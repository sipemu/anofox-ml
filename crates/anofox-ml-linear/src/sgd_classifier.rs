//! SGD-based linear classifier.
//!
//! Supports hinge (linear SVM), log_loss (logistic regression), and
//! modified_huber loss functions, trained with stochastic gradient descent.
//! Binary problems train a single model (positive class = `classes[1]`);
//! multi-class problems use one-vs-rest. Stopping, `partial_fit`,
//! truncated-gradient L1 and averaging (ASGD) follow scikit-learn's
//! `SGDClassifier` semantics.

use crate::sgd_common::{
    default_n_iter_no_change, default_t, default_validation_fraction, run_sgd, sigmoid, softplus,
    validation_split, LearningRate, Penalty, SgdConfig, SgdRun, SgdWeights, Stopping,
};
use anofox_ml_core::{
    ClassWeight, Fit, FitWeighted, Float, PartialFit, Predict, PredictProba, Result, RustMlError,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, AsArray, Ix2};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Loss function for SGD classifier.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ClassifierLoss {
    /// Hinge loss (linear SVM). Not differentiable at 1, but subgradient works.
    Hinge,
    /// Log loss (logistic regression).
    Log,
    /// Modified Huber: smooth hinge that gives probability estimates.
    ModifiedHuber,
    /// Perceptron loss: hinge with zero threshold.
    Perceptron,
}

impl Default for ClassifierLoss {
    fn default() -> Self {
        ClassifierLoss::Hinge
    }
}

impl ClassifierLoss {
    /// `(loss, dloss/dz)` for decision value `z` and label `y ∈ {-1, +1}`.
    #[inline]
    pub(crate) fn eval(self, z: f64, y: f64) -> (f64, f64) {
        let margin = y * z;
        match self {
            ClassifierLoss::Hinge => {
                if margin < 1.0 {
                    (1.0 - margin, -y)
                } else {
                    (0.0, 0.0)
                }
            }
            // ln(1 + exp(-m)) and its derivative -y·σ(-m), both overflow-free.
            ClassifierLoss::Log => (softplus(-margin), -y * sigmoid(-margin)),
            ClassifierLoss::ModifiedHuber => {
                if margin >= 1.0 {
                    (0.0, 0.0)
                } else if margin >= -1.0 {
                    ((1.0 - margin).powi(2), -2.0 * y * (1.0 - margin))
                } else {
                    (-4.0 * margin, -4.0 * y)
                }
            }
            ClassifierLoss::Perceptron => {
                if margin <= 0.0 {
                    (-margin, -y)
                } else {
                    (0.0, 0.0)
                }
            }
        }
    }
}

/// Stochastic Gradient Descent classifier.
///
/// Linear classifier trained via SGD, supporting multiple loss functions.
/// Binary problems train one model; for more than two classes one-vs-rest
/// decomposition is used.
///
/// Stopping (sklearn semantics): an epoch whose average training loss is
/// `> best_loss - tol` increments a no-improvement counter, any other epoch
/// resets it; training stops once the counter reaches `n_iter_no_change`.
/// With `early_stopping = true` a stratified `validation_fraction` of the
/// data is held out and its accuracy is used instead. `tol = None` disables
/// the criterion.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SgdClassifier {
    pub loss: ClassifierLoss,
    pub penalty: Penalty,
    pub alpha: f64,
    pub l1_ratio: f64,
    pub max_iter: usize,
    /// Stopping tolerance; `None` disables the stopping criterion.
    pub tol: Option<f64>,
    /// Number of epochs without improvement before stopping (default 5).
    #[serde(default = "default_n_iter_no_change")]
    pub n_iter_no_change: usize,
    /// Hold out `validation_fraction` of the data and stop on its accuracy.
    #[serde(default)]
    pub early_stopping: bool,
    /// Fraction of the training data used for early stopping (default 0.1).
    #[serde(default = "default_validation_fraction")]
    pub validation_fraction: f64,
    /// Averaged SGD: 0 = off, `k >= 1` = average the iterates once `k`
    /// samples have been seen (sklearn `average=True` is `1`).
    #[serde(default)]
    pub average: usize,
    pub eta0: f64,
    pub power_t: f64,
    pub learning_rate: LearningRate,
    pub shuffle: bool,
    pub seed: u64,
    /// Per-class weights (sklearn's `class_weight`): in each binary
    /// sub-problem the positive class gets its weight and the negative side
    /// the weight of the negative class (binary) or 1 (one-vs-rest).
    #[serde(default)]
    pub class_weight: Option<ClassWeight>,
}

impl SgdClassifier {
    pub fn new() -> Self {
        Self {
            loss: ClassifierLoss::Hinge,
            penalty: Penalty::L2,
            alpha: 0.0001,
            l1_ratio: 0.15,
            max_iter: 1000,
            tol: Some(1e-3),
            n_iter_no_change: 5,
            early_stopping: false,
            validation_fraction: 0.1,
            average: 0,
            eta0: 0.01,
            power_t: 0.5,
            learning_rate: LearningRate::Optimal,
            shuffle: true,
            seed: 0,
            class_weight: None,
        }
    }

    pub fn with_loss(mut self, loss: ClassifierLoss) -> Self {
        self.loss = loss;
        self
    }
    pub fn with_penalty(mut self, penalty: Penalty) -> Self {
        self.penalty = penalty;
        self
    }
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }
    pub fn with_l1_ratio(mut self, l1_ratio: f64) -> Self {
        self.l1_ratio = l1_ratio;
        self
    }
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }
    /// Set the stopping tolerance: `with_tol(1e-3)` or `with_tol(None)`
    /// (run all `max_iter` epochs).
    pub fn with_tol(mut self, tol: impl Into<Option<f64>>) -> Self {
        self.tol = tol.into();
        self
    }
    pub fn with_n_iter_no_change(mut self, n: usize) -> Self {
        self.n_iter_no_change = n;
        self
    }
    pub fn with_early_stopping(mut self, early_stopping: bool) -> Self {
        self.early_stopping = early_stopping;
        self
    }
    pub fn with_validation_fraction(mut self, fraction: f64) -> Self {
        self.validation_fraction = fraction;
        self
    }
    pub fn with_average(mut self, average: usize) -> Self {
        self.average = average;
        self
    }
    pub fn with_eta0(mut self, eta0: f64) -> Self {
        self.eta0 = eta0;
        self
    }
    pub fn with_power_t(mut self, power_t: f64) -> Self {
        self.power_t = power_t;
        self
    }
    pub fn with_learning_rate(mut self, lr: LearningRate) -> Self {
        self.learning_rate = lr;
        self
    }
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
    pub fn with_class_weight(mut self, class_weight: Option<ClassWeight>) -> Self {
        self.class_weight = class_weight;
        self
    }

    fn config(&self) -> SgdConfig {
        SgdConfig {
            penalty: self.penalty,
            alpha: self.alpha,
            l1_ratio: self.l1_ratio,
            eta0: self.eta0,
            power_t: self.power_t,
            learning_rate: self.learning_rate,
            average: self.average,
        }
    }

    fn validate(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<()> {
        if x.nrows() != y.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but y has {} elements",
                x.nrows(),
                y.len()
            )));
        }
        if x.is_empty() {
            return Err(RustMlError::EmptyInput("training data is empty".into()));
        }
        if self.alpha < 0.0 {
            return Err(RustMlError::InvalidParameter(
                "alpha must be non-negative".into(),
            ));
        }
        if self.n_iter_no_change == 0 {
            return Err(RustMlError::InvalidParameter(
                "n_iter_no_change must be >= 1".into(),
            ));
        }
        Ok(())
    }
}

impl Default for SgdClassifier {
    fn default() -> Self {
        Self::new()
    }
}

/// A fitted SGD linear classifier.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedSgdClassifier<F: Float> {
    /// Effective weight vectors: one for a binary problem (positive class
    /// `classes[1]`), one per class (OvR) otherwise. Averaged when
    /// `average > 0`.
    weights: Vec<Array1<F>>,
    /// Effective biases, aligned with `weights`.
    biases: Vec<F>,
    /// Sorted unique class labels.
    classes: Vec<F>,
    n_features: usize,
    /// Full optimiser state per model (plain + averaged iterate), used to
    /// resume in `partial_fit`.
    #[serde(default)]
    models: Vec<SgdWeights>,
    /// Loss used for training (decides `predict_proba` support).
    #[serde(default)]
    loss: ClassifierLoss,
    /// Epochs run by the last `fit` / `partial_fit` call (max over OvR models).
    #[serde(default)]
    n_iter_: usize,
    /// Learning-rate step counter (next `t`); persists across `partial_fit`.
    #[serde(default = "default_t")]
    t_: usize,
}

impl<F: Float> FittedSgdClassifier<F> {
    /// Return the unique class labels.
    pub fn classes(&self) -> &[F] {
        &self.classes
    }

    /// Return weight vectors: a single vector for binary problems (scores
    /// the positive class `classes[1]`), one per class (OvR) otherwise.
    pub fn weights(&self) -> &[Array1<F>] {
        &self.weights
    }

    /// Return biases, aligned with [`weights`](Self::weights).
    pub fn biases(&self) -> &[F] {
        &self.biases
    }

    /// Number of epochs run by the last `fit` / `partial_fit` call
    /// (sklearn `n_iter_`; the maximum over OvR sub-problems).
    pub fn n_iter(&self) -> usize {
        self.n_iter_
    }

    /// Learning-rate step counter: `1 + number of SGD updates performed`
    /// per model (sklearn `t_`).
    pub fn t(&self) -> usize {
        self.t_
    }

    /// Compute raw decision function values.
    ///
    /// Shape `(n_samples, 1)` for binary problems (positive = `classes[1]`,
    /// like sklearn's 1-D output) and `(n_samples, n_classes)` otherwise.
    pub fn decision_function<'a>(&self, x: impl AsArray<'a, F, Ix2>) -> Result<Array2<F>> {
        let x: ArrayView2<F> = x.into();
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        let n = x.nrows();
        let k = self.weights.len();
        let mut scores = Array2::zeros((n, k));
        for (c, (w, &b)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            for i in 0..n {
                let mut s = b;
                for j in 0..self.n_features {
                    s = s + x[[i, j]] * w[j];
                }
                scores[[i, c]] = s;
            }
        }
        Ok(scores)
    }
}

fn sorted_unique(y: &[f64]) -> Vec<f64> {
    let mut v = y.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    v.dedup();
    v
}

/// The ±1 target of each sub-problem: binary → one problem with
/// `classes[1]` positive, otherwise one OvR problem per class.
fn binary_targets(y: ArrayView1<f64>, classes: &[f64]) -> Vec<Vec<f64>> {
    let positives: Vec<f64> = if classes.len() == 2 {
        vec![classes[1]]
    } else {
        classes.to_vec()
    };
    positives
        .iter()
        .map(|&cls| {
            y.iter()
                .map(|&v| if v == cls { 1.0 } else { -1.0 })
                .collect()
        })
        .collect()
}

impl SgdClassifier {
    fn train(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        sample_weight: Option<ArrayView1<f64>>,
        classes: Vec<f64>,
        state: Option<FittedSgdClassifier<f64>>,
        partial: bool,
    ) -> Result<FittedSgdClassifier<f64>> {
        let p = x.ncols();
        let targets = binary_targets(y, &classes);
        let n_models = targets.len();
        if let Some(w) = sample_weight {
            anofox_ml_core::check_sample_weight(w, y.len())?;
        }
        // Per-sub-problem sample weights: sample_weight × (positive /
        // negative) class weight, as sklearn's `_fit_binary` /
        // `_fit_multiclass`.
        let class_w: Option<Vec<f64>> = match &self.class_weight {
            None => None,
            Some(cw) => {
                let per_sample = anofox_ml_core::compute_sample_weight(cw, y)?;
                let mut per_class = vec![1.0; classes.len()];
                for (i, &v) in y.iter().enumerate() {
                    if let Some(k) = classes.iter().position(|&c| c == v) {
                        per_class[k] = per_sample[i];
                    }
                }
                Some(per_class)
            }
        };
        let sub_weights: Vec<Option<Vec<f64>>> = targets
            .iter()
            .enumerate()
            .map(|(m, yb)| {
                if sample_weight.is_none() && class_w.is_none() {
                    return None;
                }
                let (pos, neg) = match &class_w {
                    None => (1.0, 1.0),
                    Some(cw) if n_models == 1 => (cw[1], cw[0]),
                    Some(cw) => (cw[m], 1.0),
                };
                Some(
                    yb.iter()
                        .enumerate()
                        .map(|(i, &t)| {
                            let c = if t > 0.0 { pos } else { neg };
                            c * sample_weight.map_or(1.0, |w| w[i])
                        })
                        .collect(),
                )
            })
            .collect();

        let (mut models, t_start) = match state {
            Some(s) => {
                if s.n_features != p {
                    return Err(RustMlError::ShapeMismatch(format!(
                        "previous fit had {} features, batch has {}",
                        s.n_features, p
                    )));
                }
                let models = if s.models.len() == n_models {
                    s.models
                } else if s.weights.len() == n_models {
                    // State deserialised from an older version: no optimiser
                    // state, resume from the stored weights.
                    s.weights
                        .into_iter()
                        .zip(s.biases)
                        .map(|(w, b)| SgdWeights::new(w, b))
                        .collect()
                } else {
                    return Err(RustMlError::InvalidParameter(format!(
                        "state has {} models but {} classes need {}",
                        s.weights.len(),
                        classes.len(),
                        n_models
                    )));
                };
                (models, s.t_)
            }
            None => (vec![SgdWeights::zeros(p); n_models], 1),
        };

        let cfg = self.config();
        let loss = self.loss;
        let loss_fn = |z: f64, yi: f64| loss.eval(z, yi);

        // Early stopping: one stratified split shared by all sub-problems.
        let split = if !partial && self.early_stopping {
            let ys = y.to_vec();
            let mut rng = StdRng::seed_from_u64(self.seed);
            Some(validation_split(
                x.nrows(),
                self.validation_fraction,
                Some(&ys),
                &mut rng,
            )?)
        } else {
            None
        };

        let mut n_iter = 0;
        let mut t_end = t_start;
        for (c_idx, (model, yb)) in models.iter_mut().zip(targets.iter()).enumerate() {
            let sw = sub_weights[c_idx].as_deref();
            let run: SgdRun = if partial {
                let seed = self
                    .seed
                    .wrapping_add(c_idx as u64)
                    .wrapping_add(t_start as u64);
                let mut rng = StdRng::seed_from_u64(seed);
                let mut idx: Vec<usize> = (0..x.nrows()).collect();
                run_sgd(
                    x,
                    yb,
                    sw,
                    &mut idx,
                    model,
                    &cfg,
                    loss_fn,
                    1,
                    &Stopping::never(),
                    self.shuffle,
                    &mut rng,
                    t_start,
                )
            } else {
                let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(c_idx as u64));
                match &split {
                    Some((train_idx, val_idx)) => {
                        let score = |w: &Array1<f64>, b: f64| {
                            let wt = |i: usize| sample_weight.map_or(1.0, |w| w[i]);
                            let (mut correct, mut total) = (0.0, 0.0);
                            for &i in val_idx {
                                let z = b + x.row(i).dot(w);
                                total += wt(i);
                                if (z > 0.0) == (yb[i] > 0.0) {
                                    correct += wt(i);
                                }
                            }
                            correct / total
                        };
                        let stopping = Stopping {
                            tol: self.tol,
                            n_iter_no_change: self.n_iter_no_change,
                            validation: Some(&score),
                        };
                        let mut idx = train_idx.clone();
                        run_sgd(
                            x,
                            yb,
                            sw,
                            &mut idx,
                            model,
                            &cfg,
                            loss_fn,
                            self.max_iter,
                            &stopping,
                            self.shuffle,
                            &mut rng,
                            t_start,
                        )
                    }
                    None => {
                        let stopping = Stopping {
                            tol: self.tol,
                            n_iter_no_change: self.n_iter_no_change,
                            validation: None,
                        };
                        let mut idx: Vec<usize> = (0..x.nrows()).collect();
                        run_sgd(
                            x,
                            yb,
                            sw,
                            &mut idx,
                            model,
                            &cfg,
                            loss_fn,
                            self.max_iter,
                            &stopping,
                            self.shuffle,
                            &mut rng,
                            t_start,
                        )
                    }
                }
            };
            n_iter = n_iter.max(run.n_iter);
            t_end = t_end.max(run.t);
        }

        let (weights, biases) = models
            .iter()
            .map(|m| {
                let (w, b) = m.effective();
                (w.clone(), b)
            })
            .unzip();

        Ok(FittedSgdClassifier {
            weights,
            biases,
            classes,
            n_features: p,
            models,
            loss: self.loss,
            n_iter_: n_iter,
            t_: t_end,
        })
    }
}

impl Fit<f64> for SgdClassifier {
    type Fitted = FittedSgdClassifier<f64>;

    fn fit_view(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<Self::Fitted> {
        self.validate(x, y)?;
        let classes = sorted_unique(&y.to_vec());
        if classes.len() < 2 {
            return Err(RustMlError::InvalidParameter(
                "need at least 2 classes".into(),
            ));
        }
        self.train(x, y, None, classes, None, false)
    }
}

impl FitWeighted<f64> for SgdClassifier {
    type Fitted = FittedSgdClassifier<f64>;

    /// Each SGD update is scaled by the sample's weight times its class
    /// weight (sklearn); the early-stopping accuracy is weighted too.
    fn fit_weighted_view(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        sample_weight: Option<ArrayView1<f64>>,
    ) -> Result<Self::Fitted> {
        self.validate(x, y)?;
        let classes = sorted_unique(&y.to_vec());
        if classes.len() < 2 {
            return Err(RustMlError::InvalidParameter(
                "need at least 2 classes".into(),
            ));
        }
        self.train(x, y, sample_weight, classes, None, false)
    }
}

impl PartialFit<f64> for SgdClassifier {
    type Fitted = FittedSgdClassifier<f64>;

    /// Run exactly one SGD epoch over the batch, resuming from `state`
    /// (weights, averaged weights and the learning-rate counter `t_`), as
    /// sklearn's `partial_fit`. `tol` / `early_stopping` do not apply.
    fn partial_fit_view(
        &self,
        state: Option<Self::Fitted>,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        classes: Option<&[f64]>,
    ) -> Result<Self::Fitted> {
        self.validate(x, y)?;

        // Establish the class set: from state, or from `classes` on first call,
        // or by deriving from the batch (must have ≥ 2 unique labels then).
        let class_list: Vec<f64> = if let Some(s) = &state {
            s.classes.clone()
        } else if let Some(c) = classes {
            sorted_unique(c)
        } else {
            let v = sorted_unique(&y.to_vec());
            if v.len() < 2 {
                return Err(RustMlError::InvalidParameter(
                    "first partial_fit needs classes= or batch with ≥ 2 unique labels".into(),
                ));
            }
            v
        };
        if class_list.len() < 2 {
            return Err(RustMlError::InvalidParameter(
                "need at least 2 classes".into(),
            ));
        }
        if let Some(bad) = y.iter().find(|v| !class_list.contains(v)) {
            return Err(RustMlError::InvalidParameter(format!(
                "label {bad} is not in classes {class_list:?}"
            )));
        }

        if self.class_weight == Some(ClassWeight::Balanced) {
            return Err(RustMlError::InvalidParameter(
                "class_weight=Balanced is not supported by partial_fit (sklearn); \
                 pass Manual weights computed on the full data"
                    .into(),
            ));
        }
        self.train(x, y, None, class_list, state, true)
    }
}

impl Predict<f64> for FittedSgdClassifier<f64> {
    fn predict_view(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        let scores = self.decision_function(x)?;
        let n = x.nrows();
        let mut preds = Array1::zeros(n);

        if self.weights.len() == 1 {
            for i in 0..n {
                preds[i] = if scores[[i, 0]] > 0.0 {
                    self.classes[1]
                } else {
                    self.classes[0]
                };
            }
            return Ok(preds);
        }

        for i in 0..n {
            let mut best_c = 0;
            let mut best_s = scores[[i, 0]];
            for c in 1..self.classes.len() {
                if scores[[i, c]] > best_s {
                    best_s = scores[[i, c]];
                    best_c = c;
                }
            }
            preds[i] = self.classes[best_c];
        }
        Ok(preds)
    }
}

impl PredictProba<f64> for FittedSgdClassifier<f64> {
    /// Class probabilities, columns ordered as [`classes`](Self::classes).
    ///
    /// Only for `ClassifierLoss::Log` (sigmoid of the decision value) and
    /// `ClassifierLoss::ModifiedHuber` (`(clip(s, -1, 1) + 1) / 2`), as in
    /// sklearn. Multi-class OvR scores are normalised to sum to one
    /// (uniform when all OvR probabilities are zero).
    fn predict_proba_view(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        let link: fn(f64) -> f64 = match self.loss {
            ClassifierLoss::Log => sigmoid,
            ClassifierLoss::ModifiedHuber => |s: f64| (s.clamp(-1.0, 1.0) + 1.0) / 2.0,
            other => {
                return Err(RustMlError::InvalidParameter(format!(
                    "predict_proba is only available for Log and ModifiedHuber loss, not {other:?}"
                )))
            }
        };
        let scores = self.decision_function(x)?;
        let n = x.nrows();
        let k = self.classes.len();
        let mut proba = Array2::zeros((n, k));
        if self.weights.len() == 1 {
            for i in 0..n {
                let p1 = link(scores[[i, 0]]);
                proba[[i, 0]] = 1.0 - p1;
                proba[[i, 1]] = p1;
            }
            return Ok(proba);
        }
        for i in 0..n {
            let mut sum = 0.0;
            for c in 0..k {
                let v = link(scores[[i, c]]);
                proba[[i, c]] = v;
                sum += v;
            }
            for c in 0..k {
                proba[[i, c]] = if sum > 0.0 {
                    proba[[i, c]] / sum
                } else {
                    1.0 / k as f64
                };
            }
        }
        Ok(proba)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn make_binary_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec(
            (12, 2),
            vec![
                0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 0.0, 1.0, 3.0, 3.0, 3.5, 3.5, 4.0, 3.0, 3.0, 4.0,
                0.5, 0.0, 0.0, 0.5, 3.5, 3.0, 3.0, 3.5,
            ],
        )
        .unwrap();
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        (x, y)
    }

    #[test]
    fn test_log_loss_stable_for_huge_margins() {
        for &z in &[1e3, 1e6, 1e300, -1e3, -1e6, -1e300] {
            for &y in &[1.0, -1.0] {
                let (l, g) = ClassifierLoss::Log.eval(z, y);
                assert!(l.is_finite() && l >= 0.0, "loss({z}, {y}) = {l}");
                assert!(g.is_finite() && g.abs() <= 1.0, "grad({z}, {y}) = {g}");
                if y * z > 0.0 {
                    assert!(l < 1e-300 && g.abs() < 1e-300);
                } else {
                    assert!((l - (y * z).abs()).abs() <= 1e-9 * (y * z).abs());
                    assert!((g.abs() - 1.0).abs() < 1e-15);
                }
            }
        }
        let (l0, g0) = ClassifierLoss::Log.eval(0.0, 1.0);
        assert!((l0 - std::f64::consts::LN_2).abs() < 1e-15);
        assert!((g0 + 0.5).abs() < 1e-15);
    }

    #[test]
    fn test_sgd_partial_fit_resumes_from_state() {
        // partial_fit on two halves should fit a classifier that achieves
        // training accuracy ≥ that of one-shot fit on a single half.
        let (x, y) = make_binary_data();
        let n = x.nrows();
        let half = n / 2;
        let x1 = x.slice(ndarray::s![..half, ..]).to_owned();
        let y1 = y.slice(ndarray::s![..half]).to_owned();
        let x2 = x.slice(ndarray::s![half.., ..]).to_owned();
        let y2 = y.slice(ndarray::s![half..]).to_owned();

        let clf = SgdClassifier::new()
            .with_loss(ClassifierLoss::Log)
            .with_max_iter(200)
            .with_alpha(0.001);
        let classes = [0.0_f64, 1.0];

        // partial_fit runs one epoch per call (sklearn semantics), so stream
        // the two halves repeatedly.
        let mut s2 = PartialFit::partial_fit(&clf, None, &x1, &y1, Some(&classes)).unwrap();
        s2 = PartialFit::partial_fit(&clf, Some(s2), &x2, &y2, None).unwrap();
        for _ in 0..200 {
            s2 = PartialFit::partial_fit(&clf, Some(s2), &x1, &y1, None).unwrap();
            s2 = PartialFit::partial_fit(&clf, Some(s2), &x2, &y2, None).unwrap();
        }

        let preds = s2.predict(&x).unwrap();
        let correct: usize = preds.iter().zip(y.iter()).filter(|(&p, &t)| p == t).count();
        // Should classify the majority correctly after seeing both halves.
        assert!(correct >= 10, "partial_fit too inaccurate: {correct}/12");
        assert_eq!(s2.classes(), &[0.0_f64, 1.0]);
    }

    #[test]
    fn test_sgd_classifier_hinge() {
        let (x, y) = make_binary_data();
        let clf = SgdClassifier::new()
            .with_loss(ClassifierLoss::Hinge)
            .with_max_iter(500)
            .with_alpha(0.001);
        let fitted = clf.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        let correct: usize = preds.iter().zip(y.iter()).filter(|(&p, &t)| p == t).count();
        assert!(
            correct >= 8,
            "should classify most points correctly, got {}/12",
            correct
        );
    }

    #[test]
    fn test_sgd_classifier_log_loss() {
        let (x, y) = make_binary_data();
        let clf = SgdClassifier::new()
            .with_loss(ClassifierLoss::Log)
            .with_max_iter(500)
            .with_alpha(0.001);
        let fitted = clf.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        let correct: usize = preds.iter().zip(y.iter()).filter(|(&p, &t)| p == t).count();
        assert!(
            correct >= 8,
            "should classify most points correctly, got {}/12",
            correct
        );
    }

    #[test]
    fn test_sgd_classifier_modified_huber() {
        let (x, y) = make_binary_data();
        let clf = SgdClassifier::new()
            .with_loss(ClassifierLoss::ModifiedHuber)
            .with_max_iter(500);
        let fitted = clf.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(p == 0.0 || p == 1.0);
        }
    }

    #[test]
    fn test_sgd_classifier_perceptron() {
        let (x, y) = make_binary_data();
        let clf = SgdClassifier::new()
            .with_loss(ClassifierLoss::Perceptron)
            .with_penalty(Penalty::None)
            .with_max_iter(500);
        let fitted = clf.fit(&x, &y).unwrap();

        assert_eq!(fitted.classes().len(), 2);
    }

    #[test]
    fn test_sgd_classifier_multiclass() {
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.5, 0.5, 0.0, 0.5, 3.0, 0.0, 3.5, 0.5, 3.0, 0.5, 1.5, 3.0, 1.0, 3.5,
                2.0, 3.0,
            ],
        )
        .unwrap();
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];

        let clf = SgdClassifier::new()
            .with_loss(ClassifierLoss::Hinge)
            .with_max_iter(1000)
            .with_alpha(0.001);
        let fitted = clf.fit(&x, &y).unwrap();

        assert_eq!(fitted.classes().len(), 3);
        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(p == 0.0 || p == 1.0 || p == 2.0);
        }
    }

    #[test]
    fn test_sgd_classifier_decision_function() {
        let (x, y) = make_binary_data();
        let clf = SgdClassifier::new().with_max_iter(100);
        let fitted = clf.fit(&x, &y).unwrap();

        let scores = fitted.decision_function(x.view()).unwrap();
        assert_eq!(scores.nrows(), 12);
        assert_eq!(scores.ncols(), 1, "binary problems train a single model");
    }

    #[test]
    fn test_sgd_classifier_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 2), vec![0.0; 6]).unwrap();
        let y = array![0.0, 1.0];
        assert!(SgdClassifier::new().fit(&x, &y).is_err());
    }

    #[test]
    fn test_sgd_classifier_empty_input() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<f64>::zeros(0);
        assert!(SgdClassifier::new().fit(&x, &y).is_err());
    }

    #[test]
    fn test_sgd_classifier_single_class() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![0.0, 0.0, 0.0];
        assert!(SgdClassifier::new().fit(&x, &y).is_err());
    }

    #[test]
    fn test_sgd_classifier_elastic_net() {
        let (x, y) = make_binary_data();
        let clf = SgdClassifier::new()
            .with_penalty(Penalty::ElasticNet)
            .with_l1_ratio(0.5)
            .with_max_iter(500);
        let fitted = clf.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 12);
    }

    #[test]
    fn test_sgd_classifier_constant_lr() {
        let (x, y) = make_binary_data();
        let clf = SgdClassifier::new()
            .with_learning_rate(LearningRate::Constant)
            .with_eta0(0.01)
            .with_max_iter(500);
        let fitted = clf.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 12);
    }
}

impl anofox_ml_core::ClassifierScore<f64> for FittedSgdClassifier<f64> {}
