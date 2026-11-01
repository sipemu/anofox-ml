use anofox_ml_core::{Fit, FitWeighted, Float, Predict, PredictProba, Result, RustMlError};
use anofox_ml_trees::{DecisionTreeRegressor, FittedDecisionTreeRegressor};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::forest_common::{argmax_first, class_index, sorted_classes};

/// Gradient boosting classifier parameters (unfitted state).
///
/// Follows Friedman's gradient boosting as implemented by scikit-learn's
/// `GradientBoostingClassifier`:
///
/// - **Binary**: one regression tree per round fitted to the negative
///   gradient `y - p` of the log loss; each leaf then takes the Newton step
///   `sum(y - p) / sum(p (1 - p))`. The initial score is the log-odds of the
///   positive class.
/// - **Multiclass** (K > 2): K trees per round, tree k fitted to `y_k - p_k`
///   of the multinomial deviance (softmax probabilities), with leaf values
///   `(K - 1) / K * sum(y_k - p_k) / sum(p_k (1 - p_k))`. Initial scores are
///   the (centred) log class priors.
///
/// With `subsample < 1` one in-bag sample (`max(1, floor(subsample * n))`
/// rows, drawn without replacement) is drawn per round and shared by all K
/// trees; leaf values are estimated from in-bag rows only.
///
/// With `n_iter_no_change`, a stratified `validation_fraction` of the data
/// is held out and boosting stops once the validation log loss has not
/// improved by `tol` within the last `n_iter_no_change` rounds (sklearn's
/// rule).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GradientBoostingClassifier {
    /// Number of boosting rounds (each round adds K trees for K > 2 classes).
    pub n_estimators: usize,
    /// Shrinkage applied to each tree's contribution.
    pub learning_rate: f64,
    /// Maximum depth of each tree.
    pub max_depth: Option<usize>,
    /// Minimum samples required to split a node.
    pub min_samples_split: usize,
    /// Minimum samples required in a leaf node.
    pub min_samples_leaf: usize,
    /// Fraction of training samples used per tree.
    pub subsample: f64,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Features drawn per split of every tree (`None` = all).
    #[serde(default)]
    pub max_features: Option<anofox_ml_trees::MaxFeatures>,
    /// Early stopping: stop when the validation loss has not improved by
    /// `tol` for this many rounds (`None` = off).
    #[serde(default)]
    pub n_iter_no_change: Option<usize>,
    /// Fraction of the training data held out for early stopping.
    #[serde(default = "crate::gradient_boosting_regressor::default_validation_fraction")]
    pub validation_fraction: f64,
    /// Minimum validation-loss improvement for early stopping.
    #[serde(default = "crate::gradient_boosting_regressor::default_tol")]
    pub tol: f64,
}

impl GradientBoostingClassifier {
    /// Create a new `GradientBoostingClassifier` with default parameters.
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: Some(3),
            min_samples_split: 2,
            min_samples_leaf: 1,
            subsample: 1.0,
            seed: 0,
            max_features: None,
            n_iter_no_change: None,
            validation_fraction: 0.1,
            tol: 1e-4,
        }
    }

    /// Features drawn per split.
    pub fn with_max_features(mut self, mf: Option<anofox_ml_trees::MaxFeatures>) -> Self {
        self.max_features = mf;
        self
    }

    /// Enable early stopping on a held-out (stratified) `validation_fraction`.
    pub fn with_early_stopping(
        mut self,
        n_iter_no_change: Option<usize>,
        validation_fraction: f64,
        tol: f64,
    ) -> Self {
        self.n_iter_no_change = n_iter_no_change;
        self.validation_fraction = validation_fraction;
        self.tol = tol;
        self
    }

    /// Set the number of boosting rounds.
    pub fn with_n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators;
        self
    }

    /// Set the learning rate (shrinkage).
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the maximum depth of each tree.
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set the minimum number of samples required to split a node.
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split;
        self
    }

    /// Set the minimum number of samples required in a leaf node.
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }

    /// Set the fraction of samples used per boosting round.
    pub fn with_subsample(mut self, subsample: f64) -> Self {
        self.subsample = subsample;
        self
    }

    /// Set the random seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// Salt of the per-tree seed stream (see `fit_impl`).
pub(crate) const TREE_SEED_SALT: u64 = 0x9E37_79B9_7F4A_7C15;

/// sklearn's half binomial / half multinomial loss of one row of raw
/// scores: `log(1 + e^r) - y r` (one column) or `logsumexp(r) - r_y`.
fn raw_log_loss<F: Float>(raw: ndarray::ArrayView1<'_, F>, label: usize) -> F {
    if raw.len() == 1 {
        let r = raw[0];
        // log(1 + e^r), stable for large |r|.
        let softplus = r.max(F::zero()) + (-r.abs()).exp().ln_1p();
        softplus - if label == 1 { r } else { F::zero() }
    } else {
        let m = raw.iter().copied().fold(F::neg_infinity(), F::max);
        let lse = m + raw
            .iter()
            .map(|&r| (r - m).exp())
            .fold(F::zero(), |a, b| a + b)
            .ln();
        lse - raw[label]
    }
}

impl Default for GradientBoostingClassifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted gradient boosting classifier.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedGradientBoostingClassifier<F: Float> {
    /// Unique class labels sorted in ascending order.
    classes: Vec<F>,
    /// For binary: a single list of trees operating on log-odds.
    /// For multiclass: one list of trees per class (`tree_sets[k][m]` is the
    /// class-k tree of round m of the multinomial model).
    tree_sets: Vec<Vec<FittedDecisionTreeRegressor<F>>>,
    /// Initial raw score per tree set.
    initial_values: Vec<F>,
    /// Learning rate.
    learning_rate: F,
    /// Number of features expected.
    n_features: usize,
}

impl<F: Float> Fit<F> for GradientBoostingClassifier {
    type Fitted = FittedGradientBoostingClassifier<F>;

    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Self::Fitted> {
        self.fit_impl(x, y, None, None)
    }
}

impl<F: Float> FitWeighted<F> for GradientBoostingClassifier {
    type Fitted = FittedGradientBoostingClassifier<F>;

    /// Per-sample weights (sklearn's `sample_weight`).
    fn fit_weighted(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        sample_weight: Option<&Array1<F>>,
    ) -> Result<Self::Fitted> {
        self.fit_impl(x, y, sample_weight, None)
    }
}

impl GradientBoostingClassifier {
    /// sklearn's `warm_start=True`: keep the rounds of `prev` and add more
    /// until there are `n_estimators`, fitted on the same `(X, y)`. The
    /// random draws of the earlier rounds are replayed, so without early
    /// stopping the result equals a single fit with `n_estimators` rounds.
    /// The early-stopping history restarts, as in sklearn. Asking for fewer
    /// rounds than `prev` has is an error; as many returns `prev` unchanged.
    pub fn fit_warm_start<F: Float>(
        &self,
        prev: FittedGradientBoostingClassifier<F>,
        x: &Array2<F>,
        y: &Array1<F>,
        sample_weight: Option<&Array1<F>>,
    ) -> Result<FittedGradientBoostingClassifier<F>> {
        self.fit_impl(x, y, sample_weight, Some(prev))
    }

    fn fit_impl<F: Float>(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        sample_weight: Option<&Array1<F>>,
        prev: Option<FittedGradientBoostingClassifier<F>>,
    ) -> Result<FittedGradientBoostingClassifier<F>> {
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
        if self.n_estimators == 0 {
            return Err(RustMlError::InvalidParameter(
                "n_estimators must be > 0".into(),
            ));
        }
        if self.learning_rate <= 0.0 {
            return Err(RustMlError::InvalidParameter(
                "learning_rate must be > 0".into(),
            ));
        }
        if self.subsample <= 0.0 || self.subsample > 1.0 {
            return Err(RustMlError::InvalidParameter(
                "subsample must be in (0, 1]".into(),
            ));
        }

        let classes = sorted_classes(y);
        let n_classes = classes.len();
        if n_classes < 2 {
            return Err(RustMlError::InvalidParameter(
                "y must contain at least 2 distinct classes".into(),
            ));
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();
        let lr = F::from_f64(self.learning_rate).unwrap();
        // Binary: one tree per round on the log-odds. Multiclass: K trees per
        // round, one per class, on the multinomial deviance.
        let n_trees_per_round = if n_classes == 2 { 1 } else { n_classes };
        if let Some(p) = &prev {
            if p.n_features != n_features || p.classes != classes {
                return Err(RustMlError::ShapeMismatch(
                    "warm start needs the features and classes of the previous fit".into(),
                ));
            }
            if p.learning_rate != lr {
                return Err(RustMlError::InvalidParameter(
                    "warm start needs the learning_rate of the previous fit".into(),
                ));
            }
            let done = p.tree_sets[0].len();
            if done > self.n_estimators {
                return Err(RustMlError::InvalidParameter(format!(
                    "warm start: n_estimators={} is less than the {done} rounds already fitted",
                    self.n_estimators
                )));
            }
            if done == self.n_estimators {
                return Ok(p.clone());
            }
        }

        // Encoded labels 0..K-1.
        let y_idx: Vec<usize> = y
            .iter()
            .map(|&v| class_index(&classes, v).expect("label is in the class list"))
            .collect();

        if let Some(w) = sample_weight {
            anofox_ml_core::check_sample_weight(w, n_samples)?;
        }
        let weight = |i: usize| sample_weight.map_or(F::one(), |w| w[i]);

        // Early stopping: hold out a stratified validation split.
        let (train_rows, val_rows): (Vec<usize>, Vec<usize>) = match self.n_iter_no_change {
            Some(_) => {
                if !(self.validation_fraction > 0.0 && self.validation_fraction < 1.0) {
                    return Err(RustMlError::InvalidParameter(
                        "validation_fraction must be in (0, 1)".into(),
                    ));
                }
                let (mut train, mut val) = anofox_ml_core::stratified_shuffle_split(
                    x,
                    y,
                    1,
                    self.validation_fraction,
                    self.seed,
                )?
                .remove(0);
                train.sort_unstable();
                val.sort_unstable();
                (train, val)
            }
            None => ((0..n_samples).collect(), Vec::new()),
        };
        let n_train = train_rows.len();

        // Initial raw prediction (sklearn's `init='prior'` / DummyClassifier)
        // from the training rows: binary = log-odds of the positive class;
        // multiclass = log priors, centred so they sum to zero (symmetric
        // multinomial logit link).
        let eps = F::epsilon();
        let mut counts = vec![F::zero(); n_classes];
        for &i in &train_rows {
            counts[y_idx[i]] += weight(i);
        }
        let n_f = counts.iter().fold(F::zero(), |a, &b| a + b);
        let priors: Vec<F> = counts
            .iter()
            .map(|&c| clamp(c / n_f, eps, F::one() - eps))
            .collect();
        let initial_values: Vec<F> = if let Some(p) = &prev {
            p.initial_values.clone()
        } else if n_classes == 2 {
            vec![(priors[1] / (F::one() - priors[1])).ln()]
        } else {
            let logs: Vec<F> = priors.iter().map(|p| p.ln()).collect();
            let mean = logs.iter().copied().fold(F::zero(), |a, b| a + b)
                / F::from_usize(n_classes).unwrap();
            logs.iter().map(|&l| l - mean).collect()
        };

        let tree_params = DecisionTreeRegressor {
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            max_features: self.max_features,
            sample_weight: None,
            seed: 0,
            ..DecisionTreeRegressor::new()
        };

        // Leaf update factor: 1 for binary, (K-1)/K for multinomial (Friedman).
        let factor = if n_classes == 2 {
            F::one()
        } else {
            F::from_usize(n_classes - 1).unwrap() / F::from_usize(n_classes).unwrap()
        };

        let mut raw = Array2::<F>::zeros((n_samples, n_trees_per_round));
        for (mut col, &init) in raw.columns_mut().into_iter().zip(&initial_values) {
            col.fill(init);
        }

        let mut rng = StdRng::seed_from_u64(self.seed);
        // Per-tree seeds for feature sampling, from a stream of their own so
        // the in-bag draws do not depend on `max_features`.
        let mut tree_seeds = StdRng::seed_from_u64(self.seed ^ TREE_SEED_SALT);
        // Same in-bag size as sklearn: max(1, int(subsample * n)).
        let n_inbag = ((self.subsample * n_train as f64) as usize).max(1);
        let use_subsample = n_inbag < n_train;

        let mut tree_sets: Vec<Vec<FittedDecisionTreeRegressor<F>>> = (0..n_trees_per_round)
            .map(|_| Vec::with_capacity(self.n_estimators))
            .collect();
        let mut probs = Array2::<F>::zeros((n_samples, n_trees_per_round));
        let mut neg_grad = Array1::<F>::zeros(n_samples);
        let mut in_bag: Vec<usize> = train_rows.clone();
        let mut loss_history = vec![F::infinity(); self.n_iter_no_change.unwrap_or(0)];

        // Warm start: take over the previous rounds and replay their random
        // draws so the streams continue where the previous fit stopped.
        let mut begin = 0;
        if let Some(p) = prev {
            raw = p.raw_predictions(x)?;
            begin = p.tree_sets[0].len();
            for _ in 0..begin {
                if use_subsample {
                    train_rows.clone().shuffle(&mut rng);
                }
                if self.max_features.is_some() {
                    for _ in 0..n_trees_per_round {
                        let _: u64 = tree_seeds.gen();
                    }
                }
            }
            tree_sets = p.tree_sets;
        }

        for iteration in begin..self.n_estimators {
            // Probabilities from the raw predictions at the start of the
            // round (shared by all K trees of the round).
            compute_probs(&raw, &mut probs);

            // One in-bag sample per round, shared across classes.
            if use_subsample {
                in_bag.clone_from(&train_rows);
                in_bag.shuffle(&mut rng);
                in_bag.truncate(n_inbag);
                in_bag.sort_unstable();
            }

            for k in 0..n_trees_per_round {
                // Class whose indicator is the target of this tree.
                let target_class = if n_classes == 2 { 1 } else { k };
                for i in 0..n_samples {
                    let y_k = if y_idx[i] == target_class {
                        F::one()
                    } else {
                        F::zero()
                    };
                    neg_grad[i] = y_k - probs[[i, k]];
                }

                let round_params;
                let params = if self.max_features.is_some() {
                    round_params = tree_params.clone().with_seed(tree_seeds.gen());
                    &round_params
                } else {
                    &tree_params
                };
                let mut tree: FittedDecisionTreeRegressor<F> = if in_bag.len() < n_samples {
                    crate::forest_common::fit_tree_on_rows(
                        params,
                        x,
                        &neg_grad,
                        &in_bag,
                        sample_weight,
                    )?
                } else {
                    params.fit_weighted(x, &neg_grad, sample_weight)?
                };

                // Newton step per leaf over the in-bag samples:
                //   factor * sum(y - p) / sum(p (1 - p)).
                let leaves = tree.apply(x)?;
                let n_leaves = tree.n_leaves();
                let mut num = vec![F::zero(); n_leaves];
                let mut den = vec![F::zero(); n_leaves];
                for &i in &in_bag {
                    let p = probs[[i, k]];
                    let w = weight(i);
                    num[leaves[i]] += w * neg_grad[i];
                    den[leaves[i]] += w * p * (F::one() - p);
                }
                let values: Vec<F> = num
                    .iter()
                    .zip(den.iter())
                    .map(|(&nu, &de)| factor * safe_divide(nu, de))
                    .collect();
                tree.set_leaf_values(&values)?;

                // Update raw predictions on all samples (in- and out-of-bag).
                for i in 0..n_samples {
                    raw[[i, k]] += lr * values[leaves[i]];
                }
                tree_sets[k].push(tree);
            }

            if let Some(n_no_change) = self.n_iter_no_change {
                let (mut num, mut den) = (F::zero(), F::zero());
                for &i in &val_rows {
                    let w = weight(i);
                    num += w * raw_log_loss(raw.row(i), y_idx[i]);
                    den += w;
                }
                let val_loss = num / den;
                let tol = F::from_f64(self.tol).unwrap();
                if loss_history.iter().any(|&h| val_loss + tol < h) {
                    loss_history[iteration % n_no_change] = val_loss;
                } else {
                    break;
                }
            }
        }

        Ok(FittedGradientBoostingClassifier {
            classes,
            tree_sets,
            initial_values,
            learning_rate: lr,
            n_features,
        })
    }
}

impl<F: Float> Predict<F> for FittedGradientBoostingClassifier<F> {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        let raw = self.raw_predictions(x)?;
        let preds = if self.classes.len() == 2 {
            raw.column(0)
                .iter()
                .map(|&r| {
                    if r >= F::zero() {
                        self.classes[1]
                    } else {
                        self.classes[0]
                    }
                })
                .collect()
        } else {
            raw.rows()
                .into_iter()
                .map(|row| self.classes[argmax_first(row.iter().copied())])
                .collect()
        };
        Ok(preds)
    }
}

impl<F: Float> FittedGradientBoostingClassifier<F> {
    /// Number of boosting rounds.
    pub fn n_estimators(&self) -> usize {
        self.tree_sets.first().map_or(0, |ts| ts.len())
    }

    /// The unique classes discovered during training.
    pub fn classes(&self) -> &[F] {
        &self.classes
    }

    /// Raw ensemble scores (sklearn's `decision_function`).
    ///
    /// Shape `(n_samples, 1)` for binary classification (log-odds of the
    /// second class) and `(n_samples, n_classes)` for multiclass.
    pub fn decision_function(&self, x: &Array2<F>) -> Result<Array2<F>> {
        self.raw_predictions(x)
    }

    /// Predict class probabilities for each sample.
    ///
    /// For binary classification, returns shape `(n_samples, 2)` with columns
    /// `[P(class_0), P(class_1)]` (sigmoid of the log-odds). For multiclass,
    /// returns `(n_samples, n_classes)` via softmax of the raw scores.
    pub fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>> {
        let raw = self.raw_predictions(x)?;
        let n_samples = raw.nrows();
        if self.classes.len() == 2 {
            let mut proba = Array2::<F>::zeros((n_samples, 2));
            for i in 0..n_samples {
                let p1 = sigmoid(raw[[i, 0]]);
                proba[[i, 0]] = F::one() - p1;
                proba[[i, 1]] = p1;
            }
            Ok(proba)
        } else {
            let mut proba = Array2::<F>::zeros(raw.raw_dim());
            compute_probs(&raw, &mut proba);
            Ok(proba)
        }
    }

    /// Feature importances averaged across all trees, normalized to sum to 1.
    pub fn feature_importances(&self) -> Array1<F> {
        let mut importances = vec![F::zero(); self.n_features];
        let mut total_trees = 0usize;

        for tree_set in &self.tree_sets {
            for tree in tree_set {
                let tree_imp = tree.feature_importances();
                for (j, &imp) in tree_imp.iter().enumerate() {
                    importances[j] += imp;
                }
                total_trees += 1;
            }
        }

        if total_trees > 0 {
            let total_f = F::from_usize(total_trees).unwrap();
            for imp in &mut importances {
                *imp /= total_f;
            }
        }

        // Normalize to sum to 1
        let sum: F = importances.iter().copied().fold(F::zero(), |a, b| a + b);
        if sum > F::zero() {
            Array1::from_vec(importances.into_iter().map(|v| v / sum).collect())
        } else {
            Array1::zeros(self.n_features)
        }
    }

    /// Raw scores: one column per tree set.
    fn raw_predictions(&self, x: &Array2<F>) -> Result<Array2<F>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        let n_samples = x.nrows();
        let mut raw = Array2::<F>::zeros((n_samples, self.tree_sets.len()));
        for (k, trees) in self.tree_sets.iter().enumerate() {
            let mut col = raw.column_mut(k);
            col.fill(self.initial_values[k]);
            for tree in trees {
                let tree_preds = tree.predict(x)?;
                col.scaled_add(self.learning_rate, &tree_preds);
            }
        }
        Ok(raw)
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Probabilities from raw scores: sigmoid for a single column (binary),
/// softmax across columns otherwise.
fn compute_probs<F: Float>(raw: &Array2<F>, probs: &mut Array2<F>) {
    let n_cols = raw.ncols();
    for (r, mut p) in raw.rows().into_iter().zip(probs.rows_mut()) {
        if n_cols == 1 {
            p[0] = sigmoid(r[0]);
        } else {
            let max = r.iter().copied().fold(F::neg_infinity(), F::max);
            let mut sum = F::zero();
            for (pk, &rk) in p.iter_mut().zip(r.iter()) {
                *pk = (rk - max).exp();
                sum += *pk;
            }
            p.mapv_inplace(|v| v / sum);
        }
    }
}

/// `num / den`, or 0 when the division is undefined (sklearn's `_safe_divide`).
fn safe_divide<F: Float>(num: F, den: F) -> F {
    if den == F::zero() {
        return F::zero();
    }
    let v = num / den;
    if v.is_finite() {
        v
    } else {
        F::zero()
    }
}

/// Sigmoid function: 1 / (1 + exp(-x)).
fn sigmoid<F: Float>(x: F) -> F {
    F::one() / (F::one() + (-x).exp())
}

/// Clamp value to [lo, hi].
fn clamp<F: Float>(x: F, lo: F, hi: F) -> F {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_raw_log_loss() {
        let ln2 = std::f64::consts::LN_2;
        assert_abs_diff_eq!(raw_log_loss(array![0.0].view(), 1), ln2, epsilon = 1e-12);
        // log(1 + e^2) - 2 for the positive class, log(1 + e^2) otherwise.
        let sp = (1.0 + 2f64.exp()).ln();
        assert_abs_diff_eq!(
            raw_log_loss(array![2.0].view(), 1),
            sp - 2.0,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(raw_log_loss(array![2.0].view(), 0), sp, epsilon = 1e-12);
        assert_abs_diff_eq!(raw_log_loss(array![800.0].view(), 0), 800.0, epsilon = 1e-9);
        let k3 = 3f64.ln();
        assert_abs_diff_eq!(
            raw_log_loss(array![1.0, 1.0, 1.0].view(), 2),
            k3,
            epsilon = 1e-12
        );
    }

    /// Data generated with numpy (seed 0); expected values from
    /// `sklearn.ensemble.GradientBoostingClassifier(n_estimators=10,
    /// learning_rate=0.3, max_depth=2, random_state=0)` (sklearn 1.8.0).
    fn sklearn_data() -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let x = array![
            [0.195, 0.861],
            [0.411, 0.18],
            [-0.305, 0.584],
            [-0.25, 1.567],
            [1.855, -0.466],
            [1.167, 0.116],
            [0.272, 1.702],
            [-1.716, -1.651],
            [-1.919, 1.33],
            [1.113, 1.48],
            [1.914, 1.197],
            [-0.154, 1.122],
            [-1.527, 0.56],
            [-1.427, 1.779],
            [0.087, -0.341],
            [-0.942, 1.097],
            [-0.175, 0.274],
            [-1.925, 0.471],
            [0.448, 0.468],
            [1.775, 0.727],
            [-0.562, -0.252],
            [0.791, -1.759],
            [0.667, 0.683],
            [-1.158, -1.484]
        ];
        let yb = array![
            0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0
        ];
        let ym = array![
            1.0, 1.0, 0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 0.0, 2.0, 2.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0,
            0.0, 1.0, 2.0, 1.0, 2.0, 2.0, 0.0
        ];
        (x, yb, ym)
    }

    fn sklearn_params() -> GradientBoostingClassifier {
        GradientBoostingClassifier::new()
            .with_n_estimators(10)
            .with_learning_rate(0.3)
            .with_max_depth(Some(2))
    }

    #[test]
    fn test_binary_matches_sklearn_newton_leaves() {
        let (x, yb, _) = sklearn_data();
        let fitted: FittedGradientBoostingClassifier<f64> = sklearn_params().fit(&x, &yb).unwrap();
        let dec = fitted.decision_function(&x).unwrap();
        let expected_dec = [-0.745538118363, -1.399793583139, -2.921933153658];
        for (i, e) in expected_dec.iter().enumerate() {
            assert_abs_diff_eq!(dec[[i, 0]], *e, epsilon = 1e-9);
        }
        let proba = fitted.predict_proba(&x).unwrap();
        let expected_p1 = [
            0.321794298769,
            0.197848868723,
            0.051079918131,
            0.872466485398,
            0.896776002215,
            0.896776002215,
        ];
        for (i, e) in expected_p1.iter().enumerate() {
            assert_abs_diff_eq!(proba[[i, 1]], *e, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_multiclass_matches_sklearn_multinomial() {
        let (x, _, ym) = sklearn_data();
        let fitted: FittedGradientBoostingClassifier<f64> = sklearn_params().fit(&x, &ym).unwrap();
        let dec = fitted.decision_function(&x).unwrap();
        let expected_dec = [
            [-1.870807292507, 1.843153671841, -2.367592091472],
            [-1.870807292507, 2.157190310048, -2.371135515195],
            [1.810505864826, 0.192775578674, -2.357844717096],
        ];
        for (i, row) in expected_dec.iter().enumerate() {
            for (k, e) in row.iter().enumerate() {
                assert_abs_diff_eq!(dec[[i, k]], *e, epsilon = 1e-9);
            }
        }
        let proba = fitted.predict_proba(&x).unwrap();
        let expected = [
            [0.023460723282, 0.962263803941, 0.014275472777],
            [0.017314608499, 0.972186996962, 0.010498394539],
            [0.823841205888, 0.163407570736, 0.012751223377],
            [0.023460723282, 0.962263803941, 0.014275472777],
            [0.009381725877, 0.010909220565, 0.979709053558],
            [0.009215989995, 0.024077754438, 0.966706255567],
        ];
        for (i, row) in expected.iter().enumerate() {
            for (k, e) in row.iter().enumerate() {
                assert_abs_diff_eq!(proba[[i, k]], *e, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn test_multiclass_subsample_shared_across_classes() {
        // Regression: every class used to reseed the RNG, so each class saw
        // an identical subsample *and* the classes were fit independently
        // (OvR). Now one in-bag sample per round is shared by the K trees and
        // the probabilities are a proper multinomial model.
        let (x, _, ym) = sklearn_data();
        let fitted: FittedGradientBoostingClassifier<f64> = sklearn_params()
            .with_subsample(0.7)
            .with_seed(3)
            .fit(&x, &ym)
            .unwrap();
        assert_eq!(fitted.n_estimators(), 10);
        let proba = fitted.predict_proba(&x).unwrap();
        for row in proba.rows() {
            assert_abs_diff_eq!(row.sum(), 1.0, epsilon = 1e-12);
        }
        // Deterministic for a given seed.
        let again: FittedGradientBoostingClassifier<f64> = sklearn_params()
            .with_subsample(0.7)
            .with_seed(3)
            .fit(&x, &ym)
            .unwrap();
        assert_eq!(proba, again.predict_proba(&x).unwrap());
    }

    #[test]
    fn test_basic_binary_classification() {
        let x = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [10.0, 1.0],
            [11.0, 1.0],
            [12.0, 1.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let gb = GradientBoostingClassifier {
            n_estimators: 50,
            learning_rate: 0.1,
            max_depth: Some(3),
            seed: 42,
            ..Default::default()
        };
        let fitted: FittedGradientBoostingClassifier<f64> = gb.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(*p, *t, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_multiclass_classification() {
        // Three classes with clearly separable data.
        let x = array![
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [5.0, 5.0],
            [5.5, 5.0],
            [6.0, 5.0],
            [10.0, 10.0],
            [10.5, 10.0],
            [11.0, 10.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];

        let gb = GradientBoostingClassifier {
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: Some(3),
            seed: 42,
            ..Default::default()
        };
        let fitted: FittedGradientBoostingClassifier<f64> = gb.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(*p, *t, epsilon = 1e-10);
        }

        // Verify all 3 classes were detected.
        assert_eq!(fitted.classes().len(), 3);
    }

    #[test]
    fn test_reproducibility() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let gb = GradientBoostingClassifier {
            n_estimators: 20,
            seed: 123,
            ..Default::default()
        };

        let fitted1: FittedGradientBoostingClassifier<f64> = gb.fit(&x, &y).unwrap();
        let fitted2: FittedGradientBoostingClassifier<f64> = gb.fit(&x, &y).unwrap();

        let preds1 = fitted1.predict(&x).unwrap();
        let preds2 = fitted2.predict(&x).unwrap();

        for (a, b) in preds1.iter().zip(preds2.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-15);
        }
    }

    #[test]
    fn test_subsample_binary() {
        let x = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [10.0, 1.0],
            [11.0, 1.0],
            [12.0, 1.0],
            [13.0, 1.0]
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let gb = GradientBoostingClassifier {
            n_estimators: 80,
            learning_rate: 0.1,
            max_depth: Some(3),
            subsample: 0.75,
            seed: 42,
            ..Default::default()
        };
        let fitted: FittedGradientBoostingClassifier<f64> = gb.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(*p, *t, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_shape_mismatch_error() {
        let x = array![[1.0], [2.0]];
        let y = array![0.0, 1.0, 2.0];

        let gb = GradientBoostingClassifier::default();
        let result: std::result::Result<FittedGradientBoostingClassifier<f64>, _> = gb.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_predict_wrong_features_error() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0];

        let gb = GradientBoostingClassifier {
            n_estimators: 5,
            seed: 0,
            ..Default::default()
        };
        let fitted: FittedGradientBoostingClassifier<f64> = gb.fit(&x, &y).unwrap();

        let x_bad = array![[1.0], [2.0]];
        let result = fitted.predict(&x_bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_single_class_error() {
        let x = array![[1.0], [2.0], [3.0]];
        let y = array![1.0, 1.0, 1.0];

        let gb = GradientBoostingClassifier::default();
        let result: std::result::Result<FittedGradientBoostingClassifier<f64>, _> = gb.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_parameters() {
        let x = array![[1.0], [2.0]];
        let y = array![0.0, 1.0];

        let gb = GradientBoostingClassifier {
            n_estimators: 0,
            ..Default::default()
        };
        assert!(Fit::<f64>::fit(&gb, &x, &y).is_err());

        let gb = GradientBoostingClassifier {
            learning_rate: -0.1,
            ..Default::default()
        };
        assert!(Fit::<f64>::fit(&gb, &x, &y).is_err());
    }

    #[test]
    fn test_n_estimators_one() {
        let x = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [10.0, 1.0],
            [11.0, 1.0],
            [12.0, 1.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let gb = GradientBoostingClassifier {
            n_estimators: 1,
            learning_rate: 0.1,
            max_depth: Some(3),
            seed: 42,
            ..Default::default()
        };
        let fitted: FittedGradientBoostingClassifier<f64> = gb.fit(&x, &y).unwrap();
        assert_eq!(fitted.n_estimators(), 1);

        // Even a single boosting round should produce predictions with the correct length.
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), y.len());
    }

    #[test]
    fn test_predictions_are_valid_labels() {
        let x = array![
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [5.0, 5.0],
            [5.5, 5.0],
            [6.0, 5.0],
            [10.0, 10.0],
            [10.5, 10.0],
            [11.0, 10.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];

        let gb = GradientBoostingClassifier {
            n_estimators: 50,
            learning_rate: 0.1,
            max_depth: Some(3),
            seed: 42,
            ..Default::default()
        };
        let fitted: FittedGradientBoostingClassifier<f64> = gb.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        let valid_labels: std::collections::HashSet<u64> = y.iter().map(|v| v.to_bits()).collect();
        for &p in preds.iter() {
            assert!(
                valid_labels.contains(&p.to_bits()),
                "prediction {p} is not a valid training label"
            );
        }
    }

    #[test]
    fn test_subsample_impact() {
        // With subsample < 1.0, the model should still produce reasonable
        // predictions on clearly separable data.
        let x = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [10.0, 1.0],
            [11.0, 1.0],
            [12.0, 1.0],
            [13.0, 1.0]
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let gb = GradientBoostingClassifier {
            n_estimators: 80,
            learning_rate: 0.1,
            max_depth: Some(3),
            subsample: 0.5,
            seed: 7,
            ..Default::default()
        };
        let fitted: FittedGradientBoostingClassifier<f64> = gb.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        let correct: usize = preds
            .iter()
            .zip(y.iter())
            .filter(|(p, t)| (*p - *t).abs() < 1e-10)
            .count();
        let accuracy = correct as f64 / y.len() as f64;
        assert!(
            accuracy >= 0.75,
            "subsample=0.5 should still achieve reasonable accuracy, got {accuracy}"
        );
    }

    #[test]
    fn test_learning_rate_zero_error_or_degrades() {
        // A very small learning rate with few estimators should produce weaker
        // predictions than a normal learning rate (less overfitting / underfitting
        // with few rounds).
        let x = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [10.0, 1.0],
            [11.0, 1.0],
            [12.0, 1.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        // Normal learning rate should fit well with enough estimators.
        let gb_normal = GradientBoostingClassifier {
            n_estimators: 50,
            learning_rate: 0.1,
            max_depth: Some(3),
            seed: 42,
            ..Default::default()
        };
        let fitted_normal: FittedGradientBoostingClassifier<f64> = gb_normal.fit(&x, &y).unwrap();
        let preds_normal = fitted_normal.predict(&x).unwrap();
        let correct_normal: usize = preds_normal
            .iter()
            .zip(y.iter())
            .filter(|(p, t)| (*p - *t).abs() < 1e-10)
            .count();

        // Tiny learning rate with the same number of estimators should learn slower.
        let gb_tiny = GradientBoostingClassifier {
            n_estimators: 50,
            learning_rate: 0.001,
            max_depth: Some(3),
            seed: 42,
            ..Default::default()
        };
        let fitted_tiny: FittedGradientBoostingClassifier<f64> = gb_tiny.fit(&x, &y).unwrap();
        let preds_tiny = fitted_tiny.predict(&x).unwrap();
        let correct_tiny: usize = preds_tiny
            .iter()
            .zip(y.iter())
            .filter(|(p, t)| (*p - *t).abs() < 1e-10)
            .count();

        // The normal learning rate should be at least as accurate (likely better).
        assert!(
            correct_normal >= correct_tiny,
            "normal lr ({correct_normal} correct) should be >= tiny lr ({correct_tiny} correct)"
        );
    }

    #[test]
    fn test_warm_start_equals_single_fit() {
        let x = Array2::from_shape_fn((90, 3), |(i, j)| ((i * (j + 2)) as f64 * 0.41).sin());
        for n_classes in [2usize, 3] {
            let y = Array1::from_shape_fn(90, |i| {
                let s = x[[i, 0]] + 0.5 * x[[i, 1]];
                if n_classes == 2 {
                    f64::from(u8::from(s > 0.0))
                } else if s < -0.4 {
                    0.0
                } else if s < 0.4 {
                    1.0
                } else {
                    2.0
                }
            });
            let gb = |n| {
                GradientBoostingClassifier::new()
                    .with_n_estimators(n)
                    .with_subsample(0.6)
                    .with_max_depth(Some(2))
                    .with_max_features(Some(anofox_ml_trees::MaxFeatures::Fixed(2)))
                    .with_seed(9)
            };
            let full = gb(10).fit(&x, &y).unwrap();
            let part = gb(4).fit(&x, &y).unwrap();
            let warm = gb(10).fit_warm_start(part, &x, &y, None).unwrap();
            assert_eq!(
                warm.predict_proba(&x).unwrap(),
                full.predict_proba(&x).unwrap()
            );
            assert!(gb(3).fit_warm_start(full, &x, &y, None).is_err());
        }
    }
}

impl<F: Float> PredictProba<F> for FittedGradientBoostingClassifier<F> {
    fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>> {
        Self::predict_proba(self, x)
    }
}
