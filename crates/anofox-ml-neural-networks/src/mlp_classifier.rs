use anofox_ml_core::{Fit, Float, PartialFit, Predict, Result, RustMlError};
use ndarray::{Array1, Array2, ArrayView2, Axis};

use crate::activation::Activation;
use crate::network::{predict_output, DenseLayer, OutputKind};
use crate::solver::{LearningRateSchedule, Solver, StochasticOptimizer};
use crate::trainer::{initialize, rng_for, train, TrainConfig, TrainState};

/// Multi-layer perceptron classifier (unfitted). Mirrors
/// `sklearn.neural_network.MLPClassifier`: binary problems use a single
/// logistic output unit, multi-class problems a softmax layer; the loss is
/// the log-loss computed from the logits plus `alpha/(2·n_batch)·Σ||W||²`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct MlpClassifier {
    /// Sizes of hidden layers, e.g. `vec![100]` for one hidden layer of 100 units.
    pub hidden_layer_sizes: Vec<usize>,
    /// Activation function for hidden layers.
    pub activation: Activation,
    /// Optimization solver.
    pub solver: Solver,
    /// Initial learning rate (sklearn `learning_rate_init`; sgd/adam).
    pub learning_rate: f64,
    /// Maximum number of epochs (sgd/adam) or iterations (lbfgs).
    pub max_iter: usize,
    /// Optimisation tolerance: minimum loss (or validation score)
    /// improvement per epoch; projected-gradient tolerance for lbfgs.
    pub tol: f64,
    /// Random seed for weight initialization, shuffling and the
    /// early-stopping split.
    pub seed: u64,
    /// Mini-batch size. `None` means full batch.
    pub batch_size: Option<usize>,
    /// L2 regularization strength (scaled by `1/n_batch` like sklearn).
    pub alpha: f64,
    /// Learning-rate schedule for `Solver::Sgd` (sklearn `learning_rate`).
    pub learning_rate_schedule: LearningRateSchedule,
    /// Exponent of the `InvScaling` schedule.
    pub power_t: f64,
    /// Momentum for `Solver::Sgd`, in `[0, 1]`.
    pub momentum: f64,
    /// Use Nesterov momentum (`Solver::Sgd`).
    pub nesterovs_momentum: bool,
    /// Adam first-moment decay.
    pub beta_1: f64,
    /// Adam second-moment decay.
    pub beta_2: f64,
    /// Adam numerical-stability constant.
    pub epsilon: f64,
    /// Shuffle samples every epoch (sgd/adam).
    pub shuffle: bool,
    /// Epochs without `tol` improvement before stopping (or, for the
    /// adaptive schedule, before dividing the learning rate by 5).
    pub n_iter_no_change: usize,
    /// Hold out `validation_fraction` of the data (stratified) and stop when
    /// the validation accuracy stops improving; the best weights are kept.
    pub early_stopping: bool,
    /// Fraction of training data used for early stopping, in `(0, 1)`.
    pub validation_fraction: f64,
    /// Maximum number of loss evaluations (lbfgs).
    pub max_fun: usize,
    /// Explicit initial `(W, b)` per layer instead of random initialisation
    /// (used to reproduce sklearn trajectories in tests).
    #[doc(hidden)]
    pub initial_weights: Option<Vec<(Array2<f64>, Array1<f64>)>>,
}

impl Default for MlpClassifier {
    fn default() -> Self {
        Self {
            hidden_layer_sizes: vec![100],
            activation: Activation::Relu,
            solver: Solver::Adam,
            learning_rate: 0.001,
            max_iter: 200,
            tol: 1e-4,
            seed: 0,
            batch_size: Some(200),
            alpha: 1e-4,
            learning_rate_schedule: LearningRateSchedule::Constant,
            power_t: 0.5,
            momentum: 0.9,
            nesterovs_momentum: true,
            beta_1: 0.9,
            beta_2: 0.999,
            epsilon: 1e-8,
            shuffle: true,
            n_iter_no_change: 10,
            early_stopping: false,
            validation_fraction: 0.1,
            max_fun: 15000,
            initial_weights: None,
        }
    }
}

macro_rules! mlp_builders {
    ($t:ty) => {
        impl $t {
            pub fn with_hidden_layer_sizes(mut self, v: Vec<usize>) -> Self {
                self.hidden_layer_sizes = v;
                self
            }
            pub fn with_activation(mut self, v: Activation) -> Self {
                self.activation = v;
                self
            }
            pub fn with_solver(mut self, v: Solver) -> Self {
                self.solver = v;
                self
            }
            pub fn with_learning_rate(mut self, v: f64) -> Self {
                self.learning_rate = v;
                self
            }
            pub fn with_learning_rate_schedule(mut self, v: LearningRateSchedule) -> Self {
                self.learning_rate_schedule = v;
                self
            }
            pub fn with_power_t(mut self, v: f64) -> Self {
                self.power_t = v;
                self
            }
            pub fn with_max_iter(mut self, v: usize) -> Self {
                self.max_iter = v;
                self
            }
            pub fn with_tol(mut self, v: f64) -> Self {
                self.tol = v;
                self
            }
            pub fn with_seed(mut self, v: u64) -> Self {
                self.seed = v;
                self
            }
            pub fn with_batch_size(mut self, v: Option<usize>) -> Self {
                self.batch_size = v;
                self
            }
            pub fn with_alpha(mut self, v: f64) -> Self {
                self.alpha = v;
                self
            }
            pub fn with_momentum(mut self, v: f64) -> Self {
                self.momentum = v;
                self
            }
            pub fn with_nesterovs_momentum(mut self, v: bool) -> Self {
                self.nesterovs_momentum = v;
                self
            }
            pub fn with_shuffle(mut self, v: bool) -> Self {
                self.shuffle = v;
                self
            }
            pub fn with_n_iter_no_change(mut self, v: usize) -> Self {
                self.n_iter_no_change = v;
                self
            }
            pub fn with_early_stopping(mut self, v: bool) -> Self {
                self.early_stopping = v;
                self
            }
            pub fn with_validation_fraction(mut self, v: f64) -> Self {
                self.validation_fraction = v;
                self
            }
            pub fn with_max_fun(mut self, v: usize) -> Self {
                self.max_fun = v;
                self
            }
            /// Start from explicit `(W_i, b_i)` per layer (`W_i` has shape
            /// `(n_in, n_out)`) instead of random initialisation — e.g. the
            /// `coefs_` / `intercepts_` of a sklearn model.
            #[doc(hidden)]
            pub fn with_initial_weights(mut self, v: Vec<(Array2<f64>, Array1<f64>)>) -> Self {
                self.initial_weights = Some(v);
                self
            }

            pub(crate) fn train_config(&self) -> TrainConfig {
                TrainConfig {
                    hidden_layer_sizes: self.hidden_layer_sizes.clone(),
                    activation: self.activation,
                    solver: self.solver,
                    learning_rate: self.learning_rate,
                    learning_rate_schedule: self.learning_rate_schedule,
                    power_t: self.power_t,
                    momentum: self.momentum,
                    nesterovs_momentum: self.nesterovs_momentum,
                    beta_1: self.beta_1,
                    beta_2: self.beta_2,
                    epsilon: self.epsilon,
                    alpha: self.alpha,
                    batch_size: self.batch_size,
                    max_iter: self.max_iter,
                    tol: self.tol,
                    n_iter_no_change: self.n_iter_no_change,
                    early_stopping: self.early_stopping,
                    validation_fraction: self.validation_fraction,
                    shuffle: self.shuffle,
                    max_fun: self.max_fun,
                    initial_weights: self.initial_weights.clone(),
                }
            }
        }
    };
}
pub(crate) use mlp_builders;

mlp_builders!(MlpClassifier);

/// Training diagnostics shared by both fitted MLPs (sklearn's trailing
/// underscore attributes) plus the resumable optimiser state.
macro_rules! fitted_accessors {
    ($t:ident) => {
        impl<F: Float> $t<F> {
            /// Training loss per epoch (sgd/adam; empty for lbfgs), sklearn
            /// `loss_curve_`. Accumulates across `partial_fit` / warm starts.
            pub fn loss_curve(&self) -> &[f64] {
                &self.loss_curve
            }
            /// Epochs (sgd/adam) or iterations (lbfgs) run by the last
            /// `fit` / `partial_fit` call (sklearn `n_iter_`).
            pub fn n_iter(&self) -> usize {
                self.n_iter
            }
            /// Final training loss (sklearn `loss_`).
            pub fn loss(&self) -> f64 {
                self.loss
            }
            /// Lowest epoch loss seen (sklearn `best_loss_`; `None` with
            /// early stopping or lbfgs).
            pub fn best_loss(&self) -> Option<f64> {
                self.best_loss
            }
            /// Validation scores per epoch with early stopping.
            pub fn validation_scores(&self) -> &[f64] {
                &self.validation_scores
            }
            /// Best validation score with early stopping.
            pub fn best_validation_score(&self) -> Option<f64> {
                self.best_validation_score
            }
            /// Samples seen by the stochastic solver (sklearn `t_`).
            pub fn t(&self) -> usize {
                self.t
            }
            /// Weight matrices per layer (sklearn `coefs_`).
            pub fn coefs(&self) -> Vec<&Array2<F>> {
                self.layers.iter().map(|l| &l.weights).collect()
            }
            /// Bias vectors per layer (sklearn `intercepts_`).
            pub fn intercepts(&self) -> Vec<Array1<F>> {
                self.layers
                    .iter()
                    .map(|l| l.biases.row(0).to_owned())
                    .collect()
            }

            fn into_state(self) -> TrainState<F> {
                TrainState {
                    layers: self.layers,
                    loss_curve: self.loss_curve,
                    loss: self.loss,
                    n_iter: self.n_iter,
                    t: self.t,
                    best_loss: self.best_loss,
                    no_improvement_count: self.no_improvement_count,
                    validation_scores: self.validation_scores,
                    best_validation_score: self.best_validation_score,
                    optimizer: self.optimizer,
                }
            }
        }
    };
}
pub(crate) use fitted_accessors;

/// Fitted multi-layer perceptron classifier.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedMlpClassifier<F: Float> {
    layers: Vec<DenseLayer<F>>,
    activation: Activation,
    class_labels: Vec<F>,
    n_features: usize,
    #[serde(default)]
    loss_curve: Vec<f64>,
    #[serde(default)]
    n_iter: usize,
    #[serde(default)]
    loss: f64,
    #[serde(default)]
    best_loss: Option<f64>,
    #[serde(default)]
    validation_scores: Vec<f64>,
    #[serde(default)]
    best_validation_score: Option<f64>,
    #[serde(default)]
    t: usize,
    #[serde(default)]
    no_improvement_count: usize,
    #[serde(skip)]
    optimizer: Option<StochasticOptimizer<F>>,
}

fitted_accessors!(FittedMlpClassifier);

fn output_kind<F: Float>(layers: &[DenseLayer<F>]) -> OutputKind {
    if layers.last().map_or(0, |l| l.weights.ncols()) == 1 {
        OutputKind::Logistic
    } else {
        OutputKind::Softmax
    }
}

/// Fraction of rows whose predicted class equals the target class.
fn accuracy_from_output<F: Float>(kind: OutputKind, out: &Array2<F>, y: ArrayView2<F>) -> f64 {
    let n = out.nrows();
    if out.iter().any(|v| !v.is_finite()) {
        return f64::NAN;
    }
    let half = F::from_f64(0.5).unwrap();
    let correct = (0..n)
        .filter(|&i| match kind {
            OutputKind::Logistic => (out[[i, 0]] > half) == (y[[i, 0]] > half),
            _ => argmax(out.row(i).iter()) == argmax(y.row(i).iter()),
        })
        .count();
    correct as f64 / n as f64
}

fn argmax<'a, F: Float + 'a>(it: impl Iterator<Item = &'a F>) -> usize {
    let mut best = 0;
    let mut best_v = F::neg_infinity();
    for (i, &v) in it.enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

impl<F: Float> FittedMlpClassifier<F> {
    /// Return class probability estimates.
    pub fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        let kind = output_kind(&self.layers);
        let out = predict_output(&self.layers, self.activation, kind, x.view());
        Ok(match kind {
            OutputKind::Logistic => {
                let mut p = Array2::zeros((out.nrows(), 2));
                for i in 0..out.nrows() {
                    p[[i, 0]] = F::one() - out[[i, 0]];
                    p[[i, 1]] = out[[i, 0]];
                }
                p
            }
            _ => out,
        })
    }

    /// Class labels in sorted order (sklearn `classes_`).
    pub fn classes(&self) -> &[F] {
        &self.class_labels
    }
}

fn label_index<F: Float>(labels: &[F], v: F) -> Option<usize> {
    labels
        .iter()
        .position(|&c| (c - v).abs() < F::from_f64(1e-10).unwrap())
}

impl MlpClassifier {
    fn fit_impl<F: Float>(
        &self,
        prev: Option<FittedMlpClassifier<F>>,
        x: &Array2<F>,
        y: &Array1<F>,
        incremental: bool,
        classes: Option<&[F]>,
    ) -> Result<FittedMlpClassifier<F>> {
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
        let cfg = self.train_config();
        cfg.validate()?;

        let class_labels: Vec<F> = match (&prev, classes) {
            (Some(p), _) => {
                if p.n_features != x.ncols() {
                    return Err(RustMlError::ShapeMismatch(format!(
                        "expected {} features, got {}",
                        p.n_features,
                        x.ncols()
                    )));
                }
                p.class_labels.clone()
            }
            (None, Some(c)) => {
                let mut c = c.to_vec();
                c.sort_by(anofox_ml_core::labels::label_cmp);
                c.dedup();
                c
            }
            (None, None) => {
                let mut seen = std::collections::BTreeMap::new();
                for &v in y.iter() {
                    seen.entry(v.to_f64().unwrap().to_bits()).or_insert(v);
                }
                let mut c: Vec<F> = seen.into_values().collect();
                c.sort_by(anofox_ml_core::labels::label_cmp);
                c
            }
        };
        let n_classes = class_labels.len();
        if n_classes < 2 {
            return Err(RustMlError::InvalidParameter(
                "need at least 2 classes for classification".into(),
            ));
        }
        let n_out = if n_classes == 2 { 1 } else { n_classes };
        let mut strata = Vec::with_capacity(y.len());
        let mut targets = Array2::<F>::zeros((y.len(), n_out));
        for (i, &v) in y.iter().enumerate() {
            let c = label_index(&class_labels, v).ok_or_else(|| {
                RustMlError::InvalidParameter(format!(
                    "label {} is not one of the known classes",
                    v.to_f64().unwrap()
                ))
            })?;
            strata.push(c);
            if n_out == 1 {
                targets[[i, 0]] = if c == 1 { F::one() } else { F::zero() };
            } else {
                targets[[i, c]] = F::one();
            }
        }

        let mut rng = rng_for(self.seed);
        let mut state = match prev {
            Some(p) => {
                let n_out_prev = p.layers.last().map_or(0, |l| l.weights.ncols());
                if n_out_prev != n_out {
                    return Err(RustMlError::InvalidParameter(
                        "previous model has an incompatible output layer".into(),
                    ));
                }
                p.into_state()
            }
            None => initialize(&cfg, x.ncols(), n_out, &mut rng)?,
        };
        let kind = if n_out == 1 {
            OutputKind::Logistic
        } else {
            OutputKind::Softmax
        };
        let activation = self.activation;
        let scorer = move |layers: &[DenseLayer<F>], xv: ArrayView2<F>, yv: ArrayView2<F>| {
            let out = predict_output(layers, activation, kind, xv);
            accuracy_from_output(kind, &out, yv)
        };
        train(
            &cfg,
            &mut state,
            kind,
            x,
            &targets,
            &mut rng,
            incremental,
            Some(&strata),
            &scorer,
        )?;
        Ok(FittedMlpClassifier {
            layers: state.layers,
            activation: self.activation,
            class_labels,
            n_features: x.ncols(),
            loss_curve: state.loss_curve,
            n_iter: state.n_iter,
            loss: state.loss,
            best_loss: state.best_loss,
            validation_scores: state.validation_scores,
            best_validation_score: state.best_validation_score,
            t: state.t,
            no_improvement_count: state.no_improvement_count,
            optimizer: state.optimizer,
        })
    }

    /// Continue training from `previous` (sklearn `warm_start=True`):
    /// weights, `loss_curve` and `best_loss` carry over, the optimiser is
    /// re-initialised and up to `max_iter` further epochs run.
    pub fn fit_warm_start<F: Float>(
        &self,
        previous: FittedMlpClassifier<F>,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<FittedMlpClassifier<F>> {
        self.fit_impl(Some(previous), x, y, false, None)
    }
}

impl<F: Float> Fit<F> for MlpClassifier {
    type Fitted = FittedMlpClassifier<F>;

    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Self::Fitted> {
        self.fit_impl(None, x, y, false, None)
    }
}

/// One epoch over `(x, y)` (sklearn `partial_fit`). On the first call pass
/// all `classes` (inferred from `y` if `None`); the optimiser state is kept
/// between calls in memory.
impl<F: Float> PartialFit<F> for MlpClassifier {
    type Fitted = FittedMlpClassifier<F>;

    fn partial_fit(
        &self,
        state: Option<Self::Fitted>,
        x: &Array2<F>,
        y: &Array1<F>,
        classes: Option<&[F]>,
    ) -> Result<Self::Fitted> {
        self.fit_impl(state, x, y, true, classes)
    }
}

impl<F: Float> Predict<F> for FittedMlpClassifier<F> {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        let probs = self.predict_proba(x)?;
        let half = F::from_f64(0.5).unwrap();
        Ok(probs
            .axis_iter(Axis(0))
            .map(|row| {
                if row.len() == 2 {
                    // sklearn LabelBinarizer threshold: p(class 1) > 0.5
                    self.class_labels[usize::from(row[1] > half)]
                } else {
                    self.class_labels[argmax(row.iter())]
                }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn xor_classification() {
        // XOR is not linearly separable, requires hidden layer
        let x = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0],];
        let y = array![0.0, 1.0, 1.0, 0.0];

        let mlp = MlpClassifier {
            hidden_layer_sizes: vec![10, 10],
            activation: Activation::Relu,
            solver: Solver::Adam,
            learning_rate: 0.01,
            max_iter: 500,
            tol: 1e-6,
            seed: 42,
            batch_size: None, // full batch
            alpha: 0.0,
            ..Default::default()
        };

        let fitted: FittedMlpClassifier<f64> = mlp.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(*p, *t, epsilon = 1e-10);
        }
    }

    #[test]
    fn simple_three_class() {
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.0],
            [10.0, 0.0],
            [10.1, 0.1],
            [10.2, 0.0],
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];

        let mlp = MlpClassifier {
            hidden_layer_sizes: vec![20],
            learning_rate: 0.01,
            max_iter: 300,
            seed: 123,
            batch_size: None,
            alpha: 0.0,
            ..Default::default()
        };

        let fitted: FittedMlpClassifier<f64> = mlp.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        let correct: usize = preds
            .iter()
            .zip(y.iter())
            .filter(|(p, t)| (**p - **t).abs() < 1e-10)
            .count();
        assert!(correct >= 7, "expected >= 7 correct, got {correct}");
    }

    #[test]
    fn predict_proba_sums_to_one() {
        let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let mlp = MlpClassifier {
            hidden_layer_sizes: vec![5],
            max_iter: 10,
            seed: 7,
            ..Default::default()
        };
        let fitted: FittedMlpClassifier<f64> = mlp.fit(&x, &y).unwrap();
        let proba = fitted.predict_proba(&x).unwrap();
        assert_eq!(proba.ncols(), 2);
        for row in proba.rows() {
            assert_abs_diff_eq!(row.sum(), 1.0, epsilon = 1e-6);
        }
        // Binary → single logistic output unit, as in sklearn.
        assert_eq!(fitted.coefs().last().unwrap().ncols(), 1);
    }

    #[test]
    fn shape_mismatch_error() {
        let x = array![[1.0, 2.0]];
        let y = array![0.0, 1.0];
        let result: std::result::Result<FittedMlpClassifier<f64>, _> =
            MlpClassifier::default().fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn empty_input_error() {
        let x: Array2<f64> = Array2::zeros((0, 2));
        let y: Array1<f64> = Array1::zeros(0);
        let result: std::result::Result<FittedMlpClassifier<f64>, _> =
            MlpClassifier::default().fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn single_class_error() {
        let x = array![[1.0], [2.0], [3.0]];
        let y = array![0.0, 0.0, 0.0];
        let result: std::result::Result<FittedMlpClassifier<f64>, _> =
            MlpClassifier::default().fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn predict_wrong_features_error() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0];
        let mlp = MlpClassifier {
            hidden_layer_sizes: vec![5],
            max_iter: 5,
            ..Default::default()
        };
        let fitted: FittedMlpClassifier<f64> = mlp.fit(&x, &y).unwrap();
        assert!(fitted.predict(&array![[1.0]]).is_err());
    }

    #[test]
    fn reproducibility() {
        let x = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let y = array![0.0, 1.0, 1.0, 0.0];
        let mlp = MlpClassifier {
            hidden_layer_sizes: vec![10],
            max_iter: 50,
            seed: 999,
            batch_size: None,
            ..Default::default()
        };
        let p1 = mlp.fit(&x, &y).unwrap().predict_proba(&x).unwrap();
        let p2 = mlp.fit(&x, &y).unwrap().predict_proba(&x).unwrap();
        for (a, b) in p1.iter().zip(p2.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-12);
        }
    }
}
