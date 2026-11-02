use anofox_ml_core::{Fit, Float, PartialFit, Predict, Result, RustMlError};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, AsArray, Axis, Ix2};

use crate::activation::Activation;
use crate::mlp_classifier::{fitted_accessors, mlp_builders};
use crate::network::{predict_output, DenseLayer, OutputKind};
use crate::solver::{LearningRateSchedule, Solver, StochasticOptimizer};
use crate::trainer::{initialize, rng_for, train, TrainConfig, TrainState};

/// Multi-layer perceptron regressor (unfitted). Mirrors
/// `sklearn.neural_network.MLPRegressor`: identity output, squared loss
/// `½·Σ_outputs (ŷ - y)²` averaged over the batch, plus
/// `alpha/(2·n_batch)·Σ||W||²`. Single-output via [`Fit`], multi-output via
/// [`MlpRegressor::fit_multi`].
///
/// Note: for `n_outputs > 1` sklearn reports the squared loss averaged over
/// the outputs while its gradient is not; here loss and gradient are
/// consistent (summed over outputs). Single-output losses are identical.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct MlpRegressor {
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
    /// Hold out `validation_fraction` of the data and stop when the
    /// validation R² stops improving; the best weights are kept.
    pub early_stopping: bool,
    /// Fraction of training data used for early stopping, in `(0, 1)`.
    pub validation_fraction: f64,
    /// Maximum number of loss evaluations (lbfgs).
    pub max_fun: usize,
    /// Explicit initial `(W, b)` per layer instead of random initialisation.
    #[doc(hidden)]
    pub initial_weights: Option<Vec<(Array2<f64>, Array1<f64>)>>,
}

impl Default for MlpRegressor {
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

mlp_builders!(MlpRegressor);

/// Fitted multi-layer perceptron regressor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedMlpRegressor<F: Float> {
    layers: Vec<DenseLayer<F>>,
    activation: Activation,
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

fitted_accessors!(FittedMlpRegressor);

/// sklearn `r2_score(multioutput="uniform_average")` (force_finite).
fn r2_uniform<F: Float>(pred: &Array2<F>, y: ArrayView2<F>) -> f64 {
    if pred.iter().any(|v| !v.is_finite()) {
        return f64::NAN;
    }
    let n = y.nrows() as f64;
    let mut total = 0.0;
    for j in 0..y.ncols() {
        let mean = y.column(j).iter().map(|v| v.to_f64().unwrap()).sum::<f64>() / n;
        let (mut ss_res, mut ss_tot) = (0.0, 0.0);
        for i in 0..y.nrows() {
            let t = y[[i, j]].to_f64().unwrap();
            let p = pred[[i, j]].to_f64().unwrap();
            ss_res += (t - p) * (t - p);
            ss_tot += (t - mean) * (t - mean);
        }
        total += if ss_tot == 0.0 {
            if ss_res == 0.0 {
                1.0
            } else {
                0.0
            }
        } else {
            1.0 - ss_res / ss_tot
        };
    }
    total / y.ncols() as f64
}

impl MlpRegressor {
    fn fit_impl<F: Float>(
        &self,
        prev: Option<FittedMlpRegressor<F>>,
        x: ArrayView2<F>,
        y: ArrayView2<F>,
        incremental: bool,
    ) -> Result<FittedMlpRegressor<F>> {
        if x.nrows() != y.nrows() {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but y has {} rows",
                x.nrows(),
                y.nrows()
            )));
        }
        if x.is_empty() || y.ncols() == 0 {
            return Err(RustMlError::EmptyInput("training data is empty".into()));
        }
        let cfg = self.train_config();
        cfg.validate()?;
        let n_outputs = y.ncols();
        let mut rng = rng_for(self.seed);
        let mut state = match prev {
            Some(p) => {
                if p.n_features != x.ncols() || p.n_outputs() != n_outputs {
                    return Err(RustMlError::ShapeMismatch(format!(
                        "previous model has {} features / {} outputs, got {} / {}",
                        p.n_features,
                        p.n_outputs(),
                        x.ncols(),
                        n_outputs
                    )));
                }
                p.into_state()
            }
            None => initialize(&cfg, x.ncols(), n_outputs, &mut rng)?,
        };
        let activation = self.activation;
        let scorer = move |layers: &[DenseLayer<F>], xv: ArrayView2<F>, yv: ArrayView2<F>| {
            let out = predict_output(layers, activation, OutputKind::Identity, xv);
            r2_uniform(&out, yv)
        };
        train(
            &cfg,
            &mut state,
            OutputKind::Identity,
            x,
            y,
            &mut rng,
            incremental,
            None,
            &scorer,
        )?;
        Ok(FittedMlpRegressor {
            layers: state.layers,
            activation: self.activation,
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

    /// Fit a multi-output regressor; `y` has shape `(n_samples, n_outputs)`.
    pub fn fit_multi<F: Float>(
        &self,
        x: &Array2<F>,
        y: &Array2<F>,
    ) -> Result<FittedMlpRegressor<F>> {
        self.fit_impl(None, x.view(), y.view(), false)
    }

    /// One epoch on a multi-output batch (sklearn `partial_fit`).
    pub fn partial_fit_multi<F: Float>(
        &self,
        state: Option<FittedMlpRegressor<F>>,
        x: &Array2<F>,
        y: &Array2<F>,
    ) -> Result<FittedMlpRegressor<F>> {
        self.fit_impl(state, x.view(), y.view(), true)
    }

    /// Continue training from `previous` (sklearn `warm_start=True`).
    pub fn fit_warm_start<F: Float>(
        &self,
        previous: FittedMlpRegressor<F>,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<FittedMlpRegressor<F>> {
        self.fit_impl(
            Some(previous),
            x.view(),
            y.view().insert_axis(Axis(1)),
            false,
        )
    }

    /// Multi-output variant of [`MlpRegressor::fit_warm_start`].
    pub fn fit_warm_start_multi<F: Float>(
        &self,
        previous: FittedMlpRegressor<F>,
        x: &Array2<F>,
        y: &Array2<F>,
    ) -> Result<FittedMlpRegressor<F>> {
        self.fit_impl(Some(previous), x.view(), y.view(), false)
    }
}

impl<F: Float> Fit<F> for MlpRegressor {
    type Fitted = FittedMlpRegressor<F>;

    fn fit_view(&self, x: ArrayView2<F>, y: ArrayView1<F>) -> Result<Self::Fitted> {
        if x.nrows() != y.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but y has {} elements",
                x.nrows(),
                y.len()
            )));
        }
        self.fit_impl(None, x, y.insert_axis(Axis(1)), false)
    }
}

impl<F: Float> PartialFit<F> for MlpRegressor {
    type Fitted = FittedMlpRegressor<F>;

    fn partial_fit_view(
        &self,
        state: Option<Self::Fitted>,
        x: ArrayView2<F>,
        y: ArrayView1<F>,
        _classes: Option<&[F]>,
    ) -> Result<Self::Fitted> {
        if x.nrows() != y.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but y has {} elements",
                x.nrows(),
                y.len()
            )));
        }
        self.fit_impl(state, x, y.insert_axis(Axis(1)), true)
    }
}

impl<F: Float> FittedMlpRegressor<F> {
    /// Number of regression targets.
    pub fn n_outputs(&self) -> usize {
        self.layers.last().map_or(0, |l| l.weights.ncols())
    }

    /// Predict all outputs, shape `(n_samples, n_outputs)`.
    pub fn predict_multi<'a>(&self, x: impl AsArray<'a, F, Ix2>) -> Result<Array2<F>> {
        let x: ArrayView2<F> = x.into();
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        Ok(predict_output(
            &self.layers,
            self.activation,
            OutputKind::Identity,
            x.view(),
        ))
    }
}

impl<F: Float> Predict<F> for FittedMlpRegressor<F> {
    /// Single-output prediction; use [`FittedMlpRegressor::predict_multi`]
    /// for models fitted with several outputs.
    fn predict_view(&self, x: ArrayView2<F>) -> Result<Array1<F>> {
        if self.n_outputs() != 1 {
            return Err(RustMlError::InvalidParameter(format!(
                "model has {} outputs; use predict_multi",
                self.n_outputs()
            )));
        }
        Ok(self.predict_multi(x)?.column(0).to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn simple_linear_regression() {
        // y = 2*x + 1
        let x = array![
            [0.0],
            [0.25],
            [0.5],
            [0.75],
            [1.0],
            [1.25],
            [1.5],
            [1.75],
            [2.0],
            [2.25],
        ];
        let y = array![1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5];

        let mlp = MlpRegressor {
            hidden_layer_sizes: vec![20],
            activation: Activation::Relu,
            solver: Solver::Adam,
            learning_rate: 0.01,
            max_iter: 500,
            tol: 1e-7,
            seed: 42,
            batch_size: None,
            alpha: 0.0,
            // Adam with lr=0.01 overshoots for ~10 epochs here; sklearn's
            // n_iter_no_change=10 rule would stop during that bump.
            n_iter_no_change: 500,
            ..Default::default()
        };

        let fitted: FittedMlpRegressor<f64> = mlp.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(*p, *t, epsilon = 0.5);
        }
    }

    #[test]
    fn quadratic_regression() {
        let x = array![
            [0.0],
            [0.5],
            [1.0],
            [1.5],
            [2.0],
            [2.5],
            [3.0],
            [3.5],
            [4.0],
            [4.5],
        ];
        let y = array![0.0, 0.25, 1.0, 2.25, 4.0, 6.25, 9.0, 12.25, 16.0, 20.25];

        let mlp = MlpRegressor {
            hidden_layer_sizes: vec![20, 10],
            learning_rate: 0.005,
            max_iter: 1000,
            tol: 1e-8,
            seed: 123,
            batch_size: None,
            alpha: 0.0,
            ..Default::default()
        };

        let fitted: FittedMlpRegressor<f64> = mlp.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        let mse: f64 = preds
            .iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / y.len() as f64;
        assert!(mse < 5.0, "MSE too high: {mse}");
    }

    #[test]
    fn shape_mismatch_error() {
        let x = array![[1.0, 2.0]];
        let y = array![0.0, 1.0];
        let result: std::result::Result<FittedMlpRegressor<f64>, _> =
            MlpRegressor::default().fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn empty_input_error() {
        let x: Array2<f64> = Array2::zeros((0, 2));
        let y: Array1<f64> = Array1::zeros(0);
        let result: std::result::Result<FittedMlpRegressor<f64>, _> =
            MlpRegressor::default().fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn predict_wrong_features_error() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 2.0];
        let mlp = MlpRegressor {
            hidden_layer_sizes: vec![5],
            max_iter: 5,
            ..Default::default()
        };
        let fitted: FittedMlpRegressor<f64> = mlp.fit(&x, &y).unwrap();
        assert!(fitted.predict(&array![[1.0]]).is_err());
    }

    #[test]
    fn reproducibility() {
        let x = array![[0.0], [1.0], [2.0], [3.0]];
        let y = array![0.0, 1.0, 4.0, 9.0];
        let mlp = MlpRegressor {
            hidden_layer_sizes: vec![10],
            max_iter: 50,
            seed: 42,
            batch_size: None,
            ..Default::default()
        };
        let p1 = mlp.fit(&x, &y).unwrap().predict(&x).unwrap();
        let p2 = mlp.fit(&x, &y).unwrap().predict(&x).unwrap();
        for (a, b) in p1.iter().zip(p2.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-12);
        }
    }
}
