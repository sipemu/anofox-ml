//! SGD-based linear regressor.
//!
//! Supports squared_error, huber, and epsilon_insensitive loss functions,
//! trained with stochastic gradient descent. Stopping, `partial_fit`,
//! truncated-gradient L1 and averaging (ASGD) follow scikit-learn's
//! `SGDRegressor` semantics.

use crate::sgd_common::{
    default_n_iter_no_change, default_t, default_validation_fraction, run_sgd, validation_split,
    LearningRate, Penalty, SgdConfig, SgdWeights, Stopping,
};
use anofox_ml_core::{Fit, Float, PartialFit, Predict, Result, RustMlError};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Loss function for SGD regressor.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum RegressorLoss {
    /// Squared error (L2 loss): `0.5 * (y - f(x))^2`.
    SquaredError,
    /// Huber loss: squared for |r| <= epsilon, linear otherwise.
    Huber,
    /// Epsilon-insensitive: zero loss for |r| <= epsilon (SVR-style).
    EpsilonInsensitive,
}

impl Default for RegressorLoss {
    fn default() -> Self {
        RegressorLoss::SquaredError
    }
}

impl RegressorLoss {
    /// `(loss, dloss/dz)` for prediction `z` and target `y`.
    #[inline]
    fn eval(self, z: f64, y: f64, eps: f64) -> (f64, f64) {
        let r = z - y;
        match self {
            RegressorLoss::SquaredError => (0.5 * r * r, r),
            RegressorLoss::Huber => {
                if r.abs() <= eps {
                    (0.5 * r * r, r)
                } else {
                    (eps * r.abs() - 0.5 * eps * eps, eps * r.signum())
                }
            }
            RegressorLoss::EpsilonInsensitive => {
                if r.abs() <= eps {
                    (0.0, 0.0)
                } else {
                    (r.abs() - eps, r.signum())
                }
            }
        }
    }
}

/// Stochastic Gradient Descent regressor.
///
/// Linear regressor trained via SGD, supporting multiple loss functions
/// and regularization options.
///
/// Stopping (sklearn semantics): an epoch whose average training loss is
/// `> best_loss - tol` increments a no-improvement counter, any other epoch
/// resets it; training stops once the counter reaches `n_iter_no_change`.
/// With `early_stopping = true` a `validation_fraction` of the data is held
/// out and its R² is used instead (`score < best_score + tol` counts as no
/// improvement). `tol = None` disables the criterion.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SgdRegressor {
    pub loss: RegressorLoss,
    pub penalty: Penalty,
    pub alpha: f64,
    pub l1_ratio: f64,
    pub max_iter: usize,
    /// Stopping tolerance; `None` disables the stopping criterion.
    pub tol: Option<f64>,
    /// Number of epochs without improvement before stopping (default 5).
    #[serde(default = "default_n_iter_no_change")]
    pub n_iter_no_change: usize,
    /// Hold out `validation_fraction` of the data and stop on its R².
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
    /// Epsilon parameter for Huber and EpsilonInsensitive losses.
    pub epsilon: f64,
    pub shuffle: bool,
    pub seed: u64,
}

impl SgdRegressor {
    pub fn new() -> Self {
        Self {
            loss: RegressorLoss::SquaredError,
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
            power_t: 0.25,
            learning_rate: LearningRate::InvScaling,
            epsilon: 0.1,
            shuffle: true,
            seed: 0,
        }
    }

    pub fn with_loss(mut self, loss: RegressorLoss) -> Self {
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
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
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

    fn validate(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
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

impl Default for SgdRegressor {
    fn default() -> Self {
        Self::new()
    }
}

/// A fitted SGD linear regressor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedSgdRegressor<F: Float> {
    state: SgdWeights,
    n_features: usize,
    /// Epochs run by the last `fit` / `partial_fit` call.
    #[serde(default)]
    n_iter_: usize,
    /// Learning-rate step counter (next `t`); persists across `partial_fit`.
    #[serde(default = "default_t")]
    t_: usize,
    #[serde(skip)]
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> FittedSgdRegressor<F> {
    /// Return the weight vector (the averaged one when `average > 0`).
    pub fn weights(&self) -> &Array1<f64> {
        self.state.effective().0
    }

    /// Return the bias (intercept; averaged when `average > 0`).
    pub fn bias(&self) -> f64 {
        self.state.effective().1
    }

    /// Number of epochs run by the last `fit` / `partial_fit` call
    /// (sklearn `n_iter_`).
    pub fn n_iter(&self) -> usize {
        self.n_iter_
    }

    /// Learning-rate step counter: `1 + number of SGD updates performed`
    /// (sklearn `t_`).
    pub fn t(&self) -> usize {
        self.t_
    }
}

/// R² of `(w, b)` on the rows `idx` (higher is better).
fn r2_on(x: &Array2<f64>, y: &Array1<f64>, idx: &[usize], w: &Array1<f64>, b: f64) -> f64 {
    let n = idx.len() as f64;
    let mean = idx.iter().map(|&i| y[i]).sum::<f64>() / n;
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    for &i in idx {
        let pred = b + x.row(i).dot(w);
        ss_res += (y[i] - pred).powi(2);
        ss_tot += (y[i] - mean).powi(2);
    }
    if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        -ss_res
    }
}

impl SgdRegressor {
    fn train(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        state: Option<FittedSgdRegressor<f64>>,
        partial: bool,
    ) -> Result<FittedSgdRegressor<f64>> {
        self.validate(x, y)?;
        let p = x.ncols();
        let (mut weights, t_start) = match state {
            Some(s) => {
                if s.n_features != p {
                    return Err(RustMlError::ShapeMismatch(format!(
                        "previous fit had {} features, batch has {}",
                        s.n_features, p
                    )));
                }
                (s.state, s.t_)
            }
            None => (SgdWeights::zeros(p), 1),
        };
        let cfg = self.config();
        let eps = self.epsilon;
        let loss = self.loss;
        let ys = y
            .as_slice()
            .map(|s| s.to_vec())
            .unwrap_or_else(|| y.to_vec());
        let loss_fn = |z: f64, yi: f64| loss.eval(z, yi, eps);

        let run = if partial {
            // One epoch per call; reseed per call so successive shuffles differ.
            let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(t_start as u64));
            let mut idx: Vec<usize> = (0..x.nrows()).collect();
            run_sgd(
                x,
                &ys,
                &mut idx,
                &mut weights,
                &cfg,
                loss_fn,
                1,
                &Stopping::never(),
                self.shuffle,
                &mut rng,
                t_start,
            )
        } else {
            let mut rng = StdRng::seed_from_u64(self.seed);
            if self.early_stopping {
                let (mut train_idx, val_idx) =
                    validation_split(x.nrows(), self.validation_fraction, None, &mut rng)?;
                let score = |w: &Array1<f64>, b: f64| r2_on(x, y, &val_idx, w, b);
                let stopping = Stopping {
                    tol: self.tol,
                    n_iter_no_change: self.n_iter_no_change,
                    validation: Some(&score),
                };
                run_sgd(
                    x,
                    &ys,
                    &mut train_idx,
                    &mut weights,
                    &cfg,
                    loss_fn,
                    self.max_iter,
                    &stopping,
                    self.shuffle,
                    &mut rng,
                    t_start,
                )
            } else {
                let mut idx: Vec<usize> = (0..x.nrows()).collect();
                let stopping = Stopping {
                    tol: self.tol,
                    n_iter_no_change: self.n_iter_no_change,
                    validation: None,
                };
                run_sgd(
                    x,
                    &ys,
                    &mut idx,
                    &mut weights,
                    &cfg,
                    loss_fn,
                    self.max_iter,
                    &stopping,
                    self.shuffle,
                    &mut rng,
                    t_start,
                )
            }
        };

        Ok(FittedSgdRegressor {
            state: weights,
            n_features: p,
            n_iter_: run.n_iter,
            t_: run.t,
            _marker: std::marker::PhantomData,
        })
    }
}

impl Fit<f64> for SgdRegressor {
    type Fitted = FittedSgdRegressor<f64>;

    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        self.train(x, y, None, false)
    }
}

impl PartialFit<f64> for SgdRegressor {
    type Fitted = FittedSgdRegressor<f64>;

    /// Run exactly one SGD epoch over the batch, resuming from `state`
    /// (weights, averaged weights and the learning-rate counter `t_`).
    /// `classes` is ignored; `tol` / `early_stopping` do not apply.
    fn partial_fit(
        &self,
        state: Option<Self::Fitted>,
        x: &Array2<f64>,
        y: &Array1<f64>,
        _classes: Option<&[f64]>,
    ) -> Result<Self::Fitted> {
        self.train(x, y, state, true)
    }
}

impl Predict<f64> for FittedSgdRegressor<f64> {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        let (w, b) = self.state.effective();
        Ok(x.dot(w) + b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_linear_data() -> (Array2<f64>, Array1<f64>) {
        // y = 2 + 3*x1 + 0.5*x2
        let x = Array2::from_shape_vec(
            (20, 2),
            (0..20)
                .flat_map(|i| vec![i as f64, (i as f64) * 0.5])
                .collect(),
        )
        .unwrap();
        let y = Array1::from_vec(
            (0..20)
                .map(|i| 2.0 + 3.0 * (i as f64) + 0.5 * (i as f64) * 0.5)
                .collect(),
        );
        (x, y)
    }

    #[test]
    fn test_sgd_regressor_squared_error() {
        let (x, y) = make_linear_data();
        let reg = SgdRegressor::new()
            .with_loss(RegressorLoss::SquaredError)
            .with_max_iter(2000)
            .with_eta0(0.001)
            .with_learning_rate(LearningRate::Constant)
            .with_alpha(0.0);
        let fitted = reg.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        // Check that predictions are reasonable (not exact due to SGD noise)
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(*p, *t, epsilon = 5.0);
        }
    }

    #[test]
    fn test_sgd_regressor_huber() {
        let (x, y) = make_linear_data();
        let reg = SgdRegressor::new()
            .with_loss(RegressorLoss::Huber)
            .with_epsilon(1.0)
            .with_max_iter(2000)
            .with_eta0(0.001)
            .with_learning_rate(LearningRate::Constant)
            .with_alpha(0.0);
        let fitted = reg.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 20);
        for &p in preds.iter() {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn test_sgd_regressor_epsilon_insensitive() {
        let (x, y) = make_linear_data();
        let reg = SgdRegressor::new()
            .with_loss(RegressorLoss::EpsilonInsensitive)
            .with_epsilon(0.5)
            .with_max_iter(2000)
            .with_eta0(0.001)
            .with_learning_rate(LearningRate::Constant)
            .with_alpha(0.0);
        let fitted = reg.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 20);
    }

    #[test]
    fn test_sgd_regressor_l1_penalty() {
        let (x, y) = make_linear_data();
        let fitted = SgdRegressor::new()
            .with_penalty(Penalty::L1)
            .with_alpha(0.01)
            .with_max_iter(500)
            .fit(&x, &y)
            .unwrap();

        // L1 should produce sparser weights than no regularization
        assert!(fitted.weights().iter().all(|w| w.is_finite()));
    }

    #[test]
    fn test_sgd_regressor_elastic_net() {
        let (x, y) = make_linear_data();
        let fitted = SgdRegressor::new()
            .with_penalty(Penalty::ElasticNet)
            .with_l1_ratio(0.5)
            .with_max_iter(500)
            .fit(&x, &y)
            .unwrap();

        assert_eq!(fitted.weights().len(), 2);
    }

    #[test]
    fn test_sgd_regressor_inv_scaling() {
        let (x, y) = make_linear_data();
        let fitted = SgdRegressor::new()
            .with_learning_rate(LearningRate::InvScaling)
            .with_eta0(0.1)
            .with_power_t(0.25)
            .with_max_iter(1000)
            .fit(&x, &y)
            .unwrap();

        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 20);
    }

    #[test]
    fn test_sgd_regressor_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 2), vec![0.0; 6]).unwrap();
        let y = Array1::from_vec(vec![1.0, 2.0]);
        assert!(SgdRegressor::new().fit(&x, &y).is_err());
    }

    #[test]
    fn test_sgd_regressor_empty_input() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<f64>::zeros(0);
        assert!(SgdRegressor::new().fit(&x, &y).is_err());
    }

    #[test]
    fn test_sgd_regressor_predict_shape_mismatch() {
        let (x, y) = make_linear_data();
        let fitted = SgdRegressor::new().with_max_iter(10).fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((2, 3), vec![0.0; 6]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_sgd_regressor_weights_and_bias() {
        let (x, y) = make_linear_data();
        let fitted = SgdRegressor::new().with_max_iter(100).fit(&x, &y).unwrap();

        assert_eq!(fitted.weights().len(), 2);
        assert!(fitted.bias().is_finite());
    }
}

impl anofox_ml_core::RegressorScore<f64> for FittedSgdRegressor<f64> {}
