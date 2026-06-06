//! SGD-based linear regressor.
//!
//! Supports squared_error, huber, and epsilon_insensitive loss functions,
//! trained with stochastic gradient descent.

use crate::sgd_common::{compute_lr, penalty_gradient, LearningRate, Penalty};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rustml_core::{Fit, Float, Predict, Result, RustMlError};

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

/// Stochastic Gradient Descent regressor.
///
/// Linear regressor trained via SGD, supporting multiple loss functions
/// and regularization options.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SgdRegressor {
    pub loss: RegressorLoss,
    pub penalty: Penalty,
    pub alpha: f64,
    pub l1_ratio: f64,
    pub max_iter: usize,
    pub tol: f64,
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
            tol: 1e-3,
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
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
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
}

impl Default for SgdRegressor {
    fn default() -> Self {
        Self::new()
    }
}

/// A fitted SGD linear regressor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedSgdRegressor<F: Float> {
    weights: Array1<F>,
    bias: F,
    n_features: usize,
}

impl<F: Float> FittedSgdRegressor<F> {
    /// Return the weight vector.
    pub fn weights(&self) -> &Array1<F> {
        &self.weights
    }

    /// Return the bias (intercept).
    pub fn bias(&self) -> F {
        self.bias
    }
}

impl Fit<f64> for SgdRegressor {
    type Fitted = FittedSgdRegressor<f64>;

    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
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

        let n = x.nrows();
        let p = x.ncols();
        let mut w = Array1::zeros(p);
        let mut b = 0.0;
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut indices: Vec<usize> = (0..n).collect();
        let mut t: usize = 1;
        let eps = self.epsilon;

        for _epoch in 0..self.max_iter {
            if self.shuffle {
                indices.shuffle(&mut rng);
            }

            let mut total_loss = 0.0;

            for &i in &indices {
                let eta = compute_lr(self.learning_rate, self.eta0, self.alpha, t, self.power_t);
                t += 1;

                // Compute prediction: z = w · x_i + b
                let mut z = b;
                for j in 0..p {
                    z += w[j] * x[[i, j]];
                }
                let r = z - y[i]; // residual

                // Compute loss gradient w.r.t. z
                let dloss = match self.loss {
                    RegressorLoss::SquaredError => {
                        total_loss += 0.5 * r * r;
                        r
                    }
                    RegressorLoss::Huber => {
                        if r.abs() <= eps {
                            total_loss += 0.5 * r * r;
                            r
                        } else {
                            total_loss += eps * r.abs() - 0.5 * eps * eps;
                            eps * r.signum()
                        }
                    }
                    RegressorLoss::EpsilonInsensitive => {
                        if r.abs() <= eps {
                            0.0
                        } else {
                            total_loss += r.abs() - eps;
                            r.signum()
                        }
                    }
                };

                // Update weights
                if dloss != 0.0 {
                    for j in 0..p {
                        w[j] -= eta
                            * (dloss * x[[i, j]]
                                + penalty_gradient(w[j], self.alpha, self.penalty, self.l1_ratio));
                    }
                    b -= eta * dloss;
                } else {
                    for j in 0..p {
                        w[j] -=
                            eta * penalty_gradient(w[j], self.alpha, self.penalty, self.l1_ratio);
                    }
                }
            }

            let avg_loss = total_loss / n as f64;
            if avg_loss < self.tol {
                break;
            }
        }

        Ok(FittedSgdRegressor {
            weights: w,
            bias: b,
            n_features: p,
        })
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
        let n = x.nrows();
        let mut preds = Array1::zeros(n);
        for i in 0..n {
            let mut z = self.bias;
            for j in 0..self.n_features {
                z += self.weights[j] * x[[i, j]];
            }
            preds[i] = z;
        }
        Ok(preds)
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

impl rustml_core::RegressorScore<f64> for FittedSgdRegressor<f64> {}
