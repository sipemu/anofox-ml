//! Linear Support Vector Regression (LinearSVR) using SGD on the primal.

use ndarray::{Array1, Array2};
use rustml_core::{Fit, Float, Predict, Result, RustMlError};

/// Linear Support Vector Regressor parameters (unfitted state).
///
/// Solves the epsilon-insensitive SVR objective in the primal via
/// stochastic gradient descent:
///
/// `min  0.5 * ||w||^2  +  C * sum(max(0, |y_i - w . x_i - b| - epsilon))`
///
/// Uses the type-state pattern: call [`Fit::fit`] to produce a
/// [`FittedLinearSvr`] that can make predictions.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LinearSvr {
    /// Regularization parameter. Larger values penalize errors more.
    pub c: f64,
    /// Width of the epsilon-insensitive tube.
    pub epsilon: f64,
    /// Maximum number of SGD epochs.
    pub max_iter: usize,
    /// Tolerance for the stopping criterion (change in loss).
    pub tol: f64,
}

impl LinearSvr {
    /// Create a new `LinearSvr` with default parameters.
    pub fn new() -> Self {
        Self {
            c: 1.0,
            epsilon: 0.1,
            max_iter: 1000,
            tol: 1e-4,
        }
    }

    /// Set the regularization parameter C.
    pub fn with_c(mut self, c: f64) -> Self {
        self.c = c;
        self
    }

    /// Set the width of the epsilon-insensitive tube.
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set the maximum number of SGD epochs.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the tolerance for the stopping criterion.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Validate parameters before fitting.
    fn validate(&self) -> Result<()> {
        if self.c <= 0.0 {
            return Err(RustMlError::InvalidParameter("C must be positive".into()));
        }
        if self.epsilon < 0.0 {
            return Err(RustMlError::InvalidParameter(
                "epsilon must be non-negative".into(),
            ));
        }
        if self.max_iter == 0 {
            return Err(RustMlError::InvalidParameter(
                "max_iter must be at least 1".into(),
            ));
        }
        if self.tol <= 0.0 {
            return Err(RustMlError::InvalidParameter(
                "tol must be positive".into(),
            ));
        }
        Ok(())
    }
}

impl Default for LinearSvr {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Linear Support Vector Regressor.
///
/// Stores the learned weight vector and bias term for making predictions
/// via `y = w . x + b`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedLinearSvr<F: Float> {
    /// Weight vector, shape `(n_features,)`.
    weights: Array1<F>,
    /// Bias (intercept) term.
    bias: F,
    /// Number of features expected at prediction time.
    n_features: usize,
}

impl<F: Float> FittedLinearSvr<F> {
    /// Returns a reference to the weight vector.
    pub fn weights(&self) -> &Array1<F> {
        &self.weights
    }

    /// Returns the bias (intercept) term.
    pub fn bias(&self) -> F {
        self.bias
    }

    /// Returns the number of features the model was trained on.
    pub fn n_features(&self) -> usize {
        self.n_features
    }
}

impl<F: Float> Predict<F> for FittedLinearSvr<F> {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        if x.is_empty() {
            return Err(RustMlError::EmptyInput(
                "prediction input must not be empty".into(),
            ));
        }
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }

        let predictions: Vec<F> = x
            .rows()
            .into_iter()
            .map(|row| row.dot(&self.weights) + self.bias)
            .collect();

        Ok(Array1::from_vec(predictions))
    }
}

/// Train a linear SVR model using stochastic gradient descent on the primal.
///
/// For each epoch, iterates over all samples and applies sub-gradient
/// updates for the epsilon-insensitive loss plus L2 regularization.
fn fit_linear_svr<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    c: F,
    epsilon: F,
    max_iter: usize,
    tol: F,
) -> (Array1<F>, F) {
    let n_samples = x.nrows();
    let n_features = x.ncols();
    let zero = F::zero();
    let one = F::one();
    let n_f = F::from_usize(n_samples).unwrap();

    let mut w = Array1::<F>::zeros(n_features);
    let mut b = zero;

    // Initial learning rate; decays as 1/epoch.
    let lr_0 = F::from_f64(0.01).unwrap();

    let mut prev_loss = F::from_f64(f64::MAX).unwrap();

    for epoch in 0..max_iter {
        let lr = lr_0 / (one + F::from_usize(epoch).unwrap() * F::from_f64(0.001).unwrap());

        for i in 0..n_samples {
            let xi = x.row(i);
            let residual = xi.dot(&w) + b - y[i];
            let abs_residual = residual.abs();

            if abs_residual > epsilon {
                // Sub-gradient of epsilon-insensitive loss
                let sign = if residual > zero { one } else { -one };
                // Update weights: SGD step for regularization + loss
                // gradient of 0.5*||w||^2 is w, scaled by 1/n for per-sample
                // gradient of C * max(0, |r| - eps) is C * sign(r) * x_i
                let grad_w = &w / n_f + &xi * (c * sign);
                w = w - &grad_w * lr;
                b = b - lr * c * sign;
            } else {
                // Only regularization decay
                w = &w * (one - lr / n_f);
            }
        }

        // Compute loss for convergence check
        let mut loss = F::from_f64(0.5).unwrap() * w.dot(&w);
        for i in 0..n_samples {
            let residual = (x.row(i).dot(&w) + b - y[i]).abs();
            if residual > epsilon {
                loss = loss + c * (residual - epsilon);
            }
        }

        let change = (prev_loss - loss).abs();
        if change < tol && epoch > 0 {
            break;
        }
        prev_loss = loss;
    }

    (w, b)
}

impl<F: Float> Fit<F> for LinearSvr {
    type Fitted = FittedLinearSvr<F>;

    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Self::Fitted> {
        self.validate()?;

        if x.is_empty() || y.is_empty() {
            return Err(RustMlError::EmptyInput(
                "training data must not be empty".into(),
            ));
        }
        if x.nrows() != y.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but y has {} elements",
                x.nrows(),
                y.len()
            )));
        }

        let c = F::from_f64(self.c).unwrap();
        let epsilon = F::from_f64(self.epsilon).unwrap();
        let tol = F::from_f64(self.tol).unwrap();

        let (weights, bias) = fit_linear_svr(x, y, c, epsilon, self.max_iter, tol);

        Ok(FittedLinearSvr {
            n_features: x.ncols(),
            weights,
            bias,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_basic_linear_regression() {
        // y = 2*x + 1
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0],
            [10.0]
        ];
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0];

        let svr = LinearSvr::new()
            .with_c(10.0)
            .with_epsilon(0.1)
            .with_max_iter(5000);
        let fitted: FittedLinearSvr<f64> = svr.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(*p, *t, epsilon = 2.0);
        }
    }

    #[test]
    fn test_epsilon_tube() {
        // Points inside the epsilon tube should not generate loss,
        // so a larger epsilon should allow more slack / smaller weights.
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let svr_small_eps = LinearSvr::new()
            .with_c(10.0)
            .with_epsilon(0.01)
            .with_max_iter(3000);
        let fitted_small: FittedLinearSvr<f64> = svr_small_eps.fit(&x, &y).unwrap();

        let svr_large_eps = LinearSvr::new()
            .with_c(10.0)
            .with_epsilon(5.0)
            .with_max_iter(3000);
        let fitted_large: FittedLinearSvr<f64> = svr_large_eps.fit(&x, &y).unwrap();

        // With a very large epsilon, most points are inside the tube
        // so the weight norm should be smaller (more regularization dominates).
        let norm_small = fitted_small.weights().dot(fitted_small.weights());
        let norm_large = fitted_large.weights().dot(fitted_large.weights());
        assert!(
            norm_large < norm_small,
            "larger epsilon should yield smaller weight norm: large_eps={}, small_eps={}",
            norm_large,
            norm_small
        );
    }

    #[test]
    fn test_c_effect() {
        // Higher C should fit the data more closely (lower training error).
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let svr_low_c = LinearSvr::new()
            .with_c(0.001)
            .with_epsilon(0.01)
            .with_max_iter(3000);
        let fitted_low: FittedLinearSvr<f64> = svr_low_c.fit(&x, &y).unwrap();

        let svr_high_c = LinearSvr::new()
            .with_c(100.0)
            .with_epsilon(0.01)
            .with_max_iter(3000);
        let fitted_high: FittedLinearSvr<f64> = svr_high_c.fit(&x, &y).unwrap();

        let preds_low = fitted_low.predict(&x).unwrap();
        let preds_high = fitted_high.predict(&x).unwrap();

        let mse_low: f64 = preds_low
            .iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / y.len() as f64;
        let mse_high: f64 = preds_high
            .iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / y.len() as f64;

        assert!(
            mse_high < mse_low,
            "higher C should yield lower training MSE: high_c_mse={}, low_c_mse={}",
            mse_high,
            mse_low
        );
    }

    #[test]
    fn test_epsilon_effect() {
        // Larger epsilon means predictions can be further from targets
        // without penalty, so training error should generally be larger.
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let svr_small = LinearSvr::new()
            .with_c(10.0)
            .with_epsilon(0.01)
            .with_max_iter(3000);
        let fitted_small: FittedLinearSvr<f64> = svr_small.fit(&x, &y).unwrap();

        let svr_large = LinearSvr::new()
            .with_c(10.0)
            .with_epsilon(3.0)
            .with_max_iter(3000);
        let fitted_large: FittedLinearSvr<f64> = svr_large.fit(&x, &y).unwrap();

        let preds_small = fitted_small.predict(&x).unwrap();
        let preds_large = fitted_large.predict(&x).unwrap();

        let mse_small: f64 = preds_small
            .iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / y.len() as f64;
        let mse_large: f64 = preds_large
            .iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / y.len() as f64;

        assert!(
            mse_small <= mse_large + 1.0,
            "smaller epsilon should generally yield tighter fit: small={}, large={}",
            mse_small,
            mse_large
        );
    }

    #[test]
    fn test_predict_shape_mismatch() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 2.0];

        let svr = LinearSvr::new();
        let fitted: FittedLinearSvr<f64> = svr.fit(&x, &y).unwrap();

        // 3 features instead of 2
        let x_bad = array![[1.0, 2.0, 3.0]];
        let result = fitted.predict(&x_bad);
        assert!(result.is_err());
        match result {
            Err(RustMlError::ShapeMismatch(_)) => {}
            other => panic!("expected ShapeMismatch error, got {:?}", other),
        }
    }

    #[test]
    fn test_empty_input() {
        // Empty training data
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<f64>::zeros(0);

        let svr = LinearSvr::new();
        let result: Result<FittedLinearSvr<f64>> = svr.fit(&x, &y);
        assert!(result.is_err());
        match result {
            Err(RustMlError::EmptyInput(_)) => {}
            other => panic!("expected EmptyInput error, got {:?}", other),
        }

        // Empty prediction data
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let y_train = array![1.0, 2.0];
        let fitted: FittedLinearSvr<f64> = svr.fit(&x_train, &y_train).unwrap();

        let x_empty = Array2::<f64>::zeros((0, 2));
        let result = fitted.predict(&x_empty);
        assert!(result.is_err());
        match result {
            Err(RustMlError::EmptyInput(_)) => {}
            other => panic!("expected EmptyInput error, got {:?}", other),
        }
    }

    #[test]
    fn test_f32_support() {
        let x: Array2<f32> = array![[1.0f32], [2.0], [3.0], [4.0], [5.0]];
        let y: Array1<f32> = array![2.0f32, 4.0, 6.0, 8.0, 10.0];

        let svr = LinearSvr::new()
            .with_c(10.0)
            .with_epsilon(0.1)
            .with_max_iter(3000);
        let fitted: FittedLinearSvr<f32> = svr.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 5);
        for &p in preds.iter() {
            assert!(p.is_finite(), "prediction should be finite, got {}", p);
        }
    }

    #[test]
    fn test_builder_pattern() {
        let svr = LinearSvr::new()
            .with_c(0.5)
            .with_epsilon(0.2)
            .with_max_iter(500)
            .with_tol(1e-3);
        assert_eq!(svr.c, 0.5);
        assert_eq!(svr.epsilon, 0.2);
        assert_eq!(svr.max_iter, 500);
        assert_eq!(svr.tol, 1e-3);
    }

    #[test]
    fn test_default() {
        let svr = LinearSvr::default();
        assert_eq!(svr.c, 1.0);
        assert_eq!(svr.epsilon, 0.1);
        assert_eq!(svr.max_iter, 1000);
        assert_eq!(svr.tol, 1e-4);
    }

    #[test]
    fn test_multidimensional_regression() {
        // y = x0 + 2*x1
        let x = array![
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0],
            [3.0, 3.0],
            [4.0, 2.0],
            [2.0, 4.0],
            [5.0, 1.0],
            [1.0, 5.0]
        ];
        let y = array![3.0, 4.0, 5.0, 9.0, 8.0, 10.0, 7.0, 11.0];

        let svr = LinearSvr::new()
            .with_c(50.0)
            .with_epsilon(0.1)
            .with_max_iter(5000);
        let fitted: FittedLinearSvr<f64> = svr.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(*p, *t, epsilon = 3.0);
        }
    }

    #[test]
    fn test_fit_shape_mismatch() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 2.0, 3.0]; // wrong length

        let svr = LinearSvr::new();
        let result: Result<FittedLinearSvr<f64>> = svr.fit(&x, &y);
        assert!(result.is_err());
        match result {
            Err(RustMlError::ShapeMismatch(_)) => {}
            other => panic!("expected ShapeMismatch error, got {:?}", other),
        }
    }

    #[test]
    fn test_invalid_parameters() {
        let x = array![[1.0], [2.0]];
        let y = array![1.0, 2.0];

        // Invalid C
        let svr = LinearSvr::new().with_c(-1.0);
        assert!(Fit::<f64>::fit(&svr, &x, &y).is_err());

        // Invalid epsilon
        let svr = LinearSvr::new().with_epsilon(-0.1);
        assert!(Fit::<f64>::fit(&svr, &x, &y).is_err());

        // Invalid max_iter
        let svr = LinearSvr::new().with_max_iter(0);
        assert!(Fit::<f64>::fit(&svr, &x, &y).is_err());

        // Invalid tol
        let svr = LinearSvr::new().with_tol(-1e-4);
        assert!(Fit::<f64>::fit(&svr, &x, &y).is_err());
    }

    #[test]
    fn test_accessors() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![1.0, 2.0, 3.0];

        let svr = LinearSvr::new();
        let fitted: FittedLinearSvr<f64> = svr.fit(&x, &y).unwrap();

        assert_eq!(fitted.n_features(), 2);
        assert_eq!(fitted.weights().len(), 2);
        assert!(fitted.bias().is_finite());
    }
}
