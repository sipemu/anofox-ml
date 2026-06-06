//! Lasso (L1-regularized) regression wrapper.
//!
//! This is a convenience wrapper around `ElasticNetRegressor` with `alpha = 1.0`
//! (pure L1 regularization).

use crate::convert::{col_to_ndarray, ndarray_to_col, ndarray_to_mat};
use anofox_regression::solvers::{ElasticNetRegressor as InnerElasticNet, FittedElasticNet};
use anofox_regression::{FittedRegressor as _, Regressor as _};
use ndarray::{Array1, Array2};
use rustml_core::{Fit, Predict, Result, RustMlError};

/// Lasso regression estimator with pure L1 regularization.
///
/// Minimizes: `||y - Xβ||² + λ||β||₁`
///
/// This is equivalent to Elastic Net with `alpha = 1.0`.
#[derive(Debug, Clone)]
pub struct LassoRegressor {
    lambda: f64,
    with_intercept: bool,
}

impl LassoRegressor {
    pub fn new() -> Self {
        Self {
            lambda: 1.0,
            with_intercept: true,
        }
    }

    pub fn with_lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda;
        self
    }

    pub fn with_intercept(mut self, include: bool) -> Self {
        self.with_intercept = include;
        self
    }
}

impl Default for LassoRegressor {
    fn default() -> Self {
        Self::new()
    }
}

/// A fitted Lasso regression model.
#[derive(Debug, Clone)]
pub struct FittedLassoRegressor {
    inner: FittedElasticNet,
    n_features: usize,
}

impl FittedLassoRegressor {
    pub fn coefficients(&self) -> Array1<f64> {
        col_to_ndarray(self.inner.coefficients())
    }

    pub fn intercept(&self) -> Option<f64> {
        self.inner.intercept()
    }

    pub fn r_squared(&self) -> f64 {
        self.inner.r_squared()
    }
}

impl Fit<f64> for LassoRegressor {
    type Fitted = FittedLassoRegressor;

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
        if self.lambda < 0.0 {
            return Err(RustMlError::InvalidParameter(
                "lambda must be non-negative".into(),
            ));
        }

        let x_mat = ndarray_to_mat(x);
        let y_col = ndarray_to_col(y);

        let inner_model = InnerElasticNet::builder()
            .with_intercept(self.with_intercept)
            .lambda(self.lambda)
            .alpha(1.0)
            .build();

        let fitted = inner_model
            .fit(&x_mat, &y_col)
            .map_err(|e| RustMlError::InvalidParameter(e.to_string()))?;

        Ok(FittedLassoRegressor {
            inner: fitted,
            n_features: x.ncols(),
        })
    }
}

impl Predict<f64> for FittedLassoRegressor {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }

        let x_mat = ndarray_to_mat(x);
        let preds = self.inner.predict(&x_mat);
        Ok(col_to_ndarray(&preds))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_lasso_basic() {
        // y = 2 + 3x
        let x = Array2::from_shape_vec((10, 1), (0..10).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_vec((0..10).map(|i| 2.0 + 3.0 * i as f64).collect());

        let fitted = LassoRegressor::new().with_lambda(0.01).fit(&x, &y).unwrap();

        // Lasso with small lambda should be close to OLS
        assert!(fitted.r_squared() > 0.99);
        assert_abs_diff_eq!(fitted.coefficients()[0], 3.0, epsilon = 0.1);
    }

    #[test]
    fn test_lasso_shrinks_coefficients() {
        let x = Array2::from_shape_vec((10, 1), (0..10).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_vec((0..10).map(|i| 2.0 + 3.0 * i as f64).collect());

        let fitted_small = LassoRegressor::new().with_lambda(0.01).fit(&x, &y).unwrap();
        let fitted_large = LassoRegressor::new()
            .with_lambda(100.0)
            .fit(&x, &y)
            .unwrap();

        // Larger lambda should shrink coefficients more
        assert!(
            fitted_large.coefficients()[0].abs() < fitted_small.coefficients()[0].abs(),
            "larger lambda should shrink coefficients: small={}, large={}",
            fitted_small.coefficients()[0],
            fitted_large.coefficients()[0]
        );
    }

    #[test]
    fn test_lasso_sparsity() {
        // Lasso with large lambda should push some coefficients toward zero
        let x = Array2::from_shape_vec((20, 3), (0..60).map(|i| i as f64 * 0.1).collect()).unwrap();
        let y = Array1::from_vec((0..20).map(|i| 1.0 + 2.0 * i as f64 * 0.1).collect());

        let fitted = LassoRegressor::new().with_lambda(10.0).fit(&x, &y).unwrap();

        // At least some coefficients should be very small (near zero)
        let coeffs = fitted.coefficients();
        let near_zero_count = coeffs.iter().filter(|c| c.abs() < 0.01).count();
        assert!(
            near_zero_count > 0,
            "Lasso with large lambda should produce sparse coefficients, got {:?}",
            coeffs
        );
    }

    #[test]
    fn test_lasso_negative_lambda() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = LassoRegressor::new().with_lambda(-1.0).fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_lasso_shape_mismatch() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];

        let result = LassoRegressor::new().fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_lasso_empty_input() {
        let x = Array2::<f64>::zeros((0, 1));
        let y = Array1::<f64>::zeros(0);

        let result = LassoRegressor::new().fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_lasso_predict_wrong_features() {
        let x = Array2::from_shape_vec((5, 2), vec![0.0; 10]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let fitted = LassoRegressor::new().with_lambda(0.01).fit(&x, &y).unwrap();

        let x_wrong = Array2::from_shape_vec((3, 3), vec![0.0; 9]).unwrap();
        assert!(fitted.predict(&x_wrong).is_err());
    }

    #[test]
    fn test_lasso_no_intercept() {
        let x = Array2::from_shape_vec((10, 1), (0..10).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_vec((0..10).map(|i| 3.0 * i as f64).collect());

        let fitted = LassoRegressor::new()
            .with_lambda(0.01)
            .with_intercept(false)
            .fit(&x, &y)
            .unwrap();

        assert!(fitted.intercept().is_none());
    }

    #[test]
    fn test_lasso_default() {
        let lasso = LassoRegressor::default();
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        // Default should work without errors
        let result = lasso.fit(&x, &y);
        assert!(result.is_ok());
    }
}

impl rustml_core::RegressorScore<f64> for FittedLassoRegressor {}
