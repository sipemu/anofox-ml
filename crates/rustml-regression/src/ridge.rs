//! Ridge (L2-regularized) regression wrapper.

use crate::convert::{col_to_ndarray, ndarray_to_col, ndarray_to_mat};
use anofox_regression::solvers::{FittedRidge, RidgeRegressor as InnerRidge};
use anofox_regression::{FittedRegressor as _, Regressor as _};
use ndarray::{Array1, Array2};
use rustml_core::{Fit, Predict, Result, RustMlError};

/// Ridge regression estimator with L2 regularization.
///
/// Minimizes: `||y - Xβ||² + λ||β||²`
#[derive(Debug, Clone)]
pub struct RidgeRegressor {
    lambda: f64,
    with_intercept: bool,
}

impl RidgeRegressor {
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

impl Default for RidgeRegressor {
    fn default() -> Self {
        Self::new()
    }
}

/// A fitted Ridge regression model.
#[derive(Debug, Clone)]
pub struct FittedRidgeRegressor {
    inner: FittedRidge,
    n_features: usize,
}

impl FittedRidgeRegressor {
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

impl Fit<f64> for RidgeRegressor {
    type Fitted = FittedRidgeRegressor;

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

        let inner_model = InnerRidge::builder()
            .with_intercept(self.with_intercept)
            .lambda(self.lambda)
            .build();

        let fitted = inner_model
            .fit(&x_mat, &y_col)
            .map_err(|e| RustMlError::InvalidParameter(e.to_string()))?;

        Ok(FittedRidgeRegressor {
            inner: fitted,
            n_features: x.ncols(),
        })
    }
}

impl Predict<f64> for FittedRidgeRegressor {
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
    fn test_ridge_basic() {
        // y = 2 + 3x
        let x = Array2::from_shape_vec((10, 1), (0..10).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_vec((0..10).map(|i| 2.0 + 3.0 * i as f64).collect());

        let fitted = RidgeRegressor::new()
            .with_lambda(0.01)
            .fit(&x, &y)
            .unwrap();

        // Ridge with small lambda should be close to OLS
        assert!(fitted.r_squared() > 0.99);
        assert_abs_diff_eq!(fitted.coefficients()[0], 3.0, epsilon = 0.1);
    }

    #[test]
    fn test_ridge_shrinks_coefficients() {
        let x = Array2::from_shape_vec((10, 1), (0..10).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_vec((0..10).map(|i| 2.0 + 3.0 * i as f64).collect());

        let fitted_small = RidgeRegressor::new()
            .with_lambda(0.01)
            .fit(&x, &y)
            .unwrap();
        let fitted_large = RidgeRegressor::new()
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
    fn test_ridge_negative_lambda() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = RidgeRegressor::new().with_lambda(-1.0).fit(&x, &y);
        assert!(result.is_err());
    }
}
