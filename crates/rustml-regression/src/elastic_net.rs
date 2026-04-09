//! Elastic Net regression wrapper (combined L1 and L2 regularization).

use crate::convert::{col_to_ndarray, ndarray_to_col, ndarray_to_mat};
use anofox_regression::solvers::{ElasticNetRegressor as InnerElasticNet, FittedElasticNet};
use anofox_regression::{FittedRegressor as _, Regressor as _};
use ndarray::{Array1, Array2};
use rustml_core::{Fit, Predict, Result, RustMlError};

/// Elastic Net regression estimator.
///
/// Minimizes: `||y - Xβ||² + λ(α||β||₁ + (1-α)||β||₂²)`
///
/// - `alpha = 1.0` is pure Lasso (L1)
/// - `alpha = 0.0` is pure Ridge (L2)
/// - `0 < alpha < 1` is a mix of both
#[derive(Debug, Clone)]
pub struct ElasticNetRegressor {
    lambda: f64,
    alpha: f64,
    with_intercept: bool,
}

impl ElasticNetRegressor {
    pub fn new() -> Self {
        Self {
            lambda: 1.0,
            alpha: 0.5,
            with_intercept: true,
        }
    }

    pub fn with_lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda;
        self
    }

    /// Set the L1 ratio (alpha). 1.0 = pure Lasso, 0.0 = pure Ridge.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn with_intercept(mut self, include: bool) -> Self {
        self.with_intercept = include;
        self
    }
}

impl Default for ElasticNetRegressor {
    fn default() -> Self {
        Self::new()
    }
}

/// A fitted Elastic Net regression model.
#[derive(Debug, Clone)]
pub struct FittedElasticNetRegressor {
    inner: FittedElasticNet,
    n_features: usize,
}

impl FittedElasticNetRegressor {
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

impl Fit<f64> for ElasticNetRegressor {
    type Fitted = FittedElasticNetRegressor;

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
        if !(0.0..=1.0).contains(&self.alpha) {
            return Err(RustMlError::InvalidParameter(
                "alpha must be between 0.0 and 1.0".into(),
            ));
        }

        let x_mat = ndarray_to_mat(x);
        let y_col = ndarray_to_col(y);

        let inner_model = InnerElasticNet::builder()
            .with_intercept(self.with_intercept)
            .lambda(self.lambda)
            .alpha(self.alpha)
            .build();

        let fitted = inner_model
            .fit(&x_mat, &y_col)
            .map_err(|e| RustMlError::InvalidParameter(e.to_string()))?;

        Ok(FittedElasticNetRegressor {
            inner: fitted,
            n_features: x.ncols(),
        })
    }
}

impl Predict<f64> for FittedElasticNetRegressor {
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
    use ndarray::array;

    #[test]
    fn test_elastic_net_basic() {
        let x = Array2::from_shape_vec((10, 1), (0..10).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_vec((0..10).map(|i| 2.0 + 3.0 * i as f64).collect());

        let fitted = ElasticNetRegressor::new()
            .with_lambda(0.01)
            .with_alpha(0.5)
            .fit(&x, &y)
            .unwrap();

        assert!(fitted.r_squared() > 0.99);
    }

    #[test]
    fn test_elastic_net_invalid_alpha() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        assert!(ElasticNetRegressor::new()
            .with_alpha(1.5)
            .fit(&x, &y)
            .is_err());
        assert!(ElasticNetRegressor::new()
            .with_alpha(-0.1)
            .fit(&x, &y)
            .is_err());
    }
}
