//! Quantile regression wrapper.

use crate::convert::{col_to_ndarray, ndarray_to_col, ndarray_to_mat};
use anofox_regression::{FittedQuantile, QuantileRegressor as InnerQuantile};
use anofox_regression::{FittedRegressor as _, Regressor as _};
use ndarray::{Array1, Array2};
use rustml_core::{Fit, Predict, Result, RustMlError};

/// Quantile regression estimator.
///
/// Estimates conditional quantiles of the response variable.
/// Set `quantile = 0.5` for median regression.
#[derive(Debug, Clone)]
pub struct QuantileRegressor {
    quantile: f64,
    with_intercept: bool,
}

impl QuantileRegressor {
    pub fn new(quantile: f64) -> Self {
        Self {
            quantile,
            with_intercept: true,
        }
    }

    pub fn with_intercept(mut self, include: bool) -> Self {
        self.with_intercept = include;
        self
    }
}

/// A fitted quantile regression model.
#[derive(Debug, Clone)]
pub struct FittedQuantileRegressor {
    inner: FittedQuantile,
    n_features: usize,
}

impl FittedQuantileRegressor {
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

impl Fit<f64> for QuantileRegressor {
    type Fitted = FittedQuantileRegressor;

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
        if self.quantile <= 0.0 || self.quantile >= 1.0 {
            return Err(RustMlError::InvalidParameter(
                "quantile must be between 0 and 1 (exclusive)".into(),
            ));
        }

        let x_mat = ndarray_to_mat(x);
        let y_col = ndarray_to_col(y);

        let inner_model = InnerQuantile::builder()
            .tau(self.quantile)
            .with_intercept(self.with_intercept)
            .build();

        let fitted = inner_model
            .fit(&x_mat, &y_col)
            .map_err(|e| RustMlError::InvalidParameter(e.to_string()))?;

        Ok(FittedQuantileRegressor {
            inner: fitted,
            n_features: x.ncols(),
        })
    }
}

impl Predict<f64> for FittedQuantileRegressor {
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
    fn test_quantile_median() {
        // Simple linear data — median should match mean for symmetric errors
        let x = Array2::from_shape_vec((10, 1), (0..10).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_vec((0..10).map(|i| 2.0 + 3.0 * i as f64).collect());

        let fitted = QuantileRegressor::new(0.5).fit(&x, &y).unwrap();
        assert!(fitted.r_squared() > 0.99);
    }

    #[test]
    fn test_quantile_invalid_tau() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        assert!(QuantileRegressor::new(0.0).fit(&x, &y).is_err());
        assert!(QuantileRegressor::new(1.0).fit(&x, &y).is_err());
        assert!(QuantileRegressor::new(-0.1).fit(&x, &y).is_err());
    }
}

impl rustml_core::RegressorScore<f64> for FittedQuantileRegressor {}
