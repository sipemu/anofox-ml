//! Isotonic (monotonic) regression wrapper.

use crate::convert::{col_to_ndarray, ndarray_to_col, ndarray_to_mat};
use anofox_ml_core::{Fit, Predict, Result, RustMlError};
use anofox_regression::{FittedIsotonic, IsotonicRegressor as InnerIsotonic};
use anofox_regression::{FittedRegressor as _, Regressor as _};
use ndarray::{Array1, Array2};

/// Isotonic regression estimator.
///
/// Fits a monotonic (non-decreasing or non-increasing) step function to 1D data.
/// Requires exactly one feature column.
#[derive(Debug, Clone)]
pub struct IsotonicRegressor {
    increasing: bool,
}

impl IsotonicRegressor {
    pub fn new() -> Self {
        Self { increasing: true }
    }

    pub fn with_increasing(mut self, increasing: bool) -> Self {
        self.increasing = increasing;
        self
    }
}

impl Default for IsotonicRegressor {
    fn default() -> Self {
        Self::new()
    }
}

/// A fitted isotonic regression model.
#[derive(Debug, Clone)]
pub struct FittedIsotonicRegressor {
    inner: FittedIsotonic,
}

impl FittedIsotonicRegressor {
    pub fn r_squared(&self) -> f64 {
        self.inner.r_squared()
    }
}

impl Fit<f64> for IsotonicRegressor {
    type Fitted = FittedIsotonicRegressor;

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
        if x.ncols() != 1 {
            return Err(RustMlError::InvalidParameter(
                "isotonic regression requires exactly one feature column".into(),
            ));
        }

        let x_mat = ndarray_to_mat(x);
        let y_col = ndarray_to_col(y);

        let inner_model = InnerIsotonic::builder().increasing(self.increasing).build();

        let fitted = inner_model
            .fit(&x_mat, &y_col)
            .map_err(|e| RustMlError::InvalidParameter(e.to_string()))?;

        Ok(FittedIsotonicRegressor { inner: fitted })
    }
}

impl Predict<f64> for FittedIsotonicRegressor {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if x.ncols() != 1 {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected 1 feature, got {}",
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
    fn test_isotonic_increasing() {
        let x = Array2::from_shape_vec((6, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        // Noisy increasing data with a violation: 3.0 > 2.5
        let y = array![1.0, 2.0, 3.0, 2.5, 4.0, 5.0];

        let fitted = IsotonicRegressor::new().fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        // After isotonic fit, predictions should be non-decreasing
        for i in 1..preds.len() {
            assert!(
                preds[i] >= preds[i - 1] - 1e-10,
                "predictions should be non-decreasing: preds[{}]={} < preds[{}]={}",
                i,
                preds[i],
                i - 1,
                preds[i - 1]
            );
        }
    }

    #[test]
    fn test_isotonic_multi_feature_error() {
        let x = Array2::from_shape_vec((5, 2), vec![0.0; 10]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        assert!(IsotonicRegressor::new().fit(&x, &y).is_err());
    }
}

impl anofox_ml_core::RegressorScore<f64> for FittedIsotonicRegressor {}
