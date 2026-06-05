//! Weighted Least Squares regression wrapper.

use crate::convert::{col_to_ndarray, ndarray_to_col, ndarray_to_mat};
use anofox_regression::solvers::{FittedWls, WlsRegressor as InnerWls};
use anofox_regression::{FittedRegressor as _, Regressor as _};
use ndarray::{Array1, Array2};
use rustml_core::{Fit, Predict, Result, RustMlError};

/// Weighted Least Squares regression estimator.
///
/// Minimizes: `Σ w_i (y_i - x_i'β)²`
#[derive(Debug, Clone)]
pub struct WlsRegressor {
    weights: Array1<f64>,
    with_intercept: bool,
}

impl WlsRegressor {
    pub fn new(weights: Array1<f64>) -> Self {
        Self {
            weights,
            with_intercept: true,
        }
    }

    pub fn with_intercept(mut self, include: bool) -> Self {
        self.with_intercept = include;
        self
    }
}

/// A fitted WLS regression model.
#[derive(Debug, Clone)]
pub struct FittedWlsRegressor {
    inner: FittedWls,
    n_features: usize,
}

impl FittedWlsRegressor {
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

impl Fit<f64> for WlsRegressor {
    type Fitted = FittedWlsRegressor;

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
        if self.weights.len() != x.nrows() {
            return Err(RustMlError::ShapeMismatch(format!(
                "weights has {} elements but X has {} rows",
                self.weights.len(),
                x.nrows()
            )));
        }

        let x_mat = ndarray_to_mat(x);
        let y_col = ndarray_to_col(y);
        let w_col = ndarray_to_col(&self.weights);

        let inner_model = InnerWls::builder()
            .with_intercept(self.with_intercept)
            .weights(w_col)
            .build();

        let fitted = inner_model
            .fit(&x_mat, &y_col)
            .map_err(|e| RustMlError::InvalidParameter(e.to_string()))?;

        Ok(FittedWlsRegressor {
            inner: fitted,
            n_features: x.ncols(),
        })
    }
}

impl Predict<f64> for FittedWlsRegressor {
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
    fn test_wls_basic() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 5.0, 8.0, 11.0, 14.0];
        let w = array![1.0, 1.0, 1.0, 1.0, 1.0];

        // Uniform weights should give same result as OLS
        let fitted = WlsRegressor::new(w).fit(&x, &y).unwrap();
        assert!(fitted.r_squared() > 0.99);
    }

    #[test]
    fn test_wls_weight_mismatch() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let w = array![1.0, 1.0, 1.0];

        assert!(WlsRegressor::new(w).fit(&x, &y).is_err());
    }
}

impl rustml_core::RegressorScore<f64> for FittedWlsRegressor {}
