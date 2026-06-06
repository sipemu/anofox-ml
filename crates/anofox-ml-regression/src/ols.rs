//! Ordinary Least Squares regression wrapper.

use crate::convert::{col_to_ndarray, ndarray_to_col, ndarray_to_mat};
use anofox_ml_core::{Fit, Predict, Result, RustMlError};
use anofox_regression::solvers::{FittedOls, OlsRegressor as InnerOls};
use anofox_regression::{FittedRegressor as _, Regressor as _};
use ndarray::{Array1, Array2};

/// Ordinary Least Squares regression estimator.
///
/// Wraps `anofox_regression::OlsRegressor` and implements the anofox-ml
/// [`Fit`]/[`Predict`] type-state pattern.
///
/// # Example
///
/// ```rust,ignore
/// use anofox_ml_regression::OlsRegressor;
/// use anofox_ml_core::{Fit, Predict};
/// use ndarray::{array, Array2};
///
/// let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
/// let y = array![2.0, 5.0, 8.0, 11.0, 14.0];
///
/// let fitted = OlsRegressor::new().fit(&x, &y).unwrap();
/// let preds = fitted.predict(&Array2::from_shape_vec((1, 1), vec![5.0]).unwrap()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct OlsRegressor {
    with_intercept: bool,
    confidence_level: f64,
}

impl OlsRegressor {
    pub fn new() -> Self {
        Self {
            with_intercept: true,
            confidence_level: 0.95,
        }
    }

    pub fn with_intercept(mut self, include: bool) -> Self {
        self.with_intercept = include;
        self
    }

    pub fn with_confidence_level(mut self, level: f64) -> Self {
        self.confidence_level = level;
        self
    }
}

impl Default for OlsRegressor {
    fn default() -> Self {
        Self::new()
    }
}

/// A fitted OLS regression model.
#[derive(Debug, Clone)]
pub struct FittedOlsRegressor {
    inner: FittedOls,
    n_features: usize,
}

impl FittedOlsRegressor {
    /// Get the regression coefficients (slope terms).
    pub fn coefficients(&self) -> Array1<f64> {
        col_to_ndarray(self.inner.coefficients())
    }

    /// Get the intercept term, if the model was fit with one.
    pub fn intercept(&self) -> Option<f64> {
        self.inner.intercept()
    }

    /// Get the R-squared statistic.
    pub fn r_squared(&self) -> f64 {
        self.inner.r_squared()
    }
}

impl Fit<f64> for OlsRegressor {
    type Fitted = FittedOlsRegressor;

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

        let x_mat = ndarray_to_mat(x);
        let y_col = ndarray_to_col(y);

        let inner_model = InnerOls::builder()
            .with_intercept(self.with_intercept)
            .confidence_level(self.confidence_level)
            .build();

        let fitted = inner_model
            .fit(&x_mat, &y_col)
            .map_err(|e| RustMlError::InvalidParameter(e.to_string()))?;

        Ok(FittedOlsRegressor {
            inner: fitted,
            n_features: x.ncols(),
        })
    }
}

impl Predict<f64> for FittedOlsRegressor {
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
    fn test_ols_simple_linear() {
        // y = 2 + 3x
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 5.0, 8.0, 11.0, 14.0];

        let fitted = OlsRegressor::new().fit(&x, &y).unwrap();

        assert_abs_diff_eq!(fitted.coefficients()[0], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(fitted.intercept().unwrap(), 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(fitted.r_squared(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ols_predict() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 5.0, 8.0, 11.0, 14.0];

        let fitted = OlsRegressor::new().fit(&x, &y).unwrap();

        let x_new = Array2::from_shape_vec((2, 1), vec![10.0, 11.0]).unwrap();
        let preds = fitted.predict(&x_new).unwrap();

        assert_abs_diff_eq!(preds[0], 32.0, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[1], 35.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ols_shape_mismatch() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];

        let result = OlsRegressor::new().fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_ols_empty_input() {
        let x = Array2::<f64>::zeros((0, 1));
        let y = array![];

        let result = OlsRegressor::new().fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_ols_no_intercept() {
        // y = 3x (through origin)
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 6.0, 9.0, 12.0, 15.0];

        let fitted = OlsRegressor::new()
            .with_intercept(false)
            .fit(&x, &y)
            .unwrap();

        assert_abs_diff_eq!(fitted.coefficients()[0], 3.0, epsilon = 1e-10);
        assert!(fitted.intercept().is_none());
    }

    #[test]
    fn test_ols_multivariate() {
        // y = 1 + 2*x1 + 3*x2
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 2.0],
        )
        .unwrap();
        let y = array![3.0, 4.0, 6.0, 5.0, 7.0];

        let fitted = OlsRegressor::new().fit(&x, &y).unwrap();

        assert_abs_diff_eq!(fitted.coefficients()[0], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(fitted.coefficients()[1], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(fitted.intercept().unwrap(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ols_predict_feature_mismatch() {
        let x = Array2::from_shape_vec((5, 2), vec![0.0; 10]).unwrap();
        let y = array![0.0, 0.0, 0.0, 0.0, 0.0];

        let fitted = OlsRegressor::new().fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((2, 3), vec![0.0; 6]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }
}

impl anofox_ml_core::RegressorScore<f64> for FittedOlsRegressor {}
