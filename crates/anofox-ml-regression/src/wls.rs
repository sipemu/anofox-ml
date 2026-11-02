//! Weighted Least Squares regression wrapper.

use crate::convert::{col_to_ndarray, ndarray_to_col, ndarray_to_mat};
use crate::summary::{
    interval_from, summary_from_result, IntervalKind, PredictionInterval, RegressionSummary,
};
use anofox_ml_core::{Fit, Predict, Result, RustMlError};
use anofox_regression::solvers::{FittedWls, WlsRegressor as InnerWls};
use anofox_regression::{FittedRegressor as _, Regressor as _};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Weighted Least Squares regression estimator.
///
/// Minimizes: `Σ w_i (y_i - x_i'β)²`
///
/// For sklearn-style `fit(X, y, sample_weight)` use
/// `OlsRegressor`'s `FitWeighted` implementation instead.
#[derive(Debug, Clone)]
pub struct WlsRegressor {
    weights: Array1<f64>,
    with_intercept: bool,
    inference: bool,
    confidence_level: f64,
}

impl WlsRegressor {
    pub fn new(weights: Array1<f64>) -> Self {
        Self {
            weights,
            with_intercept: true,
            inference: false,
            confidence_level: 0.95,
        }
    }

    pub fn with_intercept(mut self, include: bool) -> Self {
        self.with_intercept = include;
        self
    }

    /// Compute SE / t / p / CIs and enable `predict_interval`. Default false.
    pub fn with_inference(mut self, inference: bool) -> Self {
        self.inference = inference;
        self
    }

    /// Confidence level of the summary intervals. Default 0.95.
    pub fn with_confidence_level(mut self, level: f64) -> Self {
        self.confidence_level = level;
        self
    }
}

/// A fitted WLS regression model.
#[derive(Debug, Clone)]
pub struct FittedWlsRegressor {
    inner: FittedWls,
    n_features: usize,
    inference: bool,
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

    /// Coefficient table (requires `with_inference(true)`).
    pub fn summary(&self) -> Result<RegressionSummary> {
        summary_from_result(self.inner.result())
    }

    /// Confidence / prediction intervals at `level` (requires
    /// `with_inference(true)`). Prediction intervals assume unit weight for
    /// the new observations.
    pub fn predict_interval(
        &self,
        x: &Array2<f64>,
        kind: IntervalKind,
        level: f64,
    ) -> Result<PredictionInterval> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        if !self.inference {
            return Err(crate::summary::not_computed());
        }
        interval_from(&self.inner, &ndarray_to_mat(x.view()), kind, level)
    }
}

impl Fit<f64> for WlsRegressor {
    type Fitted = FittedWlsRegressor;

    fn fit_view(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<Self::Fitted> {
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

        crate::convert::resolve_sample_weight(Some(self.weights.view()), x.nrows())?;
        crate::summary::validate_level(self.confidence_level)?;
        let x_mat = ndarray_to_mat(x);
        let y_col = ndarray_to_col(y);
        let w_col = ndarray_to_col(self.weights.view());

        let inner_model = InnerWls::builder()
            .with_intercept(self.with_intercept)
            .weights(w_col)
            .compute_inference(self.inference)
            .confidence_level(self.confidence_level)
            .build();

        let fitted = inner_model
            .fit(&x_mat, &y_col)
            .map_err(|e| RustMlError::InvalidParameter(e.to_string()))?;

        Ok(FittedWlsRegressor {
            inner: fitted,
            n_features: x.ncols(),
            inference: self.inference,
        })
    }
}

impl Predict<f64> for FittedWlsRegressor {
    fn predict_view(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
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

    #[test]
    fn test_wls_summary_matches_statsmodels() {
        // statsmodels WLS(y, [1, x], weights=w)
        let x = Array2::from_shape_vec((8, 1), (0..8).map(|i| i as f64).collect()).unwrap();
        let y = array![1.2, 2.9, 5.3, 6.8, 9.4, 10.6, 13.3, 14.9];
        let w = array![1.0, 2.0, 0.5, 1.0, 3.0, 1.0, 0.2, 1.5];
        let f = WlsRegressor::new(w)
            .with_inference(true)
            .fit(&x, &y)
            .unwrap();
        let s = f.summary().unwrap();
        assert!((s.coefficients[0] - 1.9869420654911845).abs() < 1e-10);
        assert!((s.std_errors[0] - 0.051259156448969244).abs() < 1e-10);
        let xn = Array2::from_shape_vec((1, 1), vec![10.0]).unwrap();
        let ci = f
            .predict_interval(&xn, IntervalKind::Confidence, 0.95)
            .unwrap();
        assert!((ci.lower[0] - 20.106821161066065).abs() < 1e-9);
        assert!((ci.upper[0] - 21.850277075709762).abs() < 1e-9);
    }

    #[test]
    fn test_wls_rejects_negative_weights() {
        let x = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        assert!(WlsRegressor::new(array![1.0, -1.0, 1.0])
            .fit(&x, &y)
            .is_err());
    }
}

impl anofox_ml_core::RegressorScore<f64> for FittedWlsRegressor {}
