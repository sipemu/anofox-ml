//! Ordinary Least Squares regression wrapper.

use crate::convert::{col_to_ndarray, ndarray_to_col, ndarray_to_mat};
use crate::summary::{
    interval_from, summary_from_result, IntervalKind, PredictionInterval, RegressionSummary,
};
use anofox_ml_core::{Fit, FitWeighted, Predict, Result, RustMlError};
use anofox_regression::solvers::{
    FittedOls, FittedWls, OlsRegressor as InnerOls, WlsRegressor as InnerWls,
};
use anofox_regression::{FittedRegressor, RegressionResult, Regressor as _};
use ndarray::{Array1, Array2};

/// Ordinary Least Squares regression estimator.
///
/// Wraps `anofox_regression::OlsRegressor` and implements the anofox-ml
/// [`Fit`]/[`Predict`] type-state pattern. [`FitWeighted`] fits weighted
/// least squares (`Σ wᵢ (yᵢ - xᵢᵀβ)²`, sklearn's `sample_weight`).
///
/// Inference is opt-in: with `with_inference(true)` the fitted model offers
/// [`FittedOlsRegressor::summary`] (SE, t, p and CIs at `confidence_level`)
/// and [`FittedOlsRegressor::predict_interval`].
///
/// # Example
///
/// ```rust,ignore
/// use anofox_ml_regression::{IntervalKind, OlsRegressor};
/// use anofox_ml_core::{Fit, Predict};
/// use ndarray::{array, Array2};
///
/// let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
/// let y = array![2.1, 4.9, 8.2, 10.9, 14.1];
///
/// let fitted = OlsRegressor::new().with_inference(true).fit(&x, &y).unwrap();
/// let table = fitted.summary().unwrap();
/// let pi = fitted.predict_interval(&x, IntervalKind::Prediction, 0.95).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct OlsRegressor {
    with_intercept: bool,
    confidence_level: f64,
    inference: bool,
}

impl OlsRegressor {
    pub fn new() -> Self {
        Self {
            with_intercept: true,
            confidence_level: 0.95,
            inference: false,
        }
    }

    pub fn with_intercept(mut self, include: bool) -> Self {
        self.with_intercept = include;
        self
    }

    /// Confidence level of the coefficient intervals reported by
    /// [`FittedOlsRegressor::summary`]. Default 0.95.
    pub fn with_confidence_level(mut self, level: f64) -> Self {
        self.confidence_level = level;
        self
    }

    /// Compute standard errors, t statistics, p-values and CIs, and keep
    /// `(XᵀX)⁻¹` for prediction intervals. Default false.
    pub fn with_inference(mut self, inference: bool) -> Self {
        self.inference = inference;
        self
    }
}

impl Default for OlsRegressor {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
enum OlsInner {
    Ols(FittedOls),
    Wls(FittedWls),
}

/// A fitted OLS (or, via [`FitWeighted`], WLS) regression model.
#[derive(Debug, Clone)]
pub struct FittedOlsRegressor {
    inner: OlsInner,
    n_features: usize,
    inference: bool,
}

impl FittedOlsRegressor {
    fn result(&self) -> &RegressionResult {
        match &self.inner {
            OlsInner::Ols(m) => m.result(),
            OlsInner::Wls(m) => m.result(),
        }
    }

    /// Get the regression coefficients (slope terms).
    pub fn coefficients(&self) -> Array1<f64> {
        col_to_ndarray(&self.result().coefficients)
    }

    /// Get the intercept term, if the model was fit with one.
    pub fn intercept(&self) -> Option<f64> {
        self.result().intercept
    }

    /// Get the R-squared statistic.
    pub fn r_squared(&self) -> f64 {
        self.result().r_squared
    }

    /// Coefficient table (requires `with_inference(true)`).
    pub fn summary(&self) -> Result<RegressionSummary> {
        summary_from_result(self.result())
    }

    /// Confidence (mean response) or prediction (new observation) intervals
    /// at `level` (requires `with_inference(true)`).
    pub fn predict_interval(
        &self,
        x: &Array2<f64>,
        kind: IntervalKind,
        level: f64,
    ) -> Result<PredictionInterval> {
        self.check_features(x)?;
        if !self.inference {
            return Err(crate::summary::not_computed());
        }
        let x_mat = ndarray_to_mat(x);
        match &self.inner {
            OlsInner::Ols(m) => interval_from(m, &x_mat, kind, level),
            OlsInner::Wls(m) => interval_from(m, &x_mat, kind, level),
        }
    }

    fn check_features(&self, x: &Array2<f64>) -> Result<()> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        Ok(())
    }
}

impl Fit<f64> for OlsRegressor {
    type Fitted = FittedOlsRegressor;

    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        crate::convert::check_xy(x, y)?;
        crate::summary::validate_level(self.confidence_level)?;

        let x_mat = ndarray_to_mat(x);
        let y_col = ndarray_to_col(y);

        let inner_model = InnerOls::builder()
            .with_intercept(self.with_intercept)
            .compute_inference(self.inference)
            .confidence_level(self.confidence_level)
            .build();

        let fitted = inner_model
            .fit(&x_mat, &y_col)
            .map_err(|e| RustMlError::InvalidParameter(e.to_string()))?;

        Ok(FittedOlsRegressor {
            inner: OlsInner::Ols(fitted),
            n_features: x.ncols(),
            inference: self.inference,
        })
    }
}

impl FitWeighted<f64> for OlsRegressor {
    type Fitted = FittedOlsRegressor;

    /// Weighted least squares; `None` is identical to [`Fit::fit`].
    fn fit_weighted(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        sample_weight: Option<&Array1<f64>>,
    ) -> Result<Self::Fitted> {
        let Some(w) = sample_weight else {
            return self.fit(x, y);
        };
        crate::convert::check_xy(x, y)?;
        crate::summary::validate_level(self.confidence_level)?;
        let w = crate::convert::resolve_sample_weight(Some(w), y.len())?;
        let fitted = InnerWls::builder()
            .with_intercept(self.with_intercept)
            .weights(ndarray_to_col(&w))
            .compute_inference(self.inference)
            .confidence_level(self.confidence_level)
            .build()
            .fit(&ndarray_to_mat(x), &ndarray_to_col(y))
            .map_err(|e| RustMlError::InvalidParameter(e.to_string()))?;
        Ok(FittedOlsRegressor {
            inner: OlsInner::Wls(fitted),
            n_features: x.ncols(),
            inference: self.inference,
        })
    }
}

impl Predict<f64> for FittedOlsRegressor {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        self.check_features(x)?;
        let x_mat = ndarray_to_mat(x);
        let preds = match &self.inner {
            OlsInner::Ols(m) => m.predict(&x_mat),
            OlsInner::Wls(m) => m.predict(&x_mat),
        };
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

    // statsmodels 0.14 OLS / WLS on x = 0..8,
    // y = [1.2, 2.9, 5.3, 6.8, 9.4, 10.6, 13.3, 14.9].
    fn sm_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec((8, 1), (0..8).map(|i| i as f64).collect()).unwrap();
        let y = array![1.2, 2.9, 5.3, 6.8, 9.4, 10.6, 13.3, 14.9];
        (x, y)
    }

    #[test]
    fn test_ols_summary_matches_statsmodels() {
        let (x, y) = sm_data();
        let f = OlsRegressor::new()
            .with_inference(true)
            .with_confidence_level(0.9)
            .fit(&x, &y)
            .unwrap();
        let s = f.summary().unwrap();
        assert_eq!(s.confidence_level, 0.9);
        assert_abs_diff_eq!(s.coefficients[0], 1.9809523809523801, epsilon = 1e-10);
        assert_abs_diff_eq!(s.std_errors[0], 0.04734045489525958, epsilon = 1e-10);
        assert_abs_diff_eq!(s.p_values[0], 1.2461036970099873e-08, epsilon = 1e-13);
        assert_abs_diff_eq!(s.conf_lower[0], 1.8889613425292875, epsilon = 1e-9);
        assert_abs_diff_eq!(s.conf_upper[0], 2.0729434193754726, epsilon = 1e-9);
        let ic = s.intercept.unwrap();
        assert_abs_diff_eq!(ic.std_error, 0.19803933124401543, epsilon = 1e-10);
    }

    #[test]
    fn test_ols_predict_interval_matches_statsmodels() {
        let (x, y) = sm_data();
        let f = OlsRegressor::new()
            .with_inference(true)
            .fit(&x, &y)
            .unwrap();
        let xn = Array2::from_shape_vec((1, 1), vec![10.0]).unwrap();
        let ci = f
            .predict_interval(&xn, IntervalKind::Confidence, 0.95)
            .unwrap();
        let pi = f
            .predict_interval(&xn, IntervalKind::Prediction, 0.95)
            .unwrap();
        assert_abs_diff_eq!(ci.lower[0], 20.127832707040625, epsilon = 1e-9);
        assert_abs_diff_eq!(ci.upper[0], 21.724548245340316, epsilon = 1e-9);
        assert_abs_diff_eq!(pi.lower[0], 19.83031232394724, epsilon = 1e-9);
        assert_abs_diff_eq!(pi.upper[0], 22.0220686284337, epsilon = 1e-9);
    }

    #[test]
    fn test_ols_inference_is_opt_in() {
        let (x, y) = sm_data();
        let f = OlsRegressor::new().fit(&x, &y).unwrap();
        assert!(f.summary().is_err());
        assert!(f
            .predict_interval(&x, IntervalKind::Confidence, 0.95)
            .is_err());
        assert!(OlsRegressor::new()
            .with_confidence_level(1.5)
            .fit(&x, &y)
            .is_err());
    }

    #[test]
    fn test_ols_fit_weighted_matches_statsmodels_wls() {
        let (x, y) = sm_data();
        let w = array![1.0, 2.0, 0.5, 1.0, 3.0, 1.0, 0.2, 1.5];
        let f = OlsRegressor::new()
            .with_inference(true)
            .fit_weighted(&x, &y, Some(&w))
            .unwrap();
        assert_abs_diff_eq!(f.coefficients()[0], 1.9869420654911845, epsilon = 1e-10);
        assert_abs_diff_eq!(f.intercept().unwrap(), 1.1091284634760674, epsilon = 1e-10);
        assert_abs_diff_eq!(
            f.summary().unwrap().std_errors[0],
            0.051259156448969244,
            epsilon = 1e-10
        );
        let xn = Array2::from_shape_vec((1, 1), vec![10.0]).unwrap();
        let ci = f
            .predict_interval(&xn, IntervalKind::Confidence, 0.95)
            .unwrap();
        assert_abs_diff_eq!(ci.lower[0], 20.106821161066065, epsilon = 1e-9);
        assert_abs_diff_eq!(ci.upper[0], 21.850277075709762, epsilon = 1e-9);

        // None == plain fit.
        let a = OlsRegressor::new().fit_weighted(&x, &y, None).unwrap();
        let b = OlsRegressor::new().fit(&x, &y).unwrap();
        assert_abs_diff_eq!(a.coefficients()[0], b.coefficients()[0], epsilon = 1e-14);
    }

    #[test]
    fn test_ols_integer_weights_equal_row_duplication() {
        let (x, y) = sm_data();
        let wi = array![1.0, 2.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0];
        let fw = OlsRegressor::new().fit_weighted(&x, &y, Some(&wi)).unwrap();
        let mut xs = vec![];
        let mut ys = vec![];
        for i in 0..8 {
            for _ in 0..(wi[i] as usize) {
                xs.push(x[[i, 0]]);
                ys.push(y[i]);
            }
        }
        let xd = Array2::from_shape_vec((xs.len(), 1), xs).unwrap();
        let fd = OlsRegressor::new().fit(&xd, &Array1::from_vec(ys)).unwrap();
        assert_abs_diff_eq!(fw.coefficients()[0], fd.coefficients()[0], epsilon = 1e-10);
        assert_abs_diff_eq!(
            fw.intercept().unwrap(),
            fd.intercept().unwrap(),
            epsilon = 1e-10
        );
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
