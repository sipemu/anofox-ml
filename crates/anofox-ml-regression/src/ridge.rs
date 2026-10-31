//! Ridge (L2-regularized) regression wrapper.

use crate::convert::{ndarray_to_col, ndarray_to_mat};
use crate::summary::{
    interval_from, summary_from_result, IntervalKind, PredictionInterval, RegressionSummary,
};
use anofox_ml_core::{Fit, FitWeighted, Predict, Result, RustMlError};
use anofox_regression::solvers::{FittedRidge, RidgeRegressor as InnerRidge};
use anofox_regression::{FittedRegressor as _, Regressor as _};
use ndarray::{Array1, Array2};

/// Ridge regression estimator with L2 regularization.
///
/// Minimizes: `||y - Xβ||² + λ||β||²`
#[derive(Debug, Clone)]
pub struct RidgeRegressor {
    lambda: f64,
    with_intercept: bool,
    inference: bool,
    confidence_level: f64,
}

impl RidgeRegressor {
    pub fn new() -> Self {
        Self {
            lambda: 1.0,
            with_intercept: true,
            inference: false,
            confidence_level: 0.95,
        }
    }

    /// Compute (ridge) standard errors / t / p / CIs and enable
    /// `predict_interval` on the unweighted fit. Default false.
    ///
    /// Standard errors are the upstream sandwich `σ² A XᵀX A`,
    /// `A = (XᵀX + λP)⁻¹`, conditional on `lambda`. For `lambda > 0` the
    /// t statistics, p-values and coefficient CIs are reported as `NaN`
    /// (a test centred on a shrunken, biased estimate is not valid).
    pub fn with_inference(mut self, inference: bool) -> Self {
        self.inference = inference;
        self
    }

    /// Confidence level of the summary intervals. Default 0.95.
    pub fn with_confidence_level(mut self, level: f64) -> Self {
        self.confidence_level = level;
        self
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
    inference: bool,
    coef: Array1<f64>,
    intercept: Option<f64>,
}

impl FittedRidgeRegressor {
    pub fn coefficients(&self) -> Array1<f64> {
        self.coef.clone()
    }

    pub fn intercept(&self) -> Option<f64> {
        self.intercept
    }

    pub fn r_squared(&self) -> f64 {
        self.inner.r_squared()
    }

    /// Coefficient table (requires `with_inference(true)`).
    pub fn summary(&self) -> Result<RegressionSummary> {
        if !self.inference {
            return Err(crate::summary::not_computed());
        }
        summary_from_result(self.inner.result())
    }

    /// Confidence / prediction intervals at `level` (requires
    /// `with_inference(true)`).
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
        interval_from(&self.inner, &ndarray_to_mat(x), kind, level)
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

        crate::summary::validate_level(self.confidence_level)?;
        let inner_model = InnerRidge::builder()
            .with_intercept(self.with_intercept)
            .lambda(self.lambda)
            .compute_inference(self.inference)
            .confidence_level(self.confidence_level)
            .build();

        let fitted = inner_model
            .fit(&x_mat, &y_col)
            .map_err(|e| RustMlError::InvalidParameter(e.to_string()))?;

        let (coef, intercept) = crate::convert::predict_params(fitted.result());
        Ok(FittedRidgeRegressor {
            inner: fitted,
            n_features: x.ncols(),
            inference: self.inference,
            coef,
            intercept: self.with_intercept.then_some(intercept),
        })
    }
}

impl FitWeighted<f64> for RidgeRegressor {
    type Fitted = FittedRidgeRegressor;

    /// Minimises `Σ wᵢ (yᵢ − xᵢβ − β₀)² + λ‖β‖²` (sklearn's `sample_weight`).
    ///
    /// As in sklearn, X and y are centred by their weighted means and each
    /// row is scaled by `√wᵢ`; the upstream ridge solver then fits the
    /// rescaled problem without intercept, and `β₀ = ȳ_w − x̄_wᵀβ`.
    /// `r_squared` is the weighted R². Inference is not available for a
    /// weighted fit (`summary` / `predict_interval` return an error).
    /// `None` is identical to [`Fit::fit`].
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
        if self.lambda < 0.0 {
            return Err(RustMlError::InvalidParameter(
                "lambda must be non-negative".into(),
            ));
        }
        let w = crate::convert::resolve_sample_weight(Some(w), y.len())?;
        let sqrt_w = w.mapv(f64::sqrt);

        let (x_mean, y_mean) = if self.with_intercept {
            let wsum = w.sum();
            (w.dot(x) / wsum, w.dot(y) / wsum)
        } else {
            (Array1::zeros(x.ncols()), 0.0)
        };
        let xs = (x - &x_mean) * &sqrt_w.view().insert_axis(ndarray::Axis(1));
        let ys = (y - y_mean) * &sqrt_w;

        let fitted = InnerRidge::builder()
            .with_intercept(false)
            .lambda(self.lambda)
            .build()
            .fit(&ndarray_to_mat(&xs), &ndarray_to_col(&ys))
            .map_err(|e| RustMlError::InvalidParameter(e.to_string()))?;
        let (coef, _) = crate::convert::predict_params(fitted.result());
        let intercept = self.with_intercept.then(|| y_mean - x_mean.dot(&coef));
        Ok(FittedRidgeRegressor {
            inner: fitted,
            n_features: x.ncols(),
            inference: false,
            coef,
            intercept,
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

        crate::convert::linear_predict(x, &self.coef, self.intercept.unwrap_or(0.0))
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

        let fitted = RidgeRegressor::new().with_lambda(0.01).fit(&x, &y).unwrap();

        // Ridge with small lambda should be close to OLS
        assert!(fitted.r_squared() > 0.99);
        assert_abs_diff_eq!(fitted.coefficients()[0], 3.0, epsilon = 0.1);
    }

    #[test]
    fn test_ridge_shrinks_coefficients() {
        let x = Array2::from_shape_vec((10, 1), (0..10).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_vec((0..10).map(|i| 2.0 + 3.0 * i as f64).collect());

        let fitted_small = RidgeRegressor::new().with_lambda(0.01).fit(&x, &y).unwrap();
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
    fn test_ridge_inference_opt_in_and_tiny_lambda_matches_ols() {
        let x = Array2::from_shape_vec((8, 1), (0..8).map(|i| i as f64).collect()).unwrap();
        let y = array![1.2, 2.9, 5.3, 6.8, 9.4, 10.6, 13.3, 14.9];
        let plain = RidgeRegressor::new().fit(&x, &y).unwrap();
        assert!(plain.summary().is_err());
        assert!(plain
            .predict_interval(&x, IntervalKind::Confidence, 0.95)
            .is_err());

        // lambda -> 0: ridge inference reduces to OLS (statsmodels values).
        let f = RidgeRegressor::new()
            .with_lambda(1e-10)
            .with_inference(true)
            .fit(&x, &y)
            .unwrap();
        let s = f.summary().unwrap();
        assert!((s.std_errors[0] - 0.04734045489525958).abs() < 1e-6);
        let xn = Array2::from_shape_vec((1, 1), vec![10.0]).unwrap();
        let ci = f
            .predict_interval(&xn, IntervalKind::Confidence, 0.95)
            .unwrap();
        assert!((ci.lower[0] - 20.127832707040625).abs() < 1e-5);
        assert!((ci.upper[0] - 21.724548245340316).abs() < 1e-5);
        let pi = f
            .predict_interval(&xn, IntervalKind::Prediction, 0.95)
            .unwrap();
        assert!(pi.lower[0] < ci.lower[0] && pi.upper[0] > ci.upper[0]);
    }

    #[test]
    fn test_ridge_negative_lambda() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = RidgeRegressor::new().with_lambda(-1.0).fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_ridge_uniform_weights_match_unweighted() {
        // Sample-weighted fit with all-ones weights should match unweighted
        // (within numerical noise — anofox + our XᵀWX solver share the same
        // closed-form but use different solvers).
        let x = Array2::from_shape_vec((6, 1), (0..6).map(|i| i as f64).collect()).unwrap();
        let y = array![1.0_f64, 3.0, 5.0, 7.0, 9.0, 11.0]; // 2x + 1
        let rr = RidgeRegressor::new().with_lambda(0.01);
        let unw = rr.fit(&x, &y).unwrap();
        let w = Array1::<f64>::ones(6);
        let weighted = rr.fit_weighted(&x, &y, Some(&w)).unwrap();
        // Both should recover slope ~ 2 and intercept ~ 1.
        assert!((unw.coefficients()[0] - 2.0).abs() < 0.05);
        assert_abs_diff_eq!(
            weighted.coefficients()[0],
            unw.coefficients()[0],
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            weighted.intercept().unwrap(),
            unw.intercept().unwrap(),
            epsilon = 1e-10
        );
        assert!(weighted.summary().is_err());
    }

    #[test]
    fn test_ridge_weights_match_sklearn() {
        // sklearn 1.8: Ridge(alpha=0.7).fit(X, y, sample_weight=w)
        let x = array![
            [0.5, 1.2],
            [1.0, -0.3],
            [1.5, 2.2],
            [2.0, 0.1],
            [2.5, 1.7],
            [3.0, -1.0],
            [3.5, 0.4]
        ];
        let y = array![1.3, 2.1, 4.0, 3.2, 5.6, 3.9, 6.1];
        let w = array![1.0, 2.0, 0.5, 3.0, 1.0, 0.0, 2.5];
        let f = RidgeRegressor::new()
            .with_lambda(0.7)
            .fit_weighted(&x, &y, Some(&w))
            .unwrap();
        let (c, b) = (f.coefficients(), f.intercept().unwrap());
        assert_abs_diff_eq!(c[0], 1.505_442_95, epsilon = 1e-8);
        assert_abs_diff_eq!(c[1], 0.494_618_58, epsilon = 1e-8);
        assert_abs_diff_eq!(b, 0.476_371_221_034_365_36, epsilon = 1e-8);
        // Normal equations of the weighted, centred problem must hold.
        let wsum = w.sum();
        let xm = w.dot(&x) / wsum;
        let ym = w.dot(&y) / wsum;
        assert_abs_diff_eq!(b, ym - xm.dot(&c), epsilon = 1e-12);
        let xc = &x - &xm;
        let r = &y - ym - xc.dot(&c);
        let grad = xc.t().dot(&(&r * &w)) - &c * 0.7;
        assert!(grad.iter().all(|g| g.abs() < 1e-9), "grad = {grad}");
    }

    #[test]
    fn test_ridge_high_weight_anchor_dominates() {
        // 5 noisy points + 1 high-weight anchor at (10, 100). The fit should
        // pull strongly toward the anchor.
        let x = Array2::from_shape_vec((6, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0, 10.0]).unwrap();
        let y = array![0.0, 0.5, 0.5, 0.0, 0.0, 100.0];
        let w = array![1.0, 1.0, 1.0, 1.0, 1.0, 1e6];
        let fitted = RidgeRegressor::new()
            .with_lambda(1e-6)
            .fit_weighted(&x, &y, Some(&w))
            .unwrap();
        let p = fitted
            .predict(&Array2::from_shape_vec((1, 1), vec![10.0]).unwrap())
            .unwrap();
        assert!((p[0] - 100.0).abs() < 1.0, "anchor pred = {}", p[0]);
    }
}

impl anofox_ml_core::RegressorScore<f64> for FittedRidgeRegressor {}
