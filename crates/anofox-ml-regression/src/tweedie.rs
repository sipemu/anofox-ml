//! Tweedie / Gamma GLM regressors.
//!
//! The penalised objective follows
//! `sklearn.linear_model.{TweedieRegressor, GammaRegressor}`:
//!
//! ```text
//! min  1/(2 Σw) Σ wᵢ d_p(yᵢ, μᵢ) + α/2 ‖β‖²      (intercept unpenalised)
//! ```
//!
//! with `d_p` the Tweedie unit deviance of variance power `p` and the link
//! chosen like sklearn's `link='auto'` (identity for `p <= 0`, log
//! otherwise). The fit is delegated to `anofox_regression`'s
//! `TweedieRegressor` (penalised IRLS, upstream `lambda = alpha · Σw`, prior
//! weights `w`), with sample weights (`FitWeighted`), offsets and opt-in Wald
//! inference.
//!
//! `alpha` defaults to sklearn's 1.0.

use crate::convert::{
    check_row_vector, check_xy, coef_from_col, ndarray_to_col, ndarray_to_mat,
    resolve_sample_weight, upstream_err,
};
use crate::glm::{check_response, upstream_lambda, GlmFit};
use crate::summary::{glm_summary_from_result, RegressionSummary, StatisticKind};
use anofox_ml_core::{Fit, FitWeighted, Predict, Result, RustMlError};
use anofox_regression::solvers::{
    FittedRegressor as _, Regressor as _, TweedieRegressor as InnerTweedie,
};
use anofox_regression::{GlmFamily, TweedieFamily};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, AsArray, Ix1, Ix2};
use serde::{Deserialize, Serialize};

/// Tweedie GLM regressor.
///
/// `power=0` → Gaussian, `power=1` → Poisson, `1<power<2` → compound
/// Poisson-Gamma, `power=2` → Gamma, `power=3` → Inverse Gaussian
/// (`0 < power < 1` is not a valid distribution and is rejected; sklearn's
/// negative powers are not supported).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TweedieRegressor {
    power: f64,
    alpha: f64,
    link_power: f64,
    with_intercept: bool,
    max_iter: usize,
    tol: f64,
    inference: bool,
    confidence_level: f64,
}

impl TweedieRegressor {
    pub fn new(power: f64) -> Self {
        Self {
            power,
            alpha: 1.0,
            // sklearn link='auto': identity for power <= 0, log otherwise.
            link_power: if power <= 0.0 { 1.0 } else { 0.0 },
            with_intercept: true,
            max_iter: 100,
            tol: 1e-8,
            inference: false,
            confidence_level: 0.95,
        }
    }

    /// L2 penalty strength on the mean deviance (sklearn's `alpha`). Default 0.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Link power `q`: `0` → log link, `1` → identity, `q` → `μ^q`.
    pub fn with_link_power(mut self, link_power: f64) -> Self {
        self.link_power = link_power;
        self
    }

    pub fn with_intercept(mut self, include: bool) -> Self {
        self.with_intercept = include;
        self
    }

    /// Maximum IRLS iterations. Default 100.
    pub fn with_max_iter(mut self, m: usize) -> Self {
        self.max_iter = m;
        self
    }

    /// IRLS convergence tolerance. Default 1e-8.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Compute Wald inference at fit time (estimated dispersion, t
    /// statistics unless `power == 1`). Default false.
    pub fn with_inference(mut self, inference: bool) -> Self {
        self.inference = inference;
        self
    }

    pub fn with_confidence_level(mut self, level: f64) -> Self {
        self.confidence_level = level;
        self
    }

    /// Fit with an optional offset on the link scale (e.g. `ln(exposure)`
    /// for the log link).
    pub fn fit_with_offset<'a, 'b>(
        &self,
        x: impl AsArray<'a, f64, Ix2>,
        y: impl AsArray<'b, f64, Ix1>,
        offset: Option<&Array1<f64>>,
    ) -> Result<FittedTweedieRegressor> {
        self.fit_impl(x.into(), y.into(), None, offset)
    }

    /// Fit with sample weights (sklearn's `sample_weight`, R's prior
    /// weights) and an optional offset on the link scale.
    pub fn fit_weighted_with_offset<'a, 'b>(
        &self,
        x: impl AsArray<'a, f64, Ix2>,
        y: impl AsArray<'b, f64, Ix1>,
        sample_weight: Option<&Array1<f64>>,
        offset: Option<&Array1<f64>>,
    ) -> Result<FittedTweedieRegressor> {
        self.fit_impl(x.into(), y.into(), sample_weight.map(|w| w.view()), offset)
    }

    fn fit_impl(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        sample_weight: Option<ArrayView1<f64>>,
        offset: Option<&Array1<f64>>,
    ) -> Result<FittedTweedieRegressor> {
        check_xy(x, y)?;
        let p = self.power;
        if !p.is_finite() || p < 0.0 || (p > 0.0 && p < 1.0) {
            return Err(RustMlError::InvalidParameter(format!(
                "Tweedie power must be 0 or >= 1 (negative powers are not supported), got {p}"
            )));
        }
        if !self.link_power.is_finite() {
            return Err(RustMlError::InvalidParameter(
                "link_power must be finite".into(),
            ));
        }
        if p >= 2.0 {
            check_response(y, "Tweedie (power >= 2)", |v| v > 0.0, "strictly positive")?;
        } else if p >= 1.0 {
            check_response(y, "Tweedie (1 <= power < 2)", |v| v >= 0.0, "non-negative")?;
        } else {
            check_response(y, "Tweedie", |_| true, "")?;
        }
        check_row_vector(offset, y.len(), "offset")?;
        if self.inference {
            crate::summary::validate_level(self.confidence_level)?;
        }
        let w = resolve_sample_weight(sample_weight, y.len())?;
        let mut b = InnerTweedie::builder()
            .var_power(p)
            .link_power(self.link_power)
            .with_intercept(self.with_intercept)
            .lambda(upstream_lambda(self.alpha, w.sum())?)
            .max_iterations(self.max_iter)
            .tolerance(self.tol)
            .compute_inference(self.inference)
            .confidence_level(self.confidence_level);
        if sample_weight.is_some() {
            b = b.weights(ndarray_to_col(w.view()));
        }
        if let Some(o) = offset {
            b = b.offset(ndarray_to_col(o.view()));
        }
        let inner = b
            .build()
            .fit(&ndarray_to_mat(x), &ndarray_to_col(y))
            .map_err(upstream_err)?;
        let res = inner.result();
        let summary = if self.inference {
            let kind = if p == 1.0 {
                StatisticKind::Z
            } else {
                StatisticKind::T
            };
            Some(glm_summary_from_result(res, kind, inner.dispersion)?)
        } else {
            None
        };
        Ok(FittedTweedieRegressor {
            fit: GlmFit {
                coef: coef_from_col(&res.coefficients),
                intercept: res.intercept,
                deviance: inner.deviance,
                null_deviance: inner.null_deviance,
                n_iter: inner.iterations,
                converged: inner.converged,
                summary,
                n_features: x.ncols(),
            },
            power: p,
            link_power: self.link_power,
        })
    }
}

/// Convenience: Gamma regression (power=2, log link).
pub fn gamma_regressor() -> TweedieRegressor {
    TweedieRegressor::new(2.0)
}

/// Fitted Tweedie GLM regressor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedTweedieRegressor {
    fit: GlmFit,
    power: f64,
    link_power: f64,
}

impl FittedTweedieRegressor {
    pub fn coefficients(&self) -> Array1<f64> {
        self.fit.coef.clone()
    }

    pub fn intercept(&self) -> Option<f64> {
        self.fit.intercept
    }

    /// Deviance-based pseudo R² (`1 - D/D_null`).
    pub fn r_squared(&self) -> f64 {
        self.fit.d_squared()
    }

    pub fn deviance(&self) -> f64 {
        self.fit.deviance
    }

    pub fn power(&self) -> f64 {
        self.power
    }

    pub fn n_iter(&self) -> usize {
        self.fit.n_iter
    }

    pub fn converged(&self) -> bool {
        self.fit.converged
    }

    /// Coefficient table (requires `with_inference(true)`).
    pub fn summary(&self) -> Result<&RegressionSummary> {
        self.fit.summary()
    }

    /// Linear predictor `b + Xβ` (link scale, no offset).
    pub fn predict_linear<'a>(&self, x: impl AsArray<'a, f64, Ix2>) -> Result<Array1<f64>> {
        let x: ArrayView2<f64> = x.into();
        self.fit.linear_predictor(x.view(), None)
    }

    /// Predicted mean with an offset on the link scale.
    pub fn predict_with_offset(
        &self,
        x: &Array2<f64>,
        offset: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        let fam = TweedieFamily::new(self.power, self.link_power);
        Ok(self
            .fit
            .linear_predictor(x.view(), Some(offset))?
            .mapv(|e| GlmFamily::link_inverse(&fam, e)))
    }
}

impl Fit<f64> for TweedieRegressor {
    type Fitted = FittedTweedieRegressor;

    fn fit_view(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<Self::Fitted> {
        self.fit_with_offset(x, y, None)
    }
}

impl FitWeighted<f64> for TweedieRegressor {
    type Fitted = FittedTweedieRegressor;

    fn fit_weighted_view(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        sample_weight: Option<ArrayView1<f64>>,
    ) -> Result<Self::Fitted> {
        self.fit_impl(x, y, sample_weight, None)
    }
}

impl Predict<f64> for FittedTweedieRegressor {
    fn predict_view(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        let fam = TweedieFamily::new(self.power, self.link_power);
        Ok(self
            .fit
            .linear_predictor(x, None)?
            .mapv(|e| GlmFamily::link_inverse(&fam, e)))
    }
}

// ---------------------------------------------------------------------------
// GammaRegressor — convenience wrapper matching sklearn's class.
// ---------------------------------------------------------------------------

/// Gamma GLM regressor (Tweedie with power=2 and log link), mirroring
/// `sklearn.linear_model.GammaRegressor`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GammaRegressor {
    inner: TweedieRegressor,
}

impl GammaRegressor {
    pub fn new() -> Self {
        Self {
            inner: TweedieRegressor::new(2.0),
        }
    }
    /// L2 penalty strength on the mean deviance (sklearn's `alpha`). Default 0.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.inner = self.inner.with_alpha(alpha);
        self
    }
    pub fn with_intercept(mut self, include: bool) -> Self {
        self.inner = self.inner.with_intercept(include);
        self
    }
    pub fn with_max_iter(mut self, m: usize) -> Self {
        self.inner = self.inner.with_max_iter(m);
        self
    }
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.inner = self.inner.with_tol(tol);
        self
    }
    pub fn with_inference(mut self, inference: bool) -> Self {
        self.inner = self.inner.with_inference(inference);
        self
    }
    pub fn with_confidence_level(mut self, level: f64) -> Self {
        self.inner = self.inner.with_confidence_level(level);
        self
    }
    /// See [`TweedieRegressor::fit_with_offset`].
    pub fn fit_with_offset(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        offset: Option<&Array1<f64>>,
    ) -> Result<FittedTweedieRegressor> {
        self.inner.fit_with_offset(x.view(), y.view(), offset)
    }

    /// See [`TweedieRegressor::fit_weighted_with_offset`].
    pub fn fit_weighted_with_offset(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        sample_weight: Option<&Array1<f64>>,
        offset: Option<&Array1<f64>>,
    ) -> Result<FittedTweedieRegressor> {
        self.inner
            .fit_weighted_with_offset(x.view(), y.view(), sample_weight, offset)
    }
}

impl Default for GammaRegressor {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Gamma GLM regressor.
pub type FittedGammaRegressor = FittedTweedieRegressor;

impl Fit<f64> for GammaRegressor {
    type Fitted = FittedTweedieRegressor;
    fn fit_view(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<Self::Fitted> {
        self.inner.fit(x, y)
    }
}

impl FitWeighted<f64> for GammaRegressor {
    type Fitted = FittedTweedieRegressor;
    fn fit_weighted_view(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        sample_weight: Option<ArrayView1<f64>>,
    ) -> Result<Self::Fitted> {
        self.inner.fit_weighted_view(x, y, sample_weight)
    }
}

impl anofox_ml_core::RegressorScore<f64> for FittedTweedieRegressor {}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_gamma_predictions_positive() {
        let x =
            Array2::from_shape_vec((8, 1), vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]).unwrap();
        let y = array![1.0, 1.5, 2.0, 2.5, 3.5, 5.0, 7.0, 12.0];

        let fitted = GammaRegressor::new().fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(p > 0.0, "Gamma predictions must be positive, got {p}");
        }
    }

    #[test]
    fn test_tweedie_power_1p5() {
        let x = Array2::from_shape_vec((10, 1), (0..10).map(|i| i as f64).collect()).unwrap();
        let y = array![0.0, 0.1, 0.5, 0.0, 1.2, 0.0, 2.0, 3.5, 0.0, 4.0];
        let fitted = TweedieRegressor::new(1.5).fit(&x, &y).unwrap();
        assert!(fitted.converged());
        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(p >= 0.0);
        }
    }

    #[test]
    fn test_tweedie_rejects_invalid_power_and_targets() {
        let x = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        assert!(TweedieRegressor::new(0.5).fit(&x, &y).is_err());
        assert!(GammaRegressor::new()
            .fit(&x, &array![1.0, 0.0, 2.0])
            .is_err());
        assert!(TweedieRegressor::new(1.5)
            .fit(&x, &array![1.0, -1.0, 2.0])
            .is_err());
    }

    #[test]
    fn test_tweedie_gaussian_identity_alpha0_is_ols() {
        // power = 0 with the auto (identity) link and alpha = 0 is OLS.
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 3.1, 4.9, 7.2, 8.8];
        let f = TweedieRegressor::new(0.0)
            .with_alpha(0.0)
            .fit(&x, &y)
            .unwrap();
        // OLS slope = Sxy/Sxx = 19.7/10.
        assert!((f.coefficients()[0] - 1.97).abs() < 1e-10);
    }
}
