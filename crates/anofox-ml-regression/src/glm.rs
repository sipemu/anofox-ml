//! GLM (Generalized Linear Model) wrappers for Poisson and Binomial regression.
//!
//! Both delegate the fit to `anofox_regression` (`PoissonRegressor`,
//! `BinomialRegressor`, penalised IRLS) and convert at the ndarray boundary.
//! They support an offset / exposure, an sklearn-scaled ridge `alpha`
//! (upstream `lambda = alpha · Σw`, intercept unpenalised) and opt-in Wald
//! inference ([`RegressionSummary`]). Binomial additionally takes sample
//! (trial) weights.
//!
//! Blocked on anofox-regression (#72): upstream's penalised IRLS does not
//! converge at sklearn's default `alpha = 1.0` and Poisson has no sample
//! weights, so `alpha` defaults to 0 (unpenalised, as R's `glm`) and
//! `PoissonRegressor` does not implement `FitWeighted` yet.

use crate::convert::{
    check_n_features, check_row_vector, check_xy, coef_from_col, exposure_to_offset,
    ndarray_to_col, ndarray_to_mat, resolve_sample_weight, upstream_err,
};
use crate::summary::{glm_summary_from_result, not_computed, RegressionSummary, StatisticKind};
use anofox_ml_core::{Fit, FitWeighted, Predict, Result};
use anofox_regression::solvers::{
    BinomialRegressor as InnerBinomial, FittedRegressor as _, PoissonRegressor as InnerPoisson,
    Regressor as _,
};
use anofox_regression::{BinomialFamily, BinomialLink as InnerLink, GlmFamily};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

pub(crate) fn check_response(
    y: &Array1<f64>,
    family: &str,
    ok: impl Fn(f64) -> bool,
    what: &str,
) -> Result<()> {
    if y.iter().any(|&v| !v.is_finite() || !ok(v)) {
        return Err(anofox_ml_core::RustMlError::InvalidParameter(format!(
            "{family} regression requires {what} finite targets"
        )));
    }
    Ok(())
}

/// Coefficients and diagnostics of a fitted GLM, copied out of the upstream
/// model so that fitted wrappers stay serializable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GlmFit {
    pub coef: Array1<f64>,
    pub intercept: Option<f64>,
    pub deviance: f64,
    pub null_deviance: f64,
    pub n_iter: usize,
    pub converged: bool,
    pub summary: Option<RegressionSummary>,
    pub n_features: usize,
}

impl GlmFit {
    /// Deviance-based pseudo R² (`1 - D/D_null`, sklearn's `d2` score).
    pub fn d_squared(&self) -> f64 {
        if self.null_deviance > 0.0 {
            1.0 - self.deviance / self.null_deviance
        } else {
            0.0
        }
    }

    pub fn summary(&self) -> Result<&RegressionSummary> {
        self.summary.as_ref().ok_or_else(not_computed)
    }

    /// `b + Xβ (+ offset)`.
    pub fn linear_predictor(
        &self,
        x: &Array2<f64>,
        offset: Option<&Array1<f64>>,
    ) -> Result<Array1<f64>> {
        check_n_features(x, self.n_features)?;
        check_row_vector(offset, x.nrows(), "offset")?;
        let mut eta = x.dot(&self.coef) + self.intercept.unwrap_or(0.0);
        if let Some(o) = offset {
            eta += o;
        }
        Ok(eta)
    }
}

/// Upstream ridge strength for sklearn's `alpha` on the weighted mean
/// deviance: `1/(2Σw) Σ wᵢ dᵢ + α/2 ‖β‖²` ⇔ `½ Σ wᵢ dᵢ + (α Σw)/2 ‖β‖²`.
pub(crate) fn upstream_lambda(alpha: f64, weight_sum: f64) -> Result<f64> {
    if !(alpha >= 0.0 && alpha.is_finite()) {
        return Err(anofox_ml_core::RustMlError::InvalidParameter(format!(
            "alpha must be finite and >= 0, got {alpha}"
        )));
    }
    Ok(alpha * weight_sum)
}

// ---------------------------------------------------------------------------
// Poisson
// ---------------------------------------------------------------------------

/// Poisson GLM regression estimator (log link).
///
/// Models count data where `E[Y] = exp(Xβ + offset)`. The penalised
/// objective follows `sklearn.linear_model.PoissonRegressor`:
///
/// ```text
/// 1/(2n) Σ d(yᵢ, μᵢ) + α/2 ‖β‖²      (intercept unpenalised)
/// ```
///
/// with `d` the Poisson unit deviance. `alpha` defaults to 0 (unpenalised
/// maximum likelihood, R's `glm(family = poisson)`); sklearn's default of
/// 1.0 is blocked on anofox-regression (#72).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PoissonRegressor {
    alpha: f64,
    with_intercept: bool,
    max_iter: usize,
    tol: f64,
    inference: bool,
    confidence_level: f64,
}

impl PoissonRegressor {
    pub fn new() -> Self {
        Self {
            alpha: 0.0,
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

    pub fn with_intercept(mut self, include: bool) -> Self {
        self.with_intercept = include;
        self
    }

    /// Maximum IRLS iterations. Default 100.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// IRLS convergence tolerance. Default 1e-8.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Compute Wald inference (SE, z, p, CI) at fit time. Default false.
    pub fn with_inference(mut self, inference: bool) -> Self {
        self.inference = inference;
        self
    }

    /// Confidence level of the summary intervals. Default 0.95.
    pub fn with_confidence_level(mut self, level: f64) -> Self {
        self.confidence_level = level;
        self
    }

    /// Fit with an optional offset on the linear-predictor (log) scale.
    pub fn fit_with_offset(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        offset: Option<&Array1<f64>>,
    ) -> Result<FittedPoissonRegressor> {
        check_xy(x, y)?;
        check_response(y, "Poisson", |v| v >= 0.0, "non-negative")?;
        check_row_vector(offset, y.len(), "offset")?;
        if self.inference {
            crate::summary::validate_level(self.confidence_level)?;
        }
        let mut b = InnerPoisson::log()
            .with_intercept(self.with_intercept)
            .lambda(upstream_lambda(self.alpha, y.len() as f64)?)
            .max_iterations(self.max_iter)
            .tolerance(self.tol)
            .compute_inference(self.inference)
            .confidence_level(self.confidence_level);
        if let Some(o) = offset {
            b = b.offset(ndarray_to_col(o));
        }
        let inner = b
            .build()
            .fit(&ndarray_to_mat(x), &ndarray_to_col(y))
            .map_err(upstream_err)?;
        let res = inner.result();
        let summary = if self.inference {
            Some(glm_summary_from_result(
                res,
                StatisticKind::Z,
                inner.dispersion,
            )?)
        } else {
            None
        };
        Ok(FittedPoissonRegressor {
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
        })
    }

    /// Rate model: fit with `offset = ln(exposure)` (e.g. counts per
    /// person-year). Predict rates with `predict`, counts with
    /// [`FittedPoissonRegressor::predict_with_exposure`].
    pub fn fit_with_exposure(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        exposure: &Array1<f64>,
    ) -> Result<FittedPoissonRegressor> {
        check_row_vector(Some(exposure), y.len(), "exposure")?;
        let offset = exposure_to_offset(exposure)?;
        self.fit_with_offset(x, y, Some(&offset))
    }
}

impl Default for PoissonRegressor {
    fn default() -> Self {
        Self::new()
    }
}

/// A fitted Poisson regression model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedPoissonRegressor {
    fit: GlmFit,
}

impl FittedPoissonRegressor {
    pub fn coefficients(&self) -> Array1<f64> {
        self.fit.coef.clone()
    }

    pub fn intercept(&self) -> Option<f64> {
        self.fit.intercept
    }

    /// Deviance-based pseudo R² (`1 - D/D_null`, sklearn's `d2` score).
    pub fn r_squared(&self) -> f64 {
        self.fit.d_squared()
    }

    /// Residual deviance at the solution.
    pub fn deviance(&self) -> f64 {
        self.fit.deviance
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

    /// Linear predictor `b + Xβ` (log scale, no offset).
    pub fn predict_linear(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        self.fit.linear_predictor(x, None)
    }

    /// Predicted mean `exp(b + Xβ + offset)`.
    pub fn predict_with_offset(
        &self,
        x: &Array2<f64>,
        offset: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        Ok(self.fit.linear_predictor(x, Some(offset))?.mapv(f64::exp))
    }

    /// Predicted counts for the given exposures: `exposure · exp(b + Xβ)`.
    pub fn predict_with_exposure(
        &self,
        x: &Array2<f64>,
        exposure: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        check_row_vector(Some(exposure), x.nrows(), "exposure")?;
        let offset = exposure_to_offset(exposure)?;
        self.predict_with_offset(x, &offset)
    }
}

impl Fit<f64> for PoissonRegressor {
    type Fitted = FittedPoissonRegressor;

    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        self.fit_with_offset(x, y, None)
    }
}

impl Predict<f64> for FittedPoissonRegressor {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        Ok(self.fit.linear_predictor(x, None)?.mapv(f64::exp))
    }
}

// ---------------------------------------------------------------------------
// Binomial
// ---------------------------------------------------------------------------

/// Link function of [`BinomialRegressor`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum BinomialLink {
    /// `log(μ / (1-μ))` (canonical; logistic regression).
    #[default]
    Logit,
    /// `Φ⁻¹(μ)`.
    Probit,
    /// `log(-log(1-μ))`.
    Cloglog,
}

impl BinomialLink {
    fn inner(self) -> InnerLink {
        match self {
            BinomialLink::Logit => InnerLink::Logit,
            BinomialLink::Probit => InnerLink::Probit,
            BinomialLink::Cloglog => InnerLink::Cloglog,
        }
    }

    fn family(self) -> BinomialFamily {
        BinomialFamily::new(self.inner())
    }
}

/// Binomial GLM regression estimator.
///
/// Models binary outcomes or proportions `y ∈ [0, 1]` (use `sample_weight`
/// for the number of trials, as R's `glm(weights = )`) with
/// `E[Y] = g⁻¹(Xβ + offset)` and a logit (default), probit or cloglog link.
/// Optional ridge `alpha` with the same scaling as [`PoissonRegressor`]
/// (default 0: unpenalised maximum likelihood).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BinomialRegressor {
    link: BinomialLink,
    alpha: f64,
    with_intercept: bool,
    max_iter: usize,
    tol: f64,
    inference: bool,
    confidence_level: f64,
}

impl BinomialRegressor {
    pub fn new() -> Self {
        Self {
            link: BinomialLink::Logit,
            alpha: 0.0,
            with_intercept: true,
            max_iter: 100,
            tol: 1e-8,
            inference: false,
            confidence_level: 0.95,
        }
    }

    pub fn with_intercept(mut self, include: bool) -> Self {
        self.with_intercept = include;
        self
    }

    /// Link function. Default [`BinomialLink::Logit`].
    pub fn with_link(mut self, link: BinomialLink) -> Self {
        self.link = link;
        self
    }

    /// L2 penalty strength on the weighted mean deviance. Default 0.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Compute Wald inference (SE, z, p, CI) at fit time. Default false.
    pub fn with_inference(mut self, inference: bool) -> Self {
        self.inference = inference;
        self
    }

    pub fn with_confidence_level(mut self, level: f64) -> Self {
        self.confidence_level = level;
        self
    }

    /// Fit with optional sample (trial) weights and an optional offset on the
    /// link scale.
    pub fn fit_with_offset(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        sample_weight: Option<&Array1<f64>>,
        offset: Option<&Array1<f64>>,
    ) -> Result<FittedBinomialRegressor> {
        check_xy(x, y)?;
        check_response(y, "Binomial", |v| (0.0..=1.0).contains(&v), "[0, 1]-valued")?;
        check_row_vector(offset, y.len(), "offset")?;
        if self.inference {
            crate::summary::validate_level(self.confidence_level)?;
        }
        let w = resolve_sample_weight(sample_weight, y.len())?;
        let mut b = InnerBinomial::builder()
            .link(self.link.inner())
            .with_intercept(self.with_intercept)
            .lambda(upstream_lambda(self.alpha, w.sum())?)
            .max_iterations(self.max_iter)
            .tolerance(self.tol)
            .compute_inference(self.inference)
            .confidence_level(self.confidence_level);
        if sample_weight.is_some() {
            b = b.weights(ndarray_to_col(&w));
        }
        if let Some(o) = offset {
            b = b.offset(ndarray_to_col(o));
        }
        let inner = b
            .build()
            .fit(&ndarray_to_mat(x), &ndarray_to_col(y))
            .map_err(upstream_err)?;
        let res = inner.result();
        let summary = if self.inference {
            Some(glm_summary_from_result(res, StatisticKind::Z, 1.0)?)
        } else {
            None
        };
        Ok(FittedBinomialRegressor {
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
            link: self.link,
        })
    }
}

impl Default for BinomialRegressor {
    fn default() -> Self {
        Self::new()
    }
}

/// A fitted Binomial regression model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedBinomialRegressor {
    fit: GlmFit,
    #[serde(default)]
    link: BinomialLink,
}

impl FittedBinomialRegressor {
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

    pub fn link(&self) -> BinomialLink {
        self.link
    }

    pub fn n_iter(&self) -> usize {
        self.fit.n_iter
    }

    pub fn converged(&self) -> bool {
        self.fit.converged
    }

    /// Coefficient table (requires `with_inference(true)`). For the logit
    /// link, [`RegressionSummary::odds_ratios`] gives odds ratios with CIs.
    pub fn summary(&self) -> Result<&RegressionSummary> {
        self.fit.summary()
    }

    /// Linear predictor `b + Xβ` (link scale).
    pub fn predict_linear(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        self.fit.linear_predictor(x, None)
    }

    /// Predicted probability with an offset on the link scale.
    pub fn predict_with_offset(
        &self,
        x: &Array2<f64>,
        offset: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        let fam = self.link.family();
        Ok(self
            .fit
            .linear_predictor(x, Some(offset))?
            .mapv(|e| fam.link_inverse(e)))
    }
}

impl Fit<f64> for BinomialRegressor {
    type Fitted = FittedBinomialRegressor;

    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        self.fit_with_offset(x, y, None, None)
    }
}

impl FitWeighted<f64> for BinomialRegressor {
    type Fitted = FittedBinomialRegressor;

    fn fit_weighted(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        sample_weight: Option<&Array1<f64>>,
    ) -> Result<Self::Fitted> {
        self.fit_with_offset(x, y, sample_weight, None)
    }
}

impl Predict<f64> for FittedBinomialRegressor {
    /// Predicted probabilities `g⁻¹(b + Xβ)`.
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let fam = self.link.family();
        Ok(self
            .fit
            .linear_predictor(x, None)?
            .mapv(|e| fam.link_inverse(e)))
    }
}

impl anofox_ml_core::RegressorScore<f64> for FittedPoissonRegressor {}
impl anofox_ml_core::RegressorScore<f64> for FittedBinomialRegressor {}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_poisson_basic() {
        let x =
            Array2::from_shape_vec((8, 1), vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]).unwrap();
        let y = array![1.0, 1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 8.0];

        let fitted = PoissonRegressor::new().fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(p > 0.0, "Poisson predictions must be positive, got {}", p);
        }
    }

    #[test]
    fn test_binomial_basic() {
        let x = Array2::from_shape_vec((8, 1), vec![-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0])
            .unwrap();
        let y = array![0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0];

        let fitted = BinomialRegressor::new().fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!((0.0..=1.0).contains(&p));
        }
    }

    #[test]
    fn test_poisson_rejects_negative_counts() {
        let x = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        assert!(PoissonRegressor::new()
            .fit(&x, &array![1.0, -1.0, 2.0])
            .is_err());
    }

    #[test]
    fn test_binomial_integer_weights_equal_row_duplication() {
        let x = Array2::from_shape_vec((6, 1), vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]).unwrap();
        let y = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let w = array![1.0, 2.0, 1.0, 3.0, 1.0, 1.0];
        let xd =
            Array2::from_shape_vec((9, 1), vec![-2.0, -1.0, -1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 3.0])
                .unwrap();
        let yd = array![0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0];
        let a = BinomialRegressor::new()
            .fit_weighted(&x, &y, Some(&w))
            .unwrap();
        let b = BinomialRegressor::new().fit(&xd, &yd).unwrap();
        assert_abs_diff_eq!(a.coefficients()[0], b.coefficients()[0], epsilon = 1e-8);
        assert_abs_diff_eq!(
            a.intercept().unwrap(),
            b.intercept().unwrap(),
            epsilon = 1e-8
        );
    }

    #[test]
    fn test_poisson_exposure_recovers_rate() {
        // Counts are exposure * exp(0.5 + 0.3 x) exactly: a rate model with
        // offset = ln(exposure) must recover the rate coefficients.
        let xs: [f64; 6] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let exposure = array![1.0, 2.0, 0.5, 4.0, 3.0, 1.5];
        let y = Array1::from_iter(
            xs.iter()
                .zip(exposure.iter())
                .map(|(&x, &e)| e * (0.5 + 0.3 * x).exp()),
        );
        let x = Array2::from_shape_vec((6, 1), xs.to_vec()).unwrap();
        let fitted = PoissonRegressor::new()
            .fit_with_exposure(&x, &y, &exposure)
            .unwrap();
        assert_abs_diff_eq!(fitted.coefficients()[0], 0.3, epsilon = 1e-6);
        assert_abs_diff_eq!(fitted.intercept().unwrap(), 0.5, epsilon = 1e-6);
        let counts = fitted.predict_with_exposure(&x, &exposure).unwrap();
        for (c, t) in counts.iter().zip(y.iter()) {
            assert_abs_diff_eq!(*c, *t, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_summary_requires_inference() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 2.0, 5.0, 9.0];
        let f = PoissonRegressor::new().fit(&x, &y).unwrap();
        assert!(f.summary().is_err());
        let f = PoissonRegressor::new()
            .with_inference(true)
            .fit(&x, &y)
            .unwrap();
        assert!(f.summary().is_ok());
    }

    #[test]
    fn test_binomial_links_differ_and_are_valid() {
        let x = Array2::from_shape_vec((8, 1), vec![-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0])
            .unwrap();
        let y = array![0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0];
        let mut slopes = vec![];
        for link in [
            BinomialLink::Logit,
            BinomialLink::Probit,
            BinomialLink::Cloglog,
        ] {
            let f = BinomialRegressor::new()
                .with_link(link)
                .fit(&x, &y)
                .unwrap();
            assert!(f.converged());
            assert_eq!(f.link(), link);
            slopes.push(f.coefficients()[0]);
        }
        // Probit slopes are ~1/1.6-1/1.8 of logit slopes.
        let ratio = slopes[1] / slopes[0];
        assert!(ratio > 0.5 && ratio < 0.7, "probit/logit ratio {ratio}");
    }
}
