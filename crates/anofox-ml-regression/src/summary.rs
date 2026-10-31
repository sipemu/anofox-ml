//! Opt-in statistical inference for the linear / GLM wrappers.
//!
//! Estimators that support inference expose a `with_inference(true)` builder
//! switch; their fitted models then return a [`RegressionSummary`]
//! (coefficients, standard errors, t / z statistics, p-values and confidence
//! intervals, like R's `summary(lm)` / `summary(glm)`), and the linear models
//! additionally offer `predict_interval` ([`PredictionInterval`]).

use crate::convert::col_to_ndarray;
use anofox_ml_core::{Result, RustMlError};
use anofox_regression::{FittedRegressor, IntervalType, RegressionResult};
use faer::Mat;
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Which reference distribution the test statistics follow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StatisticKind {
    /// Student-t with the residual degrees of freedom (linear models, GLMs
    /// with an estimated dispersion such as Gamma / Tweedie).
    T,
    /// Standard normal (GLMs with a fixed dispersion: Poisson, Binomial,
    /// logistic regression).
    Z,
}

/// Inference for the intercept term.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterceptSummary {
    pub estimate: f64,
    pub std_error: f64,
    pub statistic: f64,
    pub p_value: f64,
    pub conf_lower: f64,
    pub conf_upper: f64,
}

/// Coefficient table of a fitted model (R's `summary()` coefficient block).
///
/// All arrays are indexed by feature (same order as `coefficients()`); the
/// intercept is reported separately.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegressionSummary {
    pub coefficients: Array1<f64>,
    pub std_errors: Array1<f64>,
    /// t or z statistics (see [`RegressionSummary::statistic_kind`]).
    pub statistics: Array1<f64>,
    /// Two-sided p-values.
    pub p_values: Array1<f64>,
    pub conf_lower: Array1<f64>,
    pub conf_upper: Array1<f64>,
    pub intercept: Option<InterceptSummary>,
    pub confidence_level: f64,
    pub statistic_kind: StatisticKind,
    /// Residual degrees of freedom used for t statistics.
    pub residual_df: f64,
    /// Dispersion (residual variance for linear models; 1 for Poisson /
    /// Binomial; the Pearson estimate for Gamma / Tweedie).
    pub dispersion: f64,
}

impl RegressionSummary {
    /// `exp` of the coefficients and of their confidence limits: odds ratios
    /// for logit links, rate / mean ratios for log links.
    pub fn exp_coefficients(&self) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        (
            self.coefficients.mapv(f64::exp),
            self.conf_lower.mapv(f64::exp),
            self.conf_upper.mapv(f64::exp),
        )
    }

    /// Alias for [`Self::exp_coefficients`] (logistic / binomial-logit models).
    pub fn odds_ratios(&self) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        self.exp_coefficients()
    }
}

/// Kind of interval returned by `predict_interval`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntervalKind {
    /// Interval for the mean response `E[y | x]`.
    Confidence,
    /// Interval for a new observation `y | x` (adds the residual variance).
    Prediction,
}

/// Point predictions with interval bounds and standard errors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PredictionInterval {
    pub fit: Array1<f64>,
    pub lower: Array1<f64>,
    pub upper: Array1<f64>,
    pub se: Array1<f64>,
}

pub(crate) fn not_computed() -> RustMlError {
    RustMlError::InvalidParameter(
        "inference was not computed: fit the estimator with `.with_inference(true)`".into(),
    )
}

pub(crate) fn validate_level(level: f64) -> Result<()> {
    if level.is_nan() || level <= 0.0 || level >= 1.0 {
        return Err(RustMlError::InvalidParameter(format!(
            "confidence level must be in (0, 1), got {level}"
        )));
    }
    Ok(())
}

/// Build a [`RegressionSummary`] from an upstream linear-model result
/// (OLS / WLS / Ridge, t statistics). Statistics upstream does not report
/// (t / p / CI of a penalised ridge fit) are `NaN`.
pub(crate) fn summary_from_result(r: &RegressionResult) -> Result<RegressionSummary> {
    let se = col_to_ndarray(r.std_errors.as_ref().ok_or_else(not_computed)?);
    let d = se.len();
    let or_nan = |c: &Option<faer::Col<f64>>| match c {
        Some(c) => col_to_ndarray(c),
        None => Array1::from_elem(d, f64::NAN),
    };
    let intercept = match r.intercept {
        Some(b) => {
            let (cl, cu) = r.intercept_conf_interval.unwrap_or((f64::NAN, f64::NAN));
            Some(InterceptSummary {
                estimate: b,
                std_error: r.intercept_std_error.unwrap_or(f64::NAN),
                statistic: r.intercept_t_statistic.unwrap_or(f64::NAN),
                p_value: r.intercept_p_value.unwrap_or(f64::NAN),
                conf_lower: cl,
                conf_upper: cu,
            })
        }
        None => None,
    };
    Ok(RegressionSummary {
        coefficients: col_to_ndarray(&r.coefficients),
        std_errors: se,
        statistics: or_nan(&r.t_statistics),
        p_values: or_nan(&r.p_values),
        conf_lower: or_nan(&r.conf_interval_lower),
        conf_upper: or_nan(&r.conf_interval_upper),
        intercept,
        confidence_level: r.confidence_level,
        statistic_kind: StatisticKind::T,
        residual_df: r.residual_df() as f64,
        dispersion: r.mse,
    })
}

/// Build a [`RegressionSummary`] from an upstream GLM result. `kind` is the
/// reference distribution of the statistics (z for fixed-dispersion
/// families, t when the dispersion is estimated) and `dispersion` the
/// family's dispersion estimate.
pub(crate) fn glm_summary_from_result(
    r: &RegressionResult,
    kind: StatisticKind,
    dispersion: f64,
) -> Result<RegressionSummary> {
    let mut s = summary_from_result(r)?;
    s.statistic_kind = kind;
    s.dispersion = dispersion;
    Ok(s)
}

/// Confidence / prediction intervals from an upstream linear model.
pub(crate) fn interval_from<M: FittedRegressor>(
    m: &M,
    x: &Mat<f64>,
    kind: IntervalKind,
    level: f64,
) -> Result<PredictionInterval> {
    validate_level(level)?;
    let it = match kind {
        IntervalKind::Confidence => IntervalType::Confidence,
        IntervalKind::Prediction => IntervalType::Prediction,
    };
    let r = m.predict_with_interval(x, Some(it), level);
    let out = PredictionInterval {
        fit: col_to_ndarray(&r.fit),
        lower: col_to_ndarray(&r.lower),
        upper: col_to_ndarray(&r.upper),
        se: col_to_ndarray(&r.se),
    };
    if out.se.iter().any(|v| v.is_nan()) && x.nrows() > 0 {
        return Err(not_computed());
    }
    Ok(out)
}
