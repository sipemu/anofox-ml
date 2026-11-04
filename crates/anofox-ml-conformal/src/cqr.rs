//! Conformalized quantile regression (CQR).

use std::marker::PhantomData;

use anofox_ml_core::{Fit, Float, Predict, Result, RustMlError};
use ndarray::{Array1, ArrayView1, ArrayView2, Axis};

use crate::quantile::{check_alpha, lower_quantile};
use crate::split::{check_xy, split_indices, too_few};

/// Conformalized quantile regression (Romano, Patterson & Candès 2019):
/// fits quantile regressors at `α/2`, `1 − α/2` and the median on part of
/// the data, then widens (or narrows) the quantile band by the conformal
/// quantile of the calibration errors. Intervals adapt to
/// heteroscedasticity through the quantile models and keep marginal
/// coverage `≥ 1 − α`. Equivalent to MAPIE's
/// `ConformalizedQuantileRegressor`.
///
/// `make` builds a regressor for a quantile level, e.g.
/// `|q| GradientBoostingRegressor::new().with_loss(Loss::Quantile).with_alpha(q)`.
/// Unlike the other conformal regressors, `alpha` is fixed at fit time
/// because the quantile models are trained for it. To conformalize three
/// models you fitted yourself, use [`conformalize_quantiles`].
pub struct ConformalizedQuantileRegressor<B, E> {
    pub make: B,
    pub alpha: f64,
    /// Fraction of the rows held out for calibration, in (0, 1).
    pub calibration_size: f64,
    pub seed: u64,
    /// One correction for both bounds (Romano et al.) instead of a separate
    /// correction per bound at level `1 − α/2` each (MAPIE's default).
    pub symmetric_correction: bool,
    pub allow_infinite_bounds: bool,
    _estimator: PhantomData<fn() -> E>,
}

impl<B: Fn(f64) -> E, E> ConformalizedQuantileRegressor<B, E> {
    pub fn new(alpha: f64, make: B) -> Self {
        Self {
            make,
            alpha,
            calibration_size: 0.25,
            seed: 0,
            symmetric_correction: false,
            allow_infinite_bounds: false,
            _estimator: PhantomData,
        }
    }

    pub fn with_calibration_size(mut self, calibration_size: f64) -> Self {
        self.calibration_size = calibration_size;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_symmetric_correction(mut self, symmetric: bool) -> Self {
        self.symmetric_correction = symmetric;
        self
    }

    pub fn with_allow_infinite_bounds(mut self, allow: bool) -> Self {
        self.allow_infinite_bounds = allow;
        self
    }
}

impl<F: Float, B: Fn(f64) -> E, E: Fit<F>> Fit<F> for ConformalizedQuantileRegressor<B, E>
where
    E::Fitted: Predict<F>,
{
    type Fitted = FittedConformalizedQuantileRegressor<E::Fitted, F>;

    fn fit_view(&self, x: ArrayView2<F>, y: ArrayView1<F>) -> Result<Self::Fitted> {
        check_xy(x, y)?;
        check_alpha(self.alpha)?;
        let (train, cal) = split_indices(x.nrows(), self.calibration_size, self.seed)?;
        let (xt, yt) = (x.select(Axis(0), &train), y.select(Axis(0), &train));
        let fit_at = |q: f64| (self.make)(q).fit_view(xt.view(), yt.view());
        let lower = fit_at(self.alpha / 2.0)?;
        let upper = fit_at(1.0 - self.alpha / 2.0)?;
        let median = fit_at(0.5)?;
        Ok(conformalize_quantiles(
            lower,
            upper,
            median,
            x.select(Axis(0), &cal).view(),
            y.select(Axis(0), &cal).view(),
            self.alpha,
        )?
        .with_symmetric_correction(self.symmetric_correction)
        .with_allow_infinite_bounds(self.allow_infinite_bounds))
    }
}

/// Calibrate fitted quantile models for the `alpha / 2` (`lower`),
/// `1 − alpha / 2` (`upper`) and 0.5 (`median`) quantiles on rows they
/// were not trained on (MAPIE's `ConformalizedQuantileRegressor(prefit=True)`).
pub fn conformalize_quantiles<F: Float, M: Predict<F>>(
    lower: M,
    upper: M,
    median: M,
    x_cal: ArrayView2<F>,
    y_cal: ArrayView1<F>,
    alpha: f64,
) -> Result<FittedConformalizedQuantileRegressor<M, F>> {
    check_xy(x_cal, y_cal)?;
    check_alpha(alpha)?;
    let lo = lower.predict_view(x_cal)?;
    let hi = upper.predict_view(x_cal)?;
    let scores_low: Vec<F> = lo.iter().zip(y_cal).map(|(&q, &y)| q - y).collect();
    let scores_up: Vec<F> = hi.iter().zip(y_cal).map(|(&q, &y)| y - q).collect();
    if scores_low.iter().chain(&scores_up).any(|s| s.is_nan()) {
        return Err(RustMlError::InvalidParameter(
            "NaN in the calibration targets or quantile predictions".into(),
        ));
    }
    Ok(FittedConformalizedQuantileRegressor {
        lower,
        upper,
        median,
        scores_low,
        scores_up,
        alpha,
        symmetric_correction: false,
        allow_infinite_bounds: false,
    })
}

/// Quantile models plus their calibration errors.
#[derive(Debug, Clone)]
pub struct FittedConformalizedQuantileRegressor<M, F> {
    lower: M,
    upper: M,
    median: M,
    /// `q_lo(x) − y`: positive when the lower quantile is above `y`.
    scores_low: Vec<F>,
    /// `y − q_hi(x)`: positive when `y` is above the upper quantile.
    scores_up: Vec<F>,
    alpha: f64,
    symmetric_correction: bool,
    allow_infinite_bounds: bool,
}

impl<M: Predict<F>, F: Float> FittedConformalizedQuantileRegressor<M, F> {
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn with_symmetric_correction(mut self, symmetric: bool) -> Self {
        self.symmetric_correction = symmetric;
        self
    }

    pub fn with_allow_infinite_bounds(mut self, allow: bool) -> Self {
        self.allow_infinite_bounds = allow;
        self
    }

    /// Calibration errors of the lower and upper quantile models.
    pub fn conformity_scores(&self) -> (&[F], &[F]) {
        (&self.scores_low, &self.scores_up)
    }

    /// The corrections `(lower, upper)` added outside the quantile band
    /// (negative values narrow it).
    pub fn corrections(&self) -> Result<(F, F)> {
        let q = |scores: &[F], level: f64| -> Result<F> {
            let mut s = scores.to_vec();
            match lower_quantile(&mut s, level) {
                Some(v) => Ok(v),
                None if self.allow_infinite_bounds => Ok(F::infinity()),
                None => Err(too_few(scores.len(), self.alpha)),
            }
        };
        if self.symmetric_correction {
            let both: Vec<F> = self
                .scores_low
                .iter()
                .zip(&self.scores_up)
                .map(|(&a, &b)| a.max(b))
                .collect();
            let c = q(&both, 1.0 - self.alpha)?;
            Ok((c, c))
        } else {
            // Same float formula as MAPIE (alpha halved first).
            let level = 1.0 - self.alpha / 2.0;
            Ok((q(&self.scores_low, level)?, q(&self.scores_up, level)?))
        }
    }

    /// Prediction intervals `(lower, upper)` at the `alpha` given at fit.
    pub fn predict_interval(&self, x: ArrayView2<F>) -> Result<(Array1<F>, Array1<F>)> {
        let (c_lo, c_up) = self.corrections()?;
        let lo = self.lower.predict_view(x)?.mapv(|v| v - c_lo);
        let hi = self.upper.predict_view(x)?.mapv(|v| v + c_up);
        Ok((lo, hi))
    }
}

impl<M: Predict<F>, F: Float> Predict<F> for FittedConformalizedQuantileRegressor<M, F> {
    /// The median model's prediction.
    fn predict_view(&self, x: ArrayView2<F>) -> Result<Array1<F>> {
        self.median.predict_view(x)
    }
}
