//! Split conformal regression.

use anofox_ml_core::{Fit, Float, Predict, Result, RustMlError};
use ndarray::{Array1, ArrayView1, ArrayView2, Axis};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use crate::quantile::{check_alpha, lower_quantile, needed};

/// Split conformal regressor: fits `estimator` on part of the data and
/// uses the absolute residuals on the rest (the calibration set) to turn
/// its predictions into intervals with marginal coverage `≥ 1 − α`
/// (Lei et al. 2018). Equivalent to MAPIE's `SplitConformalRegressor`
/// with the `absolute` conformity score.
///
/// To calibrate an already fitted model on your own calibration data, use
/// [`conformalize`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitConformalRegressor<E> {
    pub estimator: E,
    /// Fraction of the rows held out for calibration, in (0, 1).
    pub calibration_size: f64,
    /// Seeds the random train / calibration split.
    pub seed: u64,
    /// Return infinite bounds instead of an error when the calibration set
    /// is too small for the requested `alpha`.
    pub allow_infinite_bounds: bool,
}

impl<E> SplitConformalRegressor<E> {
    pub fn new(estimator: E) -> Self {
        Self {
            estimator,
            calibration_size: 0.25,
            seed: 0,
            allow_infinite_bounds: false,
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

    pub fn with_allow_infinite_bounds(mut self, allow: bool) -> Self {
        self.allow_infinite_bounds = allow;
        self
    }
}

impl<F: Float, E: Fit<F>> Fit<F> for SplitConformalRegressor<E>
where
    E::Fitted: Predict<F>,
{
    type Fitted = FittedSplitConformalRegressor<E::Fitted, F>;

    fn fit_view(&self, x: ArrayView2<F>, y: ArrayView1<F>) -> Result<Self::Fitted> {
        check_xy(x, y)?;
        let (train, cal) = split_indices(x.nrows(), self.calibration_size, self.seed)?;
        let (train, cal) = (&train[..], &cal[..]);
        let model = self.estimator.fit_view(
            x.select(Axis(0), train).view(),
            y.select(Axis(0), train).view(),
        )?;
        let mut fitted = conformalize(
            model,
            x.select(Axis(0), cal).view(),
            y.select(Axis(0), cal).view(),
        )?;
        fitted.allow_infinite_bounds = self.allow_infinite_bounds;
        Ok(fitted)
    }
}

/// Calibrate a fitted `model` on `(x_cal, y_cal)`, which it must not have
/// been trained on (MAPIE's `SplitConformalRegressor(prefit=True)` +
/// `conformalize`).
pub fn conformalize<F: Float, M: Predict<F>>(
    model: M,
    x_cal: ArrayView2<F>,
    y_cal: ArrayView1<F>,
) -> Result<FittedSplitConformalRegressor<M, F>> {
    check_xy(x_cal, y_cal)?;
    let pred = model.predict_view(x_cal)?;
    let scores: Vec<F> = y_cal
        .iter()
        .zip(pred.iter())
        .map(|(&y, &p)| (y - p).abs())
        .collect();
    if scores.iter().any(|s| s.is_nan()) {
        return Err(RustMlError::InvalidParameter(
            "NaN in the calibration targets or predictions".into(),
        ));
    }
    Ok(FittedSplitConformalRegressor {
        model,
        scores,
        allow_infinite_bounds: false,
    })
}

/// A model plus its calibration scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedSplitConformalRegressor<M, F> {
    model: M,
    scores: Vec<F>,
    allow_infinite_bounds: bool,
}

impl<M: Predict<F>, F: Float> FittedSplitConformalRegressor<M, F> {
    /// The underlying fitted model.
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Absolute residuals on the calibration set.
    pub fn conformity_scores(&self) -> &[F] {
        &self.scores
    }

    pub fn with_allow_infinite_bounds(mut self, allow: bool) -> Self {
        self.allow_infinite_bounds = allow;
        self
    }

    /// Half-width of the intervals at miscoverage `alpha` (`+∞` when
    /// allowed and the calibration set is too small).
    pub fn margin(&self, alpha: f64) -> Result<F> {
        check_alpha(alpha)?;
        let mut s = self.scores.clone();
        match lower_quantile(&mut s, 1.0 - alpha) {
            Some(q) => Ok(q),
            None if self.allow_infinite_bounds => Ok(F::infinity()),
            None => Err(too_few(self.scores.len(), alpha)),
        }
    }

    /// Prediction intervals `(lower, upper)` with marginal coverage
    /// `≥ 1 − alpha`.
    pub fn predict_interval(&self, x: ArrayView2<F>, alpha: f64) -> Result<(Array1<F>, Array1<F>)> {
        let q = self.margin(alpha)?;
        let pred = self.model.predict_view(x)?;
        Ok((pred.mapv(|p| p - q), pred.mapv(|p| p + q)))
    }
}

impl<M: Predict<F>, F: Float> Predict<F> for FittedSplitConformalRegressor<M, F> {
    fn predict_view(&self, x: ArrayView2<F>) -> Result<Array1<F>> {
        self.model.predict_view(x)
    }
}

pub(crate) fn too_few(n: usize, alpha: f64) -> RustMlError {
    RustMlError::InvalidParameter(format!(
        "{n} calibration scores are too few for alpha = {alpha}: at least {} are needed \
         (or allow infinite bounds)",
        needed(alpha)
    ))
}

pub(crate) fn check_xy<F: Float>(x: ArrayView2<F>, y: ArrayView1<F>) -> Result<()> {
    if x.nrows() != y.len() {
        return Err(RustMlError::ShapeMismatch(format!(
            "x has {} rows, y has {} values",
            x.nrows(),
            y.len()
        )));
    }
    if x.nrows() == 0 {
        return Err(RustMlError::EmptyInput("no rows".into()));
    }
    Ok(())
}

/// Seeded random split of `0..n` into `(train, held_out)`, with
/// `⌈fraction · n⌉` held-out rows (at least one row on each side).
pub(crate) fn split_indices(
    n: usize,
    fraction: f64,
    seed: u64,
) -> Result<(Vec<usize>, Vec<usize>)> {
    if !(fraction > 0.0 && fraction < 1.0) {
        return Err(RustMlError::InvalidParameter(format!(
            "split fraction must lie in (0, 1), got {fraction}"
        )));
    }
    let n_out = ((fraction * n as f64).ceil() as usize).min(n.saturating_sub(1));
    if n_out == 0 || n < 2 {
        return Err(RustMlError::InvalidParameter(format!(
            "{n} rows are too few to split"
        )));
    }
    let mut perm: Vec<usize> = (0..n).collect();
    perm.shuffle(&mut rand::rngs::StdRng::seed_from_u64(seed));
    let train = perm.split_off(n_out);
    Ok((train, perm))
}
