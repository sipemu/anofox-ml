//! Split conformal regression with residual-normalised scores.

use anofox_ml_core::{Fit, Float, Predict, Result, RustMlError};
use ndarray::{Array1, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

use crate::quantile::{check_alpha, lower_quantile};
use crate::split::{check_xy, split_indices, too_few};

/// Floor of the predicted residual scale (MAPIE's `EPSILON`).
const EPS: f64 = 1e-8;

/// Split conformal regressor with locally adaptive widths (Lei et al.
/// 2018, §5.2): a second model `residual_estimator` learns the size of the
/// errors, `log |y − ŷ|`, and the score is `|y − ŷ| / σ̂(x)` with
/// `σ̂(x) = max(exp(r̂(x)), 1e-8)`. Intervals are `ŷ ∓ q · σ̂(x)`: wide
/// where the model is unsure, narrow where it is accurate, with the same
/// marginal coverage `≥ 1 − α`. Equivalent to MAPIE's
/// `SplitConformalRegressor` with `ResidualNormalisedScore`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizedConformalRegressor<E, R> {
    pub estimator: E,
    pub residual_estimator: R,
    /// Fraction of the rows held out for calibration, in (0, 1).
    pub calibration_size: f64,
    /// Fraction of the held-out rows used to fit `residual_estimator`; the
    /// rest computes the scores.
    pub residual_size: f64,
    pub seed: u64,
    pub allow_infinite_bounds: bool,
}

impl<E, R> NormalizedConformalRegressor<E, R> {
    pub fn new(estimator: E, residual_estimator: R) -> Self {
        Self {
            estimator,
            residual_estimator,
            calibration_size: 0.5,
            residual_size: 0.25,
            seed: 0,
            allow_infinite_bounds: false,
        }
    }

    pub fn with_calibration_size(mut self, calibration_size: f64) -> Self {
        self.calibration_size = calibration_size;
        self
    }

    pub fn with_residual_size(mut self, residual_size: f64) -> Self {
        self.residual_size = residual_size;
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

impl<F: Float, E: Fit<F>, R: Fit<F>> Fit<F> for NormalizedConformalRegressor<E, R>
where
    E::Fitted: Predict<F>,
    R::Fitted: Predict<F>,
{
    type Fitted = FittedNormalizedConformalRegressor<E::Fitted, R::Fitted, F>;

    fn fit_view(&self, x: ArrayView2<F>, y: ArrayView1<F>) -> Result<Self::Fitted> {
        check_xy(x, y)?;
        let (train, held) = split_indices(x.nrows(), self.calibration_size, self.seed)?;
        let model = self.estimator.fit_view(
            x.select(Axis(0), &train).view(),
            y.select(Axis(0), &train).view(),
        )?;
        let (cal, res) = split_indices(held.len(), self.residual_size, self.seed.wrapping_add(1))?;
        let pick = |idx: &[usize]| idx.iter().map(|&i| held[i]).collect::<Vec<usize>>();
        let (cal, res) = (pick(&cal), pick(&res));
        let residual_model = fit_residual_model(
            &self.residual_estimator,
            &model,
            x.select(Axis(0), &res).view(),
            y.select(Axis(0), &res).view(),
        )?;
        Ok(conformalize_normalized(
            model,
            residual_model,
            x.select(Axis(0), &cal).view(),
            y.select(Axis(0), &cal).view(),
        )?
        .with_allow_infinite_bounds(self.allow_infinite_bounds))
    }
}

/// Fit `residual_estimator` to `log(max(|y − model(x)|, 1e-8))`: the
/// residual model expected by [`conformalize_normalized`].
pub fn fit_residual_model<F: Float, M: Predict<F>, R: Fit<F>>(
    residual_estimator: &R,
    model: &M,
    x: ArrayView2<F>,
    y: ArrayView1<F>,
) -> Result<R::Fitted> {
    check_xy(x, y)?;
    let eps = F::from_f64(EPS).expect("eps fits in F");
    let pred = model.predict_view(x)?;
    let targets: Array1<F> = y
        .iter()
        .zip(pred.iter())
        .map(|(&t, &p)| (t - p).abs().max(eps).ln())
        .collect();
    residual_estimator.fit_view(x, targets.view())
}

/// Calibrate `model` with a `residual_model` that predicts the log of the
/// absolute residual (see [`fit_residual_model`]) on rows neither was
/// trained on.
pub fn conformalize_normalized<F: Float, M: Predict<F>, RM: Predict<F>>(
    model: M,
    residual_model: RM,
    x_cal: ArrayView2<F>,
    y_cal: ArrayView1<F>,
) -> Result<FittedNormalizedConformalRegressor<M, RM, F>> {
    check_xy(x_cal, y_cal)?;
    let pred = model.predict_view(x_cal)?;
    let scale = scale(&residual_model, x_cal)?;
    let scores: Vec<F> = y_cal
        .iter()
        .zip(pred.iter().zip(scale.iter()))
        .map(|(&y, (&p, &s))| (y - p).abs() / s)
        .collect();
    if scores.iter().any(|s| s.is_nan()) {
        return Err(RustMlError::InvalidParameter(
            "NaN in the calibration targets or predictions".into(),
        ));
    }
    Ok(FittedNormalizedConformalRegressor {
        model,
        residual_model,
        scores,
        allow_infinite_bounds: false,
    })
}

fn scale<F: Float, RM: Predict<F>>(residual_model: &RM, x: ArrayView2<F>) -> Result<Array1<F>> {
    let eps = F::from_f64(EPS).expect("eps fits in F");
    Ok(residual_model.predict_view(x)?.mapv(|r| r.exp().max(eps)))
}

/// A model, its residual-scale model and the normalised calibration scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedNormalizedConformalRegressor<M, RM, F> {
    model: M,
    residual_model: RM,
    scores: Vec<F>,
    allow_infinite_bounds: bool,
}

impl<M: Predict<F>, RM: Predict<F>, F: Float> FittedNormalizedConformalRegressor<M, RM, F> {
    pub fn model(&self) -> &M {
        &self.model
    }

    pub fn residual_model(&self) -> &RM {
        &self.residual_model
    }

    /// `|y − ŷ| / σ̂(x)` on the calibration rows.
    pub fn conformity_scores(&self) -> &[F] {
        &self.scores
    }

    pub fn with_allow_infinite_bounds(mut self, allow: bool) -> Self {
        self.allow_infinite_bounds = allow;
        self
    }

    /// Predicted error scale `σ̂(x)`.
    pub fn predict_scale(&self, x: ArrayView2<F>) -> Result<Array1<F>> {
        scale(&self.residual_model, x)
    }

    /// Prediction intervals `(lower, upper)` with marginal coverage
    /// `≥ 1 − alpha`.
    pub fn predict_interval(&self, x: ArrayView2<F>, alpha: f64) -> Result<(Array1<F>, Array1<F>)> {
        check_alpha(alpha)?;
        let mut s = self.scores.clone();
        let q = match lower_quantile(&mut s, 1.0 - alpha) {
            Some(q) => q,
            None if self.allow_infinite_bounds => F::infinity(),
            None => return Err(too_few(self.scores.len(), alpha)),
        };
        let pred = self.model.predict_view(x)?;
        let width = self.predict_scale(x)?.mapv(|s| q * s);
        Ok((&pred - &width, &pred + &width))
    }
}

impl<M: Predict<F>, RM: Predict<F>, F: Float> Predict<F>
    for FittedNormalizedConformalRegressor<M, RM, F>
{
    fn predict_view(&self, x: ArrayView2<F>) -> Result<Array1<F>> {
        self.model.predict_view(x)
    }
}
