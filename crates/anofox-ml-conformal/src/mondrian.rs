//! Mondrian (group-conditional) split conformal regression.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use anofox_ml_core::{Fit, Float, Predict, Result, RustMlError};
use ndarray::{Array1, ArrayView1, ArrayView2, Axis};

use crate::quantile::{check_alpha, lower_quantile};
use crate::split::{check_xy, split_indices, too_few};

/// Mondrian split conformal regressor (Vovk et al. 2005): the calibration
/// quantile is taken within each group (a store, a segment, a sensor), so
/// the intervals cover `≥ 1 − α` **within every group**, not only on
/// average. Groups with harder targets get wider intervals. Each group
/// needs enough calibration rows for `alpha` (`⌈(1 − α)(n_g + 1)⌉ ≤ n_g`).
///
/// The [`Fit`] trait has no place for groups, so fit with
/// [`fit_with_groups`](Self::fit_with_groups), or calibrate a fitted model
/// with [`conformalize_mondrian`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MondrianConformalRegressor<E> {
    pub estimator: E,
    /// Fraction of the rows held out for calibration, in (0, 1).
    pub calibration_size: f64,
    pub seed: u64,
    /// Return infinite bounds for groups with too few calibration rows
    /// instead of an error.
    pub allow_infinite_bounds: bool,
}

impl<E> MondrianConformalRegressor<E> {
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

    /// Fit `estimator` on part of the rows (all groups together) and
    /// calibrate per group on the rest; `groups` holds one group id per row.
    pub fn fit_with_groups<F: Float>(
        &self,
        x: ArrayView2<F>,
        y: ArrayView1<F>,
        groups: &[usize],
    ) -> Result<FittedMondrianRegressor<E::Fitted, F>>
    where
        E: Fit<F>,
        E::Fitted: Predict<F>,
    {
        check_xy(x, y)?;
        check_groups(groups, x.nrows())?;
        let (train, cal) = split_indices(x.nrows(), self.calibration_size, self.seed)?;
        let model = self.estimator.fit_view(
            x.select(Axis(0), &train).view(),
            y.select(Axis(0), &train).view(),
        )?;
        let cal_groups: Vec<usize> = cal.iter().map(|&i| groups[i]).collect();
        Ok(conformalize_mondrian(
            model,
            x.select(Axis(0), &cal).view(),
            y.select(Axis(0), &cal).view(),
            &cal_groups,
        )?
        .with_allow_infinite_bounds(self.allow_infinite_bounds))
    }
}

pub(crate) fn check_groups(groups: &[usize], n: usize) -> Result<()> {
    if groups.len() != n {
        return Err(RustMlError::ShapeMismatch(format!(
            "{} group ids for {n} rows",
            groups.len()
        )));
    }
    Ok(())
}

/// Calibrate a fitted `model` per group on rows it was not trained on;
/// `groups` holds the group id of every calibration row.
pub fn conformalize_mondrian<F: Float, M: Predict<F>>(
    model: M,
    x_cal: ArrayView2<F>,
    y_cal: ArrayView1<F>,
    groups: &[usize],
) -> Result<FittedMondrianRegressor<M, F>> {
    check_xy(x_cal, y_cal)?;
    check_groups(groups, x_cal.nrows())?;
    let pred = model.predict_view(x_cal)?;
    let mut scores: BTreeMap<usize, Vec<F>> = BTreeMap::new();
    for ((&y, &p), &g) in y_cal.iter().zip(pred.iter()).zip(groups) {
        let s = (y - p).abs();
        if s.is_nan() {
            return Err(RustMlError::InvalidParameter(
                "NaN in the calibration targets or predictions".into(),
            ));
        }
        scores.entry(g).or_default().push(s);
    }
    Ok(FittedMondrianRegressor {
        model,
        scores,
        allow_infinite_bounds: false,
    })
}

/// A model plus calibration scores per group.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedMondrianRegressor<M, F> {
    model: M,
    scores: BTreeMap<usize, Vec<F>>,
    allow_infinite_bounds: bool,
}

impl<M: Predict<F>, F: Float> FittedMondrianRegressor<M, F> {
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Groups seen in calibration, with their number of calibration rows.
    pub fn group_sizes(&self) -> BTreeMap<usize, usize> {
        self.scores.iter().map(|(&g, s)| (g, s.len())).collect()
    }

    pub fn with_allow_infinite_bounds(mut self, allow: bool) -> Self {
        self.allow_infinite_bounds = allow;
        self
    }

    /// Half-width of the intervals of group `group` at `alpha`.
    pub fn margin(&self, group: usize, alpha: f64) -> Result<F> {
        check_alpha(alpha)?;
        let scores = self.scores.get(&group).ok_or_else(|| {
            RustMlError::InvalidParameter(format!("group {group} has no calibration rows"))
        })?;
        let mut s = scores.clone();
        match lower_quantile(&mut s, 1.0 - alpha) {
            Some(q) => Ok(q),
            None if self.allow_infinite_bounds => Ok(F::infinity()),
            None => Err(RustMlError::InvalidParameter(format!(
                "group {group}: {}",
                too_few(scores.len(), alpha)
            ))),
        }
    }

    /// Intervals `(lower, upper)` for rows `x` of groups `groups` (one id
    /// per row), covering `≥ 1 − alpha` within each group.
    pub fn predict_interval(
        &self,
        x: ArrayView2<F>,
        groups: &[usize],
        alpha: f64,
    ) -> Result<(Array1<F>, Array1<F>)> {
        check_groups(groups, x.nrows())?;
        let mut margins = BTreeMap::new();
        for &g in groups {
            if let std::collections::btree_map::Entry::Vacant(e) = margins.entry(g) {
                e.insert(self.margin(g, alpha)?);
            }
        }
        let pred = self.model.predict_view(x)?;
        let q: Array1<F> = groups.iter().map(|g| margins[g]).collect();
        Ok((&pred - &q, &pred + &q))
    }
}

impl<M: Predict<F>, F: Float> Predict<F> for FittedMondrianRegressor<M, F> {
    fn predict_view(&self, x: ArrayView2<F>) -> Result<Array1<F>> {
        self.model.predict_view(x)
    }
}
