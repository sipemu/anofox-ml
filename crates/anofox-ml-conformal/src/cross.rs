//! Cross-conformal regression: CV+, jackknife+, CV-minmax.

use anofox_ml_core::utils::{shuffled_k_fold, CvSplit};
use anofox_ml_core::{Fit, Float, Predict, Result, RustMlError};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

use crate::quantile::{check_alpha, lower_quantile};
use crate::split::{check_xy, too_few};

/// How the fold models become an interval.
#[derive(Debug, Clone, Serialize, Deserialize, Copy, PartialEq, Eq, Default)]
pub enum CrossMethod {
    /// CV+ (jackknife+ with leave-one-out splits): quantiles of each fold
    /// model's prediction ∓ the out-of-fold residuals of its own fold
    /// (Barber et al. 2021). Coverage `≥ 1 − 2α` in theory, about `1 − α` in
    /// practice.
    #[default]
    Plus,
    /// CV-minmax: the lowest / highest fold-model prediction ∓ the
    /// conformal quantile of the out-of-fold residuals. Coverage `≥ 1 − α`,
    /// wider intervals.
    MinMax,
    /// The full-data model ∓ the conformal quantile of the out-of-fold
    /// residuals. Cheapest, no coverage guarantee. Point predictions are
    /// always the full-data model's (the interval's centre).
    Base,
}

/// Point prediction of the fitted cross-conformal regressor (MAPIE's
/// `aggregate_point_predictions`).
#[derive(Debug, Clone, Serialize, Deserialize, Copy, PartialEq, Eq, Default)]
pub enum PointPrediction {
    /// Mean of the fold models, each weighted by its fold's size.
    #[default]
    Mean,
    /// Median over the training rows of the prediction of the fold model
    /// that did not see that row.
    Median,
    /// The model fitted on all rows.
    Full,
}

/// Cross-conformal regressor: fits `estimator` on all rows and once per CV
/// fold; the out-of-fold absolute residuals calibrate the intervals.
/// Equivalent to MAPIE's `CrossConformalRegressor` with the `absolute`
/// conformity score. Jackknife+ is [`CrossMethod::Plus`] with
/// leave-one-out splits ([`fit_with_splits`](Self::fit_with_splits) with
/// `anofox_ml_core::utils::leave_one_out`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossConformalRegressor<E> {
    pub estimator: E,
    /// Number of folds (shuffled K-fold); see also
    /// [`fit_with_splits`](Self::fit_with_splits).
    pub cv: usize,
    pub method: CrossMethod,
    pub point_prediction: PointPrediction,
    pub seed: u64,
    /// Return infinite bounds instead of an error when there are too few
    /// rows for the requested `alpha`.
    pub allow_infinite_bounds: bool,
}

impl<E> CrossConformalRegressor<E> {
    pub fn new(estimator: E) -> Self {
        Self {
            estimator,
            cv: 5,
            method: CrossMethod::Plus,
            point_prediction: PointPrediction::Mean,
            seed: 0,
            allow_infinite_bounds: false,
        }
    }

    pub fn with_cv(mut self, cv: usize) -> Self {
        self.cv = cv;
        self
    }

    pub fn with_method(mut self, method: CrossMethod) -> Self {
        self.method = method;
        self
    }

    pub fn with_point_prediction(mut self, point_prediction: PointPrediction) -> Self {
        self.point_prediction = point_prediction;
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

impl<E> CrossConformalRegressor<E> {
    /// Fit over caller-supplied `(train, test)` splits whose test sets
    /// partition the rows (K-fold, group K-fold, leave-one-out, ...).
    pub fn fit_with_splits<F: Float>(
        &self,
        x: ArrayView2<F>,
        y: ArrayView1<F>,
        splits: &[CvSplit],
    ) -> Result<FittedCrossConformalRegressor<E::Fitted, F>>
    where
        E: Fit<F>,
        E::Fitted: Predict<F>,
    {
        check_xy(x, y)?;
        let n = x.nrows();
        if splits.len() < 2 {
            return Err(RustMlError::InvalidParameter(
                "cross-conformal regression needs at least 2 splits".into(),
            ));
        }
        let mut fold_of = vec![usize::MAX; n];
        for (k, (_, test)) in splits.iter().enumerate() {
            for &i in test {
                if i >= n || fold_of[i] != usize::MAX {
                    return Err(RustMlError::InvalidParameter(
                        "the test sets of the splits must partition the rows".into(),
                    ));
                }
                fold_of[i] = k;
            }
        }
        if fold_of.contains(&usize::MAX) {
            return Err(RustMlError::InvalidParameter(
                "the test sets of the splits must partition the rows".into(),
            ));
        }

        let mut residuals = vec![F::zero(); n];
        let mut folds = Vec::with_capacity(splits.len());
        for (train, test) in splits {
            let model = self.estimator.fit_view(
                x.select(Axis(0), train).view(),
                y.select(Axis(0), train).view(),
            )?;
            let pred = model.predict_view(x.select(Axis(0), test).view())?;
            for (&i, &p) in test.iter().zip(pred.iter()) {
                residuals[i] = (y[i] - p).abs();
            }
            folds.push(model);
        }
        if residuals.iter().any(|r| r.is_nan()) {
            return Err(RustMlError::InvalidParameter(
                "NaN in the targets or out-of-fold predictions".into(),
            ));
        }
        let full = self.estimator.fit_view(x, y)?;
        Ok(FittedCrossConformalRegressor {
            full,
            folds,
            fold_of,
            residuals,
            method: self.method,
            point_prediction: self.point_prediction,
            allow_infinite_bounds: self.allow_infinite_bounds,
        })
    }
}

impl<F: Float, E: Fit<F>> Fit<F> for CrossConformalRegressor<E>
where
    E::Fitted: Predict<F>,
{
    type Fitted = FittedCrossConformalRegressor<E::Fitted, F>;

    fn fit_view(&self, x: ArrayView2<F>, y: ArrayView1<F>) -> Result<Self::Fitted> {
        let splits = shuffled_k_fold(x.nrows(), self.cv, self.seed)?;
        self.fit_with_splits(x, y, &splits)
    }
}

/// Full-data model, fold models and out-of-fold residuals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedCrossConformalRegressor<M, F> {
    full: M,
    folds: Vec<M>,
    /// Fold (index into `folds`) whose test set holds each training row.
    fold_of: Vec<usize>,
    residuals: Vec<F>,
    method: CrossMethod,
    point_prediction: PointPrediction,
    allow_infinite_bounds: bool,
}

impl<M: Predict<F>, F: Float> FittedCrossConformalRegressor<M, F> {
    /// The model fitted on all rows.
    pub fn full_model(&self) -> &M {
        &self.full
    }

    pub fn fold_models(&self) -> &[M] {
        &self.folds
    }

    /// Out-of-fold absolute residual of every training row.
    pub fn conformity_scores(&self) -> &[F] {
        &self.residuals
    }

    /// Predictions of every fold model, one column per fold.
    fn fold_predictions(&self, x: ArrayView2<F>) -> Result<Array2<F>> {
        let mut out = Array2::zeros((x.nrows(), self.folds.len()));
        for (k, m) in self.folds.iter().enumerate() {
            out.column_mut(k).assign(&m.predict_view(x)?);
        }
        Ok(out)
    }

    fn margin(&self, alpha: f64) -> Result<F> {
        let mut r = self.residuals.clone();
        match lower_quantile(&mut r, 1.0 - alpha) {
            Some(q) => Ok(q),
            None if self.allow_infinite_bounds => Ok(F::infinity()),
            None => Err(too_few(self.residuals.len(), alpha)),
        }
    }

    /// Prediction intervals `(lower, upper)` at miscoverage `alpha`.
    pub fn predict_interval(&self, x: ArrayView2<F>, alpha: f64) -> Result<(Array1<F>, Array1<F>)> {
        check_alpha(alpha)?;
        let m = x.nrows();
        match self.method {
            CrossMethod::Base => {
                let q = self.margin(alpha)?;
                let pred = self.full.predict_view(x)?;
                Ok((pred.mapv(|p| p - q), pred.mapv(|p| p + q)))
            }
            CrossMethod::MinMax => {
                let q = self.margin(alpha)?;
                let fp = self.fold_predictions(x)?;
                let lo = fp.map_axis(Axis(1), |r| r.iter().copied().fold(F::infinity(), F::min));
                let hi = fp.map_axis(Axis(1), |r| {
                    r.iter().copied().fold(F::neg_infinity(), F::max)
                });
                Ok((lo.mapv(|p| p - q), hi.mapv(|p| p + q)))
            }
            CrossMethod::Plus => {
                let fp = self.fold_predictions(x)?;
                let n = self.residuals.len();
                // Same float formulas as MAPIE, so the order statistics agree.
                let level_up = 1.0 - alpha;
                let level_low = (1.0 - 2.0 * alpha) + alpha;
                let mut lower = Array1::zeros(m);
                let mut upper = Array1::zeros(m);
                let mut buf = vec![F::zero(); n];
                for r in 0..m {
                    for i in 0..n {
                        buf[i] = fp[[r, self.fold_of[i]]] + self.residuals[i];
                    }
                    upper[r] = match lower_quantile(&mut buf, level_up) {
                        Some(v) => v,
                        None if self.allow_infinite_bounds => F::infinity(),
                        None => return Err(too_few(n, alpha)),
                    };
                    // lower = −(the level_low quantile of −(μ₋ᵢ(x) − Rᵢ))
                    for i in 0..n {
                        buf[i] = self.residuals[i] - fp[[r, self.fold_of[i]]];
                    }
                    lower[r] = match lower_quantile(&mut buf, level_low) {
                        Some(v) => -v,
                        None if self.allow_infinite_bounds => F::neg_infinity(),
                        None => return Err(too_few(n, alpha)),
                    };
                }
                Ok((lower, upper))
            }
        }
    }
}

impl<M: Predict<F>, F: Float> Predict<F> for FittedCrossConformalRegressor<M, F> {
    fn predict_view(&self, x: ArrayView2<F>) -> Result<Array1<F>> {
        let point = match self.method {
            CrossMethod::Base => PointPrediction::Full,
            _ => self.point_prediction,
        };
        match point {
            PointPrediction::Full => self.full.predict_view(x),
            PointPrediction::Mean => {
                let fp = self.fold_predictions(x)?;
                let mut counts = vec![0usize; self.folds.len()];
                for &k in &self.fold_of {
                    counts[k] += 1;
                }
                let n = F::from_usize(self.fold_of.len()).expect("fits in F");
                Ok(fp.map_axis(Axis(1), |r| {
                    r.iter().zip(&counts).fold(F::zero(), |acc, (&p, &c)| {
                        acc + p * F::from_usize(c).expect("fits")
                    }) / n
                }))
            }
            PointPrediction::Median => {
                let fp = self.fold_predictions(x)?;
                let n = self.fold_of.len();
                let mut buf = vec![F::zero(); n];
                Ok(fp.map_axis(Axis(1), |r| {
                    for (b, &k) in buf.iter_mut().zip(&self.fold_of) {
                        *b = r[k];
                    }
                    buf.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
                    let two = F::one() + F::one();
                    if n % 2 == 1 {
                        buf[n / 2]
                    } else {
                        (buf[n / 2 - 1] + buf[n / 2]) / two
                    }
                }))
            }
        }
    }
}
