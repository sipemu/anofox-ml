//! Multivariate imputation by chained regressions.
//!
//! Mirrors `sklearn.impute.IterativeImputer` (sklearn 1.8): after an initial
//! [`SimpleImputer`] pass, each feature with missing values is regressed on
//! the other features and its missing entries are replaced by the
//! predictions, feature after feature, for up to `max_iter` rounds. Fitting
//! stops early once the largest absolute row sum of the change between rounds
//! drops below `tol · max|X_observed|`. Every fitted regressor is kept, and
//! `transform` replays the whole sequence on new data.
//!
//! The default regressor is [`BayesianRidge`], as in sklearn. Any
//! [`ImputationEstimator`] can be used instead; [`AnyRegressor`] adapts a
//! regressor implementing [`Fit`] + [`Predict`]. With `sample_posterior` the
//! imputations are draws from the (truncated) Gaussian predictive
//! distribution, which needs a model that reports a predictive standard
//! deviation (BayesianRidge does). The random draws, the `Random` imputation
//! order and `n_nearest_features` use this crate's seeded generator, so they
//! do not reproduce sklearn's numbers.

use anofox_ml_core::{Fit, FitUnsupervised, Predict, Result, RustMlError, Transform};
use anofox_ml_regression::{BayesianRidge, FittedBayesianRidge};
use ndarray::{Array1, Array2, Axis};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::missing_indicator::{FittedMissingIndicator, MissingIndicator};
use crate::simple_imputer::{FittedSimpleImputer, ImputeStrategy, SimpleImputer};
use crate::special::{norm_cdf, norm_ppf};

/// A regressor that `IterativeImputer` fits once per feature and round.
pub trait ImputationEstimator: Send + Sync {
    type Model: ImputationModel;
    fn fit_model(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Model>;
}

/// A fitted per-feature regressor.
pub trait ImputationModel: Send + Sync {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>>;

    /// Predictive standard deviation, needed for `sample_posterior`. `None`
    /// means the model has none.
    fn predict_std(&self, _x: &Array2<f64>) -> Option<Result<Array1<f64>>> {
        None
    }
}

impl ImputationEstimator for BayesianRidge {
    type Model = FittedBayesianRidge;

    fn fit_model(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Model> {
        self.fit(x, y)
    }
}

impl ImputationModel for FittedBayesianRidge {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        Predict::predict(self, x)
    }

    fn predict_std(&self, x: &Array2<f64>) -> Option<Result<Array1<f64>>> {
        Some(FittedBayesianRidge::predict_std(self, x))
    }
}

/// Adapts any regressor implementing [`Fit`] whose fitted model implements
/// [`Predict`] (no predictive standard deviation).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AnyRegressor<E>(pub E);

/// Fitted model of an [`AnyRegressor`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AnyModel<M>(pub M);

impl<E> ImputationEstimator for AnyRegressor<E>
where
    E: Fit<f64> + Send + Sync,
    E::Fitted: Predict<f64> + Send + Sync,
{
    type Model = AnyModel<E::Fitted>;

    fn fit_model(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Model> {
        Ok(AnyModel(self.0.fit(x, y)?))
    }
}

impl<M: Predict<f64> + Send + Sync> ImputationModel for AnyModel<M> {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        self.0.predict(x)
    }
}

/// Order in which the features are imputed in each round.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum ImputationOrder {
    /// Fewest missing values first.
    #[default]
    Ascending,
    /// Most missing values first.
    Descending,
    /// Left to right.
    Roman,
    /// Right to left.
    Arabic,
    /// A new random order every round.
    Random,
}

/// Parameters for `IterativeImputer` (unfitted state).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IterativeImputer<E = BayesianRidge> {
    /// Regressor fitted for each feature in each round.
    pub estimator: E,
    /// Draw imputations from the predictive distribution instead of using
    /// the predicted mean.
    pub sample_posterior: bool,
    /// Maximum number of rounds.
    pub max_iter: usize,
    /// Early-stopping tolerance (unused with `sample_posterior`).
    pub tol: f64,
    /// Regress each feature on this many other features, drawn with
    /// probability proportional to their absolute correlation.
    pub n_nearest_features: Option<usize>,
    /// Strategy of the initial imputation (`Constant` uses `fill_value`).
    pub initial_strategy: ImputeStrategy,
    pub fill_value: Option<f64>,
    pub imputation_order: ImputationOrder,
    /// Only impute (and fit regressors for) features with missing values at
    /// fit time.
    pub skip_complete: bool,
    /// Lower bound of the imputed values: empty (none), one value for all
    /// features, or one per feature.
    pub min_value: Vec<f64>,
    /// Upper bound, like `min_value`.
    pub max_value: Vec<f64>,
    pub random_state: u64,
    pub add_indicator: bool,
    pub keep_empty_features: bool,
}

impl Default for IterativeImputer<BayesianRidge> {
    fn default() -> Self {
        Self::with_estimator(BayesianRidge::new())
    }
}

impl IterativeImputer<BayesianRidge> {
    /// IterativeImputer with sklearn's defaults (BayesianRidge, 10 rounds).
    pub fn new() -> Self {
        Self::default()
    }
}

impl<E> IterativeImputer<E> {
    /// Use `estimator` as the per-feature regressor.
    pub fn with_estimator(estimator: E) -> Self {
        Self {
            estimator,
            sample_posterior: false,
            max_iter: 10,
            tol: 1e-3,
            n_nearest_features: None,
            initial_strategy: ImputeStrategy::Mean,
            fill_value: None,
            imputation_order: ImputationOrder::Ascending,
            skip_complete: false,
            min_value: Vec::new(),
            max_value: Vec::new(),
            random_state: 0,
            add_indicator: false,
            keep_empty_features: false,
        }
    }

    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    pub fn with_sample_posterior(mut self, sample_posterior: bool) -> Self {
        self.sample_posterior = sample_posterior;
        self
    }

    pub fn with_n_nearest_features(mut self, n: usize) -> Self {
        self.n_nearest_features = Some(n);
        self
    }

    pub fn with_initial_strategy(mut self, strategy: ImputeStrategy) -> Self {
        self.initial_strategy = strategy;
        self
    }

    pub fn with_fill_value(mut self, value: f64) -> Self {
        self.fill_value = Some(value);
        self
    }

    pub fn with_imputation_order(mut self, order: ImputationOrder) -> Self {
        self.imputation_order = order;
        self
    }

    pub fn with_skip_complete(mut self, skip: bool) -> Self {
        self.skip_complete = skip;
        self
    }

    /// Bounds of the imputed values (see the field docs for the shapes).
    pub fn with_bounds(mut self, min_value: Vec<f64>, max_value: Vec<f64>) -> Self {
        self.min_value = min_value;
        self.max_value = max_value;
        self
    }

    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = seed;
        self
    }

    pub fn with_add_indicator(mut self, add_indicator: bool) -> Self {
        self.add_indicator = add_indicator;
        self
    }

    pub fn with_keep_empty_features(mut self, keep: bool) -> Self {
        self.keep_empty_features = keep;
        self
    }
}

/// One fitted step: impute `feature` from `neighbors` with `model`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ImputationStep<M> {
    pub feature: usize,
    pub neighbors: Vec<usize>,
    pub model: M,
}

/// Fitted `IterativeImputer`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedIterativeImputer<M> {
    initial: FittedSimpleImputer<f64>,
    /// Feature was entirely missing at fit time.
    is_empty: Vec<bool>,
    keep_empty_features: bool,
    sequence: Vec<ImputationStep<M>>,
    n_iter: usize,
    min_value: Vec<f64>,
    max_value: Vec<f64>,
    sample_posterior: bool,
    random_state: u64,
    indicator: Option<FittedMissingIndicator>,
}

/// Per-feature bound vector (sklearn's `_validate_limit`), restricted to the
/// kept features.
fn validate_limit(
    limit: &[f64],
    default: f64,
    is_empty: &[bool],
    keep_empty: bool,
    what: &str,
) -> Result<Vec<f64>> {
    let d = is_empty.len();
    let full = match limit.len() {
        0 => vec![default; d],
        1 => vec![limit[0]; d],
        n if n == d => limit.to_vec(),
        n => {
            return Err(RustMlError::ShapeMismatch(format!(
                "{what} should have 1 or {d} entries, got {n}"
            )))
        }
    };
    Ok(full
        .into_iter()
        .zip(is_empty)
        .filter(|(_, &e)| keep_empty || !e)
        .map(|(v, _)| v)
        .collect())
}

/// The observed data (kept features, empty kept features filled), the
/// initially imputed working matrix and the missing mask.
fn initial_imputation(
    x: &Array2<f64>,
    initial: &FittedSimpleImputer<f64>,
    is_empty: &[bool],
    keep_empty: bool,
) -> Result<(Array2<f64>, Array2<f64>, Array2<bool>)> {
    let filled = initial.transform(x)?;
    let kept: Vec<usize> = (0..x.ncols())
        .filter(|&j| keep_empty || !is_empty[j])
        .collect();
    let mut obs = x.select(Axis(1), &kept);
    for (c, &j) in kept.iter().enumerate() {
        if is_empty[j] {
            obs.column_mut(c).assign(&filled.column(c));
        }
    }
    let mask = obs.mapv(f64::is_nan);
    Ok((obs, filled, mask))
}

fn rows_where(mask: &Array2<bool>, j: usize, missing: bool) -> Vec<usize> {
    (0..mask.nrows())
        .filter(|&i| mask[[i, j]] == missing)
        .collect()
}

/// Draw from `N(mu, sigma²)` truncated to `[lo, hi]` by inverting the CDF
/// (in the lower tail, for accuracy).
fn truncnorm_draw(rng: &mut StdRng, mu: f64, sigma: f64, lo: f64, hi: f64) -> f64 {
    let a = (lo - mu) / sigma;
    let b = (hi - mu) / sigma;
    let u: f64 = rng.gen();
    let z = if a > 0.0 {
        // Both bounds in the upper tail: sample the mirrored lower tail.
        let (pa, pb) = (norm_cdf(-b), norm_cdf(-a));
        -norm_ppf(pa + u * (pb - pa))
    } else {
        let (pa, pb) = (norm_cdf(a), norm_cdf(b));
        norm_ppf(pa + u * (pb - pa))
    };
    (mu + sigma * z).clamp(lo, hi)
}

/// Impute feature `j` of `xt` in place with `model` (sklearn's
/// `_impute_one_feature` after the fit).
#[allow(clippy::too_many_arguments)]
fn impute_feature<M: ImputationModel>(
    xt: &mut Array2<f64>,
    mask: &Array2<bool>,
    j: usize,
    neighbors: &[usize],
    model: &M,
    bounds: (f64, f64),
    sample_posterior: bool,
    rng: &mut StdRng,
) -> Result<()> {
    let rows = rows_where(mask, j, true);
    if rows.is_empty() {
        return Ok(());
    }
    let xq = xt.select(Axis(0), &rows).select(Axis(1), neighbors);
    let (lo, hi) = bounds;
    let values: Vec<f64> = if sample_posterior {
        let mus = model.predict(&xq)?;
        let sigmas = model.predict_std(&xq).ok_or_else(|| {
            RustMlError::InvalidParameter(
                "sample_posterior needs an estimator with a predictive standard deviation".into(),
            )
        })??;
        mus.iter()
            .zip(sigmas.iter())
            .map(|(&mu, &s)| {
                if mu < lo {
                    lo
                } else if mu > hi {
                    hi
                } else if s > 0.0 {
                    truncnorm_draw(rng, mu, s, lo, hi)
                } else {
                    mu
                }
            })
            .collect()
    } else {
        model
            .predict(&xq)?
            .iter()
            .map(|v| v.clamp(lo, hi))
            .collect()
    };
    for (&i, v) in rows.iter().zip(values) {
        xt[[i, j]] = v;
    }
    Ok(())
}

impl<E: ImputationEstimator> IterativeImputer<E> {
    /// Features to impute in a round (sklearn's `_get_ordered_idx`).
    fn ordered_features(&self, mask: &Array2<bool>, rng: &mut StdRng) -> Vec<usize> {
        let n = mask.nrows() as f64;
        let frac: Vec<f64> = (0..mask.ncols())
            .map(|j| mask.column(j).iter().filter(|&&m| m).count() as f64 / n)
            .collect();
        let candidates: Vec<usize> = (0..frac.len())
            .filter(|&j| !self.skip_complete || frac[j] > 0.0)
            .collect();
        // Stable sort by missing fraction; skipped (complete) features sort
        // first, so dropping the first `n_skipped` keeps the candidates.
        let ascending = || {
            let mut idx: Vec<usize> = (0..frac.len()).collect();
            idx.sort_by(|&a, &b| frac[a].total_cmp(&frac[b]));
            idx.split_off(frac.len() - candidates.len())
        };
        match self.imputation_order {
            ImputationOrder::Roman => candidates,
            ImputationOrder::Arabic => candidates.into_iter().rev().collect(),
            ImputationOrder::Ascending => ascending(),
            ImputationOrder::Descending => ascending().into_iter().rev().collect(),
            ImputationOrder::Random => {
                let mut c = candidates;
                c.shuffle(rng);
                c
            }
        }
    }

    /// Absolute correlation matrix with zero diagonal, columns normalised to
    /// sum 1 (sklearn's `_get_abs_corr_mat`); `None` without
    /// `n_nearest_features`.
    fn abs_corr(&self, xt: &Array2<f64>) -> Option<Array2<f64>> {
        let d = xt.ncols();
        let k = self.n_nearest_features?;
        if k >= d {
            return None;
        }
        let tolerance = 1e-6;
        let mean = xt.mean_axis(Axis(0)).expect("rows > 0");
        let c = xt - &mean;
        let cov = c.t().dot(&c);
        let mut m = Array2::<f64>::zeros((d, d));
        for a in 0..d {
            for b in 0..d {
                let r = (cov[[a, b]] / (cov[[a, a]] * cov[[b, b]]).sqrt()).abs();
                m[[a, b]] = if r.is_nan() {
                    tolerance
                } else {
                    r.max(tolerance)
                };
            }
            m[[a, a]] = 0.0;
        }
        for b in 0..d {
            let s = m.column(b).sum();
            if s > 0.0 {
                m.column_mut(b).mapv_inplace(|v| v / s);
            }
        }
        Some(m)
    }

    /// Regressors of feature `j` (sklearn's `_get_neighbor_feat_idx`).
    fn neighbors(
        &self,
        d: usize,
        j: usize,
        corr: Option<&Array2<f64>>,
        rng: &mut StdRng,
    ) -> Vec<usize> {
        match (self.n_nearest_features, corr) {
            (Some(k), Some(p)) => {
                // Weighted sampling without replacement.
                let mut w: Vec<f64> = p.column(j).to_vec();
                let mut chosen = Vec::with_capacity(k);
                for _ in 0..k {
                    let total: f64 = w.iter().sum();
                    let mut u = rng.gen::<f64>() * total;
                    let mut pick = w.iter().rposition(|&v| v > 0.0).unwrap_or(0);
                    for (i, &v) in w.iter().enumerate() {
                        if v > 0.0 && u < v {
                            pick = i;
                            break;
                        }
                        u -= v;
                    }
                    chosen.push(pick);
                    w[pick] = 0.0;
                }
                chosen
            }
            _ => (0..d).filter(|&i| i != j).collect(),
        }
    }
}

impl<E: ImputationEstimator> FitUnsupervised<f64> for IterativeImputer<E> {
    type Fitted = FittedIterativeImputer<E::Model>;

    fn fit(&self, x: &Array2<f64>) -> Result<Self::Fitted> {
        Ok(self.fit_transform(x)?.0)
    }

    fn fit_with_target(
        &self,
        x: &Array2<f64>,
        _y: Option<&Array1<f64>>,
    ) -> Result<(Self::Fitted, Option<Array2<f64>>)> {
        let (fitted, out) = self.fit_transform(x)?;
        Ok((fitted, Some(out)))
    }
}

impl<E: ImputationEstimator> IterativeImputer<E> {
    /// Fit and return the imputed training data (sklearn's `fit_transform`).
    pub fn fit_transform(
        &self,
        x: &Array2<f64>,
    ) -> Result<(FittedIterativeImputer<E::Model>, Array2<f64>)> {
        if x.is_empty() {
            return Err(RustMlError::EmptyInput("input array is empty".into()));
        }
        if x.iter().any(|v| v.is_infinite()) {
            return Err(RustMlError::InvalidParameter(
                "IterativeImputer input contains infinite values".into(),
            ));
        }
        let mut rng = StdRng::seed_from_u64(self.random_state);
        let is_empty: Vec<bool> = (0..x.ncols())
            .map(|j| x.column(j).iter().all(|v| v.is_nan()))
            .collect();
        let mut simple = SimpleImputer::<f64>::new()
            .with_strategy(self.initial_strategy)
            .with_keep_empty_features(self.keep_empty_features);
        if let Some(v) = self.fill_value {
            simple = simple.with_fill_value(v);
        }
        let initial = FitUnsupervised::<f64>::fit(&simple, x)?;
        let (obs, mut xt, mask) =
            initial_imputation(x, &initial, &is_empty, self.keep_empty_features)?;
        let indicator = if self.add_indicator {
            let mi = MissingIndicator::new().with_error_on_new(false);
            Some(FitUnsupervised::<f64>::fit(&mi, x)?)
        } else {
            None
        };
        let min_value = validate_limit(
            &self.min_value,
            f64::NEG_INFINITY,
            &is_empty,
            self.keep_empty_features,
            "min_value",
        )?;
        let max_value = validate_limit(
            &self.max_value,
            f64::INFINITY,
            &is_empty,
            self.keep_empty_features,
            "max_value",
        )?;
        let mut fitted = FittedIterativeImputer {
            initial,
            is_empty,
            keep_empty_features: self.keep_empty_features,
            sequence: Vec::new(),
            n_iter: 0,
            min_value,
            max_value,
            sample_posterior: self.sample_posterior,
            random_state: self.random_state,
            indicator,
        };
        let d = xt.ncols();
        if self.max_iter == 0 || mask.iter().all(|&m| m) || d == 1 {
            let out = fitted.finish(xt, &obs, &mask, x)?;
            return Ok((fitted, out));
        }
        if fitted
            .min_value
            .iter()
            .zip(&fitted.max_value)
            .any(|(lo, hi)| lo >= hi)
        {
            return Err(RustMlError::InvalidParameter(
                "one or more features have min_value >= max_value".into(),
            ));
        }

        let mut order = self.ordered_features(&mask, &mut rng);
        let corr = self.abs_corr(&xt);
        let normalized_tol = self.tol
            * obs
                .iter()
                .zip(mask.iter())
                .filter(|(_, &m)| !m)
                .fold(0.0f64, |acc, (&v, _)| acc.max(v.abs()));
        let mut previous = xt.clone();
        for round in 1..=self.max_iter {
            fitted.n_iter = round;
            if self.imputation_order == ImputationOrder::Random {
                order = self.ordered_features(&mask, &mut rng);
            }
            for &j in &order {
                let neighbors = self.neighbors(d, j, corr.as_ref(), &mut rng);
                let train = rows_where(&mask, j, false);
                let xtr = xt.select(Axis(0), &train).select(Axis(1), &neighbors);
                let ytr = train.iter().map(|&i| xt[[i, j]]).collect::<Array1<f64>>();
                let model = self.estimator.fit_model(&xtr, &ytr)?;
                impute_feature(
                    &mut xt,
                    &mask,
                    j,
                    &neighbors,
                    &model,
                    (fitted.min_value[j], fitted.max_value[j]),
                    self.sample_posterior,
                    &mut rng,
                )?;
                fitted.sequence.push(ImputationStep {
                    feature: j,
                    neighbors,
                    model,
                });
            }
            if !self.sample_posterior {
                // np.linalg.norm(ord=inf) of a matrix: largest absolute row sum.
                let change = (&xt - &previous)
                    .rows()
                    .into_iter()
                    .map(|r| r.iter().map(|v| v.abs()).sum::<f64>())
                    .fold(0.0, f64::max);
                if change < normalized_tol {
                    break;
                }
                previous.assign(&xt);
            }
        }
        let out = fitted.finish(xt, &obs, &mask, x)?;
        Ok((fitted, out))
    }
}

impl<M: ImputationModel> FittedIterativeImputer<M> {
    /// Restore the observed values and append the indicator.
    fn finish(
        &self,
        mut xt: Array2<f64>,
        obs: &Array2<f64>,
        mask: &Array2<bool>,
        x: &Array2<f64>,
    ) -> Result<Array2<f64>> {
        ndarray::Zip::from(&mut xt)
            .and(obs)
            .and(mask)
            .for_each(|t, &o, &m| {
                if !m {
                    *t = o;
                }
            });
        match &self.indicator {
            Some(ind) => {
                let m = ind.transform(x)?;
                ndarray::concatenate(Axis(1), &[xt.view(), m.view()])
                    .map_err(|e| RustMlError::ShapeMismatch(e.to_string()))
            }
            None => Ok(xt),
        }
    }

    /// The fitted steps in order (`imputation_sequence_`).
    pub fn imputation_sequence(&self) -> &[ImputationStep<M>] {
        &self.sequence
    }

    /// Number of rounds run (`n_iter_`).
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }

    /// The initial imputer (`initial_imputer_`).
    pub fn initial_imputer(&self) -> &FittedSimpleImputer<f64> {
        &self.initial
    }
}

impl<M: ImputationModel> Transform<f64> for FittedIterativeImputer<M> {
    /// Replay the fitted sequence on new data. With `sample_posterior` the
    /// draws restart from `random_state` on every call.
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let (obs, mut xt, mask) =
            initial_imputation(x, &self.initial, &self.is_empty, self.keep_empty_features)?;
        if self.n_iter > 0 && !mask.iter().all(|&m| m) {
            let mut rng = StdRng::seed_from_u64(self.random_state);
            for step in &self.sequence {
                impute_feature(
                    &mut xt,
                    &mask,
                    step.feature,
                    &step.neighbors,
                    &step.model,
                    (self.min_value[step.feature], self.max_value[step.feature]),
                    self.sample_posterior,
                    &mut rng,
                )?;
            }
        }
        self.finish(xt, &obs, &mask, x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    fn data() -> Array2<f64> {
        let n = f64::NAN;
        array![
            [1.0, 2.0, n, 4.0],
            [3.0, 4.0, 3.0, n],
            [n, 6.0, 5.0, 8.0],
            [8.0, 8.0, 7.0, 1.0],
            [2.5, n, 1.0, 0.5],
            [6.0, 1.5, 2.0, 3.5],
            [0.0, 9.0, n, n],
            [4.0, 3.0, 6.5, 2.0],
            [5.0, 7.0, 4.0, 6.0],
            [n, 2.5, 3.5, 2.5]
        ]
    }

    fn assert_close(a: &Array2<f64>, b: &[&[f64]], tol: f64) {
        assert_eq!(a.nrows(), b.len());
        for (i, row) in b.iter().enumerate() {
            assert_eq!(a.ncols(), row.len());
            for (j, v) in row.iter().enumerate() {
                assert_abs_diff_eq!(a[[i, j]], *v, epsilon = tol);
            }
        }
    }

    #[test]
    fn test_default_matches_sklearn() {
        // sklearn 1.8: IterativeImputer().fit(X) (BayesianRidge, stops after
        // 2 rounds), transform(X) and transform(Q).
        let x = data();
        let f = FitUnsupervised::<f64>::fit(&IterativeImputer::new(), &x).unwrap();
        assert_eq!(f.n_iter(), 2);
        assert_eq!(f.imputation_sequence().len(), 8);
        assert_close(
            &f.transform(&x).unwrap(),
            &[
                &[1.0, 2.0, 3.855_414_974_923_31, 4.0],
                &[3.0, 4.0, 3.0, 3.437_581_015_937_479],
                &[3.686_699_736_915_880_7, 6.0, 5.0, 8.0],
                &[8.0, 8.0, 7.0, 1.0],
                &[2.5, 4.772_804_492_367_64, 1.0, 0.5],
                &[6.0, 1.5, 2.0, 3.5],
                &[0.0, 9.0, 4.007_944_398_562_672, 3.440_248_877_386_484_3],
                &[4.0, 3.0, 6.5, 2.0],
                &[5.0, 7.0, 4.0, 6.0],
                &[3.687_101_043_339_392_7, 2.5, 3.5, 2.5],
            ],
            1e-8,
        );
        let q = array![
            [f64::NAN, 5.0, f64::NAN, 2.0],
            [1.0, f64::NAN, f64::NAN, f64::NAN]
        ];
        assert_close(
            &f.transform(&q).unwrap(),
            &[
                &[3.687_943_770_102_885_7, 5.0, 3.985_384_432_006_104, 2.0],
                &[
                    1.0,
                    4.777_014_403_424_943_5,
                    3.923_616_235_718_252_5,
                    3.438_545_577_713_815_4,
                ],
            ],
            1e-8,
        );
        // fit_transform gives the same training output.
        let (_, out) = IterativeImputer::new().fit_transform(&x).unwrap();
        assert_eq!(out, f.transform(&x).unwrap());
    }

    #[test]
    fn test_ols_bounds_descending_matches_sklearn() {
        // sklearn 1.8: IterativeImputer(LinearRegression(), max_iter=3,
        //   imputation_order='descending', min_value=-1, max_value=7.5)
        let x = data();
        let imp = IterativeImputer::with_estimator(AnyRegressor(
            anofox_ml_regression::OlsRegressor::new(),
        ))
        .with_max_iter(3)
        .with_imputation_order(ImputationOrder::Descending)
        .with_bounds(vec![-1.0], vec![7.5]);
        let f = FitUnsupervised::<f64>::fit(&imp, &x).unwrap();
        assert_eq!(f.n_iter(), 3);
        assert_close(
            &f.transform(&x).unwrap(),
            &[
                &[1.0, 2.0, 3.329_351_237_352_406, 4.0],
                &[3.0, 4.0, 3.0, 5.013_316_992_617_280_5],
                &[0.188_299_082_034_936_84, 6.0, 5.0, 8.0],
                &[8.0, 8.0, 7.0, 1.0],
                &[2.5, -1.0, 1.0, 0.5],
                &[6.0, 1.5, 2.0, 3.5],
                &[0.0, 9.0, 7.363_073_694_201_313, 7.5],
                &[4.0, 3.0, 6.5, 2.0],
                &[5.0, 7.0, 4.0, 6.0],
                &[3.996_832_857_870_954_7, 2.5, 3.5, 2.5],
            ],
            1e-9,
        );
    }

    #[test]
    fn test_empty_feature_indicator_and_keep() {
        // sklearn: an all-NaN third column is dropped (indicator kept), or
        // kept as zeros with keep_empty_features=True.
        let x = data();
        let mut cols: Vec<Array1<f64>> = (0..4).map(|j| x.column(j).to_owned()).collect();
        cols.insert(2, Array1::from_elem(10, f64::NAN));
        let views: Vec<_> = cols.iter().map(|c| c.view().insert_axis(Axis(1))).collect();
        let x2 = ndarray::concatenate(Axis(1), &views).unwrap();

        let f = FitUnsupervised::<f64>::fit(
            &IterativeImputer::new()
                .with_max_iter(2)
                .with_add_indicator(true),
            &x2,
        )
        .unwrap();
        let t = f.transform(&x2).unwrap();
        assert_eq!(t.dim(), (10, 9));
        assert_close(
            &t.slice(ndarray::s![..3, ..]).to_owned(),
            &[
                &[1.0, 2.0, 3.855_414_974_923_31, 4.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                &[
                    3.0,
                    4.0,
                    3.0,
                    3.437_581_015_937_479,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                ],
                &[
                    3.686_699_736_915_880_7,
                    6.0,
                    5.0,
                    8.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                ],
            ],
            1e-8,
        );
        let f = FitUnsupervised::<f64>::fit(
            &IterativeImputer::new()
                .with_max_iter(2)
                .with_keep_empty_features(true),
            &x2,
        )
        .unwrap();
        assert_close(
            &f.transform(&x2)
                .unwrap()
                .slice(ndarray::s![..3, ..])
                .to_owned(),
            &[
                &[1.0, 2.0, 0.0, 3.855_414_974_923_309_3, 4.0],
                &[3.0, 4.0, 0.0, 3.0, 3.437_581_015_937_479],
                &[3.686_699_736_915_880_7, 6.0, 0.0, 5.0, 8.0],
            ],
            1e-8,
        );
    }

    #[test]
    fn test_skip_complete_and_sample_posterior() {
        let x = data();
        // With skip_complete a complete column gets no regressor.
        let mut xc = x.clone();
        xc.column_mut(1)
            .assign(&array![2.0, 4.0, 6.0, 8.0, 5.0, 1.5, 9.0, 3.0, 7.0, 2.5]);
        let f = FitUnsupervised::<f64>::fit(
            &IterativeImputer::new()
                .with_max_iter(1)
                .with_skip_complete(true),
            &xc,
        )
        .unwrap();
        assert!(f.imputation_sequence().iter().all(|s| s.feature != 1));
        assert_eq!(f.imputation_sequence().len(), 3);

        // Posterior draws stay within the bounds, are reproducible and keep
        // the observed values.
        let imp = IterativeImputer::new()
            .with_sample_posterior(true)
            .with_bounds(vec![0.0], vec![9.0])
            .with_random_state(3);
        let (f, a) = imp.fit_transform(&x).unwrap();
        assert_eq!(f.n_iter(), 10);
        let (_, b) = imp.fit_transform(&x).unwrap();
        assert_eq!(a, b);
        for (&v, &o) in a.iter().zip(x.iter()) {
            if o.is_nan() {
                assert!((0.0..=9.0).contains(&v));
            } else {
                assert_eq!(v, o);
            }
        }
        // sample_posterior needs a predictive standard deviation.
        let ols = IterativeImputer::with_estimator(AnyRegressor(
            anofox_ml_regression::OlsRegressor::new(),
        ))
        .with_sample_posterior(true);
        assert!(FitUnsupervised::<f64>::fit(&ols, &x).is_err());
    }

    #[test]
    fn test_truncnorm_draw_in_bounds() {
        let mut rng = StdRng::seed_from_u64(0);
        for &(mu, s, lo, hi) in &[
            (0.0, 1.0, -0.5, 0.5),
            (0.0, 1.0, 6.0, 7.0),
            (5.0, 2.0, -1.0, 0.0),
        ] {
            for _ in 0..200 {
                let v = truncnorm_draw(&mut rng, mu, s, lo, hi);
                assert!((lo..=hi).contains(&v), "{v} not in [{lo}, {hi}]");
            }
        }
    }
}
