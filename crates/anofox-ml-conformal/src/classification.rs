//! Conformal prediction sets for classifiers: LAC and APS.

use anofox_ml_core::{Fit, Float, Predict, PredictProba, Result, RustMlError};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::mondrian::check_groups;
use crate::quantile::{check_alpha, lower_quantile};
use crate::split::{check_xy, split_indices, too_few};

/// Tolerance of the set-membership comparisons (MAPIE's `EPSILON`).
const EPS: f64 = 1e-8;

/// How a class's conformity is scored.
#[derive(Debug, Clone, Serialize, Deserialize, Copy, PartialEq, Default)]
pub enum ClassificationScore {
    /// Least ambiguous set-valued classifier (Sadinle et al. 2019): score
    /// `1 − p̂_y(x)`; a set holds every class with `p̂ ≥ 1 − q`. Smallest
    /// sets on average; they may be empty and adapt little to difficulty.
    #[default]
    Lac,
    /// Adaptive prediction sets (Romano, Sesia & Candès 2020): score = the
    /// probability mass of the classes ranked up to and including the true
    /// one; a set holds the most probable classes until their mass reaches
    /// `q`. Sets grow on hard inputs.
    Aps(ApsMode),
    /// Regularized APS (Angelopoulos et al. 2021): APS plus a penalty
    /// `λ · max(0, rank − k_reg)` on every class ranked after `k_reg`, which
    /// keeps sets small when the model puts mass on many unlikely classes.
    Raps(RapsParams),
}

/// Settings of [`ClassificationScore::Raps`].
#[derive(Debug, Clone, Serialize, Deserialize, Copy, PartialEq)]
pub struct RapsParams {
    pub mode: ApsMode,
    /// Rank after which classes are penalised; `None`: tuned per `alpha` as
    /// the conformal quantile of the true class's rank on the tuning rows,
    /// plus one.
    pub k_reg: Option<usize>,
    /// Penalty weight; `None`: tuned per `alpha` on the tuning rows as the
    /// value of `[0.001, 0.01, 0.1, 0.2, 0.5]` giving the smallest sets.
    pub lambda: Option<f64>,
    /// Fraction of the calibration rows held out for tuning `k_reg` /
    /// `lambda` (only when one of them is `None`).
    pub tuning_size: f64,
}

impl RapsParams {
    pub fn new(mode: ApsMode) -> Self {
        Self {
            mode,
            k_reg: None,
            lambda: None,
            tuning_size: 0.2,
        }
    }

    pub fn with_k_reg(mut self, k_reg: usize) -> Self {
        self.k_reg = Some(k_reg);
        self
    }

    pub fn with_lambda(mut self, lambda: f64) -> Self {
        self.lambda = Some(lambda);
        self
    }

    pub fn with_tuning_size(mut self, tuning_size: f64) -> Self {
        self.tuning_size = tuning_size;
        self
    }

    fn needs_tuning(&self) -> bool {
        self.k_reg.is_none() || self.lambda.is_none()
    }
}

/// λ values searched when [`RapsParams::lambda`] is `None` (from the paper).
const LAMBDA_GRID: [f64; 5] = [0.001, 0.01, 0.1, 0.2, 0.5];

/// Treatment of the class that crosses the APS threshold (MAPIE's
/// `include_last_label`).
#[derive(Debug, Clone, Serialize, Deserialize, Copy, PartialEq, Eq, Default)]
pub enum ApsMode {
    /// Always include it: coverage `≥ 1 − α`, conservative.
    #[default]
    Include,
    /// Never include it: smaller sets, coverage can fall below `1 − α`.
    Exclude,
    /// Randomised scores and a randomised decision for the crossing class:
    /// coverage `1 − α` on average, not just at least (seeded).
    Randomized,
}

/// Split conformal classifier: fits `estimator` on part of the rows and
/// calibrates its `predict_proba` on the rest, giving prediction sets that
/// contain the true class with probability `≥ 1 − α`. Equivalent to
/// MAPIE's `SplitConformalClassifier` with the `lac` / `aps` scores.
///
/// Probability columns are taken to follow the sorted distinct training
/// labels (the convention of anofox-ml's classifiers). To calibrate a model
/// you fitted yourself, use [`conformalize_classifier`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitConformalClassifier<E> {
    pub estimator: E,
    pub score: ClassificationScore,
    /// Fraction of the rows held out for calibration, in (0, 1).
    pub calibration_size: f64,
    /// Seeds the split and, for [`ApsMode::Randomized`], the randomisation.
    pub seed: u64,
    /// Return the set of all classes instead of an error when there are too
    /// few calibration rows for the requested `alpha`.
    pub allow_full_sets: bool,
}

impl<E> SplitConformalClassifier<E> {
    pub fn new(estimator: E) -> Self {
        Self {
            estimator,
            score: ClassificationScore::Lac,
            calibration_size: 0.25,
            seed: 0,
            allow_full_sets: false,
        }
    }

    pub fn with_score(mut self, score: ClassificationScore) -> Self {
        self.score = score;
        self
    }

    pub fn with_calibration_size(mut self, calibration_size: f64) -> Self {
        self.calibration_size = calibration_size;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_allow_full_sets(mut self, allow: bool) -> Self {
        self.allow_full_sets = allow;
        self
    }
}

impl<E> SplitConformalClassifier<E> {
    /// Mondrian fit: train `estimator` on part of the rows (all groups
    /// together) and calibrate the threshold per group on the rest;
    /// `groups` holds one group id per row. Predict with
    /// [`FittedSplitConformalClassifier::predict_set_mondrian`].
    pub fn fit_with_groups<F: Float>(
        &self,
        x: ArrayView2<F>,
        y: ArrayView1<F>,
        groups: &[usize],
    ) -> Result<FittedSplitConformalClassifier<E::Fitted, F>>
    where
        E: Fit<F>,
        E::Fitted: Predict<F> + PredictProba<F>,
    {
        check_xy(x, y)?;
        check_groups(groups, x.nrows())?;
        let (train, cal) = split_indices(x.nrows(), self.calibration_size, self.seed)?;
        let y_train = y.select(Axis(0), &train);
        let mut classes = y_train.to_vec();
        classes.sort_by(|a, b| a.partial_cmp(b).expect("no NaN labels"));
        classes.dedup();
        let model = self
            .estimator
            .fit_view(x.select(Axis(0), &train).view(), y_train.view())?;
        let cal_groups: Vec<usize> = cal.iter().map(|&i| groups[i]).collect();
        Ok(conformalize_classifier_mondrian(
            model,
            classes,
            x.select(Axis(0), &cal).view(),
            y.select(Axis(0), &cal).view(),
            &cal_groups,
            self.score,
            self.seed,
        )?
        .with_allow_full_sets(self.allow_full_sets))
    }
}

impl<F: Float, E: Fit<F>> Fit<F> for SplitConformalClassifier<E>
where
    E::Fitted: Predict<F> + PredictProba<F>,
{
    type Fitted = FittedSplitConformalClassifier<E::Fitted, F>;

    fn fit_view(&self, x: ArrayView2<F>, y: ArrayView1<F>) -> Result<Self::Fitted> {
        check_xy(x, y)?;
        let (train, cal) = split_indices(x.nrows(), self.calibration_size, self.seed)?;
        let y_train = y.select(Axis(0), &train);
        let mut classes = y_train.to_vec();
        classes.sort_by(|a, b| a.partial_cmp(b).expect("no NaN labels"));
        classes.dedup();
        let model = self
            .estimator
            .fit_view(x.select(Axis(0), &train).view(), y_train.view())?;
        Ok(conformalize_classifier(
            model,
            classes,
            x.select(Axis(0), &cal).view(),
            y.select(Axis(0), &cal).view(),
            self.score,
            self.seed,
        )?
        .with_allow_full_sets(self.allow_full_sets))
    }
}

/// Calibrate a fitted classifier on rows it was not trained on. `classes`
/// are the labels of its `predict_proba` columns, in order; every
/// calibration label must be one of them. `seed` drives the APS / RAPS
/// randomisation and, for RAPS with tuning, the split into calibration and
/// tuning rows (use [`conformalize_raps`] to choose that split yourself).
pub fn conformalize_classifier<F: Float, M: Predict<F> + PredictProba<F>>(
    model: M,
    classes: Vec<F>,
    x_cal: ArrayView2<F>,
    y_cal: ArrayView1<F>,
    score: ClassificationScore,
    seed: u64,
) -> Result<FittedSplitConformalClassifier<M, F>> {
    check_xy(x_cal, y_cal)?;
    if let ClassificationScore::Raps(params) = score {
        if params.needs_tuning() {
            let (cal, tune) = split_indices(x_cal.nrows(), params.tuning_size, seed ^ 0x5EED)?;
            return conformalize_raps(
                model,
                classes,
                x_cal.select(Axis(0), &cal).view(),
                y_cal.select(Axis(0), &cal).view(),
                x_cal.select(Axis(0), &tune).view(),
                y_cal.select(Axis(0), &tune).view(),
                params,
                seed,
            );
        }
    }
    calibrate(model, classes, x_cal, y_cal, score, seed, None)
}

/// [`conformalize_classifier`] with RAPS and an explicit tuning set: rows
/// `(x_tune, y_tune)` choose `k_reg` / `lambda` (where `params` leaves them
/// `None`), rows `(x_cal, y_cal)` give the conformity scores.
#[allow(clippy::too_many_arguments)]
pub fn conformalize_raps<F: Float, M: Predict<F> + PredictProba<F>>(
    model: M,
    classes: Vec<F>,
    x_cal: ArrayView2<F>,
    y_cal: ArrayView1<F>,
    x_tune: ArrayView2<F>,
    y_tune: ArrayView1<F>,
    params: RapsParams,
    seed: u64,
) -> Result<FittedSplitConformalClassifier<M, F>> {
    check_xy(x_tune, y_tune)?;
    let tune_proba = model.predict_proba_view(x_tune)?;
    let tune_labels = encode(&classes, y_tune)?;
    calibrate(
        model,
        classes,
        x_cal,
        y_cal,
        ClassificationScore::Raps(params),
        seed,
        Some((tune_proba, tune_labels)),
    )
}

/// Mondrian (group-conditional) [`conformalize_classifier`]: the threshold
/// is calibrated within each group, so sets cover `≥ 1 − α` within every
/// group. `groups` holds the group id of every calibration row; predict
/// with [`FittedSplitConformalClassifier::predict_set_mondrian`]. For RAPS,
/// `k_reg` / `lambda` are tuned on all groups together.
pub fn conformalize_classifier_mondrian<F: Float, M: Predict<F> + PredictProba<F>>(
    model: M,
    classes: Vec<F>,
    x_cal: ArrayView2<F>,
    y_cal: ArrayView1<F>,
    groups: &[usize],
    score: ClassificationScore,
    seed: u64,
) -> Result<FittedSplitConformalClassifier<M, F>> {
    check_xy(x_cal, y_cal)?;
    check_groups(groups, x_cal.nrows())?;
    let (cal, mut fitted) = match score {
        ClassificationScore::Raps(params) if params.needs_tuning() => {
            let (cal, tune) = split_indices(x_cal.nrows(), params.tuning_size, seed ^ 0x5EED)?;
            let fitted = conformalize_raps(
                model,
                classes,
                x_cal.select(Axis(0), &cal).view(),
                y_cal.select(Axis(0), &cal).view(),
                x_cal.select(Axis(0), &tune).view(),
                y_cal.select(Axis(0), &tune).view(),
                params,
                seed,
            )?;
            (cal, fitted)
        }
        _ => {
            let fitted = calibrate(model, classes, x_cal, y_cal, score, seed, None)?;
            ((0..x_cal.nrows()).collect(), fitted)
        }
    };
    fitted.groups = Some(cal.iter().map(|&i| groups[i]).collect());
    Ok(fitted)
}

/// Column index of every label in `classes`.
fn encode<F: Float>(classes: &[F], y: ArrayView1<F>) -> Result<Vec<usize>> {
    y.iter()
        .map(|&label| {
            classes.iter().position(|&c| c == label).ok_or_else(|| {
                RustMlError::InvalidParameter(format!(
                    "calibration label {label:?} is not one of the classes"
                ))
            })
        })
        .collect()
}

fn calibrate<F: Float, M: Predict<F> + PredictProba<F>>(
    model: M,
    classes: Vec<F>,
    x_cal: ArrayView2<F>,
    y_cal: ArrayView1<F>,
    score: ClassificationScore,
    seed: u64,
    tuning: Option<(Array2<F>, Vec<usize>)>,
) -> Result<FittedSplitConformalClassifier<M, F>> {
    check_xy(x_cal, y_cal)?;
    if classes.len() < 2 {
        return Err(RustMlError::InvalidParameter(
            "conformal classification needs at least 2 classes".into(),
        ));
    }
    let proba = model.predict_proba_view(x_cal)?;
    if proba.ncols() != classes.len() {
        return Err(RustMlError::ShapeMismatch(format!(
            "predict_proba returned {} columns for {} classes",
            proba.ncols(),
            classes.len()
        )));
    }
    let labels = encode(&classes, y_cal)?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut scores = Vec::with_capacity(labels.len());
    let mut ranks = Vec::with_capacity(labels.len());
    for (row, &k) in proba.rows().into_iter().zip(&labels) {
        let s = match score {
            ClassificationScore::Lac => F::one() - row[k],
            ClassificationScore::Aps(mode) | ClassificationScore::Raps(RapsParams { mode, .. }) => {
                let mut s = mass_up_to(row, k);
                if mode == ApsMode::Randomized {
                    s -= F::from_f64(rng.gen::<f64>()).expect("fits") * row[k];
                }
                s
            }
        };
        if s.is_nan() {
            return Err(RustMlError::InvalidParameter(
                "NaN in the calibration probabilities".into(),
            ));
        }
        scores.push(s);
        ranks.push(rank_of(row, k));
    }
    Ok(FittedSplitConformalClassifier {
        model,
        classes,
        scores,
        ranks,
        groups: None,
        tuning,
        score,
        seed,
        allow_full_sets: false,
    })
}

/// Classes in decreasing probability; ties put the higher index first
/// (numpy's `flip(argsort(p))` for small arrays).
fn order_desc<F: Float>(p: ArrayView1<F>) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..p.len()).collect();
    idx.sort_by(|&a, &b| p[a].partial_cmp(&p[b]).expect("no NaN"));
    idx.reverse();
    idx
}

/// 1-based rank of class `k` in decreasing probability.
fn rank_of<F: Float>(p: ArrayView1<F>, k: usize) -> usize {
    order_desc(p)
        .iter()
        .position(|&i| i == k)
        .expect("k is a column")
        + 1
}

/// Probability mass of the classes ranked up to and including class `k`.
fn mass_up_to<F: Float>(p: ArrayView1<F>, k: usize) -> F {
    let mut sum = F::zero();
    for i in order_desc(p) {
        sum += p[i];
        if i == k {
            break;
        }
    }
    sum
}

/// Calibrated classifier: model, classes and calibration scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedSplitConformalClassifier<M, F> {
    model: M,
    classes: Vec<F>,
    /// LAC / APS scores; for RAPS the unregularized APS part.
    scores: Vec<F>,
    /// 1-based rank of the true class of every calibration row (RAPS).
    ranks: Vec<usize>,
    /// Group of every calibration row (Mondrian).
    groups: Option<Vec<usize>>,
    /// RAPS tuning rows: probabilities and true columns.
    tuning: Option<(Array2<F>, Vec<usize>)>,
    score: ClassificationScore,
    seed: u64,
    allow_full_sets: bool,
}

/// RAPS regularization `(k_reg, lambda)` in use for one `alpha`.
#[derive(Debug, Clone, Copy)]
struct Reg<F> {
    k: usize,
    lambda: F,
}

impl<M: Predict<F> + PredictProba<F>, F: Float> FittedSplitConformalClassifier<M, F> {
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Labels of the set columns.
    pub fn classes(&self) -> &[F] {
        &self.classes
    }

    pub fn conformity_scores(&self) -> &[F] {
        &self.scores
    }

    pub fn with_allow_full_sets(mut self, allow: bool) -> Self {
        self.allow_full_sets = allow;
        self
    }

    /// RAPS: the `(k_reg, lambda)` used at `alpha` (tuned or fixed).
    pub fn raps_regularization(&self, alpha: f64) -> Result<Option<(usize, f64)>> {
        Ok(self
            .regularization(alpha)?
            .map(|r| (r.k, r.lambda.to_f64().expect("finite"))))
    }

    fn regularization(&self, alpha: f64) -> Result<Option<Reg<F>>> {
        let ClassificationScore::Raps(params) = self.score else {
            return Ok(None);
        };
        check_alpha(alpha)?;
        let tuning = || {
            self.tuning
                .as_ref()
                .ok_or_else(|| RustMlError::InvalidParameter("RAPS tuning rows are missing".into()))
        };
        let k = match params.k_reg {
            Some(k) => k,
            None => {
                let (proba, labels) = tuning()?;
                let mut pos: Vec<F> = proba
                    .rows()
                    .into_iter()
                    .zip(labels)
                    .map(|(r, &y)| F::from_usize(rank_of(r, y) - 1).expect("fits"))
                    .collect();
                let q = lower_quantile(&mut pos, 1.0 - alpha)
                    .ok_or_else(|| too_few(labels.len(), alpha))?;
                q.to_usize().expect("a rank") + 1
            }
        };
        let lambda = match params.lambda {
            Some(l) => F::from_f64(l).expect("fits"),
            None => self.tune_lambda(tuning()?, k, params.mode, alpha)?,
        };
        Ok(Some(Reg { k, lambda }))
    }

    /// The grid value of λ giving the smallest mean set on the tuning rows.
    fn tune_lambda(
        &self,
        tuning: &(Array2<F>, Vec<usize>),
        k: usize,
        mode: ApsMode,
        alpha: f64,
    ) -> Result<F> {
        let (proba, labels) = tuning;
        let eps = F::from_f64(EPS).expect("fits");
        // Randomisation plays no part in the choice (as in MAPIE).
        let mode = if mode == ApsMode::Exclude {
            ApsMode::Exclude
        } else {
            ApsMode::Include
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let (mut best, mut best_size) = (F::zero(), f64::MAX);
        for &l in &LAMBDA_GRID {
            let reg = Reg {
                k,
                lambda: F::from_f64(l).expect("fits"),
            };
            let mut scores: Vec<F> = proba
                .rows()
                .into_iter()
                .zip(labels)
                .map(|(r, &y)| mass_up_to(r, y) + penalty(rank_of(r, y), reg))
                .collect();
            let q = lower_quantile(&mut scores, 1.0 - alpha)
                .ok_or_else(|| too_few(labels.len(), alpha))?;
            let mut size = 0usize;
            let mut set = Array1::from_elem(proba.ncols(), false);
            for r in proba.rows() {
                aps_row(r, q, mode, eps, &mut rng, set.view_mut(), Some(reg));
                size += set.iter().filter(|&&b| b).count();
            }
            let size = size as f64 / labels.len() as f64;
            if size < best_size - EPS {
                best = reg.lambda;
                best_size = size;
            }
        }
        Ok(best)
    }

    /// The calibrated threshold `q` at `alpha` (`None` when there are too
    /// few calibration rows and full sets are allowed).
    pub fn threshold(&self, alpha: f64) -> Result<Option<F>> {
        check_alpha(alpha)?;
        let reg = self.regularization(alpha)?;
        self.threshold_over(alpha, reg, |_| true)
    }

    /// Mondrian: the threshold of group `group` (`None`: full sets allowed
    /// and too few rows in the group).
    pub fn group_threshold(&self, group: usize, alpha: f64) -> Result<Option<F>> {
        check_alpha(alpha)?;
        let reg = self.regularization(alpha)?;
        self.group_threshold_with(group, alpha, reg)
    }

    fn group_threshold_with(
        &self,
        group: usize,
        alpha: f64,
        reg: Option<Reg<F>>,
    ) -> Result<Option<F>> {
        let groups = self.groups.as_ref().ok_or_else(|| {
            RustMlError::InvalidParameter(
                "calibrated without groups; use conformalize_classifier_mondrian".into(),
            )
        })?;
        if !groups.contains(&group) {
            return Err(RustMlError::InvalidParameter(format!(
                "group {group} has no calibration rows"
            )));
        }
        self.threshold_over(alpha, reg, |i| groups[i] == group)
            .map_err(|e| RustMlError::InvalidParameter(format!("group {group}: {e}")))
    }

    /// Conformal threshold over the calibration rows selected by `keep`.
    fn threshold_over(
        &self,
        alpha: f64,
        reg: Option<Reg<F>>,
        keep: impl Fn(usize) -> bool,
    ) -> Result<Option<F>> {
        let mut s: Vec<F> = (0..self.scores.len())
            .filter(|&i| keep(i))
            .map(|i| self.scores[i] + reg.map_or(F::zero(), |r| penalty(self.ranks[i], r)))
            .collect();
        let n = s.len();
        match lower_quantile(&mut s, 1.0 - alpha) {
            Some(q) => Ok(Some(q)),
            None if self.allow_full_sets => Ok(None),
            None => Err(too_few(n, alpha)),
        }
    }

    /// Prediction sets at miscoverage `alpha`: `sets[[i, k]]` is true when
    /// class `classes()[k]` is in the set of row `i`.
    pub fn predict_set(&self, x: ArrayView2<F>, alpha: f64) -> Result<Array2<bool>> {
        let q = self.threshold(alpha)?;
        let reg = self.regularization(alpha)?;
        self.sets(x, |_| q, reg)
    }

    /// Mondrian prediction sets: row `i` uses the threshold of `groups[i]`,
    /// so coverage holds within each group.
    pub fn predict_set_mondrian(
        &self,
        x: ArrayView2<F>,
        groups: &[usize],
        alpha: f64,
    ) -> Result<Array2<bool>> {
        check_alpha(alpha)?;
        check_groups(groups, x.nrows())?;
        let reg = self.regularization(alpha)?;
        let mut q = std::collections::BTreeMap::new();
        for &g in groups {
            if let std::collections::btree_map::Entry::Vacant(e) = q.entry(g) {
                e.insert(self.group_threshold_with(g, alpha, reg)?);
            }
        }
        self.sets(x, |i| q[&groups[i]], reg)
    }

    /// Sets with threshold `q(i)` for row `i` (`None`: all classes).
    fn sets(
        &self,
        x: ArrayView2<F>,
        q: impl Fn(usize) -> Option<F>,
        reg: Option<Reg<F>>,
    ) -> Result<Array2<bool>> {
        let proba = self.model.predict_proba_view(x)?;
        let (n, k) = proba.dim();
        if k != self.classes.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "predict_proba returned {k} columns for {} classes",
                self.classes.len()
            )));
        }
        let eps = F::from_f64(EPS).expect("fits");
        let mut sets = Array2::from_elem((n, k), false);
        // A separate stream from the calibration draws, the same on every call.
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed ^ 0x9E37_79B9_7F4A_7C15);
        for (i, (mut set, p)) in sets.rows_mut().into_iter().zip(proba.rows()).enumerate() {
            let Some(qi) = q(i) else {
                set.fill(true);
                continue;
            };
            match self.score {
                ClassificationScore::Lac => {
                    for (s, &v) in set.iter_mut().zip(p) {
                        *s = (F::one() - v) - qi <= eps;
                    }
                }
                ClassificationScore::Aps(mode) => aps_row(p, qi, mode, eps, &mut rng, set, None),
                ClassificationScore::Raps(params) => {
                    aps_row(p, qi, params.mode, eps, &mut rng, set, reg)
                }
            }
        }
        Ok(sets)
    }

    /// Prediction sets as lists of class labels, one per row.
    pub fn predict_set_labels(&self, x: ArrayView2<F>, alpha: f64) -> Result<Vec<Vec<F>>> {
        let sets = self.predict_set(x, alpha)?;
        Ok(sets
            .rows()
            .into_iter()
            .map(|r| {
                r.iter()
                    .zip(&self.classes)
                    .filter(|(&inside, _)| inside)
                    .map(|(_, &c)| c)
                    .collect()
            })
            .collect())
    }
}

/// RAPS penalty of the class at 1-based `rank`.
fn penalty<F: Float>(rank: usize, reg: Reg<F>) -> F {
    reg.lambda * F::from_usize(rank.saturating_sub(reg.k)).expect("fits")
}

/// One APS / RAPS set (MAPIE's `_get_last_included_proba` +
/// `get_prediction_sets`); `reg` adds the RAPS penalty.
fn aps_row<F: Float>(
    p: ArrayView1<F>,
    q: F,
    mode: ApsMode,
    eps: F,
    rng: &mut rand::rngs::StdRng,
    mut set: ndarray::ArrayViewMut1<bool>,
    reg: Option<Reg<F>>,
) {
    let k = p.len();
    // Cumulative (regularized) mass in decreasing-probability order, by class.
    let mut cum = vec![F::zero(); k];
    let mut sum = F::zero();
    for (j, i) in order_desc(p).into_iter().enumerate() {
        sum += p[i];
        cum[i] = sum + reg.map_or(F::zero(), |r| penalty(j + 1, r));
    }
    let last = match mode {
        ApsMode::Include | ApsMode::Randomized => {
            // The class with the smallest cumulative mass reaching q.
            (0..k)
                .filter(|&i| cum[i] - q >= -eps)
                .min_by(|&a, &b| (cum[a] - q).partial_cmp(&(cum[b] - q)).expect("no NaN"))
                .unwrap_or_else(|| argmax(&cum))
        }
        ApsMode::Exclude => {
            // The class with the largest cumulative mass not above q (or the
            // top class when even it exceeds q).
            let min_cum = cum.iter().copied().fold(F::infinity(), F::min);
            let t = q.max(min_cum);
            let mut best: Option<usize> = None;
            for i in 0..k {
                if cum[i] - t <= eps && best.is_none_or(|b| cum[i] - t > cum[b] - t) {
                    best = Some(i);
                }
            }
            best.unwrap_or_else(|| argmax(&cum))
        }
    };
    let mut p_last = p[last];
    if p_last <= eps {
        // Zero-probability crossing class: use the smallest non-zero one.
        p_last = p
            .iter()
            .copied()
            .filter(|&v| v >= eps)
            .fold(F::infinity(), F::min);
    }
    for (s, &v) in set.iter_mut().zip(p) {
        *s = v - p_last >= -eps;
    }
    if mode == ApsMode::Randomized {
        let denom = match reg {
            None => p_last,
            Some(r) => {
                // MAPIE's RAPS `_compute_v_parameter`: L is the set size
                // before randomisation.
                let l = set.iter().filter(|&&b| b).count();
                let over = F::from_usize(l.saturating_sub(r.k)).expect("fits");
                let flag = if l > r.k { F::one() } else { F::zero() };
                p_last - r.lambda * over + r.lambda * flag
            }
        };
        let v = (cum[last] - q) / denom;
        let u = F::from_f64(rng.gen::<f64>()).expect("fits");
        set[last] = v - u <= eps;
    }
}

fn argmax<F: Float>(v: &[F]) -> usize {
    let mut best = 0;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}

impl<M: Predict<F> + PredictProba<F>, F: Float> Predict<F>
    for FittedSplitConformalClassifier<M, F>
{
    /// The underlying classifier's prediction.
    fn predict_view(&self, x: ArrayView2<F>) -> Result<Array1<F>> {
        self.model.predict_view(x)
    }
}
