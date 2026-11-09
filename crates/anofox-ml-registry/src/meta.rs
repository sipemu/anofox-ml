//! Meta-estimators over registered classifiers (sklearn's
//! `OneVsRestClassifier`, `OneVsOneClassifier`, `FixedThresholdClassifier`
//! and `TunedThresholdClassifierCV`).
//!
//! As in sklearn, a binary model's score is its `decision_function` when it
//! has one, else the positive-class column of `predict_proba` (the
//! threshold classifiers prefer `predict_proba`, `response_method="auto"`).
//! The registry knows each kind's capabilities, so this mirrors sklearn's
//! duck typing exactly.

use anofox_ml_core::{Predict, PredictProba, Result, RustMlError};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

use crate::estimator::{RegistryEstimator, RegistryFitted};
use crate::model::Out;

fn err(msg: impl Into<String>) -> RustMlError {
    RustMlError::InvalidParameter(msg.into())
}

/// Sorted distinct labels and each row's class index.
fn encode(y: ArrayView1<f64>) -> Result<(Vec<f64>, Vec<usize>)> {
    if y.iter().any(|v| !v.is_finite()) {
        return Err(err("labels must be finite"));
    }
    let mut classes = y.to_vec();
    classes.sort_by(f64::total_cmp);
    classes.dedup();
    let idx = y
        .iter()
        .map(|&v| classes.partition_point(|&c| c < v))
        .collect();
    Ok((classes, idx))
}

fn fit_binary(
    est: &RegistryEstimator,
    x: ArrayView2<f64>,
    y01: &Array1<f64>,
) -> Result<RegistryFitted> {
    let mut model = crate::make(&est.kind, &est.params)?;
    model.fit(x, Some(y01.view()), None)?;
    Ok(RegistryFitted(model))
}

fn has(m: &RegistryFitted, capability: &str) -> bool {
    m.model().capabilities().contains(&capability)
}

/// A binary decision function as one value per row (an `n × 2` output
/// becomes `col1 − col0`).
fn decision_1d(m: &RegistryFitted, x: ArrayView2<f64>) -> Result<Array1<f64>> {
    match m.model().decision_function(x)? {
        Out::A1(a) => Ok(a),
        Out::A2(a) if a.ncols() == 1 => Ok(a.column(0).to_owned()),
        Out::A2(a) if a.ncols() == 2 => Ok(&a.column(1) - &a.column(0)),
        Out::A2(a) => Err(err(format!(
            "{}: a binary decision_function with {} columns",
            m.model().kind(),
            a.ncols()
        ))),
    }
}

/// Probability of class column `col` of a binary model.
fn proba_col(m: &RegistryFitted, x: ArrayView2<f64>, col: usize) -> Result<Array1<f64>> {
    let p = m.predict_proba(x)?;
    if p.ncols() != 2 {
        return Err(err(format!(
            "{}: predict_proba of a binary model has {} columns",
            m.model().kind(),
            p.ncols()
        )));
    }
    Ok(p.column(col).to_owned())
}

/// sklearn's `_predict_binary`: decision function, else P(class 1).
fn binary_score(m: &RegistryFitted, x: ArrayView2<f64>, use_decision: bool) -> Result<Array1<f64>> {
    if use_decision {
        decision_1d(m, x)
    } else {
        proba_col(m, x, 1)
    }
}

fn argmax_rows(s: &Array2<f64>) -> Vec<usize> {
    s.axis_iter(Axis(0))
        .map(|r| {
            r.iter()
                .enumerate()
                .fold(0, |b, (j, &v)| if v > r[b] { j } else { b })
        })
        .collect()
}

fn check_scores(use_decision: bool, m: &RegistryFitted) -> Result<()> {
    if !use_decision && !has(m, "predict_proba") {
        return Err(err(format!(
            "{} has neither decision_function nor predict_proba",
            m.model().kind()
        )));
    }
    Ok(())
}

// ============================================================
// One-vs-rest
// ============================================================

/// One binary model per class (one model for two classes).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct OneVsRestConfig {
    pub estimator: String,
    pub estimator_params: serde_json::Value,
}

impl Default for OneVsRestConfig {
    fn default() -> Self {
        Self {
            estimator: "LogisticRegressor".into(),
            estimator_params: serde_json::json!({}),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub(crate) struct OneVsRestFit {
    classes: Vec<f64>,
    models: Vec<RegistryFitted>,
    use_decision: bool,
}

impl OneVsRestConfig {
    pub fn fit(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<OneVsRestFit> {
        let (classes, idx) = encode(y)?;
        if classes.len() < 2 {
            return Err(err("OneVsRestClassifier needs at least 2 classes"));
        }
        let est = RegistryEstimator::new(&self.estimator, self.estimator_params.clone());
        let targets: Vec<usize> = if classes.len() == 2 {
            vec![1]
        } else {
            (0..classes.len()).collect()
        };
        let models = targets
            .iter()
            .map(|&c| {
                let y01 = idx
                    .iter()
                    .map(|&i| if i == c { 1.0 } else { 0.0 })
                    .collect();
                fit_binary(&est, x, &y01)
            })
            .collect::<Result<Vec<_>>>()?;
        let use_decision = has(&models[0], "decision_function");
        check_scores(use_decision, &models[0])?;
        Ok(OneVsRestFit {
            classes,
            models,
            use_decision,
        })
    }
}

impl OneVsRestFit {
    fn scores(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        let cols = self
            .models
            .iter()
            .map(|m| binary_score(m, x, self.use_decision))
            .collect::<Result<Vec<_>>>()?;
        let views: Vec<_> = cols.iter().map(|c| c.view()).collect();
        ndarray::stack(Axis(1), &views).map_err(|e| err(e.to_string()))
    }

    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        let s = self.scores(x)?;
        if self.classes.len() == 2 {
            let thr = if self.use_decision { 0.0 } else { 0.5 };
            return Ok(s.column(0).mapv(|v| self.classes[(v > thr) as usize]));
        }
        Ok(argmax_rows(&s)
            .into_iter()
            .map(|k| self.classes[k])
            .collect())
    }

    pub fn predict_proba(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        let cols = self
            .models
            .iter()
            .map(|m| proba_col(m, x, 1))
            .collect::<Result<Vec<_>>>()?;
        if self.classes.len() == 2 {
            let p = &cols[0];
            return Ok(Array2::from_shape_fn((x.nrows(), 2), |(i, j)| {
                if j == 1 {
                    p[i]
                } else {
                    1.0 - p[i]
                }
            }));
        }
        let views: Vec<_> = cols.iter().map(|c| c.view()).collect();
        let mut p = ndarray::stack(Axis(1), &views).map_err(|e| err(e.to_string()))?;
        for mut row in p.rows_mut() {
            let s = row.sum();
            row /= s;
        }
        Ok(p)
    }

    pub fn decision_function(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        if !self.use_decision {
            return Err(err("the base estimator has no decision_function"));
        }
        self.scores(x)
    }
}

// ============================================================
// One-vs-one
// ============================================================

/// One binary model per pair of classes, combined by votes plus a
/// tie-breaking confidence (sklearn's `_ovr_decision_function`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct OneVsOneConfig {
    #[serde(flatten)]
    pub base: OneVsRestConfig,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct OneVsOneFit {
    classes: Vec<f64>,
    /// Pairs `(i, j)`, `i < j`, in sklearn's order; class `j` is the
    /// positive (1) label.
    models: Vec<RegistryFitted>,
    use_decision: bool,
}

impl OneVsOneConfig {
    pub fn fit(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<OneVsOneFit> {
        let (classes, idx) = encode(y)?;
        let k = classes.len();
        if k < 2 {
            return Err(err("OneVsOneClassifier needs at least 2 classes"));
        }
        let est = RegistryEstimator::new(&self.base.estimator, self.base.estimator_params.clone());
        let mut models = Vec::new();
        for i in 0..k {
            for j in i + 1..k {
                let rows: Vec<usize> = (0..idx.len())
                    .filter(|&r| idx[r] == i || idx[r] == j)
                    .collect();
                let y01: Array1<f64> = rows
                    .iter()
                    .map(|&r| if idx[r] == j { 1.0 } else { 0.0 })
                    .collect();
                models.push(fit_binary(&est, x.select(Axis(0), &rows).view(), &y01)?);
            }
        }
        let use_decision = has(&models[0], "decision_function");
        check_scores(use_decision, &models[0])?;
        Ok(OneVsOneFit {
            classes,
            models,
            use_decision,
        })
    }
}

impl OneVsOneFit {
    /// Votes plus `conf / (3 (|conf| + 1))`, `n × classes`.
    fn ovo_scores(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        let (n, k) = (x.nrows(), self.classes.len());
        let mut votes = Array2::<f64>::zeros((n, k));
        let mut conf = Array2::<f64>::zeros((n, k));
        let mut m = 0;
        for i in 0..k {
            for j in i + 1..k {
                let model = &self.models[m];
                let pred = model.predict(x)?;
                let c = binary_score(model, x, self.use_decision)?;
                for r in 0..n {
                    conf[[r, i]] -= c[r];
                    conf[[r, j]] += c[r];
                    if pred[r] == 0.0 {
                        votes[[r, i]] += 1.0;
                    } else if pred[r] == 1.0 {
                        votes[[r, j]] += 1.0;
                    }
                }
                m += 1;
            }
        }
        Ok(votes + conf.mapv(|c| c / (3.0 * (c.abs() + 1.0))))
    }

    pub fn decision_function(&self, x: ArrayView2<f64>) -> Result<Out> {
        let s = self.ovo_scores(x)?;
        if self.classes.len() == 2 {
            return Ok(Out::A1(s.column(1).to_owned()));
        }
        Ok(Out::A2(s))
    }

    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        let s = self.ovo_scores(x)?;
        if self.classes.len() == 2 {
            let thr = if self.use_decision { 0.0 } else { 0.5 };
            return Ok(s.column(1).mapv(|v| self.classes[(v > thr) as usize]));
        }
        Ok(argmax_rows(&s)
            .into_iter()
            .map(|k| self.classes[k])
            .collect())
    }
}

// ============================================================
// Decision thresholds
// ============================================================

/// Which score a threshold classifier thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub(crate) enum ResponseMethod {
    /// `predict_proba` when available, else `decision_function`.
    #[default]
    Auto,
    PredictProba,
    DecisionFunction,
}

/// A fitted binary classifier plus how its score is read.
#[derive(Serialize, Deserialize)]
pub(crate) struct ThresholdFit {
    classes: Vec<f64>,
    model: RegistryFitted,
    use_proba: bool,
    pos: usize,
    threshold: f64,
    /// TunedThresholdClassifierCV: the best mean CV score.
    best_score: Option<f64>,
}

fn binary_classes(y: ArrayView1<f64>, pos_label: Option<f64>) -> Result<(Vec<f64>, usize)> {
    let (classes, _) = encode(y)?;
    if classes.len() != 2 {
        return Err(err(format!(
            "threshold classifiers need a binary target, got {} classes",
            classes.len()
        )));
    }
    let pos = match pos_label {
        Some(p) => classes
            .iter()
            .position(|&c| c == p)
            .ok_or_else(|| err(format!("pos_label {p} is not one of {classes:?}")))?,
        None => 1,
    };
    Ok((classes, pos))
}

fn use_proba(response: ResponseMethod, model: &RegistryFitted) -> Result<bool> {
    let (p, d) = (has(model, "predict_proba"), has(model, "decision_function"));
    match response {
        ResponseMethod::Auto if p => Ok(true),
        ResponseMethod::Auto if d => Ok(false),
        ResponseMethod::PredictProba if p => Ok(true),
        ResponseMethod::DecisionFunction if d => Ok(false),
        _ => Err(err(format!(
            "{} has no {response:?} response",
            model.model().kind()
        ))),
    }
}

/// The score of the positive class (sklearn's `_get_response_values`: the
/// decision function is negated when the positive class is `classes[0]`).
fn response(
    model: &RegistryFitted,
    x: ArrayView2<f64>,
    use_proba: bool,
    pos: usize,
) -> Result<Array1<f64>> {
    if use_proba {
        proba_col(model, x, pos)
    } else {
        let d = decision_1d(model, x)?;
        Ok(if pos == 1 { d } else { -d })
    }
}

impl ThresholdFit {
    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        let s = response(&self.model, x, self.use_proba, self.pos)?;
        let neg = 1 - self.pos;
        Ok(s.mapv(|v| self.classes[if v >= self.threshold { self.pos } else { neg }]))
    }

    pub fn predict_proba(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        self.model.predict_proba(x)
    }

    pub fn decision_function(&self, x: ArrayView2<f64>) -> Result<Out> {
        self.model.model().decision_function(x)
    }
}

/// `FixedThresholdClassifier`: a binary classifier with a set threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct FixedThresholdConfig {
    pub estimator: String,
    pub estimator_params: serde_json::Value,
    /// `None`: 0.5 for probabilities, 0 for decision values.
    pub threshold: Option<f64>,
    pub response_method: ResponseMethod,
    pub pos_label: Option<f64>,
}

impl Default for FixedThresholdConfig {
    fn default() -> Self {
        Self {
            estimator: "LogisticRegressor".into(),
            estimator_params: serde_json::json!({}),
            threshold: None,
            response_method: ResponseMethod::Auto,
            pos_label: None,
        }
    }
}

impl FixedThresholdConfig {
    pub fn fit(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<ThresholdFit> {
        let (classes, pos) = binary_classes(y, self.pos_label)?;
        let mut model = crate::make(&self.estimator, &self.estimator_params)?;
        model.fit(x, Some(y), None)?;
        let model = RegistryFitted(model);
        let use_proba = use_proba(self.response_method, &model)?;
        let threshold = self.threshold.unwrap_or(if use_proba { 0.5 } else { 0.0 });
        Ok(ThresholdFit {
            classes,
            model,
            use_proba,
            pos,
            threshold,
            best_score: None,
        })
    }
}

/// Metric a [`TunedThresholdConfig`] maximises.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub(crate) enum ThresholdScoring {
    #[default]
    BalancedAccuracy,
    Accuracy,
    F1,
    Precision,
    Recall,
}

/// Binary metric of predictions (`true` = positive) against the truth;
/// undefined ratios count as 0 (sklearn's `zero_division` default).
fn binary_metric(scoring: ThresholdScoring, truth: &[bool], pred: &[bool]) -> f64 {
    let (mut tp, mut fp, mut tn, mut fneg) = (0.0, 0.0, 0.0, 0.0);
    for (&t, &p) in truth.iter().zip(pred) {
        match (t, p) {
            (true, true) => tp += 1.0,
            (false, true) => fp += 1.0,
            (false, false) => tn += 1.0,
            (true, false) => fneg += 1.0,
        }
    }
    let ratio = |a: f64, b: f64| if b > 0.0 { a / b } else { 0.0 };
    match scoring {
        ThresholdScoring::Accuracy => (tp + tn) / truth.len() as f64,
        ThresholdScoring::Precision => ratio(tp, tp + fp),
        ThresholdScoring::Recall => ratio(tp, tp + fneg),
        ThresholdScoring::F1 => ratio(2.0 * tp, 2.0 * tp + fp + fneg),
        ThresholdScoring::BalancedAccuracy => {
            let mut recalls = Vec::new();
            if tp + fneg > 0.0 {
                recalls.push(tp / (tp + fneg));
            }
            if tn + fp > 0.0 {
                recalls.push(tn / (tn + fp));
            }
            recalls.iter().sum::<f64>() / recalls.len() as f64
        }
    }
}

fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![a];
    }
    let step = (b - a) / (n - 1) as f64;
    (0..n)
        .map(|i| if i == n - 1 { b } else { a + step * i as f64 })
        .collect()
}

/// numpy's `interp` (clamped at both ends; `xp` ascending).
fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    if x <= xp[0] {
        return fp[0];
    }
    if x >= xp[xp.len() - 1] {
        return fp[fp.len() - 1];
    }
    let j = xp.partition_point(|&v| v <= x);
    let (x0, x1, y0, y1) = (xp[j - 1], xp[j], fp[j - 1], fp[j]);
    if x1 == x0 {
        return y1;
    }
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

/// `TunedThresholdClassifierCV`: the threshold that maximises the mean CV
/// score (sklearn's defaults: unshuffled stratified 5-fold, 100 thresholds,
/// balanced accuracy, refit on all rows).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct TunedThresholdConfig {
    pub estimator: String,
    pub estimator_params: serde_json::Value,
    pub scoring: ThresholdScoring,
    pub response_method: ResponseMethod,
    pub thresholds: usize,
    pub cv: usize,
    pub pos_label: Option<f64>,
}

impl Default for TunedThresholdConfig {
    fn default() -> Self {
        Self {
            estimator: "LogisticRegressor".into(),
            estimator_params: serde_json::json!({}),
            scoring: ThresholdScoring::BalancedAccuracy,
            response_method: ResponseMethod::Auto,
            thresholds: 100,
            cv: 5,
            pos_label: None,
        }
    }
}

impl TunedThresholdConfig {
    pub fn fit(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<ThresholdFit> {
        if self.thresholds < 2 {
            return Err(err("thresholds must be at least 2"));
        }
        let (classes, pos) = binary_classes(y, self.pos_label)?;
        let splits = anofox_ml_core::stratified_k_fold_ordered(y, self.cv)?;
        let mut fold_thresholds = Vec::new();
        let mut fold_scores = Vec::new();
        let mut use_proba_any = None;
        for (train, test) in &splits {
            let mut model = crate::make(&self.estimator, &self.estimator_params)?;
            let yt = y.select(Axis(0), train);
            model.fit(x.select(Axis(0), train).view(), Some(yt.view()), None)?;
            let model = RegistryFitted(model);
            let up = use_proba(self.response_method, &model)?;
            use_proba_any = Some(up);
            let xv = x.select(Axis(0), test);
            let s = response(&model, xv.view(), up, pos)?;
            let truth: Vec<bool> = test.iter().map(|&i| y[i] == classes[pos]).collect();
            let (lo, hi) = s
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
                    (a.min(v), b.max(v))
                });
            let ths = linspace(lo, hi, self.thresholds);
            if (ths[0] - ths[ths.len() - 1]).abs() <= 1e-8 + 1e-5 * ths[ths.len() - 1].abs() {
                return Err(err(
                    "the estimator makes constant predictions; the threshold cannot be tuned",
                ));
            }
            let scores: Vec<f64> = ths
                .iter()
                .map(|&t| {
                    let pred: Vec<bool> = s.iter().map(|&v| v >= t).collect();
                    binary_metric(self.scoring, &truth, &pred)
                })
                .collect();
            fold_thresholds.push(ths);
            fold_scores.push(scores);
        }
        let lo = fold_thresholds
            .iter()
            .map(|t| t[0])
            .fold(f64::INFINITY, f64::min);
        let hi = fold_thresholds
            .iter()
            .map(|t| t[t.len() - 1])
            .fold(f64::NEG_INFINITY, f64::max);
        let candidates = linspace(lo, hi, self.thresholds);
        let objective: Vec<f64> = candidates
            .iter()
            .map(|&t| {
                fold_thresholds
                    .iter()
                    .zip(&fold_scores)
                    .map(|(th, sc)| interp(t, th, sc))
                    .sum::<f64>()
                    / fold_thresholds.len() as f64
            })
            .collect();
        let best = objective
            .iter()
            .enumerate()
            .fold(0, |b, (i, &v)| if v > objective[b] { i } else { b });

        let mut model = crate::make(&self.estimator, &self.estimator_params)?;
        model.fit(x, Some(y), None)?;
        let model = RegistryFitted(model);
        let use_proba = use_proba(self.response_method, &model)?;
        debug_assert_eq!(Some(use_proba), use_proba_any);
        Ok(ThresholdFit {
            classes,
            model,
            use_proba,
            pos,
            threshold: candidates[best],
            best_score: Some(objective[best]),
        })
    }
}

// ============================================================
// Self-training
// ============================================================

/// How a [`SelfTrainingConfig`] picks the pseudo-labels of a round.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub(crate) enum SelfTrainingCriterion {
    /// Unlabelled rows whose top probability exceeds `threshold`.
    Threshold(f64),
    /// The `k` unlabelled rows with the highest top probability.
    KBest(usize),
}

impl Default for SelfTrainingCriterion {
    fn default() -> Self {
        SelfTrainingCriterion::Threshold(0.75)
    }
}

/// `SelfTrainingClassifier`: fit on the labelled rows (`y ≠ −1`), add the
/// confident predictions on unlabelled rows as labels, repeat, then refit.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct SelfTrainingConfig {
    pub estimator: String,
    pub estimator_params: serde_json::Value,
    pub criterion: SelfTrainingCriterion,
    /// Rounds of pseudo-labelling (`None`: until nothing changes).
    pub max_iter: Option<usize>,
}

impl Default for SelfTrainingConfig {
    fn default() -> Self {
        Self {
            estimator: "LogisticRegressor".into(),
            estimator_params: serde_json::json!({}),
            criterion: SelfTrainingCriterion::default(),
            max_iter: Some(10),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub(crate) struct SelfTrainingFit {
    model: RegistryFitted,
    transduction: Array1<f64>,
    labeled_iter: Array1<f64>,
    n_iter: usize,
    termination: String,
}

impl SelfTrainingConfig {
    pub fn fit(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<SelfTrainingFit> {
        let n = y.len();
        if x.nrows() != n {
            return Err(err(format!("x has {} rows but y has {n}", x.nrows())));
        }
        let mut has_label: Vec<bool> = y.iter().map(|&v| v != -1.0).collect();
        if !has_label.iter().any(|&h| h) {
            return Err(err("no labelled rows (all labels are −1)"));
        }
        let mut transduction = y.to_owned();
        let mut labeled_iter =
            Array1::from_iter(has_label.iter().map(|&h| if h { 0.0 } else { -1.0 }));
        let fit_on = |mask: &[bool], t: &Array1<f64>| -> Result<RegistryFitted> {
            let rows: Vec<usize> = (0..n).filter(|&i| mask[i]).collect();
            let mut m = crate::make(&self.estimator, &self.estimator_params)?;
            m.fit(
                x.select(Axis(0), &rows).view(),
                Some(t.select(Axis(0), &rows).view()),
                None,
            )?;
            Ok(RegistryFitted(m))
        };
        let mut n_iter = 0;
        let mut termination = String::new();
        while has_label.iter().any(|&h| !h) && self.max_iter.is_none_or(|m| n_iter < m) {
            n_iter += 1;
            let model = fit_on(&has_label, &transduction)?;
            let unl: Vec<usize> = (0..n).filter(|&i| !has_label[i]).collect();
            let proba = model.predict_proba(x.select(Axis(0), &unl).view())?;
            // Class labels of the probability columns: the sorted labels
            // the model was fitted on.
            let mut classes: Vec<f64> = (0..n)
                .filter(|&i| has_label[i])
                .map(|i| transduction[i])
                .collect();
            classes.sort_by(f64::total_cmp);
            classes.dedup();
            let best: Vec<(usize, f64)> = proba
                .rows()
                .into_iter()
                .map(|r| {
                    r.iter()
                        .enumerate()
                        .fold(
                            (0, f64::NEG_INFINITY),
                            |b, (j, &v)| if v > b.1 { (j, v) } else { b },
                        )
                })
                .collect();
            let selected: Vec<usize> = match self.criterion {
                SelfTrainingCriterion::Threshold(t) => {
                    (0..unl.len()).filter(|&k| best[k].1 > t).collect()
                }
                SelfTrainingCriterion::KBest(k) => {
                    let mut order: Vec<usize> = (0..unl.len()).collect();
                    order.sort_by(|&a, &b| best[b].1.total_cmp(&best[a].1).then(a.cmp(&b)));
                    order.truncate(k.min(unl.len()));
                    order
                }
            };
            for &k in &selected {
                let i = unl[k];
                transduction[i] = classes[best[k].0];
                has_label[i] = true;
                labeled_iter[i] = n_iter as f64;
            }
            if selected.is_empty() {
                termination = "no_change".into();
                break;
            }
        }
        if self.max_iter == Some(n_iter) {
            termination = "max_iter".into();
        }
        if has_label.iter().all(|&h| h) {
            termination = "all_labeled".into();
        }
        let model = fit_on(&has_label, &transduction)?;
        Ok(SelfTrainingFit {
            model,
            transduction,
            labeled_iter,
            n_iter,
            termination,
        })
    }
}

impl SelfTrainingFit {
    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        self.model.predict(x)
    }

    pub fn predict_proba(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        self.model.predict_proba(x)
    }
}

// ============================================================
// Voting
// ============================================================

/// One base estimator of a multi-estimator meta kind.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct EstimatorSpec {
    pub estimator: String,
    #[serde(default = "empty_params")]
    pub estimator_params: serde_json::Value,
}

fn empty_params() -> serde_json::Value {
    serde_json::json!({})
}

impl EstimatorSpec {
    fn fit(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<RegistryFitted> {
        let mut model = crate::make(&self.estimator, &self.estimator_params)?;
        model.fit(x, Some(y), None)?;
        Ok(RegistryFitted(model))
    }
}

/// The estimators' weights (1 each by default), checked against their
/// number.
fn voting_weights(
    kind: &str,
    specs: &[EstimatorSpec],
    weights: &Option<Vec<f64>>,
) -> Result<Vec<f64>> {
    if specs.is_empty() {
        return Err(err(format!("{kind} needs at least one estimator")));
    }
    match weights {
        None => Ok(vec![1.0; specs.len()]),
        Some(w) if w.len() == specs.len() => Ok(w.clone()),
        Some(w) => Err(err(format!(
            "{kind}: {} weights for {} estimators",
            w.len(),
            specs.len()
        ))),
    }
}

/// `Σ w_k a_k / Σ w_k`, summed in estimator order (numpy's `average`).
fn weighted_average<D: ndarray::Dimension>(
    arrays: &[ndarray::Array<f64, D>],
    w: &[f64],
) -> ndarray::Array<f64, D> {
    let mut sum = &arrays[0] * w[0];
    for (a, &wk) in arrays.iter().zip(w).skip(1) {
        sum = sum + a * wk;
    }
    let total: f64 = w.iter().sum();
    sum / total
}

/// Averages the predictions of its estimators (sklearn's `VotingRegressor`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct VotingRegressorConfig {
    pub estimators: Vec<EstimatorSpec>,
    pub weights: Option<Vec<f64>>,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct VotingRegressorFit {
    models: Vec<RegistryFitted>,
    weights: Vec<f64>,
}

impl VotingRegressorConfig {
    pub fn fit(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<VotingRegressorFit> {
        let weights = voting_weights("VotingRegressor", &self.estimators, &self.weights)?;
        let models = self
            .estimators
            .iter()
            .map(|s| s.fit(x, y))
            .collect::<Result<Vec<_>>>()?;
        Ok(VotingRegressorFit { models, weights })
    }
}

impl VotingRegressorFit {
    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        let preds = self
            .models
            .iter()
            .map(|m| m.predict(x))
            .collect::<Result<Vec<_>>>()?;
        Ok(weighted_average(&preds, &self.weights))
    }
}

/// Hard (weighted majority of the predicted classes) or soft (weighted
/// mean of the probabilities) voting.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Voting {
    #[default]
    Hard,
    Soft,
}

/// sklearn's `VotingClassifier`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct VotingClassifierConfig {
    pub estimators: Vec<EstimatorSpec>,
    pub voting: Voting,
    pub weights: Option<Vec<f64>>,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct VotingClassifierFit {
    classes: Vec<f64>,
    models: Vec<RegistryFitted>,
    weights: Vec<f64>,
    voting: Voting,
}

impl VotingClassifierConfig {
    pub fn fit(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<VotingClassifierFit> {
        let weights = voting_weights("VotingClassifier", &self.estimators, &self.weights)?;
        let (classes, _) = encode(y)?;
        let models = self
            .estimators
            .iter()
            .map(|s| s.fit(x, y))
            .collect::<Result<Vec<_>>>()?;
        if self.voting == Voting::Soft {
            if let Some(m) = models.iter().find(|m| !has(m, "predict_proba")) {
                return Err(err(format!(
                    "VotingClassifier: soft voting needs predict_proba, which {} lacks",
                    m.model().kind()
                )));
            }
        }
        Ok(VotingClassifierFit {
            classes,
            models,
            weights,
            voting: self.voting,
        })
    }
}

impl VotingClassifierFit {
    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        let index = match self.voting {
            Voting::Soft => argmax_rows(&self.predict_proba(x)?),
            Voting::Hard => {
                // Weighted class counts per row; ties go to the first class.
                let mut votes = Array2::<f64>::zeros((x.nrows(), self.classes.len()));
                for (m, &w) in self.models.iter().zip(&self.weights) {
                    for (i, p) in m.predict(x)?.iter().enumerate() {
                        let k = self.classes.partition_point(|&c| c < *p);
                        if k == self.classes.len() || self.classes[k] != *p {
                            return Err(err(format!(
                                "{} predicted an unknown class {p}",
                                m.model().kind()
                            )));
                        }
                        votes[[i, k]] += w;
                    }
                }
                argmax_rows(&votes)
            }
        };
        Ok(index.into_iter().map(|k| self.classes[k]).collect())
    }

    pub fn predict_proba(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        if self.voting == Voting::Hard {
            return Err(err(
                "VotingClassifier has predict_proba only with voting=\"soft\"",
            ));
        }
        let probas = self
            .models
            .iter()
            .map(|m| m.predict_proba(x))
            .collect::<Result<Vec<_>>>()?;
        Ok(weighted_average(&probas, &self.weights))
    }
}

// ============================================================
// Stacking
// ============================================================

/// The output a classifier's base estimators feed the final one
/// (sklearn's `stack_method`; `auto`: the first of `predict_proba`,
/// `decision_function`, `predict` the estimator has).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum StackMethod {
    #[default]
    Auto,
    PredictProba,
    DecisionFunction,
    Predict,
}

impl StackMethod {
    fn resolve(self, m: &RegistryFitted) -> Result<StackMethod> {
        let method = match self {
            StackMethod::Auto if has(m, "predict_proba") => StackMethod::PredictProba,
            StackMethod::Auto if has(m, "decision_function") => StackMethod::DecisionFunction,
            StackMethod::Auto => StackMethod::Predict,
            other => other,
        };
        let name = match method {
            StackMethod::PredictProba => "predict_proba",
            StackMethod::DecisionFunction => "decision_function",
            _ => "predict",
        };
        if !has(m, name) {
            return Err(err(format!(
                "StackingClassifier: {} has no {name}",
                m.model().kind()
            )));
        }
        Ok(method)
    }

    /// The meta-features of one base model (sklearn's
    /// `_concatenate_predictions`: one column of a binary `predict_proba`).
    fn features(
        self,
        m: &RegistryFitted,
        x: ArrayView2<f64>,
        n_classes: usize,
    ) -> Result<Array2<f64>> {
        let as_column = |a: Array1<f64>| a.insert_axis(Axis(1));
        Ok(match self {
            StackMethod::PredictProba => {
                let p = m.predict_proba(x)?;
                if n_classes == 2 {
                    p.slice(ndarray::s![.., 1..]).to_owned()
                } else {
                    p
                }
            }
            StackMethod::DecisionFunction => match m.model().decision_function(x)? {
                Out::A1(a) => as_column(a),
                Out::A2(a) => a,
            },
            _ => as_column(m.predict(x)?),
        })
    }
}

/// Out-of-fold meta-features of every base estimator and the estimators
/// refitted on all rows (sklearn's `cross_val_predict` and `estimators_`;
/// `full`: those refits, when the caller has them already).
fn stack(
    kind: &str,
    specs: &[EstimatorSpec],
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    splits: &[anofox_ml_core::CvSplit],
    full: Option<Vec<RegistryFitted>>,
    features: impl Fn(usize, &RegistryFitted, ArrayView2<f64>) -> Result<Array2<f64>>,
) -> Result<(Array2<f64>, Vec<RegistryFitted>)> {
    if specs.is_empty() {
        return Err(err(format!("{kind} needs at least one estimator")));
    }
    let models = match full {
        Some(m) => m,
        None => specs
            .iter()
            .map(|s| s.fit(x, y))
            .collect::<Result<Vec<_>>>()?,
    };
    let mut blocks = Vec::with_capacity(specs.len());
    for (k, spec) in specs.iter().enumerate() {
        let mut block: Option<Array2<f64>> = None;
        for (train, test) in splits {
            let fold = spec.fit(
                x.select(Axis(0), train).view(),
                y.select(Axis(0), train).view(),
            )?;
            let f = features(k, &fold, x.select(Axis(0), test).view())?;
            let b = block.get_or_insert_with(|| Array2::zeros((x.nrows(), f.ncols())));
            if f.ncols() != b.ncols() {
                return Err(err(format!(
                    "{kind}: {} gave {} columns on one fold and {} on another",
                    spec.estimator,
                    f.ncols(),
                    b.ncols()
                )));
            }
            for (r, &i) in test.iter().enumerate() {
                b.row_mut(i).assign(&f.row(r));
            }
        }
        blocks.push(block.ok_or_else(|| err(format!("{kind}: no folds")))?);
    }
    let views: Vec<_> = blocks.iter().map(|b| b.view()).collect();
    let meta = ndarray::concatenate(Axis(1), &views).map_err(|e| err(e.to_string()))?;
    Ok((meta, models))
}

/// `[meta | x]` with `passthrough`.
fn with_passthrough(
    meta: Array2<f64>,
    x: ArrayView2<f64>,
    passthrough: bool,
) -> Result<Array2<f64>> {
    if !passthrough {
        return Ok(meta);
    }
    ndarray::concatenate(Axis(1), &[meta.view(), x.view()]).map_err(|e| err(e.to_string()))
}

fn check_cv(kind: &str, cv: usize) -> Result<()> {
    if cv < 2 {
        return Err(err(format!("{kind}: cv must be at least 2, got {cv}")));
    }
    Ok(())
}

/// sklearn's `StackingRegressor`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct StackingRegressorConfig {
    pub estimators: Vec<EstimatorSpec>,
    pub final_estimator: EstimatorSpec,
    pub cv: usize,
    pub passthrough: bool,
}

impl Default for StackingRegressorConfig {
    fn default() -> Self {
        Self {
            estimators: Vec::new(),
            final_estimator: EstimatorSpec {
                estimator: "RidgeCrossValidated".into(),
                estimator_params: empty_params(),
            },
            cv: 5,
            passthrough: false,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub(crate) struct StackingRegressorFit {
    models: Vec<RegistryFitted>,
    final_model: RegistryFitted,
    passthrough: bool,
}

impl StackingRegressorConfig {
    pub fn fit(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<StackingRegressorFit> {
        check_cv("StackingRegressor", self.cv)?;
        let splits = anofox_ml_core::k_fold(x.nrows(), self.cv)?;
        let (meta, models) = stack(
            "StackingRegressor",
            &self.estimators,
            x,
            y,
            &splits,
            None,
            |_, m, xs| Ok(m.predict(xs)?.insert_axis(Axis(1))),
        )?;
        let meta = with_passthrough(meta, x, self.passthrough)?;
        let final_model = self.final_estimator.fit(meta.view(), y)?;
        Ok(StackingRegressorFit {
            models,
            final_model,
            passthrough: self.passthrough,
        })
    }
}

impl StackingRegressorFit {
    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        let cols = self
            .models
            .iter()
            .map(|m| Ok(m.predict(x)?.insert_axis(Axis(1))))
            .collect::<Result<Vec<_>>>()?;
        let views: Vec<_> = cols.iter().map(|c| c.view()).collect();
        let meta = ndarray::concatenate(Axis(1), &views).map_err(|e| err(e.to_string()))?;
        self.final_model
            .predict(with_passthrough(meta, x, self.passthrough)?.view())
    }
}

/// sklearn's `StackingClassifier`: the labels are encoded `0..k` for the
/// base and final estimators.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct StackingClassifierConfig {
    pub estimators: Vec<EstimatorSpec>,
    pub final_estimator: EstimatorSpec,
    pub cv: usize,
    pub passthrough: bool,
    pub stack_method: StackMethod,
}

impl Default for StackingClassifierConfig {
    fn default() -> Self {
        Self {
            estimators: Vec::new(),
            final_estimator: EstimatorSpec {
                estimator: "LogisticRegressor".into(),
                estimator_params: empty_params(),
            },
            cv: 5,
            passthrough: false,
            stack_method: StackMethod::Auto,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub(crate) struct StackingClassifierFit {
    classes: Vec<f64>,
    models: Vec<RegistryFitted>,
    methods: Vec<StackMethod>,
    final_model: RegistryFitted,
    passthrough: bool,
}

impl StackingClassifierConfig {
    pub fn fit(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<StackingClassifierFit> {
        check_cv("StackingClassifier", self.cv)?;
        let (classes, idx) = encode(y)?;
        let y_enc: Array1<f64> = idx.iter().map(|&i| i as f64).collect();
        let splits = anofox_ml_core::stratified_k_fold_ordered(&y_enc, self.cv)?;
        // The stack method of each base estimator, from its fit on all rows.
        let full = self
            .estimators
            .iter()
            .map(|s| s.fit(x, y_enc.view()))
            .collect::<Result<Vec<_>>>()?;
        let methods = full
            .iter()
            .map(|m| self.stack_method.resolve(m))
            .collect::<Result<Vec<_>>>()?;
        let n_classes = classes.len();
        let (meta, models) = stack(
            "StackingClassifier",
            &self.estimators,
            x,
            y_enc.view(),
            &splits,
            Some(full),
            |k, m, xs| methods[k].features(m, xs, n_classes),
        )?;
        let meta = with_passthrough(meta, x, self.passthrough)?;
        let final_model = self.final_estimator.fit(meta.view(), y_enc.view())?;
        Ok(StackingClassifierFit {
            classes,
            models,
            methods,
            final_model,
            passthrough: self.passthrough,
        })
    }
}

impl StackingClassifierFit {
    fn meta(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        let blocks = self
            .models
            .iter()
            .zip(&self.methods)
            .map(|(m, method)| method.features(m, x, self.classes.len()))
            .collect::<Result<Vec<_>>>()?;
        let views: Vec<_> = blocks.iter().map(|b| b.view()).collect();
        let meta = ndarray::concatenate(Axis(1), &views).map_err(|e| err(e.to_string()))?;
        with_passthrough(meta, x, self.passthrough)
    }

    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        let encoded = self.final_model.predict(self.meta(x)?.view())?;
        encoded
            .iter()
            .map(|&v| {
                self.classes
                    .get(v as usize)
                    .copied()
                    .ok_or_else(|| err(format!("final estimator predicted an unknown class {v}")))
            })
            .collect()
    }

    pub fn predict_proba(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        self.final_model.predict_proba(self.meta(x)?.view())
    }
}

// ============================================================
// Probability calibration
// ============================================================

/// sklearn's `CalibratedClassifierCV(method=...)`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum CalibrationMethod {
    #[default]
    Sigmoid,
    Isotonic,
}

/// One class's map from scores to probabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
enum Calibrator {
    /// `expit(-(a·f + b))` (sklearn's `_SigmoidCalibration`).
    Sigmoid { a: f64, b: f64 },
    /// sklearn's `IsotonicRegression(out_of_bounds="clip")`: thresholds,
    /// linearly interpolated (scipy's `interp1d`), clipped to their range.
    Isotonic { x: Vec<f64>, y: Vec<f64> },
}

fn expit(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// sklearn's `_sigmoid_calibration` (Platt): the half-binomial loss of the
/// prior-smoothed targets at `raw = -(a f + b)`, minimised by Newton's
/// method (sklearn: L-BFGS-B to `gtol = 1e-6`, the same minimum).
fn fit_sigmoid(scores: &[f64], y01: &[f64]) -> Calibrator {
    let max_abs = scores.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let scale = if max_abs >= 30.0 { max_abs } else { 1.0 };
    let f: Vec<f64> = scores.iter().map(|v| v / scale).collect();
    let prior0 = y01.iter().filter(|&&v| v <= 0.0).count() as f64;
    let prior1 = y01.len() as f64 - prior0;
    let t: Vec<f64> = y01
        .iter()
        .map(|&v| {
            if v > 0.0 {
                (prior1 + 1.0) / (prior1 + 2.0)
            } else {
                1.0 / (prior0 + 2.0)
            }
        })
        .collect();
    let loss = |a: f64, b: f64| -> f64 {
        f.iter()
            .zip(&t)
            .map(|(&fi, &ti)| {
                let raw = -(a * fi + b);
                // log(1 + exp(raw)) - t raw, stably.
                raw.max(0.0) + (-raw.abs()).exp().ln_1p() - ti * raw
            })
            .sum()
    };
    let (mut a, mut b) = (0.0, ((prior0 + 1.0) / (prior1 + 1.0)).ln());
    let mut value = loss(a, b);
    for _ in 0..200 {
        // Gradient and Hessian in (a, b); d raw / d(a, b) = -(f, 1).
        let (mut ga, mut gb, mut haa, mut hab, mut hbb) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for (&fi, &ti) in f.iter().zip(&t) {
            let p = expit(-(a * fi + b));
            let g = p - ti;
            let h = p * (1.0 - p);
            ga -= g * fi;
            gb -= g;
            haa += h * fi * fi;
            hab += h * fi;
            hbb += h;
        }
        if ga.abs().max(gb.abs()) < 1e-13 * f.len() as f64 {
            break;
        }
        let det = haa * hbb - hab * hab;
        if det.abs() < 1e-300 {
            break;
        }
        let (da, db) = (-(hbb * ga - hab * gb) / det, -(haa * gb - hab * ga) / det);
        let mut step = 1.0;
        let mut moved = false;
        while step > 1e-12 {
            let (na, nb) = (a + step * da, b + step * db);
            let nv = loss(na, nb);
            if nv <= value {
                (a, b, value) = (na, nb, nv);
                moved = true;
                break;
            }
            step /= 2.0;
        }
        if !moved {
            break;
        }
    }
    Calibrator::Sigmoid { a: a / scale, b }
}

/// sklearn's `IsotonicRegression(out_of_bounds="clip").fit`: rows sorted
/// by score (then label), equal scores pooled (`_make_unique`, to numpy's
/// resolution), pool adjacent violators
/// (`_inplace_contiguous_isotonic_regression`), interior points equal to
/// both neighbours dropped.
fn fit_isotonic(scores: &[f64], y01: &[f64]) -> Calibrator {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&i, &j| {
        scores[i]
            .total_cmp(&scores[j])
            .then(y01[i].total_cmp(&y01[j]))
    });
    let (mut xs, mut ys, mut ws) = (Vec::new(), Vec::new(), Vec::new());
    let (mut cx, mut cy, mut cw) = (scores[order[0]], 0.0, 0.0);
    for &k in &order {
        if scores[k] - cx >= 1e-15 {
            xs.push(cx);
            ys.push(cy / cw);
            ws.push(cw);
            (cx, cy, cw) = (scores[k], y01[k], 1.0);
        } else {
            cy += y01[k];
            cw += 1.0;
        }
    }
    xs.push(cx);
    ys.push(cy / cw);
    ws.push(cw);
    // Pool adjacent violators, as sklearn's Cython (single pass with
    // backtracking; equal neighbours pool).
    let n = ys.len();
    let mut target: Vec<usize> = (0..n).collect();
    let mut i = 0;
    while i < n {
        let mut k = target[i] + 1;
        if k == n {
            break;
        }
        if ys[i] < ys[k] {
            i = k;
            continue;
        }
        let (mut sum_wy, mut sum_w) = (ws[i] * ys[i], ws[i]);
        loop {
            let prev_y = ys[k];
            sum_wy += ws[k] * ys[k];
            sum_w += ws[k];
            k = target[k] + 1;
            if k == n || prev_y < ys[k] {
                ys[i] = sum_wy / sum_w;
                ws[i] = sum_w;
                target[i] = k - 1;
                target[k - 1] = i;
                if i > 0 {
                    i = target[i - 1];
                }
                break;
            }
        }
    }
    let mut i = 0;
    while i < n {
        let k = target[i] + 1;
        for j in i + 1..k {
            ys[j] = ys[i];
        }
        i = k;
    }
    let keep: Vec<bool> = (0..n)
        .map(|j| j == 0 || j == n - 1 || ys[j] != ys[j - 1] || ys[j] != ys[j + 1])
        .collect();
    let pick = |v: &[f64]| {
        v.iter()
            .zip(&keep)
            .filter(|(_, &k)| k)
            .map(|(&a, _)| a)
            .collect()
    };
    Calibrator::Isotonic {
        x: pick(&xs),
        y: pick(&ys),
    }
}

impl Calibrator {
    fn fit(method: CalibrationMethod, scores: &[f64], y01: &[f64]) -> Calibrator {
        match method {
            CalibrationMethod::Sigmoid => fit_sigmoid(scores, y01),
            CalibrationMethod::Isotonic => fit_isotonic(scores, y01),
        }
    }

    fn predict(&self, s: f64) -> f64 {
        match self {
            Calibrator::Sigmoid { a, b } => expit(-(a * s + b)),
            Calibrator::Isotonic { x, y } => {
                if x.len() == 1 {
                    return y[0];
                }
                let s = s.clamp(x[0], x[x.len() - 1]);
                // scipy's interp1d: searchsorted (left), clipped to 1..n-1.
                let hi = x.partition_point(|&v| v < s).clamp(1, x.len() - 1);
                let lo = hi - 1;
                let slope = (y[hi] - y[lo]) / (x[hi] - x[lo]);
                slope * (s - x[lo]) + y[lo]
            }
        }
    }
}

/// The scores sklearn calibrates (`response_method=["decision_function",
/// "predict_proba"]`): one column for two classes (the decision function,
/// or P(class 1)), one per class otherwise.
fn calibration_scores(
    m: &RegistryFitted,
    x: ArrayView2<f64>,
    n_classes: usize,
) -> Result<Array2<f64>> {
    if has(m, "decision_function") {
        if n_classes == 2 {
            return Ok(decision_1d(m, x)?.insert_axis(Axis(1)));
        }
        return match m.model().decision_function(x)? {
            Out::A2(a) if a.ncols() == n_classes => Ok(a),
            other => Err(err(format!(
                "{}: decision_function of shape {:?} for {n_classes} classes",
                m.model().kind(),
                match other {
                    Out::A1(a) => vec![a.len()],
                    Out::A2(a) => a.shape().to_vec(),
                }
            ))),
        };
    }
    if !has(m, "predict_proba") {
        return Err(err(format!(
            "CalibratedClassifierCV: {} has neither decision_function nor predict_proba",
            m.model().kind()
        )));
    }
    let p = m.predict_proba(x)?;
    if p.ncols() != n_classes {
        return Err(err(format!(
            "{}: predict_proba has {} columns for {n_classes} classes",
            m.model().kind(),
            p.ncols()
        )));
    }
    Ok(if n_classes == 2 {
        p.slice(ndarray::s![.., 1..]).to_owned()
    } else {
        p
    })
}

/// A base model and its per-class calibrators (sklearn's
/// `_CalibratedClassifier`).
#[derive(Serialize, Deserialize)]
struct CalibratedPair {
    model: RegistryFitted,
    calibrators: Vec<Calibrator>,
}

impl CalibratedPair {
    fn fit(
        model: RegistryFitted,
        scores: &Array2<f64>,
        idx: &[usize],
        n_classes: usize,
        method: CalibrationMethod,
    ) -> CalibratedPair {
        let calibrators = (0..scores.ncols())
            .map(|c| {
                // The positive class of column c (class 1 for two classes).
                let pos = if n_classes == 2 { 1 } else { c };
                let y01: Vec<f64> = idx.iter().map(|&i| f64::from(i == pos)).collect();
                Calibrator::fit(method, &scores.column(c).to_vec(), &y01)
            })
            .collect();
        CalibratedPair { model, calibrators }
    }

    fn predict_proba(&self, x: ArrayView2<f64>, n_classes: usize) -> Result<Array2<f64>> {
        let scores = calibration_scores(&self.model, x, n_classes)?;
        let mut p = Array2::<f64>::zeros((x.nrows(), n_classes));
        for (c, cal) in self.calibrators.iter().enumerate() {
            let col = if n_classes == 2 { 1 } else { c };
            for i in 0..x.nrows() {
                p[[i, col]] = cal.predict(scores[[i, c]]);
            }
        }
        if n_classes == 2 {
            for i in 0..x.nrows() {
                p[[i, 0]] = 1.0 - p[[i, 1]];
            }
        } else {
            for mut row in p.rows_mut() {
                let total: f64 = row.sum();
                if total != 0.0 {
                    row.mapv_inplace(|v| v / total);
                } else {
                    row.fill(1.0 / n_classes as f64);
                }
            }
        }
        p.mapv_inplace(|v| if v > 1.0 && v <= 1.0 + 1e-5 { 1.0 } else { v });
        Ok(p)
    }
}

/// sklearn's `CalibratedClassifierCV` over a registered classifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct CalibratedClassifierConfig {
    pub estimator: String,
    pub estimator_params: serde_json::Value,
    pub method: CalibrationMethod,
    pub cv: usize,
    pub ensemble: bool,
}

impl Default for CalibratedClassifierConfig {
    fn default() -> Self {
        Self {
            estimator: "LinearSvc".into(),
            estimator_params: empty_params(),
            method: CalibrationMethod::Sigmoid,
            cv: 5,
            ensemble: true,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub(crate) struct CalibratedClassifierFit {
    classes: Vec<f64>,
    pairs: Vec<CalibratedPair>,
}

impl CalibratedClassifierConfig {
    pub fn fit(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<CalibratedClassifierFit> {
        check_cv("CalibratedClassifierCV", self.cv)?;
        let (classes, idx) = encode(y)?;
        if classes.len() < 2 {
            return Err(err("CalibratedClassifierCV needs at least 2 classes"));
        }
        let k = classes.len();
        let spec = EstimatorSpec {
            estimator: self.estimator.clone(),
            estimator_params: self.estimator_params.clone(),
        };
        let y_enc: Array1<f64> = idx.iter().map(|&i| i as f64).collect();
        let splits = anofox_ml_core::stratified_k_fold_ordered(&y_enc, self.cv)?;
        let pairs = if self.ensemble {
            // One calibrated model per fold, calibrated on its test rows.
            splits
                .iter()
                .map(|(train, test)| {
                    let model = spec.fit(
                        x.select(Axis(0), train).view(),
                        y.select(Axis(0), train).view(),
                    )?;
                    let scores = calibration_scores(&model, x.select(Axis(0), test).view(), k)?;
                    let test_idx: Vec<usize> = test.iter().map(|&i| idx[i]).collect();
                    Ok(CalibratedPair::fit(
                        model,
                        &scores,
                        &test_idx,
                        k,
                        self.method,
                    ))
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            // Out-of-fold scores of all rows; the model refitted on all rows.
            let mut oof: Option<Array2<f64>> = None;
            for (train, test) in &splits {
                let model = spec.fit(
                    x.select(Axis(0), train).view(),
                    y.select(Axis(0), train).view(),
                )?;
                let s = calibration_scores(&model, x.select(Axis(0), test).view(), k)?;
                let all = oof.get_or_insert_with(|| Array2::zeros((x.nrows(), s.ncols())));
                for (r, &i) in test.iter().enumerate() {
                    all.row_mut(i).assign(&s.row(r));
                }
            }
            let model = spec.fit(x, y)?;
            vec![CalibratedPair::fit(
                model,
                &oof.expect("cv >= 2 folds"),
                &idx,
                k,
                self.method,
            )]
        };
        Ok(CalibratedClassifierFit { classes, pairs })
    }
}

impl CalibratedClassifierFit {
    pub fn predict_proba(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        let k = self.classes.len();
        let mut mean = Array2::<f64>::zeros((x.nrows(), k));
        for pair in &self.pairs {
            mean = mean + pair.predict_proba(x, k)?;
        }
        Ok(mean / self.pairs.len() as f64)
    }

    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        Ok(argmax_rows(&self.predict_proba(x)?)
            .into_iter()
            .map(|c| self.classes[c])
            .collect())
    }
}
