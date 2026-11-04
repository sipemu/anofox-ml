//! The search loop: rustuna study → cross-validated registry estimator.

use anofox_ml_core::utils::{default_cv_splits, CvSplit};
use anofox_ml_core::{Result, RustMlError};
use anofox_ml_registry::{DynModel, Out};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rustuna_core::sampler::RandomSampler;
use rustuna_core::storage::InMemoryStorage;
use rustuna_core::study::{create_study, Direction, Study};
use rustuna_core::trial::TrialStateValues;
use rustuna_sampler::tpe::TpeBuilder;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::metric::Metric;
use crate::space::SearchSpace;

/// Sampler that proposes the next parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Sampler {
    /// Tree-structured Parzen estimator (Optuna's default).
    #[default]
    Tpe,
    Random,
}

/// Search settings. Deserializes from JSON with every field optional.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TuneOptions {
    /// Number of trials (parameter sets evaluated).
    pub n_trials: usize,
    pub sampler: Sampler,
    /// Random trials before TPE starts modelling (TPE only).
    pub n_startup_trials: usize,
    /// Seeds the sampler and the CV splits.
    pub seed: u64,
    /// Number of CV folds (stratified for class labels, shuffled K-fold
    /// otherwise). Ignored by [`tune_with_splits`].
    pub cv: usize,
    /// Metric name (see [`Metric`]); default: accuracy for classifiers, R²
    /// otherwise.
    pub metric: Option<String>,
    /// Fixed estimator parameters, applied to every trial.
    pub params: Value,
    /// Refit the best parameters on all rows and return the model.
    pub refit: bool,
}

impl Default for TuneOptions {
    fn default() -> Self {
        Self {
            n_trials: 50,
            sampler: Sampler::Tpe,
            n_startup_trials: 10,
            seed: 0,
            cv: 5,
            metric: None,
            params: Value::Object(Map::new()),
            refit: true,
        }
    }
}

/// One evaluated parameter set.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TrialRecord {
    pub number: usize,
    /// The searched parameters of this trial.
    pub params: Value,
    /// Mean CV score; `None` when the trial failed.
    pub score: Option<f64>,
    pub fold_scores: Vec<f64>,
    /// Why the trial failed (an invalid parameter combination, a fit error,
    /// a NaN score).
    pub error: Option<String>,
}

/// Outcome of a search.
#[derive(Serialize)]
pub struct TuneResult {
    pub kind: String,
    pub metric: String,
    pub maximize: bool,
    /// Full estimator parameters of the best trial: defaults, overridden by
    /// the fixed `params`, overridden by the searched values.
    pub best_params: Value,
    pub best_score: f64,
    pub best_trial: usize,
    pub trials: Vec<TrialRecord>,
    /// The estimator refitted with `best_params` on all rows (when
    /// `refit`). `to_bytes` gives its persisted form.
    #[serde(skip)]
    pub model: Option<Box<dyn DynModel>>,
}

impl std::fmt::Debug for TuneResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TuneResult")
            .field("kind", &self.kind)
            .field("metric", &self.metric)
            .field("best_params", &self.best_params)
            .field("best_score", &self.best_score)
            .field("best_trial", &self.best_trial)
            .field("n_trials", &self.trials.len())
            .field("model", &self.model.as_ref().map(|m| m.kind()))
            .finish()
    }
}

fn invalid(msg: impl Into<String>) -> RustMlError {
    RustMlError::InvalidParameter(msg.into())
}

fn study_err(e: rustuna_core::Error) -> RustMlError {
    RustMlError::InvalidParameter(format!("rustuna: {e}"))
}

/// Search the hyperparameters of registry estimator `kind` over `space`
/// (see [`SearchSpace`]) by `options.cv`-fold cross-validation.
///
/// ```
/// use ndarray::{Array1, Array2};
/// use serde_json::json;
///
/// let x = Array2::from_shape_fn((60, 2), |(i, j)| ((i * 7 + j * 3) % 11) as f64);
/// let y = Array1::from_shape_fn(60, |i| if x[[i, 0]] > 5.0 { 1.0 } else { 0.0 });
/// let options = anofox_ml_tune::TuneOptions { n_trials: 8, cv: 3, ..Default::default() };
/// let result = anofox_ml_tune::tune(
///     "DecisionTreeClassifier",
///     &json!({"max_depth": {"int": [1, 6]}}),
///     x.view(),
///     y.view(),
///     &options,
/// )
/// .unwrap();
/// assert_eq!(result.metric, "accuracy");
/// assert!(result.best_score > 0.9);
/// assert!(result.model.unwrap().is_fitted());
/// ```
pub fn tune(
    kind: &str,
    space: &Value,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    options: &TuneOptions,
) -> Result<TuneResult> {
    let splits = default_cv_splits(x, y, options.cv, options.seed)?;
    tune_with_splits(kind, space, x, y, None, &splits, options)
}

/// [`tune`] over caller-supplied `(train, test)` splits (group, time-series
/// or any other splitter of `anofox_ml_core`), with optional sample
/// weights passed to every fit.
pub fn tune_with_splits(
    kind: &str,
    space: &Value,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    sample_weight: Option<ArrayView1<f64>>,
    splits: &[CvSplit],
    options: &TuneOptions,
) -> Result<TuneResult> {
    let space = SearchSpace::from_json(space)?;
    let metric = match &options.metric {
        Some(name) => Metric::from_name(name)?,
        None => Metric::default_for(kind),
    };
    let fixed = check_params(kind, &space, &options.params)?;
    if options.n_trials == 0 {
        return Err(invalid("n_trials must be at least 1"));
    }
    if splits.is_empty() {
        return Err(invalid("no CV splits"));
    }
    if x.nrows() != y.len() || sample_weight.is_some_and(|w| w.len() != y.len()) {
        return Err(RustMlError::ShapeMismatch(format!(
            "x has {} rows, y {} values",
            x.nrows(),
            y.len()
        )));
    }

    let study = new_study(kind, metric, options)?;
    let mut trials = Vec::with_capacity(options.n_trials);
    for _ in 0..options.n_trials {
        let mut trial = study.ask().map_err(study_err)?;
        let number = trial.number;
        let suggested = space.suggest(&mut trial)?;
        let mut params = fixed.clone();
        params.extend(suggested.clone());
        let outcome = cv_scores(
            kind,
            &Value::Object(params),
            x,
            y,
            sample_weight,
            splits,
            metric,
        );
        let record = match outcome {
            Ok(fold_scores) => {
                let mean = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
                if mean.is_finite() {
                    TrialRecord {
                        number: number as usize,
                        params: Value::Object(suggested),
                        score: Some(mean),
                        fold_scores,
                        error: None,
                    }
                } else {
                    TrialRecord {
                        number: number as usize,
                        params: Value::Object(suggested),
                        score: None,
                        fold_scores,
                        error: Some(format!("{} is not finite", metric.name())),
                    }
                }
            }
            Err(e) => TrialRecord {
                number: number as usize,
                params: Value::Object(suggested),
                score: None,
                fold_scores: Vec::new(),
                error: Some(e.to_string()),
            },
        };
        let state = match record.score {
            Some(s) => TrialStateValues::Complete(vec![s]),
            None => TrialStateValues::Fail,
        };
        study.tell(number, state).map_err(study_err)?;
        trials.push(record);
    }

    let best = trials
        .iter()
        .filter_map(|t| t.score.map(|s| (t, s)))
        .reduce(|a, b| {
            let better = if metric.maximize() {
                b.1 > a.1
            } else {
                b.1 < a.1
            };
            if better {
                b
            } else {
                a
            }
        })
        .map(|(t, _)| t)
        .ok_or_else(|| {
            let first = trials
                .iter()
                .find_map(|t| t.error.clone())
                .unwrap_or_default();
            invalid(format!("every trial failed; first error: {first}"))
        })?;

    let mut best_params = fixed;
    best_params.extend(best.params.as_object().cloned().unwrap_or_default());
    let best_params = Value::Object(best_params);
    let model = if options.refit {
        let mut model = anofox_ml_registry::make(kind, &best_params)?;
        model.fit(x, Some(y), sample_weight)?;
        Some(model)
    } else {
        None
    };
    Ok(TuneResult {
        kind: kind.to_string(),
        metric: metric.name().to_string(),
        maximize: metric.maximize(),
        best_params,
        best_score: best.score.expect("best has a score"),
        best_trial: best.number,
        trials,
        model,
    })
}

/// Validate the fixed and searched parameter names against the estimator's
/// parameters; returns the defaults merged with the fixed parameters.
fn check_params(kind: &str, space: &SearchSpace, fixed: &Value) -> Result<Map<String, Value>> {
    let defaults = anofox_ml_registry::default_params(kind)?;
    let known = defaults
        .as_object()
        .ok_or_else(|| invalid(format!("{kind} has no parameters to search")))?;
    let fixed = match fixed {
        Value::Null => Map::new(),
        Value::Object(m) => m.clone(),
        _ => return Err(invalid("params must be a JSON object")),
    };
    for name in space.names().chain(fixed.keys().map(String::as_str)) {
        if !known.contains_key(name) {
            let mut names: Vec<&String> = known.keys().collect();
            names.sort();
            return Err(invalid(format!(
                "{kind} has no parameter '{name}' (known: {names:?})"
            )));
        }
    }
    if let Some(name) = space.names().find(|n| fixed.contains_key(*n)) {
        return Err(invalid(format!(
            "'{name}' is both fixed (params) and searched"
        )));
    }
    let mut params = known.clone();
    params.extend(fixed);
    Ok(params)
}

fn new_study(kind: &str, metric: Metric, options: &TuneOptions) -> Result<Study> {
    let direction = if metric.maximize() {
        Direction::Maximize
    } else {
        Direction::Minimize
    };
    let name = format!("{kind}-{}", metric.name());
    match options.sampler {
        Sampler::Tpe => create_study(
            &name,
            InMemoryStorage::new(),
            TpeBuilder::new()
                .n_startup_trials(options.n_startup_trials)
                .seed(options.seed)
                .build(),
            vec![direction],
        ),
        Sampler::Random => create_study(
            &name,
            InMemoryStorage::new(),
            RandomSampler::seed_from_u64(options.seed),
            vec![direction],
        ),
    }
    .map_err(study_err)
}

/// Fit and score `kind` with `params` on every split.
fn cv_scores(
    kind: &str,
    params: &Value,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    sample_weight: Option<ArrayView1<f64>>,
    splits: &[CvSplit],
    metric: Metric,
) -> Result<Vec<f64>> {
    splits
        .iter()
        .map(|(train, test)| {
            let x_train = x.select(Axis(0), train);
            let y_train = y.select(Axis(0), train);
            let w_train = sample_weight.map(|w| w.select(Axis(0), train));
            let x_test = x.select(Axis(0), test);
            let y_test = y.select(Axis(0), test);
            let mut model = anofox_ml_registry::make(kind, params)?;
            model.fit(
                x_train.view(),
                Some(y_train.view()),
                w_train.as_ref().map(Array1::view),
            )?;
            if metric.needs_proba() {
                let proba = as_2d(model.predict_proba(x_test.view())?);
                let mut classes = y_train.to_vec();
                classes.sort_by(f64::total_cmp);
                classes.dedup();
                metric.score_proba(&y_test, &proba, &classes)
            } else {
                let pred = as_1d(model.predict(x_test.view())?)?;
                metric.score(&y_test, &pred)
            }
        })
        .collect()
}

fn as_1d(out: Out) -> Result<Array1<f64>> {
    match out {
        Out::A1(a) => Ok(a),
        Out::A2(a) if a.ncols() == 1 => Ok(a.column(0).to_owned()),
        Out::A2(a) => Err(RustMlError::ShapeMismatch(format!(
            "predict returned {} columns",
            a.ncols()
        ))),
    }
}

fn as_2d(out: Out) -> Array2<f64> {
    match out {
        Out::A2(a) => a,
        Out::A1(a) => a.insert_axis(Axis(1)),
    }
}
