//! The search loop: rustuna study → cross-validated registry estimator.

use anofox_ml_core::utils::{default_cv_splits, CvSplit};
use anofox_ml_core::{Result, RustMlError};
use anofox_ml_registry::{DynModel, Out};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rustuna_core::sampler::RandomSampler;
use rustuna_core::storage::InMemoryStorage;
use rustuna_core::study::{create_study, get_pareto_front, Direction, Study};
use rustuna_core::trial::TrialStateValues;
use rustuna_sampler::nsgaii::NsgaiiBuilder;
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
    /// NSGA-II genetic algorithm, for multi-objective searches.
    Nsgaii,
}

/// What a trial is scored on: a [`Metric`], the mean fit time per fold
/// (seconds, `fit_time`) or the mean size of the persisted fold models
/// (bytes, `model_size`). Time and size are minimised.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Objective {
    Metric(Metric),
    FitTime,
    ModelSize,
}

impl Objective {
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "fit_time" => Ok(Objective::FitTime),
            "model_size" => Ok(Objective::ModelSize),
            _ => Metric::from_name(name)
                .map(Objective::Metric)
                .map_err(|e| invalid(format!("{e}, or fit_time / model_size"))),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Objective::Metric(m) => m.name(),
            Objective::FitTime => "fit_time",
            Objective::ModelSize => "model_size",
        }
    }

    pub fn maximize(self) -> bool {
        matches!(self, Objective::Metric(m) if m.maximize())
    }
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
    /// Further objectives for a multi-objective search: metric names,
    /// `fit_time` or `model_size` (see [`Objective`]). The result then
    /// reports the Pareto front; `best_*` still refer to `metric`.
    pub objectives: Vec<String>,
    /// Population size of the NSGA-II sampler.
    pub population_size: usize,
    /// Fixed estimator parameters, applied to every trial.
    pub params: Value,
    /// Refit the best parameters on all rows and return the model.
    pub refit: bool,
    /// Weight the test-fold scores by the sample weights too (scikit-learn
    /// >= 1.8's search scorers); otherwise the weights only go to the fits.
    pub weighted_scoring: bool,
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
            objectives: Vec::new(),
            population_size: 50,
            params: Value::Object(Map::new()),
            refit: true,
            weighted_scoring: false,
        }
    }
}

/// One evaluated parameter set.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TrialRecord {
    pub number: usize,
    /// The searched parameters of this trial.
    pub params: Value,
    /// Mean CV score of `metric`; `None` when the trial failed.
    pub score: Option<f64>,
    /// `metric` per fold.
    pub fold_scores: Vec<f64>,
    /// Mean of every objective, `metric` first, then `objectives`; empty
    /// when the trial failed.
    pub values: Vec<f64>,
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
    /// Every objective with its direction: `metric` first.
    pub objectives: Vec<(String, bool)>,
    /// Trials no other trial beats on every objective (only the best trial
    /// for a single objective).
    pub pareto_front: Vec<usize>,
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
    run(kind, space, x, y, sample_weight, splits, options, None)
}

/// Translates the suggested values of a trial into estimator parameters.
pub type ParamMap<'a> = &'a dyn Fn(&Map<String, Value>) -> Result<Map<String, Value>>;

/// [`tune_with_splits`] for a search space in another vocabulary than the
/// estimator's parameters (e.g. the scikit-learn names of the Python
/// wrappers): `map_params` turns each trial's suggested values into
/// estimator parameters, which override the defaults and `options.params`.
/// The space's names are not checked against the estimator; an unknown
/// mapped parameter fails the trial. `TrialRecord::params` keep the
/// suggested values, `best_params` the mapped ones.
#[allow(clippy::too_many_arguments)]
pub fn tune_with_splits_mapped(
    kind: &str,
    space: &Value,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    sample_weight: Option<ArrayView1<f64>>,
    splits: &[CvSplit],
    options: &TuneOptions,
    map_params: ParamMap<'_>,
) -> Result<TuneResult> {
    run(
        kind,
        space,
        x,
        y,
        sample_weight,
        splits,
        options,
        Some(map_params),
    )
}

#[allow(clippy::too_many_arguments)]
fn run(
    kind: &str,
    space: &Value,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    sample_weight: Option<ArrayView1<f64>>,
    splits: &[CvSplit],
    options: &TuneOptions,
    map_params: Option<ParamMap<'_>>,
) -> Result<TuneResult> {
    let space = SearchSpace::from_json(space)?;
    let (metric, objectives) = objectives_of(kind, options)?;
    let fixed = check_params(
        kind,
        map_params.is_none().then_some(&space),
        &options.params,
    )?;
    let mapped = |suggested: &Map<String, Value>| -> Result<Map<String, Value>> {
        let mut params = fixed.clone();
        params.extend(match map_params {
            Some(map) => map(suggested)?,
            None => suggested.clone(),
        });
        Ok(params)
    };
    if options.n_trials == 0 {
        return Err(invalid("n_trials must be at least 1"));
    }
    check_data(x, y, sample_weight, splits)?;

    let study = new_study(kind, &objectives, options)?;
    let mut trials = Vec::with_capacity(options.n_trials);
    for _ in 0..options.n_trials {
        let mut trial = study.ask().map_err(study_err)?;
        let number = trial.number;
        let suggested = space.suggest(&mut trial)?;
        let record = trial_record(
            number as usize,
            suggested,
            &mapped,
            kind,
            x,
            y,
            sample_weight,
            splits,
            &objectives,
            options.weighted_scoring,
        );
        let state = if record.values.is_empty() {
            TrialStateValues::Fail
        } else {
            TrialStateValues::Complete(record.values.clone())
        };
        study.tell(number, state).map_err(study_err)?;
        trials.push(record);
    }

    if trials.iter().all(|t| t.score.is_none()) {
        let first = trials
            .iter()
            .find_map(|t| t.error.clone())
            .unwrap_or_default();
        return Err(invalid(format!("every trial failed; first error: {first}")));
    }
    // Candidates for the best trial: every completed trial, or the Pareto
    // front of a multi-objective search (so a trial that ties on the metric
    // but loses on another objective is never "best").
    let candidates: Vec<usize> = if objectives.len() == 1 {
        trials
            .iter()
            .filter(|t| t.score.is_some())
            .map(|t| t.number)
            .collect()
    } else {
        let mut front: Vec<usize> = get_pareto_front(&study)
            .map_err(study_err)?
            .into_iter()
            .map(|n| n as usize)
            .collect();
        front.sort_unstable();
        front
    };
    let score = |n: usize| trials[n].score.expect("completed trial");
    let best_number = candidates
        .iter()
        .copied()
        .reduce(|a, b| {
            let better = if metric.maximize() {
                score(b) > score(a)
            } else {
                score(b) < score(a)
            };
            if better {
                b
            } else {
                a
            }
        })
        .expect("at least one completed trial");
    let best = &trials[best_number];
    let pareto_front = if objectives.len() == 1 {
        vec![best_number]
    } else {
        candidates
    };

    let best_params = Value::Object(mapped(best.params.as_object().expect("an object"))?);
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
        objectives: objectives
            .iter()
            .map(|o| (o.name().to_string(), o.maximize()))
            .collect(),
        pareto_front,
        best_params,
        best_score: best.score.expect("best has a score"),
        best_trial: best.number,
        trials,
        model,
    })
}

/// The metric and every objective of a search, the metric first.
fn objectives_of(kind: &str, options: &TuneOptions) -> Result<(Metric, Vec<Objective>)> {
    let metric = match &options.metric {
        Some(name) => Metric::from_name(name)?,
        None => Metric::default_for(kind),
    };
    let mut objectives = vec![Objective::Metric(metric)];
    for name in &options.objectives {
        let o = Objective::from_name(name)?;
        if objectives.contains(&o) {
            return Err(invalid(format!("objective '{name}' is listed twice")));
        }
        objectives.push(o);
    }
    Ok((metric, objectives))
}

fn check_data(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    sample_weight: Option<ArrayView1<f64>>,
    splits: &[CvSplit],
) -> Result<()> {
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
    Ok(())
}

/// Cross-validate one parameter set: the record of trial `number`, with
/// its score or why it failed.
#[allow(clippy::too_many_arguments)]
fn trial_record(
    number: usize,
    suggested: Map<String, Value>,
    mapped: ParamMap<'_>,
    kind: &str,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    sample_weight: Option<ArrayView1<f64>>,
    splits: &[CvSplit],
    objectives: &[Objective],
    weighted_scoring: bool,
) -> TrialRecord {
    let outcome = mapped(&suggested).and_then(|params| {
        cv_evaluate(
            kind,
            &Value::Object(params),
            x,
            y,
            sample_weight,
            splits,
            objectives,
            weighted_scoring,
        )
    });
    let mut record = TrialRecord {
        number,
        params: Value::Object(suggested),
        score: None,
        fold_scores: Vec::new(),
        values: Vec::new(),
        error: None,
    };
    match outcome {
        Ok(folds) => {
            let n = folds.len() as f64;
            let means: Vec<f64> = (0..objectives.len())
                .map(|k| folds.iter().map(|f| f[k]).sum::<f64>() / n)
                .collect();
            record.fold_scores = folds.iter().map(|f| f[0]).collect();
            match means.iter().position(|v| !v.is_finite()) {
                None => {
                    record.score = Some(means[0]);
                    record.values = means;
                }
                Some(k) => {
                    record.error = Some(format!("{} is not finite", objectives[k].name()));
                }
            }
        }
        Err(e) => record.error = Some(e.to_string()),
    }
    record
}

/// Cross-validate fixed parameter sets in order (the candidates of
/// scikit-learn's `GridSearchCV` / `RandomizedSearchCV`), one
/// [`TrialRecord`] each, scored on `options.metric` and
/// `options.objectives`; `options.params` are fixed for every candidate.
/// `map_params`, as in [`tune_with_splits_mapped`], turns each candidate
/// into estimator parameters (the candidates' names are then not checked).
/// The sampler options, `n_trials` and `refit` are not used: choosing and
/// refitting the best candidate is the caller's.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_candidates(
    kind: &str,
    candidates: &[Map<String, Value>],
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    sample_weight: Option<ArrayView1<f64>>,
    splits: &[CvSplit],
    options: &TuneOptions,
    map_params: Option<ParamMap<'_>>,
) -> Result<Vec<TrialRecord>> {
    if candidates.is_empty() {
        return Err(invalid("no candidates to evaluate"));
    }
    let (_, objectives) = objectives_of(kind, options)?;
    let fixed = check_params(kind, None, &options.params)?;
    if map_params.is_none() {
        if let Some(name) = candidates
            .iter()
            .flat_map(Map::keys)
            .find(|n| !fixed.contains_key(*n))
        {
            return Err(invalid(format!("{kind} has no parameter '{name}'")));
        }
    }
    check_data(x, y, sample_weight, splits)?;
    let mapped = |suggested: &Map<String, Value>| -> Result<Map<String, Value>> {
        let mut params = fixed.clone();
        params.extend(match map_params {
            Some(map) => map(suggested)?,
            None => suggested.clone(),
        });
        Ok(params)
    };
    Ok(candidates
        .iter()
        .enumerate()
        .map(|(number, c)| {
            trial_record(
                number,
                c.clone(),
                &mapped,
                kind,
                x,
                y,
                sample_weight,
                splits,
                &objectives,
                options.weighted_scoring,
            )
        })
        .collect())
}

/// Validate the fixed and (when given) searched parameter names against the
/// estimator's parameters; returns the defaults merged with the fixed
/// parameters.
fn check_params(
    kind: &str,
    space: Option<&SearchSpace>,
    fixed: &Value,
) -> Result<Map<String, Value>> {
    let defaults = anofox_ml_registry::default_params(kind)?;
    let known = defaults
        .as_object()
        .ok_or_else(|| invalid(format!("{kind} has no parameters to search")))?;
    let fixed = match fixed {
        Value::Null => Map::new(),
        Value::Object(m) => m.clone(),
        _ => return Err(invalid("params must be a JSON object")),
    };
    let searched = space.into_iter().flat_map(SearchSpace::names);
    for name in searched.chain(fixed.keys().map(String::as_str)) {
        if !known.contains_key(name) {
            let mut names: Vec<&String> = known.keys().collect();
            names.sort();
            return Err(invalid(format!(
                "{kind} has no parameter '{name}' (known: {names:?})"
            )));
        }
    }
    if let Some(name) = space
        .into_iter()
        .flat_map(SearchSpace::names)
        .find(|n| fixed.contains_key(*n))
    {
        return Err(invalid(format!(
            "'{name}' is both fixed (params) and searched"
        )));
    }
    let mut params = known.clone();
    params.extend(fixed);
    Ok(params)
}

fn new_study(kind: &str, objectives: &[Objective], options: &TuneOptions) -> Result<Study> {
    let directions: Vec<Direction> = objectives
        .iter()
        .map(|o| {
            if o.maximize() {
                Direction::Maximize
            } else {
                Direction::Minimize
            }
        })
        .collect();
    let names: Vec<&str> = objectives.iter().map(|o| o.name()).collect();
    let name = format!("{kind}-{}", names.join("-"));
    match options.sampler {
        Sampler::Tpe => create_study(
            &name,
            InMemoryStorage::new(),
            TpeBuilder::new()
                .n_startup_trials(options.n_startup_trials)
                .seed(options.seed)
                .build(),
            directions,
        ),
        Sampler::Random => create_study(
            &name,
            InMemoryStorage::new(),
            RandomSampler::seed_from_u64(options.seed),
            directions,
        ),
        Sampler::Nsgaii => {
            if options.population_size < 2 {
                return Err(invalid("population_size must be at least 2"));
            }
            create_study(
                &name,
                InMemoryStorage::new(),
                NsgaiiBuilder::new()
                    .population_size(options.population_size)
                    .seed(options.seed)
                    .build(),
                directions,
            )
        }
    }
    .map_err(study_err)
}

/// Fit `kind` with `params` on every split and evaluate every objective
/// (weighted by the test rows' sample weights with `weighted_scoring`);
/// returns one row of objective values per split.
#[allow(clippy::too_many_arguments)]
fn cv_evaluate(
    kind: &str,
    params: &Value,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    sample_weight: Option<ArrayView1<f64>>,
    splits: &[CvSplit],
    objectives: &[Objective],
    weighted_scoring: bool,
) -> Result<Vec<Vec<f64>>> {
    let metrics = || {
        objectives.iter().filter_map(|o| match o {
            Objective::Metric(m) => Some(*m),
            _ => None,
        })
    };
    let need_pred = metrics().any(|m| !m.needs_proba());
    let need_proba = metrics().any(Metric::needs_proba);
    splits
        .iter()
        .map(|(train, test)| {
            let x_train = x.select(Axis(0), train);
            let y_train = y.select(Axis(0), train);
            let w_train = sample_weight.map(|w| w.select(Axis(0), train));
            let x_test = x.select(Axis(0), test);
            let y_test = y.select(Axis(0), test);
            let w_test = sample_weight
                .filter(|_| weighted_scoring)
                .map(|w| w.select(Axis(0), test));
            let mut model = anofox_ml_registry::make(kind, params)?;
            let start = std::time::Instant::now();
            model.fit(
                x_train.view(),
                Some(y_train.view()),
                w_train.as_ref().map(Array1::view),
            )?;
            let fit_time = start.elapsed().as_secs_f64();
            let pred = if need_pred {
                Some(as_1d(model.predict(x_test.view())?)?)
            } else {
                None
            };
            let proba = if need_proba {
                Some(as_2d(model.predict_proba(x_test.view())?))
            } else {
                None
            };
            let mut classes = y_train.to_vec();
            classes.sort_by(f64::total_cmp);
            classes.dedup();
            objectives
                .iter()
                .map(|o| match o {
                    Objective::Metric(m) if m.needs_proba() => m.score_proba(
                        &y_test,
                        proba.as_ref().expect("computed"),
                        &classes,
                        w_test.as_ref(),
                    ),
                    Objective::Metric(m) => {
                        m.score(&y_test, pred.as_ref().expect("computed"), w_test.as_ref())
                    }
                    Objective::FitTime => Ok(fit_time),
                    Objective::ModelSize => Ok(model.to_bytes()?.len() as f64),
                })
                .collect()
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
