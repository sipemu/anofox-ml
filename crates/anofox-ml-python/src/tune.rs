//! `_native.tune`: hyperparameter search of `anofox-ml-tune`.

use anofox_ml_core::utils::{default_cv_splits, CvSplit};
use anofox_ml_core::RustMlError;
use anofox_ml_tune::TuneOptions;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::{Map, Value};

use crate::native::{to_py_err, PyModel};

/// `map_params(suggested_json) -> params_json` (see [`tune`]).
fn call_map(
    map_params: Option<&Py<PyAny>>,
    suggested: &Map<String, Value>,
) -> anofox_ml_core::Result<Map<String, Value>> {
    let callback = map_params.expect("only called with a callback");
    let out = Python::attach(|py| -> PyResult<String> {
        callback
            .call1(py, (Value::Object(suggested.clone()).to_string(),))?
            .extract::<String>(py)
    })
    .map_err(|e| RustMlError::InvalidParameter(e.to_string()))?;
    match serde_json::from_str(&out) {
        Ok(Value::Object(m)) => Ok(m),
        _ => Err(RustMlError::InvalidParameter(
            "map_params must return a JSON object".into(),
        )),
    }
}

/// Search the parameters of registry estimator `kind`.
///
/// `space_json` / `options_json` are the JSON forms of
/// `anofox_ml_tune::SearchSpace` / `TuneOptions`. `splits` are
/// `(train, test)` index lists (default: `options.cv` folds). `map_params`,
/// when given, is called with each trial's suggested values as JSON and
/// returns the estimator parameters as JSON; an exception fails that trial.
/// Returns the result as JSON and the refitted model (when `refit`).
#[pyfunction]
#[pyo3(signature = (kind, space_json, x, y, options_json=None, sample_weight=None, splits=None, map_params=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn tune(
    py: Python<'_>,
    kind: &str,
    space_json: &str,
    x: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    options_json: Option<&str>,
    sample_weight: Option<PyReadonlyArray1<'_, f64>>,
    splits: Option<Vec<CvSplit>>,
    map_params: Option<Py<PyAny>>,
) -> PyResult<(String, Option<PyModel>)> {
    let json_err = |what: &str, e: serde_json::Error| PyValueError::new_err(format!("{what}: {e}"));
    let space: Value = serde_json::from_str(space_json).map_err(|e| json_err("search space", e))?;
    let options: TuneOptions = match options_json {
        Some(s) => serde_json::from_str(s).map_err(|e| json_err("options", e))?,
        None => TuneOptions::default(),
    };
    let (x, y) = (x.as_array(), y.as_array());
    let w = sample_weight.as_ref().map(|w| w.as_array());
    let splits = match splits {
        Some(s) => s,
        None => default_cv_splits(x, y, options.cv, options.seed).map_err(to_py_err)?,
    };
    let map = |suggested: &Map<String, Value>| call_map(map_params.as_ref(), suggested);
    let result = py
        .detach(|| match map_params {
            Some(_) => anofox_ml_tune::tune_with_splits_mapped(
                kind, &space, x, y, w, &splits, &options, &map,
            ),
            None => anofox_ml_tune::tune_with_splits(kind, &space, x, y, w, &splits, &options),
        })
        .map_err(to_py_err)?;
    let json = serde_json::to_string(&result)
        .map_err(|e| PyValueError::new_err(format!("result: {e}")))?;
    Ok((json, result.model.map(PyModel::from_inner)))
}

/// Cross-validate the parameter sets `candidates_json` (a JSON list of
/// objects) of registry estimator `kind` over `splits`, in order: the
/// candidates of `GridSearchCV` / `RandomizedSearchCV`. `options_json` and
/// `map_params` are as in [`tune`] (only the metric, objectives and fixed
/// params of the options are used). Returns the trial records as JSON.
#[pyfunction]
#[pyo3(signature = (kind, candidates_json, x, y, splits, options_json=None, sample_weight=None, map_params=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_candidates(
    py: Python<'_>,
    kind: &str,
    candidates_json: &str,
    x: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    splits: Vec<CvSplit>,
    options_json: Option<&str>,
    sample_weight: Option<PyReadonlyArray1<'_, f64>>,
    map_params: Option<Py<PyAny>>,
) -> PyResult<String> {
    let json_err = |what: &str, e: serde_json::Error| PyValueError::new_err(format!("{what}: {e}"));
    let candidates: Vec<Map<String, Value>> =
        serde_json::from_str(candidates_json).map_err(|e| json_err("candidates", e))?;
    let options: TuneOptions = match options_json {
        Some(s) => serde_json::from_str(s).map_err(|e| json_err("options", e))?,
        None => TuneOptions::default(),
    };
    let (x, y) = (x.as_array(), y.as_array());
    let w = sample_weight.as_ref().map(|w| w.as_array());
    let map = |suggested: &Map<String, Value>| call_map(map_params.as_ref(), suggested);
    let records = py
        .detach(|| {
            let map_ref: Option<anofox_ml_tune::ParamMap<'_>> = match map_params {
                Some(_) => Some(&map),
                None => None,
            };
            anofox_ml_tune::evaluate_candidates(
                kind,
                &candidates,
                x,
                y,
                w,
                &splits,
                &options,
                map_ref,
            )
        })
        .map_err(to_py_err)?;
    serde_json::to_string(&records).map_err(|e| PyValueError::new_err(format!("records: {e}")))
}
