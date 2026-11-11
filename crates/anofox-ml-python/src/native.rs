//! `_native.Model`: the single Python class behind every estimator wrapper.

use ndarray::ArrayView2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyAttributeError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use anofox_ml_core::RustMlError;

use anofox_ml_registry::{self as registry, DynModel, Out};

/// Map a Rust error to the closest Python exception: `NotFitted` →
/// `RuntimeError` (the Python layer re-raises sklearn's `NotFittedError`),
/// unsupported or unavailable methods (e.g. `predict_proba` without
/// `probability=true`) → `AttributeError`, as sklearn, everything else →
/// `ValueError`.
pub(crate) fn to_py_err(e: RustMlError) -> PyErr {
    match e {
        RustMlError::NotFitted(m) => PyRuntimeError::new_err(format!("NotFitted: {m}")),
        RustMlError::InvalidParameter(m)
            if m.contains(" does not support ")
                || m.contains("is only available")
                || m.contains("is not available") =>
        {
            PyAttributeError::new_err(m)
        }
        other => PyValueError::new_err(other.to_string()),
    }
}

fn out_to_py(py: Python<'_>, out: Out) -> Py<PyAny> {
    match out {
        Out::A1(a) => PyArray1::from_owned_array(py, a).into_any().unbind(),
        Out::A2(a) => PyArray2::from_owned_array(py, a).into_any().unbind(),
    }
}

fn parse_params(params_json: Option<&str>) -> PyResult<serde_json::Value> {
    match params_json {
        None => Ok(serde_json::Value::Null),
        Some(s) => serde_json::from_str(s)
            .map_err(|e| PyValueError::new_err(format!("parameters are not valid JSON: {e}"))),
    }
}

/// An estimator from the Rust registry: its configuration and, after `fit`,
/// its fitted state. Compute-heavy calls release the GIL.
#[pyclass(name = "Model", module = "anofox_ml_python._native")]
pub struct PyModel {
    inner: Box<dyn DynModel>,
}

#[pymethods]
impl PyModel {
    /// Create an unfitted estimator of `kind`; `params_json` holds the
    /// overrides of the default configuration (Rust field names).
    #[new]
    #[pyo3(signature = (kind, params_json=None))]
    fn new(kind: &str, params_json: Option<&str>) -> PyResult<Self> {
        let params = parse_params(params_json)?;
        Ok(Self {
            inner: registry::make(kind, &params).map_err(to_py_err)?,
        })
    }

    /// Restore a fitted estimator from [`PyModel::to_bytes`].
    #[staticmethod]
    fn from_bytes(kind: &str, data: &[u8]) -> PyResult<Self> {
        Ok(Self {
            inner: registry::load(kind, data).map_err(to_py_err)?,
        })
    }

    /// Default configuration of `kind` as JSON.
    #[staticmethod]
    fn default_params(kind: &str) -> PyResult<String> {
        Ok(registry::default_params(kind)
            .map_err(to_py_err)?
            .to_string())
    }

    /// Names of all registered estimators.
    #[staticmethod]
    fn kinds() -> Vec<&'static str> {
        registry::kinds()
    }

    #[getter]
    fn kind(&self) -> &'static str {
        self.inner.kind()
    }

    /// Methods this estimator supports besides `fit`.
    fn capabilities(&self) -> Vec<&'static str> {
        self.inner.capabilities()
    }

    /// Current configuration as JSON.
    fn params_json(&self) -> PyResult<String> {
        Ok(self.inner.params().map_err(to_py_err)?.to_string())
    }

    fn is_fitted(&self) -> bool {
        self.inner.is_fitted()
    }

    /// Fitted state as JSON (for fitted attributes such as coefficients).
    fn fitted_json(&self) -> PyResult<String> {
        self.inner.fitted_json().map_err(to_py_err)
    }

    /// Configuration and fitted state in the versioned binary model format.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.inner.to_bytes().map_err(to_py_err)?;
        Ok(PyBytes::new(py, &bytes))
    }

    #[pyo3(signature = (x, y=None, sample_weight=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: Option<PyReadonlyArray1<'_, f64>>,
        sample_weight: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<()> {
        let (x, y, w) = (
            x.as_array(),
            y.as_ref().map(|y| y.as_array()),
            sample_weight.as_ref().map(|w| w.as_array()),
        );
        let inner = &mut self.inner;
        py.detach(|| inner.fit(x, y, w)).map_err(to_py_err)
    }

    fn predict(&self, py: Python<'_>, x: PyReadonlyArray2<'_, f64>) -> PyResult<Py<PyAny>> {
        self.call(py, x, |m, x| m.predict(x))
    }

    fn predict_proba(&self, py: Python<'_>, x: PyReadonlyArray2<'_, f64>) -> PyResult<Py<PyAny>> {
        self.call(py, x, |m, x| m.predict_proba(x))
    }

    /// Class log-probabilities: the estimator's own where it has them (naive
    /// Bayes stays finite where `predict_proba` underflows).
    fn predict_log_proba(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Py<PyAny>> {
        self.call(py, x, |m, x| m.predict_log_proba(x))
    }

    /// Standard deviation of the predictive distribution (Gaussian process).
    fn predict_std(&self, py: Python<'_>, x: PyReadonlyArray2<'_, f64>) -> PyResult<Py<PyAny>> {
        self.call(py, x, |m, x| m.predict_std(x))
    }

    fn decision_function(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Py<PyAny>> {
        self.call(py, x, |m, x| m.decision_function(x))
    }

    /// Scores of the second block (the Y scores of Cca / PLS).
    fn transform_y(&self, py: Python<'_>, y: PyReadonlyArray2<'_, f64>) -> PyResult<Py<PyAny>> {
        self.call(py, y, |m, y| m.transform_y(y))
    }

    /// Distances and training-row indices of the `k` nearest training rows.
    #[pyo3(signature = (x, k=None))]
    fn kneighbors(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        k: Option<usize>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let x = x.as_array();
        let inner: &dyn DynModel = self.inner.as_ref();
        let (dist, idx) = py.detach(|| inner.kneighbors(x, k)).map_err(to_py_err)?;
        let idx = idx.mapv(|i| i as i64);
        Ok((
            PyArray2::from_owned_array(py, dist).into_any().unbind(),
            PyArray2::from_owned_array(py, idx).into_any().unbind(),
        ))
    }

    /// Coefficients of the fitted linear model.
    fn coef(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let out = self.inner.coef().map_err(to_py_err)?;
        Ok(out_to_py(py, out))
    }

    /// Fold one batch into the model (sklearn's `partial_fit`).
    #[pyo3(signature = (x, y=None, sample_weight=None, classes=None))]
    fn partial_fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: Option<PyReadonlyArray1<'_, f64>>,
        sample_weight: Option<PyReadonlyArray1<'_, f64>>,
        classes: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let (x, y, w) = (
            x.as_array(),
            y.as_ref().map(|y| y.as_array()),
            sample_weight.as_ref().map(|w| w.as_array()),
        );
        let inner = &mut self.inner;
        py.detach(|| inner.partial_fit(x, y, w, classes.as_deref()))
            .map_err(to_py_err)
    }

    /// Fit a text vectorizer on documents (one string each).
    fn fit_text(&mut self, py: Python<'_>, docs: Vec<String>) -> PyResult<()> {
        let inner = &mut self.inner;
        py.detach(|| {
            let refs: Vec<&str> = docs.iter().map(String::as_str).collect();
            inner.fit_text(&refs)
        })
        .map_err(to_py_err)
    }

    /// Encode documents with a fitted text vectorizer (dense).
    fn transform_text(&self, py: Python<'_>, docs: Vec<String>) -> PyResult<Py<PyAny>> {
        let inner: &dyn DynModel = self.inner.as_ref();
        let out = py
            .detach(|| {
                let refs: Vec<&str> = docs.iter().map(String::as_str).collect();
                inner.transform_text(&refs)
            })
            .map_err(to_py_err)?;
        Ok(out_to_py(py, out))
    }

    /// Fit with an offset on the link scale (the GLMs).
    #[pyo3(signature = (x, y, offset, sample_weight=None))]
    fn fit_with_offset(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        offset: PyReadonlyArray1<'_, f64>,
        sample_weight: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<()> {
        let (x, y, o, w) = (
            x.as_array(),
            y.as_array(),
            offset.as_array(),
            sample_weight.as_ref().map(|w| w.as_array()),
        );
        let inner = &mut self.inner;
        py.detach(|| inner.fit_with_offset(x, Some(y), w, o))
            .map_err(to_py_err)
    }

    /// Fit with query groups (the ranking objectives).
    #[pyo3(signature = (x, y, group, sample_weight=None))]
    fn fit_with_group(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        group: Vec<usize>,
        sample_weight: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<()> {
        let (x, y, w) = (
            x.as_array(),
            y.as_array(),
            sample_weight.as_ref().map(|w| w.as_array()),
        );
        let inner = &mut self.inner;
        py.detach(|| inner.fit_with_group(x, Some(y), w, &group))
            .map_err(to_py_err)
    }

    /// Fit with text columns besides `x` (CatBoost's `text_features`):
    /// `text[c][r]` is row `r` of text column `c`.
    #[pyo3(signature = (x, text, y, sample_weight=None))]
    fn fit_with_text(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        text: Vec<Vec<String>>,
        y: PyReadonlyArray1<'_, f64>,
        sample_weight: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<()> {
        let (x, y, w) = (
            x.as_array(),
            y.as_array(),
            sample_weight.as_ref().map(|w| w.as_array()),
        );
        let inner = &mut self.inner;
        py.detach(|| inner.fit_with_text(x, &text, Some(y), w))
            .map_err(to_py_err)
    }

    /// Predictions of a model fitted with `fit_with_text`.
    fn predict_with_text(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        text: Vec<Vec<String>>,
    ) -> PyResult<Py<PyAny>> {
        self.call(py, x, |m, x| m.predict_with_text(x, &text))
    }

    /// Class probabilities of a model fitted with `fit_with_text`.
    fn predict_proba_with_text(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        text: Vec<Vec<String>>,
    ) -> PyResult<Py<PyAny>> {
        self.call(py, x, |m, x| m.predict_proba_with_text(x, &text))
    }

    /// Fit with embedding columns besides `x` (CatBoost's
    /// `embedding_features`): `embeddings[c]` holds one vector per row.
    #[pyo3(signature = (x, embeddings, y, sample_weight=None))]
    fn fit_with_embeddings(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        embeddings: Vec<PyReadonlyArray2<'_, f64>>,
        y: PyReadonlyArray1<'_, f64>,
        sample_weight: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<()> {
        let (x, y, w) = (
            x.as_array(),
            y.as_array(),
            sample_weight.as_ref().map(|w| w.as_array()),
        );
        let e: Vec<_> = embeddings.iter().map(|e| e.as_array()).collect();
        let inner = &mut self.inner;
        py.detach(|| inner.fit_with_embeddings(x, &e, Some(y), w))
            .map_err(to_py_err)
    }

    /// Predictions of a model fitted with `fit_with_embeddings`.
    fn predict_with_embeddings(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        embeddings: Vec<PyReadonlyArray2<'_, f64>>,
    ) -> PyResult<Py<PyAny>> {
        let e: Vec<_> = embeddings.iter().map(|e| e.as_array()).collect();
        self.call(py, x, |m, x| m.predict_with_embeddings(x, &e))
    }

    /// Class probabilities of a model fitted with `fit_with_embeddings`.
    fn predict_proba_with_embeddings(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        embeddings: Vec<PyReadonlyArray2<'_, f64>>,
    ) -> PyResult<Py<PyAny>> {
        let e: Vec<_> = embeddings.iter().map(|e| e.as_array()).collect();
        self.call(py, x, |m, x| m.predict_proba_with_embeddings(x, &e))
    }

    /// Predictions with an offset on the link scale (the GLMs).
    fn predict_with_offset(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        offset: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Py<PyAny>> {
        let o = offset.as_array();
        self.call(py, x, |m, x| m.predict_with_offset(x, o))
    }

    fn transform(&self, py: Python<'_>, x: PyReadonlyArray2<'_, f64>) -> PyResult<Py<PyAny>> {
        self.call(py, x, |m, x| m.transform(x))
    }

    fn inverse_transform(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Py<PyAny>> {
        self.call(py, x, |m, x| m.inverse_transform(x))
    }

    fn score_samples(&self, py: Python<'_>, x: PyReadonlyArray2<'_, f64>) -> PyResult<Py<PyAny>> {
        self.call(py, x, |m, x| m.score_samples(x))
    }

    /// Conformal prediction intervals at miscoverage `alpha`: `n × 2`
    /// (lower, upper).
    fn predict_interval(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        alpha: f64,
    ) -> PyResult<Py<PyAny>> {
        self.call(py, x, |m, x| m.predict_interval(x, alpha))
    }

    /// Conformal prediction sets at miscoverage `alpha`: `n × classes`, 1
    /// for the classes in each row's set.
    fn predict_set(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        alpha: f64,
    ) -> PyResult<Py<PyAny>> {
        self.call(py, x, |m, x| m.predict_set(x, alpha))
    }

    /// Per-sample result of the training fit (cluster / outlier labels).
    fn labels(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let out = self.inner.labels().map_err(to_py_err)?;
        Ok(out_to_py(py, out))
    }

    /// Embedding of the training data (manifold learners).
    fn embedding(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let out = self.inner.embedding().map_err(to_py_err)?;
        Ok(out_to_py(py, out))
    }

    /// Per-feature importances of the fitted model.
    fn feature_importances(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let out = self.inner.feature_importances().map_err(to_py_err)?;
        Ok(out_to_py(py, out))
    }

    /// Predictions after each boosting stage: `n × stages`.
    fn staged_predict(&self, py: Python<'_>, x: PyReadonlyArray2<'_, f64>) -> PyResult<Py<PyAny>> {
        self.call(py, x, |m, x| m.staged_predict(x))
    }

    /// Merge JSON overrides into the configuration, keeping a fitted state.
    fn set_params_json(&mut self, params_json: &str) -> PyResult<()> {
        let params = parse_params(Some(params_json))?;
        self.inner.set_params(&params).map_err(to_py_err)
    }

    /// Continue training the fitted model (sklearn's `warm_start=True`).
    #[pyo3(signature = (x, y=None, sample_weight=None))]
    fn warm_start(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: Option<PyReadonlyArray1<'_, f64>>,
        sample_weight: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<()> {
        let (x, y, w) = (
            x.as_array(),
            y.as_ref().map(|y| y.as_array()),
            sample_weight.as_ref().map(|w| w.as_array()),
        );
        let inner = &mut self.inner;
        py.detach(|| inner.warm_start(x, y, w)).map_err(to_py_err)
    }

    /// Fit with an explicit validation set for early stopping.
    #[pyo3(signature = (x, y, sample_weight, x_val, y_val, sample_weight_val=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit_with_validation(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        sample_weight: Option<PyReadonlyArray1<'_, f64>>,
        x_val: PyReadonlyArray2<'_, f64>,
        y_val: PyReadonlyArray1<'_, f64>,
        sample_weight_val: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<()> {
        let (x, y, w) = (
            x.as_array(),
            y.as_array(),
            sample_weight.as_ref().map(|w| w.as_array()),
        );
        let (xv, yv, wv) = (
            x_val.as_array(),
            y_val.as_array(),
            sample_weight_val.as_ref().map(|w| w.as_array()),
        );
        let inner = &mut self.inner;
        py.detach(|| inner.fit_with_validation(x, Some(y), w, xv, Some(yv), wv))
            .map_err(to_py_err)
    }
}

impl PyModel {
    pub(crate) fn from_inner(inner: Box<dyn DynModel>) -> Self {
        Self { inner }
    }

    fn call(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        f: impl FnOnce(&dyn DynModel, ArrayView2<f64>) -> anofox_ml_core::Result<Out> + Send,
    ) -> PyResult<Py<PyAny>> {
        let x = x.as_array();
        let inner: &dyn DynModel = self.inner.as_ref();
        let out = py.detach(|| f(inner, x)).map_err(to_py_err)?;
        Ok(out_to_py(py, out))
    }
}
