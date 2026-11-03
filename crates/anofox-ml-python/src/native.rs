//! `_native.Model`: the single Python class behind every estimator wrapper.

use ndarray::ArrayView2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyAttributeError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use anofox_ml_core::RustMlError;

use crate::model::{DynModel, Out};
use crate::registry;

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

    /// Configuration and fitted state as versioned bincode bytes.
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

    fn decision_function(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Py<PyAny>> {
        self.call(py, x, |m, x| m.decision_function(x))
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
}

impl PyModel {
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
