//! Stateless text / feature hashing helpers.

use numpy::PyArray2;
use pyo3::prelude::*;

use crate::native::to_py_err;

/// sklearn's `FeatureHasher.transform` on `(name, value)` pairs per sample:
/// a dense `n × n_features` array.
#[pyfunction]
pub fn feature_hash(
    py: Python<'_>,
    samples: Vec<Vec<(String, f64)>>,
    n_features: usize,
    alternate_sign: bool,
) -> PyResult<Py<PyAny>> {
    let hasher = anofox_ml_text::FeatureHasher {
        n_features,
        alternate_sign,
    };
    let out = py
        .detach(|| hasher.transform(&samples))
        .map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, out).into_any().unbind())
}
