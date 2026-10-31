use ndarray::Array1;
use numpy::PyReadonlyArray1;
use pyo3::PyErr;

pub(crate) fn to_array1(y: PyReadonlyArray1<'_, f64>) -> Array1<f64> {
    y.as_array().to_owned()
}

pub(crate) fn py_err(e: impl std::fmt::Display) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
}

pub(crate) fn parse_average(average: &str) -> pyo3::PyResult<anofox_ml_metrics::Average> {
    match average {
        "macro" => Ok(anofox_ml_metrics::Average::Macro),
        "micro" => Ok(anofox_ml_metrics::Average::Micro),
        "weighted" => Ok(anofox_ml_metrics::Average::Weighted),
        other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unknown average mode '{}'; expected 'macro', 'micro', or 'weighted'",
            other
        ))),
    }
}
