use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use crate::helpers::{parse_average, py_err, to_array1};

#[pyfunction]
pub fn accuracy_score<'py>(
    y_true: PyReadonlyArray1<'py, f64>,
    y_pred: PyReadonlyArray1<'py, f64>,
) -> PyResult<f64> {
    rustml_metrics::accuracy_score(&to_array1(y_true), &to_array1(y_pred)).map_err(py_err)
}

#[pyfunction]
pub fn mse<'py>(
    y_true: PyReadonlyArray1<'py, f64>,
    y_pred: PyReadonlyArray1<'py, f64>,
) -> PyResult<f64> {
    rustml_metrics::mse(&to_array1(y_true), &to_array1(y_pred)).map_err(py_err)
}

#[pyfunction]
pub fn r2_score<'py>(
    y_true: PyReadonlyArray1<'py, f64>,
    y_pred: PyReadonlyArray1<'py, f64>,
) -> PyResult<f64> {
    rustml_metrics::r2_score(&to_array1(y_true), &to_array1(y_pred)).map_err(py_err)
}

#[pyfunction]
pub fn mae<'py>(
    y_true: PyReadonlyArray1<'py, f64>,
    y_pred: PyReadonlyArray1<'py, f64>,
) -> PyResult<f64> {
    rustml_metrics::mae(&to_array1(y_true), &to_array1(y_pred)).map_err(py_err)
}

#[pyfunction]
#[pyo3(signature = (y_true, y_pred, average="macro"))]
pub fn precision_score<'py>(
    y_true: PyReadonlyArray1<'py, f64>,
    y_pred: PyReadonlyArray1<'py, f64>,
    average: &str,
) -> PyResult<f64> {
    let avg = parse_average(average)?;
    rustml_metrics::precision_score(&to_array1(y_true), &to_array1(y_pred), avg).map_err(py_err)
}

#[pyfunction]
#[pyo3(signature = (y_true, y_pred, average="macro"))]
pub fn recall_score<'py>(
    y_true: PyReadonlyArray1<'py, f64>,
    y_pred: PyReadonlyArray1<'py, f64>,
    average: &str,
) -> PyResult<f64> {
    let avg = parse_average(average)?;
    rustml_metrics::recall_score(&to_array1(y_true), &to_array1(y_pred), avg).map_err(py_err)
}

#[pyfunction]
#[pyo3(signature = (y_true, y_pred, average="macro"))]
pub fn f1_score<'py>(
    y_true: PyReadonlyArray1<'py, f64>,
    y_pred: PyReadonlyArray1<'py, f64>,
    average: &str,
) -> PyResult<f64> {
    let avg = parse_average(average)?;
    rustml_metrics::f1_score_avg(&to_array1(y_true), &to_array1(y_pred), avg).map_err(py_err)
}
