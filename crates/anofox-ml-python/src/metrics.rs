use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::helpers::{parse_average, py_err, to_array1};

#[pyfunction]
pub fn accuracy_score<'py>(
    y_true: PyReadonlyArray1<'py, f64>,
    y_pred: PyReadonlyArray1<'py, f64>,
) -> PyResult<f64> {
    anofox_ml_metrics::accuracy_score(&to_array1(y_true), &to_array1(y_pred)).map_err(py_err)
}

#[pyfunction]
pub fn mse<'py>(
    y_true: PyReadonlyArray1<'py, f64>,
    y_pred: PyReadonlyArray1<'py, f64>,
) -> PyResult<f64> {
    anofox_ml_metrics::mse(&to_array1(y_true), &to_array1(y_pred)).map_err(py_err)
}

#[pyfunction]
pub fn r2_score<'py>(
    y_true: PyReadonlyArray1<'py, f64>,
    y_pred: PyReadonlyArray1<'py, f64>,
) -> PyResult<f64> {
    anofox_ml_metrics::r2_score(&to_array1(y_true), &to_array1(y_pred)).map_err(py_err)
}

#[pyfunction]
pub fn mae<'py>(
    y_true: PyReadonlyArray1<'py, f64>,
    y_pred: PyReadonlyArray1<'py, f64>,
) -> PyResult<f64> {
    anofox_ml_metrics::mae(&to_array1(y_true), &to_array1(y_pred)).map_err(py_err)
}

#[pyfunction]
#[pyo3(signature = (y_true, y_pred, average="macro"))]
pub fn precision_score<'py>(
    y_true: PyReadonlyArray1<'py, f64>,
    y_pred: PyReadonlyArray1<'py, f64>,
    average: &str,
) -> PyResult<f64> {
    let avg = parse_average(average)?;
    anofox_ml_metrics::precision_score(&to_array1(y_true), &to_array1(y_pred), avg).map_err(py_err)
}

#[pyfunction]
#[pyo3(signature = (y_true, y_pred, average="macro"))]
pub fn recall_score<'py>(
    y_true: PyReadonlyArray1<'py, f64>,
    y_pred: PyReadonlyArray1<'py, f64>,
    average: &str,
) -> PyResult<f64> {
    let avg = parse_average(average)?;
    anofox_ml_metrics::recall_score(&to_array1(y_true), &to_array1(y_pred), avg).map_err(py_err)
}

#[pyfunction]
#[pyo3(signature = (y_true, y_pred, average="macro"))]
pub fn f1_score<'py>(
    y_true: PyReadonlyArray1<'py, f64>,
    y_pred: PyReadonlyArray1<'py, f64>,
    average: &str,
) -> PyResult<f64> {
    let avg = parse_average(average)?;
    anofox_ml_metrics::f1_score_avg(&to_array1(y_true), &to_array1(y_pred), avg).map_err(py_err)
}

/// Metrics of the form `f(y_true, y) -> float`.
macro_rules! pair_metrics {
    ($($name:ident),* $(,)?) => {
        $(
            #[pyfunction]
            pub fn $name<'py>(
                y_true: PyReadonlyArray1<'py, f64>,
                y: PyReadonlyArray1<'py, f64>,
            ) -> PyResult<f64> {
                anofox_ml_metrics::$name(&to_array1(y_true), &to_array1(y)).map_err(py_err)
            }
        )*
    };
}

pair_metrics!(
    balanced_accuracy_score,
    cohen_kappa_score,
    explained_variance_score,
    matthews_corrcoef,
    max_error,
    mean_absolute_percentage_error,
    mean_squared_log_error,
    median_absolute_error,
    adjusted_rand_score,
    normalized_mutual_info_score,
    log_loss,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
);

#[pyfunction]
pub fn silhouette_score<'py>(
    x: PyReadonlyArray2<'py, f64>,
    labels: PyReadonlyArray1<'py, f64>,
) -> PyResult<f64> {
    anofox_ml_metrics::silhouette_score(&x.as_array().to_owned(), &to_array1(labels))
        .map_err(py_err)
}

#[pyfunction]
pub fn confusion_matrix<'py>(
    py: Python<'py>,
    y_true: PyReadonlyArray1<'py, f64>,
    y_pred: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let m = anofox_ml_metrics::confusion_matrix(&to_array1(y_true), &to_array1(y_pred))
        .map_err(py_err)?;
    Ok(PyArray2::from_owned_array(py, m))
}
