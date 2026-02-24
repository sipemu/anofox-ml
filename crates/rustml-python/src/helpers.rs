use ndarray::{Array1, Array2};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::PyErr;

pub(crate) fn to_array2(x: PyReadonlyArray2<'_, f64>) -> Array2<f64> {
    x.as_array().to_owned()
}

pub(crate) fn to_array1(y: PyReadonlyArray1<'_, f64>) -> Array1<f64> {
    y.as_array().to_owned()
}

pub(crate) fn py_err(e: impl std::fmt::Display) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
}

pub(crate) fn not_fitted() -> PyErr {
    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Not fitted")
}

pub(crate) fn parse_kernel(kernel: &str, gamma: f64, degree: usize, coef0: f64) -> pyo3::PyResult<rustml_svm::SvmKernel> {
    match kernel {
        "linear" => Ok(rustml_svm::SvmKernel::Linear),
        "rbf" => Ok(rustml_svm::SvmKernel::Rbf { gamma }),
        "poly" | "polynomial" => Ok(rustml_svm::SvmKernel::Polynomial { degree, coef0 }),
        other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unknown kernel '{}'; expected 'linear', 'rbf', or 'poly'",
            other
        ))),
    }
}

pub(crate) fn parse_activation(s: &str) -> pyo3::PyResult<rustml_neural_networks::Activation> {
    match s {
        "relu" => Ok(rustml_neural_networks::Activation::Relu),
        "tanh" => Ok(rustml_neural_networks::Activation::Tanh),
        "sigmoid" => Ok(rustml_neural_networks::Activation::Sigmoid),
        "identity" => Ok(rustml_neural_networks::Activation::Identity),
        other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unknown activation '{}'; expected 'relu', 'tanh', 'sigmoid', or 'identity'",
            other
        ))),
    }
}

pub(crate) fn parse_solver(s: &str) -> pyo3::PyResult<rustml_neural_networks::Solver> {
    match s {
        "adam" => Ok(rustml_neural_networks::Solver::Adam),
        "sgd" => Ok(rustml_neural_networks::Solver::Sgd),
        other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unknown solver '{}'; expected 'adam' or 'sgd'",
            other
        ))),
    }
}

pub(crate) fn parse_average(average: &str) -> pyo3::PyResult<rustml_metrics::Average> {
    match average {
        "macro" => Ok(rustml_metrics::Average::Macro),
        "micro" => Ok(rustml_metrics::Average::Micro),
        "weighted" => Ok(rustml_metrics::Average::Weighted),
        other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unknown average mode '{}'; expected 'macro', 'micro', or 'weighted'",
            other
        ))),
    }
}
