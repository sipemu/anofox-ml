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

/// sklearn-style `gamma`: a positive float, `"scale"` or `"auto"`.
#[derive(pyo3::FromPyObject)]
pub(crate) enum PyGamma {
    Value(f64),
    Name(String),
}

pub(crate) fn parse_gamma(gamma: PyGamma) -> pyo3::PyResult<anofox_ml_svm::Gamma> {
    match gamma {
        PyGamma::Value(v) => Ok(anofox_ml_svm::Gamma::Value(v)),
        PyGamma::Name(s) => match s.as_str() {
            "scale" => Ok(anofox_ml_svm::Gamma::Scale),
            "auto" => Ok(anofox_ml_svm::Gamma::Auto),
            other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "unknown gamma '{other}'; expected a float, 'scale' or 'auto'"
            ))),
        },
    }
}

pub(crate) fn parse_kernel(
    kernel: &str,
    gamma: PyGamma,
    degree: usize,
    coef0: f64,
) -> pyo3::PyResult<anofox_ml_svm::SvmKernel> {
    let gamma = parse_gamma(gamma)?;
    match kernel {
        "linear" => Ok(anofox_ml_svm::SvmKernel::Linear),
        "rbf" => Ok(anofox_ml_svm::SvmKernel::Rbf { gamma }),
        "poly" | "polynomial" => Ok(anofox_ml_svm::SvmKernel::Polynomial {
            degree,
            gamma,
            coef0,
        }),
        "sigmoid" => Ok(anofox_ml_svm::SvmKernel::Sigmoid { gamma, coef0 }),
        "precomputed" => Ok(anofox_ml_svm::SvmKernel::Precomputed),
        other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unknown kernel '{}'; expected 'linear', 'rbf', 'poly', 'sigmoid' or 'precomputed'",
            other
        ))),
    }
}

pub(crate) fn parse_activation(s: &str) -> pyo3::PyResult<anofox_ml_neural_networks::Activation> {
    match s {
        "relu" => Ok(anofox_ml_neural_networks::Activation::Relu),
        "tanh" => Ok(anofox_ml_neural_networks::Activation::Tanh),
        "sigmoid" => Ok(anofox_ml_neural_networks::Activation::Sigmoid),
        "identity" => Ok(anofox_ml_neural_networks::Activation::Identity),
        other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unknown activation '{}'; expected 'relu', 'tanh', 'sigmoid', or 'identity'",
            other
        ))),
    }
}

pub(crate) fn parse_solver(s: &str) -> pyo3::PyResult<anofox_ml_neural_networks::Solver> {
    match s {
        "adam" => Ok(anofox_ml_neural_networks::Solver::Adam),
        "sgd" => Ok(anofox_ml_neural_networks::Solver::Sgd),
        "lbfgs" => Ok(anofox_ml_neural_networks::Solver::Lbfgs),
        other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unknown solver '{}'; expected 'adam', 'sgd' or 'lbfgs'",
            other
        ))),
    }
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
