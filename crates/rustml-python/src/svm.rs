use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use rustml_core::{Fit, Predict};

use crate::helpers::{not_fitted, parse_kernel, py_err, to_array1, to_array2};

// ---------------------------------------------------------------------------
// LinearSvc
// ---------------------------------------------------------------------------
#[pyclass]
pub struct LinearSvc {
    inner: rustml_svm::LinearSvc,
    fitted: Option<rustml_svm::FittedLinearSvc<f64>>,
}

#[pymethods]
impl LinearSvc {
    #[new]
    #[pyo3(signature = (c=1.0, max_iter=1000, tol=1e-4, seed=0))]
    fn new(c: f64, max_iter: usize, tol: f64, seed: u64) -> Self {
        Self {
            inner: rustml_svm::LinearSvc::new()
                .with_c(c)
                .with_max_iter(max_iter)
                .with_tol(tol)
                .with_seed(seed),
            fitted: None,
        }
    }

    fn fit<'py>(
        &mut self,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<()> {
        self.fitted = Some(Fit::fit(&self.inner, &to_array2(x), &to_array1(y)).map_err(py_err)?);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let result = fitted.predict(&to_array2(x)).map_err(py_err)?;
        Ok(PyArray1::from_owned_array(py, result))
    }

    fn decision_function<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let result = fitted.decision_function(&to_array2(x)).map_err(py_err)?;
        Ok(PyArray2::from_owned_array(py, result))
    }

    fn save_json(&self, path: &str) -> PyResult<()> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        rustml_core::persistence::save_json(fitted, path).map_err(py_err)
    }

    #[staticmethod]
    fn load_json(path: &str) -> PyResult<Self> {
        let fitted = rustml_core::persistence::load_json(path).map_err(py_err)?;
        Ok(Self { inner: rustml_svm::LinearSvc::new(), fitted: Some(fitted) })
    }

    fn save_bincode(&self, path: &str) -> PyResult<()> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        rustml_core::persistence::save_bincode(fitted, path).map_err(py_err)
    }

    #[staticmethod]
    fn load_bincode(path: &str) -> PyResult<Self> {
        let fitted = rustml_core::persistence::load_bincode(path).map_err(py_err)?;
        Ok(Self { inner: rustml_svm::LinearSvc::new(), fitted: Some(fitted) })
    }
}

// ---------------------------------------------------------------------------
// Svc (kernel SVM)
// ---------------------------------------------------------------------------
#[pyclass]
pub struct Svc {
    inner: rustml_svm::Svc,
    fitted: Option<rustml_svm::FittedSvc<f64>>,
}

#[pymethods]
impl Svc {
    #[new]
    #[pyo3(signature = (c=1.0, kernel="rbf", gamma=1.0, degree=3, coef0=0.0, max_iter=1000, tol=1e-4, seed=0))]
    fn new(
        c: f64,
        kernel: &str,
        gamma: f64,
        degree: usize,
        coef0: f64,
        max_iter: usize,
        tol: f64,
        seed: u64,
    ) -> PyResult<Self> {
        let k = parse_kernel(kernel, gamma, degree, coef0)?;
        Ok(Self {
            inner: rustml_svm::Svc::new()
                .with_c(c)
                .with_kernel(k)
                .with_max_iter(max_iter)
                .with_tol(tol)
                .with_seed(seed),
            fitted: None,
        })
    }

    fn fit<'py>(
        &mut self,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<()> {
        self.fitted = Some(Fit::fit(&self.inner, &to_array2(x), &to_array1(y)).map_err(py_err)?);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let result = fitted.predict(&to_array2(x)).map_err(py_err)?;
        Ok(PyArray1::from_owned_array(py, result))
    }

    fn decision_function<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let result = fitted.decision_function(&to_array2(x)).map_err(py_err)?;
        Ok(PyArray2::from_owned_array(py, result))
    }

    fn save_json(&self, path: &str) -> PyResult<()> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        rustml_core::persistence::save_json(fitted, path).map_err(py_err)
    }

    #[staticmethod]
    fn load_json(path: &str) -> PyResult<Self> {
        let fitted = rustml_core::persistence::load_json(path).map_err(py_err)?;
        Ok(Self { inner: rustml_svm::Svc::new(), fitted: Some(fitted) })
    }

    fn save_bincode(&self, path: &str) -> PyResult<()> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        rustml_core::persistence::save_bincode(fitted, path).map_err(py_err)
    }

    #[staticmethod]
    fn load_bincode(path: &str) -> PyResult<Self> {
        let fitted = rustml_core::persistence::load_bincode(path).map_err(py_err)?;
        Ok(Self { inner: rustml_svm::Svc::new(), fitted: Some(fitted) })
    }
}
