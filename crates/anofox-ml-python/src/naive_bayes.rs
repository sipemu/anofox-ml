use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use anofox_ml_core::{Fit, Predict};

use crate::helpers::{not_fitted, py_err, to_array1, to_array2};

// ---------------------------------------------------------------------------
// Gaussian Naive Bayes
// ---------------------------------------------------------------------------
#[pyclass]
pub struct GaussianNB {
    inner: anofox_ml_naive_bayes::GaussianNB,
    fitted: Option<anofox_ml_naive_bayes::FittedGaussianNB<f64>>,
}

#[pymethods]
impl GaussianNB {
    #[new]
    #[pyo3(signature = (var_smoothing=1e-9))]
    fn new(var_smoothing: f64) -> Self {
        Self {
            inner: anofox_ml_naive_bayes::GaussianNB::new().with_var_smoothing(var_smoothing),
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

    fn save_json(&self, path: &str) -> PyResult<()> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        anofox_ml_core::persistence::save_json(fitted, path).map_err(py_err)
    }

    #[staticmethod]
    fn load_json(path: &str) -> PyResult<Self> {
        let fitted = anofox_ml_core::persistence::load_json(path).map_err(py_err)?;
        Ok(Self {
            inner: anofox_ml_naive_bayes::GaussianNB::new(),
            fitted: Some(fitted),
        })
    }

    fn save_bincode(&self, path: &str) -> PyResult<()> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        anofox_ml_core::persistence::save_bincode(fitted, path).map_err(py_err)
    }

    #[staticmethod]
    fn load_bincode(path: &str) -> PyResult<Self> {
        let fitted = anofox_ml_core::persistence::load_bincode(path).map_err(py_err)?;
        Ok(Self {
            inner: anofox_ml_naive_bayes::GaussianNB::new(),
            fitted: Some(fitted),
        })
    }
}
