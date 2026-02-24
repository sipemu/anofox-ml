use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use rustml_core::{Fit, Predict};

use crate::helpers::{not_fitted, py_err, to_array1, to_array2};

// ---------------------------------------------------------------------------
// KNN Classifier
// ---------------------------------------------------------------------------
#[pyclass]
pub struct KnnClassifier {
    inner: rustml_neighbors::KnnClassifier,
    fitted: Option<rustml_neighbors::FittedKnnClassifier<f64>>,
}

#[pymethods]
impl KnnClassifier {
    #[new]
    #[pyo3(signature = (n_neighbors=5))]
    fn new(n_neighbors: usize) -> Self {
        Self {
            inner: rustml_neighbors::KnnClassifier::new(n_neighbors),
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
        rustml_core::persistence::save_json(fitted, path).map_err(py_err)
    }

    #[staticmethod]
    fn load_json(path: &str) -> PyResult<Self> {
        let fitted = rustml_core::persistence::load_json(path).map_err(py_err)?;
        Ok(Self { inner: rustml_neighbors::KnnClassifier::new(5), fitted: Some(fitted) })
    }

    fn save_bincode(&self, path: &str) -> PyResult<()> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        rustml_core::persistence::save_bincode(fitted, path).map_err(py_err)
    }

    #[staticmethod]
    fn load_bincode(path: &str) -> PyResult<Self> {
        let fitted = rustml_core::persistence::load_bincode(path).map_err(py_err)?;
        Ok(Self { inner: rustml_neighbors::KnnClassifier::new(5), fitted: Some(fitted) })
    }
}

// ---------------------------------------------------------------------------
// KNN Regressor
// ---------------------------------------------------------------------------
#[pyclass]
pub struct KnnRegressor {
    inner: rustml_neighbors::KnnRegressor,
    fitted: Option<rustml_neighbors::FittedKnnRegressor<f64>>,
}

#[pymethods]
impl KnnRegressor {
    #[new]
    #[pyo3(signature = (n_neighbors=5))]
    fn new(n_neighbors: usize) -> Self {
        Self {
            inner: rustml_neighbors::KnnRegressor::new(n_neighbors),
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
        rustml_core::persistence::save_json(fitted, path).map_err(py_err)
    }

    #[staticmethod]
    fn load_json(path: &str) -> PyResult<Self> {
        let fitted = rustml_core::persistence::load_json(path).map_err(py_err)?;
        Ok(Self { inner: rustml_neighbors::KnnRegressor::new(5), fitted: Some(fitted) })
    }

    fn save_bincode(&self, path: &str) -> PyResult<()> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        rustml_core::persistence::save_bincode(fitted, path).map_err(py_err)
    }

    #[staticmethod]
    fn load_bincode(path: &str) -> PyResult<Self> {
        let fitted = rustml_core::persistence::load_bincode(path).map_err(py_err)?;
        Ok(Self { inner: rustml_neighbors::KnnRegressor::new(5), fitted: Some(fitted) })
    }
}
