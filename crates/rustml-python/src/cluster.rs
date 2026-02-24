use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use rustml_core::{FitUnsupervised, Predict};

use crate::helpers::{not_fitted, py_err, to_array2};

// ---------------------------------------------------------------------------
// KMeans
// ---------------------------------------------------------------------------
#[pyclass]
pub struct KMeans {
    inner: rustml_cluster::KMeans,
    fitted: Option<rustml_cluster::FittedKMeans<f64>>,
}

#[pymethods]
impl KMeans {
    #[new]
    #[pyo3(signature = (n_clusters=3, max_iter=300, seed=42))]
    fn new(n_clusters: usize, max_iter: usize, seed: u64) -> Self {
        Self {
            inner: rustml_cluster::KMeans::new(n_clusters)
                .with_max_iter(max_iter)
                .with_seed(seed),
            fitted: None,
        }
    }

    fn fit<'py>(&mut self, x: PyReadonlyArray2<'py, f64>) -> PyResult<()> {
        self.fitted = Some(FitUnsupervised::fit(&self.inner, &to_array2(x)).map_err(py_err)?);
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

    #[getter]
    fn labels<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(PyArray1::from_owned_array(py, fitted.labels().clone()))
    }

    #[getter]
    fn inertia(&self) -> PyResult<f64> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(fitted.inertia())
    }

    fn save_json(&self, path: &str) -> PyResult<()> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        rustml_core::persistence::save_json(fitted, path).map_err(py_err)
    }

    #[staticmethod]
    fn load_json(path: &str) -> PyResult<Self> {
        let fitted = rustml_core::persistence::load_json(path).map_err(py_err)?;
        Ok(Self { inner: rustml_cluster::KMeans::new(3), fitted: Some(fitted) })
    }

    fn save_bincode(&self, path: &str) -> PyResult<()> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        rustml_core::persistence::save_bincode(fitted, path).map_err(py_err)
    }

    #[staticmethod]
    fn load_bincode(path: &str) -> PyResult<Self> {
        let fitted = rustml_core::persistence::load_bincode(path).map_err(py_err)?;
        Ok(Self { inner: rustml_cluster::KMeans::new(3), fitted: Some(fitted) })
    }
}

// ---------------------------------------------------------------------------
// DBSCAN
// ---------------------------------------------------------------------------
#[pyclass]
pub struct Dbscan {
    inner: rustml_cluster::Dbscan,
    fitted: Option<rustml_cluster::FittedDbscan<f64>>,
}

#[pymethods]
impl Dbscan {
    #[new]
    #[pyo3(signature = (eps=0.5, min_samples=5))]
    fn new(eps: f64, min_samples: usize) -> Self {
        Self {
            inner: rustml_cluster::Dbscan::new(eps, min_samples),
            fitted: None,
        }
    }

    fn fit<'py>(&mut self, x: PyReadonlyArray2<'py, f64>) -> PyResult<()> {
        self.fitted = Some(FitUnsupervised::fit(&self.inner, &to_array2(x)).map_err(py_err)?);
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

    #[getter]
    fn labels<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(PyArray1::from_owned_array(py, fitted.labels().clone()))
    }

    #[getter]
    fn n_clusters(&self) -> PyResult<usize> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(fitted.n_clusters())
    }

    #[getter]
    fn core_sample_indices(&self) -> PyResult<Vec<usize>> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(fitted.core_sample_indices().to_vec())
    }

    fn save_json(&self, path: &str) -> PyResult<()> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        rustml_core::persistence::save_json(fitted, path).map_err(py_err)
    }

    #[staticmethod]
    fn load_json(path: &str) -> PyResult<Self> {
        let fitted = rustml_core::persistence::load_json(path).map_err(py_err)?;
        Ok(Self { inner: rustml_cluster::Dbscan::new(0.5, 5), fitted: Some(fitted) })
    }

    fn save_bincode(&self, path: &str) -> PyResult<()> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        rustml_core::persistence::save_bincode(fitted, path).map_err(py_err)
    }

    #[staticmethod]
    fn load_bincode(path: &str) -> PyResult<Self> {
        let fitted = rustml_core::persistence::load_bincode(path).map_err(py_err)?;
        Ok(Self { inner: rustml_cluster::Dbscan::new(0.5, 5), fitted: Some(fitted) })
    }
}
