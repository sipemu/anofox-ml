use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

use rustml_core::{FitUnsupervised, Predict, PredictProba};

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
        Ok(Self {
            inner: rustml_cluster::KMeans::new(3),
            fitted: Some(fitted),
        })
    }

    fn save_bincode(&self, path: &str) -> PyResult<()> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        rustml_core::persistence::save_bincode(fitted, path).map_err(py_err)
    }

    #[staticmethod]
    fn load_bincode(path: &str) -> PyResult<Self> {
        let fitted = rustml_core::persistence::load_bincode(path).map_err(py_err)?;
        Ok(Self {
            inner: rustml_cluster::KMeans::new(3),
            fitted: Some(fitted),
        })
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
        Ok(Self {
            inner: rustml_cluster::Dbscan::new(0.5, 5),
            fitted: Some(fitted),
        })
    }

    fn save_bincode(&self, path: &str) -> PyResult<()> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        rustml_core::persistence::save_bincode(fitted, path).map_err(py_err)
    }

    #[staticmethod]
    fn load_bincode(path: &str) -> PyResult<Self> {
        let fitted = rustml_core::persistence::load_bincode(path).map_err(py_err)?;
        Ok(Self {
            inner: rustml_cluster::Dbscan::new(0.5, 5),
            fitted: Some(fitted),
        })
    }
}

// ---------------------------------------------------------------------------
// HDBSCAN
// ---------------------------------------------------------------------------
#[pyclass]
pub struct Hdbscan {
    inner: rustml_cluster::Hdbscan,
    fitted: Option<rustml_cluster::FittedHdbscan>,
}

#[pymethods]
impl Hdbscan {
    #[new]
    #[pyo3(signature = (min_samples=5, min_cluster_size=5))]
    fn new(min_samples: usize, min_cluster_size: usize) -> Self {
        Self {
            inner: rustml_cluster::Hdbscan::new(min_samples, min_cluster_size),
            fitted: None,
        }
    }
    fn fit<'py>(&mut self, x: PyReadonlyArray2<'py, f64>) -> PyResult<()> {
        self.fitted = Some(FitUnsupervised::fit(&self.inner, &to_array2(x)).map_err(py_err)?);
        Ok(())
    }
    #[getter]
    fn labels<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let f = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(PyArray1::from_owned_array(py, f.labels.clone()))
    }
    #[getter]
    fn n_clusters(&self) -> PyResult<usize> {
        let f = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(f.n_clusters)
    }
}

// ---------------------------------------------------------------------------
// MeanShift
// ---------------------------------------------------------------------------
#[pyclass]
pub struct MeanShift {
    inner: rustml_cluster::MeanShift,
    fitted: Option<rustml_cluster::FittedMeanShift>,
}

#[pymethods]
impl MeanShift {
    #[new]
    #[pyo3(signature = (bandwidth=1.0))]
    fn new(bandwidth: f64) -> Self {
        Self {
            inner: rustml_cluster::MeanShift::new(bandwidth),
            fitted: None,
        }
    }
    fn fit<'py>(&mut self, x: PyReadonlyArray2<'py, f64>) -> PyResult<()> {
        self.fitted = Some(FitUnsupervised::fit(&self.inner, &to_array2(x)).map_err(py_err)?);
        Ok(())
    }
    #[getter]
    fn labels<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let f = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(PyArray1::from_owned_array(py, f.labels.clone()))
    }
    #[getter]
    fn cluster_centers<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let f = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(PyArray2::from_owned_array(py, f.cluster_centers.clone()))
    }
}

// ---------------------------------------------------------------------------
// AffinityPropagation
// ---------------------------------------------------------------------------
#[pyclass]
pub struct AffinityPropagation {
    inner: rustml_cluster::AffinityPropagation,
    fitted: Option<rustml_cluster::FittedAffinityPropagation>,
}

#[pymethods]
impl AffinityPropagation {
    #[new]
    #[pyo3(signature = (damping=0.9, preference=None, n_neighbors=None))]
    fn new(damping: f64, preference: Option<f64>, n_neighbors: Option<usize>) -> Self {
        let mut ap = rustml_cluster::AffinityPropagation::new().with_damping(damping);
        if let Some(p) = preference {
            ap = ap.with_preference(p);
        }
        if let Some(k) = n_neighbors {
            ap = ap.with_n_neighbors(k);
        }
        Self {
            inner: ap,
            fitted: None,
        }
    }
    fn fit<'py>(&mut self, x: PyReadonlyArray2<'py, f64>) -> PyResult<()> {
        self.fitted = Some(FitUnsupervised::fit(&self.inner, &to_array2(x)).map_err(py_err)?);
        Ok(())
    }
    #[getter]
    fn labels<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let f = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(PyArray1::from_owned_array(py, f.labels.clone()))
    }
    #[getter]
    fn cluster_centers_indices(&self) -> PyResult<Vec<usize>> {
        let f = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(f.cluster_centers_indices.clone())
    }
}

// ---------------------------------------------------------------------------
// BayesianGaussianMixture
// ---------------------------------------------------------------------------
#[pyclass]
pub struct BayesianGaussianMixture {
    inner: rustml_cluster::BayesianGaussianMixture,
    fitted: Option<rustml_cluster::FittedBayesianGaussianMixture>,
}

#[pymethods]
impl BayesianGaussianMixture {
    #[new]
    #[pyo3(signature = (n_components=3, weight_concentration_prior=0.01, max_iter=200, seed=0))]
    fn new(
        n_components: usize,
        weight_concentration_prior: f64,
        max_iter: usize,
        seed: u64,
    ) -> Self {
        Self {
            inner: rustml_cluster::BayesianGaussianMixture::new(n_components)
                .with_concentration(weight_concentration_prior)
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
        let f = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let out = f.predict(&to_array2(x)).map_err(py_err)?;
        Ok(PyArray1::from_owned_array(py, out))
    }
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let f = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let out = f.predict_proba(&to_array2(x)).map_err(py_err)?;
        Ok(PyArray2::from_owned_array(py, out))
    }
    #[getter]
    fn weights<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let f = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(PyArray1::from_owned_array(py, f.weights.clone()))
    }
    #[getter]
    fn means<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let f = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(PyArray2::from_owned_array(py, f.means.clone()))
    }
    #[getter]
    fn n_iter(&self) -> PyResult<usize> {
        let f = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(f.n_iter)
    }
    #[getter]
    fn lower_bound(&self) -> PyResult<f64> {
        let f = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(f.lower_bound)
    }
}
