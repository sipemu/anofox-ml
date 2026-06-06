use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

use anofox_ml_core::FitUnsupervised;

use crate::helpers::{not_fitted, py_err, to_array2};

// ---------------------------------------------------------------------------
// TSne
// ---------------------------------------------------------------------------
#[pyclass]
pub struct TSne {
    inner: anofox_ml_manifold::TSne,
    fitted: Option<anofox_ml_manifold::FittedTSne>,
}

#[pymethods]
impl TSne {
    #[new]
    #[pyo3(signature = (n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=500, seed=0, method="exact"))]
    fn new(
        n_components: usize,
        perplexity: f64,
        learning_rate: f64,
        n_iter: usize,
        seed: u64,
        method: &str,
    ) -> PyResult<Self> {
        let m = match method {
            "exact" => anofox_ml_manifold::TSneMethod::Exact,
            "barnes_hut" => anofox_ml_manifold::TSneMethod::BarnesHut,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown method '{other}'; expected 'exact' or 'barnes_hut'"
                )));
            }
        };
        Ok(Self {
            inner: anofox_ml_manifold::TSne::new(n_components)
                .with_perplexity(perplexity)
                .with_learning_rate(learning_rate)
                .with_n_iter(n_iter)
                .with_seed(seed)
                .with_method(m),
            fitted: None,
        })
    }
    fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = FitUnsupervised::fit(&self.inner, &to_array2(x)).map_err(py_err)?;
        let emb = fitted.embedding.clone();
        self.fitted = Some(fitted);
        Ok(PyArray2::from_owned_array(py, emb))
    }
    #[getter]
    fn kl_divergence(&self) -> PyResult<f64> {
        let f = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(f.kl_divergence)
    }
    #[getter]
    fn n_iter(&self) -> PyResult<usize> {
        let f = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(f.n_iter)
    }
}

// ---------------------------------------------------------------------------
// Isomap
// ---------------------------------------------------------------------------
#[pyclass]
pub struct Isomap {
    inner: anofox_ml_manifold::Isomap,
    fitted: Option<anofox_ml_manifold::FittedIsomap>,
}

#[pymethods]
impl Isomap {
    #[new]
    #[pyo3(signature = (n_components=2, n_neighbors=5))]
    fn new(n_components: usize, n_neighbors: usize) -> Self {
        Self {
            inner: anofox_ml_manifold::Isomap::new(n_components, n_neighbors),
            fitted: None,
        }
    }
    fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = FitUnsupervised::fit(&self.inner, &to_array2(x)).map_err(py_err)?;
        let emb = fitted.embedding.clone();
        self.fitted = Some(fitted);
        Ok(PyArray2::from_owned_array(py, emb))
    }
}

// ---------------------------------------------------------------------------
// LocallyLinearEmbedding
// ---------------------------------------------------------------------------
#[pyclass]
pub struct LocallyLinearEmbedding {
    inner: anofox_ml_manifold::LocallyLinearEmbedding,
    fitted: Option<anofox_ml_manifold::FittedLocallyLinearEmbedding>,
}

#[pymethods]
impl LocallyLinearEmbedding {
    #[new]
    #[pyo3(signature = (n_components=2, n_neighbors=5))]
    fn new(n_components: usize, n_neighbors: usize) -> Self {
        Self {
            inner: anofox_ml_manifold::LocallyLinearEmbedding::new(n_components, n_neighbors),
            fitted: None,
        }
    }
    fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = FitUnsupervised::fit(&self.inner, &to_array2(x)).map_err(py_err)?;
        let emb = fitted.embedding.clone();
        self.fitted = Some(fitted);
        Ok(PyArray2::from_owned_array(py, emb))
    }
}
