use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use anofox_ml_core::{Fit, FitUnsupervised, Predict};

use crate::helpers::{not_fitted, py_err, to_array1, to_array2};

// ---------------------------------------------------------------------------
// KNN Classifier
// ---------------------------------------------------------------------------
#[pyclass]
pub struct KnnClassifier {
    inner: anofox_ml_neighbors::KnnClassifier,
    fitted: Option<anofox_ml_neighbors::FittedKnnClassifier<f64>>,
}

#[pymethods]
impl KnnClassifier {
    #[new]
    #[pyo3(signature = (n_neighbors=5))]
    fn new(n_neighbors: usize) -> Self {
        Self {
            inner: anofox_ml_neighbors::KnnClassifier::new(n_neighbors),
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
            inner: anofox_ml_neighbors::KnnClassifier::new(5),
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
            inner: anofox_ml_neighbors::KnnClassifier::new(5),
            fitted: Some(fitted),
        })
    }
}

// ---------------------------------------------------------------------------
// KNN Regressor
// ---------------------------------------------------------------------------
#[pyclass]
pub struct KnnRegressor {
    inner: anofox_ml_neighbors::KnnRegressor,
    fitted: Option<anofox_ml_neighbors::FittedKnnRegressor<f64>>,
}

#[pymethods]
impl KnnRegressor {
    #[new]
    #[pyo3(signature = (n_neighbors=5))]
    fn new(n_neighbors: usize) -> Self {
        Self {
            inner: anofox_ml_neighbors::KnnRegressor::new(n_neighbors),
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
            inner: anofox_ml_neighbors::KnnRegressor::new(5),
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
            inner: anofox_ml_neighbors::KnnRegressor::new(5),
            fitted: Some(fitted),
        })
    }
}

// ---------------------------------------------------------------------------
// LocalOutlierFactor
// ---------------------------------------------------------------------------
#[pyclass]
pub struct LocalOutlierFactor {
    inner: anofox_ml_neighbors::LocalOutlierFactor,
    fitted: Option<anofox_ml_neighbors::FittedLocalOutlierFactor>,
}

#[pymethods]
impl LocalOutlierFactor {
    #[new]
    #[pyo3(signature = (n_neighbors=20, contamination=0.1, algorithm="auto"))]
    fn new(n_neighbors: usize, contamination: f64, algorithm: &str) -> PyResult<Self> {
        let alg = match algorithm {
            "auto" => anofox_ml_neighbors::LofAlgorithm::Auto,
            "kdtree" | "kd_tree" => anofox_ml_neighbors::LofAlgorithm::KdTree,
            "brute" => anofox_ml_neighbors::LofAlgorithm::BruteForce,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown algorithm '{other}'; expected 'auto', 'kdtree', or 'brute'"
                )));
            }
        };
        Ok(Self {
            inner: anofox_ml_neighbors::LocalOutlierFactor::new(n_neighbors)
                .with_contamination(contamination)
                .with_algorithm(alg),
            fitted: None,
        })
    }
    fn fit_predict<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = FitUnsupervised::fit(&self.inner, &to_array2(x)).map_err(py_err)?;
        let preds = fitted.predictions.clone();
        self.fitted = Some(fitted);
        Ok(PyArray1::from_owned_array(py, preds))
    }
    #[getter]
    fn negative_outlier_factor<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let f = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(PyArray1::from_owned_array(
            py,
            f.negative_outlier_factor.clone(),
        ))
    }
    #[getter]
    fn threshold(&self) -> PyResult<f64> {
        let f = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(f.threshold)
    }
}
