use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use rustml_core::{Fit, Predict};

use crate::helpers::{not_fitted, py_err, to_array1, to_array2};

// ---------------------------------------------------------------------------
// Decision Tree Classifier
// ---------------------------------------------------------------------------
#[pyclass]
pub struct DecisionTreeClassifier {
    inner: rustml_trees::DecisionTreeClassifier,
    fitted: Option<rustml_trees::FittedDecisionTreeClassifier<f64>>,
}

#[pymethods]
impl DecisionTreeClassifier {
    #[new]
    #[pyo3(signature = (max_depth=None, min_samples_split=2, min_samples_leaf=1))]
    fn new(max_depth: Option<usize>, min_samples_split: usize, min_samples_leaf: usize) -> Self {
        Self {
            inner: rustml_trees::DecisionTreeClassifier::new()
                .with_max_depth(max_depth)
                .with_min_samples_split(min_samples_split)
                .with_min_samples_leaf(min_samples_leaf),
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

    fn feature_importances<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(PyArray1::from_owned_array(py, fitted.feature_importances()))
    }

    fn save_json(&self, path: &str) -> PyResult<()> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        rustml_core::persistence::save_json(fitted, path).map_err(py_err)
    }

    #[staticmethod]
    fn load_json(path: &str) -> PyResult<Self> {
        let fitted = rustml_core::persistence::load_json(path).map_err(py_err)?;
        Ok(Self {
            inner: rustml_trees::DecisionTreeClassifier::new(),
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
            inner: rustml_trees::DecisionTreeClassifier::new(),
            fitted: Some(fitted),
        })
    }
}

// ---------------------------------------------------------------------------
// Decision Tree Regressor
// ---------------------------------------------------------------------------
#[pyclass]
pub struct DecisionTreeRegressor {
    inner: rustml_trees::DecisionTreeRegressor,
    fitted: Option<rustml_trees::FittedDecisionTreeRegressor<f64>>,
}

#[pymethods]
impl DecisionTreeRegressor {
    #[new]
    #[pyo3(signature = (max_depth=None, min_samples_split=2, min_samples_leaf=1))]
    fn new(max_depth: Option<usize>, min_samples_split: usize, min_samples_leaf: usize) -> Self {
        Self {
            inner: rustml_trees::DecisionTreeRegressor::new()
                .with_max_depth(max_depth)
                .with_min_samples_split(min_samples_split)
                .with_min_samples_leaf(min_samples_leaf),
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
        Ok(Self {
            inner: rustml_trees::DecisionTreeRegressor::new(),
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
            inner: rustml_trees::DecisionTreeRegressor::new(),
            fitted: Some(fitted),
        })
    }
}
