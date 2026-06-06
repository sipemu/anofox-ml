use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use anofox_ml_core::{Fit, Predict};

use crate::helpers::{not_fitted, parse_activation, parse_solver, py_err, to_array1, to_array2};

// ---------------------------------------------------------------------------
// MLP Classifier
// ---------------------------------------------------------------------------
#[pyclass]
pub struct MlpClassifier {
    inner: anofox_ml_neural_networks::MlpClassifier,
    fitted: Option<anofox_ml_neural_networks::FittedMlpClassifier<f64>>,
}

#[pymethods]
impl MlpClassifier {
    #[new]
    #[pyo3(signature = (hidden_layer_sizes=vec![100], activation="relu", solver="adam", learning_rate=0.001, max_iter=200, tol=1e-4, seed=0, batch_size=Some(200), alpha=1e-4))]
    fn new(
        hidden_layer_sizes: Vec<usize>,
        activation: &str,
        solver: &str,
        learning_rate: f64,
        max_iter: usize,
        tol: f64,
        seed: u64,
        batch_size: Option<usize>,
        alpha: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: anofox_ml_neural_networks::MlpClassifier {
                hidden_layer_sizes,
                activation: parse_activation(activation)?,
                solver: parse_solver(solver)?,
                learning_rate,
                max_iter,
                tol,
                seed,
                batch_size,
                alpha,
            },
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

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let result = fitted.predict_proba(&to_array2(x)).map_err(py_err)?;
        Ok(PyArray2::from_owned_array(py, result))
    }

    fn save_json(&self, path: &str) -> PyResult<()> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        anofox_ml_core::persistence::save_json(fitted, path).map_err(py_err)
    }

    #[staticmethod]
    fn load_json(path: &str) -> PyResult<Self> {
        let fitted = anofox_ml_core::persistence::load_json(path).map_err(py_err)?;
        Ok(Self {
            inner: anofox_ml_neural_networks::MlpClassifier::default(),
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
            inner: anofox_ml_neural_networks::MlpClassifier::default(),
            fitted: Some(fitted),
        })
    }
}

// ---------------------------------------------------------------------------
// MLP Regressor
// ---------------------------------------------------------------------------
#[pyclass]
pub struct MlpRegressor {
    inner: anofox_ml_neural_networks::MlpRegressor,
    fitted: Option<anofox_ml_neural_networks::FittedMlpRegressor<f64>>,
}

#[pymethods]
impl MlpRegressor {
    #[new]
    #[pyo3(signature = (hidden_layer_sizes=vec![100], activation="relu", solver="adam", learning_rate=0.001, max_iter=200, tol=1e-4, seed=0, batch_size=Some(200), alpha=1e-4))]
    fn new(
        hidden_layer_sizes: Vec<usize>,
        activation: &str,
        solver: &str,
        learning_rate: f64,
        max_iter: usize,
        tol: f64,
        seed: u64,
        batch_size: Option<usize>,
        alpha: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: anofox_ml_neural_networks::MlpRegressor {
                hidden_layer_sizes,
                activation: parse_activation(activation)?,
                solver: parse_solver(solver)?,
                learning_rate,
                max_iter,
                tol,
                seed,
                batch_size,
                alpha,
            },
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

    fn save_json(&self, path: &str) -> PyResult<()> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        anofox_ml_core::persistence::save_json(fitted, path).map_err(py_err)
    }

    #[staticmethod]
    fn load_json(path: &str) -> PyResult<Self> {
        let fitted = anofox_ml_core::persistence::load_json(path).map_err(py_err)?;
        Ok(Self {
            inner: anofox_ml_neural_networks::MlpRegressor::default(),
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
            inner: anofox_ml_neural_networks::MlpRegressor::default(),
            fitted: Some(fitted),
        })
    }
}
