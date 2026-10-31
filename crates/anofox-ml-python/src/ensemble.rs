use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use anofox_ml_core::{Fit, Predict};

use crate::helpers::{not_fitted, py_err, to_array1, to_array2};

// ---------------------------------------------------------------------------
// Random Forest Classifier
// ---------------------------------------------------------------------------
#[pyclass]
pub struct RandomForestClassifier {
    inner: anofox_ml_ensemble::RandomForestClassifier,
    fitted: Option<anofox_ml_ensemble::FittedRandomForestClassifier<f64>>,
}

#[pymethods]
impl RandomForestClassifier {
    #[new]
    #[pyo3(signature = (n_estimators=100, max_depth=None, max_features=None, seed=0))]
    fn new(
        n_estimators: usize,
        max_depth: Option<usize>,
        max_features: Option<usize>,
        seed: u64,
    ) -> Self {
        // `max_features=None` keeps the estimator default (sqrt(n_features)
        // per split for the classifier, all features for the regressor, as in
        // sklearn); an integer is a fixed per-split feature count.
        let mut inner = anofox_ml_ensemble::RandomForestClassifier::new(n_estimators)
            .with_max_depth(max_depth)
            .with_seed(seed);
        if let Some(k) = max_features {
            inner = inner.with_max_features(Some(anofox_ml_trees::MaxFeatures::Fixed(k)));
        }
        Self {
            inner,
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
        crate::helpers::save_model_json(&self.inner, fitted, path)
    }

    #[staticmethod]
    fn load_json(path: &str) -> PyResult<Self> {
        let (inner, fitted) = crate::helpers::load_model_json(path, || {
            anofox_ml_ensemble::RandomForestClassifier::default()
        })?;
        Ok(Self {
            inner,
            fitted: Some(fitted),
        })
    }

    fn save_bincode(&self, path: &str) -> PyResult<()> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        crate::helpers::save_model_bincode(&self.inner, fitted, path)
    }

    #[staticmethod]
    fn load_bincode(path: &str) -> PyResult<Self> {
        let (inner, fitted) = crate::helpers::load_model_bincode(path, || {
            anofox_ml_ensemble::RandomForestClassifier::default()
        })?;
        Ok(Self {
            inner,
            fitted: Some(fitted),
        })
    }
}

// ---------------------------------------------------------------------------
// Random Forest Regressor
// ---------------------------------------------------------------------------
#[pyclass]
pub struct RandomForestRegressor {
    inner: anofox_ml_ensemble::RandomForestRegressor,
    fitted: Option<anofox_ml_ensemble::FittedRandomForestRegressor<f64>>,
}

#[pymethods]
impl RandomForestRegressor {
    #[new]
    #[pyo3(signature = (n_estimators=100, max_depth=None, max_features=None, seed=0))]
    fn new(
        n_estimators: usize,
        max_depth: Option<usize>,
        max_features: Option<usize>,
        seed: u64,
    ) -> Self {
        // `max_features=None` keeps the estimator default (sqrt(n_features)
        // per split for the classifier, all features for the regressor, as in
        // sklearn); an integer is a fixed per-split feature count.
        let mut inner = anofox_ml_ensemble::RandomForestRegressor::new(n_estimators)
            .with_max_depth(max_depth)
            .with_seed(seed);
        if let Some(k) = max_features {
            inner = inner.with_max_features(Some(anofox_ml_trees::MaxFeatures::Fixed(k)));
        }
        Self {
            inner,
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
        crate::helpers::save_model_json(&self.inner, fitted, path)
    }

    #[staticmethod]
    fn load_json(path: &str) -> PyResult<Self> {
        let (inner, fitted) = crate::helpers::load_model_json(path, || {
            anofox_ml_ensemble::RandomForestRegressor::default()
        })?;
        Ok(Self {
            inner,
            fitted: Some(fitted),
        })
    }

    fn save_bincode(&self, path: &str) -> PyResult<()> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        crate::helpers::save_model_bincode(&self.inner, fitted, path)
    }

    #[staticmethod]
    fn load_bincode(path: &str) -> PyResult<Self> {
        let (inner, fitted) = crate::helpers::load_model_bincode(path, || {
            anofox_ml_ensemble::RandomForestRegressor::default()
        })?;
        Ok(Self {
            inner,
            fitted: Some(fitted),
        })
    }
}

// ---------------------------------------------------------------------------
// Gradient Boosting Classifier
// ---------------------------------------------------------------------------
#[pyclass]
pub struct GradientBoostingClassifier {
    inner: anofox_ml_ensemble::GradientBoostingClassifier,
    fitted: Option<anofox_ml_ensemble::FittedGradientBoostingClassifier<f64>>,
}

#[pymethods]
impl GradientBoostingClassifier {
    #[new]
    #[pyo3(signature = (n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1.0, seed=0))]
    fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: usize,
        subsample: f64,
        seed: u64,
    ) -> Self {
        Self {
            inner: anofox_ml_ensemble::GradientBoostingClassifier::new()
                .with_n_estimators(n_estimators)
                .with_learning_rate(learning_rate)
                .with_max_depth(Some(max_depth))
                .with_subsample(subsample)
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

    fn save_json(&self, path: &str) -> PyResult<()> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        crate::helpers::save_model_json(&self.inner, fitted, path)
    }

    #[staticmethod]
    fn load_json(path: &str) -> PyResult<Self> {
        let (inner, fitted) = crate::helpers::load_model_json(path, || {
            anofox_ml_ensemble::GradientBoostingClassifier::new()
        })?;
        Ok(Self {
            inner,
            fitted: Some(fitted),
        })
    }

    fn save_bincode(&self, path: &str) -> PyResult<()> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        crate::helpers::save_model_bincode(&self.inner, fitted, path)
    }

    #[staticmethod]
    fn load_bincode(path: &str) -> PyResult<Self> {
        let (inner, fitted) = crate::helpers::load_model_bincode(path, || {
            anofox_ml_ensemble::GradientBoostingClassifier::new()
        })?;
        Ok(Self {
            inner,
            fitted: Some(fitted),
        })
    }
}

// ---------------------------------------------------------------------------
// Gradient Boosting Regressor
// ---------------------------------------------------------------------------
#[pyclass]
pub struct GradientBoostingRegressor {
    inner: anofox_ml_ensemble::GradientBoostingRegressor,
    fitted: Option<anofox_ml_ensemble::FittedGradientBoostingRegressor<f64>>,
}

#[pymethods]
impl GradientBoostingRegressor {
    #[new]
    #[pyo3(signature = (n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1.0, seed=0))]
    fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: usize,
        subsample: f64,
        seed: u64,
    ) -> Self {
        Self {
            inner: anofox_ml_ensemble::GradientBoostingRegressor::new()
                .with_n_estimators(n_estimators)
                .with_learning_rate(learning_rate)
                .with_max_depth(Some(max_depth))
                .with_subsample(subsample)
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

    fn save_json(&self, path: &str) -> PyResult<()> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        crate::helpers::save_model_json(&self.inner, fitted, path)
    }

    #[staticmethod]
    fn load_json(path: &str) -> PyResult<Self> {
        let (inner, fitted) = crate::helpers::load_model_json(path, || {
            anofox_ml_ensemble::GradientBoostingRegressor::new()
        })?;
        Ok(Self {
            inner,
            fitted: Some(fitted),
        })
    }

    fn save_bincode(&self, path: &str) -> PyResult<()> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        crate::helpers::save_model_bincode(&self.inner, fitted, path)
    }

    #[staticmethod]
    fn load_bincode(path: &str) -> PyResult<Self> {
        let (inner, fitted) = crate::helpers::load_model_bincode(path, || {
            anofox_ml_ensemble::GradientBoostingRegressor::new()
        })?;
        Ok(Self {
            inner,
            fitted: Some(fitted),
        })
    }
}
