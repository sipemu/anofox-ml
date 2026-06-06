use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use anofox_ml_core::{Fit, FitUnsupervised, InverseTransform, Transform};

use crate::helpers::{not_fitted, py_err, to_array1, to_array2};

// ---------------------------------------------------------------------------
// StandardScaler
// ---------------------------------------------------------------------------
#[pyclass]
pub struct StandardScaler {
    inner: anofox_ml_preprocessing::StandardScaler,
    fitted: Option<anofox_ml_preprocessing::FittedStandardScaler<f64>>,
}

#[pymethods]
impl StandardScaler {
    #[new]
    #[pyo3(signature = (with_mean=true, with_std=true))]
    fn new(with_mean: bool, with_std: bool) -> Self {
        let mut s = anofox_ml_preprocessing::StandardScaler::new();
        if !with_mean {
            s = s.with_mean(false);
        }
        if !with_std {
            s = s.with_std(false);
        }
        Self {
            inner: s,
            fitted: None,
        }
    }

    fn fit<'py>(&mut self, x: PyReadonlyArray2<'py, f64>) -> PyResult<()> {
        let arr = to_array2(x);
        self.fitted = Some(FitUnsupervised::fit(&self.inner, &arr).map_err(py_err)?);
        Ok(())
    }

    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let result = fitted.transform(&to_array2(x)).map_err(py_err)?;
        Ok(PyArray2::from_owned_array(py, result))
    }

    fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.fit(x.clone())?;
        self.transform(py, x)
    }

    fn inverse_transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let result = fitted.inverse_transform(&to_array2(x)).map_err(py_err)?;
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
            inner: anofox_ml_preprocessing::StandardScaler::new(),
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
            inner: anofox_ml_preprocessing::StandardScaler::new(),
            fitted: Some(fitted),
        })
    }
}

// ---------------------------------------------------------------------------
// MinMaxScaler
// ---------------------------------------------------------------------------
#[pyclass]
pub struct MinMaxScaler {
    inner: anofox_ml_preprocessing::MinMaxScaler<f64>,
    fitted: Option<anofox_ml_preprocessing::FittedMinMaxScaler<f64>>,
}

#[pymethods]
impl MinMaxScaler {
    #[new]
    #[pyo3(signature = (feature_min=0.0, feature_max=1.0))]
    fn new(feature_min: f64, feature_max: f64) -> Self {
        Self {
            inner: anofox_ml_preprocessing::MinMaxScaler::new()
                .with_range(feature_min, feature_max),
            fitted: None,
        }
    }

    fn fit<'py>(&mut self, x: PyReadonlyArray2<'py, f64>) -> PyResult<()> {
        self.fitted = Some(FitUnsupervised::fit(&self.inner, &to_array2(x)).map_err(py_err)?);
        Ok(())
    }

    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let result = fitted.transform(&to_array2(x)).map_err(py_err)?;
        Ok(PyArray2::from_owned_array(py, result))
    }

    fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.fit(x.clone())?;
        self.transform(py, x)
    }

    fn inverse_transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let result = fitted.inverse_transform(&to_array2(x)).map_err(py_err)?;
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
            inner: anofox_ml_preprocessing::MinMaxScaler::new(),
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
            inner: anofox_ml_preprocessing::MinMaxScaler::new(),
            fitted: Some(fitted),
        })
    }
}

// ---------------------------------------------------------------------------
// Pca
// ---------------------------------------------------------------------------
#[pyclass]
pub struct Pca {
    inner: anofox_ml_preprocessing::Pca,
    fitted: Option<anofox_ml_preprocessing::FittedPca<f64>>,
}

#[pymethods]
impl Pca {
    #[new]
    #[pyo3(signature = (n_components=2))]
    fn new(n_components: usize) -> Self {
        Self {
            inner: anofox_ml_preprocessing::Pca::new(n_components),
            fitted: None,
        }
    }

    fn fit<'py>(&mut self, x: PyReadonlyArray2<'py, f64>) -> PyResult<()> {
        self.fitted = Some(FitUnsupervised::fit(&self.inner, &to_array2(x)).map_err(py_err)?);
        Ok(())
    }

    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let result = fitted.transform(&to_array2(x)).map_err(py_err)?;
        Ok(PyArray2::from_owned_array(py, result))
    }

    fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.fit(x.clone())?;
        self.transform(py, x)
    }

    fn inverse_transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let result = fitted.inverse_transform(&to_array2(x)).map_err(py_err)?;
        Ok(PyArray2::from_owned_array(py, result))
    }

    #[getter]
    fn explained_variance<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(PyArray1::from_owned_array(
            py,
            fitted.explained_variance().clone(),
        ))
    }

    fn save_json(&self, path: &str) -> PyResult<()> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        anofox_ml_core::persistence::save_json(fitted, path).map_err(py_err)
    }

    #[staticmethod]
    fn load_json(path: &str) -> PyResult<Self> {
        let fitted = anofox_ml_core::persistence::load_json(path).map_err(py_err)?;
        Ok(Self {
            inner: anofox_ml_preprocessing::Pca::new(1),
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
            inner: anofox_ml_preprocessing::Pca::new(1),
            fitted: Some(fitted),
        })
    }
}

// ---------------------------------------------------------------------------
// VarianceThreshold
// ---------------------------------------------------------------------------
#[pyclass]
pub struct VarianceThreshold {
    inner: anofox_ml_preprocessing::VarianceThreshold,
    fitted: Option<anofox_ml_preprocessing::FittedVarianceThreshold<f64>>,
}

#[pymethods]
impl VarianceThreshold {
    #[new]
    #[pyo3(signature = (threshold=0.0))]
    fn new(threshold: f64) -> Self {
        Self {
            inner: anofox_ml_preprocessing::VarianceThreshold::new(threshold),
            fitted: None,
        }
    }

    fn fit<'py>(&mut self, x: PyReadonlyArray2<'py, f64>) -> PyResult<()> {
        self.fitted = Some(FitUnsupervised::fit(&self.inner, &to_array2(x)).map_err(py_err)?);
        Ok(())
    }

    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let result = fitted.transform(&to_array2(x)).map_err(py_err)?;
        Ok(PyArray2::from_owned_array(py, result))
    }

    fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.fit(x.clone())?;
        self.transform(py, x)
    }

    #[getter]
    fn variances<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(PyArray1::from_owned_array(py, fitted.variances().clone()))
    }

    #[getter]
    fn selected_indices(&self) -> PyResult<Vec<usize>> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(fitted.selected_indices().to_vec())
    }

    fn save_json(&self, path: &str) -> PyResult<()> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        anofox_ml_core::persistence::save_json(fitted, path).map_err(py_err)
    }

    #[staticmethod]
    fn load_json(path: &str) -> PyResult<Self> {
        let fitted = anofox_ml_core::persistence::load_json(path).map_err(py_err)?;
        Ok(Self {
            inner: anofox_ml_preprocessing::VarianceThreshold::new(0.0),
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
            inner: anofox_ml_preprocessing::VarianceThreshold::new(0.0),
            fitted: Some(fitted),
        })
    }
}

// ---------------------------------------------------------------------------
// MutualInformationSelector
// ---------------------------------------------------------------------------
#[pyclass]
pub struct MutualInformationSelector {
    inner: anofox_ml_preprocessing::MutualInformationSelector,
    fitted: Option<anofox_ml_preprocessing::FittedMutualInformationSelector<f64>>,
}

#[pymethods]
impl MutualInformationSelector {
    #[new]
    #[pyo3(signature = (n_features_to_select, n_bins=10))]
    fn new(n_features_to_select: usize, n_bins: usize) -> Self {
        Self {
            inner: anofox_ml_preprocessing::MutualInformationSelector::new(n_features_to_select)
                .with_n_bins(n_bins),
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

    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let result = fitted.transform(&to_array2(x)).map_err(py_err)?;
        Ok(PyArray2::from_owned_array(py, result))
    }

    #[getter]
    fn mi_scores<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(PyArray1::from_owned_array(py, fitted.mi_scores().clone()))
    }

    #[getter]
    fn selected_indices(&self) -> PyResult<Vec<usize>> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(fitted.selected_indices().to_vec())
    }

    fn save_json(&self, path: &str) -> PyResult<()> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        anofox_ml_core::persistence::save_json(fitted, path).map_err(py_err)
    }

    #[staticmethod]
    fn load_json(path: &str) -> PyResult<Self> {
        let fitted = anofox_ml_core::persistence::load_json(path).map_err(py_err)?;
        Ok(Self {
            inner: anofox_ml_preprocessing::MutualInformationSelector::new(1),
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
            inner: anofox_ml_preprocessing::MutualInformationSelector::new(1),
            fitted: Some(fitted),
        })
    }
}
