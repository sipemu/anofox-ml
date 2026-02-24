use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use rustml_core::{Fit, FitUnsupervised, InverseTransform, Predict, Transform};

// ---------------------------------------------------------------------------
// Helper: convert numpy → ndarray
// ---------------------------------------------------------------------------
fn to_array2(x: PyReadonlyArray2<'_, f64>) -> Array2<f64> {
    x.as_array().to_owned()
}

fn to_array1(y: PyReadonlyArray1<'_, f64>) -> Array1<f64> {
    y.as_array().to_owned()
}

// ---------------------------------------------------------------------------
// StandardScaler
// ---------------------------------------------------------------------------
#[pyclass]
struct StandardScaler {
    inner: rustml_preprocessing::StandardScaler,
    fitted: Option<rustml_preprocessing::FittedStandardScaler<f64>>,
}

#[pymethods]
impl StandardScaler {
    #[new]
    #[pyo3(signature = (with_mean=true, with_std=true))]
    fn new(with_mean: bool, with_std: bool) -> Self {
        let mut s = rustml_preprocessing::StandardScaler::new();
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
        let fitted = FitUnsupervised::fit(&self.inner, &arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Not fitted"))?;
        let arr = to_array2(x);
        let result = fitted
            .transform(&arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
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
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Not fitted"))?;
        let arr = to_array2(x);
        let result = fitted
            .inverse_transform(&arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(PyArray2::from_owned_array(py, result))
    }
}

// ---------------------------------------------------------------------------
// KNN Classifier
// ---------------------------------------------------------------------------
#[pyclass]
struct KnnClassifier {
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
        let x_arr = to_array2(x);
        let y_arr = to_array1(y);
        let fitted = Fit::fit(&self.inner, &x_arr, &y_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Not fitted"))?;
        let arr = to_array2(x);
        let result = fitted
            .predict(&arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(PyArray1::from_owned_array(py, result))
    }
}

// ---------------------------------------------------------------------------
// KNN Regressor
// ---------------------------------------------------------------------------
#[pyclass]
struct KnnRegressor {
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
        let x_arr = to_array2(x);
        let y_arr = to_array1(y);
        let fitted = Fit::fit(&self.inner, &x_arr, &y_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Not fitted"))?;
        let arr = to_array2(x);
        let result = fitted
            .predict(&arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(PyArray1::from_owned_array(py, result))
    }
}

// ---------------------------------------------------------------------------
// Decision Tree Classifier
// ---------------------------------------------------------------------------
#[pyclass]
struct DecisionTreeClassifier {
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
        let x_arr = to_array2(x);
        let y_arr = to_array1(y);
        let fitted = Fit::fit(&self.inner, &x_arr, &y_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Not fitted"))?;
        let arr = to_array2(x);
        let result = fitted
            .predict(&arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(PyArray1::from_owned_array(py, result))
    }

    fn feature_importances<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Not fitted"))?;
        Ok(PyArray1::from_owned_array(py, fitted.feature_importances()))
    }
}

// ---------------------------------------------------------------------------
// Decision Tree Regressor
// ---------------------------------------------------------------------------
#[pyclass]
struct DecisionTreeRegressor {
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
        let x_arr = to_array2(x);
        let y_arr = to_array1(y);
        let fitted = Fit::fit(&self.inner, &x_arr, &y_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Not fitted"))?;
        let arr = to_array2(x);
        let result = fitted
            .predict(&arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(PyArray1::from_owned_array(py, result))
    }
}

// ---------------------------------------------------------------------------
// Random Forest Classifier
// ---------------------------------------------------------------------------
#[pyclass]
struct RandomForestClassifier {
    inner: rustml_ensemble::RandomForestClassifier,
    fitted: Option<rustml_ensemble::FittedRandomForestClassifier<f64>>,
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
        Self {
            inner: rustml_ensemble::RandomForestClassifier::new(n_estimators)
                .with_max_depth(max_depth)
                .with_max_features(max_features)
                .with_seed(seed),
            fitted: None,
        }
    }

    fn fit<'py>(
        &mut self,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<()> {
        let x_arr = to_array2(x);
        let y_arr = to_array1(y);
        let fitted = Fit::fit(&self.inner, &x_arr, &y_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Not fitted"))?;
        let arr = to_array2(x);
        let result = fitted
            .predict(&arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(PyArray1::from_owned_array(py, result))
    }

    fn feature_importances<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Not fitted"))?;
        Ok(PyArray1::from_owned_array(py, fitted.feature_importances()))
    }
}

// ---------------------------------------------------------------------------
// Random Forest Regressor
// ---------------------------------------------------------------------------
#[pyclass]
struct RandomForestRegressor {
    inner: rustml_ensemble::RandomForestRegressor,
    fitted: Option<rustml_ensemble::FittedRandomForestRegressor<f64>>,
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
        Self {
            inner: rustml_ensemble::RandomForestRegressor::new(n_estimators)
                .with_max_depth(max_depth)
                .with_max_features(max_features)
                .with_seed(seed),
            fitted: None,
        }
    }

    fn fit<'py>(
        &mut self,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<()> {
        let x_arr = to_array2(x);
        let y_arr = to_array1(y);
        let fitted = Fit::fit(&self.inner, &x_arr, &y_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Not fitted"))?;
        let arr = to_array2(x);
        let result = fitted
            .predict(&arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(PyArray1::from_owned_array(py, result))
    }
}

// ---------------------------------------------------------------------------
// KMeans
// ---------------------------------------------------------------------------
#[pyclass]
struct KMeans {
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
        let arr = to_array2(x);
        let fitted = FitUnsupervised::fit(&self.inner, &arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Not fitted"))?;
        let arr = to_array2(x);
        let result = fitted
            .predict(&arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(PyArray1::from_owned_array(py, result))
    }

    #[getter]
    fn labels<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Not fitted"))?;
        Ok(PyArray1::from_owned_array(py, fitted.labels().clone()))
    }

    #[getter]
    fn inertia(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Not fitted"))?;
        Ok(fitted.inertia())
    }
}

// ---------------------------------------------------------------------------
// Gaussian Naive Bayes
// ---------------------------------------------------------------------------
#[pyclass]
struct GaussianNB {
    inner: rustml_naive_bayes::GaussianNB,
    fitted: Option<rustml_naive_bayes::FittedGaussianNB<f64>>,
}

#[pymethods]
impl GaussianNB {
    #[new]
    #[pyo3(signature = (var_smoothing=1e-9))]
    fn new(var_smoothing: f64) -> Self {
        Self {
            inner: rustml_naive_bayes::GaussianNB::new().with_var_smoothing(var_smoothing),
            fitted: None,
        }
    }

    fn fit<'py>(
        &mut self,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<()> {
        let x_arr = to_array2(x);
        let y_arr = to_array1(y);
        let fitted = Fit::fit(&self.inner, &x_arr, &y_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Not fitted"))?;
        let arr = to_array2(x);
        let result = fitted
            .predict(&arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(PyArray1::from_owned_array(py, result))
    }
}

// ---------------------------------------------------------------------------
// DBSCAN
// ---------------------------------------------------------------------------
#[pyclass]
struct Dbscan {
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
        let arr = to_array2(x);
        let fitted = FitUnsupervised::fit(&self.inner, &arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Not fitted"))?;
        let arr = to_array2(x);
        let result = fitted
            .predict(&arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(PyArray1::from_owned_array(py, result))
    }

    #[getter]
    fn labels<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Not fitted"))?;
        Ok(PyArray1::from_owned_array(py, fitted.labels().clone()))
    }

    #[getter]
    fn n_clusters(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Not fitted"))?;
        Ok(fitted.n_clusters())
    }

    #[getter]
    fn core_sample_indices(&self) -> PyResult<Vec<usize>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Not fitted"))?;
        Ok(fitted.core_sample_indices().to_vec())
    }
}

// ---------------------------------------------------------------------------
// Gradient Boosting Classifier
// ---------------------------------------------------------------------------
#[pyclass]
struct GradientBoostingClassifier {
    inner: rustml_ensemble::GradientBoostingClassifier,
    fitted: Option<rustml_ensemble::FittedGradientBoostingClassifier<f64>>,
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
            inner: rustml_ensemble::GradientBoostingClassifier::new()
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
        let x_arr = to_array2(x);
        let y_arr = to_array1(y);
        let fitted = Fit::fit(&self.inner, &x_arr, &y_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Not fitted"))?;
        let arr = to_array2(x);
        let result = fitted
            .predict(&arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(PyArray1::from_owned_array(py, result))
    }
}

// ---------------------------------------------------------------------------
// Gradient Boosting Regressor
// ---------------------------------------------------------------------------
#[pyclass]
struct GradientBoostingRegressor {
    inner: rustml_ensemble::GradientBoostingRegressor,
    fitted: Option<rustml_ensemble::FittedGradientBoostingRegressor<f64>>,
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
            inner: rustml_ensemble::GradientBoostingRegressor::new()
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
        let x_arr = to_array2(x);
        let y_arr = to_array1(y);
        let fitted = Fit::fit(&self.inner, &x_arr, &y_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Not fitted"))?;
        let arr = to_array2(x);
        let result = fitted
            .predict(&arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(PyArray1::from_owned_array(py, result))
    }
}

// ---------------------------------------------------------------------------
// VarianceThreshold
// ---------------------------------------------------------------------------
#[pyclass]
struct VarianceThreshold {
    inner: rustml_preprocessing::VarianceThreshold,
    fitted: Option<rustml_preprocessing::FittedVarianceThreshold<f64>>,
}

#[pymethods]
impl VarianceThreshold {
    #[new]
    #[pyo3(signature = (threshold=0.0))]
    fn new(threshold: f64) -> Self {
        Self {
            inner: rustml_preprocessing::VarianceThreshold::new(threshold),
            fitted: None,
        }
    }

    fn fit<'py>(&mut self, x: PyReadonlyArray2<'py, f64>) -> PyResult<()> {
        let arr = to_array2(x);
        let fitted = FitUnsupervised::fit(&self.inner, &arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Not fitted"))?;
        let arr = to_array2(x);
        let result = fitted
            .transform(&arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
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
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Not fitted"))?;
        Ok(PyArray1::from_owned_array(py, fitted.variances().clone()))
    }

    #[getter]
    fn selected_indices(&self) -> PyResult<Vec<usize>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Not fitted"))?;
        Ok(fitted.selected_indices().to_vec())
    }
}

// ---------------------------------------------------------------------------
// MutualInformationSelector
// ---------------------------------------------------------------------------
#[pyclass]
struct MutualInformationSelector {
    inner: rustml_preprocessing::MutualInformationSelector,
    fitted: Option<rustml_preprocessing::FittedMutualInformationSelector<f64>>,
}

#[pymethods]
impl MutualInformationSelector {
    #[new]
    #[pyo3(signature = (n_features_to_select, n_bins=10))]
    fn new(n_features_to_select: usize, n_bins: usize) -> Self {
        Self {
            inner: rustml_preprocessing::MutualInformationSelector::new(n_features_to_select)
                .with_n_bins(n_bins),
            fitted: None,
        }
    }

    fn fit<'py>(
        &mut self,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<()> {
        let x_arr = to_array2(x);
        let y_arr = to_array1(y);
        let fitted = Fit::fit(&self.inner, &x_arr, &y_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Not fitted"))?;
        let arr = to_array2(x);
        let result = fitted
            .transform(&arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(PyArray2::from_owned_array(py, result))
    }

    #[getter]
    fn mi_scores<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Not fitted"))?;
        Ok(PyArray1::from_owned_array(py, fitted.mi_scores().clone()))
    }

    #[getter]
    fn selected_indices(&self) -> PyResult<Vec<usize>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Not fitted"))?;
        Ok(fitted.selected_indices().to_vec())
    }
}

// ---------------------------------------------------------------------------
// Metrics (free functions)
// ---------------------------------------------------------------------------
#[pyfunction]
fn accuracy_score<'py>(
    y_true: PyReadonlyArray1<'py, f64>,
    y_pred: PyReadonlyArray1<'py, f64>,
) -> PyResult<f64> {
    let yt = to_array1(y_true);
    let yp = to_array1(y_pred);
    rustml_metrics::accuracy_score(&yt, &yp)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}

#[pyfunction]
fn mse<'py>(
    y_true: PyReadonlyArray1<'py, f64>,
    y_pred: PyReadonlyArray1<'py, f64>,
) -> PyResult<f64> {
    let yt = to_array1(y_true);
    let yp = to_array1(y_pred);
    rustml_metrics::mse(&yt, &yp)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}

#[pyfunction]
fn r2_score<'py>(
    y_true: PyReadonlyArray1<'py, f64>,
    y_pred: PyReadonlyArray1<'py, f64>,
) -> PyResult<f64> {
    let yt = to_array1(y_true);
    let yp = to_array1(y_pred);
    rustml_metrics::r2_score(&yt, &yp)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}

#[pyfunction]
fn mae<'py>(
    y_true: PyReadonlyArray1<'py, f64>,
    y_pred: PyReadonlyArray1<'py, f64>,
) -> PyResult<f64> {
    let yt = to_array1(y_true);
    let yp = to_array1(y_pred);
    rustml_metrics::mae(&yt, &yp)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}

/// Convert a Python string average mode to the Rust `Average` enum.
fn parse_average(average: &str) -> PyResult<rustml_metrics::Average> {
    match average {
        "macro" => Ok(rustml_metrics::Average::Macro),
        "micro" => Ok(rustml_metrics::Average::Micro),
        "weighted" => Ok(rustml_metrics::Average::Weighted),
        other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unknown average mode '{}'; expected 'macro', 'micro', or 'weighted'",
            other
        ))),
    }
}

#[pyfunction]
#[pyo3(signature = (y_true, y_pred, average="macro"))]
fn precision_score<'py>(
    y_true: PyReadonlyArray1<'py, f64>,
    y_pred: PyReadonlyArray1<'py, f64>,
    average: &str,
) -> PyResult<f64> {
    let yt = to_array1(y_true);
    let yp = to_array1(y_pred);
    let avg = parse_average(average)?;
    rustml_metrics::precision_score(&yt, &yp, avg)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (y_true, y_pred, average="macro"))]
fn recall_score<'py>(
    y_true: PyReadonlyArray1<'py, f64>,
    y_pred: PyReadonlyArray1<'py, f64>,
    average: &str,
) -> PyResult<f64> {
    let yt = to_array1(y_true);
    let yp = to_array1(y_pred);
    let avg = parse_average(average)?;
    rustml_metrics::recall_score(&yt, &yp, avg)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (y_true, y_pred, average="macro"))]
fn f1_score<'py>(
    y_true: PyReadonlyArray1<'py, f64>,
    y_pred: PyReadonlyArray1<'py, f64>,
    average: &str,
) -> PyResult<f64> {
    let yt = to_array1(y_true);
    let yp = to_array1(y_pred);
    let avg = parse_average(average)?;
    rustml_metrics::f1_score_avg(&yt, &yp, avg)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------
#[pymodule]
fn rustml_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Preprocessing
    m.add_class::<StandardScaler>()?;
    m.add_class::<VarianceThreshold>()?;
    m.add_class::<MutualInformationSelector>()?;
    // Neighbors
    m.add_class::<KnnClassifier>()?;
    m.add_class::<KnnRegressor>()?;
    // Trees
    m.add_class::<DecisionTreeClassifier>()?;
    m.add_class::<DecisionTreeRegressor>()?;
    // Ensemble
    m.add_class::<RandomForestClassifier>()?;
    m.add_class::<RandomForestRegressor>()?;
    m.add_class::<GradientBoostingClassifier>()?;
    m.add_class::<GradientBoostingRegressor>()?;
    // Clustering
    m.add_class::<KMeans>()?;
    m.add_class::<Dbscan>()?;
    // Naive Bayes
    m.add_class::<GaussianNB>()?;
    // Metrics
    m.add_function(wrap_pyfunction!(accuracy_score, m)?)?;
    m.add_function(wrap_pyfunction!(mse, m)?)?;
    m.add_function(wrap_pyfunction!(r2_score, m)?)?;
    m.add_function(wrap_pyfunction!(mae, m)?)?;
    m.add_function(wrap_pyfunction!(precision_score, m)?)?;
    m.add_function(wrap_pyfunction!(recall_score, m)?)?;
    m.add_function(wrap_pyfunction!(f1_score, m)?)?;
    Ok(())
}
