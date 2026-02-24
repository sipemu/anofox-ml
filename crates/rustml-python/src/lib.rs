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

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------
#[pymodule]
fn rustml_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<StandardScaler>()?;
    m.add_class::<KnnClassifier>()?;
    m.add_class::<KnnRegressor>()?;
    m.add_class::<DecisionTreeClassifier>()?;
    m.add_class::<DecisionTreeRegressor>()?;
    m.add_class::<RandomForestClassifier>()?;
    m.add_class::<RandomForestRegressor>()?;
    m.add_class::<KMeans>()?;
    m.add_class::<GaussianNB>()?;
    m.add_function(wrap_pyfunction!(accuracy_score, m)?)?;
    m.add_function(wrap_pyfunction!(mse, m)?)?;
    m.add_function(wrap_pyfunction!(r2_score, m)?)?;
    Ok(())
}
