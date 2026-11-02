//! Radius-neighbour estimators (sklearn's `RadiusNeighborsClassifier` and
//! `RadiusNeighborsRegressor`): every training sample within `radius` of a
//! query votes or averages, optionally weighted by inverse distance.

use anofox_ml_core::{Fit, Float, Predict, PredictProba, Result, RustMlError};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, AsArray, Ix2};

use crate::distance::DistanceMetric;
use crate::index::{NeighborAlgorithm, NeighborIndex, RadiusNeighbors};
use crate::knn_classifier::{argmax_first, encode_classes, neighbor_weights, WeightFunction};

fn check_fit<F: Float>(x: ArrayView2<F>, y: ArrayView1<F>, radius: f64) -> Result<()> {
    if x.nrows() != y.len() {
        return Err(RustMlError::ShapeMismatch(format!(
            "X has {} rows but y has {} elements",
            x.nrows(),
            y.len()
        )));
    }
    if x.is_empty() {
        return Err(RustMlError::EmptyInput("training data is empty".into()));
    }
    if radius.is_nan() || radius < 0.0 {
        return Err(RustMlError::InvalidParameter(format!(
            "radius must be >= 0, got {radius}"
        )));
    }
    Ok(())
}

/// Radius-neighbour classifier parameters (unfitted state).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RadiusNeighborsClassifier {
    /// Neighbourhood radius (inclusive). Default 1.0.
    pub radius: f64,
    pub weights: WeightFunction,
    pub metric: DistanceMetric,
    pub algorithm: NeighborAlgorithm,
    /// Label for queries without neighbours; `None` makes such queries an
    /// error (sklearn's `outlier_label=None`).
    pub outlier_label: Option<f64>,
}

impl RadiusNeighborsClassifier {
    pub fn new(radius: f64) -> Self {
        Self {
            radius,
            weights: WeightFunction::Uniform,
            metric: DistanceMetric::Euclidean,
            algorithm: NeighborAlgorithm::Auto,
            outlier_label: None,
        }
    }
    pub fn with_weights(mut self, weights: WeightFunction) -> Self {
        self.weights = weights;
        self
    }
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }
    pub fn with_algorithm(mut self, algorithm: NeighborAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }
    pub fn with_outlier_label(mut self, label: Option<f64>) -> Self {
        self.outlier_label = label;
        self
    }
}

impl Default for RadiusNeighborsClassifier {
    fn default() -> Self {
        Self::new(1.0)
    }
}

/// Fitted radius-neighbour classifier.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedRadiusNeighborsClassifier<F: Float> {
    index: NeighborIndex<F>,
    classes: Vec<F>,
    y_class: Vec<usize>,
    radius: F,
    weights: WeightFunction,
    outlier_label: Option<F>,
}

impl<F: Float> Fit<F> for RadiusNeighborsClassifier {
    type Fitted = FittedRadiusNeighborsClassifier<F>;

    fn fit_view(&self, x: ArrayView2<F>, y: ArrayView1<F>) -> Result<Self::Fitted> {
        check_fit(x, y, self.radius)?;
        let (classes, y_class) = encode_classes(y);
        Ok(FittedRadiusNeighborsClassifier {
            index: NeighborIndex::new(x, self.metric, self.algorithm, None)?,
            classes,
            y_class,
            radius: F::from_f64(self.radius).unwrap(),
            weights: self.weights,
            outlier_label: self.outlier_label.map(|l| F::from_f64(l).unwrap()),
        })
    }
}

impl<F: Float> FittedRadiusNeighborsClassifier<F> {
    /// Sorted class labels (columns of [`predict_proba`](Self::predict_proba)).
    pub fn classes(&self) -> &[F] {
        &self.classes
    }

    /// Training samples within the radius of every row of `x`.
    pub fn radius_neighbors<'a>(&self, x: impl AsArray<'a, F, Ix2>) -> Result<RadiusNeighbors<F>> {
        let x: ArrayView2<F> = x.into();
        self.index.radius_neighbors(x, self.radius)
    }

    fn no_neighbour_error(&self, nbrs: &RadiusNeighbors<F>) -> Result<()> {
        if self.outlier_label.is_none() {
            if let Some(i) = nbrs.iter().position(Vec::is_empty) {
                return Err(RustMlError::InvalidParameter(format!(
                    "no neighbours within the radius for sample {i}; set an outlier_label"
                )));
            }
        }
        Ok(())
    }

    /// Class probabilities: the (weighted) share of each class among the
    /// neighbours. A query without neighbours gets probability 1 for the
    /// outlier label if that is a training class, else all zeros.
    pub fn predict_proba<'a>(&self, x: impl AsArray<'a, F, Ix2>) -> Result<Array2<F>> {
        let x: ArrayView2<F> = x.into();
        let nbrs = self.radius_neighbors(x)?;
        self.no_neighbour_error(&nbrs)?;
        let outlier_class = self
            .outlier_label
            .and_then(|l| self.classes.iter().position(|&c| c == l));
        let mut proba = Array2::zeros((x.nrows(), self.classes.len()));
        for (mut row, nb) in proba.rows_mut().into_iter().zip(&nbrs) {
            if nb.is_empty() {
                if let Some(c) = outlier_class {
                    row[c] = F::one();
                }
                continue;
            }
            let d: Vec<F> = nb.iter().map(|n| n.1).collect();
            for (&(j, _), w) in nb.iter().zip(neighbor_weights(&d, self.weights)) {
                row[self.y_class[j]] += w;
            }
            let total = row.sum();
            if total > F::zero() {
                row /= total;
            }
        }
        Ok(proba)
    }
}

impl<F: Float> Predict<F> for FittedRadiusNeighborsClassifier<F> {
    /// The class with the largest (weighted) vote, or the outlier label for
    /// queries without neighbours.
    fn predict_view(&self, x: ArrayView2<F>) -> Result<Array1<F>> {
        let nbrs = self.radius_neighbors(x)?;
        self.no_neighbour_error(&nbrs)?;
        let proba = self.predict_proba(x)?;
        Ok(proba
            .rows()
            .into_iter()
            .zip(&nbrs)
            .map(|(row, nb)| match (nb.is_empty(), self.outlier_label) {
                (true, Some(l)) => l,
                _ => self.classes[argmax_first(row.iter().copied())],
            })
            .collect())
    }
}

impl<F: Float> PredictProba<F> for FittedRadiusNeighborsClassifier<F> {
    fn predict_proba_view(&self, x: ArrayView2<F>) -> Result<Array2<F>> {
        FittedRadiusNeighborsClassifier::predict_proba(self, x)
    }
}

/// Radius-neighbour regressor parameters (unfitted state).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RadiusNeighborsRegressor {
    /// Neighbourhood radius (inclusive). Default 1.0.
    pub radius: f64,
    pub weights: WeightFunction,
    pub metric: DistanceMetric,
    pub algorithm: NeighborAlgorithm,
}

impl RadiusNeighborsRegressor {
    pub fn new(radius: f64) -> Self {
        Self {
            radius,
            weights: WeightFunction::Uniform,
            metric: DistanceMetric::Euclidean,
            algorithm: NeighborAlgorithm::Auto,
        }
    }
    pub fn with_weights(mut self, weights: WeightFunction) -> Self {
        self.weights = weights;
        self
    }
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }
    pub fn with_algorithm(mut self, algorithm: NeighborAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }
}

impl Default for RadiusNeighborsRegressor {
    fn default() -> Self {
        Self::new(1.0)
    }
}

/// Fitted radius-neighbour regressor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedRadiusNeighborsRegressor<F: Float> {
    index: NeighborIndex<F>,
    y_train: Array1<F>,
    radius: F,
    weights: WeightFunction,
}

impl<F: Float> Fit<F> for RadiusNeighborsRegressor {
    type Fitted = FittedRadiusNeighborsRegressor<F>;

    fn fit_view(&self, x: ArrayView2<F>, y: ArrayView1<F>) -> Result<Self::Fitted> {
        check_fit(x, y, self.radius)?;
        Ok(FittedRadiusNeighborsRegressor {
            index: NeighborIndex::new(x, self.metric, self.algorithm, None)?,
            y_train: y.to_owned(),
            radius: F::from_f64(self.radius).unwrap(),
            weights: self.weights,
        })
    }
}

impl<F: Float> FittedRadiusNeighborsRegressor<F> {
    /// Training samples within the radius of every row of `x`.
    pub fn radius_neighbors<'a>(&self, x: impl AsArray<'a, F, Ix2>) -> Result<RadiusNeighbors<F>> {
        let x: ArrayView2<F> = x.into();
        self.index.radius_neighbors(x, self.radius)
    }
}

impl<F: Float> Predict<F> for FittedRadiusNeighborsRegressor<F> {
    /// (Weighted) mean target of the neighbours; NaN for queries without
    /// neighbours (sklearn).
    fn predict_view(&self, x: ArrayView2<F>) -> Result<Array1<F>> {
        let nbrs = self.radius_neighbors(x)?;
        Ok(nbrs
            .iter()
            .map(|nb| {
                if nb.is_empty() {
                    return F::nan();
                }
                let d: Vec<F> = nb.iter().map(|n| n.1).collect();
                crate::knn_regressor::weighted_mean(
                    &self.y_train,
                    nb.iter().map(|n| n.0),
                    &d,
                    self.weights,
                )
            })
            .collect())
    }
}
