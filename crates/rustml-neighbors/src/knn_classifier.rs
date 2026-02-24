use ndarray::{Array1, Array2};
use rayon::prelude::*;
use rustml_core::{Fit, Float, Predict, Result, RustMlError};

use crate::distance::{compute_distance, DistanceMetric};
use crate::kdtree::KdTree;

/// Weighting strategy for KNN.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub enum WeightFunction {
    /// All neighbors contribute equally.
    #[default]
    Uniform,
    /// Closer neighbors have more influence (weight = 1/distance).
    Distance,
}

/// KNN classifier parameters (unfitted state).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KnnClassifier {
    pub n_neighbors: usize,
    pub metric: DistanceMetric,
    pub weights: WeightFunction,
}

impl KnnClassifier {
    /// Create a new `KnnClassifier` with the given number of neighbors and default parameters.
    pub fn new(k: usize) -> Self {
        Self {
            n_neighbors: k,
            metric: DistanceMetric::default(),
            weights: WeightFunction::default(),
        }
    }

    /// Set the weighting strategy for neighbor contributions.
    pub fn with_weights(mut self, weights: WeightFunction) -> Self {
        self.weights = weights;
        self
    }

    /// Set the distance metric used to find nearest neighbors.
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }
}

impl Default for KnnClassifier {
    fn default() -> Self {
        Self::new(5)
    }
}

/// Fitted KNN classifier — stores training data (lazy learner).
///
/// Uses a KD-tree for Euclidean distance (O(log n) per query) and
/// brute-force search for other metrics.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedKnnClassifier<F: Float> {
    x_train: Array2<F>,
    y_train: Array1<F>,
    kdtree: Option<KdTree<F>>,
    n_neighbors: usize,
    metric: DistanceMetric,
    weights: WeightFunction,
}

impl<F: Float> Fit<F> for KnnClassifier {
    type Fitted = FittedKnnClassifier<F>;

    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Self::Fitted> {
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
        if self.n_neighbors == 0 {
            return Err(RustMlError::InvalidParameter(
                "n_neighbors must be > 0".into(),
            ));
        }
        if self.n_neighbors > x.nrows() {
            return Err(RustMlError::InvalidParameter(format!(
                "n_neighbors ({}) > number of training samples ({})",
                self.n_neighbors,
                x.nrows()
            )));
        }

        // Build KD-tree for Euclidean distance
        let kdtree = if self.metric == DistanceMetric::Euclidean {
            let points: Vec<(Vec<F>, usize)> = x
                .rows()
                .into_iter()
                .enumerate()
                .map(|(i, row)| (row.to_vec(), i))
                .collect();
            Some(KdTree::build(&points, x.ncols()))
        } else {
            None
        };

        Ok(FittedKnnClassifier {
            x_train: x.to_owned(),
            y_train: y.to_owned(),
            kdtree,
            n_neighbors: self.n_neighbors,
            metric: self.metric,
            weights: self.weights,
        })
    }
}

impl<F: Float + Send + Sync> Predict<F> for FittedKnnClassifier<F> {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        if x.ncols() != self.x_train.ncols() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.x_train.ncols(),
                x.ncols()
            )));
        }

        let rows: Vec<Vec<F>> = x.rows().into_iter().map(|r| r.to_vec()).collect();

        // Parallel prediction using rayon
        let predictions: Vec<F> = rows
            .par_iter()
            .map(|row| {
                let neighbors = self.find_neighbors(row);
                weighted_majority_vote(&neighbors, self.weights)
            })
            .collect();

        Ok(Array1::from_vec(predictions))
    }
}

impl<F: Float> FittedKnnClassifier<F> {
    /// Find k nearest neighbors, returning (distance, label) pairs.
    fn find_neighbors(&self, query: &[F]) -> Vec<(F, F)> {
        if let Some(ref kdtree) = self.kdtree {
            // KD-tree path (Euclidean only)
            kdtree
                .query_k_nearest(query, self.n_neighbors)
                .into_iter()
                .map(|(dist, idx)| (dist, self.y_train[idx]))
                .collect()
        } else {
            // Brute-force path
            let query_view = ndarray::ArrayView1::from(query);
            let mut distances: Vec<(F, F)> = self
                .x_train
                .rows()
                .into_iter()
                .zip(self.y_train.iter())
                .map(|(train_row, &label)| {
                    let dist = compute_distance(&query_view, &train_row, self.metric);
                    (dist, label)
                })
                .collect();
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            distances.truncate(self.n_neighbors);
            distances
        }
    }
}

/// Weighted majority vote among neighbors.
fn weighted_majority_vote<F: Float>(neighbors: &[(F, F)], weights: WeightFunction) -> F {
    let mut classes: Vec<F> = neighbors.iter().map(|&(_, c)| c).collect();
    classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    classes.dedup_by(|a, b| (*a - *b).abs() < F::from_f64(1e-9).unwrap());

    let mut best_class = classes[0];
    let mut best_weight = F::neg_infinity();

    for &class in &classes {
        let weight: F = neighbors
            .iter()
            .filter(|&&(_, c)| (c - class).abs() < F::from_f64(1e-9).unwrap())
            .map(|&(dist, _)| match weights {
                WeightFunction::Uniform => F::one(),
                WeightFunction::Distance => {
                    if dist < F::from_f64(1e-15).unwrap() {
                        F::from_f64(1e15).unwrap()
                    } else {
                        F::one() / dist
                    }
                }
            })
            .fold(F::zero(), |a, b| a + b);

        if weight > best_weight {
            best_weight = weight;
            best_class = class;
        }
    }

    best_class
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_knn_simple() {
        let x_train = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.2]
        ];
        let y_train = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let knn = KnnClassifier {
            n_neighbors: 3,
            ..Default::default()
        };
        let fitted = Fit::fit(&knn, &x_train, &y_train).unwrap();

        let x_test = array![[0.05, 0.05], [10.05, 10.05]];
        let preds = fitted.predict(&x_test).unwrap();

        assert_abs_diff_eq!(preds[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_knn_distance_weights() {
        let x_train = array![[0.0], [1.0], [2.0]];
        let y_train = array![0.0, 0.0, 1.0];

        let knn = KnnClassifier {
            n_neighbors: 3,
            weights: WeightFunction::Distance,
            ..Default::default()
        };
        let fitted = Fit::fit(&knn, &x_train, &y_train).unwrap();

        let x_test = array![[1.9]];
        let preds = fitted.predict(&x_test).unwrap();
        assert_abs_diff_eq!(preds[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_knn_manhattan() {
        let x_train = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.2]
        ];
        let y_train = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let knn = KnnClassifier {
            n_neighbors: 3,
            metric: DistanceMetric::Manhattan,
            ..Default::default()
        };
        let fitted = Fit::fit(&knn, &x_train, &y_train).unwrap();

        let x_test = array![[0.05, 0.05]];
        let preds = fitted.predict(&x_test).unwrap();
        assert_abs_diff_eq!(preds[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_shape_mismatch() {
        let x = array![[1.0, 2.0]];
        let y = array![0.0, 1.0];
        let knn = KnnClassifier::default();
        assert!(Fit::fit(&knn, &x, &y).is_err());
    }
}
