use ndarray::{Array1, Array2};
use rayon::prelude::*;
use rustml_core::{Fit, Float, Predict, Result, RustMlError};

use crate::distance::{compute_distances_batch, DistanceMetric};
use crate::kdtree::KdTree;
use crate::knn_classifier::WeightFunction;

/// KNN regressor parameters (unfitted state).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KnnRegressor {
    pub n_neighbors: usize,
    pub metric: DistanceMetric,
    pub weights: WeightFunction,
}

impl KnnRegressor {
    /// Create a new `KnnRegressor` with the given number of neighbors and default parameters.
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

impl Default for KnnRegressor {
    fn default() -> Self {
        Self::new(5)
    }
}

/// Fitted KNN regressor — stores training data (lazy learner).
///
/// Uses a KD-tree for Euclidean distance and brute-force for other metrics.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedKnnRegressor<F: Float> {
    x_train: Array2<F>,
    y_train: Array1<F>,
    kdtree: Option<KdTree<F>>,
    n_neighbors: usize,
    metric: DistanceMetric,
    weights: WeightFunction,
}

impl<F: Float> Fit<F> for KnnRegressor {
    type Fitted = FittedKnnRegressor<F>;

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

        Ok(FittedKnnRegressor {
            x_train: x.to_owned(),
            y_train: y.to_owned(),
            kdtree,
            n_neighbors: self.n_neighbors,
            metric: self.metric,
            weights: self.weights,
        })
    }
}

impl<F: Float + Send + Sync> Predict<F> for FittedKnnRegressor<F> {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        if x.ncols() != self.x_train.ncols() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.x_train.ncols(),
                x.ncols()
            )));
        }

        // Iterate ndarray rows directly — avoid unnecessary to_vec()
        let predictions: Vec<F> = (0..x.nrows())
            .into_par_iter()
            .map(|i| {
                let row = x.row(i);
                let row_slice = row.as_slice().unwrap();
                let neighbors = self.find_neighbors(row_slice);
                weighted_mean(&neighbors, self.weights)
            })
            .collect();

        Ok(Array1::from_vec(predictions))
    }
}

impl<F: Float> FittedKnnRegressor<F> {
    fn find_neighbors(&self, query: &[F]) -> Vec<(F, F)> {
        if let Some(ref kdtree) = self.kdtree {
            kdtree
                .query_k_nearest(query, self.n_neighbors)
                .into_iter()
                .map(|(dist, idx)| (dist, self.y_train[idx]))
                .collect()
        } else {
            // Brute-force path — batch distance is faster than per-row calls.
            let query_view = ndarray::ArrayView1::from(query);
            let dists = compute_distances_batch(&query_view, &self.x_train, self.metric);
            let mut distances: Vec<(F, F)> = dists
                .into_iter()
                .zip(self.y_train.iter().copied())
                .collect();
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            distances.truncate(self.n_neighbors);
            distances
        }
    }
}

/// Weighted mean of neighbor targets.
fn weighted_mean<F: Float>(neighbors: &[(F, F)], weights: WeightFunction) -> F {
    match weights {
        WeightFunction::Uniform => {
            let sum: F = neighbors
                .iter()
                .map(|&(_, y)| y)
                .fold(F::zero(), |a, b| a + b);
            sum / F::from_usize(neighbors.len()).unwrap()
        }
        WeightFunction::Distance => {
            let mut weight_sum = F::zero();
            let mut value_sum = F::zero();

            for &(dist, y) in neighbors {
                let w = if dist < F::from_f64(1e-15).unwrap() {
                    F::from_f64(1e15).unwrap()
                } else {
                    F::one() / dist
                };
                weight_sum += w;
                value_sum += w * y;
            }

            value_sum / weight_sum
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_knn_regressor_simple() {
        let x_train = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y_train = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let knn = KnnRegressor {
            n_neighbors: 3,
            ..Default::default()
        };
        let fitted = Fit::fit(&knn, &x_train, &y_train).unwrap();

        let x_test = array![[3.0]];
        let preds = fitted.predict(&x_test).unwrap();
        assert_abs_diff_eq!(preds[0], 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_knn_regressor_distance_weights() {
        let x_train = array![[1.0], [3.0], [5.0]];
        let y_train = array![10.0, 20.0, 30.0];

        let knn = KnnRegressor {
            n_neighbors: 3,
            weights: WeightFunction::Distance,
            ..Default::default()
        };
        let fitted = Fit::fit(&knn, &x_train, &y_train).unwrap();

        let x_test = array![[2.9]];
        let preds = fitted.predict(&x_test).unwrap();
        assert!(preds[0] > 18.0 && preds[0] < 22.0);
    }

    #[test]
    fn test_k1_nearest() {
        // k=1 should return the exact target value of the nearest neighbor.
        let x_train = array![[1.0], [3.0], [5.0], [7.0]];
        let y_train = array![10.0, 30.0, 50.0, 70.0];

        let knn = KnnRegressor::new(1);
        let fitted: FittedKnnRegressor<f64> = Fit::fit(&knn, &x_train, &y_train).unwrap();

        // Point at exactly 3.0 should return 30.0.
        let x_test = array![[3.0]];
        let preds = fitted.predict(&x_test).unwrap();
        assert_abs_diff_eq!(preds[0], 30.0, epsilon = 1e-10);

        // Point at 3.1 is closest to 3.0, so should return 30.0.
        let x_test2 = array![[3.1]];
        let preds2 = fitted.predict(&x_test2).unwrap();
        assert_abs_diff_eq!(preds2[0], 30.0, epsilon = 1e-10);
    }

    #[test]
    fn test_distance_weights() {
        // Verify distance weighting produces a different result than uniform.
        let x_train = array![[0.0], [1.0], [10.0]];
        let y_train = array![0.0, 100.0, 200.0];

        // Uniform: mean of all 3 = 100.0
        let knn_uniform = KnnRegressor {
            n_neighbors: 3,
            weights: WeightFunction::Uniform,
            ..Default::default()
        };
        let fitted_uniform: FittedKnnRegressor<f64> =
            Fit::fit(&knn_uniform, &x_train, &y_train).unwrap();

        // Distance-weighted: closer neighbors have more influence.
        let knn_dist = KnnRegressor {
            n_neighbors: 3,
            weights: WeightFunction::Distance,
            ..Default::default()
        };
        let fitted_dist: FittedKnnRegressor<f64> = Fit::fit(&knn_dist, &x_train, &y_train).unwrap();

        // Query at 0.9 — nearest is [1.0] (y=100), then [0.0] (y=0), then [10.0] (y=200).
        let x_test = array![[0.9]];
        let pred_uniform = fitted_uniform.predict(&x_test).unwrap();
        let pred_dist = fitted_dist.predict(&x_test).unwrap();

        // Uniform should be exactly (0 + 100 + 200) / 3.
        assert_abs_diff_eq!(pred_uniform[0], 100.0, epsilon = 1e-10);

        // Distance-weighted should differ from uniform and be biased towards
        // the nearest neighbor [1.0] (y=100). At query 0.9:
        //   dist to [1.0]=0.1 -> w=10.0, dist to [0.0]=0.9 -> w~1.11, dist to [10.0]=9.1 -> w~0.11
        // Weighted mean ~ (10*100 + 1.11*0 + 0.11*200) / (10+1.11+0.11) ~ 91.0
        // The result should be close to 100 (dominated by nearest neighbor).
        assert!(
            (pred_dist[0] - pred_uniform[0]).abs() > 1.0,
            "distance-weighted ({}) should differ from uniform ({})",
            pred_dist[0],
            pred_uniform[0]
        );
        assert!(
            pred_dist[0] > 80.0 && pred_dist[0] < 100.0,
            "distance-weighted prediction ({}) should be close to 100 (nearest neighbor value)",
            pred_dist[0]
        );
    }

    #[test]
    fn test_shape_mismatch_error() {
        let x_train = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let y_train = array![10.0, 20.0, 30.0];

        let knn = KnnRegressor::new(2);
        let fitted: FittedKnnRegressor<f64> = Fit::fit(&knn, &x_train, &y_train).unwrap();

        // Predict with 2 features instead of 3.
        let x_test = array![[1.0, 2.0]];
        let result = fitted.predict(&x_test);
        assert!(result.is_err());
        match result {
            Err(RustMlError::ShapeMismatch(msg)) => {
                assert!(msg.contains("3"), "error should mention expected features");
                assert!(msg.contains("2"), "error should mention actual features");
            }
            other => panic!("expected ShapeMismatch error, got {:?}", other),
        }
    }

    #[test]
    fn test_f32_support() {
        let x_train: Array2<f32> = array![[1.0f32], [2.0], [3.0], [4.0], [5.0]];
        let y_train: Array1<f32> = array![2.0f32, 4.0, 6.0, 8.0, 10.0];

        let knn = KnnRegressor::new(3);
        let fitted: FittedKnnRegressor<f32> = Fit::fit(&knn, &x_train, &y_train).unwrap();

        // At x=3.0, the 3 nearest are [2.0, 3.0, 4.0] with y=[4.0, 6.0, 8.0], mean=6.0.
        let x_test: Array2<f32> = array![[3.0f32]];
        let preds = fitted.predict(&x_test).unwrap();
        assert_abs_diff_eq!(preds[0], 6.0f32, epsilon = 1e-4);
    }
}
