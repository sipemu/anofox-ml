use anofox_ml_core::{Fit, Float, Predict, Result};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::distance::DistanceMetric;
use crate::index::{NeighborAlgorithm, NeighborIndex};
use crate::knn_classifier::{check_knn_fit, neighbor_weights, WeightFunction};

/// KNN regressor parameters (unfitted state).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KnnRegressor {
    pub n_neighbors: usize,
    pub metric: DistanceMetric,
    pub weights: WeightFunction,
    /// Neighbour search algorithm (sklearn's `algorithm`).
    #[serde(default)]
    pub algorithm: NeighborAlgorithm,
}

impl KnnRegressor {
    /// Create a new `KnnRegressor` with the given number of neighbors and default parameters.
    pub fn new(k: usize) -> Self {
        Self {
            n_neighbors: k,
            metric: DistanceMetric::default(),
            weights: WeightFunction::default(),
            algorithm: NeighborAlgorithm::default(),
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

    /// Set the neighbour search algorithm.
    pub fn with_algorithm(mut self, algorithm: NeighborAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }
}

impl Default for KnnRegressor {
    fn default() -> Self {
        Self::new(5)
    }
}

/// Fitted KNN regressor — stores training data (lazy learner) in a
/// [`NeighborIndex`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedKnnRegressor<F: Float> {
    index: NeighborIndex<F>,
    y_train: Array1<F>,
    n_neighbors: usize,
    weights: WeightFunction,
}

impl<F: Float> Fit<F> for KnnRegressor {
    type Fitted = FittedKnnRegressor<F>;

    fn fit_view(&self, x: ArrayView2<F>, y: ArrayView1<F>) -> Result<Self::Fitted> {
        check_knn_fit(x, y, self.n_neighbors)?;
        Ok(FittedKnnRegressor {
            index: NeighborIndex::new(x, self.metric, self.algorithm, Some(self.n_neighbors))?,
            y_train: y.to_owned(),
            n_neighbors: self.n_neighbors,
            weights: self.weights,
        })
    }
}

/// Weighted mean of the targets `y[idx]`.
pub(crate) fn weighted_mean<F: Float>(
    y: &Array1<F>,
    idx: impl Iterator<Item = usize>,
    dists: &[F],
    weights: WeightFunction,
) -> F {
    let w = neighbor_weights(dists, weights);
    let (num, den) = idx
        .zip(&w)
        .fold((F::zero(), F::zero()), |(n, d), (j, &wj)| {
            (n + wj * y[j], d + wj)
        });
    num / den
}

impl<F: Float + Send + Sync> Predict<F> for FittedKnnRegressor<F> {
    fn predict_view(&self, x: ArrayView2<F>) -> Result<Array1<F>> {
        let (dist, idx) = self.index.kneighbors(x, self.n_neighbors)?;
        Ok(Array1::from_shape_fn(x.nrows(), |i| {
            let d = dist.row(i).to_vec();
            weighted_mean(&self.y_train, idx.row(i).iter().copied(), &d, self.weights)
        }))
    }
}

impl<F: Float> FittedKnnRegressor<F> {
    /// The neighbour index.
    pub fn index(&self) -> &NeighborIndex<F> {
        &self.index
    }

    /// Distances and training indices of the `k` (default `n_neighbors`)
    /// nearest training samples of every row of `x`.
    pub fn kneighbors(
        &self,
        x: &Array2<F>,
        k: Option<usize>,
    ) -> Result<(Array2<F>, Array2<usize>)> {
        self.index
            .kneighbors(x.view(), k.unwrap_or(self.n_neighbors))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anofox_ml_core::RustMlError;
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

    /// Regression for #42: Fortran-order input used to panic.
    #[test]
    fn test_fortran_order_input() {
        use ndarray::{Array2, ShapeBuilder};
        let x = array![[0.0, 0.0], [1.0, 0.5], [2.0, 1.0], [3.0, 1.5], [4.0, 2.0]];
        let y = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let v: Vec<f64> = x.t().iter().copied().collect();
        let xf = Array2::from_shape_vec((5, 2).f(), v).unwrap();
        assert!(!xf.is_standard_layout());
        for metric in [DistanceMetric::Euclidean, DistanceMetric::Manhattan] {
            let knn = KnnRegressor::new(2).with_metric(metric);
            let a = Fit::fit(&knn, &x, &y).unwrap().predict(&x).unwrap();
            let b = Fit::fit(&knn, &xf, &y).unwrap().predict(&xf).unwrap();
            assert_eq!(a, b, "metric={metric:?}");
        }
    }
}
