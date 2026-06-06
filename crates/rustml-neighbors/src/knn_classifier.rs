use ndarray::{Array1, Array2};
use rayon::prelude::*;
use rustml_core::{Fit, Float, Predict, Result, RustMlError};

use crate::distance::{compute_distances_batch, DistanceMetric};
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

        // Parallel prediction using rayon — iterate ndarray rows directly
        let predictions: Vec<F> = (0..x.nrows())
            .into_par_iter()
            .map(|i| {
                let row = x.row(i);
                let row_slice = row.as_slice().unwrap();
                let neighbors = self.find_neighbors(row_slice);
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

/// Weighted majority vote among neighbors.
/// Uses HashMap with f64 bit representation for O(1) class lookup.
#[inline]
fn weighted_majority_vote<F: Float>(neighbors: &[(F, F)], weights: WeightFunction) -> F {
    use std::collections::HashMap;

    // Accumulate weights per class using HashMap for O(1) lookup
    let mut class_weights: HashMap<u64, (F, F)> = HashMap::new(); // key -> (class_value, total_weight)

    for &(dist, class) in neighbors {
        let w = match weights {
            WeightFunction::Uniform => F::one(),
            WeightFunction::Distance => {
                if dist < F::from_f64(1e-15).unwrap() {
                    F::from_f64(1e15).unwrap()
                } else {
                    F::one() / dist
                }
            }
        };
        let key = class.to_f64().unwrap().to_bits();
        class_weights
            .entry(key)
            .and_modify(|e| e.1 += w)
            .or_insert((class, w));
    }

    class_weights
        .into_values()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()
        .0
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

    #[test]
    fn test_k1_nearest_single() {
        // k=1 should return the label of the single nearest neighbor.
        let x_train = array![[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]];
        let y_train = array![0.0, 1.0, 2.0];

        let knn = KnnClassifier::new(1);
        let fitted: FittedKnnClassifier<f64> = Fit::fit(&knn, &x_train, &y_train).unwrap();

        // Point nearest to [5.0, 5.0] (class 1).
        let x_test = array![[4.9, 4.9]];
        let preds = fitted.predict(&x_test).unwrap();
        assert_abs_diff_eq!(preds[0], 1.0, epsilon = 1e-10);

        // Point nearest to [0.0, 0.0] (class 0).
        let x_test2 = array![[0.1, 0.1]];
        let preds2 = fitted.predict(&x_test2).unwrap();
        assert_abs_diff_eq!(preds2[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_k_equals_n_samples() {
        // k = n_samples, all neighbors participate in the vote.
        let x_train = array![[0.0], [1.0], [2.0], [3.0]];
        let y_train = array![0.0, 0.0, 0.0, 1.0];

        let knn = KnnClassifier::new(4); // k = n_samples = 4
        let fitted: FittedKnnClassifier<f64> = Fit::fit(&knn, &x_train, &y_train).unwrap();

        // Majority is class 0 (3 vs 1), should predict 0 everywhere with uniform weights.
        let x_test = array![[1.5]];
        let preds = fitted.predict(&x_test).unwrap();
        assert_abs_diff_eq!(preds[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cosine_distance() {
        // Cosine distance cares about angle, not magnitude.
        // Two directions: roughly (1,0) and (0,1).
        let x_train = array![
            [10.0, 0.1], // class 0 — points along x-axis
            [20.0, 0.2], // class 0
            [0.1, 10.0], // class 1 — points along y-axis
            [0.2, 20.0]  // class 1
        ];
        let y_train = array![0.0, 0.0, 1.0, 1.0];

        let knn = KnnClassifier::new(2).with_metric(DistanceMetric::Cosine);
        let fitted: FittedKnnClassifier<f64> = Fit::fit(&knn, &x_train, &y_train).unwrap();

        // A point along x-axis should be class 0.
        let x_test = array![[5.0, 0.05]];
        let preds = fitted.predict(&x_test).unwrap();
        assert_abs_diff_eq!(preds[0], 0.0, epsilon = 1e-10);

        // A point along y-axis should be class 1.
        let x_test2 = array![[0.05, 5.0]];
        let preds2 = fitted.predict(&x_test2).unwrap();
        assert_abs_diff_eq!(preds2[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_duplicate_training_points() {
        // Duplicate points should not cause issues.
        let x_train = array![
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [10.0, 10.0],
            [10.0, 10.0]
        ];
        let y_train = array![0.0, 0.0, 0.0, 1.0, 1.0];

        let knn = KnnClassifier::new(3);
        let fitted: FittedKnnClassifier<f64> = Fit::fit(&knn, &x_train, &y_train).unwrap();

        // Predicting at a duplicate point should work.
        let x_test = array![[1.0, 1.0]];
        let preds = fitted.predict(&x_test).unwrap();
        assert_abs_diff_eq!(preds[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_f32_support() {
        let x_train: Array2<f32> = array![[0.0f32, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1]];
        let y_train: Array1<f32> = array![0.0f32, 0.0, 1.0, 1.0];

        let knn = KnnClassifier::new(2);
        let fitted: FittedKnnClassifier<f32> = Fit::fit(&knn, &x_train, &y_train).unwrap();

        let x_test: Array2<f32> = array![[0.05f32, 0.05], [10.05, 10.05]];
        let preds = fitted.predict(&x_test).unwrap();
        assert_abs_diff_eq!(preds[0], 0.0f32, epsilon = 1e-5);
        assert_abs_diff_eq!(preds[1], 1.0f32, epsilon = 1e-5);
    }

    #[test]
    fn test_empty_input_error() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<f64>::zeros(0);

        let knn = KnnClassifier::new(3);
        let result: Result<FittedKnnClassifier<f64>> = Fit::fit(&knn, &x, &y);
        assert!(result.is_err());
        match result {
            Err(RustMlError::EmptyInput(_)) => {}
            other => panic!("expected EmptyInput error, got {:?}", other),
        }
    }

    #[test]
    fn test_k_zero_error() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0];

        let knn = KnnClassifier::new(0);
        let result: Result<FittedKnnClassifier<f64>> = Fit::fit(&knn, &x, &y);
        assert!(result.is_err());
        match result {
            Err(RustMlError::InvalidParameter(msg)) => {
                assert!(msg.contains("0") || msg.to_lowercase().contains("neighbor"));
            }
            other => panic!("expected InvalidParameter error, got {:?}", other),
        }
    }

    #[test]
    fn test_k_too_large_error() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0];

        let knn = KnnClassifier::new(5); // k=5 but only 2 samples
        let result: Result<FittedKnnClassifier<f64>> = Fit::fit(&knn, &x, &y);
        assert!(result.is_err());
        match result {
            Err(RustMlError::InvalidParameter(msg)) => {
                assert!(msg.contains("5"), "error should mention k value");
                assert!(msg.contains("2"), "error should mention sample count");
            }
            other => panic!("expected InvalidParameter error, got {:?}", other),
        }
    }

    mod prop_tests {
        use super::*;
        use proptest::prelude::*;
        use std::collections::HashSet;

        /// Generate deterministic training data for classification.
        fn make_classification_data(
            n_samples: usize,
            n_features: usize,
            n_classes: usize,
            seed: u64,
        ) -> (Array2<f64>, Array1<f64>) {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let mut x_data = Vec::with_capacity(n_samples * n_features);
            let mut y_data = Vec::with_capacity(n_samples);

            for i in 0..n_samples {
                for j in 0..n_features {
                    let mut h = DefaultHasher::new();
                    seed.hash(&mut h);
                    (i as u64).hash(&mut h);
                    (j as u64).hash(&mut h);
                    let bits = h.finish();
                    let v = (bits as f64 / u64::MAX as f64) * 20.0 - 10.0;
                    x_data.push(v);
                }
                let mut h = DefaultHasher::new();
                seed.hash(&mut h);
                (i as u64).hash(&mut h);
                0xDEAD_BEEFu64.hash(&mut h);
                let label = (h.finish() % n_classes as u64) as f64;
                y_data.push(label);
            }

            let x = Array2::from_shape_vec((n_samples, n_features), x_data).unwrap();
            let y = Array1::from_vec(y_data);
            (x, y)
        }

        proptest! {
            #[test]
            fn predictions_are_valid_training_labels(
                n_samples in 4..30usize,
                n_features in 1..5usize,
                k in 1..4usize,
                seed in 0u64..1000,
            ) {
                let n_classes = 3;
                let (x, y) = make_classification_data(n_samples, n_features, n_classes, seed);

                // k must not exceed n_samples
                let k = k.min(n_samples);

                // Collect unique training labels
                let train_labels: HashSet<u64> = y.iter()
                    .map(|&v| v.to_bits())
                    .collect();

                let knn = KnnClassifier::new(k);
                let fitted: FittedKnnClassifier<f64> = Fit::fit(&knn, &x, &y).unwrap();
                let preds = fitted.predict(&x).unwrap();

                for (i, &p) in preds.iter().enumerate() {
                    prop_assert!(
                        train_labels.contains(&p.to_bits()),
                        "prediction {} at index {} is not a valid training label",
                        p, i
                    );
                }
            }

            #[test]
            fn k1_returns_exact_nearest_neighbor(
                n_samples in 2..30usize,
                n_features in 1..5usize,
                seed in 0u64..1000,
            ) {
                let n_classes = 3;
                let (x, y) = make_classification_data(n_samples, n_features, n_classes, seed);

                let knn = KnnClassifier::new(1);
                let fitted: FittedKnnClassifier<f64> = Fit::fit(&knn, &x, &y).unwrap();

                // Predicting on the training data with k=1 should return each point's own label,
                // because the nearest neighbor of a training point is itself (distance = 0).
                let preds = fitted.predict(&x).unwrap();

                for (i, (&pred, &expected)) in preds.iter().zip(y.iter()).enumerate() {
                    prop_assert!(
                        (pred - expected).abs() < 1e-10,
                        "k=1 prediction at index {} was {} but expected {}",
                        i, pred, expected
                    );
                }
            }
        }
    }
}
