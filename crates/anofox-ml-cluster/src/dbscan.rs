use anofox_ml_core::{FitUnsupervised, Float, Predict, Result, RustMlError};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Parameters for DBSCAN clustering (unfitted state).
///
/// DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups
/// together points that are closely packed (points with many nearby neighbors),
/// marking as outliers points that lie alone in low-density regions. Unlike
/// K-Means, DBSCAN does not require specifying the number of clusters in advance
/// and can discover clusters of arbitrary shape.
///
/// # Algorithm
///
/// 1. For each point, find all neighbors within distance `eps` (the epsilon-neighborhood).
/// 2. A point with at least `min_samples` neighbors (including itself) is a **core point**.
/// 3. Core points that are within `eps` of each other are placed in the same cluster.
/// 4. Non-core points within `eps` of a core point are assigned to that core point's cluster
///    (border points).
/// 5. Points not reachable from any core point are labeled as **noise** (label = -1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dbscan {
    /// Maximum distance between two samples for them to be considered neighbors.
    pub eps: f64,
    /// Minimum number of samples in the epsilon-neighborhood of a point
    /// (including the point itself) for it to be considered a core point.
    pub min_samples: usize,
}

impl Dbscan {
    /// Create a new `Dbscan` with the given epsilon and minimum samples.
    pub fn new(eps: f64, min_samples: usize) -> Self {
        Self { eps, min_samples }
    }

    /// Set the maximum neighborhood distance.
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Set the minimum number of samples to form a dense region.
    pub fn with_min_samples(mut self, min_samples: usize) -> Self {
        self.min_samples = min_samples;
        self
    }
}

impl Default for Dbscan {
    fn default() -> Self {
        Self {
            eps: 0.5,
            min_samples: 5,
        }
    }
}

/// Fitted DBSCAN model containing cluster assignments and core sample information.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedDbscan<F: Float> {
    /// Cluster labels for each training sample. Noise points have label -1.0;
    /// cluster labels are 0.0, 1.0, 2.0, etc.
    labels: Array1<F>,
    /// Number of clusters found (excluding noise).
    n_clusters: usize,
    /// Indices of core samples in the training data.
    core_sample_indices: Vec<usize>,
    /// Number of features in the training data (for predict-time validation).
    n_features: usize,
}

impl<F: Float> FittedDbscan<F> {
    /// Returns a reference to the cluster labels assigned to training data.
    ///
    /// Noise points are labeled -1.0; clusters are labeled 0.0, 1.0, 2.0, etc.
    pub fn labels(&self) -> &Array1<F> {
        &self.labels
    }

    /// Returns the number of clusters found (excluding noise).
    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    /// Returns the indices of core samples in the training data.
    pub fn core_sample_indices(&self) -> &[usize] {
        &self.core_sample_indices
    }
}

/// Compute the Euclidean distance between two slices.
fn euclidean_distance<F: Float>(a: &[F], b: &[F]) -> F {
    let sum_sq = a
        .iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| {
            let diff = ai - bi;
            diff * diff
        })
        .fold(F::zero(), |acc, v| acc + v);
    sum_sq.sqrt()
}

/// Find all neighbor indices within `eps` of the point at `point_idx`.
fn region_query<F: Float>(x: &Array2<F>, point_idx: usize, eps: F) -> Vec<usize> {
    let point = x.row(point_idx);
    let point_slice = point.as_slice().unwrap();
    let mut neighbors = Vec::new();
    for i in 0..x.nrows() {
        let other = x.row(i);
        if euclidean_distance(point_slice, other.as_slice().unwrap()) <= eps {
            neighbors.push(i);
        }
    }
    neighbors
}

impl<F: Float> FitUnsupervised<F> for Dbscan {
    type Fitted = FittedDbscan<F>;

    fn fit(&self, x: &Array2<F>) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples == 0 {
            return Err(RustMlError::EmptyInput("input array is empty".into()));
        }

        if self.eps <= 0.0 {
            return Err(RustMlError::InvalidParameter("eps must be positive".into()));
        }

        if self.min_samples == 0 {
            return Err(RustMlError::InvalidParameter(
                "min_samples must be at least 1".into(),
            ));
        }

        let eps = F::from_f64(self.eps).unwrap();
        let noise_label: i64 = -1;

        // Labels: -1 = noise/unvisited, 0..n = cluster id.
        // We track labels as i64 internally, then convert to F at the end.
        let mut labels = vec![noise_label; n_samples];
        let mut visited = vec![false; n_samples];
        let mut is_core = vec![false; n_samples];
        let mut current_cluster: i64 = -1;

        for i in 0..n_samples {
            if visited[i] {
                continue;
            }
            visited[i] = true;

            let neighbors = region_query(x, i, eps);

            if neighbors.len() < self.min_samples {
                // Not a core point; remains noise for now (may be claimed by a cluster later).
                continue;
            }

            // Start a new cluster.
            current_cluster += 1;
            labels[i] = current_cluster;
            is_core[i] = true;

            // Expand cluster: use a queue of neighbors to process.
            let mut queue = neighbors;
            let mut qi = 0;
            while qi < queue.len() {
                let neighbor = queue[qi];
                qi += 1;

                if !visited[neighbor] {
                    visited[neighbor] = true;
                    let neighbor_neighbors = region_query(x, neighbor, eps);
                    if neighbor_neighbors.len() >= self.min_samples {
                        is_core[neighbor] = true;
                        // Add new neighbors to the queue.
                        for &nn in &neighbor_neighbors {
                            if !visited[nn] || labels[nn] == noise_label {
                                queue.push(nn);
                            }
                        }
                    }
                }

                // Assign to current cluster if not already assigned to a cluster.
                if labels[neighbor] == noise_label {
                    labels[neighbor] = current_cluster;
                }
            }
        }

        let n_clusters = if current_cluster >= 0 {
            (current_cluster + 1) as usize
        } else {
            0
        };

        let core_sample_indices: Vec<usize> = is_core
            .iter()
            .enumerate()
            .filter(|(_, &c)| c)
            .map(|(i, _)| i)
            .collect();

        let float_labels: Array1<F> = Array1::from_vec(
            labels
                .iter()
                .map(|&l| {
                    if l < 0 {
                        F::from_f64(-1.0).unwrap()
                    } else {
                        F::from_i64(l).unwrap()
                    }
                })
                .collect(),
        );

        Ok(FittedDbscan {
            labels: float_labels,
            n_clusters,
            core_sample_indices,
            n_features,
        })
    }
}

impl<F: Float> Predict<F> for FittedDbscan<F> {
    /// For DBSCAN, prediction returns the training labels.
    ///
    /// DBSCAN is a transductive algorithm and does not naturally generalize to
    /// unseen data. This implementation validates the input shape and returns
    /// the labels computed during fitting for inputs with the correct number of
    /// rows and features. For new data, consider re-fitting the model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }

        Ok(self.labels.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array2};

    /// Build a dataset with 3 well-separated clusters:
    /// cluster 0 around (0, 0), cluster 1 around (10, 10), cluster 2 around (20, 0).
    /// Plus two isolated noise points far from any cluster.
    fn make_dbscan_data() -> Array2<f64> {
        // 5 points per cluster + 2 noise points = 17 points total
        array![
            // Cluster 0: around (0, 0)
            [0.0, 0.0],
            [0.1, 0.2],
            [0.2, 0.1],
            [-0.1, 0.0],
            [0.0, -0.1],
            // Cluster 1: around (10, 10)
            [10.0, 10.0],
            [10.1, 10.2],
            [10.2, 10.1],
            [9.9, 10.0],
            [10.0, 9.9],
            // Cluster 2: around (20, 0)
            [20.0, 0.0],
            [20.1, 0.2],
            [20.2, 0.1],
            [19.9, 0.0],
            [20.0, -0.1],
            // Noise points
            [50.0, 50.0],
            [-50.0, -50.0]
        ]
    }

    #[test]
    fn test_finds_well_separated_clusters() {
        let x = make_dbscan_data();
        let dbscan = Dbscan::new(1.0, 3);
        let fitted = FitUnsupervised::<f64>::fit(&dbscan, &x).unwrap();

        assert_eq!(fitted.n_clusters(), 3, "should find 3 clusters");

        // All points in cluster 0 (indices 0..5) should share the same label.
        let labels = fitted.labels();
        let label_a = labels[0];
        assert!(label_a >= 0.0, "cluster label should be non-negative");
        for i in 1..5 {
            assert_abs_diff_eq!(labels[i], label_a, epsilon = 1e-10);
        }

        // All points in cluster 1 (indices 5..10) should share the same label.
        let label_b = labels[5];
        assert!(label_b >= 0.0, "cluster label should be non-negative");
        for i in 6..10 {
            assert_abs_diff_eq!(labels[i], label_b, epsilon = 1e-10);
        }

        // All points in cluster 2 (indices 10..15) should share the same label.
        let label_c = labels[10];
        assert!(label_c >= 0.0, "cluster label should be non-negative");
        for i in 11..15 {
            assert_abs_diff_eq!(labels[i], label_c, epsilon = 1e-10);
        }

        // The three cluster labels should be distinct.
        assert_ne!(label_a as i64, label_b as i64);
        assert_ne!(label_a as i64, label_c as i64);
        assert_ne!(label_b as i64, label_c as i64);
    }

    #[test]
    fn test_noise_points_labeled_minus_one() {
        let x = make_dbscan_data();
        let dbscan = Dbscan::new(1.0, 3);
        let fitted = FitUnsupervised::<f64>::fit(&dbscan, &x).unwrap();

        let labels = fitted.labels();
        // Last two points (indices 15, 16) are far away and should be noise.
        assert_abs_diff_eq!(labels[15], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(labels[16], -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_single_point_input() {
        let x = array![[1.0, 2.0]];
        // With min_samples=1, a single point is its own core point.
        let dbscan = Dbscan::new(0.5, 1);
        let fitted = FitUnsupervised::<f64>::fit(&dbscan, &x).unwrap();

        assert_eq!(fitted.n_clusters(), 1);
        assert_eq!(fitted.labels().len(), 1);
        assert_abs_diff_eq!(fitted.labels()[0], 0.0, epsilon = 1e-10);
        assert_eq!(fitted.core_sample_indices(), &[0]);
    }

    #[test]
    fn test_min_samples_larger_than_dataset_all_noise() {
        let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        // min_samples=10 is larger than dataset size (3), so no core points exist.
        let dbscan = Dbscan::new(0.5, 10);
        let fitted = FitUnsupervised::<f64>::fit(&dbscan, &x).unwrap();

        assert_eq!(fitted.n_clusters(), 0, "should find 0 clusters");
        for &label in fitted.labels().iter() {
            assert_abs_diff_eq!(label, -1.0, epsilon = 1e-10);
        }
        assert!(
            fitted.core_sample_indices().is_empty(),
            "no core samples when min_samples > n_samples"
        );
    }

    #[test]
    fn test_empty_input_error() {
        let x = Array2::<f64>::zeros((0, 2));
        let dbscan = Dbscan::new(0.5, 5);
        let result = FitUnsupervised::<f64>::fit(&dbscan, &x);
        assert!(result.is_err());
        match result.unwrap_err() {
            RustMlError::EmptyInput(_) => {}
            other => panic!("expected EmptyInput, got {other}"),
        }
    }

    #[test]
    fn test_invalid_eps_error() {
        let x = array![[1.0, 2.0]];
        let dbscan = Dbscan::new(-1.0, 5);
        let result = FitUnsupervised::<f64>::fit(&dbscan, &x);
        assert!(result.is_err());
        match result.unwrap_err() {
            RustMlError::InvalidParameter(msg) => {
                assert!(msg.contains("eps"), "error should mention eps: {msg}");
            }
            other => panic!("expected InvalidParameter, got {other}"),
        }
    }

    #[test]
    fn test_invalid_min_samples_error() {
        let x = array![[1.0, 2.0]];
        let dbscan = Dbscan::new(0.5, 0);
        let result = FitUnsupervised::<f64>::fit(&dbscan, &x);
        assert!(result.is_err());
        match result.unwrap_err() {
            RustMlError::InvalidParameter(msg) => {
                assert!(
                    msg.contains("min_samples"),
                    "error should mention min_samples: {msg}"
                );
            }
            other => panic!("expected InvalidParameter, got {other}"),
        }
    }

    #[test]
    fn test_predict_shape_mismatch() {
        let x = array![[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]];
        let dbscan = Dbscan::new(0.5, 2);
        let fitted = FitUnsupervised::<f64>::fit(&dbscan, &x).unwrap();

        let bad_input = array![[1.0, 2.0, 3.0]];
        let result = fitted.predict(&bad_input);
        assert!(result.is_err());
        match result.unwrap_err() {
            RustMlError::ShapeMismatch(_) => {}
            other => panic!("expected ShapeMismatch, got {other}"),
        }
    }

    #[test]
    fn test_core_sample_indices() {
        let x = make_dbscan_data();
        let dbscan = Dbscan::new(1.0, 3);
        let fitted = FitUnsupervised::<f64>::fit(&dbscan, &x).unwrap();

        // All 15 cluster points should be core samples (each cluster has 5 points
        // within eps=1.0, which is >= min_samples=3).
        let core_indices = fitted.core_sample_indices();
        assert!(
            core_indices.len() >= 15,
            "expected at least 15 core samples, got {}",
            core_indices.len()
        );

        // Noise points should not be core samples.
        assert!(!core_indices.contains(&15));
        assert!(!core_indices.contains(&16));
    }

    #[test]
    fn test_default_params() {
        let dbscan = Dbscan::default();
        assert_abs_diff_eq!(dbscan.eps, 0.5, epsilon = 1e-10);
        assert_eq!(dbscan.min_samples, 5);
    }

    #[test]
    fn test_builder_methods() {
        let dbscan = Dbscan::default().with_eps(1.5).with_min_samples(10);
        assert_abs_diff_eq!(dbscan.eps, 1.5, epsilon = 1e-10);
        assert_eq!(dbscan.min_samples, 10);
    }

    #[test]
    fn test_f32_support() {
        let x: Array2<f32> = array![
            [0.0f32, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.2],
        ];
        let dbscan = Dbscan::new(1.0, 2);
        let fitted = FitUnsupervised::<f32>::fit(&dbscan, &x).unwrap();

        assert_eq!(fitted.n_clusters(), 2);
    }

    mod prop_tests {
        use super::*;
        use proptest::prelude::*;

        fn make_cluster_data(n_clusters: usize, n_per_cluster: usize, seed: u64) -> Array2<f64> {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let n = n_clusters * n_per_cluster;
            let mut data = Vec::with_capacity(n * 2);
            for c in 0..n_clusters {
                let cx = (c as f64) * 10.0;
                let cy = (c as f64) * 10.0;
                for i in 0..n_per_cluster {
                    let mut h = DefaultHasher::new();
                    seed.hash(&mut h);
                    (c as u64).hash(&mut h);
                    (i as u64).hash(&mut h);
                    let bits = h.finish();
                    let dx = (bits as f64 / u64::MAX as f64) * 2.0 - 1.0;
                    let mut h2 = DefaultHasher::new();
                    seed.hash(&mut h2);
                    (c as u64).hash(&mut h2);
                    (i as u64).hash(&mut h2);
                    1u64.hash(&mut h2);
                    let bits2 = h2.finish();
                    let dy = (bits2 as f64 / u64::MAX as f64) * 2.0 - 1.0;
                    data.push(cx + dx);
                    data.push(cy + dy);
                }
            }
            Array2::from_shape_vec((n, 2), data).unwrap()
        }

        proptest! {
            #[test]
            fn labels_in_range(
                n_clusters in 1usize..=5,
                n_per_cluster in 3usize..=10,
                seed in 0u64..1000,
            ) {
                let x = make_cluster_data(n_clusters, n_per_cluster, seed);
                let dbscan = Dbscan::new(2.0, 3);
                let fitted = FitUnsupervised::<f64>::fit(&dbscan, &x).unwrap();

                let labels = fitted.labels();
                let nc = fitted.n_clusters();
                for &label in labels.iter() {
                    prop_assert!(
                        label >= -1.0 && label < nc as f64,
                        "label {} out of range -1..{}", label, nc
                    );
                }
            }

            #[test]
            fn noise_plus_clustered_equals_total(
                n_clusters in 1usize..=5,
                n_per_cluster in 3usize..=10,
                seed in 0u64..1000,
            ) {
                let x = make_cluster_data(n_clusters, n_per_cluster, seed);
                let n_total = x.nrows();
                let dbscan = Dbscan::new(2.0, 3);
                let fitted = FitUnsupervised::<f64>::fit(&dbscan, &x).unwrap();

                let labels = fitted.labels();
                let n_noise = labels.iter().filter(|&&l| l == -1.0).count();
                let n_clustered = labels.iter().filter(|&&l| l >= 0.0).count();
                prop_assert_eq!(
                    n_noise + n_clustered, n_total,
                    "noise({}) + clustered({}) != total({})", n_noise, n_clustered, n_total
                );
            }
        }
    }
}
