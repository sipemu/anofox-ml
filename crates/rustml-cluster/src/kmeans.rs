use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use rustml_core::{FitUnsupervised, Float, Predict, Result, RustMlError};
use serde::{Serialize, Deserialize};

/// Parameters for K-Means clustering (unfitted state).
///
/// Groups data into `n_clusters` clusters using Lloyd's algorithm with k-means++
/// initialization. The algorithm iteratively assigns points to the nearest centroid
/// and recomputes centroids as cluster means until convergence or `max_iter`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KMeans {
    /// Number of clusters to form.
    pub n_clusters: usize,
    /// Maximum number of Lloyd's iterations.
    pub max_iter: usize,
    /// Convergence tolerance: stop when max centroid shift is below this value.
    pub tol: f64,
    /// Random seed for reproducible centroid initialization.
    pub seed: u64,
}

impl KMeans {
    /// Create a new `KMeans` with the given number of clusters and default parameters.
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            max_iter: 300,
            tol: 1e-4,
            seed: 42,
        }
    }

    /// Set the maximum number of Lloyd's iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the random seed for reproducible centroid initialization.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl Default for KMeans {
    fn default() -> Self {
        Self::new(3)
    }
}

/// Fitted K-Means model containing learned centroids and training metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedKMeans<F: Float> {
    centroids: Array2<F>,
    labels: Array1<F>,
    inertia: F,
    n_iter: usize,
}

impl<F: Float> FittedKMeans<F> {
    /// Returns a reference to the cluster centroids (n_clusters x n_features).
    pub fn centroids(&self) -> &Array2<F> {
        &self.centroids
    }

    /// Returns a reference to the cluster labels assigned to training data.
    pub fn labels(&self) -> &Array1<F> {
        &self.labels
    }

    /// Returns the inertia (sum of squared distances to closest centroid).
    pub fn inertia(&self) -> F {
        self.inertia
    }

    /// Returns the number of iterations performed.
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }
}

/// Compute squared Euclidean distance between two slices using 4-accumulator
/// chunked pattern for SIMD-friendly auto-vectorization.
#[inline]
fn squared_euclidean<F: Float>(a: &[F], b: &[F]) -> F {
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut acc0 = F::zero();
    let mut acc1 = F::zero();
    let mut acc2 = F::zero();
    let mut acc3 = F::zero();

    let mut i = 0;
    for _ in 0..chunks {
        let d0 = a[i] - b[i];
        let d1 = a[i + 1] - b[i + 1];
        let d2 = a[i + 2] - b[i + 2];
        let d3 = a[i + 3] - b[i + 3];
        acc0 += d0 * d0;
        acc1 += d1 * d1;
        acc2 += d2 * d2;
        acc3 += d3 * d3;
        i += 4;
    }

    for j in 0..remainder {
        let d = a[i + j] - b[i + j];
        acc0 += d * d;
    }

    (acc0 + acc1) + (acc2 + acc3)
}

/// Find the index of the nearest centroid for a given point.
#[inline]
fn nearest_centroid<F: Float>(point: &[F], centroids: &Array2<F>) -> (usize, F) {
    let mut best_idx = 0;
    let mut best_dist = F::infinity();
    for (i, centroid) in centroids.rows().into_iter().enumerate() {
        let dist = squared_euclidean(point, centroid.as_slice().unwrap());
        if dist < best_dist {
            best_dist = dist;
            best_idx = i;
        }
    }
    (best_idx, best_dist)
}

/// Update each point's minimum distance given a newly added centroid.
fn update_min_distances<F: Float>(
    x: &Array2<F>,
    min_dists: &mut Array1<F>,
    centroid: ndarray::ArrayView1<F>,
) {
    let centroid_slice = centroid.as_slice().unwrap();
    for i in 0..x.nrows() {
        let dist = squared_euclidean(x.row(i).as_slice().unwrap(), centroid_slice);
        if dist < min_dists[i] {
            min_dists[i] = dist;
        }
    }
}

/// Sample a point index proportional to distance squared (roulette-wheel selection).
fn weighted_random_choice<F: Float>(min_dists: &Array1<F>, rng: &mut StdRng) -> usize {
    let total: F = min_dists.iter().copied().fold(F::zero(), |acc, v| acc + v);
    let threshold = F::from_f64(rng.gen_range(0.0..1.0)).unwrap() * total;
    let mut cumulative = F::zero();
    let mut chosen = min_dists.len() - 1;
    for i in 0..min_dists.len() {
        cumulative += min_dists[i];
        if cumulative >= threshold {
            chosen = i;
            break;
        }
    }
    chosen
}

/// Initialize centroids using the k-means++ algorithm.
fn kmeans_plus_plus<F: Float>(
    x: &Array2<F>,
    n_clusters: usize,
    rng: &mut StdRng,
) -> Array2<F> {
    let n_samples = x.nrows();
    let n_features = x.ncols();
    let mut centroids = Array2::<F>::zeros((n_clusters, n_features));

    // Pick first centroid randomly from data.
    let first_idx = rng.gen_range(0..n_samples);
    centroids.row_mut(0).assign(&x.row(first_idx));

    // Distance from each point to its nearest existing centroid.
    let mut min_dists = Array1::<F>::from_elem(n_samples, F::infinity());

    for k in 1..n_clusters {
        // Update min distances with the centroid just added (index k-1).
        update_min_distances(x, &mut min_dists, centroids.row(k - 1));

        // All remaining points coincide with existing centroids; pick randomly.
        let total: F = min_dists.iter().copied().fold(F::zero(), |acc, v| acc + v);
        if total == F::zero() {
            let idx = rng.gen_range(0..n_samples);
            centroids.row_mut(k).assign(&x.row(idx));
            continue;
        }

        // Sample next centroid proportional to distance squared.
        let chosen = weighted_random_choice(&min_dists, rng);
        centroids.row_mut(k).assign(&x.row(chosen));
    }

    centroids
}

impl<F: Float + Send + Sync> FitUnsupervised<F> for KMeans {
    type Fitted = FittedKMeans<F>;

    fn fit(&self, x: &Array2<F>) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples == 0 {
            return Err(RustMlError::EmptyInput("input array is empty".into()));
        }

        if self.n_clusters == 0 {
            return Err(RustMlError::InvalidParameter(
                "n_clusters must be at least 1".into(),
            ));
        }

        if self.n_clusters > n_samples {
            return Err(RustMlError::InvalidParameter(format!(
                "n_clusters ({}) must not exceed n_samples ({})",
                self.n_clusters, n_samples
            )));
        }

        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut centroids = kmeans_plus_plus(x, self.n_clusters, &mut rng);
        let mut labels = vec![0usize; n_samples];
        let tol = F::from_f64(self.tol).unwrap();
        let mut n_iter = 0;

        for iter in 0..self.max_iter {
            n_iter = iter + 1;

            // Assignment step: parallel assignment of each point to nearest centroid.
            let new_labels: Vec<usize> = (0..n_samples)
                .into_par_iter()
                .map(|i| {
                    let (best_idx, _) =
                        nearest_centroid(x.row(i).as_slice().unwrap(), &centroids);
                    best_idx
                })
                .collect();
            labels = new_labels;

            // Update step: recompute centroids as mean of assigned points.
            let mut new_centroids = Array2::<F>::zeros((self.n_clusters, n_features));
            let mut counts = vec![0usize; self.n_clusters];

            for i in 0..n_samples {
                let cluster = labels[i];
                counts[cluster] += 1;
                for j in 0..n_features {
                    new_centroids[[cluster, j]] += x[[i, j]];
                }
            }

            for k in 0..self.n_clusters {
                if counts[k] > 0 {
                    let count = F::from_usize(counts[k]).unwrap();
                    for j in 0..n_features {
                        new_centroids[[k, j]] /= count;
                    }
                } else {
                    // Empty cluster: keep previous centroid.
                    new_centroids.row_mut(k).assign(&centroids.row(k));
                }
            }

            // Check convergence: max centroid shift.
            let mut max_shift = F::zero();
            for k in 0..self.n_clusters {
                let shift = squared_euclidean(
                    centroids.row(k).as_slice().unwrap(),
                    new_centroids.row(k).as_slice().unwrap(),
                );
                if shift > max_shift {
                    max_shift = shift;
                }
            }

            centroids = new_centroids;

            if max_shift < tol {
                break;
            }
        }

        // Compute final labels and inertia.
        let mut float_labels = Array1::<F>::zeros(n_samples);
        let mut inertia = F::zero();
        for i in 0..n_samples {
            let (best_idx, dist) =
                nearest_centroid(x.row(i).as_slice().unwrap(), &centroids);
            float_labels[i] = F::from_usize(best_idx).unwrap();
            inertia += dist;
        }

        Ok(FittedKMeans {
            centroids,
            labels: float_labels,
            inertia,
            n_iter,
        })
    }
}

impl<F: Float> Predict<F> for FittedKMeans<F> {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        if x.ncols() != self.centroids.ncols() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.centroids.ncols(),
                x.ncols()
            )));
        }

        let n_samples = x.nrows();
        let mut labels = Array1::<F>::zeros(n_samples);
        for i in 0..n_samples {
            let (best_idx, _) =
                nearest_centroid(x.row(i).as_slice().unwrap(), &self.centroids);
            labels[i] = F::from_usize(best_idx).unwrap();
        }
        Ok(labels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array2, Axis};

    /// Build a dataset with 3 well-separated clusters centred at
    /// (0, 0), (10, 10), and (20, 0), each with 30 points.
    fn make_blobs() -> Array2<f64> {
        let mut rng = StdRng::seed_from_u64(123);
        let centers = [(0.0, 0.0), (10.0, 10.0), (20.0, 0.0)];
        let mut data = Array2::<f64>::zeros((90, 2));
        for (c, &(cx, cy)) in centers.iter().enumerate() {
            for i in 0..30 {
                let row = c * 30 + i;
                data[[row, 0]] = cx + (rng.gen_range(-1.0..1.0));
                data[[row, 1]] = cy + (rng.gen_range(-1.0..1.0));
            }
        }
        data
    }

    #[test]
    fn test_finds_three_clusters() {
        let x = make_blobs();
        let kmeans = KMeans::new(3);
        let fitted = FitUnsupervised::<f64>::fit(&kmeans, &x).unwrap();

        // Each of the 3 original groups should have a single unique label.
        let labels = fitted.labels();
        assert_eq!(labels.len(), 90);

        // Cluster 0 points (rows 0..30) should all share one label.
        let label_a = labels[0] as usize;
        for i in 1..30 {
            assert_eq!(labels[i] as usize, label_a, "row {i} has wrong label");
        }

        // Cluster 1 points (rows 30..60) should all share one label.
        let label_b = labels[30] as usize;
        for i in 31..60 {
            assert_eq!(labels[i] as usize, label_b, "row {i} has wrong label");
        }

        // Cluster 2 points (rows 60..90) should all share one label.
        let label_c = labels[60] as usize;
        for i in 61..90 {
            assert_eq!(labels[i] as usize, label_c, "row {i} has wrong label");
        }

        // The three labels should be distinct.
        assert_ne!(label_a, label_b);
        assert_ne!(label_a, label_c);
        assert_ne!(label_b, label_c);
    }

    #[test]
    fn test_predict_assigns_correct_clusters() {
        let x = make_blobs();
        let kmeans = KMeans::new(3);
        let fitted = FitUnsupervised::<f64>::fit(&kmeans, &x).unwrap();

        // New points near each cluster centre should get the same label as
        // the training points from that cluster.
        let new_points = array![[0.5, 0.5], [10.2, 9.8], [19.5, 0.3]];
        let predicted = fitted.predict(&new_points).unwrap();

        // The predicted label for each new point should match the label of
        // a training point from the same original cluster.
        assert_abs_diff_eq!(predicted[0], fitted.labels()[0], epsilon = 1e-10);
        assert_abs_diff_eq!(predicted[1], fitted.labels()[30], epsilon = 1e-10);
        assert_abs_diff_eq!(predicted[2], fitted.labels()[60], epsilon = 1e-10);
    }

    #[test]
    fn test_convergence_before_max_iter() {
        let x = make_blobs();
        let kmeans = KMeans {
            n_clusters: 3,
            max_iter: 300,
            tol: 1e-4,
            seed: 42,
        };
        let fitted = FitUnsupervised::<f64>::fit(&kmeans, &x).unwrap();

        // Well-separated data should converge well before 300 iterations.
        assert!(
            fitted.n_iter() < 300,
            "expected convergence before max_iter, got n_iter={}",
            fitted.n_iter()
        );
    }

    #[test]
    fn test_inertia_lower_than_random() {
        let x = make_blobs();
        let kmeans = KMeans::new(3);
        let fitted = FitUnsupervised::<f64>::fit(&kmeans, &x).unwrap();

        // Compute inertia of a "random" assignment: assign everything to a
        // single centroid at the data mean.
        let n = x.nrows() as f64;
        let mean = x.sum_axis(Axis(0)) / n;
        let random_inertia: f64 = x
            .rows()
            .into_iter()
            .map(|row| squared_euclidean(row.as_slice().unwrap(), mean.as_slice().unwrap()))
            .sum();

        assert!(
            fitted.inertia() < random_inertia,
            "k-means inertia ({}) should be less than single-centroid inertia ({})",
            fitted.inertia(),
            random_inertia
        );
    }

    #[test]
    fn test_error_n_clusters_exceeds_n_samples() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let kmeans = KMeans::new(5);
        let result = FitUnsupervised::<f64>::fit(&kmeans, &x);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("n_clusters"),
            "error should mention n_clusters: {err}"
        );
    }

    #[test]
    fn test_predict_shape_mismatch() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let kmeans = KMeans::new(2);
        let fitted = FitUnsupervised::<f64>::fit(&kmeans, &x).unwrap();

        let bad_input = array![[1.0, 2.0, 3.0]];
        let result = fitted.predict(&bad_input);
        assert!(result.is_err());
    }

    #[test]
    fn test_centroids_accessor() {
        let x = array![[0.0, 0.0], [1.0, 0.0], [10.0, 10.0], [11.0, 10.0]];
        let kmeans = KMeans::new(2);
        let fitted = FitUnsupervised::<f64>::fit(&kmeans, &x).unwrap();

        let centroids = fitted.centroids();
        assert_eq!(centroids.nrows(), 2);
        assert_eq!(centroids.ncols(), 2);
    }

    #[test]
    fn test_reproducibility() {
        let x = make_blobs();
        let kmeans = KMeans::new(3);
        let fitted1 = FitUnsupervised::<f64>::fit(&kmeans, &x).unwrap();
        let fitted2 = FitUnsupervised::<f64>::fit(&kmeans, &x).unwrap();

        // Same seed should produce identical results.
        for (a, b) in fitted1.labels().iter().zip(fitted2.labels().iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-15);
        }
        assert_abs_diff_eq!(fitted1.inertia(), fitted2.inertia(), epsilon = 1e-15);
    }

    mod prop_tests {
        use super::*;
        use proptest::prelude::*;

        /// Generate well-separated cluster data for k clusters.
        fn make_cluster_data(k: usize, seed: u64) -> Array2<f64> {
            let mut rng = StdRng::seed_from_u64(seed);
            let points_per_cluster = 15;
            let n_samples = k * points_per_cluster;
            let mut data = Array2::<f64>::zeros((n_samples, 2));

            for c in 0..k {
                let cx = (c as f64) * 100.0;
                let cy = (c as f64) * 100.0;
                for i in 0..points_per_cluster {
                    let row = c * points_per_cluster + i;
                    data[[row, 0]] = cx + rng.gen_range(-1.0..1.0);
                    data[[row, 1]] = cy + rng.gen_range(-1.0..1.0);
                }
            }
            data
        }

        proptest! {
            #[test]
            fn kmeans_labels_in_range(k in 2..5usize, seed in 0u64..1000) {
                let x = make_cluster_data(k, seed);
                let kmeans = KMeans::new(k).with_seed(seed);
                let fitted = FitUnsupervised::<f64>::fit(&kmeans, &x).unwrap();

                let labels = fitted.labels();
                for (i, &label) in labels.iter().enumerate() {
                    let l = label as usize;
                    prop_assert!(l < k,
                        "label {} at index {} is out of range [0, {})", l, i, k);
                }

                // Also check predict on the same data
                let predicted = fitted.predict(&x).unwrap();
                for (i, &label) in predicted.iter().enumerate() {
                    let l = label as usize;
                    prop_assert!(l < k,
                        "predicted label {} at index {} is out of range [0, {})", l, i, k);
                }
            }

            #[test]
            fn kmeans_deterministic(seed in 0u64..1000) {
                let x = make_cluster_data(3, seed);
                let kmeans = KMeans::new(3).with_seed(seed);

                let fitted1 = FitUnsupervised::<f64>::fit(&kmeans, &x).unwrap();
                let fitted2 = FitUnsupervised::<f64>::fit(&kmeans, &x).unwrap();

                for (a, b) in fitted1.labels().iter().zip(fitted2.labels().iter()) {
                    prop_assert!((a - b).abs() < 1e-15,
                        "non-deterministic labels: {} vs {}", a, b);
                }
                prop_assert!((fitted1.inertia() - fitted2.inertia()).abs() < 1e-15,
                    "non-deterministic inertia: {} vs {}", fitted1.inertia(), fitted2.inertia());
            }
        }
    }
}

impl<F: rustml_core::Float> rustml_core::ClassifierScore<F> for FittedKMeans<F> {}
