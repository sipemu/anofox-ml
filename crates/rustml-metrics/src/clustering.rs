use ndarray::{Array1, Array2};
use rustml_core::{Float, Result, RustMlError};

/// Mean silhouette coefficient over all samples.
///
/// For each sample `i`, the silhouette coefficient is defined as:
///
/// `s(i) = (b(i) - a(i)) / max(a(i), b(i))`
///
/// where:
/// - `a(i)` is the mean distance from sample `i` to all other samples in the
///   same cluster (mean intra-cluster distance).
/// - `b(i)` is the minimum over all other clusters of the mean distance from
///   sample `i` to all samples in that cluster (mean nearest-cluster distance).
///
/// The returned value is the mean of `s(i)` over all samples, in the range `[-1, 1]`.
/// A score near +1 indicates well-separated clusters; near 0 indicates overlapping
/// clusters; near -1 indicates misassigned samples.
///
/// # Arguments
///
/// * `x` - A 2D array of shape `(n_samples, n_features)` containing the data points.
/// * `labels` - A 1D array of shape `(n_samples,)` containing cluster labels for each sample.
///
/// # Errors
///
/// Returns an error if:
/// - `x` has zero rows or `labels` is empty.
/// - The number of rows in `x` does not match the length of `labels`.
/// - There are fewer than 2 distinct clusters.
pub fn silhouette_score<F: Float>(x: &Array2<F>, labels: &Array1<F>) -> Result<F> {
    let n_samples = x.nrows();

    if n_samples == 0 || labels.is_empty() {
        return Err(RustMlError::EmptyInput("input arrays are empty".into()));
    }
    if n_samples != labels.len() {
        return Err(RustMlError::ShapeMismatch(format!(
            "x has {} rows but labels has length {}",
            n_samples,
            labels.len()
        )));
    }

    let eps = F::from_f64(1e-9).unwrap();

    // Identify unique cluster labels
    let mut unique_labels: Vec<F> = labels.iter().copied().collect();
    unique_labels.sort_by(|a, b| a.partial_cmp(b).unwrap());
    unique_labels.dedup_by(|a, b| (*a - *b).abs() < eps);

    if unique_labels.len() < 2 {
        return Err(RustMlError::InvalidParameter(
            "silhouette score requires at least 2 clusters".into(),
        ));
    }

    // Pre-compute cluster membership: for each sample, which cluster index it belongs to
    let cluster_indices: Vec<usize> = labels
        .iter()
        .map(|&l| {
            unique_labels
                .iter()
                .position(|&c| (c - l).abs() < eps)
                .unwrap()
        })
        .collect();

    let n_clusters = unique_labels.len();
    let n_features = x.ncols();

    // Compute pairwise Euclidean distances and accumulate per-cluster sums
    // For each sample i, we need:
    //   - sum of distances to samples in the same cluster (to compute a(i))
    //   - sum of distances to samples in each other cluster (to compute b(i))
    //
    // We also need the count of samples per cluster.
    let mut cluster_counts = vec![0usize; n_clusters];
    for &ci in &cluster_indices {
        cluster_counts[ci] += 1;
    }

    let mut silhouette_sum = F::zero();

    for i in 0..n_samples {
        let ci = cluster_indices[i];

        // Accumulate distances from sample i to all other samples, grouped by cluster
        let mut dist_sums = vec![F::zero(); n_clusters];

        for j in 0..n_samples {
            if i == j {
                continue;
            }
            // Euclidean distance
            let mut d_sq = F::zero();
            for f in 0..n_features {
                let diff = x[[i, f]] - x[[j, f]];
                d_sq += diff * diff;
            }
            let d = d_sq.sqrt();
            dist_sums[cluster_indices[j]] += d;
        }

        // a(i): mean intra-cluster distance
        let a_i = if cluster_counts[ci] > 1 {
            dist_sums[ci] / F::from_usize(cluster_counts[ci] - 1).unwrap()
        } else {
            F::zero()
        };

        // b(i): minimum mean distance to any other cluster
        let mut b_i = F::infinity();
        for k in 0..n_clusters {
            if k == ci || cluster_counts[k] == 0 {
                continue;
            }
            let mean_dist = dist_sums[k] / F::from_usize(cluster_counts[k]).unwrap();
            if mean_dist < b_i {
                b_i = mean_dist;
            }
        }

        // s(i) = (b(i) - a(i)) / max(a(i), b(i))
        let max_ab = if a_i > b_i { a_i } else { b_i };
        let s_i = if max_ab > F::zero() {
            (b_i - a_i) / max_ab
        } else {
            F::zero()
        };

        silhouette_sum += s_i;
    }

    Ok(silhouette_sum / F::from_usize(n_samples).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn test_silhouette_well_separated() {
        // Two well-separated clusters: cluster 0 near origin, cluster 1 far away.
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, // cluster 0
                10.0, 10.0, 10.1, 10.0, 10.0, 10.1, // cluster 1
            ],
        )
        .unwrap();
        let labels = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let score: f64 = silhouette_score(&x, &labels).unwrap();
        // Well-separated clusters should have silhouette close to 1.
        assert!(score > 0.9, "Expected score > 0.9, got {}", score);
    }

    #[test]
    fn test_silhouette_overlapping() {
        // Two overlapping clusters: all points at similar locations.
        let x = Array2::from_shape_vec(
            (4, 2),
            vec![
                0.0, 0.0, 0.1, 0.1, 0.0, 0.1, 0.1, 0.0,
            ],
        )
        .unwrap();
        let labels = array![0.0, 0.0, 1.0, 1.0];

        let score: f64 = silhouette_score(&x, &labels).unwrap();
        // Overlapping clusters should have silhouette near 0.
        assert!(
            score.abs() < 0.5,
            "Expected score close to 0, got {}",
            score
        );
    }

    #[test]
    fn test_silhouette_misassigned() {
        // Points assigned to the wrong cluster: each cluster contains points
        // from both natural groups. With more points per natural group, the
        // silhouette becomes strongly negative.
        // Natural groups: tightly clustered near (0,0) and near (100,0).
        // Cluster 0 gets one from group A and two from group B.
        // Cluster 1 gets two from group A and one from group B.
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                0.0, 0.0, // group A, assigned to cluster 0
                100.0, 0.0, // group B, assigned to cluster 0
                100.1, 0.0, // group B, assigned to cluster 0
                0.0, 0.1, // group A, assigned to cluster 1
                0.1, 0.0, // group A, assigned to cluster 1
                100.0, 0.1, // group B, assigned to cluster 1
            ],
        )
        .unwrap();
        let labels = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let score: f64 = silhouette_score(&x, &labels).unwrap();
        // Misassigned clusters should have negative silhouette.
        assert!(score < 0.0, "Expected negative score, got {}", score);
    }

    #[test]
    fn test_silhouette_three_clusters() {
        // Three well-separated clusters.
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, // cluster 0
                10.0, 0.0, 10.1, 0.0, 10.0, 0.1, // cluster 1
                5.0, 10.0, 5.1, 10.0, 5.0, 10.1, // cluster 2
            ],
        )
        .unwrap();
        let labels = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];

        let score: f64 = silhouette_score(&x, &labels).unwrap();
        assert!(score > 0.9, "Expected score > 0.9, got {}", score);
    }

    #[test]
    fn test_silhouette_empty_error() {
        let x: Array2<f64> = Array2::zeros((0, 2));
        let labels: Array1<f64> = array![];
        assert!(silhouette_score(&x, &labels).is_err());
    }

    #[test]
    fn test_silhouette_shape_mismatch_error() {
        let x = Array2::from_shape_vec(
            (3, 2),
            vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
        )
        .unwrap();
        let labels = array![0.0, 1.0]; // wrong length
        assert!(silhouette_score(&x, &labels).is_err());
    }

    #[test]
    fn test_silhouette_single_cluster_error() {
        let x = Array2::from_shape_vec(
            (3, 2),
            vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
        )
        .unwrap();
        let labels = array![0.0, 0.0, 0.0]; // only one cluster
        assert!(silhouette_score(&x, &labels).is_err());
    }

    #[test]
    fn test_silhouette_f32() {
        let x: Array2<f32> = Array2::from_shape_vec(
            (4, 2),
            vec![
                0.0f32, 0.0, 0.1, 0.0, 10.0, 10.0, 10.1, 10.0,
            ],
        )
        .unwrap();
        let labels: Array1<f32> = array![0.0f32, 0.0, 1.0, 1.0];
        let score = silhouette_score(&x, &labels).unwrap();
        assert!(score.is_finite());
        assert!(score > 0.5f32);
    }
}
