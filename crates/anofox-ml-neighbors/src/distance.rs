use anofox_ml_core::Float;
use ndarray::{Array2, ArrayView1};

/// Distance metric for KNN.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub enum DistanceMetric {
    #[default]
    Euclidean,
    Manhattan,
    /// Cosine distance = 1 - cosine_similarity.
    ///
    /// Useful for text and high-dimensional data where the angle between
    /// vectors matters more than their magnitude.
    Cosine,
    /// Chebyshev (L∞) distance: the largest coordinate difference.
    Chebyshev,
    /// Minkowski distance `(Σ |a_i - b_i|^p)^(1/p)` with `p ≥ 1`
    /// (`p = 1` is Manhattan, `p = 2` Euclidean).
    Minkowski(f64),
    /// The inputs already are distances: at fit time the square matrix of
    /// training distances, at query time `(n_queries, n_train)` distances to
    /// the training samples (sklearn's `metric="precomputed"`).
    Precomputed,
}

impl DistanceMetric {
    /// The same metric with `Minkowski(1 | 2 | ∞)` written as Manhattan,
    /// Euclidean and Chebyshev.
    pub(crate) fn canonical(self) -> Self {
        match self {
            DistanceMetric::Minkowski(1.0) => DistanceMetric::Manhattan,
            DistanceMetric::Minkowski(2.0) => DistanceMetric::Euclidean,
            DistanceMetric::Minkowski(p) if p == f64::INFINITY => DistanceMetric::Chebyshev,
            m => m,
        }
    }

    /// Whether this is the Euclidean metric (also as `Minkowski(2)`).
    pub fn canonical_is_euclidean(self) -> bool {
        self.canonical() == DistanceMetric::Euclidean
    }

    /// Reject parameters that do not define a metric.
    pub(crate) fn validate(self) -> anofox_ml_core::Result<()> {
        if let DistanceMetric::Minkowski(p) = self {
            if p.is_nan() || p < 1.0 {
                return Err(anofox_ml_core::RustMlError::InvalidParameter(format!(
                    "Minkowski p must be >= 1, got {p}"
                )));
            }
        }
        Ok(())
    }

    /// Whether the metric is a Minkowski norm (KD-tree bounding boxes give
    /// valid lower bounds for it).
    pub(crate) fn is_minkowski(self) -> bool {
        matches!(
            self,
            DistanceMetric::Euclidean
                | DistanceMetric::Manhattan
                | DistanceMetric::Chebyshev
                | DistanceMetric::Minkowski(_)
        )
    }

    /// Reduced distance: a monotone transform of the distance that is
    /// cheaper to compute (squared for Euclidean, `Σ|d|^p` for Minkowski).
    #[inline]
    pub(crate) fn rdist<F: Float>(self, a: &[F], b: &[F]) -> F {
        match self {
            DistanceMetric::Euclidean => euclidean_squared_chunked(a, b),
            DistanceMetric::Manhattan => manhattan_chunked(a, b),
            DistanceMetric::Cosine => cosine_distance_chunked(a, b),
            DistanceMetric::Chebyshev => a
                .iter()
                .zip(b)
                .fold(F::zero(), |m, (&x, &y)| m.max((x - y).abs())),
            DistanceMetric::Minkowski(p) => {
                let p = F::from_f64(p).unwrap();
                a.iter()
                    .zip(b)
                    .fold(F::zero(), |s, (&x, &y)| s + (x - y).abs().powf(p))
            }
            DistanceMetric::Precomputed => F::nan(),
        }
    }

    /// Fold one coordinate's absolute difference into a reduced distance.
    #[inline]
    pub(crate) fn rdist_add<F: Float>(self, acc: F, diff: F) -> F {
        match self {
            DistanceMetric::Euclidean => acc + diff * diff,
            DistanceMetric::Manhattan => acc + diff,
            DistanceMetric::Chebyshev => acc.max(diff),
            DistanceMetric::Minkowski(p) => acc + diff.powf(F::from_f64(p).unwrap()),
            DistanceMetric::Cosine | DistanceMetric::Precomputed => acc,
        }
    }

    /// Distance from a reduced distance.
    #[inline]
    pub(crate) fn rdist_to_dist<F: Float>(self, r: F) -> F {
        match self {
            DistanceMetric::Euclidean => r.sqrt(),
            DistanceMetric::Minkowski(p) => r.powf(F::from_f64(1.0 / p).unwrap()),
            _ => r,
        }
    }

    /// Reduced distance from a distance.
    #[inline]
    pub(crate) fn dist_to_rdist<F: Float>(self, d: F) -> F {
        match self {
            DistanceMetric::Euclidean => d * d,
            DistanceMetric::Minkowski(p) => d.powf(F::from_f64(p).unwrap()),
            _ => d,
        }
    }
}

/// Compute the squared Euclidean distance between two slices using a
/// chunk-based accumulation pattern that is friendly to auto-vectorization.
///
/// Processing 4 elements per iteration lets LLVM emit packed SIMD
/// instructions on x86-64 (SSE2/AVX) and aarch64 (NEON) without any
/// platform-specific intrinsics.
#[inline]
fn euclidean_squared_chunked<F: Float>(a: &[F], b: &[F]) -> F {
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    // Four independent accumulators break the dependency chain and let the
    // CPU execute additions in parallel across SIMD lanes.
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

    // Handle remaining elements.
    for j in 0..remainder {
        let d = a[i + j] - b[i + j];
        acc0 += d * d;
    }

    (acc0 + acc1) + (acc2 + acc3)
}

/// Compute the Manhattan distance between two slices using chunk-based
/// accumulation for auto-vectorization.
#[inline]
fn manhattan_chunked<F: Float>(a: &[F], b: &[F]) -> F {
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut acc0 = F::zero();
    let mut acc1 = F::zero();
    let mut acc2 = F::zero();
    let mut acc3 = F::zero();

    let mut i = 0;
    for _ in 0..chunks {
        acc0 += (a[i] - b[i]).abs();
        acc1 += (a[i + 1] - b[i + 1]).abs();
        acc2 += (a[i + 2] - b[i + 2]).abs();
        acc3 += (a[i + 3] - b[i + 3]).abs();

        i += 4;
    }

    for j in 0..remainder {
        acc0 += (a[i + j] - b[i + j]).abs();
    }

    (acc0 + acc1) + (acc2 + acc3)
}

/// Compute the cosine distance between two slices.
///
/// cosine_distance = 1 - (a . b) / (||a|| * ||b||)
///
/// If either vector has zero norm the function returns `1.0` (maximum
/// distance) rather than producing NaN.
#[inline]
fn cosine_distance_chunked<F: Float>(a: &[F], b: &[F]) -> F {
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut dot0 = F::zero();
    let mut dot1 = F::zero();
    let mut dot2 = F::zero();
    let mut dot3 = F::zero();

    let mut norm_a0 = F::zero();
    let mut norm_a1 = F::zero();
    let mut norm_a2 = F::zero();
    let mut norm_a3 = F::zero();

    let mut norm_b0 = F::zero();
    let mut norm_b1 = F::zero();
    let mut norm_b2 = F::zero();
    let mut norm_b3 = F::zero();

    let mut i = 0;
    for _ in 0..chunks {
        let a0 = a[i];
        let a1 = a[i + 1];
        let a2 = a[i + 2];
        let a3 = a[i + 3];
        let b0 = b[i];
        let b1 = b[i + 1];
        let b2 = b[i + 2];
        let b3 = b[i + 3];

        dot0 += a0 * b0;
        dot1 += a1 * b1;
        dot2 += a2 * b2;
        dot3 += a3 * b3;

        norm_a0 += a0 * a0;
        norm_a1 += a1 * a1;
        norm_a2 += a2 * a2;
        norm_a3 += a3 * a3;

        norm_b0 += b0 * b0;
        norm_b1 += b1 * b1;
        norm_b2 += b2 * b2;
        norm_b3 += b3 * b3;

        i += 4;
    }

    for j in 0..remainder {
        let av = a[i + j];
        let bv = b[i + j];
        dot0 += av * bv;
        norm_a0 += av * av;
        norm_b0 += bv * bv;
    }

    let dot = (dot0 + dot1) + (dot2 + dot3);
    let norm_a_sq = (norm_a0 + norm_a1) + (norm_a2 + norm_a3);
    let norm_b_sq = (norm_b0 + norm_b1) + (norm_b2 + norm_b3);

    let denom = (norm_a_sq * norm_b_sq).sqrt();

    if denom < F::from_f64(1e-30).unwrap() {
        // At least one vector is (effectively) zero — return max distance.
        F::one()
    } else {
        // Clamp to [-1, 1] to guard against floating-point drift before
        // subtracting from 1.
        let similarity = dot / denom;
        let clamped = if similarity > F::one() {
            F::one()
        } else if similarity < -F::one() {
            -F::one()
        } else {
            similarity
        };
        F::one() - clamped
    }
}

/// Compute the distance between two points (`Precomputed` gives NaN: there
/// is nothing to compute).
///
/// The function is `#[inline]` to allow the compiler to specialize and
/// auto-vectorize the hot loop when the metric is known at the call-site.
#[inline]
pub fn compute_distance<F: Float>(
    a: &ArrayView1<F>,
    b: &ArrayView1<F>,
    metric: DistanceMetric,
) -> F {
    // Layout-agnostic: fall back to a contiguous copy for strided views (#42).
    let (a_owned, b_owned);
    let a_slice = match a.as_slice() {
        Some(s) => s,
        None => {
            a_owned = a.to_vec();
            &a_owned
        }
    };
    let b_slice = match b.as_slice() {
        Some(s) => s,
        None => {
            b_owned = b.to_vec();
            &b_owned
        }
    };

    let metric = metric.canonical();
    metric.rdist_to_dist(metric.rdist(a_slice, b_slice))
}

/// Compute distances from a single query point to all rows in a matrix.
///
/// Returns a `Vec` of distances. More efficient than calling
/// [`compute_distance`] in a loop because:
/// - The query slice is extracted once and reused.
/// - For Euclidean distance the sqrt is taken at the end of each row,
///   minimising expensive operations.
/// - The tight row loop is friendly to hardware prefetching.
#[inline]
pub fn compute_distances_batch<F: Float>(
    query: &ArrayView1<F>,
    data: &Array2<F>,
    metric: DistanceMetric,
) -> Vec<F> {
    // Layout-agnostic: copy strided query / non-C-order data once (#42).
    let q_owned;
    let q = match query.as_slice() {
        Some(s) => s,
        None => {
            q_owned = query.to_vec();
            &q_owned
        }
    };
    let data_std = crate::standard_layout(data);
    let data: &Array2<F> = &data_std;
    let n_rows = data.nrows();
    let mut result = Vec::with_capacity(n_rows);

    let metric = metric.canonical();
    for row in data.rows() {
        let r = row.as_slice().expect("standard layout");
        result.push(metric.rdist_to_dist(metric.rdist(q, r)));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array1};

    // ---- existing tests (preserved) ----

    #[test]
    fn test_euclidean() {
        let a = array![0.0, 0.0];
        let b = array![3.0, 4.0];
        assert_abs_diff_eq!(
            compute_distance(&a.view(), &b.view(), DistanceMetric::Euclidean),
            5.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_manhattan() {
        let a = array![0.0, 0.0];
        let b = array![3.0, 4.0];
        assert_abs_diff_eq!(
            compute_distance(&a.view(), &b.view(), DistanceMetric::Manhattan),
            7.0,
            epsilon = 1e-10
        );
    }

    // ---- cosine distance tests ----

    #[test]
    fn test_cosine_identical_vectors() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![1.0, 2.0, 3.0];
        assert_abs_diff_eq!(
            compute_distance(&a.view(), &b.view(), DistanceMetric::Cosine),
            0.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_cosine_orthogonal_vectors() {
        let a = array![1.0, 0.0];
        let b = array![0.0, 1.0];
        assert_abs_diff_eq!(
            compute_distance(&a.view(), &b.view(), DistanceMetric::Cosine),
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_cosine_opposite_vectors() {
        let a = array![1.0, 0.0];
        let b = array![-1.0, 0.0];
        assert_abs_diff_eq!(
            compute_distance(&a.view(), &b.view(), DistanceMetric::Cosine),
            2.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_cosine_zero_vector() {
        let a = array![0.0, 0.0, 0.0];
        let b = array![1.0, 2.0, 3.0];
        // Zero vector should return distance 1.0 (maximum).
        assert_abs_diff_eq!(
            compute_distance(&a.view(), &b.view(), DistanceMetric::Cosine),
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_cosine_both_zero_vectors() {
        let a = array![0.0, 0.0];
        let b = array![0.0, 0.0];
        assert_abs_diff_eq!(
            compute_distance(&a.view(), &b.view(), DistanceMetric::Cosine),
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_cosine_scaled_vectors() {
        // Cosine distance is scale-invariant: same direction = distance 0.
        let a = array![1.0, 2.0, 3.0];
        let b = array![2.0, 4.0, 6.0];
        assert_abs_diff_eq!(
            compute_distance(&a.view(), &b.view(), DistanceMetric::Cosine),
            0.0,
            epsilon = 1e-10
        );
    }

    // ---- batch distance tests ----

    #[test]
    fn test_batch_matches_individual_euclidean() {
        let query = array![1.0, 2.0, 3.0];
        let data = array![
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [3.0, 4.0, 5.0],
            [10.0, 20.0, 30.0]
        ];
        let batch = compute_distances_batch(&query.view(), &data, DistanceMetric::Euclidean);
        for (i, row) in data.rows().into_iter().enumerate() {
            let individual = compute_distance(&query.view(), &row, DistanceMetric::Euclidean);
            assert_abs_diff_eq!(batch[i], individual, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_batch_matches_individual_manhattan() {
        let query = array![1.0, 2.0, 3.0];
        let data = array![
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [3.0, 4.0, 5.0],
            [10.0, 20.0, 30.0]
        ];
        let batch = compute_distances_batch(&query.view(), &data, DistanceMetric::Manhattan);
        for (i, row) in data.rows().into_iter().enumerate() {
            let individual = compute_distance(&query.view(), &row, DistanceMetric::Manhattan);
            assert_abs_diff_eq!(batch[i], individual, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_batch_matches_individual_cosine() {
        let query = array![1.0, 2.0, 3.0];
        let data = array![
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [3.0, 4.0, 5.0],
            [-1.0, -2.0, -3.0]
        ];
        let batch = compute_distances_batch(&query.view(), &data, DistanceMetric::Cosine);
        for (i, row) in data.rows().into_iter().enumerate() {
            let individual = compute_distance(&query.view(), &row, DistanceMetric::Cosine);
            assert_abs_diff_eq!(batch[i], individual, epsilon = 1e-10);
        }
    }

    // ---- f32 support tests ----

    #[test]
    fn test_euclidean_f32() {
        let a: Array1<f32> = array![0.0f32, 0.0];
        let b: Array1<f32> = array![3.0f32, 4.0];
        assert_abs_diff_eq!(
            compute_distance(&a.view(), &b.view(), DistanceMetric::Euclidean),
            5.0f32,
            epsilon = 1e-5
        );
    }

    #[test]
    fn test_manhattan_f32() {
        let a: Array1<f32> = array![0.0f32, 0.0];
        let b: Array1<f32> = array![3.0f32, 4.0];
        assert_abs_diff_eq!(
            compute_distance(&a.view(), &b.view(), DistanceMetric::Manhattan),
            7.0f32,
            epsilon = 1e-5
        );
    }

    #[test]
    fn test_cosine_f32() {
        let a: Array1<f32> = array![1.0f32, 0.0];
        let b: Array1<f32> = array![0.0f32, 1.0];
        assert_abs_diff_eq!(
            compute_distance(&a.view(), &b.view(), DistanceMetric::Cosine),
            1.0f32,
            epsilon = 1e-5
        );
    }

    #[test]
    fn test_batch_f32() {
        let query: Array1<f32> = array![1.0f32, 2.0];
        let data: ndarray::Array2<f32> = array![[0.0f32, 0.0], [3.0, 4.0]];
        let batch = compute_distances_batch(&query.view(), &data, DistanceMetric::Euclidean);
        assert_eq!(batch.len(), 2);
        assert_abs_diff_eq!(batch[0], (1.0f32 + 4.0f32).sqrt(), epsilon = 1e-5);
    }

    // ---- chunking edge-case tests ----

    #[test]
    fn test_euclidean_odd_dimensions() {
        // 5 dimensions: exercises both the 4-element chunk and the remainder path.
        let a = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = array![5.0, 4.0, 3.0, 2.0, 1.0];
        let expected = (16.0 + 4.0 + 0.0 + 4.0 + 16.0_f64).sqrt();
        assert_abs_diff_eq!(
            compute_distance(&a.view(), &b.view(), DistanceMetric::Euclidean),
            expected,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_manhattan_single_dimension() {
        let a = array![3.0];
        let b = array![7.0];
        assert_abs_diff_eq!(
            compute_distance(&a.view(), &b.view(), DistanceMetric::Manhattan),
            4.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_euclidean_empty() {
        // Two zero-length vectors should have distance 0.
        let a: Array1<f64> = Array1::zeros(0);
        let b: Array1<f64> = Array1::zeros(0);
        assert_abs_diff_eq!(
            compute_distance(&a.view(), &b.view(), DistanceMetric::Euclidean),
            0.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_cosine_high_dimensional() {
        // 17 dimensions — exercises multiple full chunks + remainder.
        let a = Array1::from_vec(vec![1.0_f64; 17]);
        let b = Array1::from_vec(vec![1.0_f64; 17]);
        assert_abs_diff_eq!(
            compute_distance(&a.view(), &b.view(), DistanceMetric::Cosine),
            0.0,
            epsilon = 1e-10
        );
    }
}
