use ndarray::{Array2, ArrayView1};
use rustml_core::Float;

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

/// Compute the distance between two points.
///
/// The function is `#[inline]` to allow the compiler to specialize and
/// auto-vectorize the hot loop when the metric is known at the call-site.
#[inline]
pub fn compute_distance<F: Float>(
    a: &ArrayView1<F>,
    b: &ArrayView1<F>,
    metric: DistanceMetric,
) -> F {
    let a_slice = a.as_slice().expect("a must be contiguous");
    let b_slice = b.as_slice().expect("b must be contiguous");

    match metric {
        DistanceMetric::Euclidean => euclidean_squared_chunked(a_slice, b_slice).sqrt(),
        DistanceMetric::Manhattan => manhattan_chunked(a_slice, b_slice),
        DistanceMetric::Cosine => cosine_distance_chunked(a_slice, b_slice),
    }
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
    let q = query.as_slice().expect("query must be contiguous");
    let n_rows = data.nrows();
    let mut result = Vec::with_capacity(n_rows);

    match metric {
        DistanceMetric::Euclidean => {
            for row in data.rows() {
                let r = row.as_slice().expect("row must be contiguous");
                result.push(euclidean_squared_chunked(q, r).sqrt());
            }
        }
        DistanceMetric::Manhattan => {
            for row in data.rows() {
                let r = row.as_slice().expect("row must be contiguous");
                result.push(manhattan_chunked(q, r));
            }
        }
        DistanceMetric::Cosine => {
            for row in data.rows() {
                let r = row.as_slice().expect("row must be contiguous");
                result.push(cosine_distance_chunked(q, r));
            }
        }
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
