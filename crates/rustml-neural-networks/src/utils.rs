use ndarray::{Array1, Array2};
use rustml_core::Float;

/// Numerically stable softmax: subtract row max before exp.
///
/// Takes ownership of `logits` to avoid cloning; all callers produce
/// fresh arrays from `forward_pass`.
pub fn softmax<F: Float>(mut result: Array2<F>) -> Array2<F> {
    for mut row in result.rows_mut() {
        let max = row.fold(F::neg_infinity(), |a, &b| if b > a { b } else { a });
        row.mapv_inplace(|v| (v - max).exp());
        let sum = row.sum();
        row.mapv_inplace(|v| v / sum);
    }
    result
}

/// Cross-entropy loss: -mean(sum(y_onehot * log(probs))).
/// Clips probs for numerical stability.
pub fn cross_entropy_loss<F: Float>(probs: &Array2<F>, y_onehot: &Array2<F>) -> F {
    let eps = F::from_f64(1e-12).unwrap();
    let one = F::one();
    let n = F::from_usize(probs.nrows()).unwrap();

    let log_probs = probs.mapv(|p| {
        let clamped = if p < eps {
            eps
        } else if p > one - eps {
            one - eps
        } else {
            p
        };
        clamped.ln()
    });

    let sum: F = (&log_probs * y_onehot).sum();
    -sum / n
}

/// Mean squared error loss.
pub fn mse_loss<F: Float>(y_pred: &Array2<F>, y_true: &Array2<F>) -> F {
    let diff = y_pred - y_true;
    let sq = &diff * &diff;
    sq.sum() / F::from_usize(y_pred.len()).unwrap()
}

/// One-hot encode labels. Returns (n_samples, n_classes) matrix.
pub fn one_hot_encode<F: Float>(y: &Array1<F>, class_labels: &[F]) -> Array2<F> {
    let n_samples = y.len();
    let n_classes = class_labels.len();
    let mut encoded = Array2::zeros((n_samples, n_classes));

    for (i, &label) in y.iter().enumerate() {
        for (j, &cl) in class_labels.iter().enumerate() {
            if (label - cl).abs() < F::from_f64(1e-10).unwrap() {
                encoded[[i, j]] = F::one();
                break;
            }
        }
    }

    encoded
}

/// Shuffle indices in-place using Fisher-Yates.
pub fn shuffle_indices(indices: &mut [usize], rng: &mut impl rand::Rng) {
    let n = indices.len();
    for i in (1..n).rev() {
        let j = rng.gen_range(0..=i);
        indices.swap(i, j);
    }
}

/// Select rows from a 2D array by index.
pub fn select_rows<F: Float>(x: &Array2<F>, indices: &[usize]) -> Array2<F> {
    let ncols = x.ncols();
    let mut data = Vec::with_capacity(indices.len() * ncols);
    for &i in indices {
        data.extend_from_slice(x.row(i).as_slice().unwrap());
    }
    Array2::from_shape_vec((indices.len(), ncols), data).unwrap()
}

/// Select elements from a 1D array by index.
pub fn select_elements<F: Float>(y: &Array1<F>, indices: &[usize]) -> Array1<F> {
    Array1::from_vec(indices.iter().map(|&i| y[i]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn softmax_uniform() {
        let logits = array![[1.0, 1.0, 1.0]];
        let probs = softmax(logits);
        let third = 1.0 / 3.0;
        for &p in probs.iter() {
            assert_abs_diff_eq!(p, third, epsilon = 1e-10);
        }
    }

    #[test]
    fn softmax_rows_sum_to_one() {
        let logits = array![[1.0, 2.0, 3.0], [10.0, -1.0, 0.0]];
        let probs = softmax(logits);
        for row in probs.rows() {
            assert_abs_diff_eq!(row.sum(), 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn one_hot_basic() {
        let y = array![0.0, 1.0, 2.0, 1.0];
        let labels = vec![0.0, 1.0, 2.0];
        let encoded = one_hot_encode(&y, &labels);
        assert_eq!(encoded.nrows(), 4);
        assert_eq!(encoded.ncols(), 3);
        assert_abs_diff_eq!(encoded[[0, 0]], 1.0);
        assert_abs_diff_eq!(encoded[[1, 1]], 1.0);
        assert_abs_diff_eq!(encoded[[2, 2]], 1.0);
        assert_abs_diff_eq!(encoded[[3, 1]], 1.0);
    }

    #[test]
    fn cross_entropy_perfect() {
        // Perfect predictions: loss should be very small
        let probs = array![[0.99, 0.01], [0.01, 0.99]];
        let y_onehot = array![[1.0, 0.0], [0.0, 1.0]];
        let loss = cross_entropy_loss(&probs, &y_onehot);
        assert!(loss < 0.02);
    }

    #[test]
    fn mse_zero_error() {
        let y = array![[1.0, 2.0], [3.0, 4.0]];
        let loss = mse_loss(&y, &y);
        assert_abs_diff_eq!(loss, 0.0, epsilon = 1e-15);
    }
}
