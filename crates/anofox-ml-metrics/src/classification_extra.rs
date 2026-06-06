use anofox_ml_core::{Float, Result, RustMlError};
use ndarray::Array1;

/// Binary cross-entropy (log loss).
///
/// Computes `- mean(y_true * log(y_prob) + (1 - y_true) * log(1 - y_prob))`.
/// Probabilities are clipped to `[eps, 1 - eps]` to avoid `log(0)`.
/// Lower values indicate better predictions.
///
/// `y_true` should contain binary labels (0 or 1) and `y_prob` should contain
/// predicted probabilities in `[0, 1]`.
pub fn log_loss<F: Float>(y_true: &Array1<F>, y_prob: &Array1<F>) -> Result<F> {
    check_lengths(y_true, y_prob)?;

    let eps = F::from_f64(1e-15).unwrap();
    let one = F::one();
    let n = F::from_usize(y_true.len()).unwrap();

    let sum = y_true
        .iter()
        .zip(y_prob.iter())
        .map(|(&t, &p)| {
            // Clip probability to [eps, 1 - eps]
            let p_clipped = if p < eps {
                eps
            } else if p > one - eps {
                one - eps
            } else {
                p
            };
            t * p_clipped.ln() + (one - t) * (one - p_clipped).ln()
        })
        .fold(F::zero(), |acc, v| acc + v);

    Ok(-sum / n)
}

/// Balanced accuracy score.
///
/// Computes the macro-averaged recall across all classes. This is equivalent to
/// standard accuracy for balanced datasets, but accounts for class imbalance by
/// giving equal weight to each class regardless of its support.
///
/// For each class `c`, the per-class recall is `TP_c / (TP_c + FN_c)`. The
/// balanced accuracy is the unweighted mean of these per-class recall values.
pub fn balanced_accuracy_score<F: Float>(y_true: &Array1<F>, y_pred: &Array1<F>) -> Result<F> {
    check_lengths(y_true, y_pred)?;

    let classes = unique_sorted(y_true);
    let n_classes = classes.len();
    let eps = F::from_f64(1e-9).unwrap();

    let mut recall_sum = F::zero();

    for &c in &classes {
        let mut tp = F::zero();
        let mut fn_ = F::zero();

        for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
            if (t - c).abs() < eps {
                // This sample belongs to class c
                if (p - c).abs() < eps {
                    tp += F::one();
                } else {
                    fn_ += F::one();
                }
            }
        }

        let support = tp + fn_;
        if support > F::zero() {
            recall_sum += tp / support;
        }
    }

    Ok(recall_sum / F::from_usize(n_classes).unwrap())
}

/// Cohen's kappa coefficient.
///
/// Measures inter-rater agreement for categorical items, corrected for agreement
/// by chance. Computed as `(p_o - p_e) / (1 - p_e)` where `p_o` is the observed
/// agreement (accuracy) and `p_e` is the expected agreement by chance.
///
/// Returns 1 for perfect agreement, 0 for agreement equal to chance, and
/// negative values for agreement worse than chance.
pub fn cohen_kappa_score<F: Float>(y_true: &Array1<F>, y_pred: &Array1<F>) -> Result<F> {
    check_lengths(y_true, y_pred)?;

    let n = y_true.len();
    let n_f = F::from_usize(n).unwrap();
    let eps = F::from_f64(1e-9).unwrap();

    let classes = unique_sorted_pair(y_true, y_pred);
    let n_classes = classes.len();

    // Build confusion matrix
    let mut cm = vec![vec![F::zero(); n_classes]; n_classes];
    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        let i = classes.iter().position(|&c| (c - t).abs() < eps).unwrap();
        let j = classes.iter().position(|&c| (c - p).abs() < eps).unwrap();
        cm[i][j] += F::one();
    }

    // Observed agreement: sum of diagonal / n
    let mut p_o = F::zero();
    for i in 0..n_classes {
        p_o += cm[i][i];
    }
    p_o = p_o / n_f;

    // Expected agreement: sum over classes of (row_sum_i * col_sum_i) / n^2
    let mut p_e = F::zero();
    for i in 0..n_classes {
        let row_sum: F = (0..n_classes)
            .map(|j| cm[i][j])
            .fold(F::zero(), |a, b| a + b);
        let col_sum: F = (0..n_classes)
            .map(|j| cm[j][i])
            .fold(F::zero(), |a, b| a + b);
        p_e += row_sum * col_sum;
    }
    p_e = p_e / (n_f * n_f);

    let denom = F::one() - p_e;
    if denom.abs() < F::from_f64(1e-15).unwrap() {
        // Perfect expected agreement (all samples in one class and both agree)
        return Ok(F::one());
    }

    Ok((p_o - p_e) / denom)
}

fn unique_sorted<F: Float>(a: &Array1<F>) -> Vec<F> {
    let mut vals: Vec<F> = a.iter().copied().collect();
    vals.sort_by(|x, y| x.partial_cmp(y).unwrap());
    vals.dedup_by(|a, b| (*a - *b).abs() < F::from_f64(1e-9).unwrap());
    vals
}

fn unique_sorted_pair<F: Float>(a: &Array1<F>, b: &Array1<F>) -> Vec<F> {
    let mut vals: Vec<F> = a.iter().chain(b.iter()).copied().collect();
    vals.sort_by(|x, y| x.partial_cmp(y).unwrap());
    vals.dedup_by(|a, b| (*a - *b).abs() < F::from_f64(1e-9).unwrap());
    vals
}

fn check_lengths<F: Float>(a: &Array1<F>, b: &Array1<F>) -> Result<()> {
    if a.len() != b.len() {
        return Err(RustMlError::ShapeMismatch(format!(
            "y_true length {} != y_pred length {}",
            a.len(),
            b.len()
        )));
    }
    if a.is_empty() {
        return Err(RustMlError::EmptyInput("input arrays are empty".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    // ---------------------------------------------------------------
    // Log loss tests
    // ---------------------------------------------------------------

    #[test]
    fn test_log_loss_perfect() {
        // Perfect predictions: y_true = [0, 1], y_prob = [0, 1]
        // After clipping, log loss should be very close to 0.
        let y_true = array![0.0, 1.0, 1.0, 0.0];
        let y_prob = array![0.0, 1.0, 1.0, 0.0];
        let loss: f64 = log_loss(&y_true, &y_prob).unwrap();
        // Clipped to eps, so not exactly 0 but very small.
        assert!(loss < 1e-10);
    }

    #[test]
    fn test_log_loss_known_value() {
        // y_true = [1, 0], y_prob = [0.9, 0.1]
        // loss = -( 1*ln(0.9) + 0*ln(0.1) + 0*ln(0.9) + 1*ln(0.9) ) / 2
        //      = -( ln(0.9) + ln(0.9) ) / 2 = -ln(0.9)
        let y_true = array![1.0, 0.0];
        let y_prob = array![0.9, 0.1];
        let expected = -(0.9_f64.ln() + 0.9_f64.ln()) / 2.0;
        let loss: f64 = log_loss(&y_true, &y_prob).unwrap();
        assert_abs_diff_eq!(loss, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_log_loss_worst_case() {
        // Completely wrong predictions: y_true = [1, 0], y_prob = [0.1, 0.9]
        // loss = -(ln(0.1) + ln(0.1)) / 2 = -ln(0.1)
        let y_true = array![1.0, 0.0];
        let y_prob = array![0.1, 0.9];
        let expected = -0.1_f64.ln();
        let loss: f64 = log_loss(&y_true, &y_prob).unwrap();
        assert_abs_diff_eq!(loss, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_log_loss_clips_probabilities() {
        // Probabilities of exactly 0 and 1 should be clipped, not produce NaN/Inf.
        let y_true = array![1.0, 0.0];
        let y_prob = array![0.0, 1.0];
        let loss: f64 = log_loss(&y_true, &y_prob).unwrap();
        assert!(loss.is_finite());
        assert!(loss > 0.0);
    }

    #[test]
    fn test_log_loss_empty_error() {
        let y_true: Array1<f64> = array![];
        let y_prob: Array1<f64> = array![];
        assert!(log_loss(&y_true, &y_prob).is_err());
    }

    #[test]
    fn test_log_loss_length_mismatch_error() {
        let y_true = array![0.0, 1.0];
        let y_prob = array![0.5, 0.5, 0.5];
        assert!(log_loss(&y_true, &y_prob).is_err());
    }

    #[test]
    fn test_log_loss_f32() {
        let y_true: Array1<f32> = array![1.0f32, 0.0];
        let y_prob: Array1<f32> = array![0.9f32, 0.1];
        let loss = log_loss(&y_true, &y_prob).unwrap();
        assert!(loss.is_finite());
        assert!(loss > 0.0f32);
    }

    // ---------------------------------------------------------------
    // Balanced accuracy tests
    // ---------------------------------------------------------------

    #[test]
    fn test_balanced_accuracy_perfect() {
        let y_true = array![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
        let acc: f64 = balanced_accuracy_score(&y_true, &y_true).unwrap();
        assert_abs_diff_eq!(acc, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_balanced_accuracy_imbalanced() {
        // Imbalanced: 4 class-0, 1 class-1. Predict all as class 0.
        // Recall class 0 = 4/4 = 1.0, Recall class 1 = 0/1 = 0.0
        // Balanced accuracy = (1.0 + 0.0) / 2 = 0.5
        let y_true = array![0.0, 0.0, 0.0, 0.0, 1.0];
        let y_pred = array![0.0, 0.0, 0.0, 0.0, 0.0];
        let acc: f64 = balanced_accuracy_score(&y_true, &y_pred).unwrap();
        assert_abs_diff_eq!(acc, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_balanced_accuracy_multiclass() {
        // y_true = [0, 0, 1, 1, 2, 2]
        // y_pred = [0, 1, 1, 1, 0, 2]
        // class 0: TP=1, FN=1 -> recall = 0.5
        // class 1: TP=2, FN=0 -> recall = 1.0
        // class 2: TP=1, FN=1 -> recall = 0.5
        // balanced = (0.5 + 1.0 + 0.5) / 3 = 2/3
        let y_true = array![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
        let y_pred = array![0.0, 1.0, 1.0, 1.0, 0.0, 2.0];
        let acc: f64 = balanced_accuracy_score(&y_true, &y_pred).unwrap();
        assert_abs_diff_eq!(acc, 2.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_balanced_accuracy_binary_balanced() {
        // When dataset is balanced, balanced accuracy = standard accuracy.
        let y_true = array![0.0, 0.0, 1.0, 1.0];
        let y_pred = array![0.0, 1.0, 1.0, 1.0];
        // class 0: recall = 1/2 = 0.5, class 1: recall = 2/2 = 1.0
        // balanced = (0.5 + 1.0) / 2 = 0.75
        // standard accuracy = 3/4 = 0.75
        let acc: f64 = balanced_accuracy_score(&y_true, &y_pred).unwrap();
        assert_abs_diff_eq!(acc, 0.75, epsilon = 1e-10);
    }

    #[test]
    fn test_balanced_accuracy_empty_error() {
        let y_true: Array1<f64> = array![];
        let y_pred: Array1<f64> = array![];
        assert!(balanced_accuracy_score(&y_true, &y_pred).is_err());
    }

    #[test]
    fn test_balanced_accuracy_length_mismatch_error() {
        let y_true = array![0.0, 1.0];
        let y_pred = array![0.0, 1.0, 0.0];
        assert!(balanced_accuracy_score(&y_true, &y_pred).is_err());
    }

    // ---------------------------------------------------------------
    // Cohen's kappa tests
    // ---------------------------------------------------------------

    #[test]
    fn test_cohen_kappa_perfect() {
        let y_true = array![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
        let kappa: f64 = cohen_kappa_score(&y_true, &y_true).unwrap();
        assert_abs_diff_eq!(kappa, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cohen_kappa_chance() {
        // Two raters with no agreement beyond chance.
        // y_true = [0, 0, 1, 1], y_pred = [0, 1, 0, 1]
        // CM: [[1,1],[1,1]]
        // p_o = (1+1)/4 = 0.5
        // row sums: [2, 2], col sums: [2, 2]
        // p_e = (2*2 + 2*2) / 16 = 8/16 = 0.5
        // kappa = (0.5 - 0.5) / (1 - 0.5) = 0
        let y_true = array![0.0, 0.0, 1.0, 1.0];
        let y_pred = array![0.0, 1.0, 0.0, 1.0];
        let kappa: f64 = cohen_kappa_score(&y_true, &y_pred).unwrap();
        assert_abs_diff_eq!(kappa, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cohen_kappa_known_value() {
        // y_true = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        // y_pred = [0, 0, 1, 0, 1, 1, 1, 2, 2]
        // CM:
        //     pred 0  pred 1  pred 2
        // 0 [  2,      1,      0  ]
        // 1 [  1,      2,      0  ]
        // 2 [  0,      1,      2  ]
        //
        // p_o = (2+2+2)/9 = 6/9 = 2/3
        // row sums: [3, 3, 3], col sums: [3, 4, 2]
        // p_e = (3*3 + 3*4 + 3*2) / 81 = (9+12+6)/81 = 27/81 = 1/3
        // kappa = (2/3 - 1/3) / (1 - 1/3) = (1/3) / (2/3) = 1/2
        let y_true = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];
        let y_pred = array![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0];
        let kappa: f64 = cohen_kappa_score(&y_true, &y_pred).unwrap();
        assert_abs_diff_eq!(kappa, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_cohen_kappa_negative() {
        // Worse than chance agreement.
        // y_true = [0, 0, 1, 1], y_pred = [1, 1, 0, 0]
        // CM: [[0,2],[2,0]]
        // p_o = 0/4 = 0
        // row sums: [2,2], col sums: [2,2]
        // p_e = (2*2 + 2*2) / 16 = 8/16 = 0.5
        // kappa = (0 - 0.5) / (1 - 0.5) = -1.0
        let y_true = array![0.0, 0.0, 1.0, 1.0];
        let y_pred = array![1.0, 1.0, 0.0, 0.0];
        let kappa: f64 = cohen_kappa_score(&y_true, &y_pred).unwrap();
        assert_abs_diff_eq!(kappa, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cohen_kappa_empty_error() {
        let y_true: Array1<f64> = array![];
        let y_pred: Array1<f64> = array![];
        assert!(cohen_kappa_score(&y_true, &y_pred).is_err());
    }

    #[test]
    fn test_cohen_kappa_length_mismatch_error() {
        let y_true = array![0.0, 1.0];
        let y_pred = array![0.0, 1.0, 0.0];
        assert!(cohen_kappa_score(&y_true, &y_pred).is_err());
    }

    #[test]
    fn test_cohen_kappa_f32() {
        let y_true: Array1<f32> = array![0.0f32, 0.0, 1.0, 1.0];
        let y_pred: Array1<f32> = array![0.0f32, 0.0, 1.0, 1.0];
        let kappa = cohen_kappa_score(&y_true, &y_pred).unwrap();
        assert_abs_diff_eq!(kappa, 1.0f32, epsilon = 1e-6);
    }
}
