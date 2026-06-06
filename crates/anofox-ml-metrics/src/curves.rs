use anofox_ml_core::{Float, Result, RustMlError};
use ndarray::Array1;

/// Compute the Receiver Operating Characteristic (ROC) curve.
///
/// Returns `(fpr, tpr, thresholds)` where:
/// - `fpr` is the array of false positive rates,
/// - `tpr` is the array of true positive rates,
/// - `thresholds` is the array of decision thresholds (sorted descending).
///
/// The curve starts at `(0, 0)` with the highest threshold and ends at
/// `(1, 1)` with the lowest. `y_true` must contain binary labels (only 0
/// and 1 values).
///
/// # Errors
///
/// Returns an error if:
/// - The inputs have different lengths.
/// - The inputs are empty.
/// - `y_true` contains non-binary labels.
/// - `y_true` contains only a single class.
pub fn roc_curve<F: Float>(
    y_true: &Array1<F>,
    y_score: &Array1<F>,
) -> Result<(Array1<F>, Array1<F>, Array1<F>)> {
    check_lengths(y_true, y_score)?;
    check_binary(y_true)?;

    let eps = F::from_f64(1e-9).unwrap();

    let pos_count = y_true
        .iter()
        .filter(|&&v| (v - F::one()).abs() < eps)
        .count();
    let neg_count = y_true.len() - pos_count;

    if pos_count == 0 || neg_count == 0 {
        return Err(RustMlError::InvalidParameter(
            "ROC curve requires both positive and negative samples".into(),
        ));
    }

    let total_pos = F::from_usize(pos_count).unwrap();
    let total_neg = F::from_usize(neg_count).unwrap();

    // Sort indices by score descending
    let mut indices: Vec<usize> = (0..y_true.len()).collect();
    indices.sort_by(|&a, &b| y_score[b].partial_cmp(&y_score[a]).unwrap());

    // Walk through sorted scores and compute TPR/FPR at each distinct threshold
    let mut fpr_vec = vec![F::zero()];
    let mut tpr_vec = vec![F::zero()];
    let mut thresh_vec = vec![y_score[indices[0]] + F::one()]; // sentinel above max score

    let mut tp = F::zero();
    let mut fp = F::zero();

    let mut i = 0;
    while i < indices.len() {
        // Find end of group of tied scores
        let current_score = y_score[indices[i]];
        let mut j = i;
        while j < indices.len()
            && (y_score[indices[j]] - current_score).abs() < F::from_f64(1e-15).unwrap()
        {
            if (y_true[indices[j]] - F::one()).abs() < eps {
                tp += F::one();
            } else {
                fp += F::one();
            }
            j += 1;
        }

        fpr_vec.push(fp / total_neg);
        tpr_vec.push(tp / total_pos);
        thresh_vec.push(current_score);

        i = j;
    }

    Ok((
        Array1::from_vec(fpr_vec),
        Array1::from_vec(tpr_vec),
        Array1::from_vec(thresh_vec),
    ))
}

/// Compute the precision-recall curve.
///
/// Returns `(precision, recall, thresholds)` where:
/// - `precision` is the array of precision values,
/// - `recall` is the array of recall values,
/// - `thresholds` is the array of decision thresholds (sorted descending).
///
/// The curve starts with the precision and recall at the highest threshold
/// and includes a final point with `precision = 1, recall = 0` as a
/// sentinel. `y_true` must contain binary labels (only 0 and 1 values).
///
/// # Errors
///
/// Returns an error if:
/// - The inputs have different lengths.
/// - The inputs are empty.
/// - `y_true` contains non-binary labels.
/// - `y_true` contains no positive samples.
pub fn precision_recall_curve<F: Float>(
    y_true: &Array1<F>,
    y_score: &Array1<F>,
) -> Result<(Array1<F>, Array1<F>, Array1<F>)> {
    check_lengths(y_true, y_score)?;
    check_binary(y_true)?;

    let eps = F::from_f64(1e-9).unwrap();

    let pos_count = y_true
        .iter()
        .filter(|&&v| (v - F::one()).abs() < eps)
        .count();

    if pos_count == 0 {
        return Err(RustMlError::InvalidParameter(
            "precision-recall curve requires at least one positive sample".into(),
        ));
    }

    let total_pos = F::from_usize(pos_count).unwrap();

    // Sort indices by score descending
    let mut indices: Vec<usize> = (0..y_true.len()).collect();
    indices.sort_by(|&a, &b| y_score[b].partial_cmp(&y_score[a]).unwrap());

    let mut precision_vec = Vec::new();
    let mut recall_vec = Vec::new();
    let mut thresh_vec = Vec::new();

    let mut tp = F::zero();
    let mut fp = F::zero();

    let mut i = 0;
    while i < indices.len() {
        // Process group of tied scores
        let current_score = y_score[indices[i]];
        let mut j = i;
        while j < indices.len()
            && (y_score[indices[j]] - current_score).abs() < F::from_f64(1e-15).unwrap()
        {
            if (y_true[indices[j]] - F::one()).abs() < eps {
                tp += F::one();
            } else {
                fp += F::one();
            }
            j += 1;
        }

        let prec = tp / (tp + fp);
        let rec = tp / total_pos;

        precision_vec.push(prec);
        recall_vec.push(rec);
        thresh_vec.push(current_score);

        i = j;
    }

    // Append sentinel: precision = 1, recall = 0 (no corresponding threshold)
    precision_vec.push(F::one());
    recall_vec.push(F::zero());

    // Reverse so recall goes from 0 -> 1 (ascending recall, descending precision)
    precision_vec.reverse();
    recall_vec.reverse();

    Ok((
        Array1::from_vec(precision_vec),
        Array1::from_vec(recall_vec),
        Array1::from_vec(thresh_vec),
    ))
}

/// Brier score loss (mean squared error between true labels and predicted
/// probabilities).
///
/// Computed as `mean((y_true - y_prob)^2)`. Lower values indicate better
/// calibrated probability estimates. A perfect model has a Brier score of 0;
/// a model that always predicts 0.5 for balanced binary outcomes has a score
/// of 0.25.
///
/// `y_true` should contain binary labels (0 or 1) and `y_prob` should contain
/// predicted probabilities in `[0, 1]`.
///
/// # Errors
///
/// Returns an error if:
/// - The inputs have different lengths.
/// - The inputs are empty.
pub fn brier_score_loss<F: Float>(y_true: &Array1<F>, y_prob: &Array1<F>) -> Result<F> {
    check_lengths(y_true, y_prob)?;

    let n = F::from_usize(y_true.len()).unwrap();

    let sum = y_true
        .iter()
        .zip(y_prob.iter())
        .map(|(&t, &p)| {
            let diff = t - p;
            diff * diff
        })
        .fold(F::zero(), |acc, v| acc + v);

    Ok(sum / n)
}

fn check_lengths<F: Float>(a: &Array1<F>, b: &Array1<F>) -> Result<()> {
    if a.len() != b.len() {
        return Err(RustMlError::ShapeMismatch(format!(
            "y_true length {} != y_score length {}",
            a.len(),
            b.len()
        )));
    }
    if a.is_empty() {
        return Err(RustMlError::EmptyInput("input arrays are empty".into()));
    }
    Ok(())
}

fn check_binary<F: Float>(arr: &Array1<F>) -> Result<()> {
    let zero = F::zero();
    let one = F::one();
    let eps = F::from_f64(1e-9).unwrap();
    for &v in arr.iter() {
        if (v - zero).abs() > eps && (v - one).abs() > eps {
            return Err(RustMlError::InvalidParameter(format!(
                "expected binary labels (0 or 1), found {}",
                v
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    // ---------------------------------------------------------------
    // ROC curve tests
    // ---------------------------------------------------------------

    #[test]
    fn test_roc_curve_perfect() {
        // Perfect separation: positives have higher scores.
        let y_true = array![0.0, 0.0, 1.0, 1.0];
        let y_score = array![0.1, 0.2, 0.8, 0.9];
        let (fpr, tpr, _thresholds) = roc_curve(&y_true, &y_score).unwrap();

        // Should start at (0, 0), reach (0, 1) before (1, 1)
        assert_abs_diff_eq!(fpr[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(tpr[0], 0.0, epsilon = 1e-10);

        // At some point TPR should be 1 while FPR is still 0
        let perfect_point = fpr
            .iter()
            .zip(tpr.iter())
            .any(|(&f, &t)| (f - 0.0_f64).abs() < 1e-9 && (t - 1.0_f64).abs() < 1e-9);
        assert!(perfect_point, "Expected (0, 1) point in perfect ROC");
    }

    #[test]
    fn test_roc_curve_known_values() {
        // Sorted desc: (0.9,1), (0.8,0), (0.3,1), (0.2,0)
        let y_true = array![1.0, 0.0, 1.0, 0.0];
        let y_score = array![0.9, 0.8, 0.3, 0.2];
        let (fpr, tpr, thresholds) = roc_curve(&y_true, &y_score).unwrap();

        // Starting point
        assert_abs_diff_eq!(fpr[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(tpr[0], 0.0, epsilon = 1e-10);

        // After threshold 0.9: tp=1, fp=0 -> (0, 0.5)
        assert_abs_diff_eq!(fpr[1], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(tpr[1], 0.5, epsilon = 1e-10);

        // After threshold 0.8: tp=1, fp=1 -> (0.5, 0.5)
        assert_abs_diff_eq!(fpr[2], 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(tpr[2], 0.5, epsilon = 1e-10);

        // After threshold 0.3: tp=2, fp=1 -> (0.5, 1.0)
        assert_abs_diff_eq!(fpr[3], 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(tpr[3], 1.0, epsilon = 1e-10);

        // After threshold 0.2: tp=2, fp=2 -> (1.0, 1.0)
        assert_abs_diff_eq!(fpr[4], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(tpr[4], 1.0, epsilon = 1e-10);

        // Thresholds should be in descending order
        for i in 1..thresholds.len() {
            assert!(thresholds[i - 1] >= thresholds[i]);
        }
    }

    #[test]
    fn test_roc_curve_single_class_error() {
        let y_true = array![1.0, 1.0, 1.0];
        let y_score = array![0.5, 0.6, 0.7];
        assert!(roc_curve(&y_true, &y_score).is_err());
    }

    #[test]
    fn test_roc_curve_length_mismatch_error() {
        let y_true = array![0.0, 1.0];
        let y_score = array![0.5, 0.6, 0.7];
        assert!(roc_curve(&y_true, &y_score).is_err());
    }

    #[test]
    fn test_roc_curve_empty_error() {
        let y_true: Array1<f64> = array![];
        let y_score: Array1<f64> = array![];
        assert!(roc_curve(&y_true, &y_score).is_err());
    }

    #[test]
    fn test_roc_curve_non_binary_error() {
        let y_true = array![0.0, 1.0, 2.0];
        let y_score = array![0.5, 0.6, 0.7];
        assert!(roc_curve(&y_true, &y_score).is_err());
    }

    #[test]
    fn test_roc_curve_f32() {
        let y_true: Array1<f32> = array![0.0f32, 0.0, 1.0, 1.0];
        let y_score: Array1<f32> = array![0.1f32, 0.2, 0.8, 0.9];
        let (fpr, tpr, _) = roc_curve(&y_true, &y_score).unwrap();
        assert_abs_diff_eq!(fpr[0], 0.0f32, epsilon = 1e-6);
        assert_abs_diff_eq!(tpr[0], 0.0f32, epsilon = 1e-6);
    }

    // ---------------------------------------------------------------
    // Precision-recall curve tests
    // ---------------------------------------------------------------

    #[test]
    fn test_pr_curve_perfect() {
        // Perfect separation
        let y_true = array![0.0, 0.0, 1.0, 1.0];
        let y_score = array![0.1, 0.2, 0.8, 0.9];
        let (precision, recall, _thresholds) = precision_recall_curve(&y_true, &y_score).unwrap();

        // Last non-sentinel point should have recall = 1 and precision = 1
        // (since all positives are ranked first)
        let has_perfect = precision
            .iter()
            .zip(recall.iter())
            .any(|(&p, &r)| (p - 1.0_f64).abs() < 1e-9 && (r - 1.0_f64).abs() < 1e-9);
        assert!(has_perfect, "Expected (1.0, 1.0) point in perfect PR curve");
    }

    #[test]
    fn test_pr_curve_known_values() {
        // Sorted desc: (0.9,1), (0.8,0), (0.3,1), (0.2,0)
        let y_true = array![1.0, 0.0, 1.0, 0.0];
        let y_score = array![0.9, 0.8, 0.3, 0.2];
        let (precision, recall, thresholds) = precision_recall_curve(&y_true, &y_score).unwrap();

        // Sentinel at start (after reversal): precision=1, recall=0
        assert_abs_diff_eq!(precision[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(recall[0], 0.0, epsilon = 1e-10);

        // Thresholds should be in descending order
        for i in 1..thresholds.len() {
            assert!(thresholds[i - 1] >= thresholds[i]);
        }

        // precision and recall arrays should be longer than thresholds by 1
        // (due to the sentinel)
        assert_eq!(precision.len(), thresholds.len() + 1);
        assert_eq!(recall.len(), thresholds.len() + 1);
    }

    #[test]
    fn test_pr_curve_no_positive_error() {
        let y_true = array![0.0, 0.0, 0.0];
        let y_score = array![0.5, 0.6, 0.7];
        assert!(precision_recall_curve(&y_true, &y_score).is_err());
    }

    #[test]
    fn test_pr_curve_length_mismatch_error() {
        let y_true = array![0.0, 1.0];
        let y_score = array![0.5, 0.6, 0.7];
        assert!(precision_recall_curve(&y_true, &y_score).is_err());
    }

    #[test]
    fn test_pr_curve_empty_error() {
        let y_true: Array1<f64> = array![];
        let y_score: Array1<f64> = array![];
        assert!(precision_recall_curve(&y_true, &y_score).is_err());
    }

    #[test]
    fn test_pr_curve_f32() {
        let y_true: Array1<f32> = array![0.0f32, 0.0, 1.0, 1.0];
        let y_score: Array1<f32> = array![0.1f32, 0.2, 0.8, 0.9];
        let (precision, recall, _) = precision_recall_curve(&y_true, &y_score).unwrap();
        // Sentinel: precision=1, recall=0
        assert_abs_diff_eq!(precision[0], 1.0f32, epsilon = 1e-6);
        assert_abs_diff_eq!(recall[0], 0.0f32, epsilon = 1e-6);
    }

    // ---------------------------------------------------------------
    // Brier score loss tests
    // ---------------------------------------------------------------

    #[test]
    fn test_brier_perfect() {
        let y_true = array![0.0, 1.0, 1.0, 0.0];
        let y_prob = array![0.0, 1.0, 1.0, 0.0];
        let brier: f64 = brier_score_loss(&y_true, &y_prob).unwrap();
        assert_abs_diff_eq!(brier, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_brier_known_value() {
        // y_true = [1, 0], y_prob = [0.8, 0.4]
        // (1 - 0.8)^2 = 0.04, (0 - 0.4)^2 = 0.16
        // mean = (0.04 + 0.16) / 2 = 0.10
        let y_true = array![1.0, 0.0];
        let y_prob = array![0.8, 0.4];
        let brier: f64 = brier_score_loss(&y_true, &y_prob).unwrap();
        assert_abs_diff_eq!(brier, 0.10, epsilon = 1e-10);
    }

    #[test]
    fn test_brier_worst_case() {
        // Completely wrong: 0 predicted where 1 is true and vice versa.
        let y_true = array![0.0, 1.0];
        let y_prob = array![1.0, 0.0];
        let brier: f64 = brier_score_loss(&y_true, &y_prob).unwrap();
        assert_abs_diff_eq!(brier, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_brier_uniform() {
        // Predicting 0.5 for balanced binary: (0-0.5)^2 + (1-0.5)^2 = 0.25 + 0.25
        // mean = 0.25
        let y_true = array![0.0, 1.0];
        let y_prob = array![0.5, 0.5];
        let brier: f64 = brier_score_loss(&y_true, &y_prob).unwrap();
        assert_abs_diff_eq!(brier, 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_brier_length_mismatch_error() {
        let y_true = array![0.0, 1.0];
        let y_prob = array![0.5, 0.5, 0.5];
        assert!(brier_score_loss(&y_true, &y_prob).is_err());
    }

    #[test]
    fn test_brier_empty_error() {
        let y_true: Array1<f64> = array![];
        let y_prob: Array1<f64> = array![];
        assert!(brier_score_loss(&y_true, &y_prob).is_err());
    }

    #[test]
    fn test_brier_f32() {
        let y_true: Array1<f32> = array![0.0f32, 1.0];
        let y_prob: Array1<f32> = array![0.0f32, 1.0];
        let brier = brier_score_loss(&y_true, &y_prob).unwrap();
        assert_abs_diff_eq!(brier, 0.0f32, epsilon = 1e-6);
    }
}
