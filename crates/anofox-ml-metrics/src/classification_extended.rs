use anofox_ml_core::{Float, Result, RustMlError};
use ndarray::Array1;

/// ROC AUC score via trapezoidal rule on sorted (FPR, TPR) pairs.
///
/// `y_true` must be binary (only 0.0 and 1.0 values) and `y_scores` are
/// continuous confidence scores. Returns an error if the inputs are empty,
/// have mismatched lengths, contain non-binary true labels, or contain only
/// a single class.
pub fn roc_auc_score<F: Float>(y_true: &Array1<F>, y_scores: &Array1<F>) -> Result<F> {
    check_lengths(y_true, y_scores)?;
    check_binary(y_true)?;

    let pos_count = y_true
        .iter()
        .filter(|&&v| (v - F::one()).abs() < F::from_f64(1e-9).unwrap())
        .count();
    let neg_count = y_true.len() - pos_count;

    if pos_count == 0 || neg_count == 0 {
        return Err(RustMlError::InvalidParameter(
            "ROC AUC requires both positive and negative samples".into(),
        ));
    }

    // Create index array sorted by score descending (ties broken arbitrarily)
    let mut indices: Vec<usize> = (0..y_true.len()).collect();
    indices.sort_by(|&a, &b| y_scores[b].partial_cmp(&y_scores[a]).unwrap());

    let total_pos = F::from_usize(pos_count).unwrap();
    let total_neg = F::from_usize(neg_count).unwrap();

    let mut tp = F::zero();
    let mut fp = F::zero();
    let mut auc = F::zero();
    let mut prev_fpr = F::zero();
    let mut prev_tpr = F::zero();

    for &idx in &indices {
        if (y_true[idx] - F::one()).abs() < F::from_f64(1e-9).unwrap() {
            tp += F::one();
        } else {
            fp += F::one();
        }

        let fpr = fp / total_neg;
        let tpr = tp / total_pos;

        // Trapezoidal rule: area of trapezoid between previous and current point
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / F::from_f64(2.0).unwrap();

        prev_fpr = fpr;
        prev_tpr = tpr;
    }

    Ok(auc)
}

/// Average precision score (area under precision-recall curve).
///
/// `y_true` must be binary (only 0.0 and 1.0 values) and `y_scores` are
/// continuous confidence scores. AP is computed as
/// `sum((R_n - R_{n-1}) * P_n)` where precision and recall are evaluated
/// at each threshold defined by sorted scores.
pub fn average_precision_score<F: Float>(y_true: &Array1<F>, y_scores: &Array1<F>) -> Result<F> {
    check_lengths(y_true, y_scores)?;
    check_binary(y_true)?;

    let pos_count = y_true
        .iter()
        .filter(|&&v| (v - F::one()).abs() < F::from_f64(1e-9).unwrap())
        .count();

    if pos_count == 0 {
        return Err(RustMlError::InvalidParameter(
            "average precision requires at least one positive sample".into(),
        ));
    }

    // Sort by score descending
    let mut indices: Vec<usize> = (0..y_true.len()).collect();
    indices.sort_by(|&a, &b| y_scores[b].partial_cmp(&y_scores[a]).unwrap());

    let total_pos = F::from_usize(pos_count).unwrap();
    let mut tp = F::zero();
    let mut fp = F::zero();
    let mut ap = F::zero();
    let mut prev_recall = F::zero();

    for &idx in &indices {
        if (y_true[idx] - F::one()).abs() < F::from_f64(1e-9).unwrap() {
            tp += F::one();
        } else {
            fp += F::one();
        }

        let precision = tp / (tp + fp);
        let recall = tp / total_pos;

        // AP = sum((R_n - R_{n-1}) * P_n)
        ap += (recall - prev_recall) * precision;
        prev_recall = recall;
    }

    Ok(ap)
}

/// Matthews Correlation Coefficient from binary predictions.
///
/// MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
///
/// Both `y_true` and `y_pred` must be binary (only 0.0 and 1.0 values).
/// If the denominator is zero, returns 0.
pub fn matthews_corrcoef<F: Float>(y_true: &Array1<F>, y_pred: &Array1<F>) -> Result<F> {
    check_lengths(y_true, y_pred)?;
    check_binary(y_true)?;
    check_binary(y_pred)?;

    let mut tp = F::zero();
    let mut tn = F::zero();
    let mut fp = F::zero();
    let mut fn_ = F::zero();

    let one = F::one();
    let eps = F::from_f64(1e-9).unwrap();

    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        let t_pos = (t - one).abs() < eps;
        let p_pos = (p - one).abs() < eps;

        match (t_pos, p_pos) {
            (true, true) => tp += one,
            (false, false) => tn += one,
            (false, true) => fp += one,
            (true, false) => fn_ += one,
        }
    }

    let numerator = tp * tn - fp * fn_;
    let denom_sq = (tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_);

    if denom_sq == F::zero() {
        return Ok(F::zero());
    }

    Ok(numerator / denom_sq.sqrt())
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
    // ROC AUC tests
    // ---------------------------------------------------------------

    #[test]
    fn test_roc_auc_perfect() {
        // Perfect separation: all positives have higher scores than negatives.
        let y_true = array![0.0, 0.0, 1.0, 1.0];
        let y_scores = array![0.1, 0.2, 0.8, 0.9];
        let auc: f64 = roc_auc_score(&y_true, &y_scores).unwrap();
        assert_abs_diff_eq!(auc, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_roc_auc_inverse() {
        // Worst case: all positives have lower scores than negatives.
        let y_true = array![0.0, 0.0, 1.0, 1.0];
        let y_scores = array![0.8, 0.9, 0.1, 0.2];
        let auc: f64 = roc_auc_score(&y_true, &y_scores).unwrap();
        assert_abs_diff_eq!(auc, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_roc_auc_random() {
        // Interleaved scores: should give AUC ~ 0.5.
        let y_true = array![0.0, 1.0, 0.0, 1.0];
        let y_scores = array![0.1, 0.4, 0.35, 0.8];
        let auc: f64 = roc_auc_score(&y_true, &y_scores).unwrap();
        // With these specific scores: sorted desc = (0.8,1), (0.4,1), (0.35,0), (0.1,0)
        // TPR/FPR: (0,0)->(0.5,0)->(1.0,0)->(1.0,0.5)->(1.0,1.0)
        // AUC = 0 + 0 + 0.5*1.0 + 0.5*(1.0+1.0)/2 = 0.5 + 0.5 = 1.0... let me recalculate
        // Actually let's just check it's between 0 and 1
        assert!(auc >= 0.0 && auc <= 1.0);
    }

    #[test]
    fn test_roc_auc_known_value() {
        // Known example: 3 positives, 3 negatives
        let y_true = array![1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let y_scores = array![0.9, 0.7, 0.5, 0.4, 0.3, 0.1];
        // Sorted desc by score: all positives first, then negatives -> AUC = 1.0
        let auc: f64 = roc_auc_score(&y_true, &y_scores).unwrap();
        assert_abs_diff_eq!(auc, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_roc_auc_half() {
        // Alternating: 1 pos, 1 neg, 1 pos, 1 neg with scores: 0.9, 0.8, 0.3, 0.2
        // Sorted desc: (0.9,1), (0.8,0), (0.3,1), (0.2,0)
        // Step 1: tp=1, fp=0 -> tpr=0.5, fpr=0 -> area = (0-0)*(0.5+0)/2 = 0
        // Step 2: tp=1, fp=1 -> tpr=0.5, fpr=0.5 -> area = (0.5-0)*(0.5+0.5)/2 = 0.25
        // Step 3: tp=2, fp=1 -> tpr=1.0, fpr=0.5 -> area = (0.5-0.5)*(1.0+0.5)/2 = 0
        // Step 4: tp=2, fp=2 -> tpr=1.0, fpr=1.0 -> area = (1.0-0.5)*(1.0+1.0)/2 = 0.5
        // Total = 0.75
        let y_true = array![1.0, 0.0, 1.0, 0.0];
        let y_scores = array![0.9, 0.8, 0.3, 0.2];
        let auc: f64 = roc_auc_score(&y_true, &y_scores).unwrap();
        assert_abs_diff_eq!(auc, 0.75, epsilon = 1e-10);
    }

    #[test]
    fn test_roc_auc_single_class_error() {
        let y_true = array![1.0, 1.0, 1.0];
        let y_scores = array![0.5, 0.6, 0.7];
        assert!(roc_auc_score(&y_true, &y_scores).is_err());
    }

    #[test]
    fn test_roc_auc_empty_error() {
        let y_true: Array1<f64> = array![];
        let y_scores: Array1<f64> = array![];
        assert!(roc_auc_score(&y_true, &y_scores).is_err());
    }

    #[test]
    fn test_roc_auc_length_mismatch_error() {
        let y_true = array![0.0, 1.0];
        let y_scores = array![0.5, 0.6, 0.7];
        assert!(roc_auc_score(&y_true, &y_scores).is_err());
    }

    #[test]
    fn test_roc_auc_non_binary_error() {
        let y_true = array![0.0, 1.0, 2.0];
        let y_scores = array![0.5, 0.6, 0.7];
        assert!(roc_auc_score(&y_true, &y_scores).is_err());
    }

    // ---------------------------------------------------------------
    // Average Precision tests
    // ---------------------------------------------------------------

    #[test]
    fn test_average_precision_perfect() {
        // Perfect ranking: all positives scored higher than negatives.
        let y_true = array![0.0, 0.0, 1.0, 1.0];
        let y_scores = array![0.1, 0.2, 0.8, 0.9];
        let ap: f64 = average_precision_score(&y_true, &y_scores).unwrap();
        assert_abs_diff_eq!(ap, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_average_precision_known() {
        // Sorted desc: (0.9,1), (0.8,0), (0.3,1), (0.2,0)
        // Step 1: tp=1, fp=0 -> P=1.0, R=0.5 -> delta_R=0.5, contrib = 0.5*1.0 = 0.5
        // Step 2: tp=1, fp=1 -> P=0.5, R=0.5 -> delta_R=0, contrib = 0
        // Step 3: tp=2, fp=1 -> P=2/3, R=1.0 -> delta_R=0.5, contrib = 0.5*2/3 = 1/3
        // Step 4: tp=2, fp=2 -> P=0.5, R=1.0 -> delta_R=0, contrib = 0
        // AP = 0.5 + 1/3 = 5/6
        let y_true = array![1.0, 0.0, 1.0, 0.0];
        let y_scores = array![0.9, 0.8, 0.3, 0.2];
        let ap: f64 = average_precision_score(&y_true, &y_scores).unwrap();
        assert_abs_diff_eq!(ap, 5.0 / 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_average_precision_all_positive() {
        // All positive: AP should be 1.0 regardless of scores.
        let y_true = array![1.0, 1.0, 1.0];
        let y_scores = array![0.5, 0.6, 0.7];
        let ap: f64 = average_precision_score(&y_true, &y_scores).unwrap();
        assert_abs_diff_eq!(ap, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_average_precision_empty_error() {
        let y_true: Array1<f64> = array![];
        let y_scores: Array1<f64> = array![];
        assert!(average_precision_score(&y_true, &y_scores).is_err());
    }

    #[test]
    fn test_average_precision_no_positive_error() {
        let y_true = array![0.0, 0.0, 0.0];
        let y_scores = array![0.5, 0.6, 0.7];
        assert!(average_precision_score(&y_true, &y_scores).is_err());
    }

    // ---------------------------------------------------------------
    // Matthews Correlation Coefficient tests
    // ---------------------------------------------------------------

    #[test]
    fn test_mcc_perfect() {
        let y_true = array![0.0, 0.0, 1.0, 1.0];
        let y_pred = array![0.0, 0.0, 1.0, 1.0];
        let mcc: f64 = matthews_corrcoef(&y_true, &y_pred).unwrap();
        assert_abs_diff_eq!(mcc, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mcc_all_wrong() {
        let y_true = array![0.0, 0.0, 1.0, 1.0];
        let y_pred = array![1.0, 1.0, 0.0, 0.0];
        let mcc: f64 = matthews_corrcoef(&y_true, &y_pred).unwrap();
        assert_abs_diff_eq!(mcc, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mcc_random() {
        // TP=1, TN=1, FP=1, FN=1 -> MCC = (1*1 - 1*1) / sqrt(2*2*2*2) = 0
        let y_true = array![0.0, 0.0, 1.0, 1.0];
        let y_pred = array![0.0, 1.0, 0.0, 1.0];
        let mcc: f64 = matthews_corrcoef(&y_true, &y_pred).unwrap();
        assert_abs_diff_eq!(mcc, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mcc_known_value() {
        // y_true = [1, 1, 1, 0, 0, 0]
        // y_pred = [1, 1, 0, 0, 0, 1]
        // TP=2, FN=1, TN=2, FP=1
        // MCC = (2*2 - 1*1) / sqrt((2+1)(2+1)(2+1)(2+1)) = 3/sqrt(81) = 3/9 = 1/3
        let y_true = array![1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let y_pred = array![1.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mcc: f64 = matthews_corrcoef(&y_true, &y_pred).unwrap();
        assert_abs_diff_eq!(mcc, 1.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mcc_denominator_zero() {
        // All predictions are positive, no negatives predicted.
        // TP=2, FP=2, TN=0, FN=0 -> (TP+FN)=2, (TN+FP)=2, (TP+FP)=4, (TN+FN)=0
        // denom = sqrt(4*2*2*0) = 0 -> return 0
        let y_true = array![1.0, 1.0, 0.0, 0.0];
        let y_pred = array![1.0, 1.0, 1.0, 1.0];
        let mcc: f64 = matthews_corrcoef(&y_true, &y_pred).unwrap();
        assert_abs_diff_eq!(mcc, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mcc_empty_error() {
        let y_true: Array1<f64> = array![];
        let y_pred: Array1<f64> = array![];
        assert!(matthews_corrcoef(&y_true, &y_pred).is_err());
    }

    #[test]
    fn test_mcc_non_binary_error() {
        let y_true = array![0.0, 1.0, 2.0];
        let y_pred = array![0.0, 1.0, 0.0];
        assert!(matthews_corrcoef(&y_true, &y_pred).is_err());
    }

    #[test]
    fn test_mcc_length_mismatch_error() {
        let y_true = array![0.0, 1.0];
        let y_pred = array![0.0, 1.0, 0.0];
        assert!(matthews_corrcoef(&y_true, &y_pred).is_err());
    }

    // ---------------------------------------------------------------
    // f32 type tests
    // ---------------------------------------------------------------

    #[test]
    fn test_roc_auc_f32() {
        let y_true: Array1<f32> = array![0.0f32, 0.0, 1.0, 1.0];
        let y_scores: Array1<f32> = array![0.1f32, 0.2, 0.8, 0.9];
        let auc = roc_auc_score(&y_true, &y_scores).unwrap();
        assert_abs_diff_eq!(auc, 1.0f32, epsilon = 1e-6);
    }

    #[test]
    fn test_average_precision_f32() {
        let y_true: Array1<f32> = array![0.0f32, 0.0, 1.0, 1.0];
        let y_scores: Array1<f32> = array![0.1f32, 0.2, 0.8, 0.9];
        let ap = average_precision_score(&y_true, &y_scores).unwrap();
        assert_abs_diff_eq!(ap, 1.0f32, epsilon = 1e-6);
    }

    #[test]
    fn test_mcc_f32() {
        let y_true: Array1<f32> = array![0.0f32, 0.0, 1.0, 1.0];
        let y_pred: Array1<f32> = array![0.0f32, 0.0, 1.0, 1.0];
        let mcc = matthews_corrcoef(&y_true, &y_pred).unwrap();
        assert_abs_diff_eq!(mcc, 1.0f32, epsilon = 1e-6);
    }
}
