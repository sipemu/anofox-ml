use anofox_ml_core::{Float, Result, RustMlError};
use ndarray::Array1;

/// Cumulative false/true positive counts at every distinct score threshold,
/// mirroring sklearn's `_binary_clf_curve`.
///
/// Returns `(fps, tps, thresholds)` with `thresholds` strictly decreasing
/// (one entry per distinct score) and `fps[i]` / `tps[i]` the number of
/// negatives / positives with `score >= thresholds[i]`. Tied scores are
/// grouped into a single threshold, so the result does not depend on the
/// input order of tied samples.
///
/// `y_true` must already be validated as binary. Returns `InvalidParameter`
/// if any score is NaN.
pub(crate) fn binary_clf_curve<F: Float>(
    y_true: &Array1<F>,
    y_score: &Array1<F>,
) -> Result<(Vec<F>, Vec<F>, Vec<F>)> {
    if y_score.iter().any(|v| v.is_nan()) {
        return Err(RustMlError::InvalidParameter(
            "y_score contains NaN values".into(),
        ));
    }
    let eps = F::from_f64(1e-9).unwrap();

    // Sort indices by score descending. No NaNs remain, so the f64 total
    // order agrees with the numeric order (f32 -> f64 is order preserving).
    let mut indices: Vec<usize> = (0..y_true.len()).collect();
    indices.sort_by(|&a, &b| {
        let (sa, sb) = (y_score[a].to_f64().unwrap(), y_score[b].to_f64().unwrap());
        sb.total_cmp(&sa)
    });

    let mut fps = Vec::new();
    let mut tps = Vec::new();
    let mut thresholds = Vec::new();
    let mut tp = F::zero();
    let mut fp = F::zero();

    let mut i = 0;
    while i < indices.len() {
        let current = y_score[indices[i]];
        // `==` groups -0.0 with 0.0 and is exact otherwise (like sklearn).
        while i < indices.len() && y_score[indices[i]] == current {
            if (y_true[indices[i]] - F::one()).abs() < eps {
                tp += F::one();
            } else {
                fp += F::one();
            }
            i += 1;
        }
        fps.push(fp);
        tps.push(tp);
        thresholds.push(current);
    }

    Ok((fps, tps, thresholds))
}

/// Compute the Receiver Operating Characteristic (ROC) curve.
///
/// Returns `(fpr, tpr, thresholds)` where:
/// - `fpr` is the array of false positive rates,
/// - `tpr` is the array of true positive rates,
/// - `thresholds` is the array of decision thresholds (sorted descending).
///
/// The curve starts at `(0, 0)` with the highest threshold and ends at
/// `(1, 1)` with the lowest. Tied scores form a single threshold.
/// `y_true` must contain binary labels (only 0 and 1 values).
///
/// # Errors
///
/// Returns an error if:
/// - The inputs have different lengths.
/// - The inputs are empty.
/// - `y_true` contains non-binary labels.
/// - `y_true` contains only a single class.
/// - `y_score` contains NaN.
pub fn roc_curve<F: Float>(
    y_true: &Array1<F>,
    y_score: &Array1<F>,
) -> Result<(Array1<F>, Array1<F>, Array1<F>)> {
    check_lengths(y_true, y_score)?;
    check_binary(y_true)?;

    let (fps, tps, thresholds) = binary_clf_curve(y_true, y_score)?;
    let total_pos = *tps.last().unwrap();
    let total_neg = *fps.last().unwrap();

    if total_pos == F::zero() || total_neg == F::zero() {
        return Err(RustMlError::InvalidParameter(
            "ROC curve requires both positive and negative samples".into(),
        ));
    }

    let mut fpr_vec = Vec::with_capacity(fps.len() + 1);
    let mut tpr_vec = Vec::with_capacity(tps.len() + 1);
    let mut thresh_vec = Vec::with_capacity(thresholds.len() + 1);
    fpr_vec.push(F::zero());
    tpr_vec.push(F::zero());
    thresh_vec.push(thresholds[0] + F::one()); // sentinel above max score

    for ((&fp, &tp), &t) in fps.iter().zip(tps.iter()).zip(thresholds.iter()) {
        fpr_vec.push(fp / total_neg);
        tpr_vec.push(tp / total_pos);
        thresh_vec.push(t);
    }

    Ok((
        Array1::from_vec(fpr_vec),
        Array1::from_vec(tpr_vec),
        Array1::from_vec(thresh_vec),
    ))
}

/// Compute the precision-recall curve (same conventions as
/// `sklearn.metrics.precision_recall_curve`).
///
/// Returns `(precision, recall, thresholds)` where:
/// - `thresholds` holds the distinct scores in **increasing** order;
/// - `precision[i]` / `recall[i]` are the precision and recall when
///   predicting positive for `score >= thresholds[i]`;
/// - `precision` and `recall` have `thresholds.len() + 1` entries: the last
///   element is the sentinel `precision = 1, recall = 0`, which has no
///   threshold.
///
/// Recall is therefore non-increasing along the arrays. Tied scores form a
/// single threshold. `y_true` must contain binary labels (only 0 and 1 values).
///
/// # Errors
///
/// Returns an error if:
/// - The inputs have different lengths.
/// - The inputs are empty.
/// - `y_true` contains non-binary labels.
/// - `y_true` contains no positive samples.
/// - `y_score` contains NaN.
pub fn precision_recall_curve<F: Float>(
    y_true: &Array1<F>,
    y_score: &Array1<F>,
) -> Result<(Array1<F>, Array1<F>, Array1<F>)> {
    check_lengths(y_true, y_score)?;
    check_binary(y_true)?;

    let (fps, tps, thresholds) = binary_clf_curve(y_true, y_score)?;
    let total_pos = *tps.last().unwrap();

    if total_pos == F::zero() {
        return Err(RustMlError::InvalidParameter(
            "precision-recall curve requires at least one positive sample".into(),
        ));
    }

    let m = thresholds.len();
    let mut precision_vec = Vec::with_capacity(m + 1);
    let mut recall_vec = Vec::with_capacity(m + 1);
    // Walk from the lowest threshold (all samples predicted positive) up.
    for i in (0..m).rev() {
        precision_vec.push(tps[i] / (tps[i] + fps[i]));
        recall_vec.push(tps[i] / total_pos);
    }
    precision_vec.push(F::one());
    recall_vec.push(F::zero());

    let thresh_vec: Vec<F> = thresholds.into_iter().rev().collect();

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
        // sklearn: precision=[.5, 2/3, .5, 1, 1], recall=[1, 1, .5, .5, 0],
        //          thresholds=[.2, .3, .8, .9]
        let y_true = array![1.0, 0.0, 1.0, 0.0];
        let y_score = array![0.9, 0.8, 0.3, 0.2];
        let (precision, recall, thresholds) = precision_recall_curve(&y_true, &y_score).unwrap();

        let exp_p = [0.5, 2.0 / 3.0, 0.5, 1.0, 1.0];
        let exp_r = [1.0, 1.0, 0.5, 0.5, 0.0];
        let exp_t = [0.2, 0.3, 0.8, 0.9];
        assert_eq!(precision.len(), thresholds.len() + 1);
        assert_eq!(recall.len(), thresholds.len() + 1);
        for i in 0..exp_p.len() {
            assert_abs_diff_eq!(precision[i], exp_p[i], epsilon = 1e-12);
            assert_abs_diff_eq!(recall[i], exp_r[i], epsilon = 1e-12);
        }
        for i in 0..exp_t.len() {
            assert_abs_diff_eq!(thresholds[i], exp_t[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn test_pr_curve_matches_sklearn_issue_34() {
        // sklearn: p=[2/3, 1, 1, 1], r=[1, 1, .5, 0], thresholds=[.1, .5, .9]
        let y_true = array![0.0, 1.0, 1.0];
        let y_score = array![0.1, 0.5, 0.9];
        let (p, r, t) = precision_recall_curve(&y_true, &y_score).unwrap();
        let exp_p = [2.0 / 3.0, 1.0, 1.0, 1.0];
        let exp_r = [1.0, 1.0, 0.5, 0.0];
        let exp_t = [0.1, 0.5, 0.9];
        assert_eq!(p.len(), 4);
        assert_eq!(r.len(), 4);
        assert_eq!(t.len(), 3);
        for i in 0..4 {
            assert_abs_diff_eq!(p[i], exp_p[i], epsilon = 1e-12);
            assert_abs_diff_eq!(r[i], exp_r[i], epsilon = 1e-12);
        }
        for i in 0..3 {
            assert_abs_diff_eq!(t[i], exp_t[i], epsilon = 1e-12);
        }
        // recall non-increasing, thresholds increasing
        for i in 1..r.len() {
            assert!(r[i] <= r[i - 1]);
        }
        for i in 1..t.len() {
            assert!(t[i] > t[i - 1]);
        }
    }

    #[test]
    fn test_pr_curve_ties_grouped() {
        // y=[0,1,0,1], s=[.3,.3,.7,.7] -> thresholds [.3, .7]
        // at .7: tp=1, fp=1 -> p=.5, r=.5; at .3: p=.5, r=1
        let y_true = array![0.0, 1.0, 0.0, 1.0];
        let y_score = array![0.3, 0.3, 0.7, 0.7];
        let (p, r, t) = precision_recall_curve(&y_true, &y_score).unwrap();
        assert_eq!(t.to_vec(), vec![0.3, 0.7]);
        assert_eq!(p.to_vec(), vec![0.5, 0.5, 1.0]);
        assert_eq!(r.to_vec(), vec![1.0, 0.5, 0.0]);
    }

    #[test]
    fn test_roc_curve_ties_grouped() {
        let y_true = array![1.0, 0.0];
        let y_score = array![0.5, 0.5];
        let (fpr, tpr, thr) = roc_curve(&y_true, &y_score).unwrap();
        assert_eq!(fpr.to_vec(), vec![0.0, 1.0]);
        assert_eq!(tpr.to_vec(), vec![0.0, 1.0]);
        assert_eq!(thr.len(), 2);
    }

    #[test]
    fn test_curves_nan_score_error() {
        let y_true = array![0.0, 1.0, 1.0];
        let y_score = array![0.1, f64::NAN, 0.9];
        assert!(roc_curve(&y_true, &y_score).is_err());
        assert!(precision_recall_curve(&y_true, &y_score).is_err());
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
        // Sentinel (last element): precision=1, recall=0
        assert_abs_diff_eq!(*precision.last().unwrap(), 1.0f32, epsilon = 1e-6);
        assert_abs_diff_eq!(*recall.last().unwrap(), 0.0f32, epsilon = 1e-6);
        // First element: everything predicted positive -> recall 1
        assert_abs_diff_eq!(recall[0], 1.0f32, epsilon = 1e-6);
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
