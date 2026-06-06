use anofox_ml_core::{Float, Result, RustMlError};
use ndarray::{Array1, Array2};

/// Averaging method for multi-class metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Average {
    /// Unweighted mean of per-class metrics.
    Macro,
    /// Compute from global TP, FP, FN counts.
    Micro,
    /// Weighted mean by support (number of true instances per class).
    Weighted,
}

/// Accuracy: fraction of correct predictions.
pub fn accuracy_score<F: Float>(y_true: &Array1<F>, y_pred: &Array1<F>) -> Result<F> {
    check_lengths(y_true, y_pred)?;
    let n = F::from_usize(y_true.len()).unwrap();
    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(&t, &p)| (t - p).abs() < F::from_f64(1e-9).unwrap())
        .count();
    Ok(F::from_usize(correct).unwrap() / n)
}

/// Confusion matrix for integer-encoded class labels.
///
/// Returns a (num_classes x num_classes) matrix where entry (i, j) is the count
/// of samples with true label i and predicted label j.
pub fn confusion_matrix<F: Float>(y_true: &Array1<F>, y_pred: &Array1<F>) -> Result<Array2<F>> {
    check_lengths(y_true, y_pred)?;

    let classes = unique_sorted(y_true, y_pred);
    let n = classes.len();
    let mut matrix = Array2::<F>::zeros((n, n));

    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        let i = classes
            .iter()
            .position(|&c| (c - t).abs() < F::from_f64(1e-9).unwrap())
            .unwrap();
        let j = classes
            .iter()
            .position(|&c| (c - p).abs() < F::from_f64(1e-9).unwrap())
            .unwrap();
        matrix[[i, j]] += F::one();
    }

    Ok(matrix)
}

/// Precision for each class (macro-style): `TP / (TP + FP)`.
///
/// Returns per-class precision values.
pub fn precision<F: Float>(y_true: &Array1<F>, y_pred: &Array1<F>) -> Result<Array1<F>> {
    let cm = confusion_matrix(y_true, y_pred)?;
    let n = cm.nrows();
    let mut result = Array1::<F>::zeros(n);

    for i in 0..n {
        let tp = cm[[i, i]];
        let col_sum = (0..n).map(|r| cm[[r, i]]).fold(F::zero(), |a, b| a + b);
        result[i] = if col_sum > F::zero() {
            tp / col_sum
        } else {
            F::zero()
        };
    }

    Ok(result)
}

/// Recall for each class (macro-style): `TP / (TP + FN)`.
///
/// Returns per-class recall values.
pub fn recall<F: Float>(y_true: &Array1<F>, y_pred: &Array1<F>) -> Result<Array1<F>> {
    let cm = confusion_matrix(y_true, y_pred)?;
    let n = cm.nrows();
    let mut result = Array1::<F>::zeros(n);

    for i in 0..n {
        let tp = cm[[i, i]];
        let row_sum = (0..n).map(|c| cm[[i, c]]).fold(F::zero(), |a, b| a + b);
        result[i] = if row_sum > F::zero() {
            tp / row_sum
        } else {
            F::zero()
        };
    }

    Ok(result)
}

/// F1 score for each class: `2 * precision * recall / (precision + recall)`.
///
/// Returns per-class F1 values.
pub fn f1_score<F: Float>(y_true: &Array1<F>, y_pred: &Array1<F>) -> Result<Array1<F>> {
    let prec = precision(y_true, y_pred)?;
    let rec = recall(y_true, y_pred)?;
    let two = F::from_f64(2.0).unwrap();

    let mut result = Array1::<F>::zeros(prec.len());
    for i in 0..prec.len() {
        let denom = prec[i] + rec[i];
        result[i] = if denom > F::zero() {
            two * prec[i] * rec[i] / denom
        } else {
            F::zero()
        };
    }

    Ok(result)
}

/// Precision score with averaging.
///
/// Returns a single scalar precision value computed using the given averaging
/// strategy:
/// - **Macro**: unweighted mean of per-class precision values.
/// - **Micro**: global precision computed from total TP and FP across all classes.
/// - **Weighted**: weighted mean of per-class precision, weighted by support
///   (number of true instances per class).
pub fn precision_score<F: Float>(
    y_true: &Array1<F>,
    y_pred: &Array1<F>,
    average: Average,
) -> Result<F> {
    let cm = confusion_matrix(y_true, y_pred)?;
    let n = cm.nrows();

    match average {
        Average::Macro => {
            let per_class = precision(y_true, y_pred)?;
            let sum: F = per_class.iter().copied().sum();
            Ok(sum / F::from_usize(n).unwrap())
        }
        Average::Micro => {
            let (tp_total, fp_total, _fn_total) = global_tp_fp_fn(&cm);
            let denom = tp_total + fp_total;
            if denom > F::zero() {
                Ok(tp_total / denom)
            } else {
                Ok(F::zero())
            }
        }
        Average::Weighted => {
            let per_class = precision(y_true, y_pred)?;
            let supports = class_supports(&cm);
            weighted_average(&per_class, &supports)
        }
    }
}

/// Recall score with averaging.
///
/// Returns a single scalar recall value computed using the given averaging
/// strategy:
/// - **Macro**: unweighted mean of per-class recall values.
/// - **Micro**: global recall computed from total TP and FN across all classes.
/// - **Weighted**: weighted mean of per-class recall, weighted by support
///   (number of true instances per class).
pub fn recall_score<F: Float>(
    y_true: &Array1<F>,
    y_pred: &Array1<F>,
    average: Average,
) -> Result<F> {
    let cm = confusion_matrix(y_true, y_pred)?;
    let n = cm.nrows();

    match average {
        Average::Macro => {
            let per_class = recall(y_true, y_pred)?;
            let sum: F = per_class.iter().copied().sum();
            Ok(sum / F::from_usize(n).unwrap())
        }
        Average::Micro => {
            let (tp_total, _fp_total, fn_total) = global_tp_fp_fn(&cm);
            let denom = tp_total + fn_total;
            if denom > F::zero() {
                Ok(tp_total / denom)
            } else {
                Ok(F::zero())
            }
        }
        Average::Weighted => {
            let per_class = recall(y_true, y_pred)?;
            let supports = class_supports(&cm);
            weighted_average(&per_class, &supports)
        }
    }
}

/// F1 score with averaging.
///
/// Returns a single scalar F1 value computed using the given averaging
/// strategy:
/// - **Macro**: unweighted mean of per-class F1 values.
/// - **Micro**: global F1 computed from total TP, FP, FN across all classes.
/// - **Weighted**: weighted mean of per-class F1 values, weighted by support
///   (number of true instances per class).
pub fn f1_score_avg<F: Float>(
    y_true: &Array1<F>,
    y_pred: &Array1<F>,
    average: Average,
) -> Result<F> {
    let cm = confusion_matrix(y_true, y_pred)?;
    let n = cm.nrows();
    let two = F::from_f64(2.0).unwrap();

    match average {
        Average::Macro => {
            let per_class = f1_score(y_true, y_pred)?;
            let sum: F = per_class.iter().copied().sum();
            Ok(sum / F::from_usize(n).unwrap())
        }
        Average::Micro => {
            let (tp_total, fp_total, fn_total) = global_tp_fp_fn(&cm);
            let p_denom = tp_total + fp_total;
            let r_denom = tp_total + fn_total;
            let micro_p = if p_denom > F::zero() {
                tp_total / p_denom
            } else {
                F::zero()
            };
            let micro_r = if r_denom > F::zero() {
                tp_total / r_denom
            } else {
                F::zero()
            };
            let f1_denom = micro_p + micro_r;
            if f1_denom > F::zero() {
                Ok(two * micro_p * micro_r / f1_denom)
            } else {
                Ok(F::zero())
            }
        }
        Average::Weighted => {
            let per_class = f1_score(y_true, y_pred)?;
            let supports = class_supports(&cm);
            weighted_average(&per_class, &supports)
        }
    }
}

/// Compute global TP, FP, FN summed across all classes.
fn global_tp_fp_fn<F: Float>(cm: &Array2<F>) -> (F, F, F) {
    let n = cm.nrows();
    let mut tp_total = F::zero();
    let mut fp_total = F::zero();
    let mut fn_total = F::zero();

    for i in 0..n {
        let tp = cm[[i, i]];
        let col_sum: F = (0..n).map(|r| cm[[r, i]]).fold(F::zero(), |a, b| a + b);
        let row_sum: F = (0..n).map(|c| cm[[i, c]]).fold(F::zero(), |a, b| a + b);
        tp_total += tp;
        fp_total += col_sum - tp;
        fn_total += row_sum - tp;
    }

    (tp_total, fp_total, fn_total)
}

/// Compute support (number of true instances) for each class from confusion matrix.
fn class_supports<F: Float>(cm: &Array2<F>) -> Array1<F> {
    let n = cm.nrows();
    let mut supports = Array1::<F>::zeros(n);
    for i in 0..n {
        supports[i] = (0..n).map(|c| cm[[i, c]]).fold(F::zero(), |a, b| a + b);
    }
    supports
}

/// Weighted average of per-class values given supports.
fn weighted_average<F: Float>(values: &Array1<F>, supports: &Array1<F>) -> Result<F> {
    let total_support: F = supports.iter().copied().sum();
    if total_support > F::zero() {
        let weighted_sum: F = values
            .iter()
            .zip(supports.iter())
            .map(|(&v, &s)| v * s)
            .sum();
        Ok(weighted_sum / total_support)
    } else {
        Ok(F::zero())
    }
}

fn unique_sorted<F: Float>(a: &Array1<F>, b: &Array1<F>) -> Vec<F> {
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

    #[test]
    fn test_accuracy_perfect() {
        let y = array![0.0, 1.0, 2.0, 1.0];
        assert_abs_diff_eq!(accuracy_score(&y, &y).unwrap(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_accuracy_half() {
        let y_true = array![0.0, 1.0, 2.0, 1.0];
        let y_pred = array![0.0, 2.0, 2.0, 0.0];
        assert_abs_diff_eq!(
            accuracy_score(&y_true, &y_pred).unwrap(),
            0.5,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_confusion_matrix_binary() {
        let y_true = array![0.0, 0.0, 1.0, 1.0];
        let y_pred = array![0.0, 1.0, 0.0, 1.0];
        let cm = confusion_matrix(&y_true, &y_pred).unwrap();
        // [[1, 1], [1, 1]]
        assert_abs_diff_eq!(cm[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(cm[[0, 1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(cm[[1, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(cm[[1, 1]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_precision_recall_f1_perfect() {
        let y = array![0.0, 1.0, 2.0];
        let p = precision(&y, &y).unwrap();
        let r = recall(&y, &y).unwrap();
        let f = f1_score(&y, &y).unwrap();
        for i in 0..3 {
            assert_abs_diff_eq!(p[i], 1.0, epsilon = 1e-10);
            assert_abs_diff_eq!(r[i], 1.0, epsilon = 1e-10);
            assert_abs_diff_eq!(f[i], 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_precision_binary() {
        // Class 0: TP=1, FP=1 -> precision=0.5
        // Class 1: TP=1, FP=1 -> precision=0.5
        let y_true = array![0.0, 0.0, 1.0, 1.0];
        let y_pred = array![0.0, 1.0, 0.0, 1.0];
        let p = precision(&y_true, &y_pred).unwrap();
        assert_abs_diff_eq!(p[0], 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(p[1], 0.5, epsilon = 1e-10);
    }

    // ---------------------------------------------------------------
    // Tests for averaged precision, recall, F1
    // ---------------------------------------------------------------

    // Multi-class dataset verified against sklearn:
    //   y_true = [0, 0, 0, 1, 1, 2, 2, 2, 2]
    //   y_pred = [0, 1, 2, 0, 1, 0, 1, 2, 2]
    //
    // Confusion matrix:
    //          pred 0  pred 1  pred 2
    //  true 0 [  1,      1,      1  ]   support=3
    //  true 1 [  1,      1,      0  ]   support=2
    //  true 2 [  1,      1,      2  ]   support=4
    //
    // Per-class:
    //   class 0: TP=1, FP=2, FN=2  => prec=1/3, rec=1/3, f1=1/3
    //   class 1: TP=1, FP=2, FN=1  => prec=1/3, rec=1/2, f1=2/5
    //   class 2: TP=2, FP=1, FN=2  => prec=2/3, rec=2/4=1/2, f1=4/7

    fn multiclass_data() -> (Array1<f64>, Array1<f64>) {
        let y_true = array![0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0];
        let y_pred = array![0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 2.0, 2.0];
        (y_true, y_pred)
    }

    #[test]
    fn test_macro_precision() {
        let (y_true, y_pred) = multiclass_data();
        let result = precision_score(&y_true, &y_pred, Average::Macro).unwrap();
        // macro precision = (1/3 + 1/3 + 2/3) / 3 = (4/3) / 3 = 4/9
        let expected = 4.0 / 9.0;
        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_macro_recall() {
        let (y_true, y_pred) = multiclass_data();
        let result = recall_score(&y_true, &y_pred, Average::Macro).unwrap();
        // macro recall = (1/3 + 1/2 + 1/2) / 3 = (1/3 + 1/2 + 1/2) / 3
        //             = (2/6 + 3/6 + 3/6) / 3 = (8/6) / 3 = 8/18 = 4/9
        let expected = 4.0 / 9.0;
        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_macro_f1() {
        let (y_true, y_pred) = multiclass_data();
        let result = f1_score_avg(&y_true, &y_pred, Average::Macro).unwrap();
        // per-class F1: 1/3, 2/5, 4/7
        // macro = (1/3 + 2/5 + 4/7) / 3
        //       = (35/105 + 42/105 + 60/105) / 3 = (137/105) / 3 = 137/315
        let expected = 137.0 / 315.0;
        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_micro_precision_recall_f1_equal_accuracy() {
        // For multi-class single-label classification, micro precision = micro recall
        // = micro F1 = accuracy.
        let (y_true, y_pred) = multiclass_data();
        let acc = accuracy_score(&y_true, &y_pred).unwrap();

        let micro_p = precision_score(&y_true, &y_pred, Average::Micro).unwrap();
        let micro_r = recall_score(&y_true, &y_pred, Average::Micro).unwrap();
        let micro_f1 = f1_score_avg(&y_true, &y_pred, Average::Micro).unwrap();

        // accuracy = 4/9
        assert_abs_diff_eq!(acc, 4.0 / 9.0, epsilon = 1e-10);
        assert_abs_diff_eq!(micro_p, acc, epsilon = 1e-10);
        assert_abs_diff_eq!(micro_r, acc, epsilon = 1e-10);
        assert_abs_diff_eq!(micro_f1, acc, epsilon = 1e-10);
    }

    #[test]
    fn test_weighted_precision() {
        let (y_true, y_pred) = multiclass_data();
        let result = precision_score(&y_true, &y_pred, Average::Weighted).unwrap();
        // weighted precision = (1/3 * 3 + 1/3 * 2 + 2/3 * 4) / 9
        //                    = (1 + 2/3 + 8/3) / 9 = (3/3 + 2/3 + 8/3) / 9
        //                    = (13/3) / 9 = 13/27
        let expected = 13.0 / 27.0;
        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_weighted_recall() {
        let (y_true, y_pred) = multiclass_data();
        let result = recall_score(&y_true, &y_pred, Average::Weighted).unwrap();
        // weighted recall = (1/3 * 3 + 1/2 * 2 + 1/2 * 4) / 9
        //                 = (1 + 1 + 2) / 9 = 4/9
        let expected = 4.0 / 9.0;
        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_weighted_f1() {
        let (y_true, y_pred) = multiclass_data();
        let result = f1_score_avg(&y_true, &y_pred, Average::Weighted).unwrap();
        // weighted F1 = (1/3 * 3 + 2/5 * 2 + 4/7 * 4) / 9
        //             = (1 + 4/5 + 16/7) / 9
        //             = (35/35 + 28/35 + 80/35) / 9
        //             = (143/35) / 9 = 143/315
        let expected = 143.0 / 315.0;
        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_binary_classification_averaging() {
        // Binary case: y_true = [0, 0, 1, 1, 1], y_pred = [0, 1, 1, 1, 0]
        // CM:
        //          pred 0  pred 1
        //  true 0 [  1,      1  ]   support=2
        //  true 1 [  1,      2  ]   support=3
        //
        // class 0: TP=1, FP=1, FN=1 => prec=1/2, rec=1/2, f1=1/2
        // class 1: TP=2, FP=1, FN=1 => prec=2/3, rec=2/3, f1=2/3
        let y_true = array![0.0, 0.0, 1.0, 1.0, 1.0];
        let y_pred = array![0.0, 1.0, 1.0, 1.0, 0.0];

        // Macro
        let macro_p = precision_score(&y_true, &y_pred, Average::Macro).unwrap();
        assert_abs_diff_eq!(macro_p, (0.5 + 2.0 / 3.0) / 2.0, epsilon = 1e-10);

        let macro_r = recall_score(&y_true, &y_pred, Average::Macro).unwrap();
        assert_abs_diff_eq!(macro_r, (0.5 + 2.0 / 3.0) / 2.0, epsilon = 1e-10);

        let macro_f1 = f1_score_avg(&y_true, &y_pred, Average::Macro).unwrap();
        assert_abs_diff_eq!(macro_f1, (0.5 + 2.0 / 3.0) / 2.0, epsilon = 1e-10);

        // Micro = accuracy = 3/5
        let micro_p = precision_score(&y_true, &y_pred, Average::Micro).unwrap();
        assert_abs_diff_eq!(micro_p, 0.6, epsilon = 1e-10);

        let micro_f1 = f1_score_avg(&y_true, &y_pred, Average::Micro).unwrap();
        assert_abs_diff_eq!(micro_f1, 0.6, epsilon = 1e-10);

        // Weighted precision = (1/2 * 2 + 2/3 * 3) / 5 = (1 + 2) / 5 = 3/5
        let weighted_p = precision_score(&y_true, &y_pred, Average::Weighted).unwrap();
        assert_abs_diff_eq!(weighted_p, 0.6, epsilon = 1e-10);
    }

    #[test]
    fn test_perfect_predictions_all_averages() {
        let y = array![0.0, 1.0, 2.0, 0.0, 1.0, 2.0];

        for avg in [Average::Macro, Average::Micro, Average::Weighted] {
            let p = precision_score(&y, &y, avg).unwrap();
            let r = recall_score(&y, &y, avg).unwrap();
            let f = f1_score_avg(&y, &y, avg).unwrap();
            assert_abs_diff_eq!(p, 1.0, epsilon = 1e-10);
            assert_abs_diff_eq!(r, 1.0, epsilon = 1e-10);
            assert_abs_diff_eq!(f, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_single_class() {
        // All samples belong to class 0 and predictions are all class 0.
        let y_true = array![0.0, 0.0, 0.0, 0.0];
        let y_pred = array![0.0, 0.0, 0.0, 0.0];

        for avg in [Average::Macro, Average::Micro, Average::Weighted] {
            let p = precision_score(&y_true, &y_pred, avg).unwrap();
            let r = recall_score(&y_true, &y_pred, avg).unwrap();
            let f = f1_score_avg(&y_true, &y_pred, avg).unwrap();
            assert_abs_diff_eq!(p, 1.0, epsilon = 1e-10);
            assert_abs_diff_eq!(r, 1.0, epsilon = 1e-10);
            assert_abs_diff_eq!(f, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_f32_type() {
        // Ensure functions work with f32 as well.
        let y_true: Array1<f32> = array![0.0f32, 1.0, 2.0, 1.0];
        let y_pred: Array1<f32> = array![0.0f32, 2.0, 2.0, 1.0];

        let p = precision_score(&y_true, &y_pred, Average::Macro).unwrap();
        let r = recall_score(&y_true, &y_pred, Average::Macro).unwrap();
        let f = f1_score_avg(&y_true, &y_pred, Average::Macro).unwrap();

        // Verify they produce finite results (no NaN or Inf)
        assert!(p.is_finite());
        assert!(r.is_finite());
        assert!(f.is_finite());
    }

    #[test]
    fn test_all_wrong_predictions() {
        // No correct predictions at all.
        let y_true = array![0.0, 0.0, 1.0, 1.0];
        let y_pred = array![1.0, 1.0, 0.0, 0.0];

        // class 0: TP=0, FP=2, FN=2 => prec=0, rec=0, f1=0
        // class 1: TP=0, FP=2, FN=2 => prec=0, rec=0, f1=0

        for avg in [Average::Macro, Average::Micro, Average::Weighted] {
            let p = precision_score(&y_true, &y_pred, avg).unwrap();
            let r = recall_score(&y_true, &y_pred, avg).unwrap();
            let f = f1_score_avg(&y_true, &y_pred, avg).unwrap();
            assert_abs_diff_eq!(p, 0.0, epsilon = 1e-10);
            assert_abs_diff_eq!(r, 0.0, epsilon = 1e-10);
            assert_abs_diff_eq!(f, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_backward_compatibility() {
        // Ensure original per-class functions still work unchanged.
        let y_true = array![0.0, 0.0, 1.0, 1.0, 2.0];
        let y_pred = array![0.0, 1.0, 1.0, 2.0, 2.0];

        let p = precision(&y_true, &y_pred).unwrap();
        let r = recall(&y_true, &y_pred).unwrap();
        let f = f1_score(&y_true, &y_pred).unwrap();

        assert_eq!(p.len(), 3);
        assert_eq!(r.len(), 3);
        assert_eq!(f.len(), 3);
    }

    mod prop_tests {
        use super::*;
        use proptest::collection::vec;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn accuracy_bounded(
                labels in vec(0..5u32, 1..100),
                seed in 0u64..10000,
            ) {
                let n = labels.len();
                let y_true = Array1::from_vec(
                    labels.iter().map(|&v| v as f64).collect::<Vec<_>>()
                );
                // Generate different predictions using seed for determinism
                let y_pred = Array1::from_vec(
                    labels.iter().enumerate().map(|(i, &v)| {
                        ((v as u64 + seed + i as u64) % 5) as f64
                    }).collect::<Vec<_>>()
                );
                let acc = accuracy_score(&y_true, &y_pred).unwrap();
                prop_assert!((0.0..=1.0).contains(&acc),
                    "accuracy should be in [0, 1], got {} (n={})", acc, n);
            }

            #[test]
            fn perfect_accuracy(labels in vec(0..5u32, 1..100)) {
                let y = Array1::from_vec(
                    labels.iter().map(|&v| v as f64).collect::<Vec<_>>()
                );
                let acc = accuracy_score(&y, &y).unwrap();
                prop_assert!((acc - 1.0).abs() < 1e-10,
                    "accuracy(y, y) should be 1.0, got {}", acc);
            }
        }
    }
}
