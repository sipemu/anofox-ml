use ndarray::{Array1, Array2};
use rustml_core::{Float, Result, RustMlError};

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
        assert_abs_diff_eq!(accuracy_score(&y_true, &y_pred).unwrap(), 0.5, epsilon = 1e-10);
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
}
