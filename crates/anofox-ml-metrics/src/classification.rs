use anofox_ml_core::{Float, Result, RustMlError};
use ndarray::{Array1, Array2};

use crate::labels::{
    canon, check_labels_finite, check_sample_weight, confusion_counts, encode, sorted_unique,
    weight_at,
};

/// Averaging method for multi-class metrics.
///
/// Mirrors sklearn's `average=` argument. `average=None` (per-class scores) is
/// provided by the `*_per_class` functions and
/// [`precision_recall_fscore_support_per_class`] instead of a variant, since
/// it returns an array rather than a scalar.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Average {
    /// Unweighted mean of per-class metrics.
    Macro,
    /// Compute from global TP, FP, FN counts.
    Micro,
    /// Weighted mean by support (number of true instances per class).
    Weighted,
    /// Report only the score of the class `pos_label` (sklearn's default
    /// `average='binary'`). The union of `y_true` and `y_pred` must contain at
    /// most two distinct labels, and `pos_label` must be one of them if there
    /// are two.
    Binary {
        /// Label of the positive class (sklearn's default is `1`).
        pos_label: f64,
    },
}

impl Average {
    /// `Average::Binary { pos_label: 1.0 }` — sklearn's default.
    pub fn binary() -> Self {
        Average::Binary { pos_label: 1.0 }
    }
}

/// Value returned when a precision / recall / F-score has a zero denominator
/// (sklearn's `zero_division=`). sklearn's default `"warn"` behaves like
/// [`ZeroDivision::Zero`] (Rust has no warning channel).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZeroDivision {
    /// Use 0 (sklearn `zero_division="warn"` / `0`).
    #[default]
    Zero,
    /// Use 1.
    One,
    /// Use NaN; NaN classes are then ignored by macro / weighted averaging
    /// (like sklearn's `zero_division=np.nan`).
    NaN,
}

impl ZeroDivision {
    fn value<F: Float>(self) -> F {
        match self {
            ZeroDivision::Zero => F::zero(),
            ZeroDivision::One => F::one(),
            ZeroDivision::NaN => F::nan(),
        }
    }
}

/// Optional sklearn-style arguments shared by the classification metrics:
/// `labels=`, `sample_weight=` and `zero_division=`.
///
/// ```
/// use anofox_ml_metrics::{f1_score_with, Average, ClassificationOptions, ZeroDivision};
/// use ndarray::array;
///
/// let y_true = array![0.0, 1.0, 1.0, 2.0];
/// let y_pred = array![0.0, 1.0, 2.0, 2.0];
/// let w = array![1.0, 2.0, 1.0, 1.0];
/// let opts = ClassificationOptions::new()
///     .with_sample_weight(&w)
///     .with_zero_division(ZeroDivision::One);
/// let f1: f64 = f1_score_with(&y_true, &y_pred, Average::Macro, &opts).unwrap();
/// assert!(f1 > 0.0);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct ClassificationOptions<'a, F: Float> {
    /// Labels to report / average over (sklearn `labels=`). For the
    /// precision/recall/F family, samples whose label is not listed still count
    /// as false positives / negatives of the listed classes; for
    /// [`confusion_matrix_with`] they select (and order) the rows / columns.
    pub labels: Option<&'a [F]>,
    /// Per-sample weights (sklearn `sample_weight=`).
    pub sample_weight: Option<&'a Array1<F>>,
    /// Value for ill-defined scores (sklearn `zero_division=`).
    pub zero_division: ZeroDivision,
}

impl<'a, F: Float> ClassificationOptions<'a, F> {
    /// Defaults: all labels, unit weights, `zero_division = 0`.
    pub fn new() -> Self {
        Self {
            labels: None,
            sample_weight: None,
            zero_division: ZeroDivision::Zero,
        }
    }

    /// Restrict / order the labels.
    pub fn with_labels(mut self, labels: &'a [F]) -> Self {
        self.labels = Some(labels);
        self
    }

    /// Per-sample weights.
    pub fn with_sample_weight(mut self, sample_weight: &'a Array1<F>) -> Self {
        self.sample_weight = Some(sample_weight);
        self
    }

    /// Value for ill-defined (0/0) scores.
    pub fn with_zero_division(mut self, zero_division: ZeroDivision) -> Self {
        self.zero_division = zero_division;
        self
    }
}

/// Per-class precision, recall, F-beta and support (sklearn
/// `precision_recall_fscore_support(average=None)`).
#[derive(Debug, Clone, PartialEq)]
pub struct PerClassScores<F: Float> {
    /// Class labels, in the order of the arrays below (sorted, or the order of
    /// `ClassificationOptions::labels`).
    pub labels: Vec<F>,
    pub precision: Array1<F>,
    pub recall: Array1<F>,
    pub fscore: Array1<F>,
    /// (Weighted) number of true samples per class.
    pub support: Array1<F>,
}

/// Accuracy: fraction of correct predictions (exact label comparison).
pub fn accuracy_score<F: Float>(y_true: &Array1<F>, y_pred: &Array1<F>) -> Result<F> {
    accuracy_score_with(y_true, y_pred, None)
}

/// Accuracy with optional `sample_weight`: `sum(w * [y_true == y_pred]) / sum(w)`.
pub fn accuracy_score_with<F: Float>(
    y_true: &Array1<F>,
    y_pred: &Array1<F>,
    sample_weight: Option<&Array1<F>>,
) -> Result<F> {
    check_lengths(y_true, y_pred)?;
    check_sample_weight(sample_weight, y_true.len())?;
    let mut correct = F::zero();
    let mut total = F::zero();
    for (i, (&t, &p)) in y_true.iter().zip(y_pred.iter()).enumerate() {
        let w = weight_at(sample_weight, i);
        if canon(t) == canon(p) {
            correct += w;
        }
        total += w;
    }
    if total == F::zero() {
        return Err(RustMlError::InvalidParameter(
            "sum of sample_weight is zero".into(),
        ));
    }
    Ok(correct / total)
}

/// Confusion matrix over the sorted union of the labels in `y_true` and
/// `y_pred`.
///
/// Returns a (num_classes x num_classes) matrix where entry (i, j) is the count
/// of samples with true label i and predicted label j.
pub fn confusion_matrix<F: Float>(y_true: &Array1<F>, y_pred: &Array1<F>) -> Result<Array2<F>> {
    confusion_matrix_with(y_true, y_pred, &ClassificationOptions::new())
}

/// Confusion matrix with sklearn's `labels=` and `sample_weight=`.
///
/// With `labels`, rows / columns follow the given order and samples whose
/// true or predicted label is not listed are ignored; at least one listed
/// label must occur in `y_true`. `zero_division` is ignored.
pub fn confusion_matrix_with<F: Float>(
    y_true: &Array1<F>,
    y_pred: &Array1<F>,
    opts: &ClassificationOptions<'_, F>,
) -> Result<Array2<F>> {
    let (all, cm) = full_confusion(y_true, y_pred, opts)?;
    let Some(labels) = opts.labels else {
        return Ok(cm);
    };
    if labels.is_empty() {
        return Err(RustMlError::InvalidParameter(
            "'labels' should contain at least one label".into(),
        ));
    }
    let idx: Vec<usize> = labels.iter().map(|&l| encode(&all, l).unwrap()).collect();
    if !y_true.iter().any(|&t| {
        let t = canon(t);
        labels.iter().any(|&l| canon(l) == t)
    }) {
        return Err(RustMlError::InvalidParameter(
            "at least one label specified must be in y_true".into(),
        ));
    }
    let k = idx.len();
    let mut out = Array2::<F>::zeros((k, k));
    for (a, &ia) in idx.iter().enumerate() {
        for (b, &ib) in idx.iter().enumerate() {
            out[[a, b]] = cm[[ia, ib]];
        }
    }
    Ok(out)
}

/// Validates inputs and builds the confusion matrix over the sorted union of
/// `y_true`, `y_pred` and `opts.labels`. Returns `(sorted_labels, cm)`.
fn full_confusion<F: Float>(
    y_true: &Array1<F>,
    y_pred: &Array1<F>,
    opts: &ClassificationOptions<'_, F>,
) -> Result<(Vec<F>, Array2<F>)> {
    check_lengths(y_true, y_pred)?;
    check_labels_finite(y_true, "y_true")?;
    check_labels_finite(y_pred, "y_pred")?;
    check_sample_weight(opts.sample_weight, y_true.len())?;
    let extra = opts.labels.unwrap_or(&[]);
    let all = sorted_unique(
        y_true
            .iter()
            .chain(y_pred.iter())
            .chain(extra.iter())
            .copied(),
    )?;
    let cm = confusion_counts(y_true, y_pred, &all, opts.sample_weight);
    Ok((all, cm))
}

/// Sufficient statistics per selected label: `(labels, tp, pred_sum, true_sum)`.
type Counts<F> = (Vec<F>, Vec<F>, Vec<F>, Vec<F>);

/// Build the confusion matrix once and extract TP / predicted / true sums for
/// the selected labels (sklearn's `multilabel_confusion_matrix` semantics).
fn counts_for<F: Float>(
    y_true: &Array1<F>,
    y_pred: &Array1<F>,
    labels: Option<&[F]>,
    opts: &ClassificationOptions<'_, F>,
) -> Result<Counts<F>> {
    let o = ClassificationOptions {
        labels,
        sample_weight: opts.sample_weight,
        zero_division: opts.zero_division,
    };
    let (all, cm) = full_confusion(y_true, y_pred, &o)?;
    let selected: Vec<F> = match labels {
        Some(l) => l.iter().map(|&v| canon(v)).collect(),
        None => {
            // Present labels only (all == present when no labels were given).
            all.clone()
        }
    };
    let c = all.len();
    let mut tp = Vec::with_capacity(selected.len());
    let mut pred = Vec::with_capacity(selected.len());
    let mut truth = Vec::with_capacity(selected.len());
    for &l in &selected {
        let i = encode(&all, l).unwrap();
        tp.push(cm[[i, i]]);
        pred.push((0..c).fold(F::zero(), |acc, r| acc + cm[[r, i]]));
        truth.push((0..c).fold(F::zero(), |acc, k| acc + cm[[i, k]]));
    }
    Ok((selected, tp, pred, truth))
}

fn prf_divide<F: Float>(num: F, den: F, zd: ZeroDivision) -> F {
    if den == F::zero() {
        zd.value()
    } else {
        num / den
    }
}

/// `(precision, recall, fbeta)` from TP / predicted / true sums, as sklearn
/// computes them: `F = (1 + b²) tp / (b² true + pred)`.
fn prf_from_counts<F: Float>(tp: F, pred: F, truth: F, beta: F, zd: ZeroDivision) -> (F, F, F) {
    let b2 = beta * beta;
    let p = prf_divide(tp, pred, zd);
    let r = prf_divide(tp, truth, zd);
    let f = prf_divide((F::one() + b2) * tp, b2 * truth + pred, zd);
    (p, r, f)
}

fn check_beta<F: Float>(beta: F) -> Result<()> {
    if !beta.is_finite() || beta < F::zero() {
        return Err(RustMlError::InvalidParameter(
            "beta must be a finite non-negative number".into(),
        ));
    }
    Ok(())
}

/// Per-class precision, recall, F-beta and support (sklearn
/// `precision_recall_fscore_support(average=None)`), from a single confusion
/// matrix. Honors `labels`, `sample_weight` and `zero_division`.
pub fn precision_recall_fscore_support_per_class<F: Float>(
    y_true: &Array1<F>,
    y_pred: &Array1<F>,
    beta: F,
    opts: &ClassificationOptions<'_, F>,
) -> Result<PerClassScores<F>> {
    check_beta(beta)?;
    let (labels, tp, pred, truth) = counts_for(y_true, y_pred, opts.labels, opts)?;
    let k = labels.len();
    let mut precision = Array1::<F>::zeros(k);
    let mut recall = Array1::<F>::zeros(k);
    let mut fscore = Array1::<F>::zeros(k);
    for i in 0..k {
        let (p, r, f) = prf_from_counts(tp[i], pred[i], truth[i], beta, opts.zero_division);
        precision[i] = p;
        recall[i] = r;
        fscore[i] = f;
    }
    Ok(PerClassScores {
        labels,
        precision,
        recall,
        fscore,
        support: Array1::from_vec(truth),
    })
}

/// NaN-ignoring (weighted) mean, like sklearn's `_nanaverage`: all-NaN gives
/// NaN; all-zero weights fall back to the unweighted mean.
fn nan_average<F: Float>(values: &[F], weights: Option<&[F]>) -> F {
    let mut num = F::zero();
    let mut den = F::zero();
    let mut plain = F::zero();
    let mut count = 0usize;
    for (i, &v) in values.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        let w = weights.map_or(F::one(), |w| w[i]);
        num += w * v;
        den += w;
        plain += v;
        count += 1;
    }
    if count == 0 {
        F::nan()
    } else if den == F::zero() {
        plain / F::from_usize(count).unwrap()
    } else {
        num / den
    }
}

/// Averaged precision, recall and F-beta (sklearn
/// `precision_recall_fscore_support(average=...)`). Builds the confusion
/// matrix once.
pub fn precision_recall_fscore_support<F: Float>(
    y_true: &Array1<F>,
    y_pred: &Array1<F>,
    beta: F,
    average: Average,
    opts: &ClassificationOptions<'_, F>,
) -> Result<(F, F, F)> {
    check_beta(beta)?;
    let zd = opts.zero_division;
    let pos_storage;
    let labels = match average {
        Average::Binary { pos_label } => {
            check_lengths(y_true, y_pred)?;
            let present = sorted_unique(y_true.iter().chain(y_pred.iter()).copied())?;
            if present.len() > 2 {
                return Err(RustMlError::InvalidParameter(
                    "target is multiclass but average is Binary; choose Macro, Micro or Weighted"
                        .into(),
                ));
            }
            let pos = F::from_f64(pos_label).ok_or_else(|| {
                RustMlError::InvalidParameter("pos_label not representable".into())
            })?;
            if pos.is_nan() {
                return Err(RustMlError::InvalidParameter("pos_label is NaN".into()));
            }
            if present.len() == 2 && encode(&present, pos).is_none() {
                return Err(RustMlError::InvalidParameter(format!(
                    "pos_label={} is not a valid label; it should be one of {:?}",
                    pos_label, present
                )));
            }
            pos_storage = [pos];
            Some(&pos_storage[..])
        }
        _ => opts.labels,
    };
    let (_, tp, pred, truth) = counts_for(y_true, y_pred, labels, opts)?;

    if average == Average::Micro {
        let sum = |v: &[F]| v.iter().fold(F::zero(), |a, &b| a + b);
        return Ok(prf_from_counts(sum(&tp), sum(&pred), sum(&truth), beta, zd));
    }

    let k = tp.len();
    let mut ps = Vec::with_capacity(k);
    let mut rs = Vec::with_capacity(k);
    let mut fs = Vec::with_capacity(k);
    for i in 0..k {
        let (p, r, f) = prf_from_counts(tp[i], pred[i], truth[i], beta, zd);
        ps.push(p);
        rs.push(r);
        fs.push(f);
    }
    let weights = match average {
        Average::Weighted => Some(&truth[..]),
        _ => None,
    };
    Ok((
        nan_average(&ps, weights),
        nan_average(&rs, weights),
        nan_average(&fs, weights),
    ))
}

/// Precision for each class: `TP / (TP + FP)` (0 when undefined).
///
/// Returns per-class precision values over the sorted labels.
pub fn precision<F: Float>(y_true: &Array1<F>, y_pred: &Array1<F>) -> Result<Array1<F>> {
    precision_per_class(y_true, y_pred, &ClassificationOptions::new())
}

/// Recall for each class: `TP / (TP + FN)` (0 when undefined).
///
/// Returns per-class recall values over the sorted labels.
pub fn recall<F: Float>(y_true: &Array1<F>, y_pred: &Array1<F>) -> Result<Array1<F>> {
    recall_per_class(y_true, y_pred, &ClassificationOptions::new())
}

/// F1 score for each class: `2 * precision * recall / (precision + recall)`.
///
/// Returns per-class F1 values over the sorted labels.
pub fn f1_score<F: Float>(y_true: &Array1<F>, y_pred: &Array1<F>) -> Result<Array1<F>> {
    f1_per_class(y_true, y_pred, &ClassificationOptions::new())
}

/// Per-class precision with options (sklearn `precision_score(average=None)`).
pub fn precision_per_class<F: Float>(
    y_true: &Array1<F>,
    y_pred: &Array1<F>,
    opts: &ClassificationOptions<'_, F>,
) -> Result<Array1<F>> {
    Ok(precision_recall_fscore_support_per_class(y_true, y_pred, F::one(), opts)?.precision)
}

/// Per-class recall with options (sklearn `recall_score(average=None)`).
pub fn recall_per_class<F: Float>(
    y_true: &Array1<F>,
    y_pred: &Array1<F>,
    opts: &ClassificationOptions<'_, F>,
) -> Result<Array1<F>> {
    Ok(precision_recall_fscore_support_per_class(y_true, y_pred, F::one(), opts)?.recall)
}

/// Per-class F1 with options (sklearn `f1_score(average=None)`).
pub fn f1_per_class<F: Float>(
    y_true: &Array1<F>,
    y_pred: &Array1<F>,
    opts: &ClassificationOptions<'_, F>,
) -> Result<Array1<F>> {
    Ok(precision_recall_fscore_support_per_class(y_true, y_pred, F::one(), opts)?.fscore)
}

/// Precision score with averaging.
///
/// - **Macro**: unweighted mean of per-class precision values.
/// - **Micro**: global precision computed from total TP and FP across all classes.
/// - **Weighted**: weighted mean of per-class precision, weighted by support.
/// - **Binary**: precision of `pos_label`.
pub fn precision_score<F: Float>(
    y_true: &Array1<F>,
    y_pred: &Array1<F>,
    average: Average,
) -> Result<F> {
    precision_score_with(y_true, y_pred, average, &ClassificationOptions::new())
}

/// Recall score with averaging (see [`precision_score`] for the strategies).
pub fn recall_score<F: Float>(
    y_true: &Array1<F>,
    y_pred: &Array1<F>,
    average: Average,
) -> Result<F> {
    recall_score_with(y_true, y_pred, average, &ClassificationOptions::new())
}

/// F1 score with averaging (see [`precision_score`] for the strategies).
pub fn f1_score_avg<F: Float>(
    y_true: &Array1<F>,
    y_pred: &Array1<F>,
    average: Average,
) -> Result<F> {
    f1_score_with(y_true, y_pred, average, &ClassificationOptions::new())
}

/// Precision with averaging, `labels`, `sample_weight` and `zero_division`.
pub fn precision_score_with<F: Float>(
    y_true: &Array1<F>,
    y_pred: &Array1<F>,
    average: Average,
    opts: &ClassificationOptions<'_, F>,
) -> Result<F> {
    Ok(precision_recall_fscore_support(y_true, y_pred, F::one(), average, opts)?.0)
}

/// Recall with averaging, `labels`, `sample_weight` and `zero_division`.
pub fn recall_score_with<F: Float>(
    y_true: &Array1<F>,
    y_pred: &Array1<F>,
    average: Average,
    opts: &ClassificationOptions<'_, F>,
) -> Result<F> {
    Ok(precision_recall_fscore_support(y_true, y_pred, F::one(), average, opts)?.1)
}

/// F1 with averaging, `labels`, `sample_weight` and `zero_division`.
pub fn f1_score_with<F: Float>(
    y_true: &Array1<F>,
    y_pred: &Array1<F>,
    average: Average,
    opts: &ClassificationOptions<'_, F>,
) -> Result<F> {
    Ok(precision_recall_fscore_support(y_true, y_pred, F::one(), average, opts)?.2)
}

/// F-beta score with averaging and options (sklearn `fbeta_score`).
pub fn fbeta_score_with<F: Float>(
    y_true: &Array1<F>,
    y_pred: &Array1<F>,
    beta: F,
    average: Average,
    opts: &ClassificationOptions<'_, F>,
) -> Result<F> {
    Ok(precision_recall_fscore_support(y_true, y_pred, beta, average, opts)?.2)
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
