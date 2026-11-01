//! Shared label-encoding helpers for the classification metrics.
//!
//! Class labels are compared **exactly** (after mapping `-0.0` to `0.0`), using
//! the IEEE total order for sorting — no epsilon matching. One pass sorts the
//! distinct labels, a binary search encodes each sample in `O(log C)`.

use anofox_ml_core::{Float, Result, RustMlError};
use ndarray::{Array1, Array2};

/// Canonical representation of a label: maps `-0.0` to `+0.0`.
pub(crate) use anofox_ml_core::labels::canonical_label as canon;

pub(crate) fn check_labels_finite<F: Float>(arr: &Array1<F>, name: &str) -> Result<()> {
    if arr.iter().any(|v| v.is_nan()) {
        return Err(RustMlError::InvalidParameter(format!(
            "{} contains NaN labels",
            name
        )));
    }
    Ok(())
}

/// Sorted distinct labels of all `values`, canonicalised.
pub(crate) fn sorted_unique<F: Float>(values: impl Iterator<Item = F>) -> Result<Vec<F>> {
    anofox_ml_core::labels::sorted_unique_labels(values)
}

/// Position of `v` in the sorted label list, if present.
#[inline]
pub(crate) fn encode<F: Float>(classes: &[F], v: F) -> Option<usize> {
    anofox_ml_core::labels::label_index(classes, v)
}

/// Validate an optional `sample_weight` vector (length + finiteness).
pub(crate) fn check_sample_weight<F: Float>(w: Option<&Array1<F>>, n: usize) -> Result<()> {
    if let Some(w) = w {
        if w.len() != n {
            return Err(RustMlError::ShapeMismatch(format!(
                "sample_weight length {} != n_samples {}",
                w.len(),
                n
            )));
        }
        if w.iter().any(|v| !v.is_finite()) {
            return Err(RustMlError::InvalidParameter(
                "sample_weight contains non-finite values".into(),
            ));
        }
    }
    Ok(())
}

#[inline]
pub(crate) fn weight_at<F: Float>(w: Option<&Array1<F>>, i: usize) -> F {
    w.map_or(F::one(), |w| w[i])
}

/// (Weighted) confusion matrix over the sorted label set `classes`, which
/// must contain every value of `y_true` and `y_pred`. Single pass over the
/// samples, `O(n log C)`.
pub(crate) fn confusion_counts<F: Float>(
    y_true: &Array1<F>,
    y_pred: &Array1<F>,
    classes: &[F],
    sample_weight: Option<&Array1<F>>,
) -> Array2<F> {
    let c = classes.len();
    let mut cm = Array2::<F>::zeros((c, c));
    for (i, (&t, &p)) in y_true.iter().zip(y_pred.iter()).enumerate() {
        let a = encode(classes, t).expect("label set must cover y_true");
        let b = encode(classes, p).expect("label set must cover y_pred");
        cm[[a, b]] += weight_at(sample_weight, i);
    }
    cm
}
