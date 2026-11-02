//! Helpers shared by all Naive Bayes estimators: label encoding, sample-weight
//! validation, the discrete count/prior machinery of sklearn's
//! `_BaseDiscreteNB`, and the `jll -> predict / predict_proba /
//! predict_log_proba` conversions of `_BaseNB`.

use anofox_ml_core::{Float, Result, RustMlError};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

/// Smallest admissible smoothing parameter when `force_alpha = false`
/// (sklearn's `_ALPHA_MIN`).
pub(crate) const ALPHA_MIN: f64 = 1e-10;

#[inline]
fn label_tol<F: Float>() -> F {
    F::from_f64(1e-12).unwrap()
}

/// Sorted unique labels (tolerance `1e-12`).
pub(crate) fn sorted_unique<F: Float>(values: &[F]) -> Vec<F> {
    let mut labels: Vec<F> = values.to_vec();
    labels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    labels.dedup_by(|a, b| (*a - *b).abs() < label_tol());
    labels
}

/// Map every label in `y` to its index in the sorted `classes`, erroring on
/// labels that are not part of `classes` (sklearn raises `ValueError`).
pub(crate) fn encode_labels<F: Float>(y: ArrayView1<F>, classes: &[F]) -> Result<Vec<usize>> {
    let tol = label_tol::<F>();
    y.iter()
        .map(|&v| {
            classes
                .iter()
                .position(|&c| (c - v).abs() < tol)
                .ok_or_else(|| {
                    RustMlError::InvalidParameter(format!(
                        "label {v:?} is not in the declared classes {classes:?}"
                    ))
                })
        })
        .collect()
}

/// Validate the classes passed to `partial_fit`: on later calls they must be
/// identical to the classes of the first call.
pub(crate) fn check_partial_fit_classes<F: Float>(
    existing: &[F],
    classes: Option<&[F]>,
) -> Result<()> {
    if let Some(c) = classes {
        let c = sorted_unique(c);
        let same = c.len() == existing.len()
            && c.iter()
                .zip(existing.iter())
                .all(|(a, b)| (*a - *b).abs() < label_tol());
        if !same {
            return Err(RustMlError::InvalidParameter(format!(
                "`classes` = {c:?} is not the same as on the first call to partial_fit \
                 (was {existing:?})"
            )));
        }
    }
    Ok(())
}

/// Common shape checks for `fit` / `partial_fit`.
pub(crate) fn check_xy<F: Float>(
    x: ArrayView2<F>,
    y: ArrayView1<F>,
    sample_weight: Option<ArrayView1<F>>,
) -> Result<()> {
    if x.is_empty() || y.is_empty() {
        return Err(RustMlError::EmptyInput(
            "training data must not be empty".into(),
        ));
    }
    if x.nrows() != y.len() {
        return Err(RustMlError::ShapeMismatch(format!(
            "X has {} rows but y has {} elements",
            x.nrows(),
            y.len()
        )));
    }
    if let Some(w) = sample_weight {
        if w.len() != y.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "sample_weight len {} != y len {}",
                w.len(),
                y.len()
            )));
        }
        if w.iter().any(|v| !v.is_finite() || *v < F::zero()) {
            return Err(RustMlError::InvalidParameter(
                "sample_weight must be finite and non-negative".into(),
            ));
        }
    }
    Ok(())
}

/// Feature-count check at prediction time.
pub(crate) fn check_n_features<F: Float>(x: ArrayView2<F>, expected: usize) -> Result<()> {
    if x.ncols() != expected {
        return Err(RustMlError::ShapeMismatch(format!(
            "expected {} features, got {}",
            expected,
            x.ncols()
        )));
    }
    Ok(())
}

/// Weighted one-hot label matrix `Y` (shape `(n_samples, n_classes)`), as
/// built by sklearn's `LabelBinarizer` + `Y *= sample_weight`.
pub(crate) fn weighted_one_hot<F: Float>(
    idx: &[usize],
    n_classes: usize,
    sample_weight: Option<ArrayView1<F>>,
) -> Array2<F> {
    let mut y = Array2::<F>::zeros((idx.len(), n_classes));
    for (i, &c) in idx.iter().enumerate() {
        y[[i, c]] = sample_weight.map_or(F::one(), |w| w[i]);
    }
    y
}

/// Validate a smoothing parameter following sklearn 1.8's `_check_alpha`:
/// negative values are rejected; values below `1e-10` are clamped to `1e-10`
/// unless `force_alpha` is set.
pub(crate) fn check_alpha(alpha: f64, force_alpha: bool) -> Result<f64> {
    if !alpha.is_finite() || alpha < 0.0 {
        return Err(RustMlError::InvalidParameter(format!(
            "alpha must be a finite value >= 0, got {alpha}"
        )));
    }
    if alpha < ALPHA_MIN && !force_alpha {
        return Ok(ALPHA_MIN);
    }
    Ok(alpha)
}

/// Validate user-supplied class priors (length, non-negativity, sum to 1).
pub(crate) fn check_priors(priors: &[f64], n_classes: usize) -> Result<()> {
    if priors.len() != n_classes {
        return Err(RustMlError::InvalidParameter(format!(
            "number of priors ({}) must match number of classes ({n_classes})",
            priors.len()
        )));
    }
    if priors.iter().any(|p| !p.is_finite() || *p < 0.0) {
        return Err(RustMlError::InvalidParameter(
            "priors must be non-negative".into(),
        ));
    }
    let s: f64 = priors.iter().sum();
    // np.isclose(s, 1.0) with its default rtol = 1e-5, atol = 1e-8.
    if (s - 1.0).abs() > 1e-8 + 1e-5 {
        return Err(RustMlError::InvalidParameter(format!(
            "the sum of the priors should be 1, got {s}"
        )));
    }
    Ok(())
}

/// sklearn's `_update_class_log_prior`.
pub(crate) fn class_log_prior<F: Float>(
    class_count: &Array1<F>,
    fit_prior: bool,
    class_prior: Option<&[f64]>,
) -> Result<Array1<F>> {
    let k = class_count.len();
    if let Some(p) = class_prior {
        if p.len() != k {
            return Err(RustMlError::InvalidParameter(format!(
                "number of priors ({}) must match number of classes ({k})",
                p.len()
            )));
        }
        return Ok(p.iter().map(|&v| F::from_f64(v).unwrap().ln()).collect());
    }
    if fit_prior {
        let total = class_count.sum().ln();
        Ok(class_count.mapv(|c| c.ln() - total))
    } else {
        let v = -F::from_usize(k).unwrap().ln();
        Ok(Array1::from_elem(k, v))
    }
}

/// Row-wise `logsumexp` of a joint log-likelihood matrix.
fn logsumexp_rows<F: Float>(jll: &Array2<F>) -> Array1<F> {
    jll.map_axis(Axis(1), |row| {
        let m = row
            .iter()
            .copied()
            .fold(F::neg_infinity(), |a, b| if b > a { b } else { a });
        if !m.is_finite() {
            return m;
        }
        let s = row.iter().fold(F::zero(), |acc, &v| acc + (v - m).exp());
        m + s.ln()
    })
}

/// `jll − logsumexp(jll)` per row (sklearn's `predict_log_proba`).
pub(crate) fn log_proba_from_jll<F: Float>(mut jll: Array2<F>) -> Array2<F> {
    let lse = logsumexp_rows(&jll);
    for (mut row, &l) in jll.rows_mut().into_iter().zip(lse.iter()) {
        row.mapv_inplace(|v| v - l);
    }
    jll
}

/// `exp(predict_log_proba)`.
pub(crate) fn proba_from_jll<F: Float>(jll: Array2<F>) -> Array2<F> {
    log_proba_from_jll(jll).mapv(|v| v.exp())
}

/// `classes[argmax(jll, axis=1)]` (first maximum wins, as `np.argmax`).
pub(crate) fn argmax_labels<F: Float>(jll: &Array2<F>, classes: &[F]) -> Array1<F> {
    jll.map_axis(Axis(1), |row| {
        let mut best = 0;
        for (j, &v) in row.iter().enumerate() {
            if v > row[best] || (row[best].is_nan() && !v.is_nan()) {
                best = j;
            }
        }
        classes[best]
    })
}

/// Element-wise comparison helper for the sklearn parity tests: absolute
/// tolerance `tol`, scaled by the magnitude for large entries (log-probas).
#[cfg(test)]
pub(crate) fn assert_close2(actual: &Array2<f64>, expected: &[&[f64]], tol: f64, what: &str) {
    assert_eq!(actual.nrows(), expected.len(), "{what}: row count");
    for (i, row) in expected.iter().enumerate() {
        assert_eq!(actual.ncols(), row.len(), "{what}: col count");
        for (j, &e) in row.iter().enumerate() {
            let a = actual[[i, j]];
            let scale = e.abs().max(1.0);
            assert!(
                (a - e).abs() <= tol * scale,
                "{what}[{i},{j}]: got {a:e}, expected {e:e}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn log_proba_is_stable_for_large_magnitudes() {
        // exp() of these underflows; the log-space path must not.
        let jll = array![[-2000.0_f64, -1000.0, -1001.0]];
        let lp = log_proba_from_jll(jll);
        let lse = (-1000.0_f64) + (1.0 + (-1.0_f64).exp()).ln();
        assert!((lp[[0, 0]] - (-2000.0 - lse)).abs() < 1e-9);
        assert!((lp[[0, 1]] - (-1000.0 - lse)).abs() < 1e-12);
    }

    #[test]
    fn alpha_validation() {
        assert!(check_alpha(-1.0, true).is_err());
        assert_eq!(check_alpha(0.0, true).unwrap(), 0.0);
        assert_eq!(check_alpha(0.0, false).unwrap(), ALPHA_MIN);
        assert_eq!(check_alpha(0.5, false).unwrap(), 0.5);
    }
}
