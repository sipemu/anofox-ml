//! Canonical class-label handling shared by the crates.
//!
//! Labels are compared exactly (no epsilon) on a total order, with `-0.0`
//! and `0.0` treated as the same label; NaN labels are rejected.

use std::cmp::Ordering;

use ndarray::{ArrayBase, Data, Ix1};

use crate::error::{Result, RustMlError};
use crate::float::Float;

/// Canonical representative of a label (`-0.0` becomes `0.0`).
#[inline]
pub fn canonical_label<F: Float>(v: F) -> F {
    v + F::zero()
}

/// Total order on labels.
#[inline]
pub fn label_cmp<F: Float>(a: &F, b: &F) -> Ordering {
    // f32 -> f64 is exact and order-preserving.
    a.to_f64().unwrap().total_cmp(&b.to_f64().unwrap())
}

/// Sorted distinct labels; errors on NaN.
pub fn sorted_unique_labels<F: Float>(values: impl Iterator<Item = F>) -> Result<Vec<F>> {
    let mut vals: Vec<F> = Vec::new();
    for v in values {
        if v.is_nan() {
            return Err(RustMlError::InvalidParameter("labels contain NaN".into()));
        }
        vals.push(canonical_label(v));
    }
    vals.sort_by(label_cmp);
    vals.dedup_by(|a, b| label_cmp(a, b) == Ordering::Equal);
    Ok(vals)
}

/// Position of `v` in a sorted label list (from [`sorted_unique_labels`]).
#[inline]
pub fn label_index<F: Float>(classes: &[F], v: F) -> Option<usize> {
    let v = canonical_label(v);
    classes.binary_search_by(|c| label_cmp(c, &v)).ok()
}

/// Encode `y` as `(sorted classes, class index per sample)`.
pub fn encode_labels<F: Float, S: Data<Elem = F>>(
    y: &ArrayBase<S, Ix1>,
) -> Result<(Vec<F>, Vec<usize>)> {
    let classes = sorted_unique_labels(y.iter().copied())?;
    let idx = y
        .iter()
        .map(|&v| label_index(&classes, v).expect("label present"))
        .collect();
    Ok((classes, idx))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn encodes_exactly() {
        let (c, i) = encode_labels(&array![2.0, -0.0, 0.0, 1.0 + 1e-12, 2.0]).unwrap();
        assert_eq!(c, vec![0.0, 1.0 + 1e-12, 2.0]);
        assert_eq!(i, vec![2, 0, 0, 1, 2]);
        assert!(encode_labels(&array![1.0, f64::NAN]).is_err());
    }
}
