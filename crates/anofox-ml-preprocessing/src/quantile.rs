//! Shared NaN-aware order statistics used by the scalers / discretisers.
//!
//! All sorts in this crate go through [`total_cmp`], which never panics on
//! NaN (NaN sorts after every other value). Percentiles follow numpy's
//! default `method="linear"` and are computed over the non-NaN values only,
//! mirroring `np.nanpercentile` as used by sklearn's NaN-tolerant scalers.

use std::cmp::Ordering;

use anofox_ml_core::Float;

/// Total order over floats: regular ordering for numbers, NaN after
/// everything (and equal to other NaNs). Never panics.
pub(crate) fn total_cmp<F: Float>(a: &F, b: &F) -> Ordering {
    a.partial_cmp(b)
        .unwrap_or_else(|| a.is_nan().cmp(&b.is_nan()))
}

/// Total order with NaN *below* every other value. Useful for "pick the
/// largest score" / descending sorts where a NaN score must never win:
/// sort descending with `total_cmp_nan_low(b, a)` to put NaNs last.
pub(crate) fn total_cmp_nan_low<F: Float>(a: &F, b: &F) -> Ordering {
    a.partial_cmp(b)
        .unwrap_or_else(|| b.is_nan().cmp(&a.is_nan()))
}

/// Collect the non-NaN values and sort them ascending.
pub(crate) fn sorted_non_nan<F: Float>(values: impl IntoIterator<Item = F>) -> Vec<F> {
    let mut v: Vec<F> = values.into_iter().filter(|x| !x.is_nan()).collect();
    v.sort_unstable_by(total_cmp);
    v
}

/// Percentile `q ∈ [0, 1]` of an ascending, NaN-free slice using linear
/// interpolation between order statistics (numpy `method="linear"`,
/// including numpy's two-sided lerp for numerical symmetry).
///
/// Returns NaN for an empty slice (all-NaN column), like `np.nanpercentile`.
pub(crate) fn percentile_sorted<F: Float>(sorted: &[F], q: f64) -> F {
    let n = sorted.len();
    match n {
        0 => return F::nan(),
        1 => return sorted[0],
        _ => {}
    }
    let idx = q.clamp(0.0, 1.0) * (n - 1) as f64;
    let lo = (idx.floor() as usize).min(n - 1);
    let hi = (lo + 1).min(n - 1);
    let t = idx - lo as f64;
    if t == 0.0 || lo == hi {
        return sorted[lo];
    }
    let a = sorted[lo];
    let b = sorted[hi];
    let diff = b - a;
    if t >= 0.5 {
        b - diff * F::from_f64(1.0 - t).unwrap()
    } else {
        a + diff * F::from_f64(t).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_cmp_handles_nan() {
        let mut v = [3.0, f64::NAN, 1.0, 2.0, f64::NAN];
        v.sort_by(total_cmp);
        assert_eq!(&v[..3], &[1.0, 2.0, 3.0]);
        assert!(v[3].is_nan() && v[4].is_nan());
    }

    #[test]
    fn percentile_ignores_nan_and_matches_numpy() {
        let s = sorted_non_nan(vec![4.0, f64::NAN, 1.0, 3.0, 2.0]);
        assert_eq!(s, vec![1.0, 2.0, 3.0, 4.0]);
        // np.percentile([1,2,3,4], [0, 25, 50, 75, 100])
        assert_eq!(percentile_sorted(&s, 0.0), 1.0);
        assert_eq!(percentile_sorted(&s, 0.25), 1.75);
        assert_eq!(percentile_sorted(&s, 0.5), 2.5);
        assert_eq!(percentile_sorted(&s, 0.75), 3.25);
        assert_eq!(percentile_sorted(&s, 1.0), 4.0);
    }

    #[test]
    fn percentile_empty_is_nan() {
        let s: Vec<f64> = sorted_non_nan(vec![f64::NAN, f64::NAN]);
        assert!(percentile_sorted(&s, 0.5).is_nan());
    }
}
