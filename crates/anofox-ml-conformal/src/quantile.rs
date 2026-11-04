//! The finite-sample conformal quantile.

use anofox_ml_core::{Float, Result, RustMlError};

pub(crate) fn check_alpha(alpha: f64) -> Result<()> {
    if alpha > 0.0 && alpha < 1.0 {
        Ok(())
    } else {
        Err(RustMlError::InvalidParameter(format!(
            "alpha must lie in (0, 1), got {alpha}"
        )))
    }
}

/// The `⌈level·(n+1)⌉`-th smallest of `values` (numpy's
/// `quantile(values, ⌈level·(n+1)⌉ / n, method="lower")`, as MAPIE
/// computes it), or `None` when that rank exceeds `n`. Reorders `values`.
pub(crate) fn lower_quantile<F: Float>(values: &mut [F], level: f64) -> Option<F> {
    let n = values.len();
    if n == 0 {
        return None;
    }
    let q = (level * (n as f64 + 1.0)).ceil() / n as f64;
    if q > 1.0 {
        return None;
    }
    let idx = ((n - 1) as f64 * q.max(0.0)).floor() as usize;
    let (_, v, _) = values.select_nth_unstable_by(idx, |a, b| a.partial_cmp(b).expect("no NaN"));
    Some(*v)
}

/// The conformal quantile of `scores` at miscoverage `alpha`: the
/// `⌈(1−α)(n+1)⌉`-th smallest score, or `None` when `n` is too small for
/// that rank (fewer than `1/α − 1` scores), in which case a conformal
/// interval is unbounded.
///
/// ```
/// let scores: Vec<f64> = (1..=9).map(f64::from).collect();
/// // ⌈0.8 · 10⌉ = 8th smallest
/// assert_eq!(anofox_ml_conformal::conformal_quantile(&scores, 0.2).unwrap(), Some(8.0));
/// // ⌈0.95 · 10⌉ = 10 > 9 scores
/// assert_eq!(anofox_ml_conformal::conformal_quantile(&scores, 0.05).unwrap(), None);
/// ```
pub fn conformal_quantile<F: Float>(scores: &[F], alpha: f64) -> Result<Option<F>> {
    check_alpha(alpha)?;
    if scores.iter().any(|s| s.is_nan()) {
        return Err(RustMlError::InvalidParameter(
            "conformity scores contain NaN".into(),
        ));
    }
    let mut s = scores.to_vec();
    Ok(lower_quantile(&mut s, 1.0 - alpha))
}

/// Number of calibration scores needed for a bounded interval at `alpha`.
pub(crate) fn needed(alpha: f64) -> usize {
    (((1.0 - alpha) / alpha).ceil() as usize).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_is_the_finite_sample_order_statistic() {
        let scores: Vec<f64> = (1..=99).map(f64::from).collect();
        // ⌈0.9 · 100⌉ = 90
        assert_eq!(conformal_quantile(&scores, 0.1).unwrap(), Some(90.0));
        // ⌈0.5 · 100⌉ = 50
        assert_eq!(conformal_quantile(&scores, 0.5).unwrap(), Some(50.0));
        assert_eq!(conformal_quantile(&scores[..8], 0.1).unwrap(), None);
        assert!(conformal_quantile(&scores, 0.0).is_err());
        assert!(conformal_quantile(&[1.0, f64::NAN], 0.5).is_err());
        assert_eq!(needed(0.1), 9);
        assert_eq!(conformal_quantile(&scores[..9], 0.1).unwrap(), Some(9.0));
    }
}
