//! Metrics for prediction intervals (MAPIE's `regression_coverage_score`,
//! `regression_mean_width_score`) and the interval (Winkler) score.

use anofox_ml_core::{Float, Result, RustMlError};
use ndarray::Array1;

fn check<F: Float>(y: &Array1<F>, lower: &Array1<F>, upper: &Array1<F>) -> Result<()> {
    if y.len() != lower.len() || y.len() != upper.len() {
        return Err(RustMlError::ShapeMismatch(format!(
            "y has {} values, lower {}, upper {}",
            y.len(),
            lower.len(),
            upper.len()
        )));
    }
    if y.is_empty() {
        return Err(RustMlError::EmptyInput("input arrays are empty".into()));
    }
    Ok(())
}

/// Fraction of `y` inside `[lower, upper]` (bounds included).
pub fn coverage_score<F: Float>(y: &Array1<F>, lower: &Array1<F>, upper: &Array1<F>) -> Result<F> {
    check(y, lower, upper)?;
    let inside = y
        .iter()
        .zip(lower.iter().zip(upper.iter()))
        .filter(|(&v, (&l, &u))| l <= v && v <= u)
        .count();
    Ok(F::from_usize(inside).unwrap() / F::from_usize(y.len()).unwrap())
}

/// Mean of `upper − lower`.
pub fn mean_interval_width<F: Float>(lower: &Array1<F>, upper: &Array1<F>) -> Result<F> {
    check(lower, lower, upper)?;
    let sum = lower
        .iter()
        .zip(upper.iter())
        .fold(F::zero(), |acc, (&l, &u)| acc + (u - l));
    Ok(sum / F::from_usize(lower.len()).unwrap())
}

/// Mean interval score of central `(1 − alpha)` intervals (Gneiting &
/// Raftery 2007; the Winkler score): `(u − l) + (2/α)(l − y)` when
/// `y < l`, `+ (2/α)(y − u)` when `y > u`. Lower is better; it rewards
/// narrow intervals and penalises misses.
pub fn interval_score<F: Float>(
    y: &Array1<F>,
    lower: &Array1<F>,
    upper: &Array1<F>,
    alpha: F,
) -> Result<F> {
    check(y, lower, upper)?;
    if !(alpha > F::zero() && alpha < F::one()) {
        return Err(RustMlError::InvalidParameter(
            "alpha must lie in (0, 1)".into(),
        ));
    }
    let k = (F::one() + F::one()) / alpha;
    let sum =
        y.iter()
            .zip(lower.iter().zip(upper.iter()))
            .fold(F::zero(), |acc, (&v, (&l, &u))| {
                let penalty = if v < l {
                    k * (l - v)
                } else if v > u {
                    k * (v - u)
                } else {
                    F::zero()
                };
                acc + (u - l) + penalty
            });
    Ok(sum / F::from_usize(y.len()).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn interval_metrics() {
        let y = array![1.0, 2.0, 3.0, 10.0];
        let lo = array![0.0, 2.0, 3.5, 8.0];
        let hi = array![2.0, 4.0, 5.0, 9.0];
        assert_eq!(coverage_score(&y, &lo, &hi).unwrap(), 0.5);
        assert_eq!(mean_interval_width(&lo, &hi).unwrap(), 1.625);
        // widths 2+2+1.5+1 = 6.5; misses: (3.5-3)*20 = 10, (10-9)*20 = 20
        assert_eq!(
            interval_score(&y, &lo, &hi, 0.1).unwrap(),
            (6.5 + 30.0) / 4.0
        );
        assert!(coverage_score(&y, &lo, &array![1.0]).is_err());
        assert!(interval_score(&y, &lo, &hi, 0.0).is_err());
    }
}
