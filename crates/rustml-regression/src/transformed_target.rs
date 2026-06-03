//! Meta-estimator that transforms `y` before fitting and inverts on prediction.
//!
//! Mirrors `sklearn.compose.TransformedTargetRegressor` (function-based form).
//! The `func` is applied element-wise to `y` before training; `inverse_func`
//! is applied element-wise to predictions of the underlying regressor.

use ndarray::{Array1, Array2};
use rustml_core::{Fit, Predict, Result, RustMlError};

type ScalarFn = fn(f64) -> f64;

/// Wraps a regressor with a forward / inverse transformation applied to `y`.
pub struct TransformedTargetRegressor<R> {
    regressor: R,
    func: ScalarFn,
    inverse_func: ScalarFn,
    check_inverse: bool,
}

impl<R> TransformedTargetRegressor<R> {
    pub fn new(regressor: R, func: ScalarFn, inverse_func: ScalarFn) -> Self {
        Self {
            regressor,
            func,
            inverse_func,
            check_inverse: true,
        }
    }

    /// Disable the round-trip sanity check (sklearn's `check_inverse=False`).
    pub fn with_check_inverse(mut self, check: bool) -> Self {
        self.check_inverse = check;
        self
    }
}

/// Fitted version: holds the trained inner model plus the inverse function.
pub struct FittedTransformedTargetRegressor<F> {
    inner: F,
    inverse_func: ScalarFn,
}

impl<R> Fit<f64> for TransformedTargetRegressor<R>
where
    R: Fit<f64>,
{
    type Fitted = FittedTransformedTargetRegressor<R::Fitted>;

    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        if x.nrows() != y.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but y has {} elements",
                x.nrows(),
                y.len()
            )));
        }
        if y.is_empty() {
            return Err(RustMlError::EmptyInput("y is empty".into()));
        }

        // sklearn's check_inverse: assert inv(f(y)) ~= y on up to 10 points.
        if self.check_inverse {
            let n_check = y.len().min(10);
            for &yi in y.iter().take(n_check) {
                let round = (self.inverse_func)((self.func)(yi));
                if !round.is_finite() || (round - yi).abs() > 1e-4 * yi.abs().max(1.0) {
                    return Err(RustMlError::InvalidParameter(format!(
                        "func and inverse_func do not round-trip on y={yi} (got {round})"
                    )));
                }
            }
        }

        let y_trans = y.mapv(self.func);
        for &v in y_trans.iter() {
            if !v.is_finite() {
                return Err(RustMlError::InvalidParameter(format!(
                    "func produced a non-finite value: {v}"
                )));
            }
        }

        let inner = self.regressor.fit(x, &y_trans)?;
        Ok(FittedTransformedTargetRegressor {
            inner,
            inverse_func: self.inverse_func,
        })
    }
}

impl<F> Predict<f64> for FittedTransformedTargetRegressor<F>
where
    F: Predict<f64>,
{
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let raw = self.inner.predict(x)?;
        Ok(raw.mapv(self.inverse_func))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ridge::RidgeRegressor;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_log_exp_roundtrip_matches_direct_ridge() {
        // y is positive. Training Ridge on log(y) and inverting via exp must
        // produce strictly positive predictions; on identity-ish data the
        // wrapped regressor should agree with manually composing the
        // transformation outside.
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![2.0, 4.0, 8.0, 16.0, 32.0, 64.0];

        let inner = RidgeRegressor::new().with_lambda(1e-6);

        // Manual: fit Ridge on log(y), then exp().
        let y_log = y.mapv(f64::ln);
        let direct = inner.clone().fit(&x, &y_log).unwrap();
        let manual_pred = direct.predict(&x).unwrap().mapv(f64::exp);

        // Wrapped:
        let wrapped = TransformedTargetRegressor::new(inner, f64::ln, f64::exp);
        let fitted = wrapped.fit(&x, &y).unwrap();
        let wrap_pred = fitted.predict(&x).unwrap();

        for (a, b) in wrap_pred.iter().zip(manual_pred.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-9);
            assert!(*a > 0.0);
        }
    }

    #[test]
    fn test_check_inverse_rejects_bad_pair() {
        let x = array![[1.0], [2.0], [3.0]];
        let y = array![1.0, 2.0, 3.0];

        // ln/ln is not a self-inverse — should be rejected.
        let bad = TransformedTargetRegressor::new(RidgeRegressor::new(), f64::ln, f64::ln);
        assert!(bad.fit(&x, &y).is_err());
    }

    #[test]
    fn test_check_inverse_off_skips_check() {
        let x = array![[1.0], [2.0], [3.0]];
        let y = array![1.0, 2.0, 3.0];

        let lax = TransformedTargetRegressor::new(RidgeRegressor::new(), f64::ln, f64::ln)
            .with_check_inverse(false);
        // Now this fits (even though the pair is not a real inverse).
        assert!(lax.fit(&x, &y).is_ok());
    }
}
