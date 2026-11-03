//! Meta-estimator that transforms `y` before fitting and inverts on prediction.
//!
//! Mirrors `sklearn.compose.TransformedTargetRegressor` (function-based form).
//! The forward function is applied element-wise to `y` before training; the
//! inverse is applied element-wise to predictions of the underlying
//! regressor.
//!
//! The named transforms of [`TargetTransform`] can be persisted with the
//! fitted model; [`TargetTransform::Custom`] (arbitrary function pointers,
//! as passed to [`TransformedTargetRegressor::new`]) cannot.

use anofox_ml_core::{Fit, Predict, Result, RustMlError};
use ndarray::{Array1, ArrayView1, ArrayView2};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

type ScalarFn = fn(f64) -> f64;

/// The transformation applied to `y` (and inverted on predictions).
#[derive(Debug, Clone, Copy)]
pub enum TargetTransform {
    /// No transformation.
    Identity,
    /// `ln(y)`, inverted with `exp` (sklearn `func=np.log, inverse_func=np.exp`).
    Log,
    /// `ln(1 + y)`, inverted with `exp(·) − 1` (`np.log1p` / `np.expm1`).
    Log1p,
    /// `√y`, inverted with `·²` (`np.sqrt` / `np.square`).
    Sqrt,
    /// Arbitrary functions. Not serializable: saving a model that uses them
    /// is an error.
    Custom {
        func: ScalarFn,
        inverse_func: ScalarFn,
    },
}

impl TargetTransform {
    /// Forward transformation of one target value.
    pub fn forward(&self, v: f64) -> f64 {
        match self {
            TargetTransform::Identity => v,
            TargetTransform::Log => v.ln(),
            TargetTransform::Log1p => v.ln_1p(),
            TargetTransform::Sqrt => v.sqrt(),
            TargetTransform::Custom { func, .. } => func(v),
        }
    }

    /// Inverse transformation of one prediction.
    pub fn inverse(&self, v: f64) -> f64 {
        match self {
            TargetTransform::Identity => v,
            TargetTransform::Log => v.exp(),
            TargetTransform::Log1p => v.exp_m1(),
            TargetTransform::Sqrt => v * v,
            TargetTransform::Custom { inverse_func, .. } => inverse_func(v),
        }
    }
}

/// Serialized form: the named transforms only.
#[derive(Serialize, Deserialize)]
enum NamedTransform {
    Identity,
    Log,
    Log1p,
    Sqrt,
}

impl Serialize for TargetTransform {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        let named = match self {
            TargetTransform::Identity => NamedTransform::Identity,
            TargetTransform::Log => NamedTransform::Log,
            TargetTransform::Log1p => NamedTransform::Log1p,
            TargetTransform::Sqrt => NamedTransform::Sqrt,
            TargetTransform::Custom { .. } => {
                return Err(serde::ser::Error::custom(
                    "a TransformedTargetRegressor with custom functions cannot be \
                     serialized; use a named TargetTransform",
                ))
            }
        };
        named.serialize(s)
    }
}

impl<'de> Deserialize<'de> for TargetTransform {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        Ok(match NamedTransform::deserialize(d)? {
            NamedTransform::Identity => TargetTransform::Identity,
            NamedTransform::Log => TargetTransform::Log,
            NamedTransform::Log1p => TargetTransform::Log1p,
            NamedTransform::Sqrt => TargetTransform::Sqrt,
        })
    }
}

/// Wraps a regressor with a forward / inverse transformation applied to `y`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformedTargetRegressor<R> {
    regressor: R,
    transform: TargetTransform,
    check_inverse: bool,
}

impl<R> TransformedTargetRegressor<R> {
    /// Wrap `regressor` with arbitrary functions ([`TargetTransform::Custom`]).
    /// Use [`Self::with_transform`] for a model that can be saved.
    pub fn new(regressor: R, func: ScalarFn, inverse_func: ScalarFn) -> Self {
        Self::with_transform(regressor, TargetTransform::Custom { func, inverse_func })
    }

    /// Wrap `regressor` with a [`TargetTransform`].
    pub fn with_transform(regressor: R, transform: TargetTransform) -> Self {
        Self {
            regressor,
            transform,
            check_inverse: true,
        }
    }

    /// Disable the round-trip sanity check (sklearn's `check_inverse=False`).
    pub fn with_check_inverse(mut self, check: bool) -> Self {
        self.check_inverse = check;
        self
    }
}

/// Fitted version: holds the trained inner model plus the transformation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedTransformedTargetRegressor<F> {
    inner: F,
    transform: TargetTransform,
}

impl<R> Fit<f64> for TransformedTargetRegressor<R>
where
    R: Fit<f64>,
{
    type Fitted = FittedTransformedTargetRegressor<R::Fitted>;

    fn fit_view(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<Self::Fitted> {
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
                let round = self.transform.inverse(self.transform.forward(yi));
                if !round.is_finite() || (round - yi).abs() > 1e-4 * yi.abs().max(1.0) {
                    return Err(RustMlError::InvalidParameter(format!(
                        "func and inverse_func do not round-trip on y={yi} (got {round})"
                    )));
                }
            }
        }

        let y_trans = y.mapv(|v| self.transform.forward(v));
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
            transform: self.transform,
        })
    }
}

impl<F> Predict<f64> for FittedTransformedTargetRegressor<F>
where
    F: Predict<f64>,
{
    fn predict_view(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        let raw = self.inner.predict(x)?;
        Ok(raw.mapv(|v| self.transform.inverse(v)))
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

    #[test]
    fn test_named_transforms_match_functions() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![2.0, 4.0, 8.0, 16.0, 32.0, 64.0];
        let inner = crate::bayesian_ridge::BayesianRidge::new();
        for (named, f, inv) in [
            (
                TargetTransform::Log,
                f64::ln as ScalarFn,
                f64::exp as ScalarFn,
            ),
            (TargetTransform::Log1p, f64::ln_1p, f64::exp_m1),
            (TargetTransform::Sqrt, f64::sqrt, |v: f64| v * v),
            (TargetTransform::Identity, |v: f64| v, |v: f64| v),
        ] {
            let a = TransformedTargetRegressor::with_transform(inner.clone(), named)
                .fit(&x, &y)
                .unwrap()
                .predict(&x)
                .unwrap();
            let b = TransformedTargetRegressor::new(inner.clone(), f, inv)
                .fit(&x, &y)
                .unwrap()
                .predict(&x)
                .unwrap();
            for (u, v) in a.iter().zip(b.iter()) {
                assert_abs_diff_eq!(u, v, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_named_transform_persists_custom_does_not() {
        use anofox_ml_core::persistence::{from_binary_bytes, to_binary_bytes};
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![1.0, 3.0, 7.0, 15.0];
        let inner = crate::bayesian_ridge::BayesianRidge::new();
        let fitted =
            TransformedTargetRegressor::with_transform(inner.clone(), TargetTransform::Log1p)
                .fit(&x, &y)
                .unwrap();
        let back: FittedTransformedTargetRegressor<crate::bayesian_ridge::FittedBayesianRidge> =
            from_binary_bytes(&to_binary_bytes(&fitted).unwrap()).unwrap();
        assert_eq!(back.predict(&x).unwrap(), fitted.predict(&x).unwrap());

        let custom = TransformedTargetRegressor::new(inner, f64::ln_1p, f64::exp_m1)
            .fit(&x, &y)
            .unwrap();
        let err = to_binary_bytes(&custom).unwrap_err().to_string();
        assert!(err.contains("custom functions"), "{err}");
    }
}
