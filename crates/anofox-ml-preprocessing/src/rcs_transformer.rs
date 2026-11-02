//! Restricted cubic spline expansion of each feature.
//!
//! A transformer wrapper around [`anofox_regression::splines::Rcs`]
//! (Harrell's restricted / natural cubic spline, `rms::rcs` in R): with `k`
//! knots a feature becomes `k − 1` columns `x, x', x'', …` that are cubic
//! between the knots and linear beyond the outer ones. Default knots are
//! placed at the quantiles of RMS Table 2.3 (`rcs_default_knots`). sklearn
//! has no counterpart; its `SplineTransformer` builds a B-spline basis.
//!
//! The basis itself comes from anofox-regression, so the columns agree with
//! the statistical models fitted there.

use anofox_ml_core::{FitUnsupervised, Float, Result, RustMlError, Transform};
use anofox_regression::splines::{Rcs, RcsNorm};
use ndarray::{Array2, ArrayView2};

/// Normalisation of the nonlinear terms (`norm` of `Hmisc::rcspline.eval`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum SplineNorm {
    /// No normalisation (`norm = 0`).
    None,
    /// Divide by `(t_k − t_{k−1})³` (`norm = 1`).
    LastInterval,
    /// Divide by `(t_k − t₁)²` (`norm = 2`, the `rms::rcs` default), which
    /// puts the terms on the scale of `x`.
    #[default]
    Range,
}

impl From<SplineNorm> for RcsNorm {
    fn from(n: SplineNorm) -> Self {
        match n {
            SplineNorm::None => RcsNorm::None,
            SplineNorm::LastInterval => RcsNorm::LastInterval,
            SplineNorm::Range => RcsNorm::Range,
        }
    }
}

/// Parameters for `RcsTransformer` (unfitted state).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RcsTransformer {
    /// Number of knots per feature when `knots` is not given (at least 3).
    pub n_knots: usize,
    /// Explicit knots, one list per feature.
    #[serde(default)]
    pub knots: Option<Vec<Vec<f64>>>,
    /// Normalisation of the nonlinear terms.
    #[serde(default)]
    pub norm: SplineNorm,
}

impl Default for RcsTransformer {
    fn default() -> Self {
        Self {
            n_knots: 5,
            knots: None,
            norm: SplineNorm::Range,
        }
    }
}

impl RcsTransformer {
    /// `n_knots` default knots per feature.
    pub fn new(n_knots: usize) -> Self {
        Self {
            n_knots,
            ..Self::default()
        }
    }

    /// Use the given knots (one list per feature).
    pub fn with_knots(mut self, knots: Vec<Vec<f64>>) -> Self {
        self.knots = Some(knots);
        self
    }

    /// Set the normalisation of the nonlinear terms.
    pub fn with_norm(mut self, norm: SplineNorm) -> Self {
        self.norm = norm;
        self
    }
}

/// Fitted `RcsTransformer`: the knots of every feature.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedRcsTransformer<F: Float> {
    knots: Vec<Vec<f64>>,
    norm: SplineNorm,
    #[serde(skip)]
    _marker: std::marker::PhantomData<F>,
}

fn spline_err(j: usize, e: impl std::fmt::Display) -> RustMlError {
    RustMlError::InvalidParameter(format!("feature {j}: {e}"))
}

impl<F: Float> FitUnsupervised<F> for RcsTransformer {
    type Fitted = FittedRcsTransformer<F>;

    fn fit_view(&self, x: ArrayView2<F>) -> Result<Self::Fitted> {
        if x.is_empty() {
            return Err(RustMlError::EmptyInput("input array is empty".into()));
        }
        let d = x.ncols();
        let knots = match &self.knots {
            Some(k) if k.len() != d => {
                return Err(RustMlError::ShapeMismatch(format!(
                    "{} knot lists for {d} features",
                    k.len()
                )))
            }
            Some(k) => k
                .iter()
                .enumerate()
                .map(|(j, kj)| {
                    Rcs::new(kj)
                        .map(|r| r.knots().to_vec())
                        .map_err(|e| spline_err(j, e))
                })
                .collect::<Result<Vec<_>>>()?,
            None => (0..d)
                .map(|j| {
                    let col: Vec<f64> = x.column(j).iter().map(|v| v.to_f64().unwrap()).collect();
                    Rcs::from_data(&col, self.n_knots)
                        .map(|r| r.knots().to_vec())
                        .map_err(|e| spline_err(j, e))
                })
                .collect::<Result<Vec<_>>>()?,
        };
        Ok(FittedRcsTransformer {
            knots,
            norm: self.norm,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<F: Float> FittedRcsTransformer<F> {
    fn spline(&self, j: usize) -> Rcs {
        Rcs::new(&self.knots[j])
            .expect("knots validated at fit")
            .with_norm(self.norm.into())
    }

    /// Knots of each feature.
    pub fn knots(&self) -> &[Vec<f64>] {
        &self.knots
    }

    /// Number of output columns (`Σ (k_j − 1)`).
    pub fn n_output_features(&self) -> usize {
        self.knots.iter().map(|k| k.len() - 1).sum()
    }

    /// Output column names, rms style: `x0, x0', x0'', x1, …`, or built from
    /// `input_names`.
    pub fn feature_names_out(&self, input_names: Option<&[String]>) -> Result<Vec<String>> {
        let d = self.knots.len();
        if let Some(n) = input_names {
            if n.len() != d {
                return Err(RustMlError::ShapeMismatch(format!(
                    "expected {d} input names, got {}",
                    n.len()
                )));
            }
        }
        Ok((0..d)
            .flat_map(|j| {
                let name = input_names.map_or_else(|| format!("x{j}"), |n| n[j].clone());
                self.spline(j).basis(&[], &name).names
            })
            .collect())
    }
}

impl<F: Float> Transform<F> for FittedRcsTransformer<F> {
    /// Missing (non-finite) inputs give `NaN` in all columns of that feature.
    fn transform_view(&self, x: ArrayView2<F>) -> Result<Array2<F>> {
        let d = self.knots.len();
        if x.ncols() != d {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {d} features, got {}",
                x.ncols()
            )));
        }
        let mut out = Array2::<F>::zeros((x.nrows(), self.n_output_features()));
        let mut c0 = 0;
        for j in 0..d {
            let col: Vec<f64> = x.column(j).iter().map(|v| v.to_f64().unwrap()).collect();
            let basis = self.spline(j).basis(&col, "x");
            let m = basis.n_columns();
            for i in 0..x.nrows() {
                for c in 0..m {
                    out[[i, c0 + c]] = F::from_f64(basis.columns[(i, c)]).unwrap();
                }
            }
            c0 += m;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anofox_regression::splines::rcs_default_knots;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array1};

    fn data() -> Array2<f64> {
        Array2::from_shape_fn((40, 2), |(i, j)| {
            let t = i as f64;
            if j == 0 {
                (t * 0.37).sin() * 3.0 + t * 0.1
            } else {
                (t * 0.11).exp()
            }
        })
    }

    #[test]
    fn test_matches_anofox_regression_rcs() {
        let x = data();
        let f = FitUnsupervised::<f64>::fit(&RcsTransformer::new(4), &x).unwrap();
        let t = f.transform(&x).unwrap();
        assert_eq!(t.ncols(), 6);
        let mut c0 = 0;
        for j in 0..2 {
            let col: Vec<f64> = x.column(j).to_vec();
            assert_eq!(f.knots()[j], rcs_default_knots(&col, 4).unwrap());
            let b = Rcs::from_data(&col, 4).unwrap().basis(&col, "x");
            for i in 0..x.nrows() {
                for c in 0..3 {
                    assert_abs_diff_eq!(t[[i, c0 + c]], b.columns[(i, c)], epsilon = 1e-15);
                }
            }
            c0 += 3;
        }
        assert_eq!(
            f.feature_names_out(None).unwrap(),
            ["x0", "x0'", "x0''", "x1", "x1'", "x1''"]
        );
    }

    #[test]
    fn test_explicit_knots_linear_tails_and_nan() {
        // With knots [0, 1, 2] the single nonlinear term is
        // ((x)+^3 - 2 (x-1)+^3 + (x-2)+^3) / 4 (norm = Range: kd^3 = 4), which is
        // linear beyond the last knot.
        let x = array![[-1.0], [0.5], [1.5], [3.0], [4.0], [5.0], [f64::NAN]];
        let rcs = RcsTransformer::default().with_knots(vec![vec![0.0, 1.0, 2.0]]);
        let f = FitUnsupervised::<f64>::fit(&rcs, &x).unwrap();
        let t = f.transform(&x).unwrap();
        let term = |v: f64| {
            let c = |a: f64| (v - a).max(0.0).powi(3);
            (c(0.0) - 2.0 * c(1.0) + c(2.0)) / 4.0
        };
        let expected: Array1<f64> = x.column(0).iter().take(6).map(|&v| term(v)).collect();
        for i in 0..6 {
            assert_abs_diff_eq!(t[[i, 0]], x[[i, 0]], epsilon = 1e-15);
            assert_abs_diff_eq!(t[[i, 1]], expected[i], epsilon = 1e-12);
        }
        // Linear beyond the outer knot: equal steps for equal increments.
        assert_abs_diff_eq!(
            t[[5, 1]] - t[[4, 1]],
            t[[4, 1]] - t[[3, 1]],
            epsilon = 1e-12
        );
        assert!(t[[6, 0]].is_nan() && t[[6, 1]].is_nan());
    }

    #[test]
    fn test_errors() {
        let x = data();
        assert!(FitUnsupervised::<f64>::fit(&RcsTransformer::new(2), &x).is_err());
        let r = RcsTransformer::default().with_knots(vec![vec![0.0, 1.0, 2.0]]);
        assert!(FitUnsupervised::<f64>::fit(&r, &x).is_err());
    }
}
