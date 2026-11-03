//! B-spline expansion of each feature (`sklearn.preprocessing.SplineTransformer`).
//!
//! A transformer wrapper around [`anofox_regression::splines::BSpline`]: every
//! feature becomes `n_knots + degree − 1` B-spline columns (`n_knots − 1`
//! when periodic, one fewer without the bias column), in sklearn's order
//! (all splines of feature 0, then feature 1, …). Knots are placed uniformly
//! between the feature's minimum and maximum or at its quantiles, or given
//! explicitly.

use anofox_ml_core::{FitUnsupervised, Float, Result, RustMlError, Transform};
use anofox_regression::splines::{BSpline, KnotPlacement, SplineExtrapolation};
use ndarray::{Array2, ArrayView2};

/// Placement of the knots computed from data (sklearn `knots`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum SplineKnots {
    /// Equally spaced from the minimum to the maximum (`"uniform"`).
    #[default]
    Uniform,
    /// At equally spaced quantiles (`"quantile"`).
    Quantile,
}

/// Behaviour outside the outer knots (sklearn `extrapolation`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum Extrapolation {
    /// Boundary values of the splines non-zero there (`"constant"`).
    #[default]
    Constant,
    /// Linear continuation of the boundary splines (`"linear"`).
    Linear,
    /// The first / last polynomial pieces, continued (`"continue"`).
    Continue,
    /// Periodic splines with period `last knot − first knot` (`"periodic"`).
    Periodic,
    /// Values outside the knots are an error (`"error"`).
    Error,
}

impl From<Extrapolation> for SplineExtrapolation {
    fn from(e: Extrapolation) -> Self {
        match e {
            Extrapolation::Constant => SplineExtrapolation::Constant,
            Extrapolation::Linear => SplineExtrapolation::Linear,
            Extrapolation::Continue => SplineExtrapolation::Continue,
            Extrapolation::Periodic => SplineExtrapolation::Periodic,
            Extrapolation::Error => SplineExtrapolation::Error,
        }
    }
}

/// Parameters for `SplineTransformer` (unfitted state).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SplineTransformer {
    /// Number of knots per feature when `knots` is not given (at least 2).
    pub n_knots: usize,
    /// Polynomial degree of the splines.
    pub degree: usize,
    /// Knot placement from data.
    #[serde(default)]
    pub knot_placement: SplineKnots,
    /// Explicit knots, one strictly increasing list per feature.
    #[serde(default)]
    pub knots: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    pub extrapolation: Extrapolation,
    /// Keep the last spline of each feature (sklearn `include_bias`).
    pub include_bias: bool,
}

impl Default for SplineTransformer {
    /// sklearn's defaults: 5 uniform knots, cubic, constant extrapolation,
    /// with the bias column.
    fn default() -> Self {
        Self {
            n_knots: 5,
            degree: 3,
            knot_placement: SplineKnots::Uniform,
            knots: None,
            extrapolation: Extrapolation::Constant,
            include_bias: true,
        }
    }
}

impl SplineTransformer {
    /// `n_knots` knots per feature, splines of degree `degree`.
    pub fn new(n_knots: usize, degree: usize) -> Self {
        Self {
            n_knots,
            degree,
            ..Self::default()
        }
    }

    pub fn with_knot_placement(mut self, placement: SplineKnots) -> Self {
        self.knot_placement = placement;
        self
    }

    /// Use the given knots (one list per feature).
    pub fn with_knots(mut self, knots: Vec<Vec<f64>>) -> Self {
        self.knots = Some(knots);
        self
    }

    pub fn with_extrapolation(mut self, extrapolation: Extrapolation) -> Self {
        self.extrapolation = extrapolation;
        self
    }

    pub fn with_include_bias(mut self, include_bias: bool) -> Self {
        self.include_bias = include_bias;
        self
    }
}

/// Fitted `SplineTransformer`: the base knots of every feature.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedSplineTransformer<F: Float> {
    knots: Vec<Vec<f64>>,
    degree: usize,
    extrapolation: Extrapolation,
    include_bias: bool,
    #[serde(skip)]
    _marker: std::marker::PhantomData<F>,
}

fn spline_err(j: usize, e: impl std::fmt::Display) -> RustMlError {
    RustMlError::InvalidParameter(format!("feature {j}: {e}"))
}

impl<F: Float> FitUnsupervised<F> for SplineTransformer {
    type Fitted = FittedSplineTransformer<F>;

    fn fit_view(&self, x: ArrayView2<F>) -> Result<Self::Fitted> {
        if x.is_empty() {
            return Err(RustMlError::EmptyInput("input array is empty".into()));
        }
        let d = x.ncols();
        let placement = match self.knot_placement {
            SplineKnots::Uniform => KnotPlacement::Uniform,
            SplineKnots::Quantile => KnotPlacement::Quantile,
        };
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
                    BSpline::new(kj, self.degree)
                        .map(|s| s.base_knots().to_vec())
                        .map_err(|e| spline_err(j, e))
                })
                .collect::<Result<Vec<_>>>()?,
            None => (0..d)
                .map(|j| {
                    let col: Vec<f64> = x.column(j).iter().map(|v| v.to_f64().unwrap()).collect();
                    BSpline::from_data(&col, self.n_knots, self.degree, placement)
                        .map(|s| s.base_knots().to_vec())
                        .map_err(|e| spline_err(j, e))
                })
                .collect::<Result<Vec<_>>>()?,
        };
        let fitted = FittedSplineTransformer {
            knots,
            degree: self.degree,
            extrapolation: self.extrapolation,
            include_bias: self.include_bias,
            _marker: std::marker::PhantomData,
        };
        // Surface invalid combinations (periodic with degree >= n_knots) at fit.
        for j in 0..d {
            fitted
                .spline(j)
                .basis(&[], "x")
                .map_err(|e| spline_err(j, e))?;
        }
        Ok(fitted)
    }
}

impl<F: Float> FittedSplineTransformer<F> {
    fn spline(&self, j: usize) -> BSpline {
        BSpline::new(&self.knots[j], self.degree)
            .expect("knots validated at fit")
            .with_extrapolation(self.extrapolation.into())
            .with_bias(self.include_bias)
    }

    /// Base knots of each feature (sklearn's `bsplines_[j].t` without the
    /// `degree` added knots on each side).
    pub fn knots(&self) -> &[Vec<f64>] {
        &self.knots
    }

    /// Number of output columns (sklearn's `n_features_out_`).
    pub fn n_output_features(&self) -> usize {
        (0..self.knots.len())
            .map(|j| self.spline(j).n_columns())
            .sum()
    }

    /// Output column names, `x0_sp_0, x0_sp_1, …` (sklearn's
    /// `get_feature_names_out`), or built from `input_names`.
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
        let mut names = Vec::new();
        for j in 0..d {
            let name = input_names.map_or_else(|| format!("x{j}"), |n| n[j].clone());
            names.extend(
                self.spline(j)
                    .basis(&[], &name)
                    .map_err(|e| spline_err(j, e))?
                    .names,
            );
        }
        Ok(names)
    }
}

impl<F: Float> Transform<F> for FittedSplineTransformer<F> {
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
            let basis = self
                .spline(j)
                .basis(&col, "x")
                .map_err(|e| spline_err(j, e))?;
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
