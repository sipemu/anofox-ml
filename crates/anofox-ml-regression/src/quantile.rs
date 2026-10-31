//! Quantile regression wrapper.

use crate::convert::{col_to_ndarray, ndarray_to_col, ndarray_to_mat};
use anofox_ml_core::{Fit, FitWeighted, Predict, Result, RustMlError};
use anofox_regression::{FittedQuantile, QuantileRegressor as InnerQuantile};
use anofox_regression::{FittedRegressor as _, Regressor as _};
use ndarray::{Array1, Array2};

/// Quantile regression estimator.
///
/// Estimates conditional quantiles of the response variable.
/// Set `quantile = 0.5` for median regression. Unpenalised (sklearn's
/// `QuantileRegressor(alpha=0)`); [`FitWeighted`] minimises
/// `Σ wᵢ ρ_τ(yᵢ - xᵢᵀβ)`.
#[derive(Debug, Clone)]
pub struct QuantileRegressor {
    quantile: f64,
    with_intercept: bool,
}

impl QuantileRegressor {
    pub fn new(quantile: f64) -> Self {
        Self {
            quantile,
            with_intercept: true,
        }
    }

    pub fn with_intercept(mut self, include: bool) -> Self {
        self.with_intercept = include;
        self
    }
}

/// A fitted quantile regression model.
#[derive(Debug, Clone)]
pub struct FittedQuantileRegressor {
    inner: FittedQuantile,
    n_features: usize,
    coef: Array1<f64>,
    intercept: f64,
}

impl FittedQuantileRegressor {
    pub fn coefficients(&self) -> Array1<f64> {
        col_to_ndarray(self.inner.coefficients())
    }

    pub fn intercept(&self) -> Option<f64> {
        self.inner.intercept()
    }

    pub fn r_squared(&self) -> f64 {
        self.inner.r_squared()
    }
}

impl Fit<f64> for QuantileRegressor {
    type Fitted = FittedQuantileRegressor;

    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        self.fit_weighted(x, y, None)
    }
}

impl FitWeighted<f64> for QuantileRegressor {
    type Fitted = FittedQuantileRegressor;

    fn fit_weighted(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        sample_weight: Option<&Array1<f64>>,
    ) -> Result<Self::Fitted> {
        crate::convert::check_xy(x, y)?;
        if self.quantile <= 0.0 || self.quantile >= 1.0 {
            return Err(RustMlError::InvalidParameter(
                "quantile must be between 0 and 1 (exclusive)".into(),
            ));
        }

        let x_mat = ndarray_to_mat(x);
        let y_col = ndarray_to_col(y);

        let mut builder = InnerQuantile::builder()
            .tau(self.quantile)
            .with_intercept(self.with_intercept);
        if let Some(w) = sample_weight {
            let w = crate::convert::resolve_sample_weight(Some(w), y.len())?;
            builder = builder.weights(ndarray_to_col(&w));
        }

        let fitted = builder
            .build()
            .fit(&x_mat, &y_col)
            .map_err(|e| RustMlError::InvalidParameter(e.to_string()))?;

        let (coef, intercept) = crate::convert::predict_params(fitted.result());
        Ok(FittedQuantileRegressor {
            inner: fitted,
            n_features: x.ncols(),
            coef,
            intercept,
        })
    }
}

impl Predict<f64> for FittedQuantileRegressor {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }

        crate::convert::linear_predict(x, &self.coef, self.intercept)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_quantile_median() {
        // Simple linear data — median should match mean for symmetric errors
        let x = Array2::from_shape_vec((10, 1), (0..10).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_vec((0..10).map(|i| 2.0 + 3.0 * i as f64).collect());

        let fitted = QuantileRegressor::new(0.5).fit(&x, &y).unwrap();
        assert!(fitted.r_squared() > 0.99);
    }

    #[test]
    fn test_quantile_weighted_matches_sklearn() {
        // sklearn 1.8 QuantileRegressor(alpha=0, solver='highs'), exact LP.
        let xs = [
            4.17, 7.2, 0.0, 3.02, 1.47, 0.92, 1.86, 3.46, 3.97, 5.39, 4.19, 6.85, 2.04, 8.78, 0.27,
            6.7, 4.17, 5.59, 1.4, 1.98, 8.01, 9.68, 3.13, 6.92, 8.76, 8.95, 0.85, 0.39, 1.7, 8.78,
            0.98, 4.21, 9.58, 5.33, 6.92, 3.16, 6.87, 8.35, 0.18, 7.5,
        ];
        let y = array![
            4.543, 5.644, 0.046, 1.6, 0.969, 2.039, 1.578, 2.036, 3.137, 3.878, 2.85, 3.668, 4.172,
            4.611, 0.997, 4.496, 1.713, 3.586, 0.001, 1.674, 4.728, 6.028, 2.67, 4.17, 5.541,
            3.313, 1.462, 0.629, 2.605, 4.951, 1.526, 3.484, 4.981, 3.75, 2.314, 2.936, 3.776,
            7.233, 1.659, 3.593
        ];
        let w = array![
            1.99, 1.67, 2.79, 0.94, 0.38, 2.26, 2.36, 2.74, 2.81, 0.24, 0.86, 1.93, 2.86, 2.86,
            1.76, 2.76, 2.0, 1.29, 1.56, 1.89, 1.74, 2.79, 2.77, 1.31, 2.9, 0.69, 0.55, 0.58, 1.62,
            0.26, 2.85, 2.52, 0.24, 0.69, 1.13, 0.57, 2.47, 1.17, 2.83, 1.83
        ];
        let x = Array2::from_shape_vec((40, 1), xs.to_vec()).unwrap();
        let cases = [
            (0.5, false, 1.0827907742998353, 0.44612850082372324),
            (0.5, true, 1.079633001422475, 0.4554765291607397),
            (0.8, false, 1.5774863523573202, 0.45285359801488834),
            (0.8, true, 1.5762189473684212, 0.45989473684210525),
        ];
        for (tau, weighted, b, slope) in cases {
            let sw = if weighted { Some(&w) } else { None };
            let f = QuantileRegressor::new(tau)
                .fit_weighted(&x, &y, sw)
                .unwrap();
            // The upstream solver is a smoothed IRLS, so match the LP to ~1e-3.
            assert!(
                (f.coefficients()[0] - slope).abs() < 2e-3,
                "tau={tau} w={weighted}: slope {} vs {slope}",
                f.coefficients()[0]
            );
            assert!(
                (f.intercept().unwrap() - b).abs() < 1e-2,
                "tau={tau} w={weighted}: intercept {} vs {b}",
                f.intercept().unwrap()
            );
        }
    }

    #[test]
    fn test_quantile_invalid_tau() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        assert!(QuantileRegressor::new(0.0).fit(&x, &y).is_err());
        assert!(QuantileRegressor::new(1.0).fit(&x, &y).is_err());
        assert!(QuantileRegressor::new(-0.1).fit(&x, &y).is_err());
    }
}

impl anofox_ml_core::RegressorScore<f64> for FittedQuantileRegressor {}
