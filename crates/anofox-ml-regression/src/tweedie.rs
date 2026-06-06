//! Tweedie / Gamma GLM regressors.
//!
//! Mirrors `sklearn.linear_model.{TweedieRegressor, GammaRegressor}` by
//! wrapping `anofox_regression::TweedieRegressor`. sklearn's
//! `power` is the variance power; we expose it directly. Log link by default.

use crate::convert::{col_to_ndarray, ndarray_to_col, ndarray_to_mat};
use anofox_ml_core::{Fit, Predict, Result, RustMlError};
use anofox_regression::solvers::{FittedTweedie, TweedieRegressor as InnerTweedie};
use anofox_regression::{FittedRegressor as _, Regressor as _};
use ndarray::{Array1, Array2};

/// Tweedie GLM regressor.
///
/// `power=0` → Gaussian, `power=1` → Poisson, `power=2` → Gamma,
/// `power=3` → Inverse Gaussian.
#[derive(Debug, Clone)]
pub struct TweedieRegressor {
    power: f64,
    alpha: f64,
    link_power: f64,
    with_intercept: bool,
    max_iter: usize,
    tol: f64,
}

impl TweedieRegressor {
    pub fn new(power: f64) -> Self {
        Self {
            power,
            alpha: 0.0,
            link_power: if power == 0.0 { 1.0 } else { 0.0 }, // identity for Gaussian, log otherwise
            with_intercept: true,
            max_iter: 100,
            tol: 1e-4,
        }
    }

    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn with_link_power(mut self, link_power: f64) -> Self {
        self.link_power = link_power;
        self
    }

    pub fn with_intercept(mut self, include: bool) -> Self {
        self.with_intercept = include;
        self
    }

    pub fn with_max_iter(mut self, m: usize) -> Self {
        self.max_iter = m;
        self
    }
}

/// Convenience: Gamma regression (power=2, log link).
pub fn gamma_regressor() -> TweedieRegressor {
    TweedieRegressor::new(2.0)
}

/// Fitted Tweedie GLM regressor. Not yet serde-serialisable because the
/// wrapped `anofox_regression::FittedTweedie` does not derive Serialize.
#[derive(Debug, Clone)]
pub struct FittedTweedieRegressor {
    inner: FittedTweedie,
    n_features: usize,
}

impl FittedTweedieRegressor {
    pub fn coefficients(&self) -> Array1<f64> {
        col_to_ndarray(self.inner.coefficients())
    }
    pub fn intercept(&self) -> Option<f64> {
        self.inner.intercept()
    }
}

impl Fit<f64> for TweedieRegressor {
    type Fitted = FittedTweedieRegressor;

    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        if x.nrows() != y.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but y has {} elements",
                x.nrows(),
                y.len()
            )));
        }
        if x.is_empty() {
            return Err(RustMlError::EmptyInput("training data is empty".into()));
        }
        if self.alpha < 0.0 {
            return Err(RustMlError::InvalidParameter(
                "alpha must be non-negative".into(),
            ));
        }

        let x_mat = ndarray_to_mat(x);
        let y_col = ndarray_to_col(y);

        let inner_model = InnerTweedie::builder()
            .var_power(self.power)
            .link_power(self.link_power)
            .lambda(self.alpha)
            .with_intercept(self.with_intercept)
            .max_iterations(self.max_iter)
            .tolerance(self.tol)
            .build();

        let fitted = inner_model
            .fit(&x_mat, &y_col)
            .map_err(|e| RustMlError::InvalidParameter(e.to_string()))?;

        Ok(FittedTweedieRegressor {
            inner: fitted,
            n_features: x.ncols(),
        })
    }
}

impl Predict<f64> for FittedTweedieRegressor {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        let x_mat = ndarray_to_mat(x);
        let preds = self.inner.predict(&x_mat);
        Ok(col_to_ndarray(&preds))
    }
}

// ---------------------------------------------------------------------------
// GammaRegressor — convenience wrapper matching sklearn's class.
// ---------------------------------------------------------------------------

/// Gamma GLM regressor (Tweedie with power=2 and log link).
#[derive(Debug, Clone)]
pub struct GammaRegressor {
    inner: TweedieRegressor,
}

impl GammaRegressor {
    pub fn new() -> Self {
        Self {
            inner: TweedieRegressor::new(2.0),
        }
    }
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.inner = self.inner.with_alpha(alpha);
        self
    }
    pub fn with_intercept(mut self, include: bool) -> Self {
        self.inner = self.inner.with_intercept(include);
        self
    }
}

impl Default for GammaRegressor {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Gamma GLM regressor.
pub type FittedGammaRegressor = FittedTweedieRegressor;

impl Fit<f64> for GammaRegressor {
    type Fitted = FittedTweedieRegressor;
    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        self.inner.fit(x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_gamma_predictions_positive() {
        // Positive-target dataset for Gamma regression (log link → exp output).
        let x =
            Array2::from_shape_vec((8, 1), vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]).unwrap();
        let y = array![1.0, 1.5, 2.0, 2.5, 3.5, 5.0, 7.0, 12.0];

        let fitted = GammaRegressor::new().fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(p > 0.0, "Gamma predictions must be positive, got {p}");
        }
    }

    #[test]
    fn test_tweedie_power_1p5() {
        // Compound Poisson-Gamma (typical for insurance frequency-severity).
        let x = Array2::from_shape_vec((10, 1), (0..10).map(|i| i as f64).collect()).unwrap();
        let y = array![0.0, 0.1, 0.5, 0.0, 1.2, 0.0, 2.0, 3.5, 0.0, 4.0];
        let fitted = TweedieRegressor::new(1.5).fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(p >= 0.0);
        }
    }
}

impl anofox_ml_core::RegressorScore<f64> for FittedTweedieRegressor {}
