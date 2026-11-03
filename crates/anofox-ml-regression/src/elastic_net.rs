//! Elastic Net regression wrapper (combined L1 and L2 regularization).

use crate::convert::{col_to_ndarray, ndarray_to_col, ndarray_to_mat};
use anofox_ml_core::{Fit, Predict, Result, RustMlError};
use anofox_regression::solvers::{ElasticNetRegressor as InnerElasticNet, FittedElasticNet};
use anofox_regression::{FittedRegressor as _, Regressor as _};
use ndarray::{Array1, Array2};

/// Elastic Net regression estimator.
///
/// Minimizes: `||y - Xβ||² + λ(α||β||₁ + (1-α)||β||₂²)`
///
/// - `alpha = 1.0` is pure Lasso (L1)
/// - `alpha = 0.0` is pure Ridge (L2)
/// - `0 < alpha < 1` is a mix of both
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ElasticNetRegressor {
    lambda: f64,
    alpha: f64,
    with_intercept: bool,
    /// `lambda` is sklearn's `alpha` (penalty scale of `(1/2n)·RSS`),
    /// converted at fit time; otherwise upstream's raw `lambda`.
    #[serde(default)]
    sklearn_scale: bool,
}

impl ElasticNetRegressor {
    pub fn new() -> Self {
        Self {
            lambda: 1.0,
            alpha: 0.5,
            with_intercept: true,
            sklearn_scale: false,
        }
    }

    pub fn with_lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda;
        self
    }

    /// Use sklearn's penalty scale: `alpha` as in sklearn's
    /// `(1/2n)·||y − Xw||² + alpha·(l1_ratio·||w||₁ + (1 − l1_ratio)/2·||w||²)`
    /// (the mixing set with `with_alpha` is sklearn's `l1_ratio`). It is
    /// translated to upstream's raw objective at fit time.
    pub fn with_sklearn_alpha(mut self, alpha: f64) -> Self {
        self.lambda = alpha;
        self.sklearn_scale = true;
        self
    }

    /// Set the L1 ratio (alpha). 1.0 = pure Lasso, 0.0 = pure Ridge.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn with_intercept(mut self, include: bool) -> Self {
        self.with_intercept = include;
        self
    }
}

impl Default for ElasticNetRegressor {
    fn default() -> Self {
        Self::new()
    }
}

/// A fitted Elastic Net regression model.
#[derive(Debug, Clone)]
pub struct FittedElasticNetRegressor {
    inner: FittedElasticNet,
    n_features: usize,
    coef: Array1<f64>,
    intercept: f64,
}

impl FittedElasticNetRegressor {
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

impl Fit<f64> for ElasticNetRegressor {
    type Fitted = FittedElasticNetRegressor;

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
        if self.lambda < 0.0 {
            return Err(RustMlError::InvalidParameter(
                "lambda must be non-negative".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.alpha) {
            return Err(RustMlError::InvalidParameter(
                "alpha must be between 0.0 and 1.0".into(),
            ));
        }

        let x_mat = ndarray_to_mat(x);
        let y_col = ndarray_to_col(y);

        // sklearn → raw objective RSS + λ(a·|w|₁ + (1 − a)·||w||²):
        // λ = n·alpha·(1 + r), a = 2r / (1 + r) for sklearn's l1_ratio r.
        let (lambda, mix) = if self.sklearn_scale {
            let r = self.alpha;
            let n = x.nrows() as f64;
            (n * self.lambda * (1.0 + r), 2.0 * r / (1.0 + r))
        } else {
            (self.lambda, self.alpha)
        };
        let mut builder = InnerElasticNet::builder()
            .with_intercept(self.with_intercept)
            .lambda(lambda)
            .alpha(mix);
        if self.sklearn_scale {
            // sklearn is compared with a tight tolerance; upstream's
            // default stops earlier.
            builder = builder.tolerance(1e-14);
        }
        let inner_model = builder.build();

        let fitted = inner_model
            .fit(&x_mat, &y_col)
            .map_err(|e| RustMlError::InvalidParameter(e.to_string()))?;

        let (coef, intercept) = crate::convert::predict_params(fitted.result());
        Ok(FittedElasticNetRegressor {
            inner: fitted,
            n_features: x.ncols(),
            coef,
            intercept,
        })
    }
}

impl Predict<f64> for FittedElasticNetRegressor {
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
    fn test_elastic_net_basic() {
        let x = Array2::from_shape_vec((10, 1), (0..10).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_vec((0..10).map(|i| 2.0 + 3.0 * i as f64).collect());

        let fitted = ElasticNetRegressor::new()
            .with_lambda(0.01)
            .with_alpha(0.5)
            .fit(&x, &y)
            .unwrap();

        assert!(fitted.r_squared() > 0.99);
    }

    #[test]
    fn test_elastic_net_invalid_alpha() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        assert!(ElasticNetRegressor::new()
            .with_alpha(1.5)
            .fit(&x, &y)
            .is_err());
        assert!(ElasticNetRegressor::new()
            .with_alpha(-0.1)
            .fit(&x, &y)
            .is_err());
    }
}

impl anofox_ml_core::RegressorScore<f64> for FittedElasticNetRegressor {}
