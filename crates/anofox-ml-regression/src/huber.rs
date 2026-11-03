//! Huber robust regression wrapper.
//!
//! Wraps `anofox_regression::HuberRegressor` to provide robust linear regression
//! with the anofox-ml [`Fit`] / [`Predict`] type-state pattern. By default the
//! coefficients and the scale are estimated jointly, as sklearn's
//! `HuberRegressor` ([`HuberScale::Joint`]); [`HuberScale::Mad`] is IRLS at the
//! MAD scale (`MASS::rlm`).

use crate::convert::{col_to_ndarray, ndarray_to_col, ndarray_to_mat};
use anofox_ml_core::{Fit, Predict, Result, RustMlError};
use anofox_regression::solvers::FittedHuber;
use anofox_regression::solvers::HuberRegressor as InnerHuber;
use anofox_regression::solvers::HuberScale as InnerScale;
use anofox_regression::{FittedRegressor as _, Regressor as _};
use ndarray::{Array1, ArrayView1, ArrayView2};

/// How the Huber scale `σ` is estimated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum HuberScale {
    /// `σ` and the coefficients jointly minimise Owen's concomitant-scale
    /// objective `nσ + Σ_in r²/σ + Σ_out (2ε|r| − ε²σ) + α‖β‖²`, as sklearn's
    /// `HuberRegressor` (`scale_`, `outliers_`).
    #[default]
    Joint,
    /// IRLS with `σ` re-estimated by the MAD of the residuals at every step
    /// (`MASS::rlm`).
    Mad,
}

impl HuberScale {
    fn inner(self) -> InnerScale {
        match self {
            HuberScale::Joint => InnerScale::Joint,
            HuberScale::Mad => InnerScale::Mad,
        }
    }
}

/// Huber robust regression estimator.
///
/// Downweights outliers using the Huber loss: quadratic for small residuals,
/// linear for large ones. More robust than OLS while remaining efficient on
/// clean data.
///
/// The `epsilon` parameter (default 1.35) controls the transition point —
/// observations with scaled residuals larger than epsilon are downweighted.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HuberRegressor {
    epsilon: f64,
    alpha: f64,
    with_intercept: bool,
    max_iter: usize,
    tol: f64,
    /// Pre-0.10 models were fitted at the MAD scale.
    #[serde(default = "mad_scale")]
    scale_method: HuberScale,
}

fn mad_scale() -> HuberScale {
    HuberScale::Mad
}

impl HuberRegressor {
    pub fn new() -> Self {
        Self {
            epsilon: 1.35,
            alpha: 0.0001,
            with_intercept: true,
            max_iter: 100,
            tol: 1e-5,
            scale_method: HuberScale::Joint,
        }
    }

    /// Scale estimator. Default: [`HuberScale::Joint`] (sklearn).
    pub fn with_scale_method(mut self, scale_method: HuberScale) -> Self {
        self.scale_method = scale_method;
        self
    }

    /// Set the Huber threshold parameter. Must be > 1.0. Default: 1.35.
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set the L2 regularization strength. Default: 0.0001.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set whether to include an intercept (bias) term. Default: true.
    pub fn with_intercept(mut self, include: bool) -> Self {
        self.with_intercept = include;
        self
    }

    /// Set the maximum number of IRLS iterations. Default: 100.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance. Default: 1e-5.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }
}

impl Default for HuberRegressor {
    fn default() -> Self {
        Self::new()
    }
}

/// A fitted Huber robust regression model.
#[derive(Debug, Clone)]
pub struct FittedHuberRegressor {
    inner: FittedHuber,
    n_features: usize,
}

impl FittedHuberRegressor {
    /// Return the regression coefficients (excluding intercept).
    pub fn coefficients(&self) -> Array1<f64> {
        col_to_ndarray(&self.inner.result().coefficients)
    }

    /// Return the intercept term, if the model was fit with one.
    pub fn intercept(&self) -> Option<f64> {
        self.inner.result().intercept
    }

    /// Return the R-squared statistic.
    pub fn r_squared(&self) -> f64 {
        self.inner.result().r_squared
    }

    /// The estimated scale `σ` (sklearn's `scale_` with
    /// [`HuberScale::Joint`], the final MAD scale with [`HuberScale::Mad`]).
    pub fn scale(&self) -> f64 {
        self.inner.scale()
    }

    /// Return an outlier mask: `true` for observations where |residual| > epsilon * scale.
    pub fn outliers(&self) -> &[bool] {
        self.inner.outliers()
    }

    /// Return the number of detected outliers.
    pub fn n_outliers(&self) -> usize {
        self.inner.n_outliers()
    }

    /// Return the epsilon parameter used during fitting.
    pub fn epsilon(&self) -> f64 {
        self.inner.epsilon()
    }
}

impl Fit<f64> for HuberRegressor {
    type Fitted = FittedHuberRegressor;

    fn fit_view(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<Self::Fitted> {
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
        if self.epsilon <= 1.0 {
            return Err(RustMlError::InvalidParameter(
                "epsilon must be greater than 1.0".into(),
            ));
        }

        let x_mat = ndarray_to_mat(x);
        let y_col = ndarray_to_col(y);

        let inner_model = InnerHuber::builder()
            .epsilon(self.epsilon)
            .alpha(self.alpha)
            .with_intercept(self.with_intercept)
            .max_iterations(self.max_iter)
            .tolerance(self.tol)
            .scale_method(self.scale_method.inner())
            .build();

        let fitted = inner_model
            .fit(&x_mat, &y_col)
            .map_err(|e| RustMlError::InvalidParameter(e.to_string()))?;

        Ok(FittedHuberRegressor {
            inner: fitted,
            n_features: x.ncols(),
        })
    }
}

impl Predict<f64> for FittedHuberRegressor {
    fn predict_view(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array2;

    #[test]
    fn test_huber_clean_data() {
        // y = 2 + 1.5x on clean data — should match OLS closely
        let x = Array2::from_shape_vec((50, 1), (0..50).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_vec((0..50).map(|i| 2.0 + 1.5 * i as f64).collect());

        let fitted = HuberRegressor::new().fit(&x, &y).unwrap();

        assert_abs_diff_eq!(fitted.coefficients()[0], 1.5, epsilon = 0.1);
        assert!(fitted.r_squared() > 0.99);
        assert!(fitted.scale() > 0.0);
    }

    #[test]
    fn test_huber_with_outliers() {
        let mut y_vec: Vec<f64> = (0..50).map(|i| 2.0 + 1.5 * i as f64).collect();
        // Inject outliers
        y_vec[5] += 500.0;
        y_vec[15] += 500.0;
        y_vec[25] += 500.0;

        let x = Array2::from_shape_vec((50, 1), (0..50).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_vec(y_vec);

        let fitted = HuberRegressor::new().fit(&x, &y).unwrap();

        // Should still recover slope reasonably
        assert_abs_diff_eq!(fitted.coefficients()[0], 1.5, epsilon = 2.0);
        // Should detect outliers
        assert!(fitted.n_outliers() >= 3);
        assert_eq!(fitted.outliers().len(), 50);
    }

    #[test]
    fn test_huber_regularization_shrinks() {
        // Noisy data: on an exact fit the joint scale goes to 0 and the
        // penalty has (almost) no effect, as in sklearn.
        let x = Array2::from_shape_vec((50, 1), (1..=50).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_vec(
            (1..=50)
                .map(|i| 1.0 + 3.0 * i as f64 + 20.0 * (i as f64).sin())
                .collect(),
        );

        let low = HuberRegressor::new()
            .with_alpha(0.0001)
            .fit(&x, &y)
            .unwrap();
        let high = HuberRegressor::new().with_alpha(100.0).fit(&x, &y).unwrap();

        assert!(
            high.coefficients()[0].abs() < low.coefficients()[0].abs(),
            "higher alpha should shrink coefficients"
        );
    }

    #[test]
    fn test_huber_predict() {
        let x = Array2::from_shape_vec((50, 1), (0..50).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_vec((0..50).map(|i| 1.0 + 2.0 * i as f64).collect());

        let fitted = HuberRegressor::new().fit(&x, &y).unwrap();

        let x_new = Array2::from_shape_vec((3, 1), vec![100.0, 200.0, 300.0]).unwrap();
        let preds = fitted.predict(&x_new).unwrap();

        assert_abs_diff_eq!(preds[0], 201.0, epsilon = 2.0);
        assert_abs_diff_eq!(preds[1], 401.0, epsilon = 2.0);
    }

    #[test]
    fn test_huber_invalid_epsilon() {
        let x = Array2::from_shape_vec((10, 1), (0..10).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_vec((0..10).map(|i| i as f64).collect());

        assert!(HuberRegressor::new().with_epsilon(0.5).fit(&x, &y).is_err());
    }

    #[test]
    fn test_huber_shape_mismatch() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0]);

        assert!(HuberRegressor::new().fit(&x, &y).is_err());
    }
}

impl anofox_ml_core::RegressorScore<f64> for FittedHuberRegressor {}
