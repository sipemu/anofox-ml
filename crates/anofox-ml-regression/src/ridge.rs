//! Ridge (L2-regularized) regression wrapper.

use crate::convert::{col_to_ndarray, ndarray_to_col, ndarray_to_mat};
use anofox_ml_core::{Fit, FitWeighted, Predict, Result, RustMlError};
use anofox_regression::solvers::{FittedRidge, RidgeRegressor as InnerRidge};
use anofox_regression::{FittedRegressor as _, Regressor as _};
use faer::linalg::solvers::Solve;
use faer::{Mat, Side};
use ndarray::{Array1, Array2};

/// Ridge regression estimator with L2 regularization.
///
/// Minimizes: `||y - Xβ||² + λ||β||²`
#[derive(Debug, Clone)]
pub struct RidgeRegressor {
    lambda: f64,
    with_intercept: bool,
}

impl RidgeRegressor {
    pub fn new() -> Self {
        Self {
            lambda: 1.0,
            with_intercept: true,
        }
    }

    pub fn with_lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda;
        self
    }

    pub fn with_intercept(mut self, include: bool) -> Self {
        self.with_intercept = include;
        self
    }
}

impl Default for RidgeRegressor {
    fn default() -> Self {
        Self::new()
    }
}

/// A fitted Ridge regression model.
#[derive(Debug, Clone)]
pub struct FittedRidgeRegressor {
    inner: FittedRidge,
    n_features: usize,
}

impl FittedRidgeRegressor {
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

impl Fit<f64> for RidgeRegressor {
    type Fitted = FittedRidgeRegressor;

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

        let x_mat = ndarray_to_mat(x);
        let y_col = ndarray_to_col(y);

        let inner_model = InnerRidge::builder()
            .with_intercept(self.with_intercept)
            .lambda(self.lambda)
            .build();

        let fitted = inner_model
            .fit(&x_mat, &y_col)
            .map_err(|e| RustMlError::InvalidParameter(e.to_string()))?;

        Ok(FittedRidgeRegressor {
            inner: fitted,
            n_features: x.ncols(),
        })
    }
}

/// Weighted Ridge result: closed-form `(XᵀWX + λI) β = XᵀWy` with optional
/// intercept absorbed as an extra constant column. Lives outside the anofox
/// wrapper because anofox does not currently support `sample_weight`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedWeightedRidgeRegressor {
    pub coef: Array1<f64>,
    pub intercept: f64,
    n_features: usize,
}

impl Predict<f64> for FittedWeightedRidgeRegressor {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        Ok(x.dot(&self.coef).mapv(|v| v + self.intercept))
    }
}

impl FitWeighted<f64> for RidgeRegressor {
    type Fitted = FittedWeightedRidgeRegressor;

    fn fit_weighted(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        sample_weight: Option<&Array1<f64>>,
    ) -> Result<Self::Fitted> {
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
        if let Some(w) = sample_weight {
            if w.len() != y.len() {
                return Err(RustMlError::ShapeMismatch(format!(
                    "sample_weight len {} != y len {}",
                    w.len(),
                    y.len()
                )));
            }
            for &wi in w.iter() {
                if !wi.is_finite() || wi < 0.0 {
                    return Err(RustMlError::InvalidParameter(
                        "sample_weight must be non-negative finite".into(),
                    ));
                }
            }
        }

        let n = x.nrows();
        let d = x.ncols();
        let ext = if self.with_intercept { d + 1 } else { d };

        // Build design matrix with optional intercept column at the end.
        let xb = |i: usize, j: usize| -> f64 {
            if j < d {
                x[[i, j]]
            } else {
                1.0
            }
        };
        let w = |i: usize| -> f64 { sample_weight.map(|s| s[i]).unwrap_or(1.0) };

        // XᵀWX (ext × ext).
        let mut xtwx = Array2::<f64>::zeros((ext, ext));
        for a in 0..ext {
            for b in 0..ext {
                let mut s = 0.0;
                for i in 0..n {
                    s += xb(i, a) * w(i) * xb(i, b);
                }
                xtwx[[a, b]] = s;
            }
        }
        // Add λ I (skip intercept column — sklearn convention).
        for a in 0..d {
            xtwx[[a, a]] += self.lambda;
        }
        // Always add tiny ridge for numerical safety on otherwise-singular X.
        for a in 0..ext {
            xtwx[[a, a]] += 1e-12;
        }
        let mut xtwy = Array1::<f64>::zeros(ext);
        for a in 0..ext {
            let mut s = 0.0;
            for i in 0..n {
                s += xb(i, a) * w(i) * y[i];
            }
            xtwy[a] = s;
        }
        // Cholesky solve.
        let m = Mat::from_fn(ext, ext, |i, j| xtwx[[i, j]]);
        let llt = faer::linalg::solvers::Llt::new(m.as_ref(), Side::Lower)
            .map_err(|e| RustMlError::InvalidParameter(format!("Cholesky failed: {e:?}")))?;
        let rhs = Mat::from_fn(ext, 1, |i, _| xtwy[i]);
        let sol = llt.solve(&rhs);
        let mut beta = Array1::<f64>::zeros(d);
        for j in 0..d {
            beta[j] = sol[(j, 0)];
        }
        let intercept = if self.with_intercept {
            sol[(d, 0)]
        } else {
            0.0
        };
        Ok(FittedWeightedRidgeRegressor {
            coef: beta,
            intercept,
            n_features: d,
        })
    }
}

impl Predict<f64> for FittedRidgeRegressor {
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_ridge_basic() {
        // y = 2 + 3x
        let x = Array2::from_shape_vec((10, 1), (0..10).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_vec((0..10).map(|i| 2.0 + 3.0 * i as f64).collect());

        let fitted = RidgeRegressor::new().with_lambda(0.01).fit(&x, &y).unwrap();

        // Ridge with small lambda should be close to OLS
        assert!(fitted.r_squared() > 0.99);
        assert_abs_diff_eq!(fitted.coefficients()[0], 3.0, epsilon = 0.1);
    }

    #[test]
    fn test_ridge_shrinks_coefficients() {
        let x = Array2::from_shape_vec((10, 1), (0..10).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_vec((0..10).map(|i| 2.0 + 3.0 * i as f64).collect());

        let fitted_small = RidgeRegressor::new().with_lambda(0.01).fit(&x, &y).unwrap();
        let fitted_large = RidgeRegressor::new()
            .with_lambda(100.0)
            .fit(&x, &y)
            .unwrap();

        // Larger lambda should shrink coefficients more
        assert!(
            fitted_large.coefficients()[0].abs() < fitted_small.coefficients()[0].abs(),
            "larger lambda should shrink coefficients: small={}, large={}",
            fitted_small.coefficients()[0],
            fitted_large.coefficients()[0]
        );
    }

    #[test]
    fn test_ridge_negative_lambda() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = RidgeRegressor::new().with_lambda(-1.0).fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_ridge_uniform_weights_match_unweighted() {
        // Sample-weighted fit with all-ones weights should match unweighted
        // (within numerical noise — anofox + our XᵀWX solver share the same
        // closed-form but use different solvers).
        let x = Array2::from_shape_vec((6, 1), (0..6).map(|i| i as f64).collect()).unwrap();
        let y = array![1.0_f64, 3.0, 5.0, 7.0, 9.0, 11.0]; // 2x + 1
        let rr = RidgeRegressor::new().with_lambda(0.01);
        let unw = rr.fit(&x, &y).unwrap();
        let w = Array1::<f64>::ones(6);
        let weighted = rr.fit_weighted(&x, &y, Some(&w)).unwrap();
        // Both should recover slope ~ 2 and intercept ~ 1.
        assert!((unw.coefficients()[0] - 2.0).abs() < 0.05);
        assert!((weighted.coef[0] - 2.0).abs() < 0.05);
        assert!((weighted.intercept - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_ridge_high_weight_anchor_dominates() {
        // 5 noisy points + 1 high-weight anchor at (10, 100). The fit should
        // pull strongly toward the anchor.
        let x = Array2::from_shape_vec((6, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0, 10.0]).unwrap();
        let y = array![0.0, 0.5, 0.5, 0.0, 0.0, 100.0];
        let w = array![1.0, 1.0, 1.0, 1.0, 1.0, 1e6];
        let fitted = RidgeRegressor::new()
            .with_lambda(1e-6)
            .fit_weighted(&x, &y, Some(&w))
            .unwrap();
        let p = fitted
            .predict(&Array2::from_shape_vec((1, 1), vec![10.0]).unwrap())
            .unwrap();
        assert!((p[0] - 100.0).abs() < 1.0, "anchor pred = {}", p[0]);
    }
}

impl anofox_ml_core::RegressorScore<f64> for FittedRidgeRegressor {}
impl anofox_ml_core::RegressorScore<f64> for FittedWeightedRidgeRegressor {}
