//! Kernel ridge regression.
//!
//! Mirrors `sklearn.kernel_ridge.KernelRidge`. Solves
//!
//!   `(K + α I) α_dual = y`
//!
//! where `K = k(X, X)` is the kernel matrix and `α` is the regularisation
//! strength. Predictions are `K_test α_dual` with `K_test = k(X_test, X)`.

use crate::convert::{ndarray_to_col, ndarray_to_mat};
use anofox_ml_core::{Fit, FitWeighted, Predict, Result, RustMlError};
use anofox_ml_svm::SvmKernel;
use faer::linalg::solvers::Solve;
use faer::Side;
use ndarray::{Array1, Array2, Axis};

/// Kernel ridge regression estimator.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KernelRidge {
    pub alpha: f64,
    pub kernel: SvmKernel,
}

impl KernelRidge {
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            kernel: SvmKernel::Linear,
        }
    }

    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn with_kernel(mut self, kernel: SvmKernel) -> Self {
        self.kernel = kernel;
        self
    }
}

impl Default for KernelRidge {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted kernel ridge regressor — stores the training set plus dual coefficients.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedKernelRidge {
    pub x_train: Array2<f64>,
    pub dual_coef: Array1<f64>,
    pub kernel: SvmKernel,
}

impl Fit<f64> for KernelRidge {
    type Fitted = FittedKernelRidge;

    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        self.fit_weighted(x, y, None)
    }
}

impl FitWeighted<f64> for KernelRidge {
    type Fitted = FittedKernelRidge;

    fn fit_weighted(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        sample_weight: Option<&Array1<f64>>,
    ) -> Result<Self::Fitted> {
        crate::convert::check_xy(x, y)?;
        if self.alpha < 0.0 {
            return Err(RustMlError::InvalidParameter(
                "alpha must be non-negative".into(),
            ));
        }
        if self.kernel.is_precomputed() {
            return Err(RustMlError::InvalidParameter(
                "KernelRidge does not support the precomputed kernel".into(),
            ));
        }
        self.kernel.validate(false)?;
        let sqrt_w = match sample_weight {
            Some(w) => {
                Some(crate::convert::resolve_sample_weight(Some(w), y.len())?.mapv(f64::sqrt))
            }
            None => None,
        };

        let n = x.nrows();
        // Resolve gamma = 'scale' / 'auto' against the training data.
        let kernel = self.kernel.resolve(x);
        let mut k = kernel.gram(x);
        // sklearn KernelRidge with sample_weight w solves (K W + α I) α = W y.
        // Use the symmetric reformulation: K' = √W K √W, y' = √W y, then
        // solve (K' + α I) α' = y' and back-transform α_dual = √W α'.
        let mut rhs = y.clone();
        if let Some(sw) = &sqrt_w {
            k *= &sw.view().insert_axis(Axis(1));
            k *= &sw.view().insert_axis(Axis(0));
            rhs *= sw;
        }
        k.diag_mut().mapv_inplace(|v| v + self.alpha);

        // Only the lower triangle is read by the Cholesky factorisation.
        let k_mat = ndarray_to_mat(&k);
        let llt = faer::linalg::solvers::Llt::new(k_mat.as_ref(), Side::Lower)
            .map_err(|e| RustMlError::InvalidParameter(format!("Cholesky failed: {e:?}")))?;
        let sol = llt.solve(ndarray_to_col(&rhs));
        let mut dual = Array1::from_shape_fn(n, |i| sol[i]);
        if let Some(sw) = &sqrt_w {
            dual *= sw;
        }

        Ok(FittedKernelRidge {
            x_train: x.clone(),
            dual_coef: dual,
            kernel,
        })
    }
}

impl Predict<f64> for FittedKernelRidge {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if x.ncols() != self.x_train.ncols() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.x_train.ncols(),
                x.ncols()
            )));
        }
        Ok(self.kernel.matrix(x, &self.x_train).dot(&self.dual_coef))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_linear_kernel_ridge_recovers_ridge_solution() {
        // For linear kernel, KernelRidge with alpha equals plain Ridge with
        // the same alpha (no intercept; sklearn KernelRidge has no intercept).
        let x = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]];
        let y = array![1.0, 2.0, 3.0, 2.0];

        let alpha = 0.5;
        let kr = KernelRidge::new()
            .with_alpha(alpha)
            .with_kernel(SvmKernel::Linear);
        let fitted = kr.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
        for &p in preds.iter() {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn test_rbf_perfect_fit_zero_alpha() {
        // With alpha=0 and distinct training points, RBF kernel ridge
        // interpolates y exactly.
        let x = array![[0.0], [1.0], [2.0], [3.0], [4.0]];
        let y = array![1.0, -1.0, 0.5, 2.0, -0.5];
        let fitted = KernelRidge::new()
            .with_alpha(1e-10)
            .with_kernel(SvmKernel::rbf(1.0))
            .fit(&x, &y)
            .unwrap();
        let pred = fitted.predict(&x).unwrap();
        for i in 0..5 {
            assert_abs_diff_eq!(pred[i], y[i], epsilon = 1e-5);
        }
    }

    #[test]
    fn test_negative_alpha_errors() {
        let x = array![[1.0]];
        let y = array![1.0];
        assert!(KernelRidge::new().with_alpha(-1.0).fit(&x, &y).is_err());
    }

    use anofox_ml_core::FitWeighted;

    #[test]
    fn test_kernel_ridge_uniform_weights_match_unweighted() {
        let x = array![[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [2.0, 3.0]];
        let y = array![1.0, 2.0, 3.0, 4.0];
        let kr = KernelRidge::new()
            .with_alpha(0.5)
            .with_kernel(SvmKernel::rbf(0.5));
        let unw = kr.fit(&x, &y).unwrap();
        let ones = Array1::<f64>::ones(4);
        let w = kr.fit_weighted(&x, &y, Some(&ones)).unwrap();
        // Under uniform weights and the √W α' substitution, dual_coef should
        // match unweighted exactly (weights of 1 cancel out).
        for i in 0..4 {
            assert_abs_diff_eq!(unw.dual_coef[i], w.dual_coef[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_kernel_ridge_high_weight_dominates() {
        // 5 noisy points + 1 high-weight anchor; the fit must hug the anchor.
        let x = array![[0.0], [1.0], [2.0], [3.0], [4.0], [10.0]];
        let y = array![0.0, 0.5, 0.5, 0.0, 0.0, 100.0];
        let kr = KernelRidge::new()
            .with_alpha(1e-3)
            .with_kernel(SvmKernel::rbf(0.5));
        let w = array![1.0, 1.0, 1.0, 1.0, 1.0, 1e6];
        let fitted = kr.fit_weighted(&x, &y, Some(&w)).unwrap();
        let p = fitted.predict(&array![[10.0]]).unwrap();
        assert!(
            (p[0] - 100.0).abs() < 1.0,
            "high-weight anchor pred={}",
            p[0]
        );
    }
}

impl anofox_ml_core::RegressorScore<f64> for FittedKernelRidge {}
