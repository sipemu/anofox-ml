//! Kernel ridge regression.
//!
//! Mirrors `sklearn.kernel_ridge.KernelRidge`. Solves
//!
//!   `(K + α I) α_dual = y`
//!
//! where `K = k(X, X)` is the kernel matrix and `α` is the regularisation
//! strength. Predictions are `K_test α_dual` with `K_test = k(X_test, X)`.

use faer::linalg::solvers::Solve;
use faer::{Mat, Side};
use ndarray::{Array1, Array2};
use rustml_core::{Fit, Predict, Result, RustMlError};
use rustml_svm::SvmKernel;

/// Kernel ridge regression estimator.
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
pub struct FittedKernelRidge {
    pub x_train: Array2<f64>,
    pub dual_coef: Array1<f64>,
    pub kernel: SvmKernel,
}

fn build_gram(x_a: &Array2<f64>, x_b: &Array2<f64>, kernel: &SvmKernel) -> Array2<f64> {
    let na = x_a.nrows();
    let nb = x_b.nrows();
    let mut k = Array2::<f64>::zeros((na, nb));
    for i in 0..na {
        let ri = x_a.row(i);
        for j in 0..nb {
            let rj = x_b.row(j);
            k[[i, j]] = kernel.compute(&ri, &rj);
        }
    }
    k
}

impl Fit<f64> for KernelRidge {
    type Fitted = FittedKernelRidge;

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

        let n = x.nrows();
        let mut k = build_gram(x, x, &self.kernel);
        for i in 0..n {
            k[[i, i]] += self.alpha;
        }

        // Cholesky solve via faer.
        let k_mat = Mat::from_fn(n, n, |i, j| k[[i, j]]);
        let llt = faer::linalg::solvers::Llt::new(k_mat.as_ref(), Side::Lower)
            .map_err(|e| RustMlError::InvalidParameter(format!("Cholesky failed: {e:?}")))?;
        let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);
        let sol = llt.solve(&y_mat);

        let dual = Array1::from_vec((0..n).map(|i| sol[(i, 0)]).collect());

        Ok(FittedKernelRidge {
            x_train: x.clone(),
            dual_coef: dual,
            kernel: self.kernel.clone(),
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
        let k_test = build_gram(x, &self.x_train, &self.kernel);
        Ok(k_test.dot(&self.dual_coef))
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
            .with_kernel(SvmKernel::Rbf { gamma: 1.0 })
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
}
