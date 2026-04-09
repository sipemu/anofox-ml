//! Nu-Support Vector Regression (NuSVR).
//!
//! Nu-parameterized SVR where `nu` in (0, 1] controls the fraction of
//! support vectors, replacing the epsilon parameter.

use ndarray::{Array1, Array2};
use rustml_core::{Fit, Float, Predict, Result, RustMlError};

use crate::kernel::SvmKernel;
use crate::svr;

/// Nu-Support Vector Regressor (unfitted state).
///
/// Uses a nu parameter instead of epsilon to control the width of the
/// epsilon-insensitive tube. The parameter `nu` is an upper bound on the
/// fraction of training errors and a lower bound on the fraction of
/// support vectors.
///
/// Uses the type-state pattern: call [`Fit::fit`] to produce a [`FittedNuSvr`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NuSvr {
    /// Nu parameter in (0, 1]. Controls the fraction of support vectors.
    pub nu: f64,
    /// Regularization parameter. Larger values penalize errors more.
    pub c: f64,
    /// Kernel function to use.
    pub kernel: SvmKernel,
    /// Maximum number of SMO iterations.
    pub max_iter: usize,
    /// Tolerance for stopping criterion.
    pub tol: f64,
}

impl NuSvr {
    /// Create a new `NuSvr` with default parameters.
    pub fn new() -> Self {
        Self {
            nu: 0.5,
            c: 1.0,
            kernel: SvmKernel::Rbf { gamma: 1.0 },
            max_iter: 1000,
            tol: 1e-4,
        }
    }

    /// Set the nu parameter.
    pub fn with_nu(mut self, nu: f64) -> Self {
        self.nu = nu;
        self
    }

    /// Set the regularization parameter C.
    pub fn with_c(mut self, c: f64) -> Self {
        self.c = c;
        self
    }

    /// Set the kernel function.
    pub fn with_kernel(mut self, kernel: SvmKernel) -> Self {
        self.kernel = kernel;
        self
    }

    /// Set the maximum number of SMO iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the tolerance for the stopping criterion.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Validate parameters before fitting.
    fn validate(&self) -> Result<()> {
        if self.nu <= 0.0 || self.nu > 1.0 {
            return Err(RustMlError::InvalidParameter(
                "nu must be in (0, 1]".into(),
            ));
        }
        if self.c <= 0.0 {
            return Err(RustMlError::InvalidParameter("C must be positive".into()));
        }
        if self.max_iter == 0 {
            return Err(RustMlError::InvalidParameter(
                "max_iter must be at least 1".into(),
            ));
        }
        if self.tol <= 0.0 {
            return Err(RustMlError::InvalidParameter(
                "tol must be positive".into(),
            ));
        }
        match &self.kernel {
            SvmKernel::Rbf { gamma } if *gamma <= 0.0 => {
                return Err(RustMlError::InvalidParameter(
                    "gamma must be positive for RBF kernel".into(),
                ));
            }
            SvmKernel::Polynomial { degree, .. } if *degree == 0 => {
                return Err(RustMlError::InvalidParameter(
                    "degree must be at least 1 for polynomial kernel".into(),
                ));
            }
            _ => {}
        }
        Ok(())
    }
}

impl Default for NuSvr {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Nu-Support Vector Regressor.
///
/// Wraps a [`FittedSvr`](crate::FittedSvr) internally, since NuSVR converts
/// nu to an equivalent epsilon and delegates to the standard SVR solver.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedNuSvr<F: Float> {
    inner: svr::FittedSvr<F>,
}

impl<F: Float> FittedNuSvr<F> {
    /// Returns the support vectors.
    pub fn support_vectors(&self) -> &Array2<F> {
        self.inner.support_vectors()
    }

    /// Returns the number of support vectors.
    pub fn n_support(&self) -> usize {
        self.inner.n_support()
    }

    /// Returns the bias term.
    pub fn bias(&self) -> F {
        self.inner.bias()
    }
}

impl<F: Float> Predict<F> for FittedNuSvr<F> {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        self.inner.predict(x)
    }
}

/// Convert nu to an equivalent epsilon for SVR.
///
/// The epsilon tube width is derived from the target range and nu:
///   epsilon = (y_max - y_min) * (1 - nu) / 2
///
/// When nu is close to 1, epsilon is near 0 (many support vectors).
/// When nu is close to 0, epsilon is large (few support vectors).
fn nu_to_epsilon(nu: f64, y_min: f64, y_max: f64) -> f64 {
    let y_range = y_max - y_min;
    if y_range < 1e-12 {
        // Constant target: any epsilon will do
        return 0.1;
    }
    let eps = y_range * (1.0 - nu) / 2.0;
    eps.max(1e-10) // floor at a tiny positive value
}

impl<F: Float> Fit<F> for NuSvr {
    type Fitted = FittedNuSvr<F>;

    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Self::Fitted> {
        self.validate()?;

        if x.is_empty() || y.is_empty() {
            return Err(RustMlError::EmptyInput(
                "training data must not be empty".into(),
            ));
        }
        if x.nrows() != y.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but y has {} elements",
                x.nrows(),
                y.len()
            )));
        }

        // Compute target range for epsilon estimation
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;
        for &val in y.iter() {
            let v = val.to_f64().unwrap();
            if v < y_min {
                y_min = v;
            }
            if v > y_max {
                y_max = v;
            }
        }

        let epsilon = nu_to_epsilon(self.nu, y_min, y_max);

        let svr = crate::Svr::new()
            .with_c(self.c)
            .with_epsilon(epsilon)
            .with_kernel(self.kernel.clone())
            .with_max_iter(self.max_iter)
            .with_tol(self.tol);

        let inner: svr::FittedSvr<F> = svr.fit(x, y)?;
        Ok(FittedNuSvr { inner })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_linear_regression() {
        // y = 2*x on well-separated data
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0],
            [10.0]
        ];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

        // Use large nu (small epsilon tube) so model fits closely
        let nu_svr = NuSvr::new()
            .with_kernel(SvmKernel::Linear)
            .with_c(100.0)
            .with_nu(0.9)
            .with_max_iter(5000);
        let fitted: FittedNuSvr<f64> = nu_svr.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(*p, *t, epsilon = 4.0);
        }
    }

    #[test]
    fn test_rbf_regression() {
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0]
        ];
        let y = array![1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0];

        let nu_svr = NuSvr::new()
            .with_kernel(SvmKernel::Rbf { gamma: 0.1 })
            .with_c(100.0)
            .with_nu(0.8)
            .with_max_iter(5000);
        let fitted: FittedNuSvr<f64> = nu_svr.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(p.is_finite(), "prediction should be finite, got {}", p);
        }
    }

    #[test]
    fn test_small_nu_fewer_svs() {
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0],
            [10.0]
        ];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

        // Small nu => larger epsilon => fewer SVs
        let nu_svr_small = NuSvr::new()
            .with_kernel(SvmKernel::Linear)
            .with_c(100.0)
            .with_nu(0.1)
            .with_max_iter(5000);
        let fitted_small: FittedNuSvr<f64> = nu_svr_small.fit(&x, &y).unwrap();

        // Large nu => smaller epsilon => more SVs
        let nu_svr_large = NuSvr::new()
            .with_kernel(SvmKernel::Linear)
            .with_c(100.0)
            .with_nu(0.9)
            .with_max_iter(5000);
        let fitted_large: FittedNuSvr<f64> = nu_svr_large.fit(&x, &y).unwrap();

        // With small nu (large epsilon), we expect fewer or equal SVs
        assert!(
            fitted_small.n_support() <= fitted_large.n_support(),
            "small nu ({} SVs) should have <= SVs than large nu ({} SVs)",
            fitted_small.n_support(),
            fitted_large.n_support()
        );
    }

    #[test]
    fn test_support_vectors_exist() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let nu_svr = NuSvr::new()
            .with_kernel(SvmKernel::Linear)
            .with_c(10.0)
            .with_nu(0.5)
            .with_max_iter(5000);
        let fitted: FittedNuSvr<f64> = nu_svr.fit(&x, &y).unwrap();

        assert!(
            fitted.n_support() > 0,
            "should have at least one support vector"
        );
        assert!(
            fitted.n_support() <= x.nrows(),
            "cannot have more SVs than training samples"
        );
    }

    #[test]
    fn test_constant_target() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![5.0, 5.0, 5.0, 5.0];

        let nu_svr = NuSvr::new()
            .with_kernel(SvmKernel::Linear)
            .with_c(1.0)
            .with_nu(0.5)
            .with_max_iter(1000);
        let fitted: FittedNuSvr<f64> = nu_svr.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert_abs_diff_eq!(p, 5.0, epsilon = 1.0);
        }
    }

    #[test]
    fn test_empty_input_error() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<f64>::zeros(0);

        let nu_svr = NuSvr::new();
        let result: Result<FittedNuSvr<f64>> = nu_svr.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_shape_mismatch_fit() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 2.0, 3.0];

        let nu_svr = NuSvr::new();
        let result: Result<FittedNuSvr<f64>> = nu_svr.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_shape_mismatch_predict() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 2.0];

        let nu_svr = NuSvr::new()
            .with_kernel(SvmKernel::Linear)
            .with_c(10.0);
        let fitted: FittedNuSvr<f64> = nu_svr.fit(&x, &y).unwrap();

        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_invalid_nu_zero() {
        let x = array![[1.0], [2.0]];
        let y = array![1.0, 2.0];

        let nu_svr = NuSvr::new().with_nu(0.0);
        assert!(Fit::<f64>::fit(&nu_svr, &x, &y).is_err());
    }

    #[test]
    fn test_invalid_nu_negative() {
        let x = array![[1.0], [2.0]];
        let y = array![1.0, 2.0];

        let nu_svr = NuSvr::new().with_nu(-0.5);
        assert!(Fit::<f64>::fit(&nu_svr, &x, &y).is_err());
    }

    #[test]
    fn test_invalid_nu_above_one() {
        let x = array![[1.0], [2.0]];
        let y = array![1.0, 2.0];

        let nu_svr = NuSvr::new().with_nu(1.5);
        assert!(Fit::<f64>::fit(&nu_svr, &x, &y).is_err());
    }

    #[test]
    fn test_invalid_c() {
        let x = array![[1.0], [2.0]];
        let y = array![1.0, 2.0];

        let nu_svr = NuSvr::new().with_c(-1.0);
        assert!(Fit::<f64>::fit(&nu_svr, &x, &y).is_err());
    }

    #[test]
    fn test_builder_and_defaults() {
        let nu_svr = NuSvr::new()
            .with_nu(0.3)
            .with_c(5.0)
            .with_kernel(SvmKernel::Linear)
            .with_max_iter(500)
            .with_tol(1e-3);
        assert_eq!(nu_svr.nu, 0.3);
        assert_eq!(nu_svr.c, 5.0);
        assert_eq!(nu_svr.max_iter, 500);
        assert_eq!(nu_svr.tol, 1e-3);
        assert!(matches!(nu_svr.kernel, SvmKernel::Linear));

        let default = NuSvr::default();
        assert_eq!(default.nu, 0.5);
        assert_eq!(default.c, 1.0);
        assert_eq!(default.max_iter, 1000);
    }

    #[test]
    fn test_f32_support() {
        let x: Array2<f32> = array![[1.0f32], [2.0], [3.0], [4.0]];
        let y: Array1<f32> = array![2.0f32, 4.0, 6.0, 8.0];

        let nu_svr = NuSvr::new()
            .with_kernel(SvmKernel::Linear)
            .with_c(10.0)
            .with_nu(0.5)
            .with_max_iter(5000);
        let fitted: FittedNuSvr<f32> = nu_svr.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(p.is_finite());
        }
    }
}
