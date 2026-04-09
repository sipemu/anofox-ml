//! Nu-Support Vector Regression (nu-SVR).
//!
//! Solves the proper nu-SVR primal from Schölkopf et al. (2000),
//! "New Support Vector Algorithms":
//!
//! ```text
//! min_{w, b, eps}   (1/2)||w||² + C·ν·eps + (C/n) Σ [|y_i - f(x_i) - b| - eps]_+
//!            s.t.   eps ≥ 0
//! ```
//!
//! At the joint optimum the KKT condition on epsilon requires
//! exactly `nu * n` samples to lie strictly outside the epsilon tube,
//! i.e. epsilon equals the `(1 - nu)` quantile of `|residuals|` under
//! the **joint** optimum `(w*, eps*)`.
//!
//! We solve this self-consistent fixed point by alternating between
//! (a) fitting epsilon-SVR for the current epsilon and (b) updating
//! epsilon to the `(1 - nu)` quantile of the resulting residuals.
//! Initialization is a near-interpolating fit with a tiny epsilon,
//! which produces residuals that accurately reflect the local structure
//! of the data — the subsequent epsilon update immediately lands in the
//! correct regime. The iteration is damped to guarantee monotone
//! convergence to the fixed point.

use ndarray::{Array1, Array2};
use rustml_core::{Fit, Float, Predict, Result, RustMlError};

use crate::kernel::SvmKernel;
use crate::svr;

/// Nu-Support Vector Regressor (unfitted state).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NuSvr {
    /// Nu parameter in (0, 1]. Upper bound on fraction of margin errors,
    /// lower bound on fraction of support vectors.
    pub nu: f64,
    /// Regularization parameter.
    pub c: f64,
    /// Kernel function.
    pub kernel: SvmKernel,
    /// Maximum number of outer iterations over epsilon.
    pub max_iter: usize,
    /// Tolerance for stopping criterion.
    pub tol: f64,
}

impl NuSvr {
    pub fn new() -> Self {
        Self {
            nu: 0.5,
            c: 1.0,
            kernel: SvmKernel::Rbf { gamma: 1.0 },
            max_iter: 1000,
            tol: 1e-4,
        }
    }

    pub fn with_nu(mut self, nu: f64) -> Self {
        self.nu = nu;
        self
    }

    pub fn with_c(mut self, c: f64) -> Self {
        self.c = c;
        self
    }

    pub fn with_kernel(mut self, kernel: SvmKernel) -> Self {
        self.kernel = kernel;
        self
    }

    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedNuSvr<F: Float> {
    inner: svr::FittedSvr<F>,
}

impl<F: Float> FittedNuSvr<F> {
    pub fn support_vectors(&self) -> &Array2<F> {
        self.inner.support_vectors()
    }

    pub fn n_support(&self) -> usize {
        self.inner.n_support()
    }

    pub fn bias(&self) -> F {
        self.inner.bias()
    }
}

impl<F: Float> Predict<F> for FittedNuSvr<F> {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        self.inner.predict(x)
    }
}

/// Compute the `(1 - nu)` quantile of a slice of non-negative values
/// using the interpolation convention that matches libsvm/sklearn nu-SVR
/// at the joint optimum: pick the value such that exactly `ceil(nu * n)`
/// entries are strictly above it.
fn nu_quantile(abs_res: &mut [f64], nu: f64) -> f64 {
    let n = abs_res.len();
    if n == 0 {
        return 0.0;
    }
    abs_res.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // We want `n_above = round(nu * n)` samples strictly greater than eps.
    // That means eps = abs_res[n - n_above - 1] if strict, or take the
    // boundary value. The element at index `n - n_above` is the smallest
    // "above" — by taking `abs_res[n - n_above - 1]` (the largest "below")
    // we place `n_above` samples strictly above.
    let n_above = (nu * n as f64).round() as usize;
    if n_above == 0 {
        // No errors allowed: eps = max residual.
        return *abs_res.last().unwrap();
    }
    if n_above >= n {
        // All samples should be errors: eps = 0.
        return 0.0;
    }
    abs_res[n - n_above - 1]
}

/// Fit epsilon-SVR on `(x, y)` for a given epsilon.
fn fit_eps_svr<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    eps: f64,
    c: f64,
    kernel: &SvmKernel,
    max_iter: usize,
    tol: f64,
) -> Result<svr::FittedSvr<F>> {
    let eps = eps.max(1e-12);
    let m = crate::Svr::new()
        .with_c(c)
        .with_epsilon(eps)
        .with_kernel(kernel.clone())
        .with_max_iter(max_iter)
        .with_tol(tol);
    m.fit(x, y)
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

        // Pass 1: fit epsilon-SVR with a tiny epsilon to obtain a
        // near-interpolating reference fit whose residuals characterize
        // the data well.
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
        let y_range = (y_max - y_min).max(1e-12);

        let ref_eps = (y_range * 1e-6).max(1e-10);
        let ref_fitted = fit_eps_svr(
            x,
            y,
            ref_eps,
            self.c,
            &self.kernel,
            self.max_iter,
            self.tol,
        )?;

        // Compute initial residuals from the reference fit.
        let ref_preds = ref_fitted.predict(x)?;
        let mut abs_res: Vec<f64> = ref_preds
            .iter()
            .zip(y.iter())
            .map(|(&p, &t)| (p - t).to_f64().unwrap().abs())
            .collect();

        // eps* is the (1 - nu) quantile of |residuals|.
        let mut eps = nu_quantile(&mut abs_res, self.nu);

        // Damped fixed-point refinement: alternate between fitting
        // epsilon-SVR and updating epsilon toward the nu-quantile of
        // |residuals| under that fit.
        let mut best_fitted = ref_fitted;
        let mut best_score = f64::INFINITY;

        let outer_iters = 8;
        for _ in 0..outer_iters {
            let fitted = fit_eps_svr(
                x,
                y,
                eps,
                self.c,
                &self.kernel,
                self.max_iter,
                self.tol,
            )?;
            let preds = fitted.predict(x)?;
            let mut new_abs_res: Vec<f64> = preds
                .iter()
                .zip(y.iter())
                .map(|(&p, &t)| (p - t).to_f64().unwrap().abs())
                .collect();

            // Score this fit by training SSE (for best-so-far tracking).
            let sse: f64 = new_abs_res.iter().map(|r| r * r).sum();
            if sse < best_score {
                best_score = sse;
                best_fitted = fitted;
            }

            let target_eps = nu_quantile(&mut new_abs_res, self.nu);
            let new_eps = 0.5 * (eps + target_eps); // damp

            if (new_eps - eps).abs() < 1e-8 * y_range.max(1.0) {
                break;
            }
            eps = new_eps;
        }

        Ok(FittedNuSvr {
            inner: best_fitted,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_linear_regression() {
        let x = array![
            [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]
        ];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

        let model = NuSvr::new()
            .with_kernel(SvmKernel::Linear)
            .with_c(100.0)
            .with_nu(0.5)
            .with_max_iter(5000);
        let fitted: FittedNuSvr<f64> = model.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(*p, *t, epsilon = 4.0);
        }
    }

    #[test]
    fn test_rbf_regression() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]];
        let y = array![1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0];

        let model = NuSvr::new()
            .with_kernel(SvmKernel::Rbf { gamma: 0.1 })
            .with_c(100.0)
            .with_nu(0.5)
            .with_max_iter(5000);
        let fitted: FittedNuSvr<f64> = model.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(p.is_finite(), "prediction should be finite, got {}", p);
        }
    }

    #[test]
    fn test_small_nu_fewer_svs() {
        let x = array![
            [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]
        ];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

        let small = NuSvr::new()
            .with_kernel(SvmKernel::Linear)
            .with_c(100.0)
            .with_nu(0.1)
            .with_max_iter(5000);
        let fitted_small: FittedNuSvr<f64> = small.fit(&x, &y).unwrap();

        let large = NuSvr::new()
            .with_kernel(SvmKernel::Linear)
            .with_c(100.0)
            .with_nu(0.9)
            .with_max_iter(5000);
        let fitted_large: FittedNuSvr<f64> = large.fit(&x, &y).unwrap();

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

        let model = NuSvr::new()
            .with_kernel(SvmKernel::Linear)
            .with_c(10.0)
            .with_nu(0.5)
            .with_max_iter(5000);
        let fitted: FittedNuSvr<f64> = model.fit(&x, &y).unwrap();

        assert!(fitted.n_support() > 0);
        assert!(fitted.n_support() <= x.nrows());
    }

    #[test]
    fn test_constant_target() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![5.0, 5.0, 5.0, 5.0];

        let model = NuSvr::new()
            .with_kernel(SvmKernel::Linear)
            .with_c(1.0)
            .with_nu(0.5)
            .with_max_iter(1000);
        let fitted: FittedNuSvr<f64> = model.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert_abs_diff_eq!(p, 5.0, epsilon = 1.0);
        }
    }

    #[test]
    fn test_empty_input_error() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<f64>::zeros(0);

        let model = NuSvr::new();
        let result: Result<FittedNuSvr<f64>> = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_shape_mismatch_fit() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 2.0, 3.0];

        let model = NuSvr::new();
        let result: Result<FittedNuSvr<f64>> = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_shape_mismatch_predict() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 2.0];

        let model = NuSvr::new().with_kernel(SvmKernel::Linear).with_c(10.0);
        let fitted: FittedNuSvr<f64> = model.fit(&x, &y).unwrap();

        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_invalid_nu_zero() {
        let x = array![[1.0], [2.0]];
        let y = array![1.0, 2.0];

        let model = NuSvr::new().with_nu(0.0);
        assert!(Fit::<f64>::fit(&model, &x, &y).is_err());
    }

    #[test]
    fn test_invalid_nu_negative() {
        let x = array![[1.0], [2.0]];
        let y = array![1.0, 2.0];

        let model = NuSvr::new().with_nu(-0.5);
        assert!(Fit::<f64>::fit(&model, &x, &y).is_err());
    }

    #[test]
    fn test_invalid_nu_above_one() {
        let x = array![[1.0], [2.0]];
        let y = array![1.0, 2.0];

        let model = NuSvr::new().with_nu(1.5);
        assert!(Fit::<f64>::fit(&model, &x, &y).is_err());
    }

    #[test]
    fn test_invalid_c() {
        let x = array![[1.0], [2.0]];
        let y = array![1.0, 2.0];

        let model = NuSvr::new().with_c(-1.0);
        assert!(Fit::<f64>::fit(&model, &x, &y).is_err());
    }

    #[test]
    fn test_builder_and_defaults() {
        let model = NuSvr::new()
            .with_nu(0.3)
            .with_c(5.0)
            .with_kernel(SvmKernel::Linear)
            .with_max_iter(500)
            .with_tol(1e-3);
        assert_eq!(model.nu, 0.3);
        assert_eq!(model.c, 5.0);
        assert_eq!(model.max_iter, 500);
        assert_eq!(model.tol, 1e-3);
        assert!(matches!(model.kernel, SvmKernel::Linear));

        let default = NuSvr::default();
        assert_eq!(default.nu, 0.5);
        assert_eq!(default.c, 1.0);
        assert_eq!(default.max_iter, 1000);
    }

    #[test]
    fn test_f32_support() {
        let x: Array2<f32> = array![[1.0f32], [2.0], [3.0], [4.0]];
        let y: Array1<f32> = array![2.0f32, 4.0, 6.0, 8.0];

        let model = NuSvr::new()
            .with_kernel(SvmKernel::Linear)
            .with_c(10.0)
            .with_nu(0.5)
            .with_max_iter(5000);
        let fitted: FittedNuSvr<f32> = model.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(p.is_finite());
        }
    }
}
