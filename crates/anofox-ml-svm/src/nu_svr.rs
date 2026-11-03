//! Nu-Support Vector Regression (nu-SVR).
//!
//! Direct port of libsvm's nu-SVR SMO (Chang & Lin 2011, "LIBSVM: A
//! Library for Support Vector Machines", and Chang & Lin 2002,
//! "Training ν-support vector classifiers: Theory and Algorithms").
//!
//! Solves the nu-SVR dual in libsvm's parameterization:
//!
//! ```text
//! min_β  (1/2) β^T Q β + p^T β
//!  s.t.  z^T β = 0                (bias constraint)
//!        e^T β = C · ν · ℓ        (nu constraint, libsvm convention)
//!        0 ≤ β_i ≤ C              (box)
//! ```
//!
//! where `β = [α_1, α_1*, …, α_ℓ, α_ℓ*]` is a 2ℓ-dimensional vector,
//! `z_i = +1` for α slots and `-1` for α* slots, `Q[i,j] = z_i z_j
//! K(x_{⌊i/2⌋}, x_{⌊j/2⌋})`, and `p_i = -z_i y_{⌊i/2⌋}`.
//!
//! The SMO working set selection restricts pairs to same-side (same
//! `z_i`) to keep both equality constraints satisfied. Within each side
//! the standard libsvm WSS criterion `-y·G` applies (where here
//! `y = z`), and the 2-variable update uses the closed-form Newton step
//! clipped to the box.

use anofox_ml_core::{Fit, Float, Predict, Result, RustMlError};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::kernel::SvmKernel;
use crate::svr;

/// Nu-Support Vector Regressor (unfitted state).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NuSvr {
    /// Nu parameter in (0, 1]. Upper bound on fraction of margin errors,
    /// lower bound on fraction of support vectors.
    pub nu: f64,
    /// Regularization parameter (box upper bound in the dual).
    pub c: f64,
    /// Kernel function.
    pub kernel: SvmKernel,
    /// Maximum number of SMO iterations.
    pub max_iter: usize,
    /// Stopping tolerance on the maximal violating pair (default 1e-3).
    pub tol: f64,
    /// Use libsvm's shrinking heuristic (default true).
    #[serde(default = "crate::solver::default_true")]
    pub shrinking: bool,
    /// Kernel cache size in MB (default 200).
    #[serde(default = "crate::solver::default_cache_size")]
    pub cache_size: f64,
}

impl NuSvr {
    pub fn new() -> Self {
        Self {
            nu: 0.5,
            c: 1.0,
            kernel: SvmKernel::default(),
            max_iter: 10_000_000,
            tol: 1e-3,
            shrinking: true,
            cache_size: 200.0,
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

    pub fn with_shrinking(mut self, shrinking: bool) -> Self {
        self.shrinking = shrinking;
        self
    }

    pub fn with_cache_size(mut self, cache_size: f64) -> Self {
        self.cache_size = cache_size;
        self
    }

    fn validate(&self) -> Result<()> {
        if self.nu <= 0.0 || self.nu > 1.0 {
            return Err(RustMlError::InvalidParameter("nu must be in (0, 1]".into()));
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
            return Err(RustMlError::InvalidParameter("tol must be positive".into()));
        }
        self.kernel.validate(false)?;
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
    fn predict_view(&self, x: ArrayView2<F>) -> Result<Array1<F>> {
        self.inner.predict(x)
    }
}

// ---------------------------------------------------------------------------
// nu-SVR SMO (direct port of libsvm's Solver_NU for SVR_NU)
// ---------------------------------------------------------------------------

impl<F: Float> Fit<F> for NuSvr {
    type Fitted = FittedNuSvr<F>;

    fn fit_view(&self, x: ArrayView2<F>, y: ArrayView1<F>) -> Result<Self::Fitted> {
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

        let kernel = self.kernel.resolve(x);
        let x64: Array2<f64> = x.mapv(|v| v.to_f64().unwrap());
        let y64: Vec<f64> = y.iter().map(|v| v.to_f64().unwrap()).collect();
        let (w, bias, _epsilon) = crate::solver::solve_nu_svr(
            crate::solver::KernelSource::Data {
                x: &x64,
                kernel: &kernel,
            },
            &y64,
            self.c,
            self.nu,
            &crate::solver::SolverParams {
                eps: self.tol,
                shrinking: self.shrinking,
                max_iter: Some(self.max_iter),
                nu: true,
            },
            self.cache_size,
        );

        let mut sv_indices: Vec<usize> = (0..x.nrows()).filter(|&i| w[i] != 0.0).collect();
        if sv_indices.is_empty() {
            sv_indices = (0..x.nrows()).collect();
        }

        let n_features = x.ncols();
        let mut sv_matrix = Array2::<F>::zeros((sv_indices.len(), n_features));
        let mut dual_coefs = Array1::<F>::zeros(sv_indices.len());
        for (sv_pos, &orig_idx) in sv_indices.iter().enumerate() {
            for j in 0..n_features {
                sv_matrix[[sv_pos, j]] = x[[orig_idx, j]];
            }
            dual_coefs[sv_pos] = F::from_f64(w[orig_idx]).unwrap();
        }

        let inner = svr::FittedSvr::from_parts(
            sv_matrix,
            dual_coefs,
            F::from_f64(bias).unwrap(),
            kernel.clone(),
            n_features,
        );

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

        let model = NuSvr::new()
            .with_kernel(SvmKernel::Linear)
            .with_c(100.0)
            .with_nu(0.5);
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
            .with_kernel(SvmKernel::rbf(0.1))
            .with_c(100.0)
            .with_nu(0.5);
        let fitted: FittedNuSvr<f64> = model.fit(&x, &y).unwrap();

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

        let small = NuSvr::new()
            .with_kernel(SvmKernel::Linear)
            .with_c(100.0)
            .with_nu(0.1);
        let fitted_small: FittedNuSvr<f64> = small.fit(&x, &y).unwrap();

        let large = NuSvr::new()
            .with_kernel(SvmKernel::Linear)
            .with_c(100.0)
            .with_nu(0.9);
        let fitted_large: FittedNuSvr<f64> = large.fit(&x, &y).unwrap();

        assert!(
            fitted_small.n_support() <= fitted_large.n_support() + 1,
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
            .with_nu(0.5);
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
            .with_nu(0.5);
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
    }

    #[test]
    fn test_f32_support() {
        let x: Array2<f32> = array![[1.0f32], [2.0], [3.0], [4.0]];
        let y: Array1<f32> = array![2.0f32, 4.0, 6.0, 8.0];

        let model = NuSvr::new()
            .with_kernel(SvmKernel::Linear)
            .with_c(10.0)
            .with_nu(0.5);
        let fitted: FittedNuSvr<f32> = model.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(p.is_finite());
        }
    }
}
