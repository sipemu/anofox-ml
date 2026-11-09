//! Gaussian Process regression and classification.
//!
//! Mirrors `sklearn.gaussian_process.GaussianProcessRegressor` /
//! `GaussianProcessClassifier`: kernel hyperparameters are learned in `fit`
//! by maximising the log-marginal likelihood (analytic gradients, bounded
//! L-BFGS-B in log space, optional random restarts), exactly like sklearn's
//! default `optimizer="fmin_l_bfgs_b"`.
//!
//! ## Kernels supported
//!
//! See [`GpKernel`]:
//!
//! - `Rbf` / `RbfArd` — squared exponential, isotropic or one length scale
//!   per feature, `σ² exp(-||x-x'||² / (2ℓ²))`
//! - `Matern` / `MaternArd` — `ν ∈ {0.5, 1.5, 2.5, ∞}` (other `ν` → error)
//! - `RationalQuadratic` — `σ² (1 + ||x-x'||² / (2αℓ²))^(-α)`
//! - `White` — noise on the diagonal of `K(X, X)`; zero cross-covariance
//! - `Constant` — `σ²` everywhere
//! - `Sum` / `Product` — composite kernels
//! - `Bounded` — per-hyperparameter bounds / fixed parameters
//!
//! # Example
//!
//! ```
//! use anofox_ml_core::{Fit, Predict};
//! use anofox_ml_gaussian_process::{GaussianProcessRegressor, GpKernel};
//! use ndarray::{Array1, Array2};
//!
//! let x = Array2::from_shape_fn((15, 1), |(i, _)| i as f64 * 0.4);
//! let y: Array1<f64> = x.column(0).mapv(f64::sin);
//! let kernel = GpKernel::Rbf { length_scale: 1.0, signal_var: 1.0 }
//!     .add(GpKernel::White { noise_level: 1e-2 });
//! let fitted = GaussianProcessRegressor::new(kernel).fit(&x, &y).unwrap();
//! let (mean, std) = fitted.predict_with_std(&x).unwrap();
//! assert!((mean[3] - y[3]).abs() < 0.05);
//! assert!(std.iter().all(|s| *s >= 0.0));
//! println!("learned kernel: {:?}", fitted.kernel);
//! ```

pub mod classifier;
pub mod kernels;
mod linalg;

pub use classifier::{
    classifier_log_marginal_likelihood, FittedGaussianProcessClassifier,
    FittedMulticlassGaussianProcessClassifier, GaussianProcessClassifier,
    MulticlassGaussianProcessClassifier,
};
pub use kernels::{GpKernel, ParamBounds, DEFAULT_BOUNDS};

use anofox_ml_core::optim::{minimize_lbfgsb, LbfgsbOptions};
use anofox_ml_core::{Fit, Predict, Result, RustMlError};
use faer::linalg::solvers::Solve;
use faer::{Mat, Side};
use linalg::{
    cholesky, cholesky_with_jitter, col_to_array1, solve_lower, std_normal, to_array2, to_mat,
    vec_to_col,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, AsArray, Ix2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Gaussian process regressor (unfitted configuration).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GaussianProcessRegressor {
    /// Prior covariance; its free hyperparameters are optimised in `fit`.
    pub kernel: GpKernel,
    /// Value added to the diagonal of `K(X, X)` (sklearn `alpha`, 1e-10).
    pub alpha: f64,
    /// Centre and scale `y` before fitting (sklearn `normalize_y`).
    pub normalize_y: bool,
    /// Maximise the log-marginal likelihood over the kernel
    /// hyperparameters with L-BFGS-B (sklearn default `optimizer` on).
    pub optimizer: bool,
    /// Additional optimiser runs from log-uniform random starting points
    /// within the bounds (sklearn `n_restarts_optimizer`, default 0).
    pub n_restarts_optimizer: usize,
    /// Seed for the restart starting points.
    pub seed: u64,
}

impl GaussianProcessRegressor {
    pub fn new(kernel: GpKernel) -> Self {
        Self {
            kernel,
            alpha: 1e-10,
            normalize_y: false,
            optimizer: true,
            n_restarts_optimizer: 0,
            seed: 0,
        }
    }
    pub fn with_alpha(mut self, a: f64) -> Self {
        self.alpha = a;
        self
    }
    pub fn with_normalize_y(mut self, v: bool) -> Self {
        self.normalize_y = v;
        self
    }
    /// Enable / disable hyperparameter optimisation (`false` ≙ sklearn
    /// `optimizer=None`).
    pub fn with_optimizer(mut self, v: bool) -> Self {
        self.optimizer = v;
        self
    }
    pub fn with_n_restarts_optimizer(mut self, n: usize) -> Self {
        self.n_restarts_optimizer = n;
        self
    }
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }
}

/// Fitted Gaussian process regressor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedGaussianProcessRegressor {
    pub x_train: Array2<f64>,
    /// Training targets (normalised if `normalize_y`).
    pub y_train: Array1<f64>,
    /// Lower Cholesky factor of `K + αI`.
    #[serde(with = "crate::linalg::mat_serde")]
    pub l_lower: Mat<f64>,
    /// `α = (K + αI)⁻¹ y` — used in mean prediction.
    pub dual: Array1<f64>,
    /// Kernel with the optimised hyperparameters (sklearn `kernel_`).
    pub kernel: GpKernel,
    pub y_mean: f64,
    pub y_std: f64,
    /// Diagonal regulariser used (`alpha` plus any Cholesky jitter).
    pub alpha: f64,
    /// Log-marginal likelihood of the fitted kernel on the (normalised)
    /// training targets (sklearn `log_marginal_likelihood_value_`).
    pub log_marginal_likelihood_value: f64,
}

/// Log-marginal likelihood and its gradient w.r.t. the kernel's free
/// log-hyperparameters (`½ tr((ααᵀ − K⁻¹) ∂K/∂θ)`). Non-PD → `-∞`.
pub(crate) fn lml_with_grad(
    kernel: &GpKernel,
    x: ArrayView2<f64>,
    y: &Array1<f64>,
    alpha: f64,
    eval_gradient: bool,
) -> (f64, Vec<f64>) {
    let n = x.nrows();
    let (mut k, grads) = if eval_gradient {
        kernel.gram_with_gradient(x)
    } else {
        (kernel.gram(x), Vec::new())
    };
    for i in 0..n {
        k[[i, i]] += alpha;
    }
    let llt = match cholesky(&k) {
        Some(l) => l,
        None => return (f64::NEG_INFINITY, vec![0.0; grads.len()]),
    };
    let a = col_to_array1(&llt.solve(vec_to_col(y)));
    let l = llt.L();
    let log_det_half: f64 = (0..n).map(|i| l[(i, i)].ln()).sum();
    let lml = -0.5 * y.dot(&a) - log_det_half - 0.5 * n as f64 * (2.0 * std::f64::consts::PI).ln();
    if !eval_gradient || grads.is_empty() {
        return (lml, Vec::new());
    }
    let k_inv = to_array2(llt.solve(Mat::<f64>::identity(n, n)).as_ref());
    // inner = ααᵀ − K⁻¹
    let mut inner = k_inv;
    for i in 0..n {
        for j in 0..n {
            inner[[i, j]] = a[i] * a[j] - inner[[i, j]];
        }
    }
    let grad = grads.iter().map(|dk| 0.5 * (&inner * dk).sum()).collect();
    (lml, grad)
}

/// Maximise `objective(θ) = (lml, grad)` over the kernel's free
/// log-hyperparameters from `kernel.theta()` plus `n_restarts` random
/// log-uniform starts. Returns the best kernel and its LML.
pub(crate) fn optimise_kernel<FObj>(
    kernel: &GpKernel,
    n_restarts: usize,
    seed: u64,
    objective: FObj,
) -> Result<(GpKernel, f64)>
where
    FObj: Fn(&GpKernel, bool) -> (f64, Vec<f64>),
{
    let bounds = kernel.theta_bounds();
    if bounds.is_empty() {
        let (lml, _) = objective(kernel, false);
        return Ok((kernel.clone(), lml));
    }
    let opts = LbfgsbOptions::default();
    let run = |theta0: &[f64]| -> Result<(Vec<f64>, f64)> {
        let res = minimize_lbfgsb(
            |th, g| {
                let k = match kernel.with_theta(th) {
                    Ok(k) => k,
                    Err(_) => return f64::INFINITY,
                };
                let (lml, grad) = objective(&k, true);
                if !lml.is_finite() {
                    g.iter_mut().for_each(|v| *v = 0.0);
                    return f64::INFINITY;
                }
                for (gi, v) in g.iter_mut().zip(grad) {
                    *gi = -v;
                }
                -lml
            },
            theta0,
            Some(&bounds),
            &opts,
        )?;
        Ok((res.x, -res.fun))
    };
    let mut best = run(&kernel.theta())?;
    if n_restarts > 0 {
        if bounds.iter().any(|(l, h)| !l.is_finite() || !h.is_finite()) {
            return Err(RustMlError::InvalidParameter(
                "multiple optimizer restarts require finite hyperparameter bounds".into(),
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        for _ in 0..n_restarts {
            let th0: Vec<f64> = bounds
                .iter()
                .map(|&(l, h)| if h > l { rng.gen_range(l..h) } else { l })
                .collect();
            let cand = run(&th0)?;
            if cand.1 > best.1 || !best.1.is_finite() {
                best = cand;
            }
        }
    }
    if !best.1.is_finite() {
        return Err(RustMlError::InvalidParameter(
            "hyperparameter optimisation failed: log-marginal likelihood is not finite".into(),
        ));
    }
    Ok((kernel.with_theta(&best.0)?, best.1))
}

impl Fit<f64> for GaussianProcessRegressor {
    type Fitted = FittedGaussianProcessRegressor;

    fn fit_view(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<Self::Fitted> {
        let n = x.nrows();
        if x.nrows() != y.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but y has {}",
                x.nrows(),
                y.len()
            )));
        }
        if n == 0 {
            return Err(RustMlError::EmptyInput("training data is empty".into()));
        }
        if self.alpha.is_nan() || self.alpha < 0.0 {
            return Err(RustMlError::InvalidParameter(format!(
                "alpha must be >= 0, got {}",
                self.alpha
            )));
        }
        self.kernel.validate(x.ncols())?;
        let (y_mean, y_std, y_norm) = if self.normalize_y {
            let m = y.sum() / n as f64;
            let v: f64 = y.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n as f64;
            // sklearn `_handle_zeros_in_scale`: a (near-)zero scale becomes 1.
            let s = v.sqrt();
            let s = if s < 10.0 * f64::EPSILON { 1.0 } else { s };
            (m, s, y.mapv(|v| (v - m) / s))
        } else {
            (0.0, 1.0, y.to_owned())
        };

        let alpha = self.alpha;
        let (kernel, lml_opt) = if self.optimizer && self.kernel.n_free_params() > 0 {
            let (k, lml) = optimise_kernel(
                &self.kernel,
                self.n_restarts_optimizer,
                self.seed,
                |k, g| lml_with_grad(k, x, &y_norm, alpha, g),
            )?;
            (k, Some(lml))
        } else {
            (self.kernel.clone(), None)
        };

        let mut k = kernel.gram(x);
        for i in 0..n {
            k[[i, i]] += alpha;
        }
        let (llt, jitter) = cholesky_with_jitter(&k)?;
        let dual = col_to_array1(&llt.solve(vec_to_col(&y_norm)));
        let l = llt.L().to_owned();
        let log_marginal_likelihood_value = match lml_opt {
            Some(v) if jitter == 0.0 => v,
            _ => {
                let log_det_half: f64 = (0..n).map(|i| l[(i, i)].ln()).sum();
                -0.5 * y_norm.dot(&dual)
                    - log_det_half
                    - 0.5 * n as f64 * (2.0 * std::f64::consts::PI).ln()
            }
        };

        Ok(FittedGaussianProcessRegressor {
            x_train: x.to_owned(),
            y_train: y_norm,
            l_lower: l,
            dual,
            kernel,
            y_mean,
            y_std,
            alpha: alpha + jitter,
            log_marginal_likelihood_value,
        })
    }
}

impl FittedGaussianProcessRegressor {
    fn check_features(&self, x: ArrayView2<f64>) -> Result<()> {
        if x.ncols() != self.x_train.ncols() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.x_train.ncols(),
                x.ncols()
            )));
        }
        Ok(())
    }

    /// `(mean, V)` with `V = L⁻¹ K(X_train, X)` (one batched solve).
    fn mean_and_v(&self, x: ArrayView2<f64>) -> Result<(Array1<f64>, Mat<f64>)> {
        self.check_features(x.view())?;
        let k_star = self.kernel.cross(x.view(), self.x_train.view()); // (n_q, n_train)
        let mean = k_star
            .dot(&self.dual)
            .mapv(|v| v * self.y_std + self.y_mean);
        let rhs = to_mat(&k_star.t().to_owned());
        let v = solve_lower(self.l_lower.as_ref(), rhs);
        Ok((mean, v))
    }

    /// Posterior mean and standard deviation (sklearn `return_std=True`).
    pub fn predict_with_std<'a>(
        &self,
        x: impl AsArray<'a, f64, Ix2>,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        let x: ArrayView2<f64> = x.into();
        let (mean, v) = self.mean_and_v(x)?;
        let kdiag = self.kernel.diag(x.view());
        let s2 = self.y_std * self.y_std;
        let std = Array1::from_shape_fn(x.nrows(), |j| {
            let vsq: f64 = (0..v.nrows()).map(|i| v[(i, j)] * v[(i, j)]).sum();
            ((kdiag[j] - vsq).max(0.0) * s2).sqrt()
        });
        Ok((mean, std))
    }

    /// Posterior standard deviation per query point (in target scale).
    pub fn predict_std<'a>(&self, x: impl AsArray<'a, f64, Ix2>) -> Result<Array1<f64>> {
        let x: ArrayView2<f64> = x.into();
        Ok(self.predict_with_std(x)?.1)
    }

    /// Posterior mean and full covariance (sklearn `return_cov=True`).
    pub fn predict_cov<'a>(
        &self,
        x: impl AsArray<'a, f64, Ix2>,
    ) -> Result<(Array1<f64>, Array2<f64>)> {
        let x: ArrayView2<f64> = x.into();
        let (mean, v) = self.mean_and_v(x)?;
        let vtv = to_array2((v.transpose() * &v).as_ref());
        let mut cov = self.kernel.gram(x.view()) - vtv;
        let s2 = self.y_std * self.y_std;
        cov.mapv_inplace(|c| c * s2);
        Ok((mean, cov))
    }

    /// Draw `n_samples` functions from the posterior at `x`; returns an
    /// `(n_query, n_samples)` matrix (sklearn `sample_y`).
    pub fn sample_y<'a>(
        &self,
        x: impl AsArray<'a, f64, Ix2>,
        n_samples: usize,
        seed: u64,
    ) -> Result<Array2<f64>> {
        let x: ArrayView2<f64> = x.into();
        let (mean, cov) = self.predict_cov(x)?;
        let q = x.nrows();
        let eig = faer::linalg::solvers::SelfAdjointEigen::new(to_mat(&cov).as_ref(), Side::Lower)
            .map_err(|e| {
                RustMlError::InvalidParameter(format!("eigendecomposition failed: {e:?}"))
            })?;
        let u = eig.U();
        let s = eig.S().column_vector();
        let mut rng = StdRng::seed_from_u64(seed);
        let mut out = Array2::<f64>::zeros((q, n_samples));
        let mut z = vec![0.0; q];
        for c in 0..n_samples {
            for (k, zk) in z.iter_mut().enumerate() {
                *zk = std_normal(&mut rng) * s[k].max(0.0).sqrt();
            }
            for i in 0..q {
                let mut acc = mean[i];
                for (k, zk) in z.iter().enumerate() {
                    acc += u[(i, k)] * zk;
                }
                out[[i, c]] = acc;
            }
        }
        Ok(out)
    }

    /// Log-marginal likelihood of the fitted kernel (sklearn
    /// `log_marginal_likelihood_value_`).
    pub fn log_marginal_likelihood_value(&self) -> f64 {
        self.log_marginal_likelihood_value
    }

    /// Log-marginal likelihood and gradient at log-hyperparameters `theta`
    /// (sklearn `log_marginal_likelihood(theta, eval_gradient=True)`),
    /// evaluated on the stored (normalised) training targets.
    pub fn log_marginal_likelihood(&self, theta: &[f64]) -> Result<(f64, Vec<f64>)> {
        let k = self.kernel.with_theta(theta)?;
        Ok(lml_with_grad(
            &k,
            self.x_train.view(),
            &self.y_train,
            self.alpha,
            true,
        ))
    }

    /// Fitted log-hyperparameters (sklearn `kernel_.theta`).
    pub fn theta(&self) -> Vec<f64> {
        self.kernel.theta()
    }
}

impl Predict<f64> for FittedGaussianProcessRegressor {
    fn predict_view(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        self.check_features(x)?;
        let k_star = self.kernel.cross(x, self.x_train.view());
        let mean_norm = k_star.dot(&self.dual);
        Ok(mean_norm.mapv(|v| v * self.y_std + self.y_mean))
    }
}

/// Compute the log-marginal-likelihood `log p(y | X, kernel, alpha)` for a
/// given kernel and noise level on `(X, y)` (no normalisation of `y`).
///
/// `log p(y|X,θ) = -0.5 yᵀ (K + αI)⁻¹ y - 0.5 log|K + αI| - n/2 log(2π)`
///
/// Non-positive-definite kernel matrices yield `-∞`.
pub fn log_marginal_likelihood(
    x: &Array2<f64>,
    y: &Array1<f64>,
    kernel: &GpKernel,
    alpha: f64,
) -> Result<f64> {
    if y.len() != x.nrows() {
        return Err(RustMlError::ShapeMismatch(format!(
            "X has {} rows but y has {}",
            x.nrows(),
            y.len()
        )));
    }
    kernel.validate(x.ncols())?;
    Ok(lml_with_grad(kernel, x.view(), y, alpha, false).0)
}

/// Log-marginal likelihood and its analytic gradient w.r.t. the free
/// log-hyperparameters of `kernel` (see [`GpKernel::theta`]).
pub fn log_marginal_likelihood_with_gradient(
    x: &Array2<f64>,
    y: &Array1<f64>,
    kernel: &GpKernel,
    alpha: f64,
) -> Result<(f64, Vec<f64>)> {
    if y.len() != x.nrows() {
        return Err(RustMlError::ShapeMismatch(format!(
            "X has {} rows but y has {}",
            x.nrows(),
            y.len()
        )));
    }
    kernel.validate(x.ncols())?;
    Ok(lml_with_grad(kernel, x.view(), y, alpha, true))
}

/// Result of multi-parameter hyperparameter optimisation.
#[derive(Debug, Clone)]
pub struct KernelOptimResult {
    pub log_params: Vec<f64>,
    pub log_marginal_likelihood: f64,
    pub n_iter: usize,
    pub converged: bool,
}

/// Maximise the log-marginal likelihood over arbitrary log-parameters with
/// L-BFGS (unbounded) and central finite-difference gradients.
///
/// `build` turns a parameter vector (in log-space) into a `GpKernel`.
/// Prefer [`GaussianProcessRegressor`]'s built-in optimiser (analytic
/// gradients, bounds, restarts); this helper remains for custom
/// parameterisations.
#[allow(clippy::too_many_arguments)]
pub fn optimize_kernel_lbfgs(
    x: &Array2<f64>,
    y: &Array1<f64>,
    alpha: f64,
    log_params_init: &[f64],
    build: impl Fn(&[f64]) -> GpKernel,
    n_iter: usize,
    fd_step: f64,
    grad_tol: f64,
) -> Result<KernelOptimResult> {
    let neg_lml = |p: &[f64]| -> f64 {
        let k = build(p);
        if k.validate(x.ncols()).is_err() {
            return f64::INFINITY;
        }
        let v = lml_with_grad(&k, x.view(), y, alpha, false).0;
        if v.is_finite() {
            -v
        } else {
            f64::INFINITY
        }
    };
    let res = minimize_lbfgsb(
        |p, g| {
            let f0 = neg_lml(p);
            let mut pm = p.to_vec();
            for i in 0..p.len() {
                let orig = pm[i];
                pm[i] = orig + fd_step;
                let fp = neg_lml(&pm);
                pm[i] = orig - fd_step;
                let fm = neg_lml(&pm);
                pm[i] = orig;
                g[i] = (fp - fm) / (2.0 * fd_step);
            }
            f0
        },
        log_params_init,
        None,
        &LbfgsbOptions {
            max_iter: n_iter,
            pgtol: grad_tol,
            ..Default::default()
        },
    )?;
    Ok(KernelOptimResult {
        converged: res.converged(),
        log_params: res.x,
        log_marginal_likelihood: -res.fun,
        n_iter: res.n_iter,
    })
}

/// Find the length_scale (RBF kernel) that maximises log-marginal-likelihood
/// via golden-section search over `log(length_scale)`. Other kernel
/// parameters are kept fixed at the provided values.
pub fn optimize_rbf_length_scale(
    x: &Array2<f64>,
    y: &Array1<f64>,
    signal_var: f64,
    alpha: f64,
    log_lo: f64,
    log_hi: f64,
    n_iter: usize,
) -> Result<f64> {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let resphi = 2.0 - phi; // ≈ 0.382
    let mut a = log_lo;
    let mut b = log_hi;
    let mut c = b - resphi * (b - a);
    let mut d = a + resphi * (b - a);
    let f = |log_ls: f64| -> Result<f64> {
        let k = GpKernel::Rbf {
            length_scale: log_ls.exp(),
            signal_var,
        };
        log_marginal_likelihood(x, y, &k, alpha).map(|v| -v)
    };
    let mut fc = f(c)?;
    let mut fd = f(d)?;
    for _ in 0..n_iter {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - resphi * (b - a);
            fc = f(c)?;
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + resphi * (b - a);
            fd = f(d)?;
        }
    }
    Ok((0.5 * (a + b)).exp())
}

impl anofox_ml_core::RegressorScore<f64> for FittedGaussianProcessRegressor {}

#[cfg(test)]
mod tests;
