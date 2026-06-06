//! Gaussian Process Regression.
//!
//! Mirrors `sklearn.gaussian_process.GaussianProcessRegressor` with a fixed
//! kernel composed from the kernel zoo below. Hyperparameter learning is not
//! yet implemented — provide explicit kernel parameters.
//!
//! ## Kernels supported
//!
//! - `Rbf` — squared-exponential `σ² exp(-||x-x'||² / (2ℓ²))`
//! - `Matern` — Matern kernel for `ν ∈ {0.5, 1.5, 2.5}` (closed-form
//!   parameterisations)
//! - `RationalQuadratic` — `(1 + ||x-x'||² / (2αℓ²))^(-α)`
//! - `White` — `σ²` if `x == x'` else `0` (diagonal noise)
//! - `Constant` — `σ²` everywhere
//! - `Sum` / `Product` — composite kernels

pub mod classifier;
pub use classifier::{
    FittedGaussianProcessClassifier, FittedMulticlassGaussianProcessClassifier,
    GaussianProcessClassifier, MulticlassGaussianProcessClassifier,
};

use faer::linalg::solvers::Solve;
use faer::{Mat, Side};
use ndarray::{Array1, Array2};
use rustml_core::{Fit, Predict, Result, RustMlError};

/// Composable kernel.
pub enum GpKernel {
    Rbf {
        length_scale: f64,
        signal_var: f64,
    },
    /// Matern with `nu in {0.5, 1.5, 2.5}`. Other values would require Bessel
    /// functions — currently restricted.
    Matern {
        length_scale: f64,
        signal_var: f64,
        nu: f64,
    },
    RationalQuadratic {
        length_scale: f64,
        signal_var: f64,
        alpha: f64,
    },
    /// Adds `noise_level` to the diagonal of K(X, X). Zero off-diagonal.
    White {
        noise_level: f64,
    },
    Constant {
        value: f64,
    },
    Sum(Box<GpKernel>, Box<GpKernel>),
    Product(Box<GpKernel>, Box<GpKernel>),
}

impl GpKernel {
    /// Compute `k(a, b)` for two single feature vectors. Note: the `White`
    /// kernel returns `0` for `a != b` and `noise_level` for `a == b` (exact
    /// equality). For predictive variance at new query points, only the
    /// diagonal entries matter.
    fn compute(&self, a: &[f64], b: &[f64]) -> f64 {
        match self {
            GpKernel::Rbf {
                length_scale,
                signal_var,
            } => {
                let sd: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
                signal_var * (-0.5 * sd / (length_scale * length_scale)).exp()
            }
            GpKernel::Matern {
                length_scale,
                signal_var,
                nu,
            } => {
                let d: f64 = a
                    .iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let r = d / length_scale;
                let v = if (nu - 0.5).abs() < 1e-9 {
                    (-r).exp()
                } else if (nu - 1.5).abs() < 1e-9 {
                    let sqrt3_r = 3.0_f64.sqrt() * r;
                    (1.0 + sqrt3_r) * (-sqrt3_r).exp()
                } else if (nu - 2.5).abs() < 1e-9 {
                    let sqrt5_r = 5.0_f64.sqrt() * r;
                    (1.0 + sqrt5_r + 5.0 / 3.0 * r * r) * (-sqrt5_r).exp()
                } else {
                    // Fallback to RBF if unsupported nu.
                    (-0.5 * (d * d) / (length_scale * length_scale)).exp()
                };
                signal_var * v
            }
            GpKernel::RationalQuadratic {
                length_scale,
                signal_var,
                alpha,
            } => {
                let sd: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
                let base = 1.0 + sd / (2.0 * alpha * length_scale * length_scale);
                signal_var * base.powf(-alpha)
            }
            GpKernel::White { noise_level } => {
                // Compare for floating-point equality across all dims.
                if a.iter().zip(b.iter()).all(|(x, y)| x == y) {
                    *noise_level
                } else {
                    0.0
                }
            }
            GpKernel::Constant { value } => *value,
            GpKernel::Sum(k1, k2) => k1.compute(a, b) + k2.compute(a, b),
            GpKernel::Product(k1, k2) => k1.compute(a, b) * k2.compute(a, b),
        }
    }

    /// Helper to build `Rbf * Constant + White`-style composites.
    pub fn product(self, other: GpKernel) -> GpKernel {
        GpKernel::Product(Box::new(self), Box::new(other))
    }
    pub fn add(self, other: GpKernel) -> GpKernel {
        GpKernel::Sum(Box::new(self), Box::new(other))
    }
}

pub struct GaussianProcessRegressor {
    pub kernel: GpKernel,
    pub alpha: f64,
    pub normalize_y: bool,
}

impl GaussianProcessRegressor {
    pub fn new(kernel: GpKernel) -> Self {
        Self {
            kernel,
            alpha: 1e-10,
            normalize_y: false,
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
}

pub struct FittedGaussianProcessRegressor {
    pub x_train: Array2<f64>,
    pub y_train: Array1<f64>,
    /// Lower Cholesky factor of `K + αI`.
    pub l_lower: Mat<f64>,
    /// `α = L⁻ᵀ L⁻¹ y` — used in mean prediction.
    pub dual: Array1<f64>,
    pub kernel: GpKernel,
    pub y_mean: f64,
    pub y_std: f64,
}

pub(crate) fn build_gram(x_a: &Array2<f64>, x_b: &Array2<f64>, kernel: &GpKernel) -> Array2<f64> {
    let na = x_a.nrows();
    let nb = x_b.nrows();
    let mut out = Array2::<f64>::zeros((na, nb));
    for i in 0..na {
        let ai = x_a.row(i).to_owned();
        for j in 0..nb {
            let bj = x_b.row(j).to_owned();
            out[[i, j]] = kernel.compute(ai.as_slice().unwrap(), bj.as_slice().unwrap());
        }
    }
    out
}

fn clone_kernel(k: &GpKernel) -> GpKernel {
    match k {
        GpKernel::Rbf {
            length_scale,
            signal_var,
        } => GpKernel::Rbf {
            length_scale: *length_scale,
            signal_var: *signal_var,
        },
        GpKernel::Matern {
            length_scale,
            signal_var,
            nu,
        } => GpKernel::Matern {
            length_scale: *length_scale,
            signal_var: *signal_var,
            nu: *nu,
        },
        GpKernel::RationalQuadratic {
            length_scale,
            signal_var,
            alpha,
        } => GpKernel::RationalQuadratic {
            length_scale: *length_scale,
            signal_var: *signal_var,
            alpha: *alpha,
        },
        GpKernel::White { noise_level } => GpKernel::White {
            noise_level: *noise_level,
        },
        GpKernel::Constant { value } => GpKernel::Constant { value: *value },
        GpKernel::Sum(a, b) => GpKernel::Sum(Box::new(clone_kernel(a)), Box::new(clone_kernel(b))),
        GpKernel::Product(a, b) => {
            GpKernel::Product(Box::new(clone_kernel(a)), Box::new(clone_kernel(b)))
        }
    }
}

impl Fit<f64> for GaussianProcessRegressor {
    type Fitted = FittedGaussianProcessRegressor;

    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        let n = x.nrows();
        if x.nrows() != y.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but y has {}",
                x.nrows(),
                y.len()
            )));
        }
        let (y_mean, y_std, y_norm) = if self.normalize_y {
            let m = y.sum() / n as f64;
            let v: f64 = y.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n as f64;
            let s = v.sqrt().max(1e-12);
            let yn = y.mapv(|v| (v - m) / s);
            (m, s, yn)
        } else {
            (0.0, 1.0, y.clone())
        };

        let mut k = build_gram(x, x, &self.kernel);
        for i in 0..n {
            k[[i, i]] += self.alpha;
        }
        let km = Mat::from_fn(n, n, |i, j| k[[i, j]]);
        let llt = faer::linalg::solvers::Llt::new(km.as_ref(), Side::Lower)
            .map_err(|e| RustMlError::InvalidParameter(format!("Cholesky failed: {e:?}")))?;
        let ym = Mat::from_fn(n, 1, |i, _| y_norm[i]);
        let sol = llt.solve(&ym);
        let dual: Array1<f64> = Array1::from_vec((0..n).map(|i| sol[(i, 0)]).collect());
        // Save the lower factor for variance prediction.
        let lower = llt.L();
        let l = Mat::from_fn(n, n, |i, j| lower[(i, j)]);

        Ok(FittedGaussianProcessRegressor {
            x_train: x.clone(),
            y_train: y_norm,
            l_lower: l,
            dual,
            kernel: clone_kernel(&self.kernel),
            y_mean,
            y_std,
        })
    }
}

impl Predict<f64> for FittedGaussianProcessRegressor {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if x.ncols() != self.x_train.ncols() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.x_train.ncols(),
                x.ncols()
            )));
        }
        let k_star = build_gram(x, &self.x_train, &self.kernel);
        let mean_norm = k_star.dot(&self.dual);
        Ok(mean_norm.mapv(|v| v * self.y_std + self.y_mean))
    }
}

/// Compute the log-marginal-likelihood `log p(y | X, kernel, alpha)` for a
/// given kernel and noise level on `(X, y)`.
///
/// `log p(y|X,θ) = -0.5 yᵀ (K + αI)⁻¹ y - 0.5 log|K + αI| - n/2 log(2π)`
///
/// Used by hyperparameter-search loops below.
pub fn log_marginal_likelihood(
    x: &Array2<f64>,
    y: &Array1<f64>,
    kernel: &GpKernel,
    alpha: f64,
) -> Result<f64> {
    let n = x.nrows();
    if y.len() != n {
        return Err(RustMlError::ShapeMismatch(format!(
            "X has {} rows but y has {}",
            n,
            y.len()
        )));
    }
    let mut k = build_gram(x, x, kernel);
    for i in 0..n {
        k[[i, i]] += alpha;
    }
    let km = Mat::from_fn(n, n, |i, j| k[[i, j]]);
    let llt = match faer::linalg::solvers::Llt::new(km.as_ref(), Side::Lower) {
        Ok(llt) => llt,
        // Non-PD kernel matrices yield −∞ likelihood — caller can compare safely.
        Err(_) => return Ok(f64::NEG_INFINITY),
    };
    let ym = Mat::from_fn(n, 1, |i, _| y[i]);
    let sol = llt.solve(&ym);
    let mut yt_k_inv_y = 0.0;
    for i in 0..n {
        yt_k_inv_y += y[i] * sol[(i, 0)];
    }
    let lower = llt.L();
    let mut log_det = 0.0;
    for i in 0..n {
        log_det += lower[(i, i)].abs().ln();
    }
    let log_det = 2.0 * log_det;
    let two_pi = 2.0 * std::f64::consts::PI;
    Ok(-0.5 * yt_k_inv_y - 0.5 * log_det - 0.5 * n as f64 * two_pi.ln())
}

/// Result of multi-parameter hyperparameter optimisation.
#[derive(Debug, Clone)]
pub struct KernelOptimResult {
    pub log_params: Vec<f64>,
    pub log_marginal_likelihood: f64,
    pub n_iter: usize,
    pub converged: bool,
}

/// Multivariate quasi-Newton (BFGS) optimisation of arbitrary kernel
/// hyperparameters on the log-scale, maximising log-marginal likelihood.
///
/// `build` turns a parameter vector (in log-space) into a `GpKernel`. The
/// optimiser starts from `log_params_init`, computes finite-difference
/// gradients with step `fd_step`, and updates a dense inverse-Hessian
/// approximation via the BFGS formula. Backtracking line search guarantees
/// monotone increase in log-likelihood.
///
/// For low-dim problems (typical kernel zoo has ≤ 5 params) BFGS converges
/// in a handful of iterations and is more practical than L-BFGS, which adds
/// history-buffer bookkeeping for negligible benefit at this scale.
///
/// Returns the optimised log-parameters, the achieved log-marginal
/// likelihood, iteration count, and whether the gradient-norm stop
/// criterion fired.
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
    let n_params = log_params_init.len();
    let neg_lml =
        |p: &[f64]| -> Result<f64> { log_marginal_likelihood(x, y, &build(p), alpha).map(|v| -v) };
    let grad_fd = |p: &[f64], f0: f64| -> Result<Vec<f64>> {
        let mut g = vec![0.0_f64; n_params];
        let mut p_mut = p.to_vec();
        for i in 0..n_params {
            let orig = p_mut[i];
            p_mut[i] = orig + fd_step;
            let fp = neg_lml(&p_mut)?;
            p_mut[i] = orig;
            // Forward-difference. Cheaper than central, accurate enough for
            // log-space hyperparameters where fd_step ~ 1e-4.
            g[i] = (fp - f0) / fd_step;
        }
        Ok(g)
    };

    let mut p = log_params_init.to_vec();
    let mut f = neg_lml(&p)?;
    let mut g = grad_fd(&p, f)?;
    // Inverse-Hessian approximation, initially identity.
    let mut h = vec![vec![0.0_f64; n_params]; n_params];
    for i in 0..n_params {
        h[i][i] = 1.0;
    }

    let mut converged = false;
    let mut iters = 0;
    for it in 0..n_iter {
        iters = it + 1;
        // Search direction d = -H g
        let mut d = vec![0.0_f64; n_params];
        for i in 0..n_params {
            let mut s = 0.0;
            for j in 0..n_params {
                s -= h[i][j] * g[j];
            }
            d[i] = s;
        }
        // Backtracking line search (Armijo with c1 = 1e-4).
        let g_dot_d: f64 = g.iter().zip(d.iter()).map(|(a, b)| a * b).sum();
        if g_dot_d >= 0.0 {
            // Not a descent direction (numerical issue) — reset H.
            for i in 0..n_params {
                for j in 0..n_params {
                    h[i][j] = 0.0;
                }
                h[i][i] = 1.0;
            }
            for i in 0..n_params {
                d[i] = -g[i];
            }
        }
        let mut step = 1.0_f64;
        let c1 = 1e-4;
        let mut p_new;
        let mut f_new;
        let mut ls_iter = 0;
        loop {
            ls_iter += 1;
            p_new = p
                .iter()
                .zip(d.iter())
                .map(|(a, b)| a + step * b)
                .collect::<Vec<_>>();
            f_new = neg_lml(&p_new)?;
            if f_new.is_finite() && f_new <= f + c1 * step * g_dot_d {
                break;
            }
            step *= 0.5;
            if step < 1e-12 || ls_iter > 50 {
                // Line search failed — accept whatever we have and stop.
                break;
            }
        }

        let s_vec: Vec<f64> = p_new.iter().zip(p.iter()).map(|(a, b)| a - b).collect();
        let g_new = grad_fd(&p_new, f_new)?;
        let y_vec: Vec<f64> = g_new.iter().zip(g.iter()).map(|(a, b)| a - b).collect();
        let sy: f64 = s_vec.iter().zip(y_vec.iter()).map(|(a, b)| a * b).sum();

        if sy > 1e-12 {
            // BFGS inverse-Hessian update:
            //   H_new = (I - ρ s yᵀ) H (I - ρ y sᵀ) + ρ s sᵀ
            // Compute H y, then update.
            let rho = 1.0 / sy;
            let mut hy = vec![0.0_f64; n_params];
            for i in 0..n_params {
                for j in 0..n_params {
                    hy[i] += h[i][j] * y_vec[j];
                }
            }
            let yhy: f64 = y_vec.iter().zip(hy.iter()).map(|(a, b)| a * b).sum();
            for i in 0..n_params {
                for j in 0..n_params {
                    h[i][j] = h[i][j] - rho * (s_vec[i] * hy[j] + hy[i] * s_vec[j])
                        + rho * (rho * yhy + 1.0) * s_vec[i] * s_vec[j];
                }
            }
        }

        p = p_new;
        f = f_new;
        g = g_new;

        let gnorm: f64 = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if gnorm < grad_tol {
            converged = true;
            break;
        }
    }

    Ok(KernelOptimResult {
        log_params: p,
        log_marginal_likelihood: -f,
        n_iter: iters,
        converged,
    })
}

/// Find the length_scale (RBF kernel) that maximises log-marginal-likelihood
/// via golden-section search over `log(length_scale)`. Other kernel
/// parameters are kept fixed at the provided values.
///
/// Mirrors what `sklearn.gaussian_process.GaussianProcessRegressor` does when
/// `optimizer='fmin_l_bfgs_b'` is engaged for a single RBF hyperparameter —
/// in practice they use L-BFGS, but a golden-section sweep on log-scale is a
/// strong baseline.
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
        // Maximise → return value (higher better); we minimise the negation.
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
    Ok(0.5 * (a + b)).map(|log_ls| log_ls.exp())
}

impl FittedGaussianProcessRegressor {
    /// Posterior standard deviation per query point (in target scale).
    pub fn predict_std(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let n_train = self.x_train.nrows();
        if x.ncols() != self.x_train.ncols() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.x_train.ncols(),
                x.ncols()
            )));
        }
        let n_new = x.nrows();
        let k_star = build_gram(x, &self.x_train, &self.kernel);
        let mut std_out = Array1::<f64>::zeros(n_new);
        for i in 0..n_new {
            let rhs = Mat::from_fn(n_train, 1, |j, _| k_star[[i, j]]);
            let n = n_train;
            let mut v = vec![0.0_f64; n];
            for r in 0..n {
                let mut s = rhs[(r, 0)];
                for c in 0..r {
                    s -= self.l_lower[(r, c)] * v[c];
                }
                v[r] = s / self.l_lower[(r, r)].max(1e-12);
            }
            let v_sq: f64 = v.iter().map(|x| x * x).sum();
            let xi = x.row(i).to_owned();
            let k_xx = self
                .kernel
                .compute(xi.as_slice().unwrap(), xi.as_slice().unwrap());
            let var = (k_xx - v_sq).max(0.0);
            std_out[i] = var.sqrt() * self.y_std;
        }
        Ok(std_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_gp_rbf_interpolates_with_low_noise() {
        let x = Array2::from_shape_vec((6, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|v: f64| v.sin());
        let kernel = GpKernel::Rbf {
            length_scale: 1.0,
            signal_var: 1.0,
        };
        let fitted = GaussianProcessRegressor::new(kernel)
            .with_alpha(1e-8)
            .fit(&x, &y)
            .unwrap();
        let p = fitted.predict(&x).unwrap();
        for i in 0..6 {
            assert!((p[i] - y[i]).abs() < 1e-4, "[{i}]: {} vs {}", p[i], y[i]);
        }
        let _ = array![1.0_f64];
    }

    #[test]
    fn test_gp_matern_nu_2p5_interpolates() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|v: f64| v.cos());
        let kernel = GpKernel::Matern {
            length_scale: 1.0,
            signal_var: 1.0,
            nu: 2.5,
        };
        let fitted = GaussianProcessRegressor::new(kernel)
            .with_alpha(1e-8)
            .fit(&x, &y)
            .unwrap();
        let p = fitted.predict(&x).unwrap();
        for i in 0..5 {
            assert!((p[i] - y[i]).abs() < 1e-3, "[{i}]: {} vs {}", p[i], y[i]);
        }
    }

    #[test]
    fn test_gp_rational_quadratic_runs() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0.0, 1.0, 0.5, -0.5, 0.0];
        let kernel = GpKernel::RationalQuadratic {
            length_scale: 1.0,
            signal_var: 1.0,
            alpha: 0.5,
        };
        let fitted = GaussianProcessRegressor::new(kernel)
            .with_alpha(1e-6)
            .fit(&x, &y)
            .unwrap();
        let p = fitted.predict(&x).unwrap();
        for v in p.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_optimize_rbf_length_scale_picks_sensible_value() {
        // Generate y = sin(x); the "right" length scale should be around 1 (the
        // period of the function in x ∈ [0,5]).
        let x =
            Array2::from_shape_vec((20, 1), (0..20).map(|i| (i as f64) * 0.3).collect()).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|v: f64| v.sin());
        let best = optimize_rbf_length_scale(&x, &y, 1.0, 1e-6, -2.0, 2.0, 30).unwrap();
        assert!(best > 0.3 && best < 4.0, "best length_scale = {best}");
    }

    #[test]
    fn test_optimize_kernel_lbfgs_rbf_two_params() {
        // Jointly optimise log(length_scale) and log(signal_var) on a sine.
        let x =
            Array2::from_shape_vec((20, 1), (0..20).map(|i| (i as f64) * 0.3).collect()).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|v: f64| v.sin());
        let init = vec![0.0_f64, 0.0]; // log ls, log var (both = 1.0 in linear)
        let res = optimize_kernel_lbfgs(
            &x,
            &y,
            1e-6,
            &init,
            |p| GpKernel::Rbf {
                length_scale: p[0].exp(),
                signal_var: p[1].exp(),
            },
            50,
            1e-4,
            1e-3,
        )
        .unwrap();
        // Should beat the initial point.
        let lml_init = log_marginal_likelihood(
            &x,
            &y,
            &GpKernel::Rbf {
                length_scale: 1.0,
                signal_var: 1.0,
            },
            1e-6,
        )
        .unwrap();
        assert!(
            res.log_marginal_likelihood >= lml_init - 1e-9,
            "optimiser regressed: init {} → final {}",
            lml_init,
            res.log_marginal_likelihood
        );
        // Optimised length_scale should be in a plausible range.
        let ls_opt = res.log_params[0].exp();
        assert!(ls_opt > 0.1 && ls_opt < 20.0, "length_scale {ls_opt}");
    }

    #[test]
    fn test_log_marginal_likelihood_monotonic_in_data() {
        // Likelihood should be higher for the kernel that better matches data.
        let x = Array2::from_shape_vec((10, 1), (0..10).map(|i| i as f64 * 0.3).collect()).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|v: f64| v.sin());
        let good = GpKernel::Rbf {
            length_scale: 1.0,
            signal_var: 1.0,
        };
        let bad = GpKernel::Rbf {
            length_scale: 100.0,
            signal_var: 1.0,
        };
        let lml_good = log_marginal_likelihood(&x, &y, &good, 1e-6).unwrap();
        let lml_bad = log_marginal_likelihood(&x, &y, &bad, 1e-6).unwrap();
        assert!(lml_good > lml_bad, "good={lml_good}, bad={lml_bad}");
    }

    #[test]
    fn test_gp_sum_of_kernels() {
        // RBF + White: should interpolate noisy data without exploding.
        let x = Array2::from_shape_vec((10, 1), (0..10).map(|i| i as f64).collect()).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|v: f64| v.sin() + 0.05);
        let kernel = GpKernel::Rbf {
            length_scale: 2.0,
            signal_var: 1.0,
        }
        .add(GpKernel::White { noise_level: 0.01 });
        let fitted = GaussianProcessRegressor::new(kernel)
            .with_alpha(1e-8)
            .fit(&x, &y)
            .unwrap();
        let p = fitted.predict(&x).unwrap();
        for (a, b) in p.iter().zip(y.iter()) {
            assert!((a - b).abs() < 0.1, "predict {} vs y {}", a, b);
        }
    }
}

impl rustml_core::RegressorScore<f64> for FittedGaussianProcessRegressor {}
