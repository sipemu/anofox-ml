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
            "X has {} rows but y has {}", n, y.len()
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
            let k_xx = self.kernel.compute(xi.as_slice().unwrap(), xi.as_slice().unwrap());
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
        let kernel = GpKernel::Rbf { length_scale: 1.0, signal_var: 1.0 };
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
        let kernel = GpKernel::Matern { length_scale: 1.0, signal_var: 1.0, nu: 2.5 };
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
        let x = Array2::from_shape_vec((20, 1), (0..20).map(|i| (i as f64) * 0.3).collect()).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|v: f64| v.sin());
        let best = optimize_rbf_length_scale(&x, &y, 1.0, 1e-6, -2.0, 2.0, 30).unwrap();
        assert!(best > 0.3 && best < 4.0, "best length_scale = {best}");
    }

    #[test]
    fn test_log_marginal_likelihood_monotonic_in_data() {
        // Likelihood should be higher for the kernel that better matches data.
        let x = Array2::from_shape_vec((10, 1), (0..10).map(|i| i as f64 * 0.3).collect()).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|v: f64| v.sin());
        let good = GpKernel::Rbf { length_scale: 1.0, signal_var: 1.0 };
        let bad = GpKernel::Rbf { length_scale: 100.0, signal_var: 1.0 };
        let lml_good = log_marginal_likelihood(&x, &y, &good, 1e-6).unwrap();
        let lml_bad = log_marginal_likelihood(&x, &y, &bad, 1e-6).unwrap();
        assert!(lml_good > lml_bad, "good={lml_good}, bad={lml_bad}");
    }

    #[test]
    fn test_gp_sum_of_kernels() {
        // RBF + White: should interpolate noisy data without exploding.
        let x = Array2::from_shape_vec((10, 1), (0..10).map(|i| i as f64).collect()).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|v: f64| v.sin() + 0.05);
        let kernel = GpKernel::Rbf { length_scale: 2.0, signal_var: 1.0 }
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
