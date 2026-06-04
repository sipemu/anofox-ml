//! Gaussian Process Regression.
//!
//! Mirrors `sklearn.gaussian_process.GaussianProcessRegressor` with a fixed
//! RBF kernel `k(x,x') = exp(-||x-x'||² / (2ℓ²))`. Noise level `alpha` is
//! added to the diagonal of the kernel for numerical stability and to allow
//! noisy observations.
//!
//! Predictions are the posterior mean; `predict_std` returns the posterior
//! standard deviation per query point.

use faer::linalg::solvers::Solve;
use faer::{Mat, Side};
use ndarray::{Array1, Array2};
use rustml_core::{Fit, Predict, Result, RustMlError};

pub enum GpKernel {
    /// `σ² exp(-||x-x'||² / (2ℓ²))`.
    Rbf { length_scale: f64, signal_var: f64 },
}

impl GpKernel {
    fn compute(&self, a: &[f64], b: &[f64]) -> f64 {
        match self {
            GpKernel::Rbf { length_scale, signal_var } => {
                let sd: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
                signal_var * (-0.5 * sd / (length_scale * length_scale)).exp()
            }
        }
    }
}

pub struct GaussianProcessRegressor {
    pub kernel: GpKernel,
    pub alpha: f64,
    pub normalize_y: bool,
}

impl GaussianProcessRegressor {
    pub fn new(kernel: GpKernel) -> Self {
        Self { kernel, alpha: 1e-10, normalize_y: false }
    }
    pub fn with_alpha(mut self, a: f64) -> Self { self.alpha = a; self }
    pub fn with_normalize_y(mut self, v: bool) -> Self { self.normalize_y = v; self }
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

fn build_gram(x_a: &Array2<f64>, x_b: &Array2<f64>, kernel: &GpKernel) -> Array2<f64> {
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

impl Fit<f64> for GaussianProcessRegressor {
    type Fitted = FittedGaussianProcessRegressor;

    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        let n = x.nrows();
        if x.nrows() != y.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but y has {}", x.nrows(), y.len()
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
            kernel: match self.kernel {
                GpKernel::Rbf { length_scale, signal_var } => GpKernel::Rbf {
                    length_scale,
                    signal_var,
                },
            },
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
        // Solve L v = k_star' (one rhs per query point).
        // Compute K(x_new, x_new) diagonal and then σ² - ||v||² per point.
        let mut std_out = Array1::<f64>::zeros(n_new);
        for i in 0..n_new {
            let rhs = Mat::from_fn(n_train, 1, |j, _| k_star[[i, j]]);
            // Forward solve: L v = rhs.
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
    fn test_gp_interpolates_with_low_noise() {
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
        let s = fitted.predict_std(&x).unwrap();
        for &v in s.iter() {
            assert!(v < 1e-3, "std too large at training point: {v}");
        }
        let _ = array![1.0_f64];
    }
}
