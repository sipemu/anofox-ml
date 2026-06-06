//! FastICA — fixed-point Independent Component Analysis with deflation.
//!
//! Mirrors `sklearn.decomposition.FastICA` with `algorithm='deflation'` and
//! `fun='logcosh'`. Standard pipeline:
//!
//! 1. Centre and whiten X via PCA so that `cov(X) = I`.
//! 2. For each component, iterate the fixed-point update
//!    `w ← E[X g(wᵀX)] - E[g'(wᵀX)] w`, orthogonalise against previously
//!    extracted components, normalise to unit length.
//! 3. The sources `S = W X_white`; the unmixing matrix in original space is
//!    `W K` where `K` is the whitening matrix.

use anofox_ml_core::{FitUnsupervised, Result, RustMlError, Transform};
use faer::linalg::solvers::Svd;
use faer::Mat;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Debug, Clone)]
pub struct FastIca {
    pub n_components: usize,
    pub max_iter: usize,
    pub tol: f64,
    pub seed: u64,
}

impl FastIca {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            max_iter: 200,
            tol: 1e-4,
            seed: 0,
        }
    }
    pub fn with_max_iter(mut self, m: usize) -> Self {
        self.max_iter = m;
        self
    }
    pub fn with_tol(mut self, t: f64) -> Self {
        self.tol = t;
        self
    }
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedFastIca {
    /// Unmixing matrix from whitened space, shape (n_components, n_components).
    pub w: Array2<f64>,
    /// Whitening matrix `K` such that `X_centered @ K` is whitened, shape
    /// (n_features, n_components).
    pub whitening: Array2<f64>,
    /// Per-feature mean used for centring.
    pub mean: Array1<f64>,
    pub n_features: usize,
}

/// Logcosh non-linearity: `g(u) = tanh(u)`, `g'(u) = 1 - tanh(u)²`.
fn g_logcosh(u: f64) -> (f64, f64) {
    let t = u.tanh();
    (t, 1.0 - t * t)
}

impl FitUnsupervised<f64> for FastIca {
    type Fitted = FittedFastIca;

    fn fit(&self, x: &Array2<f64>) -> Result<Self::Fitted> {
        let n = x.nrows();
        let d = x.ncols();
        let k = self.n_components.min(d.min(n));
        if k == 0 {
            return Err(RustMlError::InvalidParameter("n_components >= 1".into()));
        }
        if n < 2 {
            return Err(RustMlError::EmptyInput("need at least 2 samples".into()));
        }

        // 1. Centre.
        let mut mean = Array1::<f64>::zeros(d);
        for j in 0..d {
            mean[j] = x.column(j).sum() / n as f64;
        }
        let mut xc = x.clone();
        for j in 0..d {
            for i in 0..n {
                xc[[i, j]] -= mean[j];
            }
        }

        // 2. Whiten via SVD: X_centered = U Σ Vᵀ. Whitening matrix K = V Σ⁻¹ √(n-1).
        let xm = Mat::from_fn(n, d, |i, j| xc[[i, j]]);
        let svd = Svd::new(xm.as_ref())
            .map_err(|e| RustMlError::InvalidParameter(format!("SVD failed: {e:?}")))?;
        let s = svd.S();
        let v = svd.V();
        let scale = (n as f64 - 1.0).sqrt();
        let mut k_white = Array2::<f64>::zeros((d, k));
        for c in 0..k {
            let sigma = s.column_vector()[c].max(1e-12);
            for j in 0..d {
                k_white[[j, c]] = v[(j, c)] * scale / sigma;
            }
        }
        // Whitened data: X1 = X_centered @ K, shape (n, k).
        let x1 = xc.dot(&k_white);

        // 3. Deflation extraction.
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut w = Array2::<f64>::zeros((k, k));
        for comp in 0..k {
            // Random init.
            let mut wi: Array1<f64> = Array1::from_shape_fn(k, |_| rng.gen::<f64>() * 2.0 - 1.0);
            // Normalize.
            let nrm = wi.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
            wi.mapv_inplace(|v| v / nrm);

            for _ in 0..self.max_iter {
                // Compute u = X1 @ wi (length n).
                let mut u = vec![0.0_f64; n];
                for i in 0..n {
                    let mut s = 0.0;
                    for c in 0..k {
                        s += x1[[i, c]] * wi[c];
                    }
                    u[i] = s;
                }
                // g(u) and g'(u).
                let mut gu = vec![0.0_f64; n];
                let mut g_prime_mean = 0.0_f64;
                for i in 0..n {
                    let (g, gp) = g_logcosh(u[i]);
                    gu[i] = g;
                    g_prime_mean += gp;
                }
                g_prime_mean /= n as f64;
                // New wi: E[X1 g(wᵀ X1)] - E[g'(...)] w.
                let mut new_wi = Array1::<f64>::zeros(k);
                for c in 0..k {
                    let mut s = 0.0;
                    for i in 0..n {
                        s += x1[[i, c]] * gu[i];
                    }
                    new_wi[c] = s / n as f64 - g_prime_mean * wi[c];
                }
                // Deflate: orthogonalise against previously-extracted components.
                for prev in 0..comp {
                    let mut dot = 0.0;
                    for c in 0..k {
                        dot += new_wi[c] * w[[prev, c]];
                    }
                    for c in 0..k {
                        new_wi[c] -= dot * w[[prev, c]];
                    }
                }
                // Normalise.
                let nrm = new_wi.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
                new_wi.mapv_inplace(|v| v / nrm);

                // Convergence: |1 - |<w_new, w_old>||.
                let mut dot = 0.0;
                for c in 0..k {
                    dot += new_wi[c] * wi[c];
                }
                let conv = (1.0 - dot.abs()).abs();
                wi = new_wi;
                if conv < self.tol {
                    break;
                }
            }
            for c in 0..k {
                w[[comp, c]] = wi[c];
            }
        }

        Ok(FittedFastIca {
            w,
            whitening: k_white,
            mean,
            n_features: d,
        })
    }
}

impl Transform<f64> for FittedFastIca {
    /// Returns the recovered source signals `S = (X - mean) · K · Wᵀ`.
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        let mut xc = x.clone();
        for j in 0..self.n_features {
            for i in 0..x.nrows() {
                xc[[i, j]] -= self.mean[j];
            }
        }
        let x_white = xc.dot(&self.whitening);
        Ok(x_white.dot(&self.w.t()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_fast_ica_runs() {
        // Build a mixture of two simple signals; FastICA should separate them.
        let n = 100;
        let mut s = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let t = i as f64 * 0.1;
            s[[i, 0]] = t.sin(); // sine
            s[[i, 1]] = (t * 0.3).sin().signum(); // square
        }
        // Mixing matrix.
        let a = array![[1.0_f64, 0.5], [0.5, 1.0]];
        let x = s.dot(&a);
        let fitted = FastIca::new(2).with_seed(1).fit(&x).unwrap();
        let recovered = fitted.transform(&x).unwrap();
        assert_eq!(recovered.shape(), &[n, 2]);
        for v in recovered.iter() {
            assert!(v.is_finite());
        }
    }
}
