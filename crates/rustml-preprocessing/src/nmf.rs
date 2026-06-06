//! Non-negative Matrix Factorisation.
//!
//! Mirrors `sklearn.decomposition.NMF` with the multiplicative-update solver
//! (Lee & Seung). `X ≈ W H` with `W ≥ 0`, `H ≥ 0`.

use faer::linalg::solvers::Svd;
use faer::Mat;
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rustml_core::{FitUnsupervised, Result, RustMlError};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NmfInit {
    /// Sample W, H from uniform random.
    Random,
    /// NNDSVD: deterministic init from truncated SVD (sklearn default).
    Nndsvd,
}

#[derive(Debug, Clone)]
pub struct Nmf {
    pub n_components: usize,
    pub max_iter: usize,
    pub tol: f64,
    pub seed: u64,
    pub init: NmfInit,
}

impl Nmf {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            max_iter: 200,
            tol: 1e-4,
            seed: 0,
            init: NmfInit::Nndsvd,
        }
    }
    pub fn with_init(mut self, init: NmfInit) -> Self {
        self.init = init;
        self
    }
}

/// NNDSVD initialisation (Boutsidis & Gallopoulos 2008).
///
/// 1. Compute the truncated SVD `X ≈ U Σ Vᵀ` keeping `k` triplets.
/// 2. The first component is initialised from the leading singular triplet
///    with positive sign (sign-fix).
/// 3. Each subsequent component splits the next singular vectors into their
///    positive and negative parts and picks whichever has higher norm.
fn nndsvd_init(x: &Array2<f64>, k: usize) -> Result<(Array2<f64>, Array2<f64>)> {
    let n = x.nrows();
    let d = x.ncols();
    let mat = Mat::<f64>::from_fn(n, d, |i, j| x[[i, j]]);
    let svd = Svd::new(mat.as_ref())
        .map_err(|e| RustMlError::InvalidParameter(format!("NNDSVD SVD failed: {e:?}")))?;
    let u = svd.U();
    let s = svd.S();
    let v = svd.V();
    let r = s.column_vector().nrows().min(k);

    let mut w = Array2::<f64>::zeros((n, k));
    let mut h = Array2::<f64>::zeros((k, d));

    // First component: leading singular triplet (sign-fixed positive).
    let s0 = s.column_vector()[0].max(1e-12);
    let mut u0_pos_norm = 0.0_f64;
    for i in 0..n {
        u0_pos_norm += u[(i, 0)].max(0.0).powi(2);
    }
    u0_pos_norm = u0_pos_norm.sqrt();
    let mut v0_pos_norm = 0.0_f64;
    for j in 0..d {
        v0_pos_norm += v[(j, 0)].max(0.0).powi(2);
    }
    v0_pos_norm = v0_pos_norm.sqrt();
    // If positive part norm is larger, use it; else flip sign and use negative.
    let (u_sign, v_sign) =
        if u0_pos_norm * v0_pos_norm >= (u0_pos_norm * v0_pos_norm).max(1e-12) / 2.0 {
            (1.0, 1.0)
        } else {
            (-1.0, -1.0)
        };
    let lead_scale = s0.sqrt();
    for i in 0..n {
        w[[i, 0]] = (u_sign * u[(i, 0)]).max(0.0) * lead_scale;
    }
    for j in 0..d {
        h[[0, j]] = (v_sign * v[(j, 0)]).max(0.0) * lead_scale;
    }

    // Remaining components: split positive and negative parts.
    for c in 1..r {
        let sigma = s.column_vector()[c].max(1e-12);
        // u positive / negative parts.
        let mut up = vec![0.0_f64; n];
        let mut un = vec![0.0_f64; n];
        let mut up_norm = 0.0_f64;
        let mut un_norm = 0.0_f64;
        for i in 0..n {
            let val = u[(i, c)];
            if val > 0.0 {
                up[i] = val;
                up_norm += val * val;
            } else {
                un[i] = -val;
                un_norm += val * val;
            }
        }
        up_norm = up_norm.sqrt();
        un_norm = un_norm.sqrt();
        let mut vp = vec![0.0_f64; d];
        let mut vn = vec![0.0_f64; d];
        let mut vp_norm = 0.0_f64;
        let mut vn_norm = 0.0_f64;
        for j in 0..d {
            let val = v[(j, c)];
            if val > 0.0 {
                vp[j] = val;
                vp_norm += val * val;
            } else {
                vn[j] = -val;
                vn_norm += val * val;
            }
        }
        vp_norm = vp_norm.sqrt();
        vn_norm = vn_norm.sqrt();
        // Take whichever pair (positive/positive vs negative/negative) has
        // higher Frobenius product norm.
        let pos = up_norm * vp_norm;
        let neg = un_norm * vn_norm;
        let scale = sigma.sqrt() * (pos.max(neg)).sqrt();
        if pos >= neg {
            let nrm_u = up_norm.max(1e-12);
            let nrm_v = vp_norm.max(1e-12);
            for i in 0..n {
                w[[i, c]] = up[i] / nrm_u * scale;
            }
            for j in 0..d {
                h[[c, j]] = vp[j] / nrm_v * scale;
            }
        } else {
            let nrm_u = un_norm.max(1e-12);
            let nrm_v = vn_norm.max(1e-12);
            for i in 0..n {
                w[[i, c]] = un[i] / nrm_u * scale;
            }
            for j in 0..d {
                h[[c, j]] = vn[j] / nrm_v * scale;
            }
        }
    }
    // Floor at a small epsilon (sklearn convention) to avoid zero-locks.
    let eps = 1e-6;
    for v in w.iter_mut() {
        if *v < eps {
            *v = eps;
        }
    }
    for v in h.iter_mut() {
        if *v < eps {
            *v = eps;
        }
    }
    Ok((w, h))
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedNmf {
    /// Components, shape (n_components, n_features) — sklearn's H_.
    pub components: Array2<f64>,
    /// Final reconstruction error (Frobenius).
    pub reconstruction_err: f64,
    pub n_iter: usize,
}

impl FitUnsupervised<f64> for Nmf {
    type Fitted = FittedNmf;

    fn fit(&self, x: &Array2<f64>) -> Result<Self::Fitted> {
        let n = x.nrows();
        let d = x.ncols();
        let k = self.n_components;
        if n == 0 || d == 0 {
            return Err(RustMlError::EmptyInput("empty input".into()));
        }
        if k == 0 || k > d.min(n) {
            return Err(RustMlError::InvalidParameter(format!(
                "n_components must be in 1..={}",
                d.min(n)
            )));
        }
        // Require X ≥ 0.
        for v in x.iter() {
            if *v < 0.0 {
                return Err(RustMlError::InvalidParameter("NMF requires X >= 0".into()));
            }
        }

        let (mut w, mut h) = match self.init {
            NmfInit::Nndsvd => nndsvd_init(x, k)?,
            NmfInit::Random => {
                let mut rng = StdRng::seed_from_u64(self.seed);
                let scale = (x.mean().unwrap_or(0.0).max(0.0) / k as f64)
                    .sqrt()
                    .max(1e-6);
                let w = Array2::<f64>::from_shape_fn((n, k), |_| rng.gen::<f64>() * scale + 1e-6);
                let h = Array2::<f64>::from_shape_fn((k, d), |_| rng.gen::<f64>() * scale + 1e-6);
                (w, h)
            }
        };

        let mut prev_err = f64::INFINITY;
        let mut n_iter = 0;
        for iter in 0..self.max_iter {
            n_iter = iter + 1;

            // H update: H *= (W'X) / (W'W H)
            let wt_x = w.t().dot(x);
            let wt_w = w.t().dot(&w);
            let wt_w_h = wt_w.dot(&h);
            for a in 0..k {
                for b in 0..d {
                    h[[a, b]] *= wt_x[[a, b]] / wt_w_h[[a, b]].max(1e-12);
                }
            }
            // W update: W *= (X H') / (W H H')
            let h_ht = h.dot(&h.t());
            let x_ht = x.dot(&h.t());
            let w_h_ht = w.dot(&h_ht);
            for r in 0..n {
                for a in 0..k {
                    w[[r, a]] *= x_ht[[r, a]] / w_h_ht[[r, a]].max(1e-12);
                }
            }

            // Convergence check via reconstruction error.
            let recon = w.dot(&h);
            let mut err = 0.0;
            for r in 0..n {
                for c in 0..d {
                    let dv = x[[r, c]] - recon[[r, c]];
                    err += dv * dv;
                }
            }
            err = err.sqrt();
            if (prev_err - err).abs() / prev_err.max(1e-12) < self.tol {
                prev_err = err;
                break;
            }
            prev_err = err;
        }

        Ok(FittedNmf {
            components: h,
            reconstruction_err: prev_err,
            n_iter,
        })
    }
}

impl FittedNmf {
    /// Transform new data by solving `min_W >= 0  ||X - W H||²` via MU.
    pub fn transform(&self, x: &Array2<f64>, max_iter: usize) -> Result<Array2<f64>> {
        let h = &self.components;
        let n = x.nrows();
        let k = h.nrows();
        let mut rng = StdRng::seed_from_u64(7);
        let scale = (x.mean().unwrap_or(0.0).max(0.0) / k as f64)
            .sqrt()
            .max(1e-6);
        let mut w = Array2::<f64>::from_shape_fn((n, k), |_| rng.gen::<f64>() * scale + 1e-6);
        let h_ht = h.dot(&h.t());
        let x_ht = x.dot(&h.t());
        for _ in 0..max_iter {
            let w_h_ht = w.dot(&h_ht);
            for r in 0..n {
                for a in 0..k {
                    w[[r, a]] *= x_ht[[r, a]] / w_h_ht[[r, a]].max(1e-12);
                }
            }
        }
        Ok(w)
    }

    pub fn reconstruction_err(&self) -> f64 {
        self.reconstruction_err
    }
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }
    pub fn components(&self) -> &Array2<f64> {
        &self.components
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_nmf_recovers_low_rank() {
        // Construct X = W_true H_true with k=2.
        let w_true = array![[1.0_f64, 0.0], [2.0, 0.5], [0.0, 1.0], [0.3, 2.0]];
        let h_true = array![[1.0_f64, 2.0, 3.0], [0.5, 1.5, 0.5]];
        let x = w_true.dot(&h_true);
        let nmf = Nmf::new(2);
        let fitted = nmf.fit(&x).unwrap();
        let recon = nmf.max_iter.min(0); // suppress unused field warning
        let _ = recon;
        let recon = fitted.components.clone();
        // The transform should give us back something whose product is close.
        let w = fitted.transform(&x, 200).unwrap();
        let approx = w.dot(&recon);
        let mut err = 0.0;
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                err += (x[[i, j]] - approx[[i, j]]).powi(2);
            }
        }
        let rel = err.sqrt() / x.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(rel < 0.05, "rel reconstruction error too large: {rel}");
    }
}
