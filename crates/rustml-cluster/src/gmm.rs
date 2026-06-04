//! Gaussian Mixture Model with EM training.
//!
//! Mirrors `sklearn.mixture.GaussianMixture` (`covariance_type='full'` or
//! `'diag'`). Initialization is k-means++.

use faer::linalg::solvers::Solve;
use faer::{Mat, Side};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rustml_core::{FitUnsupervised, Predict, PredictProba, Result, RustMlError};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CovarianceType {
    Full,
    Diag,
}

#[derive(Debug, Clone)]
pub struct GaussianMixture {
    pub n_components: usize,
    pub covariance_type: CovarianceType,
    pub max_iter: usize,
    pub tol: f64,
    pub reg_covar: f64,
    pub seed: u64,
    /// Number of independent random restarts (sklearn default 1).
    pub n_init: usize,
}

impl GaussianMixture {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            covariance_type: CovarianceType::Full,
            max_iter: 100,
            tol: 1e-3,
            reg_covar: 1e-6,
            seed: 0,
            n_init: 1,
        }
    }
    pub fn with_covariance_type(mut self, c: CovarianceType) -> Self { self.covariance_type = c; self }
    pub fn with_max_iter(mut self, m: usize) -> Self { self.max_iter = m; self }
    pub fn with_seed(mut self, s: u64) -> Self { self.seed = s; self }
    pub fn with_n_init(mut self, n: usize) -> Self { self.n_init = n.max(1); self }
}

#[derive(Debug, Clone)]
pub struct FittedGaussianMixture {
    pub weights: Array1<f64>,
    pub means: Array2<f64>,      // k × d
    /// Stored as either k×d (diag) or k row-major d×d (full).
    pub covariances: Vec<Array2<f64>>,
    pub log_likelihood: f64,
    pub n_iter: usize,
    pub covariance_type: CovarianceType,
}

fn pairwise_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

fn kmeans_pp_init(x: &Array2<f64>, k: usize, rng: &mut StdRng) -> Array2<f64> {
    let n = x.nrows();
    let d = x.ncols();
    let mut centers = Array2::<f64>::zeros((k, d));
    let first = rng.gen_range(0..n);
    centers.row_mut(0).assign(&x.row(first));
    let mut min_d = vec![f64::INFINITY; n];
    for c in 1..k {
        for i in 0..n {
            let sd = pairwise_sq(
                x.row(i).as_slice().unwrap(),
                centers.row(c - 1).as_slice().unwrap(),
            );
            if sd < min_d[i] {
                min_d[i] = sd;
            }
        }
        let total: f64 = min_d.iter().sum();
        if total == 0.0 {
            centers.row_mut(c).assign(&x.row(rng.gen_range(0..n)));
            continue;
        }
        let r = rng.gen::<f64>() * total;
        let mut cum = 0.0;
        let mut pick = n - 1;
        for i in 0..n {
            cum += min_d[i];
            if cum >= r {
                pick = i;
                break;
            }
        }
        centers.row_mut(c).assign(&x.row(pick));
    }
    centers
}

fn log_gauss(diff: &[f64], cov_or_diag: &Array2<f64>, cov_type: CovarianceType) -> f64 {
    let d = diff.len();
    match cov_type {
        CovarianceType::Diag => {
            // cov_or_diag is 1 × d.
            let mut q = 0.0;
            let mut logdet = 0.0;
            for j in 0..d {
                let v = cov_or_diag[[0, j]].max(1e-30);
                q += diff[j] * diff[j] / v;
                logdet += v.ln();
            }
            -0.5 * (q + logdet + d as f64 * (2.0 * std::f64::consts::PI).ln())
        }
        CovarianceType::Full => {
            let nd = d;
            let m = Mat::from_fn(nd, nd, |i, j| cov_or_diag[[i, j]]);
            let llt = match faer::linalg::solvers::Llt::new(m.as_ref(), Side::Lower) {
                Ok(l) => l,
                Err(_) => return f64::NEG_INFINITY,
            };
            let lower = llt.L();
            let mut logdet = 0.0;
            for i in 0..nd {
                logdet += lower[(i, i)].abs().ln();
            }
            let logdet = 2.0 * logdet;
            let bm = Mat::from_fn(nd, 1, |i, _| diff[i]);
            let sol = llt.solve(&bm);
            let mut q = 0.0;
            for i in 0..nd {
                q += diff[i] * sol[(i, 0)];
            }
            -0.5 * (q + logdet + nd as f64 * (2.0 * std::f64::consts::PI).ln())
        }
    }
}

fn logsumexp(v: &[f64]) -> f64 {
    let m = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if m == f64::NEG_INFINITY {
        return m;
    }
    let s: f64 = v.iter().map(|x| (x - m).exp()).sum();
    m + s.ln()
}

impl FitUnsupervised<f64> for GaussianMixture {
    type Fitted = FittedGaussianMixture;

    fn fit(&self, x: &Array2<f64>) -> Result<Self::Fitted> {
        let n = x.nrows();
        let _d = x.ncols();
        let k = self.n_components;
        if n == 0 {
            return Err(RustMlError::EmptyInput("empty input".into()));
        }
        if k == 0 || k > n {
            return Err(RustMlError::InvalidParameter("invalid n_components".into()));
        }

        // n_init restarts: pick the fit with highest log-likelihood.
        let mut best: Option<FittedGaussianMixture> = None;
        for restart in 0..self.n_init {
            let fitted = single_fit(self, x, self.seed.wrapping_add(restart as u64))?;
            match &best {
                None => best = Some(fitted),
                Some(b) if fitted.log_likelihood > b.log_likelihood => best = Some(fitted),
                _ => {}
            }
        }
        Ok(best.unwrap())
    }
}

fn single_fit(
    cfg: &GaussianMixture,
    x: &Array2<f64>,
    seed: u64,
) -> Result<FittedGaussianMixture> {
    let n = x.nrows();
    let d = x.ncols();
    let k = cfg.n_components;
    {
        // Original single-run body follows; minimally rewritten to take seed.
        let mut rng = StdRng::seed_from_u64(seed);
        let mut means = kmeans_pp_init(x, k, &mut rng);

        // Init covariances: empirical full or diagonal.
        let mut covariances: Vec<Array2<f64>> = (0..k)
            .map(|_| match cfg.covariance_type {
                CovarianceType::Diag => Array2::<f64>::ones((1, d)),
                CovarianceType::Full => Array2::<f64>::eye(d),
            })
            .collect();
        let mut weights = Array1::<f64>::from_elem(k, 1.0 / k as f64);

        let mut prev_ll = f64::NEG_INFINITY;
        let mut n_iter = 0;
        let mut log_resp = Array2::<f64>::zeros((n, k));
        let mut log_ll = f64::NEG_INFINITY;

        for iter in 0..cfg.max_iter {
            n_iter = iter + 1;
            // E-step: also accumulate log-likelihood from the same lse values
            // we compute for responsibilities — no second pass.
            let mut total_ll = 0.0_f64;
            for i in 0..n {
                let xi = x.row(i).to_owned();
                let mut logs = vec![0.0; k];
                for c in 0..k {
                    let mut diff = vec![0.0; d];
                    for j in 0..d {
                        diff[j] = xi[j] - means[[c, j]];
                    }
                    logs[c] = weights[c].ln() + log_gauss(&diff, &covariances[c], cfg.covariance_type);
                }
                let lse = logsumexp(&logs);
                total_ll += lse;
                for c in 0..k {
                    log_resp[[i, c]] = logs[c] - lse;
                }
            }
            log_ll = total_ll / n as f64;

            // M-step
            let nk: Vec<f64> = (0..k)
                .map(|c| (0..n).map(|i| log_resp[[i, c]].exp()).sum())
                .collect();
            for c in 0..k {
                let nkc = nk[c].max(1e-12);
                weights[c] = nkc / n as f64;
                // Update mean.
                for j in 0..d {
                    let mut s = 0.0;
                    for i in 0..n {
                        s += log_resp[[i, c]].exp() * x[[i, j]];
                    }
                    means[[c, j]] = s / nkc;
                }
                // Update covariance.
                match cfg.covariance_type {
                    CovarianceType::Diag => {
                        let mut diag = Array2::<f64>::zeros((1, d));
                        for j in 0..d {
                            let mu = means[[c, j]];
                            let mut s = 0.0;
                            for i in 0..n {
                                let r = log_resp[[i, c]].exp();
                                let dv = x[[i, j]] - mu;
                                s += r * dv * dv;
                            }
                            diag[[0, j]] = s / nkc + cfg.reg_covar;
                        }
                        covariances[c] = diag;
                    }
                    CovarianceType::Full => {
                        let mut sigma = Array2::<f64>::zeros((d, d));
                        for a in 0..d {
                            for b in 0..d {
                                let mut s = 0.0;
                                for i in 0..n {
                                    let r = log_resp[[i, c]].exp();
                                    let da = x[[i, a]] - means[[c, a]];
                                    let db = x[[i, b]] - means[[c, b]];
                                    s += r * da * db;
                                }
                                sigma[[a, b]] = s / nkc;
                            }
                            sigma[[a, a]] += cfg.reg_covar;
                        }
                        covariances[c] = sigma;
                    }
                }
            }

            if (log_ll - prev_ll).abs() < cfg.tol {
                break;
            }
            prev_ll = log_ll;
        }

        Ok(FittedGaussianMixture {
            weights,
            means,
            covariances,
            log_likelihood: log_ll,
            n_iter,
            covariance_type: cfg.covariance_type,
        })
    }
}

impl Predict<f64> for FittedGaussianMixture {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let k = self.weights.len();
        let d = self.means.ncols();
        if x.ncols() != d {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}", d, x.ncols()
            )));
        }
        let n = x.nrows();
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let xi = x.row(i).to_owned();
            let mut best = f64::NEG_INFINITY;
            let mut best_c = 0usize;
            for c in 0..k {
                let mut diff = vec![0.0; d];
                for j in 0..d {
                    diff[j] = xi[j] - self.means[[c, j]];
                }
                let s = self.weights[c].ln()
                    + log_gauss(&diff, &self.covariances[c], self.covariance_type);
                if s > best {
                    best = s;
                    best_c = c;
                }
            }
            out[i] = best_c as f64;
        }
        Ok(out)
    }
}

impl PredictProba<f64> for FittedGaussianMixture {
    fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let k = self.weights.len();
        let d = self.means.ncols();
        if x.ncols() != d {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}", d, x.ncols()
            )));
        }
        let n = x.nrows();
        let mut p = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            let xi = x.row(i).to_owned();
            let mut logs = vec![0.0_f64; k];
            for c in 0..k {
                let mut diff = vec![0.0; d];
                for j in 0..d {
                    diff[j] = xi[j] - self.means[[c, j]];
                }
                logs[c] = self.weights[c].ln()
                    + log_gauss(&diff, &self.covariances[c], self.covariance_type);
            }
            let max_l = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut z = 0.0;
            for c in 0..k {
                let e = (logs[c] - max_l).exp();
                p[[i, c]] = e;
                z += e;
            }
            for c in 0..k {
                p[[i, c]] /= z;
            }
        }
        Ok(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_gmm_two_well_separated_blobs() {
        let x = array![
            [0.0_f64, 0.0], [0.2, 0.1], [-0.1, 0.2], [0.1, -0.2],
            [10.0, 10.0], [10.1, 9.9], [9.8, 10.2], [10.2, 9.8],
        ];
        let fitted = GaussianMixture::new(2)
            .with_seed(0)
            .fit(&x)
            .unwrap();
        let labels = fitted.predict(&x).unwrap();
        let l0 = labels[0];
        for i in 1..4 {
            assert_eq!(labels[i], l0);
        }
        for i in 4..8 {
            assert_ne!(labels[i], l0);
        }
    }
}
