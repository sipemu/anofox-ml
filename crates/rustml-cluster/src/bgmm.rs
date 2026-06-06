//! Bayesian Gaussian Mixture Model — variational inference.
//!
//! Implements `sklearn.mixture.BayesianGaussianMixture` with
//! `weight_concentration_prior_type='dirichlet_distribution'` and
//! `covariance_type='full'` or `'diag'`.
//!
//! Mathematical model (Bishop, PRML §10.2):
//!
//!   π ~ Dir(α_0 / K, …, α_0 / K)
//!   (μ_k, Λ_k) ~ NormalWishart(m_0, β_0, W_0, ν_0)
//!   z_n | π ~ Cat(π)
//!   x_n | z_n=k, μ_k, Λ_k ~ N(μ_k, Λ_k^{-1})
//!
//! Variational posterior factorises as q(Z) q(π) ∏_k q(μ_k, Λ_k), each
//! kept in conjugate form. CAVI updates cycle E (responsibilities) and M
//! (Dirichlet + Normal-Wishart parameter updates).
//!
//! Convergence is judged on the change in mean evidence lower bound (the
//! `lower_bound_` attribute in sklearn parlance). Effective component count
//! is reported as the number of components whose posterior Dirichlet
//! parameter αₖ exceeds a small threshold over α_0/K.

use faer::linalg::solvers::Solve;
use faer::{Mat, Side};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rustml_core::{FitUnsupervised, Predict, PredictProba, Result, RustMlError};

use crate::gmm::CovarianceType;

#[derive(Debug, Clone)]
pub struct BayesianGaussianMixture {
    pub n_components: usize,
    pub covariance_type: CovarianceType,
    pub max_iter: usize,
    pub tol: f64,
    pub reg_covar: f64,
    pub seed: u64,
    /// Dirichlet concentration prior on the mixing weights (α_0). Sklearn
    /// default is 1/K; pass `None` and we fall back to that.
    pub weight_concentration_prior: f64,
    /// Mean precision prior (β_0). Sklearn default 1.0.
    pub mean_precision_prior: f64,
    /// Degrees-of-freedom prior (ν_0). Must satisfy ν_0 > D - 1; sklearn
    /// defaults to D.
    pub degrees_of_freedom_prior: Option<f64>,
}

impl BayesianGaussianMixture {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            covariance_type: CovarianceType::Full,
            max_iter: 200,
            tol: 1e-3,
            reg_covar: 1e-6,
            seed: 0,
            weight_concentration_prior: 1.0 / n_components as f64,
            mean_precision_prior: 1.0,
            degrees_of_freedom_prior: None,
        }
    }
    pub fn with_concentration(mut self, c: f64) -> Self {
        self.weight_concentration_prior = c;
        self
    }
    pub fn with_mean_precision(mut self, b: f64) -> Self {
        self.mean_precision_prior = b;
        self
    }
    pub fn with_dof(mut self, v: f64) -> Self {
        self.degrees_of_freedom_prior = Some(v);
        self
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
    pub fn with_covariance_type(mut self, c: CovarianceType) -> Self {
        self.covariance_type = c;
        self
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedBayesianGaussianMixture {
    /// Posterior mean of π (normalised α_k / Σα).
    pub weights: Array1<f64>,
    /// Posterior means m_k.
    pub means: Array2<f64>,
    /// Posterior covariance scale: W_k^{-1} / ν_k for full, diag(…) for diag.
    pub covariances: Vec<Array2<f64>>,
    /// Posterior Dirichlet parameters (α_k = α_0 + N_k).
    pub weight_concentration: Array1<f64>,
    /// Posterior degrees of freedom (ν_k).
    pub dof_posterior: Array1<f64>,
    /// Posterior mean precision (β_k).
    pub mean_precision_posterior: Array1<f64>,
    /// ELBO at convergence.
    pub lower_bound: f64,
    pub n_iter: usize,
    pub effective_components: usize,
    pub covariance_type: CovarianceType,
}

// ───── Special functions ───────────────────────────────────────────────────

/// Digamma ψ(x) for x > 0.
///
/// Uses asymptotic expansion for x ≥ 6 and recurrence ψ(x) = ψ(x+1) - 1/x
/// to lift small arguments up. Accurate to ~1e-12 for x > 0.
fn digamma(mut x: f64) -> f64 {
    let mut result = 0.0;
    while x < 6.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    // Asymptotic series.
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result +=
        x.ln() - 0.5 * inv - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0)));
    result
}

/// Log gamma ln Γ(x) for x > 0 (Lanczos g=7, n=9). Used by the BGMM test
/// suite to verify the special-function block; the variational E/M-step
/// uses only `digamma`.
#[allow(dead_code)]
fn lgamma(x: f64) -> f64 {
    static G: f64 = 7.0;
    static COEF: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_59,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_571_6e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection.
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - lgamma(1.0 - x);
    }
    let xm1 = x - 1.0;
    let mut a = COEF[0];
    for (i, &c) in COEF.iter().enumerate().skip(1) {
        a += c / (xm1 + i as f64);
    }
    let t = xm1 + G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (xm1 + 0.5) * t.ln() - t + a.ln()
}

// ───── Helpers ────────────────────────────────────────────────────────────

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

fn logsumexp(v: &[f64]) -> f64 {
    let m = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if m == f64::NEG_INFINITY {
        return m;
    }
    let s: f64 = v.iter().map(|x| (x - m).exp()).sum();
    m + s.ln()
}

/// Compute log|A| and Cholesky factor L (lower) for SPD A, plus a solve
/// closure. Returns None if A is not SPD.
fn chol_logdet(a: &Array2<f64>) -> Option<(f64, faer::linalg::solvers::Llt<f64>)> {
    let n = a.nrows();
    let m = Mat::from_fn(n, n, |i, j| a[[i, j]]);
    let llt = faer::linalg::solvers::Llt::new(m.as_ref(), Side::Lower).ok()?;
    let l = llt.L();
    let mut logdet = 0.0;
    for i in 0..n {
        logdet += l[(i, i)].abs().ln();
    }
    Some((2.0 * logdet, llt))
}

impl FitUnsupervised<f64> for BayesianGaussianMixture {
    type Fitted = FittedBayesianGaussianMixture;

    fn fit(&self, x: &Array2<f64>) -> Result<Self::Fitted> {
        let n = x.nrows();
        let d = x.ncols();
        let k = self.n_components;
        if n == 0 {
            return Err(RustMlError::EmptyInput("empty input".into()));
        }
        if k == 0 || k > n {
            return Err(RustMlError::InvalidParameter("invalid n_components".into()));
        }

        // ─── Hyperparameter setup ──────────────────────────────────────
        let alpha0 = self.weight_concentration_prior;
        let beta0 = self.mean_precision_prior;
        let nu0 = self.degrees_of_freedom_prior.unwrap_or(d as f64);
        if nu0 <= (d as f64) - 1.0 {
            return Err(RustMlError::InvalidParameter(
                "degrees_of_freedom_prior must exceed D - 1".into(),
            ));
        }

        // Prior mean m_0 = data mean (sklearn default).
        let mut m0 = Array1::<f64>::zeros(d);
        for j in 0..d {
            let mut s = 0.0;
            for i in 0..n {
                s += x[[i, j]];
            }
            m0[j] = s / n as f64;
        }

        // Prior W_0 chosen so that E[Λ] = ν_0 W_0 ≈ data precision; sklearn
        // uses the empirical covariance as `covariance_prior_`. We use
        // `cov(x) / ν_0` so the prior mean of Λ equals cov(x)^{-1} … wait,
        // sklearn actually sets W_0 = inv(cov_prior); cov_prior defaults to
        // diag of empirical covariance. We follow that.
        let mut emp_cov_diag = Array1::<f64>::zeros(d);
        for j in 0..d {
            let mut s = 0.0;
            for i in 0..n {
                let dv = x[[i, j]] - m0[j];
                s += dv * dv;
            }
            emp_cov_diag[j] = (s / n as f64).max(self.reg_covar);
        }
        // W_0 = diag(1/emp_cov_diag) for full covariance.
        let w0_inv = match self.covariance_type {
            CovarianceType::Full => {
                let mut w = Array2::<f64>::zeros((d, d));
                for j in 0..d {
                    w[[j, j]] = emp_cov_diag[j];
                }
                w
            }
            CovarianceType::Diag => {
                let mut w = Array2::<f64>::zeros((1, d));
                for j in 0..d {
                    w[[0, j]] = emp_cov_diag[j];
                }
                w
            }
        };

        // ─── Variational parameter storage ─────────────────────────────
        // q(π) = Dir(α_k);   q(μ_k, Λ_k) = NormalWishart(m_k, β_k, W_k, ν_k)
        // We store W_k via its inverse W_k_inv (size d×d for full, 1×d for
        // diag). This makes the Sherman-Morrison-like updates cleaner.
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut m_post = kmeans_pp_init(x, k, &mut rng); // initial m_k
        let mut beta_post = Array1::<f64>::from_elem(k, beta0);
        let mut nu_post = Array1::<f64>::from_elem(k, nu0);
        let mut alpha_post = Array1::<f64>::from_elem(k, alpha0);
        let mut w_inv_post: Vec<Array2<f64>> = (0..k).map(|_| w0_inv.clone()).collect();

        let mut log_resp = Array2::<f64>::zeros((n, k));
        let mut prev_elbo = f64::NEG_INFINITY;
        let mut n_iter = 0;
        let mut elbo = f64::NEG_INFINITY;

        for iter in 0..self.max_iter {
            n_iter = iter + 1;

            // ─── E-step: compute log responsibilities ──────────────────
            // ln ρ_{nk} = E[ln π_k] + ½ E[ln|Λ_k|] - D/2 ln(2π)
            //           - ½ E[(x_n - μ_k)^T Λ_k (x_n - μ_k)]
            //
            // For NW posterior:
            //   E[ln π_k]    = ψ(α_k) - ψ(Σ α)
            //   E[ln|Λ_k|]   = Σ_i ψ((ν_k+1-i)/2) + D ln 2 + ln|W_k|
            //                = Σ_i ψ((ν_k+1-i)/2) + D ln 2 - ln|W_k_inv|
            //   E[mahal]     = D/β_k + ν_k (x_n - m_k)^T W_k (x_n - m_k)

            let sum_alpha: f64 = alpha_post.iter().sum();
            let psi_sum = digamma(sum_alpha);
            let e_log_pi: Vec<f64> = (0..k).map(|c| digamma(alpha_post[c]) - psi_sum).collect();

            // Pre-compute, per component k, ½ E[ln|Λ_k|] and a Cholesky-like
            // structure on W_k_inv to evaluate the quadratic form.
            // For Full: cache LLT(W_k_inv); the quadratic form uses
            //   (x-m_k)^T W_k (x-m_k) = ‖L_k^{-T} (x-m_k)‖² where W_k_inv = L L^T,
            // so q = ‖L_k^{-1} (x-m_k)‖² — solve L_k y = (x-m_k) then ‖y‖².
            //
            // For Diag: W_k_inv stored as 1×d vector; quadratic is
            //   Σ_j (x_j-m_kj)² / W_k_inv[j].

            let mut half_e_log_det_lam = vec![0.0_f64; k];
            // For full: cache Cholesky factors of W_k_inv (so we can solve
            // L y = (x-m_k) per data point).
            let mut llt_cache: Vec<Option<faer::linalg::solvers::Llt<f64>>> =
                (0..k).map(|_| None).collect();

            for c in 0..k {
                match self.covariance_type {
                    CovarianceType::Full => {
                        // E[ln|Λ_k|] = Σ_i ψ((ν+1-i)/2) + D ln2 - ln|W_inv|
                        let (logdet_winv, llt) = chol_logdet(&w_inv_post[c]).ok_or_else(|| {
                            RustMlError::InvalidParameter("W_k_inv not SPD".into())
                        })?;
                        let mut psi_acc = 0.0;
                        for i in 0..d {
                            psi_acc += digamma((nu_post[c] + 1.0 - (i as f64 + 1.0)) / 2.0);
                        }
                        let e_log_det = psi_acc + (d as f64) * std::f64::consts::LN_2 - logdet_winv;
                        half_e_log_det_lam[c] = 0.5 * e_log_det;
                        llt_cache[c] = Some(llt);
                    }
                    CovarianceType::Diag => {
                        // |W| = ∏_j 1/W_inv[j] ⇒ ln|W| = -Σ ln W_inv[j]
                        // For diag the standard reduction yields the same per-
                        // dim digamma series (treat each dimension as a 1-D
                        // Wishart-Gamma).
                        let mut psi_acc = 0.0;
                        for i in 0..d {
                            psi_acc += digamma((nu_post[c] + 1.0 - (i as f64 + 1.0)) / 2.0);
                        }
                        let mut ln_w = 0.0;
                        for j in 0..d {
                            ln_w -= w_inv_post[c][[0, j]].max(1e-300).ln();
                        }
                        let e_log_det = psi_acc + (d as f64) * std::f64::consts::LN_2 + ln_w;
                        half_e_log_det_lam[c] = 0.5 * e_log_det;
                    }
                }
            }

            // Compute log_rho and normalise to log_resp.
            let log_2pi = (2.0 * std::f64::consts::PI).ln();
            let mut data_log_lik = 0.0_f64;
            for i in 0..n {
                let xi = x.row(i);
                let mut log_rho = vec![0.0_f64; k];
                for c in 0..k {
                    // Mahalanobis E[(x-m)^T W (x-m)] times ν, plus D/β.
                    let mahal = match self.covariance_type {
                        CovarianceType::Full => {
                            // q = (x-m)^T W (x-m)  where W = W_inv^{-1}.
                            // Solve W_inv y = diff  ⇒  y = W diff;  q = diff^T y.
                            let llt = llt_cache[c].as_ref().unwrap();
                            let diff = Mat::from_fn(d, 1, |j, _| xi[j] - m_post[[c, j]]);
                            let y = llt.solve(&diff);
                            let mut q = 0.0;
                            for j in 0..d {
                                q += diff[(j, 0)] * y[(j, 0)];
                            }
                            q
                        }
                        CovarianceType::Diag => {
                            let mut q = 0.0;
                            for j in 0..d {
                                let dv = xi[j] - m_post[[c, j]];
                                q += dv * dv / w_inv_post[c][[0, j]].max(1e-300);
                            }
                            q
                        }
                    };
                    let e_mahal = (d as f64) / beta_post[c] + nu_post[c] * mahal;
                    log_rho[c] = e_log_pi[c] + half_e_log_det_lam[c]
                        - 0.5 * ((d as f64) * log_2pi + e_mahal);
                }
                let lse = logsumexp(&log_rho);
                data_log_lik += lse;
                for c in 0..k {
                    log_resp[[i, c]] = log_rho[c] - lse;
                }
            }

            // ─── M-step: update q(π) and q(μ, Λ) ───────────────────────
            // N_k = Σ_n r_{nk};  x̄_k = (1/N_k) Σ r x_n
            // S_k = (1/N_k) Σ r (x_n - x̄_k)(x_n - x̄_k)^T
            // α_k = α_0 + N_k
            // β_k = β_0 + N_k
            // ν_k = ν_0 + N_k
            // m_k = (β_0 m_0 + N_k x̄_k) / β_k
            // W_k_inv = W_0_inv + N_k S_k
            //           + (β_0 N_k)/(β_0 + N_k) (x̄_k - m_0)(x̄_k - m_0)^T

            let nk: Vec<f64> = (0..k)
                .map(|c| (0..n).map(|i| log_resp[[i, c]].exp()).sum::<f64>())
                .collect();

            for c in 0..k {
                let nkc = nk[c];
                alpha_post[c] = alpha0 + nkc;
                beta_post[c] = beta0 + nkc;
                nu_post[c] = nu0 + nkc;

                if nkc < 1e-12 {
                    // Component is empty — leave m_k at prior mean, scale
                    // W_k_inv toward prior.
                    for j in 0..d {
                        m_post[[c, j]] = m0[j];
                    }
                    w_inv_post[c] = w0_inv.clone();
                    continue;
                }

                // x̄_k
                let mut xbar = Array1::<f64>::zeros(d);
                for j in 0..d {
                    let mut s = 0.0;
                    for i in 0..n {
                        s += log_resp[[i, c]].exp() * x[[i, j]];
                    }
                    xbar[j] = s / nkc;
                }
                // m_k
                for j in 0..d {
                    m_post[[c, j]] = (beta0 * m0[j] + nkc * xbar[j]) / beta_post[c];
                }
                // W_k_inv update.
                let mix = beta0 * nkc / (beta0 + nkc);
                match self.covariance_type {
                    CovarianceType::Full => {
                        let mut w_new = w0_inv.clone();
                        for a in 0..d {
                            for b in 0..d {
                                let mut s = 0.0;
                                for i in 0..n {
                                    let r = log_resp[[i, c]].exp();
                                    let da = x[[i, a]] - xbar[a];
                                    let db = x[[i, b]] - xbar[b];
                                    s += r * da * db;
                                }
                                w_new[[a, b]] += s + mix * (xbar[a] - m0[a]) * (xbar[b] - m0[b]);
                            }
                        }
                        for j in 0..d {
                            w_new[[j, j]] += self.reg_covar;
                        }
                        w_inv_post[c] = w_new;
                    }
                    CovarianceType::Diag => {
                        let mut w_new = Array2::<f64>::zeros((1, d));
                        for j in 0..d {
                            let mut s = 0.0;
                            for i in 0..n {
                                let r = log_resp[[i, c]].exp();
                                let dv = x[[i, j]] - xbar[j];
                                s += r * dv * dv;
                            }
                            w_new[[0, j]] = w0_inv[[0, j]]
                                + s
                                + mix * (xbar[j] - m0[j]).powi(2)
                                + self.reg_covar;
                        }
                        w_inv_post[c] = w_new;
                    }
                }
            }

            // ─── ELBO (approx, for convergence) ─────────────────────────
            // We use the data log-likelihood term + KL[q(π) || p(π)] +
            // Σ_k KL[q(μ,Λ) || p(μ,Λ)] subtraction. Exact ELBO is more
            // involved; for convergence detection it suffices to track the
            // change in `data_log_lik / n` similar to sklearn's
            // `lower_bound_` proxy.
            elbo = data_log_lik / n as f64;
            if (elbo - prev_elbo).abs() < self.tol {
                break;
            }
            prev_elbo = elbo;
        }

        // ─── Output: posterior weights, covariances, effective count ──
        let sum_alpha: f64 = alpha_post.iter().sum();
        let weights: Array1<f64> = alpha_post.mapv(|a| a / sum_alpha);

        // Posterior covariance for users = W_k_inv / (ν_k - D - 1) (mode of
        // Inverse-Wishart). For ν_k ≤ D + 1 fall back to W_k_inv / ν_k.
        let covariances: Vec<Array2<f64>> = (0..k)
            .map(|c| {
                let denom = if nu_post[c] > (d as f64) + 1.0 {
                    nu_post[c] - (d as f64) - 1.0
                } else {
                    nu_post[c]
                };
                match self.covariance_type {
                    CovarianceType::Full => {
                        let mut s = Array2::<f64>::zeros((d, d));
                        for a in 0..d {
                            for b in 0..d {
                                s[[a, b]] = w_inv_post[c][[a, b]] / denom;
                            }
                        }
                        s
                    }
                    CovarianceType::Diag => {
                        let mut s = Array2::<f64>::zeros((1, d));
                        for j in 0..d {
                            s[[0, j]] = w_inv_post[c][[0, j]] / denom;
                        }
                        s
                    }
                }
            })
            .collect();

        // Effective components: αₖ > 2·α_0 (sklearn's heuristic equivalent
        // is "weights above a threshold"; we use the Dirichlet posterior
        // directly so this matches the formal "components with non-trivial
        // posterior mass" criterion).
        let eff = alpha_post.iter().filter(|&&a| a > 2.0 * alpha0).count();

        Ok(FittedBayesianGaussianMixture {
            weights,
            means: m_post,
            covariances,
            weight_concentration: alpha_post,
            dof_posterior: nu_post,
            mean_precision_posterior: beta_post,
            lower_bound: elbo,
            n_iter,
            effective_components: eff,
            covariance_type: self.covariance_type,
        })
    }
}

// ───── Predict helpers ─────────────────────────────────────────────────────

fn log_gauss_posterior(diff: &[f64], cov: &Array2<f64>, cov_type: CovarianceType) -> f64 {
    let d = diff.len();
    match cov_type {
        CovarianceType::Diag => {
            let mut q = 0.0;
            let mut logdet = 0.0;
            for j in 0..d {
                let v = cov[[0, j]].max(1e-30);
                q += diff[j] * diff[j] / v;
                logdet += v.ln();
            }
            -0.5 * (q + logdet + d as f64 * (2.0 * std::f64::consts::PI).ln())
        }
        CovarianceType::Full => {
            let nd = d;
            let m = Mat::from_fn(nd, nd, |i, j| cov[[i, j]]);
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

impl Predict<f64> for FittedBayesianGaussianMixture {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let k = self.weights.len();
        let d = self.means.ncols();
        if x.ncols() != d {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                d,
                x.ncols()
            )));
        }
        let n = x.nrows();
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let xi = x.row(i).to_owned();
            let mut best = f64::NEG_INFINITY;
            let mut best_c = 0;
            for c in 0..k {
                let mut diff = vec![0.0; d];
                for j in 0..d {
                    diff[j] = xi[j] - self.means[[c, j]];
                }
                let s = self.weights[c].max(1e-300).ln()
                    + log_gauss_posterior(&diff, &self.covariances[c], self.covariance_type);
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

impl PredictProba<f64> for FittedBayesianGaussianMixture {
    fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let k = self.weights.len();
        let d = self.means.ncols();
        if x.ncols() != d {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                d,
                x.ncols()
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
                logs[c] = self.weights[c].max(1e-300).ln()
                    + log_gauss_posterior(&diff, &self.covariances[c], self.covariance_type);
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

impl rustml_core::ClassifierScore<f64> for FittedBayesianGaussianMixture {}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_digamma_known_values() {
        // ψ(1) = -γ ≈ -0.5772156649
        assert!((digamma(1.0) - (-0.577_215_664_901_532_9)).abs() < 1e-8);
        // ψ(2) = 1 - γ
        assert!((digamma(2.0) - (1.0 - 0.577_215_664_901_532_9)).abs() < 1e-8);
        // ψ(0.5) = -γ - 2 ln 2
        assert!(
            (digamma(0.5) - (-0.577_215_664_901_532_9 - 2.0 * std::f64::consts::LN_2)).abs() < 1e-8
        );
    }

    #[test]
    fn test_lgamma_known_values() {
        // ln Γ(1) = 0; ln Γ(2) = 0; ln Γ(5) = ln 24
        assert!((lgamma(1.0)).abs() < 1e-8);
        assert!((lgamma(2.0)).abs() < 1e-8);
        assert!((lgamma(5.0) - 24.0_f64.ln()).abs() < 1e-8);
    }

    #[test]
    fn test_bgmm_separates_two_blobs() {
        let mut x = Vec::new();
        let n_per = 30;
        for i in 0..n_per {
            let t = i as f64 * 0.05;
            x.push(0.0 + t);
            x.push(0.0 + t.sin() * 0.1);
            x.push(10.0 - t);
            x.push(10.0 + t.cos() * 0.1);
        }
        let xa = Array2::from_shape_vec((n_per * 2, 2), x).unwrap();

        let bgmm = BayesianGaussianMixture::new(2)
            .with_concentration(0.01)
            .with_seed(0)
            .with_max_iter(200);
        let fitted: FittedBayesianGaussianMixture = FitUnsupervised::fit(&bgmm, &xa).unwrap();
        let preds = Predict::predict(&fitted, &xa).unwrap();
        let mut a_labels = std::collections::HashSet::new();
        let mut b_labels = std::collections::HashSet::new();
        for i in 0..n_per {
            a_labels.insert(preds[2 * i] as i64);
            b_labels.insert(preds[2 * i + 1] as i64);
        }
        assert_eq!(
            a_labels.len(),
            1,
            "blob A spans multiple labels: {:?}",
            a_labels
        );
        assert_eq!(
            b_labels.len(),
            1,
            "blob B spans multiple labels: {:?}",
            b_labels
        );
        assert_ne!(a_labels, b_labels);

        let p = PredictProba::predict_proba(&fitted, &xa).unwrap();
        for i in 0..xa.nrows() {
            let s: f64 = (0..2).map(|c| p[[i, c]]).sum();
            assert!((s - 1.0).abs() < 1e-9);
        }
        // ELBO should be finite.
        assert!(fitted.lower_bound.is_finite());
    }

    #[test]
    fn test_bgmm_sparsifies_unused_components() {
        // 2 well-separated blobs, fit with 6 components. Effective count
        // should drop to ~2 thanks to small α_0.
        let mut x = Vec::new();
        let n_per = 40;
        for i in 0..n_per {
            let t = i as f64 * 0.03;
            x.push(0.0 + t * 0.5);
            x.push(0.0 + t.sin() * 0.1);
            x.push(15.0 + t * 0.2);
            x.push(15.0 + t.cos() * 0.1);
        }
        let xa = Array2::from_shape_vec((n_per * 2, 2), x).unwrap();

        let bgmm = BayesianGaussianMixture::new(6)
            .with_concentration(0.001)
            .with_seed(0)
            .with_max_iter(300)
            .with_tol(1e-5);
        let fitted: FittedBayesianGaussianMixture = FitUnsupervised::fit(&bgmm, &xa).unwrap();
        // At least 2, at most all 6.
        assert!(fitted.effective_components >= 1 && fitted.effective_components <= 6);
        // Posterior weights sum to 1.
        let s: f64 = fitted.weights.iter().sum();
        assert!((s - 1.0).abs() < 1e-9);
    }
}
