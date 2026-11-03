//! Bayesian Gaussian Mixture Model — variational inference.
//!
//! Implements `sklearn.mixture.BayesianGaussianMixture` with
//! `weight_concentration_prior_type='dirichlet_distribution'` for all four
//! covariance types (`full`, `tied`, `diag`, `spherical`).
//!
//! Mathematical model (Bishop, PRML §10.2):
//!
//!   π ~ Dir(α_0, …, α_0)
//!   (μ_k, Λ_k) ~ NormalWishart(m_0, β_0, W_0, ν_0)
//!   z_n | π ~ Cat(π)
//!   x_n | z_n=k, μ_k, Λ_k ~ N(μ_k, Λ_k^{-1})
//!
//! The variational posterior factorises as q(Z) q(π) ∏_k q(μ_k, Λ_k), each
//! kept in conjugate form. CAVI alternates the E-step (responsibilities) and
//! the M-step (Dirichlet + Normal-Wishart updates), exactly as sklearn: the
//! priors default to `α_0 = 1/K`, `β_0 = 1`, `m_0 = mean(X)`, `ν_0 = D` and
//! `W_0⁻¹ = cov(X)` (ddof = 1; diagonal / mean variance for the diag /
//! spherical types), and the stored `covariances` are `W_k⁻¹ / ν_k`.
//!
//! Convergence is judged on the change of the evidence lower bound
//! ([`FittedBayesianGaussianMixture::lower_bound`], sklearn's
//! `_compute_lower_bound`: the ELBO up to an additive constant that does not
//! depend on the variational parameters). The effective component count is
//! the number of components whose posterior Dirichlet parameter αₖ exceeds
//! `2·α_0`.

use anofox_ml_core::Float;
use anofox_ml_core::{FitUnsupervised, Predict, PredictProba, Result, RustMlError};
use ndarray::{Array1, Array2, ArrayView2, Axis};
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::gmm::CovarianceType;
use crate::mixture::{
    estimate_gaussian_parameters, estimate_log_gaussian_prob, initial_resp, log_det_cholesky,
    log_prob_resp, precision_cholesky, MixtureInit,
};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BayesianGaussianMixture {
    pub n_components: usize,
    pub covariance_type: CovarianceType,
    pub max_iter: usize,
    pub tol: f64,
    pub reg_covar: f64,
    pub seed: u64,
    /// Dirichlet concentration prior on the mixing weights (α_0). Sklearn
    /// default is 1/K.
    pub weight_concentration_prior: f64,
    /// Mean precision prior (β_0). Sklearn default 1.0.
    pub mean_precision_prior: f64,
    /// Degrees-of-freedom prior (ν_0). Must satisfy ν_0 > D - 1; sklearn
    /// defaults to D.
    pub degrees_of_freedom_prior: Option<f64>,
    /// Prior on the means (m_0); defaults to the data mean.
    pub mean_prior: Option<Array1<f64>>,
    /// Number of initialisations; the one with the highest lower bound wins.
    pub n_init: usize,
    /// How initial responsibilities are computed.
    pub init_params: MixtureInit,
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
            mean_prior: None,
            n_init: 1,
            init_params: MixtureInit::KMeans,
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
    pub fn with_mean_prior(mut self, m: Array1<f64>) -> Self {
        self.mean_prior = Some(m);
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
    pub fn with_reg_covar(mut self, r: f64) -> Self {
        self.reg_covar = r;
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
    pub fn with_n_init(mut self, n: usize) -> Self {
        self.n_init = n.max(1);
        self
    }
    pub fn with_init_params(mut self, init: MixtureInit) -> Self {
        self.init_params = init;
        self
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedBayesianGaussianMixture {
    /// Posterior mean of π (normalised α_k / Σα).
    pub weights: Array1<f64>,
    /// Posterior means m_k.
    pub means: Array2<f64>,
    /// `W_k⁻¹ / ν_k` (sklearn `covariances_`), in the layout of
    /// the `mixture` module: `k` `d×d` (full), one `d×d` (tied), `k` `1×d`
    /// (diag) or `k` `1×1` (spherical).
    pub covariances: Vec<Array2<f64>>,
    /// Posterior Dirichlet parameters (α_k = α_0 + N_k).
    pub weight_concentration: Array1<f64>,
    /// Posterior degrees of freedom (ν_k); all equal for `Tied`.
    pub dof_posterior: Array1<f64>,
    /// Posterior mean precision (β_k).
    pub mean_precision_posterior: Array1<f64>,
    /// Evidence lower bound (up to a constant) of the best run, sklearn's
    /// `lower_bound_`.
    pub lower_bound: f64,
    pub n_iter: usize,
    pub effective_components: usize,
    pub covariance_type: CovarianceType,
    /// Cholesky factors of `covariances⁻¹` (sklearn `precisions_cholesky_`).
    #[serde(default)]
    pub precisions_cholesky: Vec<Array2<f64>>,
    /// Whether the best run reached `tol` before `max_iter`.
    #[serde(default)]
    pub converged: bool,
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

/// Log gamma ln Γ(x) for x > 0 (Lanczos g=7, n=9), used by the ELBO.
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

// ───── Variational state ──────────────────────────────────────────────────

/// Normal-Wishart prior hyper-parameters.
struct Priors {
    alpha0: f64,
    beta0: f64,
    nu0: f64,
    m0: Array1<f64>,
    /// `W_0⁻¹` in block layout (d×d, 1×d or 1×1).
    cov_prior: Array2<f64>,
}

/// Variational posterior parameters.
struct VbState {
    wc: Array1<f64>,
    beta: Array1<f64>,
    means: Array2<f64>,
    dof: Array1<f64>,
    covariances: Vec<Array2<f64>>,
    prec_chol: Vec<Array2<f64>>,
}

fn block<T: Copy>(v: &[T], ct: CovarianceType, c: usize) -> T {
    if ct == CovarianceType::Tied {
        v[0]
    } else {
        v[c]
    }
}

impl VbState {
    /// One "M-step" from sufficient statistics (sklearn `_estimate_weights`,
    /// `_estimate_means`, `_estimate_precisions`).
    fn update(
        nk: &Array1<f64>,
        xk: &Array2<f64>,
        sk: &[Array2<f64>],
        p: &Priors,
        ct: CovarianceType,
    ) -> Result<Self> {
        let k = nk.len();
        let d = xk.ncols();
        let wc = nk + p.alpha0;
        let beta = nk + p.beta0;
        let mut means = Array2::<f64>::zeros((k, d));
        for c in 0..k {
            for j in 0..d {
                means[[c, j]] = (p.beta0 * p.m0[j] + nk[c] * xk[[c, j]]) / beta[c];
            }
        }
        let diff = |c: usize| &xk.row(c) - &p.m0;
        let (dof, covariances) = match ct {
            CovarianceType::Full => {
                let dof = nk + p.nu0;
                let covs = (0..k)
                    .map(|c| {
                        let df = diff(c);
                        let s = nk[c] * p.beta0 / beta[c];
                        let mut cov = &p.cov_prior + &(&sk[c] * nk[c]);
                        for a in 0..d {
                            for b in 0..d {
                                cov[[a, b]] += s * df[a] * df[b];
                            }
                        }
                        cov / dof[c]
                    })
                    .collect();
                (dof, covs)
            }
            CovarianceType::Tied => {
                let nbar = nk.sum() / k as f64;
                let dof_t = p.nu0 + nbar;
                let mut cov = &p.cov_prior + &(&sk[0] * nbar);
                for c in 0..k {
                    let df = diff(c);
                    let s = p.beta0 / k as f64 * nk[c] / beta[c];
                    for a in 0..d {
                        for b in 0..d {
                            cov[[a, b]] += s * df[a] * df[b];
                        }
                    }
                }
                (Array1::from_elem(k, dof_t), vec![cov / dof_t])
            }
            CovarianceType::Diag => {
                let dof = nk + p.nu0;
                let covs = (0..k)
                    .map(|c| {
                        let df = diff(c);
                        let r = p.beta0 / beta[c];
                        Array2::from_shape_fn((1, d), |(_, j)| {
                            (p.cov_prior[[0, j]] + nk[c] * (sk[c][[0, j]] + r * df[j] * df[j]))
                                / dof[c]
                        })
                    })
                    .collect();
                (dof, covs)
            }
            CovarianceType::Spherical => {
                let dof = nk + p.nu0;
                let covs = (0..k)
                    .map(|c| {
                        let df = diff(c);
                        let r = p.beta0 / beta[c];
                        let m2 = df.mapv(|v| v * v).mean().unwrap();
                        Array2::from_elem(
                            (1, 1),
                            (p.cov_prior[[0, 0]] + nk[c] * (sk[c][[0, 0]] + r * m2)) / dof[c],
                        )
                    })
                    .collect();
                (dof, covs)
            }
        };
        let prec_chol = precision_cholesky(&covariances, ct)?;
        Ok(Self {
            wc,
            beta,
            means,
            dof,
            covariances,
            prec_chol,
        })
    }
}

/// `E_q[log π_k] + E_q[log N(x_i | μ_k, Λ_k⁻¹)]`, `n × k` (sklearn
/// `_estimate_weighted_log_prob` of the Bayesian mixture).
fn weighted_log_prob(
    x: ArrayView2<f64>,
    means: &Array2<f64>,
    prec_chol: &[Array2<f64>],
    dof: &Array1<f64>,
    beta: &Array1<f64>,
    wc: &Array1<f64>,
    ct: CovarianceType,
) -> Array2<f64> {
    let d = x.ncols();
    let df = d as f64;
    let k = means.nrows();
    let mut lp = estimate_log_gaussian_prob(x, means, prec_chol, ct);
    let psi_sum = digamma(wc.sum());
    for c in 0..k {
        let log_lambda: f64 = df * std::f64::consts::LN_2
            + (0..d)
                .map(|i| digamma(0.5 * (dof[c] - i as f64)))
                .sum::<f64>();
        let shift = -0.5 * df * dof[c].ln()
            + 0.5 * (log_lambda - df / beta[c])
            + (digamma(wc[c]) - psi_sum);
        lp.column_mut(c).mapv_inplace(|v| v + shift);
    }
    lp
}

/// sklearn `_compute_lower_bound` for the Dirichlet-distribution prior.
fn lower_bound(log_resp: &Array2<f64>, s: &VbState, ct: CovarianceType, d: usize) -> f64 {
    let df = d as f64;
    let k = s.means.nrows();
    let ld = log_det_cholesky(&s.prec_chol, ct, d);
    let mut log_wishart = 0.0;
    for c in 0..k {
        let nu = s.dof[c];
        let log_det = block(&ld, ct, c) - 0.5 * df * nu.ln();
        let gl: f64 = (0..d).map(|i| lgamma(0.5 * (nu - i as f64))).sum();
        log_wishart -= nu * log_det + nu * df * 0.5 * std::f64::consts::LN_2 + gl;
    }
    let log_norm_weight = lgamma(s.wc.sum()) - s.wc.iter().map(|&a| lgamma(a)).sum::<f64>();
    let neg_entropy: f64 = log_resp
        .iter()
        .map(|&lr| {
            let r = lr.exp();
            if r > 0.0 {
                r * lr
            } else {
                0.0
            }
        })
        .sum();
    -neg_entropy
        - log_wishart
        - log_norm_weight
        - 0.5 * df * s.beta.iter().map(|b| b.ln()).sum::<f64>()
}

impl BayesianGaussianMixture {
    fn priors(&self, x: ArrayView2<f64>) -> Result<Priors> {
        let n = x.nrows();
        let d = x.ncols();
        let nu0 = self.degrees_of_freedom_prior.unwrap_or(d as f64);
        if nu0 <= (d as f64) - 1.0 {
            return Err(RustMlError::InvalidParameter(
                "degrees_of_freedom_prior must exceed D - 1".into(),
            ));
        }
        if self.weight_concentration_prior <= 0.0 || self.mean_precision_prior <= 0.0 {
            return Err(RustMlError::InvalidParameter(
                "weight_concentration_prior and mean_precision_prior must be > 0".into(),
            ));
        }
        let mean = x.mean_axis(Axis(0)).unwrap();
        let m0 = match &self.mean_prior {
            Some(m) if m.len() != d => {
                return Err(RustMlError::ShapeMismatch(format!(
                    "mean_prior has length {}, expected {d}",
                    m.len()
                )))
            }
            Some(m) => m.clone(),
            None => mean.clone(),
        };
        // Empirical covariance with ddof = 1 (numpy `np.cov(X.T)`).
        let centred = &x - &mean;
        let emp = centred.t().dot(&centred) / (n as f64 - 1.0);
        let cov_prior = match self.covariance_type {
            CovarianceType::Full | CovarianceType::Tied => emp,
            CovarianceType::Diag => Array2::from_shape_fn((1, d), |(_, j)| emp[[j, j]]),
            CovarianceType::Spherical => {
                Array2::from_elem((1, 1), (0..d).map(|j| emp[[j, j]]).sum::<f64>() / d as f64)
            }
        };
        Ok(Priors {
            alpha0: self.weight_concentration_prior,
            beta0: self.mean_precision_prior,
            nu0,
            m0,
            cov_prior,
        })
    }

    fn single_fit(
        &self,
        x: ArrayView2<f64>,
        p: &Priors,
        rng: &mut StdRng,
    ) -> Result<(VbState, f64, usize, bool)> {
        let ct = self.covariance_type;
        let d = x.ncols();
        let resp = initial_resp(x, self.n_components, self.init_params, rng)?;
        let (nk, xk, sk) = estimate_gaussian_parameters(x, &resp, self.reg_covar, ct);
        let mut s = VbState::update(&nk, &xk, &sk, p, ct)?;

        let mut lb = f64::NEG_INFINITY;
        let mut n_iter = 0;
        let mut converged = false;
        for it in 1..=self.max_iter {
            let prev = lb;
            let wlp = weighted_log_prob(x, &s.means, &s.prec_chol, &s.dof, &s.beta, &s.wc, ct);
            let (_, log_resp) = log_prob_resp(wlp);
            let resp = log_resp.mapv(f64::exp);
            let (nk, xk, sk) = estimate_gaussian_parameters(x, &resp, self.reg_covar, ct);
            s = VbState::update(&nk, &xk, &sk, p, ct)?;
            lb = lower_bound(&log_resp, &s, ct, d);
            n_iter = it;
            if (lb - prev).abs() < self.tol {
                converged = true;
                break;
            }
        }
        Ok((s, lb, n_iter, converged))
    }
}

impl<F: Float> FitUnsupervised<F> for BayesianGaussianMixture {
    type Fitted = FittedBayesianGaussianMixture;

    /// Computed in `f64` whatever the input float type.
    fn fit_view(&self, x: ArrayView2<F>) -> Result<Self::Fitted> {
        let x_f64 = anofox_ml_core::as_f64(x);
        let x = x_f64.view();
        let x_std = crate::standard_layout(x);
        let x = x_std.view();
        let n = x.nrows();
        let k = self.n_components;
        if n == 0 {
            return Err(RustMlError::EmptyInput("empty input".into()));
        }
        if n < 2 {
            return Err(RustMlError::InvalidParameter(
                "BayesianGaussianMixture needs at least 2 samples".into(),
            ));
        }
        if k == 0 || k > n {
            return Err(RustMlError::InvalidParameter(format!(
                "n_components ({k}) must be in 1..=n_samples ({n})"
            )));
        }
        if self.reg_covar < 0.0 {
            return Err(RustMlError::InvalidParameter(
                "reg_covar must be non-negative".into(),
            ));
        }
        let p = self.priors(x)?;
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut best: Option<(VbState, f64, usize, bool)> = None;
        for _ in 0..self.n_init.max(1) {
            let run = self.single_fit(x, &p, &mut rng)?;
            let better = match &best {
                None => true,
                Some(b) => run.1 > b.1 || b.1 == f64::NEG_INFINITY,
            };
            if better {
                best = Some(run);
            }
        }
        let (s, lb, n_iter, converged) = best.unwrap();
        let sum_alpha: f64 = s.wc.sum();
        let weights = s.wc.mapv(|a| a / sum_alpha);
        let eff = s.wc.iter().filter(|&&a| a > 2.0 * p.alpha0).count();
        Ok(FittedBayesianGaussianMixture {
            weights,
            means: s.means,
            covariances: s.covariances,
            weight_concentration: s.wc,
            dof_posterior: s.dof,
            mean_precision_posterior: s.beta,
            lower_bound: lb,
            n_iter,
            effective_components: eff,
            covariance_type: self.covariance_type,
            precisions_cholesky: s.prec_chol,
            converged,
        })
    }
}

// ───── Predict helpers ─────────────────────────────────────────────────────

impl FittedBayesianGaussianMixture {
    fn weighted_log_prob_checked(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        let d = self.means.ncols();
        if x.ncols() != d {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                d,
                x.ncols()
            )));
        }
        let x_std = crate::standard_layout(x.view());
        let computed;
        let pc = if self.precisions_cholesky.is_empty() {
            computed = precision_cholesky(&self.covariances, self.covariance_type)?;
            &computed
        } else {
            &self.precisions_cholesky
        };
        Ok(weighted_log_prob(
            x_std.view(),
            &self.means,
            pc,
            &self.dof_posterior,
            &self.mean_precision_posterior,
            &self.weight_concentration,
            self.covariance_type,
        ))
    }

    /// Whether the best run converged (sklearn `converged_`).
    pub fn converged(&self) -> bool {
        self.converged
    }
}

impl FittedBayesianGaussianMixture {
    fn predict_f64(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        let wlp = self.weighted_log_prob_checked(x)?;
        Ok(wlp
            .rows()
            .into_iter()
            .map(|r| {
                let mut best = 0usize;
                for c in 1..r.len() {
                    if r[c] > r[best] {
                        best = c;
                    }
                }
                best as f64
            })
            .collect())
    }
}

impl<F: Float> Predict<F> for FittedBayesianGaussianMixture {
    fn predict_view(&self, x: ArrayView2<F>) -> Result<Array1<F>> {
        Ok(anofox_ml_core::from_f64(
            self.predict_f64(anofox_ml_core::as_f64(x).view())?,
        ))
    }
}

impl FittedBayesianGaussianMixture {
    fn predict_proba_f64(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        let (_, log_resp) = log_prob_resp(self.weighted_log_prob_checked(x)?);
        Ok(log_resp.mapv(f64::exp))
    }
}

impl<F: Float> PredictProba<F> for FittedBayesianGaussianMixture {
    fn predict_proba_view(&self, x: ArrayView2<F>) -> Result<Array2<F>> {
        Ok(anofox_ml_core::from_f64(
            self.predict_proba_f64(anofox_ml_core::as_f64(x).view())?,
        ))
    }
}

impl<F: Float> anofox_ml_core::ClassifierScore<F> for FittedBayesianGaussianMixture {}

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
