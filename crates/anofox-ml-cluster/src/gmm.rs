//! Gaussian Mixture Model with EM training.
//!
//! Mirrors `sklearn.mixture.GaussianMixture` for all four covariance types
//! (`full`, `tied`, `diag`, `spherical`):
//!
//! - initial responsibilities from k-means (default), k-means++, random or
//!   random-from-data ([`MixtureInit`]), optionally overridden by
//!   `weights_init` / `means_init` / `precisions_init`;
//! - the precision Cholesky factors (`precisions_cholesky_`) are computed once
//!   per component per iteration and the E-step is batched (one GEMM per
//!   component), responsibilities are exponentiated once per M-step;
//! - convergence when the change of the mean log-likelihood lower bound is
//!   below `tol`; the best of `n_init` runs (highest lower bound) is kept;
//! - an ill-conditioned covariance is reported as an error suggesting a
//!   larger `reg_covar` instead of silently producing `-inf`/NaN.
//!
//! See the `mixture` module for the covariance storage layout.

use anofox_ml_core::{FitUnsupervised, Predict, PredictProba, Result, RustMlError};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

pub use crate::mixture::MixtureInit;
use crate::mixture::{
    check_blocks, estimate_gaussian_parameters, estimate_log_gaussian_prob, initial_resp,
    log_prob_resp, n_parameters, precision_cholesky, precision_cholesky_from_precisions,
};

/// Covariance parameterisation of a Gaussian mixture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CovarianceType {
    /// Each component has its own general covariance matrix.
    Full,
    /// Each component has its own diagonal covariance matrix.
    Diag,
    /// All components share one general covariance matrix.
    Tied,
    /// Each component has its own single variance (`σ_k² I`).
    Spherical,
}

/// Gaussian mixture estimator (unfitted).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GaussianMixture {
    pub n_components: usize,
    pub covariance_type: CovarianceType,
    pub max_iter: usize,
    /// Convergence threshold on the change of the mean log-likelihood.
    pub tol: f64,
    /// Non-negative regularisation added to the covariance diagonals.
    pub reg_covar: f64,
    pub seed: u64,
    /// Number of initialisations; the best (highest lower bound) is kept.
    pub n_init: usize,
    /// How initial responsibilities are computed.
    pub init_params: MixtureInit,
    /// User-provided initial weights (length `n_components`).
    pub weights_init: Option<Array1<f64>>,
    /// User-provided initial means (`n_components × n_features`).
    pub means_init: Option<Array2<f64>>,
    /// User-provided initial precision matrices, in the layout of
    /// the `mixture` module for the chosen covariance type.
    pub precisions_init: Option<Vec<Array2<f64>>>,
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
            init_params: MixtureInit::KMeans,
            weights_init: None,
            means_init: None,
            precisions_init: None,
        }
    }
    pub fn with_covariance_type(mut self, c: CovarianceType) -> Self {
        self.covariance_type = c;
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
    pub fn with_n_init(mut self, n: usize) -> Self {
        self.n_init = n.max(1);
        self
    }
    pub fn with_init_params(mut self, init: MixtureInit) -> Self {
        self.init_params = init;
        self
    }
    pub fn with_weights_init(mut self, w: Array1<f64>) -> Self {
        self.weights_init = Some(w);
        self
    }
    pub fn with_means_init(mut self, m: Array2<f64>) -> Self {
        self.means_init = Some(m);
        self
    }
    /// Initial precisions: `k` `d×d` (full), one `d×d` (tied), `k` `1×d`
    /// (diag) or `k` `1×1` (spherical) blocks.
    pub fn with_precisions_init(mut self, p: Vec<Array2<f64>>) -> Self {
        self.precisions_init = Some(p);
        self
    }
}

/// Fitted Gaussian mixture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedGaussianMixture {
    /// Mixing weights (sum to 1).
    pub weights: Array1<f64>,
    /// Component means, `k × d`.
    pub means: Array2<f64>,
    /// Covariances in the layout of the `mixture` module: `k` `d×d` (full),
    /// one `d×d` (tied), `k` `1×d` (diag), `k` `1×1` (spherical).
    pub covariances: Vec<Array2<f64>>,
    /// Precision Cholesky factors `U` with `Λ = U Uᵀ`, same layout as
    /// `covariances` (sklearn `precisions_cholesky_`).
    #[serde(default)]
    pub precisions_cholesky: Vec<Array2<f64>>,
    /// Mean per-sample log-likelihood of the training data under the final
    /// parameters (final E-step; equals `score(X_train)`).
    pub log_likelihood: f64,
    /// sklearn's `lower_bound_`: the mean log-likelihood computed in the
    /// E-step of the last EM iteration (i.e. before the last M-step).
    #[serde(default)]
    pub lower_bound: f64,
    pub n_iter: usize,
    /// Whether the best run reached `tol` before `max_iter`.
    #[serde(default)]
    pub converged: bool,
    pub covariance_type: CovarianceType,
}

impl FitUnsupervised<f64> for GaussianMixture {
    type Fitted = FittedGaussianMixture;

    fn fit(&self, x: &Array2<f64>) -> Result<Self::Fitted> {
        let x_std = crate::standard_layout(x);
        let x: &Array2<f64> = &x_std;
        let n = x.nrows();
        let d = x.ncols();
        let k = self.n_components;
        if n == 0 {
            return Err(RustMlError::EmptyInput("empty input".into()));
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
        self.check_inits(k, d)?;

        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut best: Option<FittedGaussianMixture> = None;
        for _ in 0..self.n_init.max(1) {
            let fitted = self.single_fit(x, &mut rng)?;
            let better = match &best {
                None => true,
                Some(b) => fitted.lower_bound > b.lower_bound || b.lower_bound == f64::NEG_INFINITY,
            };
            if better {
                best = Some(fitted);
            }
        }
        let mut best = best.unwrap();
        // Final E-step: report the likelihood of the returned parameters.
        best.log_likelihood = best.score(x)?;
        Ok(best)
    }
}

impl GaussianMixture {
    fn check_inits(&self, k: usize, d: usize) -> Result<()> {
        if let Some(w) = &self.weights_init {
            if w.len() != k {
                return Err(RustMlError::ShapeMismatch(format!(
                    "weights_init has length {}, expected {k}",
                    w.len()
                )));
            }
            if w.iter().any(|&v| !(0.0..=1.0).contains(&v)) || (w.sum() - 1.0).abs() > 1e-6 {
                return Err(RustMlError::InvalidParameter(
                    "weights_init must be in [0, 1] and sum to 1".into(),
                ));
            }
        }
        if let Some(m) = &self.means_init {
            if m.dim() != (k, d) {
                return Err(RustMlError::ShapeMismatch(format!(
                    "means_init has shape {:?}, expected ({k}, {d})",
                    m.dim()
                )));
            }
        }
        if let Some(p) = &self.precisions_init {
            check_blocks(p, self.covariance_type, k, d, "precisions_init")?;
        }
        Ok(())
    }

    fn single_fit(&self, x: &Array2<f64>, rng: &mut StdRng) -> Result<FittedGaussianMixture> {
        let n = x.nrows();
        let k = self.n_components;
        let ct = self.covariance_type;

        // ─── Initialisation (sklearn `_initialize`) ────────────────────
        let resp = initial_resp(x, k, self.init_params, rng)?;
        let (nk, means0, covariances) = estimate_gaussian_parameters(x, &resp, self.reg_covar, ct);
        let mut model = FittedGaussianMixture {
            weights: self.weights_init.clone().unwrap_or_else(|| nk / n as f64),
            means: self.means_init.clone().unwrap_or(means0),
            precisions_cholesky: match &self.precisions_init {
                Some(p) => precision_cholesky_from_precisions(p, ct)?,
                None => precision_cholesky(&covariances, ct)?,
            },
            covariances,
            log_likelihood: f64::NEG_INFINITY,
            lower_bound: f64::NEG_INFINITY,
            n_iter: 0,
            converged: false,
            covariance_type: ct,
        };
        if self.max_iter == 0 && self.precisions_init.is_some() {
            model.covariances =
                crate::mixture::covariances_from_precision_cholesky(&model.precisions_cholesky, ct);
        }

        // ─── EM ────────────────────────────────────────────────────────
        let mut lower_bound = f64::NEG_INFINITY;
        for it in 1..=self.max_iter {
            let prev = lower_bound;
            // E-step.
            let (log_norm, log_resp) = log_prob_resp(model.weighted_log_prob(x));
            lower_bound = log_norm.mean().unwrap();
            // M-step with responsibilities exponentiated once.
            let resp = log_resp.mapv(f64::exp);
            let (nk, means, covs) = estimate_gaussian_parameters(x, &resp, self.reg_covar, ct);
            model.weights = &nk / nk.sum();
            model.means = means;
            model.precisions_cholesky = precision_cholesky(&covs, ct)?;
            model.covariances = covs;
            model.n_iter = it;
            if (lower_bound - prev).abs() < self.tol {
                model.converged = true;
                break;
            }
        }
        model.lower_bound = lower_bound;
        Ok(model)
    }
}

impl FittedGaussianMixture {
    fn check_x(&self, x: &Array2<f64>) -> Result<()> {
        let d = self.means.ncols();
        if x.ncols() != d {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                d,
                x.ncols()
            )));
        }
        Ok(())
    }

    /// Precision Cholesky factors; recomputed from the covariances for
    /// models deserialised from an older format without them.
    fn prec_chol(&self) -> Result<std::borrow::Cow<'_, [Array2<f64>]>> {
        if self.precisions_cholesky.is_empty() {
            Ok(std::borrow::Cow::Owned(precision_cholesky(
                &self.covariances,
                self.covariance_type,
            )?))
        } else {
            Ok(std::borrow::Cow::Borrowed(&self.precisions_cholesky))
        }
    }

    /// `log p(x_i | k) + log w_k`, `n × k`.
    fn weighted_log_prob(&self, x: &Array2<f64>) -> Array2<f64> {
        let pc = self
            .prec_chol()
            .expect("precision Cholesky of fitted covariances");
        let mut lp = estimate_log_gaussian_prob(x, &self.means, &pc, self.covariance_type);
        let lw = self.weights.mapv(f64::ln);
        lp += &lw;
        lp
    }

    fn weighted_log_prob_checked(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.check_x(x)?;
        self.prec_chol()?;
        let x_std = crate::standard_layout(x);
        Ok(self.weighted_log_prob(&x_std))
    }

    /// Per-sample log-likelihood `log p(x_i)` (sklearn `score_samples`).
    pub fn score_samples(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        Ok(log_prob_resp(self.weighted_log_prob_checked(x)?).0)
    }

    /// Mean per-sample log-likelihood (sklearn `score`).
    pub fn score(&self, x: &Array2<f64>) -> Result<f64> {
        let s = self.score_samples(x)?;
        s.mean()
            .ok_or_else(|| RustMlError::EmptyInput("empty input".into()))
    }

    /// Number of free parameters of the model.
    pub fn n_parameters(&self) -> usize {
        n_parameters(self.covariance_type, self.means.nrows(), self.means.ncols())
    }

    /// Bayesian information criterion on `x` (lower is better).
    pub fn bic(&self, x: &Array2<f64>) -> Result<f64> {
        let n = x.nrows() as f64;
        Ok(-2.0 * self.score(x)? * n + self.n_parameters() as f64 * n.ln())
    }

    /// Akaike information criterion on `x` (lower is better).
    pub fn aic(&self, x: &Array2<f64>) -> Result<f64> {
        let n = x.nrows() as f64;
        Ok(-2.0 * self.score(x)? * n + 2.0 * self.n_parameters() as f64)
    }

    /// Whether EM converged (sklearn `converged_`).
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// Number of EM iterations of the best run (sklearn `n_iter_`).
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }
}

impl Predict<f64> for FittedGaussianMixture {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
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

impl PredictProba<f64> for FittedGaussianMixture {
    fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let (_, log_resp) = log_prob_resp(self.weighted_log_prob_checked(x)?);
        Ok(log_resp.mapv(f64::exp))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_gmm_two_well_separated_blobs() {
        let x = array![
            [0.0_f64, 0.0],
            [0.2, 0.1],
            [-0.1, 0.2],
            [0.1, -0.2],
            [10.0, 10.0],
            [10.1, 9.9],
            [9.8, 10.2],
            [10.2, 9.8],
        ];
        for ct in [
            CovarianceType::Full,
            CovarianceType::Diag,
            CovarianceType::Tied,
            CovarianceType::Spherical,
        ] {
            let fitted = GaussianMixture::new(2)
                .with_covariance_type(ct)
                .with_seed(0)
                .fit(&x)
                .unwrap();
            let labels = fitted.predict(&x).unwrap();
            let l0 = labels[0];
            for i in 1..4 {
                assert_eq!(labels[i], l0, "{ct:?}");
            }
            for i in 4..8 {
                assert_ne!(labels[i], l0, "{ct:?}");
            }
            // Final E-step: reported log-likelihood belongs to the final params.
            assert!((fitted.log_likelihood - fitted.score(&x).unwrap()).abs() < 1e-12);
        }
    }

    #[test]
    fn test_singular_covariance_errors_with_reg_covar_hint() {
        // Two exact duplicates per component and reg_covar = 0 ⇒ singular.
        let x = array![[0.0, 0.0], [0.0, 0.0], [5.0, 5.0], [5.0, 5.0]];
        let err = GaussianMixture::new(2)
            .with_reg_covar(0.0)
            .fit(&x)
            .unwrap_err();
        assert!(err.to_string().contains("reg_covar"), "{err}");
        // The default reg_covar makes it well-posed.
        let ok = GaussianMixture::new(2).fit(&x).unwrap();
        assert!(ok.score(&x).unwrap().is_finite());
    }
}

impl anofox_ml_core::ClassifierScore<f64> for FittedGaussianMixture {}

impl anofox_ml_core::PredictLogProba<f64> for FittedGaussianMixture {}
