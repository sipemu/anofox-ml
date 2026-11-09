//! Gaussian Process classification with the Laplace approximation.
//!
//! Mirrors `sklearn.gaussian_process.GaussianProcessClassifier`
//! (binary case, plus one-vs-rest for more classes).
//!
//! Algorithm (Rasmussen & Williams §3.4, Algorithms 3.1, 3.2 and 5.1):
//!
//! 1. Find the mode `f̂` of the latent posterior by Newton iterations using
//!    the Cholesky factor of `B = I + Wˢ K Wˢ` (`Wˢ = W^{1/2}`).
//! 2. The approximate log-marginal likelihood
//!    `-½ aᵀf̂ + Σ log σ(y_i f̂_i) - Σ log L_ii` and its analytic gradient
//!    are maximised over the kernel hyperparameters with L-BFGS-B
//!    (sklearn default `optimizer="fmin_l_bfgs_b"`).
//! 3. Predictive probabilities `∫ σ(f*) N(f* | μ*, σ*²) df*` use sklearn's
//!    erf-based approximation of the logistic by a sum of five probits.

use anofox_ml_core::{Fit, Predict, PredictProba, Result, RustMlError};
use faer::linalg::solvers::Solve;
use faer::Mat;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, AsArray, Ix2};

use crate::linalg::{cholesky, col_to_array1, solve_lower, vec_to_col};
use crate::{optimise_kernel, GpKernel};

/// Binary Gaussian process classifier (Laplace approximation).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GaussianProcessClassifier {
    pub kernel: GpKernel,
    /// Maximum Newton iterations for the posterior mode (sklearn
    /// `max_iter_predict`, default 100).
    pub max_iter: usize,
    /// Stop the Newton iterations once the approximate log-marginal
    /// likelihood improves by less than `tol` (sklearn hard-codes 1e-10).
    pub tol: f64,
    /// Optimise the kernel hyperparameters (sklearn default on).
    pub optimizer: bool,
    /// Additional optimiser restarts from random log-uniform points.
    pub n_restarts_optimizer: usize,
    /// Seed for the restart starting points.
    pub seed: u64,
}

impl GaussianProcessClassifier {
    pub fn new(kernel: GpKernel) -> Self {
        Self {
            kernel,
            max_iter: 100,
            tol: 1e-10,
            optimizer: true,
            n_restarts_optimizer: 0,
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

/// Fitted binary Gaussian process classifier.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedGaussianProcessClassifier {
    pub x_train: Array2<f64>,
    /// `y_train - π̂` (the dual coefficient vector at the posterior mode).
    pub alpha: Array1<f64>,
    /// Cholesky factor `L` of `I + Wˢ K Wˢ`.
    #[serde(with = "crate::linalg::mat_serde")]
    pub l_lower: Mat<f64>,
    /// `Wˢ = W^{1/2}` at the posterior mode.
    pub w_sqrt: Array1<f64>,
    /// Kernel with the optimised hyperparameters (sklearn `kernel_`).
    pub kernel: GpKernel,
    pub classes: [f64; 2],
    /// Approximate log-marginal likelihood (sklearn
    /// `log_marginal_likelihood_value_`).
    pub log_marginal_likelihood_value: f64,
}

fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// `log(1 + e^z)` without overflow.
fn softplus(z: f64) -> f64 {
    z.max(0.0) + (-z.abs()).exp().ln_1p()
}

/// Temporaries of the Laplace posterior-mode search (sklearn
/// `_posterior_mode(..., return_temporaries=True)`).
struct PosteriorMode {
    lml: f64,
    pi: Array1<f64>,
    w_sr: Array1<f64>,
    llt: faer::linalg::solvers::Llt<f64>,
    a: Array1<f64>,
}

fn posterior_mode(
    k: &Array2<f64>,
    y: &Array1<f64>,
    max_iter: usize,
    tol: f64,
) -> Option<PosteriorMode> {
    let n = y.len();
    let mut f = Array1::<f64>::zeros(n);
    let mut prev = f64::NEG_INFINITY;
    let mut out = None;
    for _ in 0..max_iter.max(1) {
        let pi = f.mapv(sigmoid);
        let w = pi.mapv(|p| p * (1.0 - p));
        let w_sr = w.mapv(f64::sqrt);
        let mut b_mat = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                b_mat[[i, j]] = w_sr[i] * k[[i, j]] * w_sr[j];
            }
            b_mat[[i, i]] += 1.0;
        }
        let llt = cholesky(&b_mat)?;
        let b = &w * &f + &(y - &pi);
        let kb = k.dot(&b);
        let rhs = &w_sr * &kb;
        let sol = col_to_array1(&llt.solve(vec_to_col(&rhs)));
        let a = &b - &(&w_sr * &sol);
        f = k.dot(&a);
        // log p(y|f) = Σ log σ((2y-1) f) = -Σ softplus(-(2y-1) f)
        let loglik: f64 = y
            .iter()
            .zip(f.iter())
            .map(|(&yi, &fi)| -softplus(-(2.0 * yi - 1.0) * fi))
            .sum();
        let l = llt.L();
        let log_det_half: f64 = (0..n).map(|i| l[(i, i)].ln()).sum();
        let lml = -0.5 * a.dot(&f) + loglik - log_det_half;
        if !lml.is_finite() {
            return None;
        }
        let done = lml - prev < tol;
        out = Some(PosteriorMode {
            lml,
            pi,
            w_sr,
            llt,
            a,
        });
        if done {
            break;
        }
        prev = lml;
    }
    out
}

/// Approximate LML and its gradient w.r.t. the free log-hyperparameters
/// (R&W Algorithm 5.1, as in sklearn).
fn gpc_lml_with_grad(
    kernel: &GpKernel,
    x: ArrayView2<f64>,
    y: &Array1<f64>,
    max_iter: usize,
    tol: f64,
    eval_gradient: bool,
) -> (f64, Vec<f64>) {
    let (k, grads) = if eval_gradient {
        kernel.gram_with_gradient(x)
    } else {
        (kernel.gram(x), Vec::new())
    };
    let pm = match posterior_mode(&k, y, max_iter, tol) {
        Some(pm) => pm,
        None => return (f64::NEG_INFINITY, vec![0.0; grads.len()]),
    };
    if !eval_gradient || grads.is_empty() {
        return (pm.lml, Vec::new());
    }
    let n = y.len();
    let w_sr = &pm.w_sr;
    // R = Wˢ B⁻¹ Wˢ
    let diag_w = Mat::<f64>::from_fn(n, n, |i, j| if i == j { w_sr[i] } else { 0.0 });
    let m = pm.llt.solve(diag_w);
    let r = Array2::from_shape_fn((n, n), |(i, j)| w_sr[i] * m[(i, j)]);
    // C = L⁻¹ Wˢ K
    let wk = Mat::<f64>::from_fn(n, n, |i, j| w_sr[i] * k[[i, j]]);
    let c = solve_lower(pm.llt.L(), wk);
    let s2: Array1<f64> = Array1::from_shape_fn(n, |j| {
        let csq: f64 = (0..n).map(|i| c[(i, j)] * c[(i, j)]).sum();
        let p = pm.pi[j];
        -0.5 * (k[[j, j]] - csq) * (p * (1.0 - p) * (1.0 - 2.0 * p))
    });
    let resid = y - &pm.pi;
    let grad = grads
        .iter()
        .map(|cj| {
            let s1 = 0.5 * pm.a.dot(&cj.dot(&pm.a)) - 0.5 * (&r.t() * cj).sum();
            let b = cj.dot(&resid);
            let s3 = &b - &k.dot(&r.dot(&b));
            s1 + s2.dot(&s3)
        })
        .collect();
    (pm.lml, grad)
}

fn validate_xy(x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<()> {
    if y.len() != x.nrows() {
        return Err(RustMlError::ShapeMismatch(format!(
            "X has {} rows but y has {}",
            x.nrows(),
            y.len()
        )));
    }
    if x.nrows() == 0 {
        return Err(RustMlError::EmptyInput("training data is empty".into()));
    }
    Ok(())
}

fn sorted_classes(y: ArrayView1<f64>) -> Vec<f64> {
    let mut classes: Vec<f64> = y.iter().copied().collect();
    classes.sort_by(|a, b| a.total_cmp(b));
    classes.dedup();
    classes
}

impl Fit<f64> for GaussianProcessClassifier {
    type Fitted = FittedGaussianProcessClassifier;

    fn fit_view(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<Self::Fitted> {
        validate_xy(x, y)?;
        self.kernel.validate(x.ncols())?;
        let classes = sorted_classes(y);
        if classes.len() != 2 {
            return Err(RustMlError::InvalidParameter(format!(
                "GPC expects 2 classes, found {}",
                classes.len()
            )));
        }
        let (neg, pos) = (classes[0], classes[1]);
        let yb: Array1<f64> = y.mapv(|v| if v == pos { 1.0 } else { 0.0 });
        let (max_iter, tol) = (self.max_iter, self.tol);

        let kernel = if self.optimizer && self.kernel.n_free_params() > 0 {
            optimise_kernel(
                &self.kernel,
                self.n_restarts_optimizer,
                self.seed,
                |k, g| gpc_lml_with_grad(k, x, &yb, max_iter, tol, g),
            )?
            .0
        } else {
            self.kernel.clone()
        };

        let k = kernel.gram(x);
        let pm = posterior_mode(&k, &yb, max_iter, tol).ok_or_else(|| {
            RustMlError::InvalidParameter("Laplace posterior mode failed (non-finite)".into())
        })?;
        Ok(FittedGaussianProcessClassifier {
            x_train: x.to_owned(),
            alpha: &yb - &pm.pi,
            l_lower: pm.llt.L().to_owned(),
            w_sqrt: pm.w_sr,
            kernel,
            classes: [neg, pos],
            log_marginal_likelihood_value: pm.lml,
        })
    }
}

/// sklearn's logistic ≈ Σ_i c_i Φ(λ_i f) approximation constants.
const LAMBDAS: [f64; 5] = [0.41, 0.4, 0.37, 0.44, 0.39];
const COEFS: [f64; 5] = [
    -1854.8214151,
    3516.89893646,
    221.29346712,
    128.12323805,
    -2010.49422654,
];

/// `∫ σ(f) N(f | mean, var) df` via sklearn's erf-based approximation.
fn predictive_probability(mean: f64, var: f64) -> f64 {
    let var = var.max(f64::MIN_POSITIVE);
    let alpha = 1.0 / (2.0 * var);
    let mut p = 0.5 * COEFS.iter().sum::<f64>();
    for (&lam, &c) in LAMBDAS.iter().zip(COEFS.iter()) {
        let gamma = lam * mean;
        let integral = (std::f64::consts::PI / alpha).sqrt()
            * libm::erf(gamma * (alpha / (alpha + lam * lam)).sqrt())
            / (2.0 * (var * 2.0 * std::f64::consts::PI).sqrt());
        p += c * integral;
    }
    p
}

impl FittedGaussianProcessClassifier {
    /// Latent posterior mean and variance at query points (one batched
    /// triangular solve).
    pub fn latent_mean_var<'a>(
        &self,
        x: impl AsArray<'a, f64, Ix2>,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        let x: ArrayView2<f64> = x.into();
        if x.ncols() != self.x_train.ncols() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.x_train.ncols(),
                x.ncols()
            )));
        }
        let k_star = self.kernel.cross(self.x_train.view(), x); // (n_train, n_q)
        let mean = k_star.t().dot(&self.alpha);
        let ws_k = Mat::<f64>::from_fn(k_star.nrows(), k_star.ncols(), |i, j| {
            self.w_sqrt[i] * k_star[[i, j]]
        });
        let v = solve_lower(self.l_lower.as_ref(), ws_k);
        let kdiag = self.kernel.diag(x);
        let var = Array1::from_shape_fn(x.nrows(), |j| {
            let vsq: f64 = (0..v.nrows()).map(|i| v[(i, j)] * v[(i, j)]).sum();
            kdiag[j] - vsq
        });
        Ok((mean, var))
    }

    /// Approximate log-marginal likelihood of the fitted kernel.
    pub fn log_marginal_likelihood_value(&self) -> f64 {
        self.log_marginal_likelihood_value
    }

    /// Fitted log-hyperparameters (sklearn `kernel_.theta`).
    pub fn theta(&self) -> Vec<f64> {
        self.kernel.theta()
    }
}

/// Approximate LML and gradient for binary labels `y` at log-hyperparameters
/// `theta` of `kernel` (sklearn `GaussianProcessClassifier.log_marginal_likelihood(theta,
/// eval_gradient=True)`); `y` must contain exactly two classes.
pub fn classifier_log_marginal_likelihood(
    x: &Array2<f64>,
    y: &Array1<f64>,
    kernel: &GpKernel,
    theta: &[f64],
) -> Result<(f64, Vec<f64>)> {
    validate_xy(x.view(), y.view())?;
    kernel.validate(x.ncols())?;
    let classes = sorted_classes(y.view());
    if classes.len() != 2 {
        return Err(RustMlError::InvalidParameter(format!(
            "GPC expects 2 classes, found {}",
            classes.len()
        )));
    }
    let yb = y.mapv(|v| if v == classes[1] { 1.0 } else { 0.0 });
    let k = kernel.with_theta(theta)?;
    Ok(gpc_lml_with_grad(&k, x.view(), &yb, 100, 1e-10, true))
}

impl Predict<f64> for FittedGaussianProcessClassifier {
    fn predict_view(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        // sklearn: sign of the latent mean.
        let (mean, _) = self.latent_mean_var(x)?;
        Ok(mean.mapv(|f| {
            if f > 0.0 {
                self.classes[1]
            } else {
                self.classes[0]
            }
        }))
    }
}

impl PredictProba<f64> for FittedGaussianProcessClassifier {
    fn predict_proba_view(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        let (mean, var) = self.latent_mean_var(x)?;
        let n = mean.len();
        let mut out = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let p1 = predictive_probability(mean[i], var[i]);
            out[[i, 0]] = 1.0 - p1;
            out[[i, 1]] = p1;
        }
        Ok(out)
    }
}

impl anofox_ml_core::ClassifierScore<f64> for FittedGaussianProcessClassifier {}

// ---------------------------------------------------------------------------
// Multi-class GPC via one-vs-rest.
// ---------------------------------------------------------------------------

/// Multi-class Gaussian Process Classifier built as a one-vs-rest stack of
/// binary `GaussianProcessClassifier` instances. Mirrors sklearn's
/// `GaussianProcessClassifier(multi_class='one_vs_rest')`; with exactly two
/// classes a single binary classifier is fitted, as in sklearn.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MulticlassGaussianProcessClassifier {
    pub kernel: GpKernel,
    pub max_iter: usize,
    pub tol: f64,
    pub optimizer: bool,
    pub n_restarts_optimizer: usize,
    pub seed: u64,
}

impl MulticlassGaussianProcessClassifier {
    pub fn new(kernel: GpKernel) -> Self {
        Self {
            kernel,
            max_iter: 100,
            tol: 1e-10,
            optimizer: true,
            n_restarts_optimizer: 0,
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

/// Fitted one-vs-rest Gaussian process classifier.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedMulticlassGaussianProcessClassifier {
    pub classes: Vec<f64>,
    /// One binary model per class (or a single one for two classes).
    pub binary: Vec<FittedGaussianProcessClassifier>,
}

impl FittedMulticlassGaussianProcessClassifier {
    /// Mean of the binary models' log-marginal likelihoods (sklearn).
    pub fn log_marginal_likelihood_value(&self) -> f64 {
        self.binary
            .iter()
            .map(|b| b.log_marginal_likelihood_value)
            .sum::<f64>()
            / self.binary.len() as f64
    }
}

impl Fit<f64> for MulticlassGaussianProcessClassifier {
    type Fitted = FittedMulticlassGaussianProcessClassifier;

    fn fit_view(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<Self::Fitted> {
        validate_xy(x, y)?;
        let classes = sorted_classes(y);
        if classes.len() < 2 {
            return Err(RustMlError::InvalidParameter(format!(
                "multi-class GPC needs ≥2 classes, found {}",
                classes.len()
            )));
        }
        let inner = GaussianProcessClassifier {
            kernel: self.kernel.clone(),
            max_iter: self.max_iter,
            tol: self.tol,
            optimizer: self.optimizer,
            n_restarts_optimizer: self.n_restarts_optimizer,
            seed: self.seed,
        };
        let binary = if classes.len() == 2 {
            vec![inner.fit(x, y)?]
        } else {
            let mut v = Vec::with_capacity(classes.len());
            for &c in &classes {
                let y_bin: Array1<f64> = y.mapv(|v| if v == c { 1.0 } else { 0.0 });
                v.push(inner.fit(x, &y_bin)?);
            }
            v
        };
        Ok(FittedMulticlassGaussianProcessClassifier { classes, binary })
    }
}

impl Predict<f64> for FittedMulticlassGaussianProcessClassifier {
    fn predict_view(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        if self.binary.len() == 1 {
            return self.binary[0].predict(x);
        }
        let proba = self.predict_proba(x)?;
        Ok(Array1::from_shape_fn(x.nrows(), |i| {
            let mut best = 0;
            for c in 1..self.classes.len() {
                if proba[[i, c]] > proba[[i, best]] {
                    best = c;
                }
            }
            self.classes[best]
        }))
    }
}

impl PredictProba<f64> for FittedMulticlassGaussianProcessClassifier {
    fn predict_proba_view(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        if self.binary.len() == 1 {
            return self.binary[0].predict_proba(x);
        }
        let n = x.nrows();
        let k = self.classes.len();
        let mut p = Array2::<f64>::zeros((n, k));
        for c in 0..k {
            let pc = self.binary[c].predict_proba(x)?;
            for i in 0..n {
                p[[i, c]] = pc[[i, 1]];
            }
        }
        // Renormalise rows to sum to 1 (sklearn does the same for OvR).
        for i in 0..n {
            let s: f64 = (0..k).map(|c| p[[i, c]]).sum::<f64>().max(1e-12);
            for c in 0..k {
                p[[i, c]] /= s;
            }
        }
        Ok(p)
    }
}

impl anofox_ml_core::ClassifierScore<f64> for FittedMulticlassGaussianProcessClassifier {}
