//! Factor analysis (sklearn's `FactorAnalysis`): a Gaussian latent-variable
//! model `x = W z + μ + ε`, `ε ~ N(0, diag(ψ))`, fitted by sklearn's
//! SVD-based EM, with optional varimax / quartimax rotation.
//!
//! The SVD is exact (sklearn's `svd_method="lapack"`). Each component's
//! sign follows the SVD and may differ from sklearn's; the covariance
//! `WᵀW + diag(ψ)`, the noise variances and the likelihood do not.

use anofox_ml_core::{FitUnsupervised, Result, RustMlError, Transform};
use faer::linalg::solvers::Svd;
use faer::Mat;
use ndarray::{Array1, Array2, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

/// Orthogonal rotation of the loadings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FactorRotation {
    Varimax,
    Quartimax,
}

/// Factor analysis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FactorAnalysis {
    /// Number of factors (`None`: the number of features).
    pub n_components: Option<usize>,
    pub tol: f64,
    pub max_iter: usize,
    /// Initial noise variances (`None`: ones).
    pub noise_variance_init: Option<Vec<f64>>,
    pub rotation: Option<FactorRotation>,
}

impl Default for FactorAnalysis {
    fn default() -> Self {
        Self {
            n_components: None,
            tol: 1e-2,
            max_iter: 1000,
            noise_variance_init: None,
            rotation: None,
        }
    }
}

/// Fitted [`FactorAnalysis`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedFactorAnalysis {
    mean: Array1<f64>,
    /// `n_components × n_features` loadings (sklearn's `components_`).
    components: Array2<f64>,
    noise_variance: Array1<f64>,
    loglike: Vec<f64>,
    n_iter: usize,
}

/// Thin SVD `(s, Vt)` of `a`, singular values descending.
fn svd(a: &Array2<f64>) -> Result<(Vec<f64>, Array2<f64>, Array2<f64>)> {
    let (r, c) = a.dim();
    let res = Svd::new_thin(Mat::<f64>::from_fn(r, c, |i, j| a[[i, j]]).as_ref())
        .map_err(|e| RustMlError::InvalidParameter(format!("SVD failed: {e:?}")))?;
    let s = res.S().column_vector();
    let k = s.nrows();
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&x, &y| s[y].total_cmp(&s[x]));
    let vals = order.iter().map(|&i| s[i]).collect();
    let vt = Array2::from_shape_fn((k, c), |(row, j)| res.V()[(j, order[row])]);
    let u = Array2::from_shape_fn((r, k), |(i, col)| res.U()[(i, order[col])]);
    Ok((vals, vt, u))
}

/// sklearn's `_ortho_rotation` on `components` (features × factors);
/// returns factors × features.
fn ortho_rotation(components: &Array2<f64>, method: FactorRotation) -> Result<Array2<f64>> {
    let (nrow, ncol) = components.dim();
    let mut rotation = Array2::<f64>::eye(ncol);
    let mut var = 0.0;
    for _ in 0..100 {
        let comp_rot = components.dot(&rotation);
        let cubed = comp_rot.mapv(|v| v.powi(3));
        let target = match method {
            FactorRotation::Varimax => {
                let col = (comp_rot.mapv(|v| v * v).sum_axis(Axis(0))) / nrow as f64;
                cubed - &(&comp_rot * &col)
            }
            FactorRotation::Quartimax => cubed,
        };
        let (s, vt, u) = svd(&components.t().dot(&target))?;
        rotation = u.dot(&vt);
        let var_new: f64 = s.iter().sum();
        if var != 0.0 && var_new < var * (1.0 + 1e-6) {
            break;
        }
        var = var_new;
    }
    Ok(components.dot(&rotation).reversed_axes())
}

impl FitUnsupervised<f64> for FactorAnalysis {
    type Fitted = FittedFactorAnalysis;

    fn fit_view(&self, x: ArrayView2<f64>) -> Result<FittedFactorAnalysis> {
        let (n, p) = x.dim();
        if n < 2 || p == 0 {
            return Err(RustMlError::EmptyInput(
                "need at least 2 rows and one column".into(),
            ));
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(RustMlError::InvalidParameter("x must be finite".into()));
        }
        let k = self.n_components.unwrap_or(p);
        if k == 0 || k > p {
            return Err(RustMlError::InvalidParameter(format!(
                "n_components must be in [1, {p}], got {k}"
            )));
        }
        let mean = x.mean_axis(Axis(0)).expect("non-empty");
        let xc = &x - &mean;
        let nsqrt = (n as f64).sqrt();
        let llconst = p as f64 * (2.0 * std::f64::consts::PI).ln() + k as f64;
        let var = xc.mapv(|v| v * v).mean_axis(Axis(0)).expect("non-empty");
        let mut psi = match &self.noise_variance_init {
            None => Array1::ones(p),
            Some(v) if v.len() == p => Array1::from(v.clone()),
            Some(v) => {
                return Err(RustMlError::InvalidParameter(format!(
                    "noise_variance_init has {} entries for {p} features",
                    v.len()
                )))
            }
        };
        const SMALL: f64 = 1e-12;
        let mut loglike = Vec::new();
        let mut old_ll = f64::NEG_INFINITY;
        let mut w = Array2::zeros((k, p));
        let mut n_iter = 0;
        for i in 0..self.max_iter {
            n_iter = i + 1;
            let sqrt_psi = psi.mapv(f64::sqrt) + SMALL;
            let scaled = &xc / &(&sqrt_psi * nsqrt);
            let (s_all, vt, _) = svd(&scaled)?;
            let s: Vec<f64> = s_all.iter().take(k).map(|v| v * v).collect();
            let unexp: f64 = s_all.iter().skip(k).map(|v| v * v).sum();
            w = Array2::from_shape_fn((k, p), |(r, j)| {
                (s[r] - 1.0).max(0.0).sqrt() * vt[[r, j]] * sqrt_psi[j]
            });
            let mut ll = llconst + s.iter().map(|v| v.ln()).sum::<f64>();
            ll += unexp + psi.iter().map(|v| v.ln()).sum::<f64>();
            ll *= -(n as f64) / 2.0;
            loglike.push(ll);
            if ll - old_ll < self.tol {
                break;
            }
            old_ll = ll;
            psi = (&var - &w.mapv(|v| v * v).sum_axis(Axis(0))).mapv(|v| v.max(SMALL));
        }
        let components = match self.rotation {
            None => w,
            Some(method) => ortho_rotation(&w.t().to_owned(), method)?,
        };
        Ok(FittedFactorAnalysis {
            mean,
            components,
            noise_variance: psi,
            loglike,
            n_iter,
        })
    }
}

/// Inverse of a small symmetric positive-definite matrix (via its SVD).
fn inv(a: &Array2<f64>) -> Result<Array2<f64>> {
    let (s, vt, u) = svd(a)?;
    if s.iter().any(|&v| v <= 0.0) {
        return Err(RustMlError::InvalidParameter("singular matrix".into()));
    }
    let n = a.nrows();
    Ok(Array2::from_shape_fn((n, n), |(i, j)| {
        (0..s.len()).map(|c| vt[[c, i]] * u[[j, c]] / s[c]).sum()
    }))
}

impl FittedFactorAnalysis {
    pub fn mean(&self) -> &Array1<f64> {
        &self.mean
    }

    pub fn components(&self) -> &Array2<f64> {
        &self.components
    }

    pub fn noise_variance(&self) -> &Array1<f64> {
        &self.noise_variance
    }

    /// Log-likelihood after each EM iteration.
    pub fn loglike(&self) -> &[f64] {
        &self.loglike
    }

    pub fn n_iter(&self) -> usize {
        self.n_iter
    }

    /// Model covariance `WᵀW + diag(ψ)`.
    pub fn covariance(&self) -> Array2<f64> {
        let mut c = self.components.t().dot(&self.components);
        for (i, v) in self.noise_variance.iter().enumerate() {
            c[[i, i]] += v;
        }
        c
    }

    /// Inverse of [`covariance`](Self::covariance), by the matrix inversion
    /// lemma as sklearn's `get_precision`.
    pub fn precision(&self) -> Result<Array2<f64>> {
        let (k, p) = self.components.dim();
        if k == p {
            return inv(&self.covariance());
        }
        let w = &self.components;
        let psi = &self.noise_variance;
        let wpsi = w / psi;
        let mut inner = wpsi.dot(&w.t());
        for i in 0..k {
            inner[[i, i]] += 1.0;
        }
        let mut prec = w.t().dot(&inv(&inner)?.dot(w));
        for i in 0..p {
            for j in 0..p {
                prec[[i, j]] /= -(psi[i] * psi[j]);
            }
            prec[[i, i]] += 1.0 / psi[i];
        }
        Ok(prec)
    }

    /// Per-row Gaussian log-likelihood under the model.
    pub fn score_samples(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        self.check(x)?;
        let xr = &x - &self.mean;
        let prec = self.precision()?;
        let (s, _, _) = svd(&prec)?;
        let logdet: f64 = s.iter().map(|v| v.ln()).sum();
        let p = x.ncols() as f64;
        let quad = (&xr * &xr.dot(&prec)).sum_axis(Axis(1));
        Ok(quad.mapv(|q| -0.5 * q - 0.5 * (p * (2.0 * std::f64::consts::PI).ln() - logdet)))
    }

    /// Mean per-row log-likelihood (sklearn's `score`).
    pub fn score(&self, x: ArrayView2<f64>) -> Result<f64> {
        Ok(self.score_samples(x)?.mean().expect("non-empty"))
    }

    fn check(&self, x: ArrayView2<f64>) -> Result<()> {
        if x.ncols() != self.mean.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.mean.len(),
                x.ncols()
            )));
        }
        Ok(())
    }
}

impl Transform<f64> for FittedFactorAnalysis {
    /// Posterior means of the factors.
    fn transform_view(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        self.check(x)?;
        let k = self.components.nrows();
        let wpsi = &self.components / &self.noise_variance;
        let mut m = wpsi.dot(&self.components.t());
        for i in 0..k {
            m[[i, i]] += 1.0;
        }
        Ok((&x - &self.mean).dot(&wpsi.t()).dot(&inv(&m)?))
    }
}
