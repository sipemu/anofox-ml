//! Non-negative Matrix Factorisation.
//!
//! Mirrors `sklearn.decomposition.NMF`: `X ≈ W H` with `W ≥ 0`, `H ≥ 0`,
//! fitted by coordinate descent (`solver='cd'`, default, Frobenius loss) or
//! multiplicative updates (`solver='mu'`, any of the Frobenius,
//! Kullback-Leibler and Itakura-Saito β-divergences), with sklearn's
//! `alpha_W` / `alpha_H` / `l1_ratio` regularisation.

use anofox_ml_core::{FitUnsupervised, Result, RustMlError, Transform};
use faer::linalg::solvers::Svd;
use faer::Mat;
use ndarray::{Array2, ArrayView2, AsArray, Ix2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Initialisation of `W` and `H` (sklearn `init`).
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub enum NmfInit {
    /// sklearn's `init=None`: `NndsvdA` when `n_components <= min(n_samples,
    /// n_features)`, else `Random`.
    #[default]
    Auto,
    /// `|N(0, 1)| · sqrt(mean(X) / n_components)` entries.
    Random,
    /// NNDSVD from the truncated SVD (exact zeros kept).
    Nndsvd,
    /// NNDSVD with its zeros filled by `mean(X)`.
    NndsvdA,
    /// NNDSVD with its zeros filled by small random values
    /// `|N(0, 1)| · mean(X) / 100`.
    NndsvdAr,
}

/// |N(0, 1)| draws (Box–Muller).
fn abs_normal(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(f64::MIN_POSITIVE);
    let u2: f64 = rng.gen();
    ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()).abs()
}

/// NMF solver (sklearn `solver`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum NmfSolver {
    /// Coordinate descent (Frobenius loss only).
    #[default]
    Cd,
    /// Multiplicative updates (any [`BetaLoss`]).
    Mu,
}

/// β-divergence minimised by the `Mu` solver (sklearn `beta_loss`).
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub enum BetaLoss {
    /// β = 2.
    #[default]
    Frobenius,
    /// β = 1.
    KullbackLeibler,
    /// β = 0.
    ItakuraSaito,
}

impl BetaLoss {
    fn beta(self) -> f64 {
        match self {
            BetaLoss::Frobenius => 2.0,
            BetaLoss::KullbackLeibler => 1.0,
            BetaLoss::ItakuraSaito => 0.0,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Nmf {
    pub n_components: usize,
    pub max_iter: usize,
    pub tol: f64,
    pub seed: u64,
    pub init: NmfInit,
    pub solver: NmfSolver,
    pub beta_loss: BetaLoss,
    /// sklearn `alpha_W` (penalty on W, scaled by `n_features`).
    pub alpha_w: f64,
    /// sklearn `alpha_H` (penalty on H, scaled by `n_samples`); `None` =
    /// same as `alpha_w`.
    pub alpha_h: Option<f64>,
    /// Elastic-net mixing of the penalties (0 = L2, 1 = L1).
    pub l1_ratio: f64,
}

impl Nmf {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            max_iter: 200,
            tol: 1e-4,
            seed: 0,
            init: NmfInit::Auto,
            solver: NmfSolver::Cd,
            beta_loss: BetaLoss::Frobenius,
            alpha_w: 0.0,
            alpha_h: None,
            l1_ratio: 0.0,
        }
    }
    pub fn with_init(mut self, init: NmfInit) -> Self {
        self.init = init;
        self
    }
    pub fn with_solver(mut self, solver: NmfSolver) -> Self {
        self.solver = solver;
        self
    }
    pub fn with_beta_loss(mut self, beta_loss: BetaLoss) -> Self {
        self.beta_loss = beta_loss;
        self
    }
    pub fn with_alpha(mut self, alpha_w: f64, alpha_h: Option<f64>, l1_ratio: f64) -> Self {
        self.alpha_w = alpha_w;
        self.alpha_h = alpha_h;
        self.l1_ratio = l1_ratio;
        self
    }
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// `(l1_W, l2_W, l1_H, l2_H)` as sklearn `_compute_regularization`.
    fn regularization(&self, n: usize, d: usize) -> (f64, f64, f64, f64) {
        let ah = self.alpha_h.unwrap_or(self.alpha_w);
        let (nf, df) = (n as f64, d as f64);
        (
            df * self.alpha_w * self.l1_ratio,
            df * self.alpha_w * (1.0 - self.l1_ratio),
            nf * ah * self.l1_ratio,
            nf * ah * (1.0 - self.l1_ratio),
        )
    }
}

/// NNDSVD initialisation (Boutsidis & Gallopoulos 2008).
///
/// 1. Compute the truncated SVD `X ≈ U Σ Vᵀ` keeping `k` triplets.
/// 2. The first component is initialised from the leading singular triplet
///    with positive sign (sign-fix).
/// 3. Each subsequent component splits the next singular vectors into their
///    positive and negative parts and picks whichever has higher norm.
fn nndsvd_init(x: ArrayView2<f64>, k: usize) -> Result<(Array2<f64>, Array2<f64>)> {
    let n = x.nrows();
    let d = x.ncols();
    let mat = Mat::<f64>::from_fn(n, d, |i, j| x[[i, j]]);
    let svd = Svd::new_thin(mat.as_ref())
        .map_err(|e| RustMlError::InvalidParameter(format!("NNDSVD SVD failed: {e:?}")))?;
    let u = svd.U();
    let s = svd.S();
    let v = svd.V();
    let r = s.column_vector().nrows().min(k);

    let mut w = Array2::<f64>::zeros((n, k));
    let mut h = Array2::<f64>::zeros((k, d));

    // First component: the leading singular vectors are of one sign, so
    // their absolute values (sklearn: sqrt(S[0]) * |U[:, 0]|, |V[0, :]|).
    let lead_scale = s.column_vector()[0].max(0.0).sqrt();
    for i in 0..n {
        w[[i, 0]] = u[(i, 0)].abs() * lead_scale;
    }
    for j in 0..d {
        h[[0, j]] = v[(j, 0)].abs() * lead_scale;
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
    // sklearn: entries below 1e-6 are set to exactly 0.
    let eps = 1e-6;
    w.mapv_inplace(|v| if v < eps { 0.0 } else { v });
    h.mapv_inplace(|v| if v < eps { 0.0 } else { v });
    Ok((w, h))
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedNmf {
    /// Components, shape (n_components, n_features) — sklearn's `components_`.
    pub components: Array2<f64>,
    /// Factor of the training data, shape (n_samples, n_components)
    /// (sklearn `fit_transform(X)`).
    #[serde(default)]
    pub w: Array2<f64>,
    /// β-divergence (square-rooted, as sklearn `reconstruction_err_`).
    pub reconstruction_err: f64,
    pub n_iter: usize,
    #[serde(skip)]
    params: Option<NmfParams>,
}

/// What `transform` needs to re-run the solver with H fixed.
#[derive(Debug, Clone, Copy)]
struct NmfParams {
    solver: NmfSolver,
    beta: f64,
    max_iter: usize,
    tol: f64,
    l1_w: f64,
    l2_w: f64,
}

const EPSILON: f64 = f64::EPSILON;

/// β-divergence `D(X || W H)` (sklearn `_beta_divergence`, square-rooted
/// for β = 2 as sklearn reports it).
fn beta_divergence(x: ArrayView2<f64>, w: &Array2<f64>, h: &Array2<f64>, beta: f64) -> f64 {
    if beta == 2.0 {
        let x_norm = x.mapv(|v| v * v).sum();
        let norm_wh = (w.t().dot(w) * h.dot(&h.t())).sum();
        let cross = (x.dot(&h.t()) * w).sum();
        let res = (x_norm + norm_wh - 2.0 * cross) / 2.0;
        return (2.0 * res.max(0.0)).sqrt();
    }
    let wh = w.dot(h);
    let mut total = 0.0;
    for (&xv, &whv) in x.iter().zip(wh.iter()) {
        let whv = whv.max(EPSILON);
        if beta == 1.0 {
            total += if xv > EPSILON {
                xv * (xv / whv).ln()
            } else {
                0.0
            } - xv
                + whv;
        } else {
            let xv = xv.max(EPSILON);
            total += xv / whv - (xv / whv).ln() - 1.0;
        }
    }
    (2.0 * total).sqrt()
}

/// One CD sweep over `W` (sklearn `_update_coordinate_descent`); returns the
/// projected-gradient violation.
fn cd_update(x: ArrayView2<f64>, w: &mut Array2<f64>, ht: &Array2<f64>, l1: f64, l2: f64) -> f64 {
    let k = w.ncols();
    let mut hht = ht.t().dot(ht);
    let mut xht = x.dot(ht);
    for t in 0..k {
        hht[[t, t]] += l2;
    }
    if l1 != 0.0 {
        xht.mapv_inplace(|v| v - l1);
    }
    let mut violation = 0.0;
    for t in 0..k {
        let hess = hht[[t, t]];
        for i in 0..w.nrows() {
            let mut grad = -xht[[i, t]];
            for r in 0..k {
                grad += hht[[t, r]] * w[[i, r]];
            }
            let pg = if w[[i, t]] == 0.0 {
                grad.min(0.0)
            } else {
                grad
            };
            violation += pg.abs();
            if hess != 0.0 {
                w[[i, t]] = (w[[i, t]] - grad / hess).max(0.0);
            }
        }
    }
    violation
}

fn gamma(beta: f64) -> f64 {
    if beta < 1.0 {
        1.0 / (2.0 - beta)
    } else if beta > 2.0 {
        1.0 / (beta - 1.0)
    } else {
        1.0
    }
}

/// Multiplicative update of `W` for `X ≈ W H` (sklearn
/// `_multiplicative_update_w`).
fn mu_update_w(
    x: ArrayView2<f64>,
    w: &mut Array2<f64>,
    h: &Array2<f64>,
    beta: f64,
    l1: f64,
    l2: f64,
) {
    let (numerator, mut denominator) = if beta == 2.0 {
        (x.dot(&h.t()), w.dot(&h.dot(&h.t())))
    } else {
        let wh = w.dot(h).mapv(|v| v.max(EPSILON));
        let num = if beta == 1.0 {
            (&x / &wh).dot(&h.t())
        } else {
            (&x * &wh.mapv(|v| v.powf(beta - 2.0))).dot(&h.t())
        };
        let den = if beta == 1.0 {
            let hsum = h.sum_axis(ndarray::Axis(1));
            Array2::from_shape_fn(w.dim(), |(_, j)| hsum[j])
        } else {
            wh.mapv(|v| v.powf(beta - 1.0)).dot(&h.t())
        };
        (num, den)
    };
    if l1 > 0.0 {
        denominator.mapv_inplace(|v| v + l1);
    }
    if l2 > 0.0 {
        denominator = denominator + &(w.mapv(|v| v * l2));
    }
    denominator.mapv_inplace(|v| if v == 0.0 { EPSILON } else { v });
    let g = gamma(beta);
    ndarray::Zip::from(w)
        .and(&numerator)
        .and(&denominator)
        .for_each(|wv, &n, &d| {
            let delta = n / d;
            *wv *= if g != 1.0 { delta.powf(g) } else { delta };
        });
}

/// Run the solver. With `update_h = false` only `W` is updated.
#[allow(clippy::too_many_arguments)]
fn solve_nmf(
    x: ArrayView2<f64>,
    w: &mut Array2<f64>,
    h: &mut Array2<f64>,
    p: &NmfParams,
    l1_h: f64,
    l2_h: f64,
    update_h: bool,
) -> usize {
    let mut n_iter = 0;
    match p.solver {
        NmfSolver::Cd => {
            let xt = x.t().to_owned();
            let mut ht = h.t().to_owned();
            let mut violation_init = 0.0;
            for it in 1..=p.max_iter {
                n_iter = it;
                let mut violation = cd_update(x, w, &ht, p.l1_w, p.l2_w);
                if update_h {
                    violation += cd_update(xt.view(), &mut ht, w, l1_h, l2_h);
                }
                if it == 1 {
                    violation_init = violation;
                }
                if violation_init == 0.0 || violation / violation_init <= p.tol {
                    break;
                }
            }
            if update_h {
                *h = ht.t().to_owned();
            }
        }
        NmfSolver::Mu => {
            let err_init = beta_divergence(x, w, h, p.beta);
            let mut previous = err_init;
            for it in 1..=p.max_iter {
                n_iter = it;
                mu_update_w(x, w, h, p.beta, p.l1_w, p.l2_w);
                if update_h {
                    // H update = W update on the transposed problem.
                    let mut ht = h.t().to_owned();
                    let wt = w.t().to_owned();
                    mu_update_w(x.t().to_owned().view(), &mut ht, &wt, p.beta, l1_h, l2_h);
                    *h = ht.t().to_owned();
                }
                if p.tol > 0.0 && it % 10 == 0 {
                    let err = beta_divergence(x, w, h, p.beta);
                    if (previous - err) / err_init < p.tol {
                        break;
                    }
                    previous = err;
                }
            }
        }
    }
    n_iter
}

impl FitUnsupervised<f64> for Nmf {
    type Fitted = FittedNmf;

    fn fit_view(&self, x: ArrayView2<f64>) -> Result<Self::Fitted> {
        let n = x.nrows();
        let d = x.ncols();
        let k = self.n_components;
        if n == 0 || d == 0 {
            return Err(RustMlError::EmptyInput("empty input".into()));
        }
        let init = match self.init {
            NmfInit::Auto if k <= d.min(n) => NmfInit::NndsvdA,
            NmfInit::Auto => NmfInit::Random,
            other => other,
        };
        if k == 0 || (k > d.min(n) && init != NmfInit::Random) {
            return Err(RustMlError::InvalidParameter(format!(
                "n_components must be in 1..={} for an NNDSVD init",
                d.min(n)
            )));
        }
        if x.iter().any(|v| !(*v >= 0.0)) {
            return Err(RustMlError::InvalidParameter("NMF requires X >= 0".into()));
        }
        if self.solver == NmfSolver::Cd && self.beta_loss != BetaLoss::Frobenius {
            return Err(RustMlError::InvalidParameter(
                "the Cd solver only supports the Frobenius loss; use NmfSolver::Mu".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.l1_ratio) || self.alpha_w < 0.0 {
            return Err(RustMlError::InvalidParameter(
                "alpha must be >= 0 and l1_ratio in [0, 1]".into(),
            ));
        }
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mean = x.mean().unwrap_or(0.0);
        let (mut w, mut h) = match init {
            NmfInit::Random => {
                let avg = (mean / k as f64).sqrt();
                // sklearn draws H first, then W.
                let h = Array2::<f64>::from_shape_fn((k, d), |_| avg * abs_normal(&mut rng));
                let w = Array2::<f64>::from_shape_fn((n, k), |_| avg * abs_normal(&mut rng));
                (w, h)
            }
            _ => {
                let (mut w, mut h) = nndsvd_init(x, k)?;
                match init {
                    NmfInit::NndsvdA => {
                        w.mapv_inplace(|v| if v == 0.0 { mean } else { v });
                        h.mapv_inplace(|v| if v == 0.0 { mean } else { v });
                    }
                    NmfInit::NndsvdAr => {
                        for v in w.iter_mut().chain(h.iter_mut()) {
                            if *v == 0.0 {
                                *v = mean * abs_normal(&mut rng) / 100.0;
                            }
                        }
                    }
                    _ => {}
                }
                (w, h)
            }
        };
        let (l1_w, l2_w, l1_h, l2_h) = self.regularization(n, d);
        let params = NmfParams {
            solver: self.solver,
            beta: self.beta_loss.beta(),
            max_iter: self.max_iter,
            tol: self.tol,
            l1_w,
            l2_w,
        };
        let n_iter = solve_nmf(x, &mut w, &mut h, &params, l1_h, l2_h, true);
        Ok(FittedNmf {
            reconstruction_err: beta_divergence(x, &w, &h, params.beta),
            components: h,
            w,
            n_iter,
            params: Some(params),
        })
    }
}

impl FittedNmf {
    /// `W` for new data with `H` fixed (sklearn `transform`): the solver is
    /// re-run from `W = sqrt(mean(X) / k)` with the fitted settings.
    pub fn transform<'a>(
        &self,
        x: impl AsArray<'a, f64, Ix2>,
        max_iter: usize,
    ) -> Result<Array2<f64>> {
        let x: ArrayView2<f64> = x.into();
        let h = &self.components;
        if x.ncols() != h.ncols() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                h.ncols(),
                x.ncols()
            )));
        }
        if x.iter().any(|v| !(*v >= 0.0)) {
            return Err(RustMlError::InvalidParameter("NMF requires X >= 0".into()));
        }
        let k = h.nrows();
        let mut p = self.params.unwrap_or(NmfParams {
            solver: NmfSolver::Cd,
            beta: 2.0,
            max_iter,
            tol: 1e-4,
            l1_w: 0.0,
            l2_w: 0.0,
        });
        p.max_iter = max_iter;
        let avg = (x.mean().unwrap_or(0.0) / k as f64).sqrt();
        let mut w = Array2::<f64>::from_elem((x.nrows(), k), avg);
        let mut hh = h.clone();
        solve_nmf(x, &mut w, &mut hh, &p, 0.0, 0.0, false);
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

impl Transform<f64> for FittedNmf {
    fn transform_view(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        let max_iter = self.params.map_or(200, |p| p.max_iter);
        FittedNmf::transform(self, x, max_iter)
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
        let w = fitted.transform(x.view(), 200).unwrap();
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

    #[test]
    fn test_solvers_match_sklearn() {
        let xv = [
            1.764, 0.4, 0.979, 2.241, 1.868, 0.977, 0.95, 0.151, 0.103, 0.411, 0.144, 1.454, 0.761,
            0.122, 0.444, 0.334, 1.494, 0.205, 0.313, 0.854, 2.553, 0.654, 0.864, 0.742, 2.27,
            1.454, 0.046, 0.187, 1.533, 1.469, 0.155, 0.378, 0.888, 1.981, 0.348, 0.156, 1.23,
            1.202, 0.387, 0.302,
        ];
        let x = Array2::from_shape_vec((8, 5), xv.to_vec()).unwrap();
        // sklearn 1.8 NMF(2, init='nndsvd', tol=1e-12, max_iter=2000, ...)
        let cases: [(Nmf, f64, [f64; 10]); 3] = [
            (
                Nmf::new(2),
                2.3547984486078186,
                [
                    1.4362113578348492,
                    0.0,
                    0.42696912776441526,
                    1.324869585184489,
                    1.3483431264201824,
                    0.1018635231086956,
                    1.5647533353747294,
                    0.7392064539390233,
                    0.0,
                    0.3206394608238117,
                ],
            ),
            (
                Nmf::new(2).with_alpha(0.01, None, 0.5),
                2.3564328019445773,
                [
                    1.233331072917809,
                    0.0,
                    0.36909420303780943,
                    1.1340079292247245,
                    1.1587260385266422,
                    0.08733210155457137,
                    1.422908804934947,
                    0.6659047755115884,
                    0.0,
                    0.28768271528304107,
                ],
            ),
            (
                Nmf::new(2)
                    .with_solver(NmfSolver::Mu)
                    .with_beta_loss(BetaLoss::KullbackLeibler),
                2.7690149876017163,
                [
                    1.3800641038026966,
                    0.3437565073395395,
                    0.6211166965579856,
                    1.3590070025771017,
                    1.4586162466355694,
                    0.0,
                    1.5430304828735824,
                    0.6020119561669455,
                    0.0,
                    0.0,
                ],
            ),
        ];
        for (i, (m, err, comps)) in cases.into_iter().enumerate() {
            let f = m
                .with_init(NmfInit::Nndsvd)
                .with_tol(1e-12)
                .with_max_iter(2000)
                .fit(&x)
                .unwrap();
            assert!(
                (f.reconstruction_err - err).abs() < 1e-5,
                "case {i}: err {}",
                f.reconstruction_err
            );
            for (j, (&got, &want)) in f.components.iter().zip(comps.iter()).enumerate() {
                assert!(
                    (got - want).abs() < 1e-4,
                    "case {i} comp {j}: {got} vs {want}"
                );
            }
            assert_eq!(f.w.dim(), (8, 2));
        }
        assert!(Nmf::new(2)
            .with_beta_loss(BetaLoss::KullbackLeibler)
            .fit(&x)
            .is_err());

        // Default init (sklearn's init=None -> 'nndsvda'):
        // sklearn 1.8 NMF(2, tol=1e-12, max_iter=2000).fit(X)
        let f = Nmf::new(2)
            .with_tol(1e-12)
            .with_max_iter(2000)
            .fit(&x)
            .unwrap();
        assert!((f.reconstruction_err - 2.354_798_448_607_818_6).abs() < 1e-8);
        let want = [
            2.445_001_341_372_870_4,
            0.0,
            0.726_870_794_058_906_8,
            2.255_453_485_478_836_4,
            2.295_414_762_418_369,
            0.234_083_107_423_126_27,
            1.665_551_707_037_105_5,
            0.824_181_298_317_364_3,
            0.115_916_162_992_167_49,
            0.459_264_354_306_825_5,
        ];
        for (got, w) in f.components.iter().zip(want) {
            assert!((got - w).abs() < 1e-8, "{got} vs {w}");
        }
    }
}
