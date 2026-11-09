//! Mini-batch NMF (sklearn's `decomposition.MiniBatchNMF`): online
//! multiplicative updates over contiguous mini-batches.
//!
//! sklearn's `_fit_transform`: `W` and `H` from `_initialize_nmf` (NNDSVDa by
//! default), then for `max_iter · ⌈n / batch_size⌉` steps over the batches
//! in order: a multiplicative update of the batch's `W` rows, the batch cost,
//! and the online `H` update `H = A / B` with `A, B` the decayed (`ρ =
//! forget_factor^(batch_size / n)`) accumulated numerators and
//! denominators. Training stops when `‖ΔH‖ / ‖H‖ <= tol` or the
//! exponentially weighted batch cost has not improved for
//! `max_no_improvement` steps. `transform` solves for `W` with `H` fixed
//! (sklearn's `_solve_W`).

use crate::nmf::{beta_divergence, gamma, initial_wh, mu_update_w, BetaLoss, NmfInit};
use anofox_ml_core::{FitUnsupervised, Result, RustMlError, Transform};
use ndarray::{s, Array2, ArrayView2};
use serde::{Deserialize, Serialize};

const EPSILON: f64 = f64::EPSILON;

/// sklearn's `MiniBatchNMF` (no `fresh_restarts`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MiniBatchNmf {
    pub n_components: usize,
    pub init: NmfInit,
    pub batch_size: usize,
    pub beta_loss: BetaLoss,
    pub tol: f64,
    /// `None`: no early stopping on the cost.
    pub max_no_improvement: Option<usize>,
    pub max_iter: usize,
    pub alpha_w: f64,
    /// `None`: `alpha_w` (sklearn's `"same"`).
    pub alpha_h: Option<f64>,
    pub l1_ratio: f64,
    pub forget_factor: f64,
    /// `transform`'s iterations (`None`: `max_iter`).
    pub transform_max_iter: Option<usize>,
    pub seed: u64,
}

impl Default for MiniBatchNmf {
    fn default() -> Self {
        Self {
            n_components: 2,
            init: NmfInit::Auto,
            batch_size: 1024,
            beta_loss: BetaLoss::Frobenius,
            tol: 1e-4,
            max_no_improvement: Some(10),
            max_iter: 200,
            alpha_w: 0.0,
            alpha_h: None,
            l1_ratio: 0.0,
            forget_factor: 0.7,
            transform_max_iter: None,
            seed: 0,
        }
    }
}

/// A fitted [`MiniBatchNmf`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedMiniBatchNmf {
    /// `H`, `n_components × n_features` (sklearn's `components_`).
    pub components: Array2<f64>,
    /// `W` of the training data (sklearn's `fit_transform` output).
    pub w: Array2<f64>,
    /// `‖X − WH‖` (β-divergence, square-rooted) on the training data.
    pub reconstruction_err: f64,
    /// Passes over the data (sklearn's `n_iter_`) and mini-batch steps
    /// (`n_steps_`).
    pub n_iter: usize,
    pub n_steps: usize,
    beta: f64,
    tol: f64,
    alpha_w: f64,
    l1_ratio: f64,
    transform_max_iter: usize,
}

/// sklearn's `_compute_regularization` on an `n × d` block: `(l1_w, l1_h,
/// l2_w, l2_h)`.
fn regularization(
    alpha_w: f64,
    alpha_h: f64,
    l1_ratio: f64,
    n: usize,
    d: usize,
) -> (f64, f64, f64, f64) {
    (
        d as f64 * alpha_w * l1_ratio,
        n as f64 * alpha_h * l1_ratio,
        d as f64 * alpha_w * (1.0 - l1_ratio),
        n as f64 * alpha_h * (1.0 - l1_ratio),
    )
}

fn frobenius(a: &Array2<f64>) -> f64 {
    a.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// sklearn's `_multiplicative_update_h` with the online accumulators.
#[allow(clippy::too_many_arguments)]
fn online_update_h(
    x: ArrayView2<f64>,
    w: &Array2<f64>,
    h: &mut Array2<f64>,
    beta: f64,
    l1: f64,
    l2: f64,
    a: &mut Array2<f64>,
    b: &mut Array2<f64>,
    rho: f64,
) {
    let (mut numerator, mut denominator) = if beta == 2.0 {
        (w.t().dot(&x), w.t().dot(w).dot(h))
    } else {
        let wh = w.dot(h).mapv(|v| v.max(EPSILON));
        let num = if beta == 1.0 {
            w.t().dot(&(&x / &wh))
        } else {
            w.t().dot(&(&x * &wh.mapv(|v| v.powf(beta - 2.0))))
        };
        let den = if beta == 1.0 {
            let wsum = w.sum_axis(ndarray::Axis(0));
            Array2::from_shape_fn(h.dim(), |(t, _)| wsum[t])
        } else {
            w.t().dot(&wh.mapv(|v| v.powf(beta - 1.0)))
        };
        (num, den)
    };
    if l1 > 0.0 {
        denominator.mapv_inplace(|v| v + l1);
    }
    if l2 > 0.0 {
        denominator = denominator + &(h.mapv(|v| v * l2));
    }
    denominator.mapv_inplace(|v| if v == 0.0 { EPSILON } else { v });
    let g = gamma(beta);
    if g != 1.0 {
        h.mapv_inplace(|v| v.powf(1.0 / g));
    }
    numerator *= &*h;
    *a *= rho;
    *b *= rho;
    *a += &numerator;
    *b += &denominator;
    *h = &*a / &*b;
    if g != 1.0 {
        h.mapv_inplace(|v| v.powf(g));
    }
}

impl FitUnsupervised<f64> for MiniBatchNmf {
    type Fitted = FittedMiniBatchNmf;

    fn fit_view(&self, x: ArrayView2<f64>) -> Result<Self::Fitted> {
        let (n, d) = x.dim();
        let k = self.n_components;
        if n == 0 || d == 0 {
            return Err(RustMlError::EmptyInput("MiniBatchNMF needs data".into()));
        }
        if x.iter().any(|v| !(*v >= 0.0)) {
            return Err(RustMlError::InvalidParameter("NMF requires X >= 0".into()));
        }
        if k == 0 || self.batch_size == 0 || self.max_iter == 0 {
            return Err(RustMlError::InvalidParameter(
                "n_components, batch_size and max_iter must be at least 1".into(),
            ));
        }
        let init = match self.init {
            NmfInit::Auto if k <= d.min(n) => NmfInit::NndsvdA,
            NmfInit::Auto => NmfInit::Random,
            other => other,
        };
        if k > d.min(n) && init != NmfInit::Random {
            return Err(RustMlError::InvalidParameter(format!(
                "n_components must be in 1..={} for an NNDSVD init",
                d.min(n)
            )));
        }
        let beta = self.beta_loss.beta();
        let alpha_h = self.alpha_h.unwrap_or(self.alpha_w);
        let (mut w, mut h) = initial_wh(x, k, init, self.seed)?;
        let batch = self.batch_size.min(n);
        let rho = self.forget_factor.powf(batch as f64 / n as f64);
        let mut a = h.clone();
        let mut b = Array2::<f64>::ones(h.dim());
        let mut h_buffer = h.clone();
        let steps_per_iter = n.div_ceil(batch);
        let n_steps_max = self.max_iter * steps_per_iter;
        let (mut ewa, mut ewa_min): (Option<f64>, Option<f64>) = (None, None);
        let mut no_improvement = 0usize;
        let mut step = 0usize;
        while step < n_steps_max {
            let s0 = (step % steps_per_iter) * batch;
            let s1 = (s0 + batch).min(n);
            let xb = x.slice(s![s0..s1, ..]);
            let size = s1 - s0;
            let (l1_w, l1_h, l2_w, l2_h) =
                regularization(self.alpha_w, alpha_h, self.l1_ratio, size, d);
            let mut wb = w.slice(s![s0..s1, ..]).to_owned();
            mu_update_w(xb, &mut wb, &h, beta, l1_w, l2_w);
            if beta < 1.0 {
                wb.mapv_inplace(|v| if v < EPSILON { 0.0 } else { v });
            }
            let div = beta_divergence(xb, &wb, &h, beta);
            let cost = (div * div / 2.0
                + l1_w * wb.sum()
                + l1_h * h.sum()
                + l2_w * wb.mapv(|v| v * v).sum()
                + l2_h * h.mapv(|v| v * v).sum())
                / size as f64;
            online_update_h(xb, &wb, &mut h, beta, l1_h, l2_h, &mut a, &mut b, rho);
            if beta <= 1.0 {
                h.mapv_inplace(|v| if v < EPSILON { 0.0 } else { v });
            }
            w.slice_mut(s![s0..s1, ..]).assign(&wb);
            step += 1;
            // sklearn's _minibatch_convergence (the first step is skipped).
            if step > 1 {
                ewa = Some(match ewa {
                    None => cost,
                    Some(e) => {
                        let alpha = (size as f64 / (n as f64 + 1.0)).min(1.0);
                        e * (1.0 - alpha) + cost * alpha
                    }
                });
                let h_diff = frobenius(&(&h - &h_buffer)) / frobenius(&h);
                if self.tol > 0.0 && h_diff <= self.tol {
                    break;
                }
                let e = ewa.expect("set above");
                if ewa_min.is_none_or(|m| e < m) {
                    no_improvement = 0;
                    ewa_min = Some(e);
                } else {
                    no_improvement += 1;
                }
                if self.max_no_improvement.is_some_and(|m| no_improvement >= m) {
                    break;
                }
            }
            h_buffer.assign(&h);
        }
        Ok(FittedMiniBatchNmf {
            reconstruction_err: beta_divergence(x, &w, &h, beta),
            components: h,
            w,
            n_iter: step.div_ceil(steps_per_iter),
            n_steps: step,
            beta,
            tol: self.tol,
            alpha_w: self.alpha_w,
            l1_ratio: self.l1_ratio,
            transform_max_iter: self.transform_max_iter.unwrap_or(self.max_iter),
        })
    }
}

impl Transform<f64> for FittedMiniBatchNmf {
    /// sklearn's `_solve_W`: multiplicative updates of `W` from
    /// `sqrt(mean(X) / k)` with `H` fixed, until `‖ΔW‖ / ‖W‖ <= tol`.
    fn transform_view(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
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
        let avg = (x.mean().unwrap_or(0.0) / k as f64).sqrt();
        let mut w = Array2::<f64>::from_elem((x.nrows(), k), avg);
        let mut buffer = w.clone();
        let (l1_w, _, l2_w, _) =
            regularization(self.alpha_w, 0.0, self.l1_ratio, x.nrows(), x.ncols());
        for _ in 0..self.transform_max_iter {
            mu_update_w(x, &mut w, h, self.beta, l1_w, l2_w);
            let diff = frobenius(&(&w - &buffer)) / frobenius(&w);
            if self.tol > 0.0 && diff <= self.tol {
                break;
            }
            buffer.assign(&w);
        }
        Ok(w)
    }
}
