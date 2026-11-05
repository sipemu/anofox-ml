//! Common types and utilities for SGD-based linear models.

use anofox_ml_core::{Result, RustMlError};
use ndarray::{Array1, ArrayView2};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
#[cfg(test)]
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

/// Regularization penalty type.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Penalty {
    /// No regularization.
    None,
    /// L2 (Ridge) regularization.
    L2,
    /// L1 (Lasso) regularization.
    L1,
    /// Elastic Net: `l1_ratio * L1 + (1 - l1_ratio) * L2`.
    ElasticNet,
}

impl Default for Penalty {
    fn default() -> Self {
        Penalty::L2
    }
}

/// Learning rate schedule.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LearningRate {
    /// Fixed learning rate: `eta = eta0`.
    Constant,
    /// Optimal: `eta = 1 / (alpha * (t + t0))`.
    Optimal,
    /// Inverse scaling: `eta = eta0 / t^power_t`.
    InvScaling,
}

impl Default for LearningRate {
    fn default() -> Self {
        LearningRate::InvScaling
    }
}

/// Apply L1/L2/ElasticNet penalty to a weight, returning the gradient contribution.
#[inline]
pub fn penalty_gradient(w: f64, alpha: f64, penalty: Penalty, l1_ratio: f64) -> f64 {
    match penalty {
        Penalty::None => 0.0,
        Penalty::L2 => alpha * w,
        Penalty::L1 => alpha * w.signum(),
        Penalty::ElasticNet => alpha * (l1_ratio * w.signum() + (1.0 - l1_ratio) * w),
    }
}

/// Compute the learning rate at iteration `t` (1-based), as sklearn's
/// `_plain_sgd`:
/// - `Optimal`: `1 / (α (t₀ + t − 1))` with `t₀ = 1 / (typw · α)` and
///   `typw = √(1 / √α)` (the initial rate is `typw` for every sklearn loss,
///   whose gradient at `(y = 1, p = −typw)` is never above 1);
/// - `InvScaling`: `eta0 / t^power_t`.
#[inline]
pub fn compute_lr(schedule: LearningRate, eta0: f64, alpha: f64, t: usize, power_t: f64) -> f64 {
    match schedule {
        LearningRate::Constant => eta0,
        LearningRate::Optimal => {
            let typw = (1.0 / alpha.sqrt()).sqrt();
            let t0 = 1.0 / (typw * alpha);
            1.0 / (alpha * (t0 + t as f64 - 1.0))
        }
        LearningRate::InvScaling => eta0 / (t as f64).powf(power_t),
    }
}

/// Numerically stable `ln(1 + exp(x))` (softplus).
///
/// Never overflows: for large positive `x` it returns `x + ln(1 + exp(-x))`,
/// otherwise `ln_1p(exp(x))`.
#[inline]
pub fn softplus(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// Numerically stable logistic sigmoid `1 / (1 + exp(-x))`.
#[inline]
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

pub(crate) fn default_n_iter_no_change() -> usize {
    5
}

pub(crate) fn default_validation_fraction() -> f64 {
    0.1
}

pub(crate) fn default_t() -> usize {
    1
}

/// Weights of one linear SGD problem (one regressor, or one binary
/// classifier), plus the averaged iterate when ASGD is enabled.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct SgdWeights {
    pub coef: Array1<f64>,
    pub intercept: f64,
    /// Averaged weights (ASGD); `None` until averaging has started.
    #[serde(default)]
    pub avg: Option<(Array1<f64>, f64)>,
    /// Truncated-gradient (L1) accumulators, kept so that `partial_fit`
    /// continues exactly where the previous call stopped: `l1_u` is the total
    /// L1 penalty each weight could have received, `l1_q[j]` the penalty
    /// actually applied to weight j.
    #[serde(default)]
    pub l1_u: f64,
    #[serde(default)]
    pub l1_q: Vec<f64>,
}

impl SgdWeights {
    pub fn zeros(p: usize) -> Self {
        Self {
            coef: Array1::zeros(p),
            intercept: 0.0,
            avg: None,
            l1_u: 0.0,
            l1_q: Vec::new(),
        }
    }

    pub fn new(coef: Array1<f64>, intercept: f64) -> Self {
        Self {
            coef,
            intercept,
            avg: None,
            l1_u: 0.0,
            l1_q: Vec::new(),
        }
    }

    /// Weights used for prediction: the averaged iterate if ASGD has started,
    /// otherwise the plain SGD iterate.
    pub fn effective(&self) -> (&Array1<f64>, f64) {
        match &self.avg {
            Some((w, b)) => (w, *b),
            None => (&self.coef, self.intercept),
        }
    }
}

/// Optimisation hyper-parameters shared by the SGD estimators.
pub(crate) struct SgdConfig {
    pub penalty: Penalty,
    pub alpha: f64,
    pub l1_ratio: f64,
    pub eta0: f64,
    pub power_t: f64,
    pub learning_rate: LearningRate,
    /// ASGD start: 0 = disabled, otherwise averaging starts once the
    /// learning-rate counter `t` reaches this value (sklearn `average`).
    pub average: usize,
}

/// Validation score callback: `(coef, intercept) -> score` (higher is better).
pub(crate) type ValidationScore<'a> = &'a dyn Fn(&Array1<f64>, f64) -> f64;

/// Stopping criterion (sklearn semantics).
pub(crate) struct Stopping<'a> {
    /// `None` disables the criterion (always run `max_epochs`).
    pub tol: Option<f64>,
    pub n_iter_no_change: usize,
    /// When set, the criterion uses this validation score instead of the
    /// average training loss.
    pub validation: Option<ValidationScore<'a>>,
}

impl Stopping<'_> {
    /// Never stop early (used by `partial_fit`).
    pub fn never() -> Stopping<'static> {
        Stopping {
            tol: None,
            n_iter_no_change: usize::MAX,
            validation: None,
        }
    }
}

/// Outcome of [`run_sgd`].
pub(crate) struct SgdRun {
    /// Number of epochs run.
    pub n_iter: usize,
    /// Learning-rate counter after the last update (next `t` to use).
    pub t: usize,
}

/// Plain/averaged SGD over the rows in `indices`, in the style of sklearn's
/// `_plain_sgd`: per-sample update with multiplicative L2 shrinkage and the
/// cumulative-penalty truncated gradient for L1 (Tsuruoka et al., 2009), so
/// L1/ElasticNet produce exact zeros.
///
/// `loss(z, y)` returns `(loss, dloss/dz)`. Stopping: an epoch whose average
/// training loss is `> best_loss - tol` (or, with a validation callback,
/// whose score is `< best_score + tol`) increments a no-improvement counter,
/// any other epoch resets it; training stops when the counter reaches
/// `n_iter_no_change`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_sgd<L>(
    x: ArrayView2<f64>,
    y: &[f64],
    sample_weight: Option<&[f64]>,
    indices: &mut [usize],
    weights: &mut SgdWeights,
    cfg: &SgdConfig,
    loss: L,
    max_epochs: usize,
    stopping: &Stopping<'_>,
    shuffle: bool,
    rng: &mut StdRng,
    t_start: usize,
) -> SgdRun
where
    L: Fn(f64, f64) -> (f64, f64),
{
    let p = x.ncols();
    let n = indices.len().max(1) as f64;
    let mut t = t_start.max(1);

    let l1_ratio = match cfg.penalty {
        Penalty::None | Penalty::L2 => 0.0,
        Penalty::L1 => 1.0,
        Penalty::ElasticNet => cfg.l1_ratio,
    };
    let use_l2 = matches!(cfg.penalty, Penalty::L2 | Penalty::ElasticNet);
    let use_l1 = matches!(cfg.penalty, Penalty::L1 | Penalty::ElasticNet);
    // Truncated-gradient state (see `SgdWeights::l1_u` / `l1_q`).
    let mut u = weights.l1_u;
    let mut q = std::mem::take(&mut weights.l1_q);
    if use_l1 && q.len() != p {
        u = 0.0;
        q = vec![0.0; p];
    }

    let mut best_loss = f64::INFINITY;
    let mut best_score = f64::NEG_INFINITY;
    let mut no_improvement = 0usize;
    let mut n_iter = 0usize;

    for _epoch in 0..max_epochs {
        if shuffle {
            indices.shuffle(rng);
        }
        n_iter += 1;
        let mut sum_loss = 0.0;

        for &i in indices.iter() {
            let eta = compute_lr(cfg.learning_rate, cfg.eta0, cfg.alpha, t, cfg.power_t);
            let row = x.row(i);
            let z = weights.intercept + row.dot(&weights.coef);
            let (l, dloss) = loss(z, y[i]);
            // sklearn 1.8 stops on the objective: the loss plus the penalty
            // of the current weights (`0.5 ‖w‖²` for the L2 part, as weight
            // decay optimises it).
            sum_loss += l;
            if cfg.penalty != Penalty::None
                && stopping.tol.is_some()
                && stopping.validation.is_none()
            {
                let mut pen = 0.0;
                if use_l2 {
                    pen += (1.0 - l1_ratio) * 0.5 * weights.coef.dot(&weights.coef);
                }
                if use_l1 {
                    pen += l1_ratio * weights.coef.iter().map(|w| w.abs()).sum::<f64>();
                }
                sum_loss += cfg.alpha * pen;
            }

            if use_l2 {
                // Do not flip signs when eta * alpha is too large.
                let scale = (1.0 - (1.0 - l1_ratio) * eta * cfg.alpha).max(0.0);
                weights.coef.mapv_inplace(|w| w * scale);
            }
            // sklearn scales the update (not the reported loss) by the
            // sample weight.
            let dloss = match sample_weight {
                Some(sw) => dloss * sw[i],
                None => dloss,
            };
            if dloss != 0.0 {
                let update = -eta * dloss;
                weights.coef.scaled_add(update, &row);
                weights.intercept += update;
            }
            if use_l1 {
                u += l1_ratio * eta * cfg.alpha;
                for (w, qj) in weights.coef.iter_mut().zip(q.iter_mut()) {
                    let before = *w;
                    if *w > 0.0 {
                        *w = (*w - (u + *qj)).max(0.0);
                    } else if *w < 0.0 {
                        *w = (*w + (u - *qj)).min(0.0);
                    }
                    *qj += *w - before;
                }
            }
            if cfg.average > 0 && t >= cfg.average {
                let k = (t - cfg.average + 1) as f64;
                match &mut weights.avg {
                    Some((aw, ab)) => {
                        aw.zip_mut_with(&weights.coef, |a, &w| *a += (w - *a) / k);
                        *ab += (weights.intercept - *ab) / k;
                    }
                    None => weights.avg = Some((weights.coef.clone(), weights.intercept)),
                }
            }
            t += 1;
        }

        if let Some(tol) = stopping.tol {
            if let Some(score_fn) = stopping.validation {
                let (w, b) = weights.effective();
                let score = score_fn(w, b);
                if score < best_score + tol {
                    no_improvement += 1;
                } else {
                    no_improvement = 0;
                }
                if score > best_score {
                    best_score = score;
                }
            } else {
                let avg_loss = sum_loss / n;
                if avg_loss > best_loss - tol {
                    no_improvement += 1;
                } else {
                    no_improvement = 0;
                }
                if avg_loss < best_loss {
                    best_loss = avg_loss;
                }
            }
            if no_improvement >= stopping.n_iter_no_change {
                break;
            }
        }
    }

    weights.l1_u = u;
    weights.l1_q = q;
    SgdRun { n_iter, t }
}

/// Split `0..n` into sorted (train, validation) index sets for early
/// stopping. With `strata = Some(labels)` the split is stratified per label
/// (as sklearn's classifier does). Errors when either side would be empty.
pub(crate) fn validation_split(
    n: usize,
    validation_fraction: f64,
    strata: Option<&[f64]>,
    rng: &mut StdRng,
) -> Result<(Vec<usize>, Vec<usize>)> {
    if !(validation_fraction > 0.0 && validation_fraction < 1.0) {
        return Err(RustMlError::InvalidParameter(
            "validation_fraction must be in (0, 1)".into(),
        ));
    }
    let mut groups: Vec<Vec<usize>> = match strata {
        None => vec![(0..n).collect()],
        Some(labels) => {
            let mut keys: Vec<f64> = labels.to_vec();
            keys.sort_by(|a, b| a.total_cmp(b));
            keys.dedup();
            keys.iter()
                .map(|k| (0..n).filter(|&i| labels[i] == *k).collect())
                .collect()
        }
    };
    let mut train = Vec::with_capacity(n);
    let mut val = Vec::new();
    for g in groups.iter_mut() {
        g.shuffle(rng);
        let n_val = (validation_fraction * g.len() as f64).round() as usize;
        val.extend_from_slice(&g[..n_val]);
        train.extend_from_slice(&g[n_val..]);
    }
    if val.is_empty() || train.is_empty() {
        return Err(RustMlError::InvalidParameter(format!(
            "early_stopping: validation_fraction={validation_fraction} on {n} samples \
             leaves an empty training or validation set"
        )));
    }
    train.sort_unstable();
    val.sort_unstable();
    Ok((train, val))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn softplus_is_finite_and_accurate() {
        assert!((softplus(0.0) - std::f64::consts::LN_2).abs() < 1e-15);
        assert_eq!(softplus(1e6), 1e6);
        assert!(softplus(-1e6) >= 0.0 && softplus(-1e6) < 1e-300);
        assert!((softplus(-40.0) - (-40.0f64).exp()).abs() < 1e-30);
        assert!(sigmoid(-1e6).is_finite() && sigmoid(1e6) == 1.0);
    }

    #[test]
    fn l1_truncated_gradient_gives_exact_zeros() {
        // Feature 1 is pure noise uncorrelated with y; strong L1 must zero it.
        let n = 200;
        let x = Array2::from_shape_fn((n, 2), |(i, j)| {
            if j == 0 {
                (i as f64 / n as f64) * 2.0 - 1.0
            } else if i % 2 == 0 {
                0.5
            } else {
                -0.5
            }
        });
        let y: Vec<f64> = (0..n).map(|i| 3.0 * x[[i, 0]]).collect();
        let cfg = SgdConfig {
            penalty: Penalty::L1,
            alpha: 0.05,
            l1_ratio: 0.15,
            eta0: 0.01,
            power_t: 0.25,
            learning_rate: LearningRate::Constant,
            average: 0,
        };
        let mut w = SgdWeights::zeros(2);
        let mut idx: Vec<usize> = (0..n).collect();
        let mut rng = StdRng::seed_from_u64(0);
        run_sgd(
            x.view(),
            &y,
            None,
            &mut idx,
            &mut w,
            &cfg,
            |z, y| {
                let r = z - y;
                (0.5 * r * r, r)
            },
            50,
            &Stopping::never(),
            true,
            &mut rng,
            1,
        );
        assert_eq!(
            w.coef[1], 0.0,
            "noise weight not exactly zero: {}",
            w.coef[1]
        );
        assert!(w.coef[0] > 1.0);
    }
}
