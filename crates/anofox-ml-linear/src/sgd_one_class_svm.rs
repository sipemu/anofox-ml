//! sklearn's `SGDOneClassSVM`: the linear One-Class SVM primal solved by
//! SGD on the hinge loss with targets `y = 1`, `alpha = nu / 2` and the
//! intercept `1 − offset` pushed down by `2 η α` at every step.
//!
//! A port of sklearn's `_plain_sgd` (one-class path, L2, dense data):
//! the `WeightVector` with its lazy scale, incremental squared norm and
//! Bottou's averaging trick, and `SequentialDataset.shuffle`, a Fisher–Yates
//! permutation from `our_rand_r` with the same seed every epoch (applied to
//! the previous epoch's order). The seed is
//! `RandomState(random_state).randint(0, 2³¹ − 1)`, so for an int
//! `random_state` the fit is sklearn's up to rounding.

use anofox_ml_core::numpy_random::RandomState;
use anofox_ml_core::{FitUnsupervised, FitUnsupervisedWeighted, Result, RustMlError};
use ndarray::{Array1, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

/// The step-size schedule (sklearn's `learning_rate`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum OneClassLearningRate {
    /// `eta0`.
    Constant,
    /// `1 / (α (t₀ + t − 1))`, `t₀` chosen so the first step is `√(1/√α)`.
    #[default]
    Optimal,
    /// `eta0 / t^power_t`.
    InvScaling,
    /// `eta0`, divided by 5 whenever `n_iter_no_change` epochs fail to
    /// improve the objective by `tol` (until it is at most `1e-6`).
    Adaptive,
}

/// sklearn's `SGDOneClassSVM` (`n_iter_no_change = 5`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SgdOneClassSvm {
    /// Upper bound on the fraction of training errors, in `(0, 1]`.
    pub nu: f64,
    pub fit_intercept: bool,
    /// Maximum number of epochs.
    pub max_iter: usize,
    /// Stop once the epoch objective has not improved by `tol` for 5 epochs
    /// (`None`: run `max_iter` epochs).
    pub tol: Option<f64>,
    pub shuffle: bool,
    pub learning_rate: OneClassLearningRate,
    pub eta0: f64,
    pub power_t: f64,
    /// Averaged SGD from update `average` on (0: off; sklearn's `True` is 1).
    pub average: usize,
    pub seed: u32,
}

impl Default for SgdOneClassSvm {
    fn default() -> Self {
        Self {
            nu: 0.5,
            fit_intercept: true,
            max_iter: 1000,
            tol: Some(1e-3),
            shuffle: true,
            learning_rate: OneClassLearningRate::Optimal,
            eta0: 0.01,
            power_t: 0.5,
            average: 0,
            seed: 0,
        }
    }
}

/// A fitted [`SgdOneClassSvm`]: decision `⟨w, x⟩ − offset`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedSgdOneClassSvm {
    /// sklearn's `coef_`.
    pub coef: Array1<f64>,
    /// sklearn's `offset_`.
    pub offset: f64,
    /// Epochs run (sklearn's `n_iter_`).
    pub n_iter: usize,
    /// Weight updates plus one (sklearn's `t_`).
    pub t: f64,
}

impl FittedSgdOneClassSvm {
    /// `⟨w, x⟩` (sklearn's `score_samples`).
    pub fn score_samples(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        if x.ncols() != self.coef.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "x has {} features, the model {}",
                x.ncols(),
                self.coef.len()
            )));
        }
        Ok(x.dot(&self.coef))
    }

    /// Signed distance to the hyperplane, positive for inliers.
    pub fn decision_function(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        Ok(self.score_samples(x)? - self.offset)
    }

    /// `1` for inliers (decision `>= 0`), `-1` for outliers.
    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        Ok(self
            .decision_function(x)?
            .mapv(|d| if d >= 0.0 { 1.0 } else { -1.0 }))
    }
}

/// sklearn's `our_rand_r` (xorshift, `RAND_R_MAX = 2³¹ − 1`).
fn our_rand_r(seed: &mut u32) -> u32 {
    if *seed == 0 {
        *seed = 1;
    }
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    *seed % 0x8000_0000
}

/// sklearn's `WeightVector` (`w = wscale · w_data`), with the averaged
/// weights `(aw + average_a · w_data) / average_b` when `averaged`.
struct WeightVector {
    w: Array1<f64>,
    wscale: f64,
    sq_norm: f64,
    aw: Option<Array1<f64>>,
    average_a: f64,
    average_b: f64,
}

impl WeightVector {
    fn dot(&self, x: ArrayView1<f64>) -> f64 {
        let mut s = 0.0;
        for (w, v) in self.w.iter().zip(x) {
            s += w * v;
        }
        s * self.wscale
    }

    fn add(&mut self, x: ArrayView1<f64>, c: f64) {
        let mut l2 = 0.0;
        for (w, v) in self.w.iter_mut().zip(x) {
            *w += v * (c / self.wscale);
            l2 += *w * *w;
        }
        self.sq_norm = l2 * (self.wscale * self.wscale);
    }

    fn add_average(&mut self, x: ArrayView1<f64>, c: f64, num_iter: f64) {
        let mu = 1.0 / num_iter;
        if let Some(aw) = &mut self.aw {
            for (a, v) in aw.iter_mut().zip(x) {
                *a += self.average_a * v * (-c / self.wscale);
            }
        }
        if num_iter > 1.0 {
            self.average_b /= 1.0 - mu;
        }
        self.average_a += mu * self.average_b * self.wscale;
    }

    fn scale(&mut self, c: f64) {
        self.wscale *= c;
        self.sq_norm *= c * c;
        if self.wscale < 1e-9 {
            self.reset_wscale();
        }
    }

    fn reset_wscale(&mut self) {
        if let Some(aw) = &mut self.aw {
            aw.scaled_add(self.average_a, &self.w);
            let b = 1.0 / self.average_b;
            aw.mapv_inplace(|a| a * b);
            self.average_a = 0.0;
            self.average_b = 1.0;
        }
        let s = self.wscale;
        self.w.mapv_inplace(|w| w * s);
        self.wscale = 1.0;
    }
}

impl SgdOneClassSvm {
    fn validate(&self) -> Result<()> {
        if !(self.nu > 0.0 && self.nu <= 1.0) {
            return Err(RustMlError::InvalidParameter(format!(
                "nu must be in (0, 1], got {}",
                self.nu
            )));
        }
        if self.learning_rate != OneClassLearningRate::Optimal && self.eta0 <= 0.0 {
            return Err(RustMlError::InvalidParameter(format!(
                "eta0 must be > 0 for this learning_rate, got {}",
                self.eta0
            )));
        }
        if self.max_iter == 0 {
            return Err(RustMlError::InvalidParameter(
                "max_iter must be >= 1".into(),
            ));
        }
        Ok(())
    }
}

impl FitUnsupervisedWeighted<f64> for SgdOneClassSvm {
    type Fitted = FittedSgdOneClassSvm;

    fn fit_unsupervised_weighted_view(
        &self,
        x: ArrayView2<f64>,
        sample_weight: Option<ArrayView1<f64>>,
    ) -> Result<FittedSgdOneClassSvm> {
        self.validate()?;
        let (n, p) = x.dim();
        if n == 0 {
            return Err(RustMlError::EmptyInput("x has no rows".into()));
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(RustMlError::InvalidParameter("x must be finite".into()));
        }
        let sw: Vec<f64> = match sample_weight {
            Some(w) if w.len() != n => {
                return Err(RustMlError::ShapeMismatch(format!(
                    "sample_weight has {} entries for {n} samples",
                    w.len()
                )))
            }
            Some(w) => w.to_vec(),
            None => vec![1.0; n],
        };

        let alpha = self.nu / 2.0;
        let seed = RandomState::new(self.seed).randint(0x7FFF_FFFF) as u32;
        let averaged = self.average > 0;
        let mut w = WeightVector {
            w: Array1::zeros(p),
            wscale: 1.0,
            sq_norm: 0.0,
            aw: averaged.then(|| Array1::zeros(p)),
            average_a: 0.0,
            average_b: 1.0,
        };
        let mut intercept = 1.0;
        let mut average_intercept = 0.0;
        let average = self.average as f64;
        let tol = self.tol.unwrap_or(f64::NEG_INFINITY);
        let optimal_init = 1.0 / ((1.0 / alpha.sqrt()).sqrt() * alpha);
        let mut eta = self.eta0;
        let mut t = 1.0;
        let mut order: Vec<usize> = (0..n).collect();
        let mut best_objective = f64::INFINITY;
        let mut no_improvement = 0;
        let mut n_iter = 0;

        for _ in 0..self.max_iter {
            n_iter += 1;
            let mut objective_sum = 0.0;
            if self.shuffle {
                let mut s = seed;
                for i in 0..n - 1 {
                    let j = i + our_rand_r(&mut s) as usize % (n - i);
                    order.swap(i, j);
                }
            }
            for &i in &order {
                let row = x.row(i);
                let pred = w.dot(row) + intercept;
                match self.learning_rate {
                    OneClassLearningRate::Optimal => {
                        eta = 1.0 / (alpha * (optimal_init + t - 1.0));
                    }
                    OneClassLearningRate::InvScaling => eta = self.eta0 / t.powf(self.power_t),
                    _ => {}
                }
                // Hinge loss with threshold 1 at y = 1, plus the objective's
                // penalty `α ‖w‖² / 2` and `ν · intercept`.
                objective_sum += (1.0 - pred).max(0.0);
                objective_sum += alpha * (0.5 * w.sq_norm.sqrt().powi(2));
                objective_sum += intercept * (alpha * 2.0);
                let dloss = if pred <= 1.0 { -1.0 } else { 0.0 };
                let update = -eta * dloss * sw[i];
                w.scale((1.0 - eta * alpha).max(0.0));
                if update != 0.0 {
                    w.add(row, update);
                }
                if self.fit_intercept {
                    let intercept_update = update - 2.0 * eta * alpha;
                    if intercept_update != 0.0 {
                        intercept += intercept_update;
                    }
                }
                if averaged && average <= t {
                    w.add_average(row, update, t - average + 1.0);
                    average_intercept += (intercept - average_intercept) / (t - average + 1.0);
                }
                t += 1.0;
            }
            if !intercept.is_finite() || w.w.iter().any(|v| !v.is_finite()) {
                return Err(RustMlError::InvalidParameter(format!(
                    "Floating-point under-/overflow occurred at epoch #{n_iter}. \
                     Scaling input data with StandardScaler or MinMaxScaler might help."
                )));
            }
            let objective = objective_sum / n as f64;
            if tol > f64::NEG_INFINITY && objective > best_objective - tol {
                no_improvement += 1;
            } else {
                no_improvement = 0;
            }
            if objective < best_objective {
                best_objective = objective;
            }
            if no_improvement >= 5 {
                if self.learning_rate == OneClassLearningRate::Adaptive && eta > 1e-6 {
                    eta /= 5.0;
                    no_improvement = 0;
                } else {
                    break;
                }
            }
        }
        w.reset_wscale();

        let t_final = 1.0 + (n_iter * n) as f64;
        let (coef, offset) = match w.aw {
            Some(aw) if average <= t_final - 1.0 => (aw, 1.0 - average_intercept),
            _ => (w.w, 1.0 - intercept),
        };
        Ok(FittedSgdOneClassSvm {
            coef,
            offset,
            n_iter,
            t: t_final,
        })
    }
}

impl FitUnsupervised<f64> for SgdOneClassSvm {
    type Fitted = FittedSgdOneClassSvm;

    fn fit_view(&self, x: ArrayView2<f64>) -> Result<FittedSgdOneClassSvm> {
        self.fit_unsupervised_weighted_view(x, None)
    }
}
