use anofox_ml_core::Float;
use ndarray::{Array2, Zip};

use crate::network::{DenseLayer, LayerGradients};

/// Optimization algorithm for weight updates.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Solver {
    /// Mini-batch SGD with (Nesterov) momentum and learning-rate schedules.
    Sgd,
    /// Adam (Kingma & Ba).
    #[default]
    Adam,
    /// Full-batch L-BFGS (quasi-Newton); best for small datasets.
    Lbfgs,
}

/// Learning-rate schedule for [`Solver::Sgd`] (sklearn `learning_rate`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LearningRateSchedule {
    /// Keep `learning_rate` fixed.
    #[default]
    Constant,
    /// `learning_rate / (t + 1)^power_t`, `t` = samples seen.
    InvScaling,
    /// Keep the rate while the loss improves; divide by 5 each time
    /// `n_iter_no_change` epochs pass without improvement (stop below 1e-6).
    Adaptive,
}

fn adam_lr(lr: f64, beta_1: f64, beta_2: f64, t: usize) -> f64 {
    let t = t as i32;
    lr * (1.0 - beta_2.powi(t)).sqrt() / (1.0 - beta_1.powi(t))
}

/// One fused in-place Adam step on a parameter array.
#[allow(clippy::too_many_arguments)]
fn adam_inplace<F: Float>(
    param: &mut Array2<F>,
    grad: &Array2<F>,
    m: &mut Array2<F>,
    v: &mut Array2<F>,
    lr_t: F,
    beta_1: F,
    beta_2: F,
    eps: F,
) {
    let one = F::one();
    Zip::from(param)
        .and(grad)
        .and(m)
        .and(v)
        .for_each(|p, &g, m, v| {
            *m = beta_1 * *m + (one - beta_1) * g;
            *v = beta_2 * *v + (one - beta_2) * g * g;
            *p -= lr_t * *m / (v.sqrt() + eps);
        });
}

/// One fused in-place (Nesterov) momentum step on a parameter array.
fn sgd_inplace<F: Float>(
    param: &mut Array2<F>,
    grad: &Array2<F>,
    vel: &mut Array2<F>,
    lr: F,
    momentum: F,
    nesterov: bool,
) {
    Zip::from(param).and(grad).and(vel).for_each(|p, &g, v| {
        *v = momentum * *v - lr * g;
        *p += if nesterov { momentum * *v - lr * g } else { *v };
    });
}

/// Stateful stochastic optimiser (sklearn `SGDOptimizer` / `AdamOptimizer`).
#[derive(Debug, Clone)]
pub(crate) enum StochasticOptimizer<F: Float> {
    Sgd {
        learning_rate_init: f64,
        learning_rate: f64,
        schedule: LearningRateSchedule,
        momentum: f64,
        nesterov: bool,
        power_t: f64,
        vel_w: Vec<Array2<F>>,
        vel_b: Vec<Array2<F>>,
    },
    Adam {
        learning_rate_init: f64,
        beta_1: f64,
        beta_2: f64,
        epsilon: f64,
        t: usize,
        m_w: Vec<Array2<F>>,
        v_w: Vec<Array2<F>>,
        m_b: Vec<Array2<F>>,
        v_b: Vec<Array2<F>>,
    },
}

impl<F: Float> StochasticOptimizer<F> {
    pub fn zeros_like(layers: &[DenseLayer<F>]) -> (Vec<Array2<F>>, Vec<Array2<F>>) {
        (
            layers
                .iter()
                .map(|l| Array2::zeros(l.weights.raw_dim()))
                .collect(),
            layers
                .iter()
                .map(|l| Array2::zeros(l.biases.raw_dim()))
                .collect(),
        )
    }

    pub fn update(&mut self, layers: &mut [DenseLayer<F>], grads: &[LayerGradients<F>]) {
        match self {
            StochasticOptimizer::Sgd {
                learning_rate,
                momentum,
                nesterov,
                vel_w,
                vel_b,
                ..
            } => {
                let lr = F::from_f64(*learning_rate).unwrap();
                let mom = F::from_f64(*momentum).unwrap();
                for (i, (layer, g)) in layers.iter_mut().zip(grads).enumerate() {
                    sgd_inplace(&mut layer.weights, &g.dw, &mut vel_w[i], lr, mom, *nesterov);
                    sgd_inplace(&mut layer.biases, &g.db, &mut vel_b[i], lr, mom, *nesterov);
                }
            }
            StochasticOptimizer::Adam {
                learning_rate_init,
                beta_1,
                beta_2,
                epsilon,
                t,
                m_w,
                v_w,
                m_b,
                v_b,
            } => {
                *t += 1;
                let lr_t = F::from_f64(adam_lr(*learning_rate_init, *beta_1, *beta_2, *t)).unwrap();
                let (b1, b2, eps) = (
                    F::from_f64(*beta_1).unwrap(),
                    F::from_f64(*beta_2).unwrap(),
                    F::from_f64(*epsilon).unwrap(),
                );
                for (i, (layer, g)) in layers.iter_mut().zip(grads).enumerate() {
                    adam_inplace(
                        &mut layer.weights,
                        &g.dw,
                        &mut m_w[i],
                        &mut v_w[i],
                        lr_t,
                        b1,
                        b2,
                        eps,
                    );
                    adam_inplace(
                        &mut layer.biases,
                        &g.db,
                        &mut m_b[i],
                        &mut v_b[i],
                        lr_t,
                        b1,
                        b2,
                        eps,
                    );
                }
            }
        }
    }

    /// End-of-epoch hook (`invscaling` schedule); `time_step` = samples seen.
    pub fn iteration_ends(&mut self, time_step: usize) {
        if let StochasticOptimizer::Sgd {
            learning_rate_init,
            learning_rate,
            schedule: LearningRateSchedule::InvScaling,
            power_t,
            ..
        } = self
        {
            *learning_rate = *learning_rate_init / ((time_step + 1) as f64).powf(*power_t);
        }
    }

    /// Called after `n_iter_no_change` epochs without improvement; returns
    /// `true` to stop (the `adaptive` schedule divides the rate by 5 instead).
    pub fn trigger_stopping(&mut self) -> bool {
        match self {
            StochasticOptimizer::Sgd {
                schedule: LearningRateSchedule::Adaptive,
                learning_rate,
                ..
            } => {
                if *learning_rate <= 1e-6 {
                    true
                } else {
                    *learning_rate /= 5.0;
                    false
                }
            }
            _ => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn adam_step_reduces_gradient() {
        let grad = array![[1.0, -2.0]];
        let mut delta = Array2::<f64>::zeros((1, 2));
        let (mut m, mut v) = (delta.clone(), delta.clone());
        let lr_t = adam_lr(0.001, 0.9, 0.999, 1);
        adam_inplace(&mut delta, &grad, &mut m, &mut v, lr_t, 0.9, 0.999, 1e-8);
        assert!(delta[[0, 0]] < 0.0);
        assert!(delta[[0, 1]] > 0.0);
        // First sklearn Adam step moves every coordinate by ≈ lr · sign(g).
        assert!((delta[[0, 0]] + 0.001).abs() < 1e-8);
        assert!((delta[[0, 1]] - 0.001).abs() < 1e-8);
    }

    #[test]
    fn nesterov_momentum_matches_sklearn_formula() {
        // sklearn: v = μv - ηg ; update = μv - ηg (nesterov)
        let mut p = array![[1.0_f64]];
        let mut v = array![[0.5_f64]];
        sgd_inplace(&mut p, &array![[2.0]], &mut v, 0.1, 0.9, true);
        let v1 = 0.9 * 0.5 - 0.1 * 2.0;
        assert!((v[[0, 0]] - v1).abs() < 1e-15);
        assert!((p[[0, 0]] - (1.0 + 0.9 * v1 - 0.2)).abs() < 1e-15);
    }
}
