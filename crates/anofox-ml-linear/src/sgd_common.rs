//! Common types and utilities for SGD-based linear models.

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

/// Compute the learning rate at iteration t.
#[inline]
pub fn compute_lr(schedule: LearningRate, eta0: f64, alpha: f64, t: usize, power_t: f64) -> f64 {
    match schedule {
        LearningRate::Constant => eta0,
        LearningRate::Optimal => {
            let t0 = 1.0 / (eta0 * alpha);
            1.0 / (alpha * (t as f64 + t0))
        }
        LearningRate::InvScaling => eta0 / (t as f64 + 1.0).powf(power_t),
    }
}
