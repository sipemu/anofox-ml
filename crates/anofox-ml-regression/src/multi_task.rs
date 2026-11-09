//! Multi-task Lasso / elastic net (sklearn's `MultiTaskLasso` /
//! `MultiTaskElasticNet`): one coefficient row per target, the features
//! selected jointly by the ℓ2,1 penalty,
//!
//! `1/(2n) ‖Y − XWᵀ‖²_F + α ρ Σ_j ‖W_{:,j}‖₂ + α (1 − ρ)/2 ‖W‖²_F`,
//!
//! solved by block coordinate descent (sklearn's
//! `enet_coordinate_descent_multi_task`) until the updates vanish, i.e.
//! to the unique minimum sklearn's duality-gap stop approaches.

use anofox_ml_core::{Result, RustMlError};
use ndarray::{Array1, Array2, ArrayView2, AsArray, Axis, Ix2};

/// sklearn's `MultiTaskElasticNet` (`MultiTaskLasso`: `l1_ratio = 1`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MultiTaskElasticNet {
    pub alpha: f64,
    pub l1_ratio: f64,
    pub fit_intercept: bool,
    pub max_iter: usize,
}

impl MultiTaskElasticNet {
    pub fn new(alpha: f64, l1_ratio: f64) -> Self {
        Self {
            alpha,
            l1_ratio,
            fit_intercept: true,
            max_iter: 100_000,
        }
    }

    /// sklearn's `MultiTaskLasso(alpha)`.
    pub fn lasso(alpha: f64) -> Self {
        Self::new(alpha, 1.0)
    }

    pub fn fit(&self, x: &Array2<f64>, y: &Array2<f64>) -> Result<FittedMultiTask> {
        let (n, p) = x.dim();
        let k = y.ncols();
        if y.nrows() != n {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {n} rows but Y has {}",
                y.nrows()
            )));
        }
        if n == 0 || p == 0 || k == 0 {
            return Err(RustMlError::EmptyInput("X and Y must be non-empty".into()));
        }
        if self.alpha.is_nan() || self.alpha < 0.0 || !(0.0..=1.0).contains(&self.l1_ratio) {
            return Err(RustMlError::InvalidParameter(format!(
                "alpha must be >= 0 and l1_ratio in [0, 1], got {} and {}",
                self.alpha, self.l1_ratio
            )));
        }
        let (x_mean, y_mean) = if self.fit_intercept {
            (x.mean_axis(Axis(0)).unwrap(), y.mean_axis(Axis(0)).unwrap())
        } else {
            (Array1::zeros(p), Array1::zeros(k))
        };
        let xc = x - &x_mean;
        let yc = y - &y_mean;
        let l1 = self.alpha * self.l1_ratio * n as f64;
        let l2 = self.alpha * (1.0 - self.l1_ratio) * n as f64;
        let norms: Vec<f64> = (0..p).map(|j| xc.column(j).dot(&xc.column(j))).collect();
        let mut w = Array2::<f64>::zeros((k, p));
        let mut r = yc.clone();
        for _ in 0..self.max_iter {
            let mut max_change = 0.0f64;
            let mut max_w = 0.0f64;
            for (j, &norm_j) in norms.iter().enumerate() {
                if norm_j == 0.0 {
                    continue;
                }
                let xj = xc.column(j);
                let old = w.column(j).to_owned();
                // tmp = X_jᵀ R + ‖X_j‖² w_j
                let tmp: Array1<f64> = r.t().dot(&xj) + &old * norm_j;
                let norm = tmp.dot(&tmp).sqrt();
                let shrink = if norm > 0.0 {
                    (1.0 - l1 / norm).max(0.0)
                } else {
                    0.0
                };
                let new = &tmp * (shrink / (norm_j + l2));
                let delta = &new - &old;
                if delta.iter().any(|&d| d != 0.0) {
                    for i in 0..n {
                        for t in 0..k {
                            r[[i, t]] -= xj[i] * delta[t];
                        }
                    }
                }
                max_change = delta.iter().fold(max_change, |m, d| m.max(d.abs()));
                max_w = new.iter().fold(max_w, |m, v| m.max(v.abs()));
                w.column_mut(j).assign(&new);
            }
            if max_change <= 1e-15 * max_w.max(1e-300) || max_change == 0.0 {
                break;
            }
        }
        let intercept = &y_mean - &w.dot(&x_mean);
        Ok(FittedMultiTask { coef: w, intercept })
    }
}

/// A fitted multi-task linear model.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedMultiTask {
    /// `(n_targets, n_features)` (sklearn `coef_`).
    pub coef: Array2<f64>,
    /// One per target (sklearn `intercept_`).
    pub intercept: Array1<f64>,
}

impl FittedMultiTask {
    /// `X Wᵀ + b`, one column per target.
    pub fn predict<'a>(&self, x: impl AsArray<'a, f64, Ix2>) -> Result<Array2<f64>> {
        let x: ArrayView2<f64> = x.into();
        if x.ncols() != self.coef.ncols() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.coef.ncols(),
                x.ncols()
            )));
        }
        Ok(x.dot(&self.coef.t()) + &self.intercept)
    }
}
