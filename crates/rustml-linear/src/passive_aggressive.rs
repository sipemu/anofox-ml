//! Passive-Aggressive online learners (Crammer et al. 2006).
//!
//! Mirrors `sklearn.linear_model.{PassiveAggressiveClassifier, PassiveAggressiveRegressor}`.
//!
//! Per-sample update rule:
//! - Compute loss `L` (hinge for classification, epsilon-insensitive for regression).
//! - If `L > 0`, take a step `w += τ * direction * x` where
//!   * PA-I: `τ = min(C, L / ||x||²)`
//!   * PA-II: `τ = L / (||x||² + 1/(2C))`
//!   * PA (original): `τ = L / ||x||²` (no upper bound).

use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rustml_core::{Fit, Predict, Result, RustMlError};

/// Variant of the Passive-Aggressive update.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum PaVariant {
    /// τ = L / ||x||² (no upper bound).
    Pa,
    /// τ = min(C, L / ||x||²) — sklearn's PA-I.
    PaI,
    /// τ = L / (||x||² + 1/(2C)) — sklearn's PA-II.
    PaII,
}

impl Default for PaVariant {
    fn default() -> Self {
        PaVariant::PaI
    }
}

// ---------------------------------------------------------------------------
// PassiveAggressiveClassifier (binary; multi-class via one-vs-rest is omitted)
// ---------------------------------------------------------------------------

/// Binary Passive-Aggressive classifier (hinge loss).
#[derive(Debug, Clone)]
pub struct PassiveAggressiveClassifier {
    pub c: f64,
    pub variant: PaVariant,
    pub max_iter: usize,
    pub tol: f64,
    pub fit_intercept: bool,
    pub shuffle: bool,
    pub seed: u64,
}

impl PassiveAggressiveClassifier {
    pub fn new() -> Self {
        Self {
            c: 1.0,
            variant: PaVariant::PaI,
            max_iter: 1000,
            tol: 1e-3,
            fit_intercept: true,
            shuffle: true,
            seed: 0,
        }
    }
    pub fn with_c(mut self, c: f64) -> Self { self.c = c; self }
    pub fn with_variant(mut self, v: PaVariant) -> Self { self.variant = v; self }
    pub fn with_max_iter(mut self, m: usize) -> Self { self.max_iter = m; self }
    pub fn with_seed(mut self, s: u64) -> Self { self.seed = s; self }
}

impl Default for PassiveAggressiveClassifier {
    fn default() -> Self { Self::new() }
}

/// Fitted binary PA classifier.
#[derive(Debug, Clone)]
pub struct FittedPassiveAggressiveClassifier {
    pub coef: Array1<f64>,
    pub intercept: f64,
    pub classes: [f64; 2],
}

impl Fit<f64> for PassiveAggressiveClassifier {
    type Fitted = FittedPassiveAggressiveClassifier;

    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        if x.nrows() != y.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but y has {} elements", x.nrows(), y.len()
            )));
        }
        if x.is_empty() {
            return Err(RustMlError::EmptyInput("training data is empty".into()));
        }

        // Find the two classes.
        let mut classes: Vec<f64> = y.iter().copied().collect();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        classes.dedup();
        if classes.len() != 2 {
            return Err(RustMlError::InvalidParameter(format!(
                "PassiveAggressiveClassifier expects 2 classes, found {}",
                classes.len()
            )));
        }
        let neg = classes[0];
        let pos = classes[1];

        let n = x.nrows();
        let d = x.ncols();
        let mut w = Array1::<f64>::zeros(d);
        let mut b = 0.0_f64;

        let mut indices: Vec<usize> = (0..n).collect();
        let mut rng = StdRng::seed_from_u64(self.seed);

        let mut prev_loss = f64::INFINITY;
        let mut n_no_improvement = 0;

        for _epoch in 0..self.max_iter {
            if self.shuffle {
                indices.shuffle(&mut rng);
            }

            let mut epoch_loss = 0.0_f64;
            for &i in &indices {
                let xi = x.row(i);
                let yi_raw = y[i];
                let yi = if yi_raw == pos { 1.0 } else { -1.0 };

                let p = xi.dot(&w) + b;
                let margin = yi * p;
                let loss = (1.0 - margin).max(0.0);
                epoch_loss += loss;
                if loss == 0.0 {
                    continue;
                }
                let norm_sq: f64 = xi.iter().map(|v| v * v).sum::<f64>()
                    + if self.fit_intercept { 1.0 } else { 0.0 };
                let tau = match self.variant {
                    PaVariant::Pa => loss / norm_sq,
                    PaVariant::PaI => self.c.min(loss / norm_sq),
                    PaVariant::PaII => loss / (norm_sq + 1.0 / (2.0 * self.c)),
                };

                for j in 0..d {
                    w[j] += tau * yi * xi[j];
                }
                if self.fit_intercept {
                    b += tau * yi;
                }
            }

            let mean_loss = epoch_loss / n as f64;
            if (prev_loss - mean_loss).abs() < self.tol {
                n_no_improvement += 1;
                if n_no_improvement >= 5 {
                    break;
                }
            } else {
                n_no_improvement = 0;
            }
            prev_loss = mean_loss;
        }

        Ok(FittedPassiveAggressiveClassifier {
            coef: w,
            intercept: b,
            classes: [neg, pos],
        })
    }
}

impl Predict<f64> for FittedPassiveAggressiveClassifier {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if x.ncols() != self.coef.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}", self.coef.len(), x.ncols()
            )));
        }
        let mut out = Array1::<f64>::zeros(x.nrows());
        for i in 0..x.nrows() {
            let s = x.row(i).dot(&self.coef) + self.intercept;
            out[i] = if s >= 0.0 { self.classes[1] } else { self.classes[0] };
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// PassiveAggressiveRegressor (epsilon-insensitive loss)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct PassiveAggressiveRegressor {
    pub c: f64,
    pub epsilon: f64,
    pub variant: PaVariant,
    pub max_iter: usize,
    pub tol: f64,
    pub fit_intercept: bool,
    pub shuffle: bool,
    pub seed: u64,
}

impl PassiveAggressiveRegressor {
    pub fn new() -> Self {
        Self {
            c: 1.0,
            epsilon: 0.1,
            variant: PaVariant::PaI,
            max_iter: 1000,
            tol: 1e-3,
            fit_intercept: true,
            shuffle: true,
            seed: 0,
        }
    }
    pub fn with_c(mut self, c: f64) -> Self { self.c = c; self }
    pub fn with_epsilon(mut self, e: f64) -> Self { self.epsilon = e; self }
    pub fn with_variant(mut self, v: PaVariant) -> Self { self.variant = v; self }
    pub fn with_max_iter(mut self, m: usize) -> Self { self.max_iter = m; self }
    pub fn with_seed(mut self, s: u64) -> Self { self.seed = s; self }
}

impl Default for PassiveAggressiveRegressor {
    fn default() -> Self { Self::new() }
}

#[derive(Debug, Clone)]
pub struct FittedPassiveAggressiveRegressor {
    pub coef: Array1<f64>,
    pub intercept: f64,
}

impl Fit<f64> for PassiveAggressiveRegressor {
    type Fitted = FittedPassiveAggressiveRegressor;

    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        if x.nrows() != y.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but y has {} elements", x.nrows(), y.len()
            )));
        }
        if x.is_empty() {
            return Err(RustMlError::EmptyInput("training data is empty".into()));
        }
        let n = x.nrows();
        let d = x.ncols();
        let mut w = Array1::<f64>::zeros(d);
        let mut b = 0.0_f64;
        let mut indices: Vec<usize> = (0..n).collect();
        let mut rng = StdRng::seed_from_u64(self.seed);

        let mut prev_loss = f64::INFINITY;
        let mut n_no_improvement = 0;

        for _epoch in 0..self.max_iter {
            if self.shuffle {
                indices.shuffle(&mut rng);
            }
            let mut epoch_loss = 0.0_f64;
            for &i in &indices {
                let xi = x.row(i);
                let p = xi.dot(&w) + b;
                let diff = y[i] - p;
                let abs_diff = diff.abs();
                let loss = (abs_diff - self.epsilon).max(0.0);
                epoch_loss += loss;
                if loss == 0.0 {
                    continue;
                }
                let sign = if diff >= 0.0 { 1.0 } else { -1.0 };
                let norm_sq: f64 = xi.iter().map(|v| v * v).sum::<f64>()
                    + if self.fit_intercept { 1.0 } else { 0.0 };
                let tau = match self.variant {
                    PaVariant::Pa => loss / norm_sq,
                    PaVariant::PaI => self.c.min(loss / norm_sq),
                    PaVariant::PaII => loss / (norm_sq + 1.0 / (2.0 * self.c)),
                };
                for j in 0..d {
                    w[j] += tau * sign * xi[j];
                }
                if self.fit_intercept {
                    b += tau * sign;
                }
            }

            let mean_loss = epoch_loss / n as f64;
            if (prev_loss - mean_loss).abs() < self.tol {
                n_no_improvement += 1;
                if n_no_improvement >= 5 {
                    break;
                }
            } else {
                n_no_improvement = 0;
            }
            prev_loss = mean_loss;
        }

        Ok(FittedPassiveAggressiveRegressor {
            coef: w,
            intercept: b,
        })
    }
}

impl Predict<f64> for FittedPassiveAggressiveRegressor {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if x.ncols() != self.coef.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}", self.coef.len(), x.ncols()
            )));
        }
        let mut out = Array1::<f64>::zeros(x.nrows());
        for i in 0..x.nrows() {
            out[i] = x.row(i).dot(&self.coef) + self.intercept;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_pa_classifier_separable() {
        let x = array![
            [-2.0, -1.0], [-1.0, -2.0], [-2.0, -2.0],
            [2.0, 1.0],   [1.0, 2.0],   [2.0, 2.0],
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let fitted = PassiveAggressiveClassifier::new()
            .with_max_iter(100)
            .with_seed(42)
            .fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_eq!(*p, *t);
        }
    }

    #[test]
    fn test_pa_regressor_recovers_line() {
        // y = 2x + 1
        let x = Array2::from_shape_vec((20, 1), (0..20).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_vec((0..20).map(|i| 2.0 * i as f64 + 1.0).collect());

        let fitted = PassiveAggressiveRegressor::new()
            .with_c(1.0)
            .with_epsilon(0.01)
            .with_max_iter(2000)
            .with_seed(0)
            .fit(&x, &y).unwrap();
        // Predictions should be reasonably close.
        let preds = fitted.predict(&x).unwrap();
        let mae: f64 = preds.iter().zip(y.iter()).map(|(p, t)| (p - t).abs()).sum::<f64>() / 20.0;
        assert!(mae < 1.0, "MAE too high: {mae}");
    }
}
