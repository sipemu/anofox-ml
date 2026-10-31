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

use anofox_ml_core::{Fit, PartialFit, Predict, Result, RustMlError};
use anofox_regression::solvers::{
    FittedRegressor as _, PaLoss, PaState, PassiveAggressiveRegressor as InnerPa, Regressor as _,
};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

/// Variant of the Passive-Aggressive update.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub enum PaVariant {
    /// τ = L / ||x||² (no upper bound).
    Pa,
    /// τ = min(C, L / ||x||²) — sklearn's PA-I.
    #[default]
    PaI,
    /// τ = L / (||x||² + 1/(2C)) — sklearn's PA-II.
    PaII,
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
    pub fn with_c(mut self, c: f64) -> Self {
        self.c = c;
        self
    }
    pub fn with_variant(mut self, v: PaVariant) -> Self {
        self.variant = v;
        self
    }
    pub fn with_max_iter(mut self, m: usize) -> Self {
        self.max_iter = m;
        self
    }
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }
}

impl Default for PassiveAggressiveClassifier {
    fn default() -> Self {
        Self::new()
    }
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
                "X has {} rows but y has {} elements",
                x.nrows(),
                y.len()
            )));
        }
        if x.is_empty() {
            return Err(RustMlError::EmptyInput("training data is empty".into()));
        }

        // Find the two classes.
        let mut classes: Vec<f64> = y.iter().copied().collect();
        classes.sort_by(anofox_ml_core::labels::label_cmp);
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
                "expected {} features, got {}",
                self.coef.len(),
                x.ncols()
            )));
        }
        let mut out = Array1::<f64>::zeros(x.nrows());
        for i in 0..x.nrows() {
            let s = x.row(i).dot(&self.coef) + self.intercept;
            out[i] = if s >= 0.0 {
                self.classes[1]
            } else {
                self.classes[0]
            };
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// PassiveAggressiveRegressor (epsilon-insensitive loss)
// ---------------------------------------------------------------------------
//
// Delegated to `anofox_regression::solvers::PassiveAggressiveRegressor`,
// which follows sklearn's `_plain_sgd` kernel: the step uses `||x||²` over the
// feature columns only (no `+1` for the intercept, as sklearn) and training
// stops after 5 epochs without the epoch loss improving by `tol · n`.

/// Passive-Aggressive regressor (sklearn `PassiveAggressiveRegressor`).
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
    pub fn with_c(mut self, c: f64) -> Self {
        self.c = c;
        self
    }
    pub fn with_epsilon(mut self, e: f64) -> Self {
        self.epsilon = e;
        self
    }
    pub fn with_variant(mut self, v: PaVariant) -> Self {
        self.variant = v;
        self
    }
    pub fn with_max_iter(mut self, m: usize) -> Self {
        self.max_iter = m;
        self
    }
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    fn inner(&self) -> Result<InnerPa> {
        if self.c.is_nan() || self.c <= 0.0 {
            return Err(RustMlError::InvalidParameter(format!(
                "C must be > 0, got {}",
                self.c
            )));
        }
        if self.epsilon.is_nan() || self.epsilon < 0.0 {
            return Err(RustMlError::InvalidParameter(format!(
                "epsilon must be >= 0, got {}",
                self.epsilon
            )));
        }
        // PA (unbounded) is PA-I with C = ∞.
        let (loss, c) = match self.variant {
            PaVariant::Pa => (PaLoss::EpsilonInsensitive, f64::INFINITY),
            PaVariant::PaI => (PaLoss::EpsilonInsensitive, self.c),
            PaVariant::PaII => (PaLoss::SquaredEpsilonInsensitive, self.c),
        };
        Ok(InnerPa::builder()
            .c(c)
            .epsilon(self.epsilon)
            .loss(loss)
            .with_intercept(self.fit_intercept)
            .max_iter(self.max_iter)
            .tolerance(self.tol)
            .shuffle(self.shuffle)
            .random_state(self.seed)
            .build())
    }
}

impl Default for PassiveAggressiveRegressor {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct FittedPassiveAggressiveRegressor {
    pub coef: Array1<f64>,
    pub intercept: f64,
    /// Epochs run by `fit` (sklearn `n_iter_`); 0 for a `partial_fit` state.
    pub n_iter: usize,
}

fn check_xy(x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
    if x.nrows() != y.len() {
        return Err(RustMlError::ShapeMismatch(format!(
            "X has {} rows but y has {} elements",
            x.nrows(),
            y.len()
        )));
    }
    if x.is_empty() {
        return Err(RustMlError::EmptyInput("training data is empty".into()));
    }
    Ok(())
}

impl Fit<f64> for PassiveAggressiveRegressor {
    type Fitted = FittedPassiveAggressiveRegressor;

    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        check_xy(x, y)?;
        let xm = faer::Mat::from_fn(x.nrows(), x.ncols(), |i, j| x[[i, j]]);
        let yc = faer::Col::from_fn(y.len(), |i| y[i]);
        let fitted = self
            .inner()?
            .fit(&xm, &yc)
            .map_err(|e| RustMlError::InvalidParameter(e.to_string()))?;
        let res = fitted.result();
        Ok(FittedPassiveAggressiveRegressor {
            coef: Array1::from_shape_fn(x.ncols(), |j| res.coefficients[j]),
            intercept: res.intercept.unwrap_or(0.0),
            n_iter: fitted.n_iter(),
        })
    }
}

impl PartialFit<f64> for PassiveAggressiveRegressor {
    type Fitted = FittedPassiveAggressiveRegressor;

    /// One pass over the batch in row order, resuming from `state`
    /// (sklearn `partial_fit` with `shuffle=False`). `classes` is ignored.
    fn partial_fit(
        &self,
        state: Option<Self::Fitted>,
        x: &Array2<f64>,
        y: &Array1<f64>,
        _classes: Option<&[f64]>,
    ) -> Result<Self::Fitted> {
        check_xy(x, y)?;
        let mut st = match state {
            Some(s) => {
                if s.coef.len() != x.ncols() {
                    return Err(RustMlError::ShapeMismatch(format!(
                        "state has {} features, X has {}",
                        s.coef.len(),
                        x.ncols()
                    )));
                }
                PaState {
                    weights: s.coef.to_vec(),
                    intercept: s.intercept,
                }
            }
            None => PaState::new(x.ncols()),
        };
        let inner = self.inner()?;
        for (row, &yi) in x.rows().into_iter().zip(y.iter()) {
            inner
                .partial_fit(&mut st, &row.to_vec(), yi)
                .map_err(|e| RustMlError::InvalidParameter(e.to_string()))?;
        }
        Ok(FittedPassiveAggressiveRegressor {
            coef: Array1::from_vec(st.weights),
            intercept: if self.fit_intercept {
                st.intercept
            } else {
                0.0
            },
            n_iter: 0,
        })
    }
}

impl Predict<f64> for FittedPassiveAggressiveRegressor {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if x.ncols() != self.coef.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.coef.len(),
                x.ncols()
            )));
        }
        Ok(x.dot(&self.coef) + self.intercept)
    }
}

impl anofox_ml_core::ClassifierScore<f64> for FittedPassiveAggressiveClassifier {}
impl anofox_ml_core::RegressorScore<f64> for FittedPassiveAggressiveRegressor {}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_pa_classifier_separable() {
        let x = array![
            [-2.0, -1.0],
            [-1.0, -2.0],
            [-2.0, -2.0],
            [2.0, 1.0],
            [1.0, 2.0],
            [2.0, 2.0],
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let fitted = PassiveAggressiveClassifier::new()
            .with_max_iter(100)
            .with_seed(42)
            .fit(&x, &y)
            .unwrap();
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
            .fit(&x, &y)
            .unwrap();
        // Predictions should be reasonably close.
        let preds = fitted.predict(&x).unwrap();
        let mae: f64 = preds
            .iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t).abs())
            .sum::<f64>()
            / 20.0;
        assert!(mae < 1.0, "MAE too high: {mae}");
    }

    /// `data.make(20, 2, m=7, coefs=[1.5, -2.0])` from the sklearn script.
    fn make() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_fn((20, 2), |(i, j)| {
            ((i * (j + 3) * 7 + j * 5 + (i * i) % 11) % 17) as f64 / 4.0 - 2.0
        });
        let y = Array1::from_shape_fn(20, |i| {
            1.5 * x[[i, 0]] - 2.0 * x[[i, 1]] + 1.0 + 0.3 * (((i * 13) % 7) as f64 - 3.0)
        });
        (x, y)
    }

    #[test]
    fn test_pa_regressor_matches_sklearn_without_shuffle() {
        // sklearn PassiveAggressiveRegressor(C=0.5, epsilon=0.1,
        // shuffle=False, loss=...) — deterministic, so exact parity.
        let (x, y) = make();
        for (variant, coef, intercept) in [
            (
                PaVariant::PaI,
                [1.6673134056881596, -1.904250783105196],
                1.1468047292027714,
            ),
            (
                PaVariant::PaII,
                [1.6366276373288773, -1.967730139596085],
                1.0981129510716956,
            ),
        ] {
            let f = PassiveAggressiveRegressor::new()
                .with_c(0.5)
                .with_variant(variant)
                .with_shuffle(false)
                .fit(&x, &y)
                .unwrap();
            assert!((f.coef[0] - coef[0]).abs() < 1e-12, "{:?}", f.coef);
            assert!((f.coef[1] - coef[1]).abs() < 1e-12, "{:?}", f.coef);
            assert!((f.intercept - intercept).abs() < 1e-12);
            assert_eq!(f.n_iter, 7);
        }
    }

    #[test]
    fn test_pa_regressor_partial_fit_matches_sklearn() {
        use anofox_ml_core::PartialFit;
        let (x, y) = make();
        let pa = PassiveAggressiveRegressor::new()
            .with_c(0.5)
            .with_shuffle(false);
        let s = pa
            .partial_fit(
                None,
                &x.slice(ndarray::s![..10, ..]).to_owned(),
                &y.slice(ndarray::s![..10]).to_owned(),
                None,
            )
            .unwrap();
        let s = pa
            .partial_fit(
                Some(s),
                &x.slice(ndarray::s![10.., ..]).to_owned(),
                &y.slice(ndarray::s![10..]).to_owned(),
                None,
            )
            .unwrap();
        assert!((s.coef[0] - 1.70998400990075).abs() < 1e-12, "{:?}", s.coef);
        assert!((s.coef[1] - -1.8758006319148437).abs() < 1e-12);
        assert!((s.intercept - 1.2089592026701308).abs() < 1e-12);
        assert!(s.predict(&x).is_ok());
    }
}
