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

/// Passive-Aggressive classifier (sklearn's: hinge loss with PA-I or PA-II
/// steps, one-vs-rest for more than two classes).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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

/// Fitted PA classifier: one weight vector per class (one-vs-rest), or a
/// single one for two classes.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedPassiveAggressiveClassifier {
    /// `(1, d)` for binary problems, else `(n_classes, d)` (sklearn `coef_`).
    pub coef: Array2<f64>,
    pub intercept: Array1<f64>,
    /// Sorted class labels.
    pub classes: Vec<f64>,
}

impl PassiveAggressiveClassifier {
    /// One binary PA run with targets `±1` (sklearn `_plain_sgd` with
    /// `learning_rate = pa1 / pa2`).
    fn fit_binary(&self, x: &Array2<f64>, yb: &[f64], seed: u64) -> (Array1<f64>, f64) {
        let (n, d) = x.dim();
        let mut w = Array1::<f64>::zeros(d);
        let mut b = 0.0_f64;
        let mut indices: Vec<usize> = (0..n).collect();
        let mut rng = StdRng::seed_from_u64(seed);
        let mut best = f64::INFINITY;
        let mut no_improvement = 0;
        let sqnorm: Vec<f64> = x.rows().into_iter().map(|r| r.dot(&r)).collect();
        for _epoch in 0..self.max_iter {
            if self.shuffle {
                indices.shuffle(&mut rng);
            }
            let mut sumloss = 0.0_f64;
            for &i in &indices {
                let xi = x.row(i);
                let yi = yb[i];
                let p = xi.dot(&w) + b;
                // sklearn uses the hinge loss for PA-I and PA-II alike; only
                // the step size differs.
                let loss = (1.0 - yi * p).max(0.0);
                sumloss += loss;
                let tau = match self.variant {
                    PaVariant::Pa | PaVariant::PaI => {
                        if sqnorm[i] == 0.0 {
                            continue;
                        }
                        let c = if self.variant == PaVariant::Pa {
                            f64::INFINITY
                        } else {
                            self.c
                        };
                        c.min(loss / sqnorm[i])
                    }
                    PaVariant::PaII => loss / (sqnorm[i] + 0.5 / self.c),
                };
                let update = tau * yi;
                if update != 0.0 {
                    w.scaled_add(update, &xi);
                    if self.fit_intercept {
                        b += update;
                    }
                }
            }
            // sklearn: no improvement when the mean loss exceeds best - tol.
            let mean_loss = sumloss / n as f64;
            if mean_loss > best - self.tol {
                no_improvement += 1;
            } else {
                no_improvement = 0;
            }
            best = best.min(mean_loss);
            if no_improvement >= 5 {
                break;
            }
        }
        (w, b)
    }
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
        let (classes, _) = anofox_ml_core::encode_labels(y)?;
        if classes.len() < 2 {
            return Err(RustMlError::InvalidParameter(
                "PassiveAggressiveClassifier needs at least 2 classes".into(),
            ));
        }
        let target =
            |c: f64| -> Vec<f64> { y.iter().map(|&v| if v == c { 1.0 } else { -1.0 }).collect() };
        let d = x.ncols();
        if classes.len() == 2 {
            let (w, b) = self.fit_binary(x, &target(classes[1]), self.seed);
            return Ok(FittedPassiveAggressiveClassifier {
                coef: w.insert_axis(ndarray::Axis(0)),
                intercept: Array1::from_elem(1, b),
                classes,
            });
        }
        // One-vs-rest, each class with its own seed (sklearn draws one per
        // class).
        let mut seeds = StdRng::seed_from_u64(self.seed);
        let mut coef = Array2::<f64>::zeros((classes.len(), d));
        let mut intercept = Array1::<f64>::zeros(classes.len());
        for (k, &c) in classes.iter().enumerate() {
            let (w, b) = self.fit_binary(x, &target(c), rand::Rng::gen(&mut seeds));
            coef.row_mut(k).assign(&w);
            intercept[k] = b;
        }
        Ok(FittedPassiveAggressiveClassifier {
            coef,
            intercept,
            classes,
        })
    }
}

impl FittedPassiveAggressiveClassifier {
    /// Signed distances to the hyperplanes: `(n, 1)` for binary problems,
    /// else `(n, n_classes)`.
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
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

impl Predict<f64> for FittedPassiveAggressiveClassifier {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let df = self.decision_function(x)?;
        Ok(df
            .rows()
            .into_iter()
            .map(|r| {
                if r.len() == 1 {
                    // sklearn: positive class when the score is > 0.
                    if r[0] > 0.0 {
                        self.classes[1]
                    } else {
                        self.classes[0]
                    }
                } else {
                    let mut best = 0;
                    for k in 1..r.len() {
                        if r[k] > r[best] {
                            best = k;
                        }
                    }
                    self.classes[best]
                }
            })
            .collect())
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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

    #[test]
    fn test_classifier_matches_sklearn_without_shuffle() {
        // sklearn 1.8 PassiveAggressiveClassifier(shuffle=False, loss=..,
        // C=0.5).fit(X, y): coef_ and intercept_ (3 classes, one-vs-rest).
        let x = Array2::from_shape_fn((60, 3), |(i, j)| {
            let t = i as f64;
            match j {
                0 => (t * 0.7).sin(),
                1 => (t * 0.31).cos(),
                _ => (i % 7) as f64 / 3.0 - 1.0,
            }
        });
        let y = Array1::from_shape_fn(60, |i| (i % 3) as f64);
        let cases = [
            (
                PaVariant::PaI,
                [
                    0.182_873_008_144_987_08,
                    -0.221_758_215_223_213_62,
                    -0.123_358_521_203_125_79,
                    0.282_071_981_290_973_2,
                    -0.130_800_264_644_157_67,
                    0.186_046_128_749_096_26,
                    -0.385_245_641_476_041_8,
                    0.154_583_831_299_847_43,
                    -0.436_066_470_118_093_63,
                ],
                [
                    -1.068_481_996_398_010_3,
                    -1.034_305_637_306_261_8,
                    -0.741_023_405_250_369_9,
                ],
            ),
            (
                PaVariant::PaII,
                [
                    0.059_719_133_178_745_94,
                    0.018_103_940_861_630_92,
                    -0.136_388_184_244_042_03,
                    0.971_271_265_449_140_9,
                    -0.511_775_901_822_441_6,
                    0.453_130_235_612_059_95,
                    -1.021_727_458_727_575,
                    0.613_821_922_587_981_4,
                    -0.438_114_864_564_824_6,
                ],
                [
                    -1.071_379_030_825_490_2,
                    -0.135_693_770_105_549_62,
                    0.026_842_619_856_511_04,
                ],
            ),
        ];
        for (variant, coef, intercept) in cases {
            let mut pa = PassiveAggressiveClassifier::new()
                .with_c(0.5)
                .with_variant(variant);
            pa.shuffle = false;
            let f = pa.fit(&x, &y).unwrap();
            for (a, b) in f.coef.iter().zip(coef) {
                assert!((a - b).abs() < 1e-12, "{variant:?}: {a} vs {b}");
            }
            for (a, b) in f.intercept.iter().zip(intercept) {
                assert!((a - b).abs() < 1e-12, "{variant:?}: {a} vs {b}");
            }
        }
    }
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
