//! PerpetualBooster: a gradient boosting machine with one complexity knob.
//!
//! Wraps the [`perpetual`](https://crates.io/crates/perpetual) crate
//! (Apache-2.0, pinned to 3.0.0-rc.2). Instead of a learning rate, tree
//! depth and number of rounds, PerpetualBooster takes a `budget`: higher
//! values fit more complex models, and the algorithm stops on its own when
//! further trees stop generalising. Missing values (`NaN`) are handled
//! natively.
//!
//! - [`PerpetualRegressor`]: squared, absolute, Huber, quantile, Poisson,
//!   Gamma, Tweedie, ... losses ([`PerpetualObjective`]).
//! - [`PerpetualClassifier`]: binary (one booster with log loss) and
//!   multiclass ([`perpetual::MultiOutputBooster`]); class labels may be any
//!   distinct values, as in the other classifiers.
//!
//! Fitted models persist through perpetual's own JSON format, so saved
//! models stay readable as long as perpetual can read them.
//!
//! Only available with the `perpetual` feature, which needs Rust 1.88.

use std::collections::HashSet;

use anofox_ml_core::{Fit, FitWeighted, Predict, PredictProba, Result, RustMlError};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use perpetual::booster::config::BoosterIO;
use perpetual::objective::Objective;
use perpetual::{Matrix, MultiOutputBooster, PerpetualBooster};
use serde::{Deserialize, Serialize};

/// Regression loss of a [`PerpetualRegressor`]. Parameters left at `None`
/// use perpetual's defaults.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub enum PerpetualObjective {
    #[default]
    SquaredLoss,
    AbsoluteLoss,
    HuberLoss {
        delta: Option<f64>,
    },
    AdaptiveHuberLoss {
        quantile: Option<f64>,
    },
    QuantileLoss {
        quantile: Option<f64>,
    },
    PoissonLoss,
    GammaLoss,
    TweedieLoss {
        p: Option<f64>,
    },
    FairLoss {
        c: Option<f64>,
    },
    MapeLoss,
    SquaredLogLoss,
}

impl PerpetualObjective {
    fn to_perpetual(self) -> Objective {
        match self {
            Self::SquaredLoss => Objective::SquaredLoss,
            Self::AbsoluteLoss => Objective::AbsoluteLoss,
            Self::HuberLoss { delta } => Objective::HuberLoss { delta },
            Self::AdaptiveHuberLoss { quantile } => Objective::AdaptiveHuberLoss { quantile },
            Self::QuantileLoss { quantile } => Objective::QuantileLoss { quantile },
            Self::PoissonLoss => Objective::PoissonLoss,
            Self::GammaLoss => Objective::GammaLoss,
            Self::TweedieLoss { p } => Objective::TweedieLoss { p },
            Self::FairLoss { c } => Objective::FairLoss { c },
            Self::MapeLoss => Objective::MapeLoss,
            Self::SquaredLogLoss => Objective::SquaredLogLoss,
        }
    }
}

/// Settings shared by the regressor and the classifier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PerpetualParams {
    /// Model complexity; higher fits more. perpetual recommends 0.5–2.
    pub budget: f32,
    /// Maximum number of histogram bins per feature.
    pub max_bin: u16,
    /// Training threads (`None`: all cores).
    pub num_threads: Option<usize>,
    pub seed: u64,
    /// Column indices treated as categorical (values are category codes).
    pub categorical_features: Option<Vec<usize>>,
    /// Grow a separate branch for missing values at each split.
    pub create_missing_branch: bool,
    /// Allow splits that only separate missing from non-missing values.
    pub allow_missing_splits: bool,
    /// Hard limit on boosting rounds.
    pub iteration_limit: Option<usize>,
    /// Hard limit on training time, in seconds.
    pub timeout: Option<f32>,
    /// Memory limit for training, in GB.
    pub memory_limit: Option<f32>,
    /// Stop after this many rounds without improvement.
    pub stopping_rounds: Option<usize>,
}

impl Default for PerpetualParams {
    fn default() -> Self {
        Self {
            budget: 0.5,
            max_bin: 256,
            num_threads: None,
            seed: 0,
            categorical_features: None,
            create_missing_branch: false,
            allow_missing_splits: true,
            iteration_limit: None,
            timeout: None,
            memory_limit: None,
            stopping_rounds: None,
        }
    }
}

impl PerpetualParams {
    fn validate(&self) -> Result<()> {
        if !(self.budget.is_finite() && self.budget > 0.0) {
            return Err(RustMlError::InvalidParameter(format!(
                "budget must be positive, got {}",
                self.budget
            )));
        }
        if self.max_bin < 2 {
            return Err(RustMlError::InvalidParameter(format!(
                "max_bin must be at least 2, got {}",
                self.max_bin
            )));
        }
        Ok(())
    }

    fn booster(&self, objective: Objective) -> PerpetualBooster {
        PerpetualBooster::default()
            .set_objective(objective)
            .set_budget(self.budget)
            .set_max_bin(self.max_bin)
            .set_num_threads(self.num_threads)
            .set_seed(self.seed)
            .set_categorical_features(self.categorical_set())
            .set_create_missing_branch(self.create_missing_branch)
            .set_allow_missing_splits(self.allow_missing_splits)
            .set_iteration_limit(self.iteration_limit)
            .set_timeout(self.timeout)
            .set_memory_limit(self.memory_limit)
            .set_stopping_rounds(self.stopping_rounds)
    }

    fn multi_booster(&self, n_boosters: usize) -> MultiOutputBooster {
        MultiOutputBooster::default()
            .set_n_boosters(n_boosters)
            .set_objective(Objective::LogLoss)
            .set_budget(self.budget)
            .set_max_bin(self.max_bin)
            .set_num_threads(self.num_threads)
            .set_seed(self.seed)
            .set_categorical_features(self.categorical_set())
            .set_create_missing_branch(self.create_missing_branch)
            .set_allow_missing_splits(self.allow_missing_splits)
            .set_iteration_limit(self.iteration_limit)
            .set_timeout(self.timeout)
            .set_memory_limit(self.memory_limit)
            .set_stopping_rounds(self.stopping_rounds)
    }

    fn categorical_set(&self) -> Option<HashSet<usize>> {
        self.categorical_features
            .as_ref()
            .map(|c| c.iter().copied().collect())
    }
}

fn perpetual_err(e: perpetual::errors::PerpetualError) -> RustMlError {
    RustMlError::InvalidParameter(format!("perpetual: {e}"))
}

/// `x` in column-major order, as perpetual's [`Matrix`] expects.
fn column_major(x: ArrayView2<f64>) -> Vec<f64> {
    x.t().iter().copied().collect()
}

fn check_fit_input(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    w: Option<ArrayView1<f64>>,
) -> Result<()> {
    if x.nrows() == 0 || x.ncols() == 0 {
        return Err(RustMlError::EmptyInput("x has no rows or columns".into()));
    }
    if y.len() != x.nrows() {
        return Err(RustMlError::ShapeMismatch(format!(
            "x has {} rows but y has {}",
            x.nrows(),
            y.len()
        )));
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(RustMlError::InvalidParameter("y must be finite".into()));
    }
    if let Some(w) = w {
        if w.len() != x.nrows() {
            return Err(RustMlError::ShapeMismatch(format!(
                "x has {} rows but sample_weight has {}",
                x.nrows(),
                w.len()
            )));
        }
        if w.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(RustMlError::InvalidParameter(
                "sample_weight must be finite and non-negative".into(),
            ));
        }
    }
    Ok(())
}

fn check_features(expected: usize, x: ArrayView2<f64>) -> Result<()> {
    if x.ncols() != expected {
        return Err(RustMlError::ShapeMismatch(format!(
            "expected {expected} features, got {}",
            x.ncols()
        )));
    }
    Ok(())
}

/// Serde for a perpetual booster: its own JSON, as a string.
mod booster_json {
    use super::BoosterIO;
    use serde::{de::Error as _, ser::Error as _, Deserialize, Deserializer, Serializer};

    pub fn serialize<B: BoosterIO, S: Serializer>(b: &B, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&b.json_dump().map_err(S::Error::custom)?)
    }

    pub fn deserialize<'de, B: BoosterIO, D: Deserializer<'de>>(d: D) -> Result<B, D::Error> {
        let json = String::deserialize(d)?;
        B::from_json(&json).map_err(D::Error::custom)
    }
}

// ============================================================
// Regressor
// ============================================================

/// PerpetualBooster for regression.
///
/// ```
/// use anofox_ml_core::{Fit, Predict};
/// use anofox_ml_ensemble::PerpetualRegressor;
/// use ndarray::Array2;
///
/// let x = Array2::from_shape_fn((200, 2), |(i, j)| ((i * (j + 3)) % 17) as f64);
/// let y = x.column(0).mapv(|v| 2.0 * v) + &x.column(1);
/// let model = PerpetualRegressor::default().fit(&x, &y).unwrap();
/// assert_eq!(model.predict(&x).unwrap().len(), 200);
/// ```
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PerpetualRegressor {
    pub objective: PerpetualObjective,
    #[serde(flatten)]
    pub params: PerpetualParams,
}

impl PerpetualRegressor {
    pub fn new(budget: f32) -> Self {
        Self::default().with_budget(budget)
    }

    pub fn with_budget(mut self, budget: f32) -> Self {
        self.params.budget = budget;
        self
    }

    pub fn with_objective(mut self, objective: PerpetualObjective) -> Self {
        self.objective = objective;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.params.seed = seed;
        self
    }

    pub fn with_num_threads(mut self, num_threads: Option<usize>) -> Self {
        self.params.num_threads = num_threads;
        self
    }
}

/// Fitted [`PerpetualRegressor`].
#[derive(Clone, Serialize, Deserialize)]
pub struct FittedPerpetualRegressor {
    #[serde(with = "booster_json")]
    booster: PerpetualBooster,
    n_features: usize,
}

impl std::fmt::Debug for FittedPerpetualRegressor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FittedPerpetualRegressor")
            .field("n_trees", &self.booster.get_prediction_trees().len())
            .field("n_features", &self.n_features)
            .finish()
    }
}

impl FittedPerpetualRegressor {
    /// The underlying perpetual booster (feature importances, contributions,
    /// ...).
    pub fn booster(&self) -> &PerpetualBooster {
        &self.booster
    }

    pub fn n_features(&self) -> usize {
        self.n_features
    }
}

impl FitWeighted<f64> for PerpetualRegressor {
    type Fitted = FittedPerpetualRegressor;

    fn fit_weighted_view(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        sample_weight: Option<ArrayView1<f64>>,
    ) -> Result<FittedPerpetualRegressor> {
        self.params.validate()?;
        check_fit_input(x, y, sample_weight)?;
        let data = column_major(x);
        let matrix = Matrix::new(&data, x.nrows(), x.ncols());
        let y = y.to_vec();
        let w = sample_weight.map(|w| w.to_vec());
        let mut booster = self.params.booster(self.objective.to_perpetual());
        booster
            .fit(&matrix, &y, w.as_deref(), None)
            .map_err(perpetual_err)?;
        Ok(FittedPerpetualRegressor {
            booster,
            n_features: x.ncols(),
        })
    }
}

impl Fit<f64> for PerpetualRegressor {
    type Fitted = FittedPerpetualRegressor;

    fn fit_view(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<FittedPerpetualRegressor> {
        self.fit_weighted_view(x, y, None)
    }
}

impl Predict<f64> for FittedPerpetualRegressor {
    fn predict_view(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        check_features(self.n_features, x)?;
        let data = column_major(x);
        let matrix = Matrix::new(&data, x.nrows(), x.ncols());
        Ok(Array1::from(self.booster.predict(&matrix, true)))
    }
}

// ============================================================
// Classifier
// ============================================================

/// PerpetualBooster for classification.
///
/// Two classes fit one booster with log loss; more fit perpetual's
/// multi-output booster (one-vs-rest or a shared-logit model, as perpetual
/// decides).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PerpetualClassifier {
    #[serde(flatten)]
    pub params: PerpetualParams,
    /// Reweight the classes of a binary problem to balance them
    /// (perpetual's `auto_class_weights`).
    pub auto_class_weights: bool,
}

impl Default for PerpetualClassifier {
    fn default() -> Self {
        Self {
            params: PerpetualParams::default(),
            auto_class_weights: true,
        }
    }
}

impl PerpetualClassifier {
    pub fn new(budget: f32) -> Self {
        Self::default().with_budget(budget)
    }

    pub fn with_budget(mut self, budget: f32) -> Self {
        self.params.budget = budget;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.params.seed = seed;
        self
    }

    pub fn with_num_threads(mut self, num_threads: Option<usize>) -> Self {
        self.params.num_threads = num_threads;
        self
    }

    pub fn with_auto_class_weights(mut self, auto: bool) -> Self {
        self.auto_class_weights = auto;
        self
    }
}

#[derive(Clone, Serialize, Deserialize)]
enum ClassBooster {
    Binary(#[serde(with = "booster_json")] PerpetualBooster),
    Multi(#[serde(with = "booster_json")] MultiOutputBooster),
}

/// Fitted [`PerpetualClassifier`].
#[derive(Clone, Serialize, Deserialize)]
pub struct FittedPerpetualClassifier {
    booster: ClassBooster,
    classes: Vec<f64>,
    n_features: usize,
}

impl std::fmt::Debug for FittedPerpetualClassifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FittedPerpetualClassifier")
            .field("classes", &self.classes)
            .field("n_features", &self.n_features)
            .finish()
    }
}

impl FittedPerpetualClassifier {
    /// Sorted distinct class labels; the columns of `predict_proba`.
    pub fn classes(&self) -> &[f64] {
        &self.classes
    }

    pub fn n_features(&self) -> usize {
        self.n_features
    }
}

impl FitWeighted<f64> for PerpetualClassifier {
    type Fitted = FittedPerpetualClassifier;

    fn fit_weighted_view(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        sample_weight: Option<ArrayView1<f64>>,
    ) -> Result<FittedPerpetualClassifier> {
        self.params.validate()?;
        check_fit_input(x, y, sample_weight)?;
        let mut classes: Vec<f64> = y.to_vec();
        classes.sort_by(f64::total_cmp);
        classes.dedup();
        if classes.len() < 2 {
            return Err(RustMlError::InvalidParameter(format!(
                "PerpetualClassifier needs at least 2 classes, got {}",
                classes.len()
            )));
        }
        let index = |v: f64| classes.partition_point(|c| *c < v);
        let data = column_major(x);
        let matrix = Matrix::new(&data, x.nrows(), x.ncols());
        let w = sample_weight.map(|w| w.to_vec());
        let n = x.nrows();
        let booster = if classes.len() == 2 {
            let y01: Vec<f64> = y.iter().map(|&v| index(v) as f64).collect();
            let mut b = self.params.booster(Objective::LogLoss);
            b.cfg.auto_class_weights = self.auto_class_weights;
            b.fit(&matrix, &y01, w.as_deref(), None)
                .map_err(perpetual_err)?;
            ClassBooster::Binary(b)
        } else {
            let k = classes.len();
            // One-hot targets, column-major (n × k).
            let mut onehot = vec![0.0; n * k];
            for (i, &v) in y.iter().enumerate() {
                onehot[index(v) * n + i] = 1.0;
            }
            let targets = Matrix::new(&onehot, n, k);
            let mut b = self.params.multi_booster(k);
            b.fit(&matrix, &targets, w.as_deref(), None)
                .map_err(perpetual_err)?;
            ClassBooster::Multi(b)
        };
        Ok(FittedPerpetualClassifier {
            booster,
            classes,
            n_features: x.ncols(),
        })
    }
}

impl Fit<f64> for PerpetualClassifier {
    type Fitted = FittedPerpetualClassifier;

    fn fit_view(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
    ) -> Result<FittedPerpetualClassifier> {
        self.fit_weighted_view(x, y, None)
    }
}

impl PredictProba<f64> for FittedPerpetualClassifier {
    fn predict_proba_view(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        check_features(self.n_features, x)?;
        let data = column_major(x);
        let matrix = Matrix::new(&data, x.nrows(), x.ncols());
        let n = x.nrows();
        match &self.booster {
            ClassBooster::Binary(b) => {
                let p1 = b.predict_proba(&matrix, true, false);
                Ok(Array2::from_shape_fn((n, 2), |(i, j)| {
                    if j == 1 {
                        p1[i]
                    } else {
                        1.0 - p1[i]
                    }
                }))
            }
            ClassBooster::Multi(b) => {
                // Row-major n × k.
                let p = b.predict_proba(&matrix, true);
                Array2::from_shape_vec((n, self.classes.len()), p)
                    .map_err(|e| RustMlError::ShapeMismatch(e.to_string()))
            }
        }
    }
}

impl Predict<f64> for FittedPerpetualClassifier {
    fn predict_view(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        let proba = self.predict_proba_view(x)?;
        Ok(proba
            .rows()
            .into_iter()
            .map(|row| {
                let best = row
                    .iter()
                    .enumerate()
                    .fold(0, |b, (j, &p)| if p > row[b] { j } else { b });
                self.classes[best]
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anofox_ml_core::persistence;
    use ndarray::Array2;

    fn regression_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_fn((300, 3), |(i, j)| {
            (((i * 7 + j * 13) % 29) as f64) / 29.0 + j as f64
        });
        let y = x
            .rows()
            .into_iter()
            .map(|r| 3.0 * r[0] - 2.0 * r[1] + (r[2] * 3.0).sin())
            .collect();
        (x, y)
    }

    fn classification_data(k: usize) -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_fn((300, 2), |(i, j)| ((i * (3 + j * 5)) % 101) as f64 / 101.0);
        // Labels 10, 20, ... to check that arbitrary labels survive.
        let y = x
            .rows()
            .into_iter()
            .map(|r| (((r[0] * k as f64) as usize).min(k - 1) as f64 + 1.0) * 10.0)
            .collect();
        (x, y)
    }

    fn r2(y: &Array1<f64>, p: &Array1<f64>) -> f64 {
        let mean = y.mean().unwrap();
        let ss_res: f64 = y.iter().zip(p).map(|(a, b)| (a - b).powi(2)).sum();
        let ss_tot: f64 = y.iter().map(|a| (a - mean).powi(2)).sum();
        1.0 - ss_res / ss_tot
    }

    #[test]
    fn regressor_fits_a_smooth_function() {
        let (x, y) = regression_data();
        let model = PerpetualRegressor::new(1.0)
            .with_num_threads(Some(1))
            .fit(&x, &y)
            .unwrap();
        let pred = model.predict(&x).unwrap();
        assert!(r2(&y, &pred) > 0.9, "r2 = {}", r2(&y, &pred));
    }

    #[test]
    fn regressor_handles_missing_values_and_weights() {
        let (mut x, y) = regression_data();
        x[[0, 0]] = f64::NAN;
        x[[5, 2]] = f64::NAN;
        let w = Array1::from_shape_fn(y.len(), |i| 1.0 + (i % 3) as f64);
        let model = PerpetualRegressor::default()
            .with_num_threads(Some(1))
            .fit_weighted(&x, &y, Some(&w))
            .unwrap();
        assert!(model.predict(&x).unwrap().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn regressor_quantile_objective_shifts_predictions() {
        let (x, y) = regression_data();
        let fit = |q| {
            PerpetualRegressor::default()
                .with_objective(PerpetualObjective::QuantileLoss { quantile: Some(q) })
                .with_num_threads(Some(1))
                .fit(&x, &y)
                .unwrap()
                .predict(&x)
                .unwrap()
                .mean()
                .unwrap()
        };
        assert!(fit(0.1) < fit(0.9));
    }

    #[test]
    fn binary_classifier_with_arbitrary_labels() {
        let (x, y) = classification_data(2);
        let model = PerpetualClassifier::default()
            .with_num_threads(Some(1))
            .fit(&x, &y)
            .unwrap();
        assert_eq!(model.classes(), &[10.0, 20.0]);
        let proba = model.predict_proba(&x).unwrap();
        assert_eq!(proba.dim(), (300, 2));
        for row in proba.rows() {
            assert!((row.sum() - 1.0).abs() < 1e-12);
        }
        let pred = model.predict(&x).unwrap();
        let acc = pred.iter().zip(&y).filter(|(a, b)| a == b).count() as f64 / 300.0;
        assert!(acc > 0.9, "accuracy {acc}");
    }

    #[test]
    fn multiclass_classifier() {
        let (x, y) = classification_data(3);
        let model = PerpetualClassifier::default()
            .with_num_threads(Some(1))
            .fit(&x, &y)
            .unwrap();
        assert_eq!(model.classes(), &[10.0, 20.0, 30.0]);
        let proba = model.predict_proba(&x).unwrap();
        assert_eq!(proba.dim(), (300, 3));
        for row in proba.rows() {
            assert!((row.sum() - 1.0).abs() < 1e-9, "{row}");
        }
        let pred = model.predict(&x).unwrap();
        let acc = pred.iter().zip(&y).filter(|(a, b)| a == b).count() as f64 / 300.0;
        assert!(acc > 0.85, "accuracy {acc}");
    }

    #[test]
    fn rejects_bad_input() {
        let (x, y) = regression_data();
        assert!(PerpetualRegressor::new(0.0).fit(&x, &y).is_err());
        assert!(PerpetualRegressor::default()
            .fit(&x, &y.slice(ndarray::s![..10]))
            .is_err());
        let one_class = Array1::from_elem(300, 1.0);
        assert!(PerpetualClassifier::default().fit(&x, &one_class).is_err());
        let model = PerpetualRegressor::default()
            .with_num_threads(Some(1))
            .fit(&x, &y)
            .unwrap();
        assert!(model.predict(&x.slice(ndarray::s![.., ..2])).is_err());
    }

    #[test]
    fn fitted_models_round_trip_through_persistence() {
        let (x, y) = regression_data();
        let reg = PerpetualRegressor::default()
            .with_num_threads(Some(1))
            .fit(&x, &y)
            .unwrap();
        let bytes = persistence::to_binary_bytes(&reg).unwrap();
        let back: FittedPerpetualRegressor = persistence::from_binary_bytes(&bytes).unwrap();
        assert_eq!(reg.predict(&x).unwrap(), back.predict(&x).unwrap());

        for k in [2, 3] {
            let (x, y) = classification_data(k);
            let clf = PerpetualClassifier::default()
                .with_num_threads(Some(1))
                .fit(&x, &y)
                .unwrap();
            let json = serde_json::to_string(&clf).unwrap();
            let back: FittedPerpetualClassifier = serde_json::from_str(&json).unwrap();
            assert_eq!(
                clf.predict_proba(&x).unwrap(),
                back.predict_proba(&x).unwrap()
            );
        }
    }

    #[test]
    fn config_json_uses_flat_parameters() {
        let cfg: PerpetualRegressor = serde_json::from_value(serde_json::json!({
            "budget": 1.5,
            "objective": { "HuberLoss": { "delta": 2.0 } },
        }))
        .unwrap();
        assert_eq!(cfg.params.budget, 1.5);
        assert_eq!(
            cfg.objective,
            PerpetualObjective::HuberLoss { delta: Some(2.0) }
        );
        let v = serde_json::to_value(PerpetualClassifier::default()).unwrap();
        assert_eq!(v["budget"], 0.5);
        assert_eq!(v["auto_class_weights"], true);
    }
}
