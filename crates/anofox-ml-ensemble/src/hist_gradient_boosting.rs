//! Histogram-based gradient boosting with sklearn's
//! `HistGradientBoostingRegressor` / `HistGradientBoostingClassifier` names
//! and defaults.
//!
//! Both are thin facades over the [`lgbm`](crate::lgbm) engine: features are
//! binned into at most `max_bins` (≤ 255) bins, trees grow leaf-wise up to
//! `max_leaf_nodes` leaves (depth-limited by `max_depth` if set), missing
//! values (`NaN`) get their own bin and a learned direction at every split,
//! and leaves are Newton steps with an L2 penalty. Sample weights, feature
//! importances and the engine's histogram subtraction come with it.

use anofox_ml_core::class_weight::{effective_sample_weight, ClassWeight};
pub use anofox_ml_core::numpy_random::TestSize;
use anofox_ml_core::numpy_random::{
    stratified_train_test_split_indices, train_test_split_indices, RandomState,
};
use anofox_ml_core::{Fit, FitWeighted, Predict, Result};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, AsArray, Axis, Ix2};

use crate::lgbm::{
    BinningMethod, EarlyStoppingScores, FittedLgbmClassifier, FittedLgbmRegressor, LgbmClassifier,
    LgbmFitOptions, LgbmObjective, LgbmRegressor, SklearnStopping,
};

/// Shared hyperparameters and builder methods of the two facades.
macro_rules! hist_gb_config {
    ($name:ident $(, $field:ident: $value:expr)*) => {
        impl $name {
            pub fn new() -> Self {
                Self {
                    n_estimators: 100,
                    learning_rate: 0.1,
                    max_leaf_nodes: Some(31),
                    max_depth: None,
                    min_samples_leaf: 20,
                    l2_regularization: 0.0,
                    max_bins: 255,
                    categorical_features: Vec::new(),
                    monotonic_cst: Vec::new(),
                    interaction_cst: None,
                    early_stopping: EarlyStopping::Auto,
                    validation_fraction: Some(TestSize::Fraction(0.1)),
                    n_iter_no_change: 10,
                    tol: 1e-7,
                    random_state: 0,
                    max_features: 1.0,
                    $($field: $value,)*
                }
            }

            /// Number of boosting rounds (sklearn's `max_iter`).
            pub fn with_n_estimators(mut self, n: usize) -> Self {
                self.n_estimators = n;
                self
            }
            pub fn with_learning_rate(mut self, lr: f64) -> Self {
                self.learning_rate = lr;
                self
            }
            /// Maximum leaves per tree (`None` = unlimited, depth-bound).
            pub fn with_max_leaf_nodes(mut self, n: Option<usize>) -> Self {
                self.max_leaf_nodes = n;
                self
            }
            pub fn with_max_depth(mut self, d: usize) -> Self {
                self.max_depth = Some(d);
                self
            }
            pub fn with_min_samples_leaf(mut self, m: usize) -> Self {
                self.min_samples_leaf = m;
                self
            }
            pub fn with_l2_regularization(mut self, l2: f64) -> Self {
                self.l2_regularization = l2;
                self
            }
            pub fn with_max_bins(mut self, b: usize) -> Self {
                self.max_bins = b;
                self
            }
            /// Column indices of categorical features (sklearn's
            /// `categorical_features`). Each distinct value is a category
            /// (at most `max_bins` per feature); `NaN` and categories unseen
            /// in training are treated as missing.
            pub fn with_categorical_features(mut self, features: Vec<usize>) -> Self {
                self.categorical_features = features;
                self
            }

            /// Per-feature monotonic constraint (sklearn's `monotonic_cst`):
            /// `1` increasing, `-1` decreasing, `0` none; one entry per
            /// feature (empty = none). Not allowed on categorical features
            /// (nor, for the classifier, with more than two classes).
            pub fn with_monotonic_cst(mut self, cst: Vec<i8>) -> Self {
                self.monotonic_cst = cst;
                self
            }
            /// Interaction constraints (sklearn's `interaction_cst`).
            pub fn with_interaction_cst(mut self, cst: InteractionConstraints) -> Self {
                self.interaction_cst = Some(cst);
                self
            }

            /// Early stopping (sklearn's `early_stopping`; default `Auto`:
            /// on above 10 000 samples).
            pub fn with_early_stopping(mut self, e: EarlyStopping) -> Self {
                self.early_stopping = e;
                self
            }
            /// Held-out share of the training data for early stopping
            /// (sklearn's `validation_fraction`, default 10 %); `None` stops
            /// on the training loss.
            pub fn with_validation_fraction(mut self, v: Option<TestSize>) -> Self {
                self.validation_fraction = v;
                self
            }
            /// Iterations without an improvement of more than `tol` before
            /// stopping (sklearn's `n_iter_no_change`, default 10).
            pub fn with_n_iter_no_change(mut self, n: usize) -> Self {
                self.n_iter_no_change = n;
                self
            }
            /// Minimum improvement of the score (sklearn's `tol`, default
            /// 1e-7).
            pub fn with_tol(mut self, tol: f64) -> Self {
                self.tol = tol;
                self
            }
            /// Seed of the validation split (sklearn's integer
            /// `random_state`; default 0, where sklearn's default `None`
            /// draws a fresh one).
            pub fn with_random_state(mut self, seed: u32) -> Self {
                self.random_state = seed;
                self
            }

            /// Share of the features tried at every split (sklearn's
            /// `max_features`, in (0, 1]; default 1). The subsets are drawn
            /// as sklearn draws them for the same `random_state`.
            pub fn with_max_features(mut self, f: f64) -> Self {
                self.max_features = f;
                self
            }

            /// sklearn's two seeds from `random_state`: the validation split's
            /// and the feature subsampling's.
            fn seeds(&self) -> (u32, u32) {
                let mut rs = RandomState::new(self.random_state);
                let split = rs.randint(u64::from(u32::MAX)) as u32;
                (split, rs.randint(u64::from(u32::MAX)) as u32)
            }

            /// Fit `engine` with sklearn's early stopping when it applies:
            /// the validation rows are split off as sklearn's
            /// `train_test_split` does (stratified by class for the
            /// classifier) and training runs on sklearn's training rows in
            /// its order.
            fn fit_engine<E: ScoredEngine>(
                &self,
                engine: &E,
                x: ArrayView2<f64>,
                y: ArrayView1<f64>,
                sample_weight: Option<ArrayView1<f64>>,
                stratify: bool,
                seed: u32,
            ) -> Result<(E::Fitted, EarlyStoppingScores)> {
                let enabled = match self.early_stopping {
                    EarlyStopping::Auto => x.nrows() > 10_000,
                    EarlyStopping::Enabled => true,
                    EarlyStopping::Disabled => false,
                };
                let opts = LgbmFitOptions {
                    sample_weight,
                    ..LgbmFitOptions::default()
                };
                if !enabled {
                    return engine.fit_scored(x, y, &opts, None);
                }
                if self.n_iter_no_change == 0 || self.tol.is_nan() || self.tol < 0.0 {
                    return Err(anofox_ml_core::RustMlError::InvalidParameter(
                        "early stopping needs n_iter_no_change >= 1 and tol >= 0".into(),
                    ));
                }
                let rule = SklearnStopping {
                    n_iter_no_change: self.n_iter_no_change,
                    tol: self.tol,
                };
                let Some(test_size) = self.validation_fraction else {
                    return engine.fit_scored(x, y, &opts, Some(rule));
                };
                let (train, val) = if stratify {
                    let mut classes: Vec<f64> = y.to_vec();
                    classes.sort_by(f64::total_cmp);
                    classes.dedup();
                    let labels: Vec<usize> = y
                        .iter()
                        .map(|v| classes.partition_point(|c| c < v))
                        .collect();
                    stratified_train_test_split_indices(&labels, test_size, seed)?
                } else {
                    train_test_split_indices(x.nrows(), test_size, seed)?
                };
                let (x_train, x_val) = (x.select(Axis(0), &train), x.select(Axis(0), &val));
                let (y_train, y_val) = (y.select(Axis(0), &train), y.select(Axis(0), &val));
                let sw = sample_weight.map(|w| (w.select(Axis(0), &train), w.select(Axis(0), &val)));
                let opts = LgbmFitOptions {
                    sample_weight: sw.as_ref().map(|(t, _)| t.view()),
                    eval_set: Some((&x_val, &y_val)),
                    eval_sample_weight: sw.as_ref().map(|(_, v)| v.view()),
                    ..LgbmFitOptions::default()
                };
                engine.fit_scored(x_train.view(), y_train.view(), &opts, Some(rule))
            }

            /// Configure `engine` for `x`: validates the categorical
            /// features and constraints against it as sklearn does.
            fn apply_constraints<E: ConstrainedEngine>(
                &self,
                x: ArrayView2<f64>,
                engine: &mut E,
            ) -> Result<()> {
                self.validate_categorical(x)?;
                if !(self.max_features > 0.0 && self.max_features <= 1.0) {
                    return Err(anofox_ml_core::RustMlError::InvalidParameter(format!(
                        "max_features must be in (0, 1], got {}",
                        self.max_features
                    )));
                }
                let p = x.ncols();
                if !self.monotonic_cst.is_empty() {
                    if self.monotonic_cst.len() != p {
                        return Err(anofox_ml_core::RustMlError::InvalidParameter(format!(
                            "monotonic_cst has {} entries but X has {p} features",
                            self.monotonic_cst.len()
                        )));
                    }
                    if self.monotonic_cst.iter().any(|c| !(-1..=1).contains(c)) {
                        return Err(anofox_ml_core::RustMlError::InvalidParameter(
                            "monotonic_cst entries must be -1, 0 or 1".into(),
                        ));
                    }
                    if self.categorical_features.iter().any(|&j| self.monotonic_cst[j] != 0) {
                        return Err(anofox_ml_core::RustMlError::InvalidParameter(
                            "categorical features cannot have monotonic constraints".into(),
                        ));
                    }
                }
                let groups = match &self.interaction_cst {
                    None => Vec::new(),
                    Some(cst) => cst.groups(p)?,
                };
                engine.set_constraints(self.monotonic_cst.clone(), groups);
                engine.set_feature_subsampling(self.max_features, self.seeds().1);
                Ok(())
            }

            /// sklearn's checks of `categorical_features` against `x`.
            fn validate_categorical(&self, x: ArrayView2<f64>) -> Result<()> {
                for &j in &self.categorical_features {
                    if j >= x.ncols() {
                        return Err(anofox_ml_core::RustMlError::InvalidParameter(format!(
                            "categorical feature {j} out of range for {} features",
                            x.ncols()
                        )));
                    }
                    let mut values: Vec<f64> =
                        x.column(j).iter().copied().filter(|v| !v.is_nan()).collect();
                    values.sort_by(f64::total_cmp);
                    values.dedup();
                    if values.len() > self.max_bins {
                        return Err(anofox_ml_core::RustMlError::InvalidParameter(format!(
                            "categorical feature {j} has {} categories, more than max_bins = {}",
                            values.len(),
                            self.max_bins
                        )));
                    }
                }
                Ok(())
            }

            fn validate(&self) -> Result<()> {
                if self.max_leaf_nodes.is_none() && self.max_depth.is_none() {
                    return Err(anofox_ml_core::RustMlError::InvalidParameter(
                        "max_leaf_nodes and max_depth cannot both be None".into(),
                    ));
                }
                if self.max_leaf_nodes.is_some_and(|n| n < 2) {
                    return Err(anofox_ml_core::RustMlError::InvalidParameter(
                        "max_leaf_nodes must be at least 2".into(),
                    ));
                }
                Ok(())
            }

            /// Leaf budget for the engine: unlimited leaves are bounded by
            /// the depth (a depth-`d` tree has at most `2^d` leaves).
            fn num_leaves(&self) -> usize {
                self.max_leaf_nodes.unwrap_or_else(|| {
                    1usize << self.max_depth.unwrap_or(0).min(usize::BITS as usize - 2)
                })
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

/// sklearn's `interaction_cst`: which features may appear together in a
/// branch of a tree.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum InteractionConstraints {
    /// Groups of feature indices; a branch only splits on features of a
    /// group that contains every feature split on above it. Features in no
    /// group form one more group.
    Groups(Vec<Vec<usize>>),
    /// Every pair of features is a group (`"pairwise"`).
    Pairwise,
    /// Every feature is a group of its own (`"no_interactions"`).
    NoInteractions,
}

impl InteractionConstraints {
    /// The groups for `p` features, deduplicated and with the unlisted
    /// features as a last group (sklearn's `_check_interaction_cst`).
    fn groups(&self, p: usize) -> Result<Vec<Vec<usize>>> {
        let mut groups: Vec<Vec<usize>> = match self {
            Self::Groups(g) => g.clone(),
            Self::Pairwise => (0..p)
                .flat_map(|i| (i + 1..p).map(move |j| vec![i, j]))
                .collect(),
            Self::NoInteractions => (0..p).map(|i| vec![i]).collect(),
        };
        if let Some(&bad) = groups.iter().flatten().find(|&&j| j >= p) {
            return Err(anofox_ml_core::RustMlError::InvalidParameter(format!(
                "interaction constraint index {bad} out of range for {p} features"
            )));
        }
        for g in &mut groups {
            g.sort_unstable();
            g.dedup();
        }
        let rest: Vec<usize> = (0..p)
            .filter(|j| !groups.iter().any(|g| g.contains(j)))
            .collect();
        if !rest.is_empty() {
            groups.push(rest);
        }
        Ok(groups)
    }
}

/// sklearn's `early_stopping`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum EarlyStopping {
    /// On when there are more than 10 000 samples (sklearn's `"auto"`).
    #[default]
    Auto,
    Enabled,
    Disabled,
}

/// Engines the facades fit with sklearn's early stopping.
trait ScoredEngine {
    type Fitted;
    fn fit_scored(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        opts: &LgbmFitOptions<'_>,
        stopping: Option<SklearnStopping>,
    ) -> Result<(Self::Fitted, EarlyStoppingScores)>;
}

impl ScoredEngine for LgbmRegressor {
    type Fitted = FittedLgbmRegressor;
    fn fit_scored(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        opts: &LgbmFitOptions<'_>,
        stopping: Option<SklearnStopping>,
    ) -> Result<(Self::Fitted, EarlyStoppingScores)> {
        LgbmRegressor::fit_scored(self, x, y, opts, stopping)
    }
}

impl ScoredEngine for LgbmClassifier {
    type Fitted = FittedLgbmClassifier;
    fn fit_scored(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        opts: &LgbmFitOptions<'_>,
        stopping: Option<SklearnStopping>,
    ) -> Result<(Self::Fitted, EarlyStoppingScores)> {
        LgbmClassifier::fit_scored(self, x, y, opts, stopping)
    }
}

/// Deserialize a field added after binary format 1: format-1 (bincode)
/// files end before it, which leaves the default.
fn lenient<'de, D, T>(d: D) -> std::result::Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::Deserialize<'de> + Default,
{
    Ok(T::deserialize(d).unwrap_or_default())
}

/// Engines the facades configure with constraints.
trait ConstrainedEngine {
    fn set_constraints(&mut self, monotone: Vec<i8>, interactions: Vec<Vec<usize>>);
    fn set_feature_subsampling(&mut self, fraction: f64, seed: u32);
}

impl ConstrainedEngine for LgbmRegressor {
    fn set_constraints(&mut self, monotone: Vec<i8>, interactions: Vec<Vec<usize>>) {
        self.monotone_constraints = monotone;
        self.interaction_cst = interactions;
    }
    fn set_feature_subsampling(&mut self, fraction: f64, seed: u32) {
        self.feature_fraction_per_split = fraction;
        self.feature_subsample_seed = seed;
    }
}

impl ConstrainedEngine for LgbmClassifier {
    fn set_constraints(&mut self, monotone: Vec<i8>, interactions: Vec<Vec<usize>>) {
        self.monotone_constraints = monotone;
        self.interaction_cst = interactions;
    }
    fn set_feature_subsampling(&mut self, fraction: f64, seed: u32) {
        self.feature_fraction_per_split = fraction;
        self.feature_subsample_seed = seed;
    }
}

/// Loss of a [`HistGradientBoostingRegressor`] (sklearn's `loss`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum HistGradientBoostingLoss {
    #[default]
    SquaredError,
    /// Leaves are refit to residual medians.
    AbsoluteError,
    /// Pinball loss at level `quantile`; leaves are refit to residual
    /// quantiles.
    Quantile,
    /// Half Poisson deviance with a log link; targets must be non-negative
    /// with a positive sum.
    Poisson,
    /// Half Gamma deviance with a log link; targets must be positive.
    Gamma,
}

/// Histogram-based gradient boosting regressor (squared error, absolute
/// error, quantile, Poisson or Gamma loss).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HistGradientBoostingRegressor {
    pub n_estimators: usize,
    pub learning_rate: f64,
    /// Maximum leaves per tree (sklearn default 31).
    #[serde(default = "default_leaves")]
    pub max_leaf_nodes: Option<usize>,
    /// Maximum depth (sklearn default `None`).
    pub max_depth: Option<usize>,
    pub min_samples_leaf: usize,
    pub l2_regularization: f64,
    pub max_bins: usize,
    /// Column indices of categorical features.
    #[serde(default)]
    pub categorical_features: Vec<usize>,
    /// Per-feature monotonic constraints (empty = none).
    #[serde(default)]
    pub monotonic_cst: Vec<i8>,
    /// Interaction constraints.
    #[serde(default)]
    pub interaction_cst: Option<InteractionConstraints>,
    #[serde(default)]
    pub early_stopping: EarlyStopping,
    #[serde(default = "default_validation_fraction")]
    pub validation_fraction: Option<TestSize>,
    #[serde(default = "default_n_iter_no_change")]
    pub n_iter_no_change: usize,
    #[serde(default = "default_tol")]
    pub tol: f64,
    #[serde(default)]
    pub random_state: u32,
    /// Share of the features tried at every split.
    #[serde(default = "default_max_features")]
    pub max_features: f64,
    #[serde(default)]
    pub loss: HistGradientBoostingLoss,
    /// Quantile level in (0, 1) for `loss = Quantile`.
    #[serde(default)]
    pub quantile: Option<f64>,
}

/// Histogram-based gradient boosting classifier (binary log-loss or
/// multinomial softmax).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HistGradientBoostingClassifier {
    pub n_estimators: usize,
    pub learning_rate: f64,
    /// Maximum leaves per tree (sklearn default 31).
    #[serde(default = "default_leaves")]
    pub max_leaf_nodes: Option<usize>,
    /// Maximum depth (sklearn default `None`).
    pub max_depth: Option<usize>,
    pub min_samples_leaf: usize,
    pub l2_regularization: f64,
    pub max_bins: usize,
    /// Column indices of categorical features.
    #[serde(default)]
    pub categorical_features: Vec<usize>,
    /// Per-feature monotonic constraints (empty = none).
    #[serde(default)]
    pub monotonic_cst: Vec<i8>,
    /// Interaction constraints.
    #[serde(default)]
    pub interaction_cst: Option<InteractionConstraints>,
    #[serde(default)]
    pub early_stopping: EarlyStopping,
    #[serde(default = "default_validation_fraction")]
    pub validation_fraction: Option<TestSize>,
    #[serde(default = "default_n_iter_no_change")]
    pub n_iter_no_change: usize,
    #[serde(default = "default_tol")]
    pub tol: f64,
    #[serde(default)]
    pub random_state: u32,
    /// Share of the features tried at every split.
    #[serde(default = "default_max_features")]
    pub max_features: f64,
    /// Class weights (sklearn's `class_weight`).
    #[serde(default)]
    pub class_weight: Option<ClassWeight>,
}

fn default_leaves() -> Option<usize> {
    Some(31)
}

fn default_validation_fraction() -> Option<TestSize> {
    Some(TestSize::Fraction(0.1))
}

fn default_n_iter_no_change() -> usize {
    10
}

fn default_tol() -> f64 {
    1e-7
}

fn default_max_features() -> f64 {
    1.0
}

hist_gb_config!(
    HistGradientBoostingRegressor,
    loss: HistGradientBoostingLoss::SquaredError,
    quantile: None
);
hist_gb_config!(HistGradientBoostingClassifier, class_weight: None);

impl HistGradientBoostingClassifier {
    /// Class weights (sklearn's `class_weight`), multiplied into the sample
    /// weights.
    pub fn with_class_weight(mut self, w: Option<ClassWeight>) -> Self {
        self.class_weight = w;
        self
    }
}

impl HistGradientBoostingRegressor {
    pub fn with_loss(mut self, loss: HistGradientBoostingLoss) -> Self {
        self.loss = loss;
        self
    }

    /// Quantile loss at level `q` (sklearn's `loss="quantile", quantile=q`).
    pub fn with_quantile(mut self, q: f64) -> Self {
        self.loss = HistGradientBoostingLoss::Quantile;
        self.quantile = Some(q);
        self
    }

    fn objective(&self) -> Result<LgbmObjective> {
        Ok(match self.loss {
            HistGradientBoostingLoss::SquaredError => LgbmObjective::Regression,
            HistGradientBoostingLoss::AbsoluteError => LgbmObjective::RegressionL1,
            HistGradientBoostingLoss::Quantile => match self.quantile {
                Some(alpha) if alpha > 0.0 && alpha < 1.0 => LgbmObjective::Quantile { alpha },
                other => {
                    return Err(anofox_ml_core::RustMlError::InvalidParameter(format!(
                        "loss = Quantile needs quantile in (0, 1), got {other:?}"
                    )))
                }
            },
            HistGradientBoostingLoss::Poisson => LgbmObjective::Poisson,
            HistGradientBoostingLoss::Gamma => LgbmObjective::Gamma,
        })
    }

    fn engine(&self) -> Result<LgbmRegressor> {
        self.validate()?;
        let mut engine = LgbmRegressor::new()
            .with_objective(self.objective()?)
            // sklearn's Poisson hessian is exactly exp(raw).
            .with_poisson_max_delta_step(0.0)
            .with_n_estimators(self.n_estimators)
            .with_learning_rate(self.learning_rate)
            .with_num_leaves(self.num_leaves())
            .with_max_depth(self.max_depth)
            .with_min_child_samples(self.min_samples_leaf)
            .with_reg_lambda(self.l2_regularization)
            .with_max_bins(self.max_bins)
            .with_categorical_features(self.categorical_features.clone())
            .with_binning(BinningMethod::Sklearn);
        engine.sklearn_compat = true;
        Ok(engine)
    }
}

impl HistGradientBoostingClassifier {
    fn engine(&self) -> Result<LgbmClassifier> {
        self.validate()?;
        let mut engine = LgbmClassifier::new()
            .with_n_estimators(self.n_estimators)
            .with_learning_rate(self.learning_rate)
            .with_num_leaves(self.num_leaves())
            .with_max_depth(self.max_depth)
            .with_min_child_samples(self.min_samples_leaf)
            .with_reg_lambda(self.l2_regularization)
            .with_max_bins(self.max_bins)
            .with_categorical_features(self.categorical_features.clone())
            .with_binning(BinningMethod::Sklearn);
        // sklearn's multinomial hessian is p(1 - p), without LightGBM's
        // K / (K - 1) factor.
        engine.softmax_hessian_factor = false;
        engine.sklearn_compat = true;
        Ok(engine)
    }
}

/// Fitted histogram-based gradient boosting regressor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedHistGradientBoostingRegressor {
    inner: FittedLgbmRegressor,
    #[serde(default, deserialize_with = "lenient")]
    scores: EarlyStoppingScores,
}

/// Iteration count and early-stopping scores of a fitted facade.
macro_rules! fitted_scores {
    ($name:ident) => {
        impl $name {
            /// Number of boosting iterations run (sklearn's `n_iter_`; fewer
            /// than `n_estimators` after early stopping).
            pub fn n_iter(&self) -> usize {
                self.inner.n_estimators()
            }

            /// Training scores (negative mean loss) before the first and
            /// after every iteration, with early stopping (sklearn's
            /// `train_score_`); empty without.
            pub fn train_score(&self) -> &[f64] {
                &self.scores.train
            }

            /// The same on the validation data (sklearn's
            /// `validation_score_`); empty without early stopping or
            /// validation data.
            pub fn validation_score(&self) -> &[f64] {
                &self.scores.validation
            }
        }
    };
}

fitted_scores!(FittedHistGradientBoostingRegressor);
fitted_scores!(FittedHistGradientBoostingClassifier);

impl FittedHistGradientBoostingRegressor {
    pub fn n_estimators(&self) -> usize {
        self.inner.n_estimators()
    }

    /// Predictions after each iteration (sklearn's `staged_predict`).
    pub fn staged_predict<'a>(
        &self,
        x: impl AsArray<'a, f64, Ix2>,
    ) -> Result<impl Iterator<Item = Array1<f64>> + '_> {
        self.inner.staged_predict(x)
    }

    /// Feature importances by total split gain (normalised to sum to 1).
    pub fn feature_importances(&self) -> Array1<f64> {
        self.inner.feature_importances()
    }
}

/// Fitted histogram-based gradient boosting classifier.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedHistGradientBoostingClassifier {
    inner: FittedLgbmClassifier,
    #[serde(default, deserialize_with = "lenient")]
    scores: EarlyStoppingScores,
}

impl FittedHistGradientBoostingClassifier {
    pub fn classes(&self) -> &[f64] {
        self.inner.classes()
    }

    pub fn n_estimators(&self) -> usize {
        self.inner.n_estimators()
    }

    /// Feature importances by total split gain (normalised to sum to 1).
    pub fn feature_importances(&self) -> Array1<f64> {
        self.inner.feature_importances()
    }

    /// Class probabilities, columns in [`classes`](Self::classes) order.
    pub fn predict_proba<'a>(&self, x: impl AsArray<'a, f64, Ix2>) -> Result<Array2<f64>> {
        let x: ArrayView2<f64> = x.into();
        self.inner.predict_proba(x.view())
    }

    /// Class probabilities after each iteration (sklearn's
    /// `staged_predict_proba`).
    pub fn staged_predict_proba<'a>(
        &self,
        x: impl AsArray<'a, f64, Ix2>,
    ) -> Result<impl Iterator<Item = Array2<f64>> + '_> {
        self.inner.staged_predict_proba(x)
    }

    /// Predicted classes after each iteration (sklearn's `staged_predict`).
    pub fn staged_predict<'a>(
        &self,
        x: impl AsArray<'a, f64, Ix2>,
    ) -> Result<impl Iterator<Item = Array1<f64>> + '_> {
        self.inner.staged_predict(x)
    }
}

impl Fit<f64> for HistGradientBoostingRegressor {
    type Fitted = FittedHistGradientBoostingRegressor;

    fn fit_view(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<Self::Fitted> {
        self.fit_weighted(x, y, None)
    }
}

impl FitWeighted<f64> for HistGradientBoostingRegressor {
    type Fitted = FittedHistGradientBoostingRegressor;

    fn fit_weighted_view(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        sample_weight: Option<ArrayView1<f64>>,
    ) -> Result<Self::Fitted> {
        let mut engine = self.engine()?;
        self.apply_constraints(x, &mut engine)?;
        let (inner, scores) =
            self.fit_engine(&engine, x, y, sample_weight, false, self.seeds().0)?;
        Ok(FittedHistGradientBoostingRegressor { inner, scores })
    }
}

impl Fit<f64> for HistGradientBoostingClassifier {
    type Fitted = FittedHistGradientBoostingClassifier;

    fn fit_view(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<Self::Fitted> {
        self.fit_weighted(x, y, None)
    }
}

impl FitWeighted<f64> for HistGradientBoostingClassifier {
    type Fitted = FittedHistGradientBoostingClassifier;

    fn fit_weighted_view(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        sample_weight: Option<ArrayView1<f64>>,
    ) -> Result<Self::Fitted> {
        let mut engine = self.engine()?;
        self.apply_constraints(x, &mut engine)?;
        if self.monotonic_cst.iter().any(|&c| c != 0) {
            let mut classes: Vec<f64> = y.to_vec();
            classes.sort_by(f64::total_cmp);
            classes.dedup();
            if classes.len() > 2 {
                return Err(anofox_ml_core::RustMlError::InvalidParameter(
                    "monotonic constraints are not supported for multiclass classification".into(),
                ));
            }
        }
        // sklearn multiplies the class weights into the sample weights
        // before the validation split.
        let weights = effective_sample_weight(y, sample_weight, self.class_weight.as_ref())?;
        let (inner, scores) = self.fit_engine(
            &engine,
            x,
            y,
            weights.as_ref().map(|w| w.view()),
            true,
            self.seeds().0,
        )?;
        Ok(FittedHistGradientBoostingClassifier { inner, scores })
    }
}

impl Predict<f64> for FittedHistGradientBoostingRegressor {
    fn predict_view(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        self.inner.predict(x)
    }
}

impl Predict<f64> for FittedHistGradientBoostingClassifier {
    fn predict_view(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        self.inner.predict(x)
    }
}

impl anofox_ml_core::PredictProba<f64> for FittedHistGradientBoostingClassifier {
    fn predict_proba_view(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        self.inner.predict_proba(x)
    }
}

impl anofox_ml_core::RegressorScore<f64> for FittedHistGradientBoostingRegressor {}
impl anofox_ml_core::ClassifierScore<f64> for FittedHistGradientBoostingClassifier {}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn data() -> (Array2<f64>, Array1<f64>) {
        let n = 200;
        let x = Array2::from_shape_fn((n, 3), |(i, j)| ((i * (j + 2) * 37) % 101) as f64 / 10.0);
        let y = x.column(0).mapv(|v| v * 2.0) - x.column(1);
        (x, y)
    }

    #[test]
    fn quantile_loss_brackets_the_data() {
        let n = 400;
        let x = Array2::from_shape_fn((n, 2), |(i, j)| ((i * (j + 3)) % 41) as f64 / 41.0);
        // y = x0 + noise uniform on [-1, 1).
        let y: Array1<f64> = (0..n)
            .map(|i| x[[i, 0]] + ((i * 7919) % 200) as f64 / 100.0 - 1.0)
            .collect();
        let frac_below = |q: f64| {
            let f = HistGradientBoostingRegressor::new()
                .with_n_estimators(50)
                .with_quantile(q)
                .fit(&x, &y)
                .unwrap();
            let p = f.predict(&x).unwrap();
            y.iter().zip(&p).filter(|(y, p)| y <= p).count() as f64 / n as f64
        };
        for q in [0.1, 0.5, 0.9] {
            let below = frac_below(q);
            assert!((below - q).abs() < 0.08, "q = {q}: {below}");
        }
        let l1 = HistGradientBoostingRegressor::new()
            .with_loss(HistGradientBoostingLoss::AbsoluteError)
            .with_n_estimators(50)
            .fit(&x, &y)
            .unwrap();
        let median = HistGradientBoostingRegressor::new()
            .with_quantile(0.5)
            .with_n_estimators(50)
            .fit(&x, &y)
            .unwrap();
        assert_eq!(l1.predict(&x).unwrap(), median.predict(&x).unwrap());
        assert!(HistGradientBoostingRegressor::new()
            .with_loss(HistGradientBoostingLoss::Quantile)
            .fit(&x, &y)
            .is_err());
        assert!(HistGradientBoostingRegressor::new()
            .with_quantile(1.0)
            .fit(&x, &y)
            .is_err());
    }

    #[test]
    fn log_link_losses_predict_positive_means_and_validate_targets() {
        let (x, y) = data();
        let counts = y.mapv(|v| (v.max(0.0) / 3.0).round());
        let positive = y.mapv(|v| v.abs() + 0.5);
        for (loss, target) in [
            (HistGradientBoostingLoss::Poisson, &counts),
            (HistGradientBoostingLoss::Gamma, &positive),
        ] {
            let p = HistGradientBoostingRegressor::new()
                .with_loss(loss)
                .with_n_estimators(30)
                .fit(&x, target)
                .unwrap()
                .predict(&x)
                .unwrap();
            assert!(p.iter().all(|&v| v > 0.0), "{loss:?}");
            // The log-link baseline keeps the mean of the predictions close
            // to the mean target.
            let rel = (p.mean().unwrap() - target.mean().unwrap()).abs() / target.mean().unwrap();
            assert!(rel < 0.05, "{loss:?}: {rel}");
        }
        let fit = |loss, y: &Array1<f64>| {
            HistGradientBoostingRegressor::new()
                .with_loss(loss)
                .fit(&x, y)
        };
        assert!(fit(HistGradientBoostingLoss::Poisson, &(&counts - 1.0)).is_err());
        assert!(fit(HistGradientBoostingLoss::Poisson, &Array1::zeros(y.len())).is_err());
        assert!(fit(HistGradientBoostingLoss::Gamma, &counts).is_err());
    }

    #[test]
    fn categorical_features_split_by_category_and_validate() {
        // y depends only on the category of feature 0 (codes not ordered by
        // effect); missing values form their own group.
        let codes = [5.0, -1.0, 2.5, 9.0];
        let effect = [3.0, -2.0, 1.0, -4.0];
        let n = 240;
        let mut x =
            Array2::from_shape_fn(
                (n, 2),
                |(i, j)| {
                    if j == 0 {
                        codes[i % 4]
                    } else {
                        (i % 7) as f64
                    }
                },
            );
        let mut y = Array1::from_shape_fn(n, |i| effect[i % 4]);
        for i in (0..n).step_by(8) {
            x[[i, 0]] = f64::NAN;
            y[i] = 6.0;
        }
        let m = HistGradientBoostingRegressor::new()
            .with_n_estimators(50)
            .with_categorical_features(vec![0])
            .fit(&x, &y)
            .unwrap();
        let p = m.predict(&x).unwrap();
        assert!(p.iter().zip(&y).all(|(a, b)| (a - b).abs() < 0.05));
        // An unseen category is treated as missing.
        let unseen = m.predict(&array![[42.0, 1.0], [f64::NAN, 1.0]]).unwrap();
        assert_eq!(unseen[0], unseen[1]);

        let fit = |features, max_bins| {
            HistGradientBoostingRegressor::new()
                .with_categorical_features(features)
                .with_max_bins(max_bins)
                .fit(&x, &y)
        };
        assert!(fit(vec![2], 255).is_err());
        assert!(fit(vec![1], 6).is_err());
        assert!(fit(vec![1], 7).is_ok());
    }

    #[test]
    fn constraints_are_enforced_and_validated() {
        let (x, _) = data();
        // A target that decreases in feature 0 where the constraint says up.
        let y = x.column(0).mapv(|v| -(v - 5.0).powi(2)) + x.column(1);
        let m = HistGradientBoostingRegressor::new()
            .with_n_estimators(30)
            .with_monotonic_cst(vec![1, 0, 0])
            .fit(&x, &y)
            .unwrap();
        let grid =
            Array2::from_shape_fn((50, 3), |(i, j)| if j == 0 { i as f64 / 5.0 } else { 3.0 });
        let p = m.predict(&grid).unwrap();
        assert!(p.windows(2).into_iter().all(|w| w[1] >= w[0]));

        // No interactions: every tree splits on a single feature, so the
        // model is additive.
        let m = HistGradientBoostingRegressor::new()
            .with_n_estimators(30)
            .with_interaction_cst(InteractionConstraints::NoInteractions)
            .fit(&x, &y)
            .unwrap();
        let at = |a: f64, b: f64| m.predict(&array![[a, b, 1.0]]).unwrap()[0];
        let interaction = at(2.0, 3.0) - at(2.0, 7.0) - at(8.0, 3.0) + at(8.0, 7.0);
        assert!(interaction.abs() < 1e-9, "{interaction}");

        let err = |m: HistGradientBoostingRegressor| m.fit(&x, &y).is_err();
        let base = || HistGradientBoostingRegressor::new().with_n_estimators(2);
        assert!(err(base().with_monotonic_cst(vec![1, 0])));
        assert!(err(base().with_monotonic_cst(vec![2, 0, 0])));
        assert!(err(base()
            .with_monotonic_cst(vec![1, 0, 0])
            .with_categorical_features(vec![0])));
        assert!(err(base().with_interaction_cst(
            InteractionConstraints::Groups(vec![vec![0, 3]])
        )));
        let multiclass = y.mapv(|v| (v.abs() as i64 % 3) as f64);
        assert!(HistGradientBoostingClassifier::new()
            .with_monotonic_cst(vec![1, 0, 0])
            .fit(&x, &multiclass)
            .is_err());
    }

    #[test]
    fn interaction_groups_add_the_unlisted_features() {
        let g = InteractionConstraints::Groups(vec![vec![2, 0, 2]])
            .groups(4)
            .unwrap();
        assert_eq!(g, vec![vec![0, 2], vec![1, 3]]);
        assert_eq!(
            InteractionConstraints::Pairwise.groups(3).unwrap(),
            vec![vec![0, 1], vec![0, 2], vec![1, 2]]
        );
    }

    #[test]
    fn early_stopping_modes() {
        let (x, y) = data();
        // Noise, so that the validation loss stops improving.
        let y = &y + &Array1::from_shape_fn(y.len(), |i| ((i * 7919) % 200) as f64 / 25.0 - 4.0);
        let fit = |m: HistGradientBoostingRegressor| m.with_n_estimators(500).fit(&x, &y);
        // `Auto` is off on 200 samples: no scores, every iteration.
        let off = fit(HistGradientBoostingRegressor::new()).unwrap();
        assert_eq!(off.n_iter(), 500);
        assert!(off.train_score().is_empty() && off.validation_score().is_empty());
        let on =
            fit(HistGradientBoostingRegressor::new().with_early_stopping(EarlyStopping::Enabled))
                .unwrap();
        assert!(on.n_iter() < 500);
        assert_eq!(on.train_score().len(), on.n_iter() + 1);
        assert_eq!(on.validation_score().len(), on.n_iter() + 1);
        let train_only = fit(HistGradientBoostingRegressor::new()
            .with_early_stopping(EarlyStopping::Enabled)
            .with_validation_fraction(None))
        .unwrap();
        assert!(train_only.validation_score().is_empty());
        assert_eq!(train_only.train_score().len(), train_only.n_iter() + 1);
        // Auto switches on above 10 000 samples.
        let xb = Array2::from_shape_fn((10_001, 1), |(i, _)| (i % 97) as f64);
        let yb = xb.column(0).mapv(|v| (v / 10.0).sin());
        let big = HistGradientBoostingRegressor::new()
            .with_n_estimators(300)
            .fit(&xb, &yb)
            .unwrap();
        assert!(!big.validation_score().is_empty());
        assert!(fit(HistGradientBoostingRegressor::new()
            .with_early_stopping(EarlyStopping::Enabled)
            .with_n_iter_no_change(0))
        .is_err());
        assert!(fit(HistGradientBoostingRegressor::new()
            .with_early_stopping(EarlyStopping::Enabled)
            .with_validation_fraction(Some(TestSize::Fraction(1.5))))
        .is_err());
    }

    #[test]
    fn max_features_is_validated_and_seeded() {
        let (x, y) = data();
        let fit = |f: f64, seed: u32| {
            HistGradientBoostingRegressor::new()
                .with_n_estimators(10)
                .with_max_features(f)
                .with_random_state(seed)
                .fit(&x, &y)
                .map(|m| m.predict(&x).unwrap())
        };
        assert!(fit(0.0, 0).is_err() && fit(1.5, 0).is_err());
        assert_eq!(fit(0.5, 1).unwrap(), fit(0.5, 1).unwrap());
        assert_ne!(fit(0.5, 1).unwrap(), fit(0.5, 2).unwrap());
        // Without subsampling the seed does not matter.
        assert_eq!(fit(1.0, 1).unwrap(), fit(1.0, 2).unwrap());
    }

    #[test]
    fn staged_predictions_end_at_predict() {
        let (x, y) = data();
        let counts = y.mapv(|v| (v.max(0.0) / 3.0).round());
        let m = HistGradientBoostingRegressor::new()
            .with_loss(HistGradientBoostingLoss::Poisson)
            .with_n_estimators(15)
            .fit(&x, &counts)
            .unwrap();
        let stages: Vec<Array1<f64>> = m.staged_predict(&x).unwrap().collect();
        assert_eq!(stages.len(), m.n_iter());
        assert_eq!(stages.last().unwrap(), &m.predict(&x).unwrap());
        assert!(stages[0].iter().all(|&v| v > 0.0));
        assert!(m.staged_predict(&x.slice(ndarray::s![.., ..2])).is_err());

        for yc in [
            y.mapv(|v| if v > 5.0 { 1.0 } else { 0.0 }),
            y.mapv(|v| (v.abs() as i64 % 3) as f64),
        ] {
            let c = HistGradientBoostingClassifier::new()
                .with_n_estimators(12)
                .fit(&x, &yc)
                .unwrap();
            let proba: Vec<Array2<f64>> = c.staged_predict_proba(&x).unwrap().collect();
            let labels: Vec<Array1<f64>> = c.staged_predict(&x).unwrap().collect();
            assert_eq!((proba.len(), labels.len()), (12, 12));
            assert_eq!(proba.last().unwrap(), &c.predict_proba(&x).unwrap());
            assert_eq!(labels.last().unwrap(), &c.predict(&x).unwrap());
        }
    }

    #[test]
    fn regressor_fits_and_matches_the_engine() {
        let (x, y) = data();
        let hgb = HistGradientBoostingRegressor::new().with_n_estimators(30);
        let f = hgb.fit(&x, &y).unwrap();
        let direct = hgb.engine().unwrap().fit(&x, &y).unwrap();
        assert_eq!(f.predict(&x).unwrap(), direct.predict(&x).unwrap());
        let r2 = anofox_ml_core::RegressorScore::score(&f, &x, &y).unwrap();
        assert!(r2 > 0.95, "r2 = {r2}");
        assert_eq!(f.feature_importances().len(), 3);
    }

    #[test]
    fn classifier_binary_and_multiclass_with_nan() {
        let (mut x, y) = data();
        x[[3, 1]] = f64::NAN;
        x[[50, 0]] = f64::NAN;
        for yc in [
            y.mapv(|v| if v > 5.0 { 1.0 } else { 0.0 }),
            y.mapv(|v| (v.abs() as i64 % 3) as f64),
        ] {
            let f = HistGradientBoostingClassifier::new()
                .with_n_estimators(30)
                .with_min_samples_leaf(5)
                .fit(&x, &yc)
                .unwrap();
            let proba = f.predict_proba(&x).unwrap();
            assert_eq!(proba.ncols(), f.classes().len());
            assert!(proba
                .rows()
                .into_iter()
                .all(|r| (r.sum() - 1.0).abs() < 1e-9));
            let acc = anofox_ml_core::ClassifierScore::score(&f, &x, &yc).unwrap();
            assert!(acc > 0.8, "accuracy = {acc}");
        }
    }

    #[test]
    fn depth_only_trees_and_validation() {
        let (x, y) = data();
        let f = HistGradientBoostingRegressor::new()
            .with_max_leaf_nodes(None)
            .with_max_depth(3)
            .fit(&x, &y)
            .unwrap();
        assert!(f.n_estimators() > 0);
        assert!(HistGradientBoostingRegressor::new()
            .with_max_leaf_nodes(None)
            .fit(&x, &y)
            .is_err());
        assert!(HistGradientBoostingClassifier::new()
            .fit(&array![[1.0], [2.0]], &array![0.0, 0.0])
            .is_err());
    }

    #[test]
    fn sample_weights_change_the_fit() {
        let (x, y) = data();
        let w = Array1::from_shape_fn(y.len(), |i| if i < 100 { 5.0 } else { 0.2 });
        let hgb = HistGradientBoostingRegressor::new().with_n_estimators(10);
        let a = hgb.fit(&x, &y).unwrap().predict(&x).unwrap();
        let b = hgb
            .fit_weighted(&x, &y, Some(&w))
            .unwrap()
            .predict(&x)
            .unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn regressor_matches_sklearn_binning() {
        // sklearn 1.8 HistGradientBoostingRegressor(max_iter=20,
        // min_samples_leaf=5[, max_bins=16]).fit(X, y).predict(X[rows]).
        // max_bins=16 takes the percentile path of sklearn's binning. The
        // facade rounds gradients to float32 as sklearn does, so the
        // predictions agree to rounding.
        let x = Array2::from_shape_fn((120, 3), |(i, j)| {
            let t = i as f64;
            match j {
                0 => (t * 0.37).sin() * 3.0 + t * 0.01,
                1 => ((t * 0.11).cos() * 4.0).round(),
                _ => (t * 0.7) % 5.3,
            }
        });
        let y = Array1::from_shape_fn(120, |i| {
            x[[i, 0]].powi(2) - 2.0 * x[[i, 1]] + x[[i, 2]].sin()
        });
        let rows = [0, 7, 33, 90, 119];
        let cases: [(usize, [f64; 5]); 2] = [
            (
                255,
                [
                    -5.777_367_549_331_124_5,
                    -3.408_584_907_578_834_6,
                    8.455_614_568_356_298,
                    17.890_444_108_490_723,
                    -3.509_011_691_821_183,
                ],
            ),
            (
                16,
                [
                    -5.885_602_490_798_531,
                    -3.395_267_404_999_707,
                    8.690_401_533_423_369,
                    17.968_895_264_862_134,
                    -3.352_575_566_258_404_5,
                ],
            ),
        ];
        for (max_bins, expected) in cases {
            let m = HistGradientBoostingRegressor::new()
                .with_n_estimators(20)
                .with_min_samples_leaf(5)
                .with_max_bins(max_bins)
                .fit(&x, &y)
                .unwrap();
            let p = m.predict(&x.select(ndarray::Axis(0), &rows)).unwrap();
            for (a, b) in p.iter().zip(expected) {
                assert!((a - b).abs() < 1e-12, "max_bins={max_bins}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn sklearn_percentile_thresholds_land_on_exact_indices() {
        // np.percentile(np.sqrt(np.arange(375.)), np.linspace(0, 100, 256)[1:-1],
        // method="midpoint"): at k = 15 the index 374 * 15 / 255 = 22 is
        // exact in numpy's arithmetic, so the threshold is a data value.
        let v: Vec<f64> = (0..375).map(|i| f64::from(i).sqrt()).collect();
        let t = crate::lgbm::sklearn_thresholds_for_test(&v, 255);
        assert_eq!(t.len(), 254);
        assert_eq!(
            &t[13..16],
            &[4.527355824977709, 4.636495727389635, 4.847405504439537]
        );
        assert!((t.iter().sum::<f64>() - 3277.3897249418715).abs() < 1e-9);
    }

    #[test]
    fn sklearn_thresholds_match_numpy() {
        // <= max_bins distinct values: midpoints.
        assert_eq!(
            crate::lgbm::sklearn_thresholds_for_test(&[1.0, 1.0, 2.0, 4.0], 4),
            vec![1.5, 3.0]
        );
        // np.percentile(arange(10), [25, 50, 75], method="midpoint")
        //   -> [2.5, 4.5, 6.5]
        let v: Vec<f64> = (0..10).map(f64::from).collect();
        assert_eq!(
            crate::lgbm::sklearn_thresholds_for_test(&v, 4),
            vec![2.5, 4.5, 6.5]
        );
    }
}
