//! The estimator registry behind `_native.Model` (see [`crate::model`]).
//!
//! Kind names are the Rust type names; the Python package maps sklearn
//! class names and parameters onto them.

use anofox_ml_core::{
    DecisionFunction, Fit, FitUnsupervised, FitUnsupervisedWeighted, FitWeighted, InverseTransform,
    Predict, PredictProba, Result, Transform,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

use anofox_ml_cluster as cluster;
use anofox_ml_discriminant as discriminant;
use anofox_ml_ensemble as ensemble;
use anofox_ml_linear as linear;
use anofox_ml_manifold as manifold;
use anofox_ml_naive_bayes as nb;
use anofox_ml_neighbors as neighbors;
use anofox_ml_neural_networks as nn;
use anofox_ml_preprocessing as pre;
use anofox_ml_regression as reg;
use anofox_ml_svm as svm;
use anofox_ml_trees as trees;

use anofox_ml_conformal as conformal;

use crate::estimator::{RegistryEstimator, RegistryFitted};
use crate::model::{need_y, no_weight, registry};

/// Compact fitted form of a linear model whose full fitted type holds
/// non-serializable solver state: enough to predict and to pickle.
/// Aliased (`NaN`) coefficients count as 0, as in the Rust `predict`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct LinearFit {
    pub coef: Array1<f64>,
    pub intercept: f64,
}

impl LinearFit {
    fn new(coef: Array1<f64>, intercept: Option<f64>) -> Self {
        Self {
            coef: coef.mapv(|c| if c.is_nan() { 0.0 } else { c }),
            intercept: intercept.unwrap_or(0.0),
        }
    }

    fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        if x.ncols() != self.coef.len() {
            return Err(anofox_ml_core::RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.coef.len(),
                x.ncols()
            )));
        }
        Ok(x.dot(&self.coef) + self.intercept)
    }
}

/// Supervised fit, no weights.
fn sup<C: Fit<f64>>(
    k: &str,
    c: &C,
    x: ArrayView2<f64>,
    y: Option<ArrayView1<f64>>,
    w: Option<ArrayView1<f64>>,
) -> Result<C::Fitted> {
    no_weight(k, w)?;
    c.fit(x, need_y(k, y)?)
}

/// Supervised fit with optional weights.
fn wsup<C: FitWeighted<f64>>(
    k: &str,
    c: &C,
    x: ArrayView2<f64>,
    y: Option<ArrayView1<f64>>,
    w: Option<ArrayView1<f64>>,
) -> Result<C::Fitted> {
    c.fit_weighted_view(x, need_y(k, y)?, w)
}

/// Fitted TargetEncoder plus the cross-fitted encoding of its training data
/// (what `fit_transform` returns; not pickled).
#[derive(Serialize, Deserialize)]
struct TargetEncoderFit {
    model: pre::FittedTargetEncoder<f64>,
    #[serde(skip)]
    train: Array2<f64>,
}

/// Unsupervised fit, no weights (`y` is ignored, as in sklearn).
fn unsup<C: FitUnsupervised<f64>>(
    k: &str,
    c: &C,
    x: ArrayView2<f64>,
    w: Option<ArrayView1<f64>>,
) -> Result<C::Fitted> {
    no_weight(k, w)?;
    FitUnsupervised::fit(c, x)
}

// ------------------------------------------------- conformal configs
//
// The conformal predictors wrap any registered estimator, given by its kind
// (`estimator`) and JSON parameters (`estimator_params`). The defaults are
// MAPIE's: a linear regression or a logistic regression.

fn default_regressor() -> String {
    "OlsRegressor".into()
}

fn empty_params() -> serde_json::Value {
    serde_json::json!({})
}

/// `SplitConformalRegressor` over a registered estimator.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct SplitConformalRegressorConfig {
    estimator: String,
    estimator_params: serde_json::Value,
    calibration_size: f64,
    seed: u64,
    allow_infinite_bounds: bool,
}

impl Default for SplitConformalRegressorConfig {
    fn default() -> Self {
        let d = conformal::SplitConformalRegressor::new(());
        Self {
            estimator: default_regressor(),
            estimator_params: empty_params(),
            calibration_size: d.calibration_size,
            seed: d.seed,
            allow_infinite_bounds: d.allow_infinite_bounds,
        }
    }
}

impl SplitConformalRegressorConfig {
    fn build(&self) -> conformal::SplitConformalRegressor<RegistryEstimator> {
        conformal::SplitConformalRegressor::new(RegistryEstimator::new(
            &self.estimator,
            self.estimator_params.clone(),
        ))
        .with_calibration_size(self.calibration_size)
        .with_seed(self.seed)
        .with_allow_infinite_bounds(self.allow_infinite_bounds)
    }
}

/// `CrossConformalRegressor` (CV+, jackknife+ with `cv` = n, minmax, base)
/// over a registered estimator.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct CrossConformalRegressorConfig {
    estimator: String,
    estimator_params: serde_json::Value,
    cv: usize,
    method: conformal::CrossMethod,
    point_prediction: conformal::PointPrediction,
    seed: u64,
    allow_infinite_bounds: bool,
}

impl Default for CrossConformalRegressorConfig {
    fn default() -> Self {
        let d = conformal::CrossConformalRegressor::new(());
        Self {
            estimator: default_regressor(),
            estimator_params: empty_params(),
            cv: d.cv,
            method: d.method,
            point_prediction: d.point_prediction,
            seed: d.seed,
            allow_infinite_bounds: d.allow_infinite_bounds,
        }
    }
}

impl CrossConformalRegressorConfig {
    fn build(&self) -> conformal::CrossConformalRegressor<RegistryEstimator> {
        conformal::CrossConformalRegressor::new(RegistryEstimator::new(
            &self.estimator,
            self.estimator_params.clone(),
        ))
        .with_cv(self.cv)
        .with_method(self.method)
        .with_point_prediction(self.point_prediction)
        .with_seed(self.seed)
        .with_allow_infinite_bounds(self.allow_infinite_bounds)
    }
}

/// `NormalizedConformalRegressor` (residual-normalised scores) over two
/// registered estimators: the point model and the model of `|residual|`.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct NormalizedConformalRegressorConfig {
    estimator: String,
    estimator_params: serde_json::Value,
    residual_estimator: String,
    residual_estimator_params: serde_json::Value,
    calibration_size: f64,
    residual_size: f64,
    seed: u64,
    allow_infinite_bounds: bool,
}

impl Default for NormalizedConformalRegressorConfig {
    fn default() -> Self {
        let d = conformal::NormalizedConformalRegressor::new((), ());
        Self {
            estimator: default_regressor(),
            estimator_params: empty_params(),
            residual_estimator: default_regressor(),
            residual_estimator_params: empty_params(),
            calibration_size: d.calibration_size,
            residual_size: d.residual_size,
            seed: d.seed,
            allow_infinite_bounds: d.allow_infinite_bounds,
        }
    }
}

impl NormalizedConformalRegressorConfig {
    fn build(
        &self,
    ) -> conformal::NormalizedConformalRegressor<RegistryEstimator, RegistryEstimator> {
        conformal::NormalizedConformalRegressor::new(
            RegistryEstimator::new(&self.estimator, self.estimator_params.clone()),
            RegistryEstimator::new(
                &self.residual_estimator,
                self.residual_estimator_params.clone(),
            ),
        )
        .with_calibration_size(self.calibration_size)
        .with_residual_size(self.residual_size)
        .with_seed(self.seed)
        .with_allow_infinite_bounds(self.allow_infinite_bounds)
    }
}

/// `SplitConformalClassifier` (LAC, APS, RAPS) over a registered
/// classifier with `predict_proba`.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct SplitConformalClassifierConfig {
    estimator: String,
    estimator_params: serde_json::Value,
    score: conformal::ClassificationScore,
    calibration_size: f64,
    seed: u64,
    allow_full_sets: bool,
}

impl Default for SplitConformalClassifierConfig {
    fn default() -> Self {
        let d = conformal::SplitConformalClassifier::new(());
        Self {
            estimator: "LogisticRegressor".into(),
            estimator_params: empty_params(),
            score: d.score,
            calibration_size: d.calibration_size,
            seed: d.seed,
            allow_full_sets: d.allow_full_sets,
        }
    }
}

impl SplitConformalClassifierConfig {
    fn build(&self) -> conformal::SplitConformalClassifier<RegistryEstimator> {
        conformal::SplitConformalClassifier::new(RegistryEstimator::new(
            &self.estimator,
            self.estimator_params.clone(),
        ))
        .with_score(self.score)
        .with_calibration_size(self.calibration_size)
        .with_seed(self.seed)
        .with_allow_full_sets(self.allow_full_sets)
    }
}

/// `(lower, upper)` as an `n × 2` array.
fn interval((lo, hi): (Array1<f64>, Array1<f64>)) -> Array2<f64> {
    ndarray::stack(ndarray::Axis(1), &[lo.view(), hi.view()]).expect("bounds of equal length")
}

/// A membership matrix as 0 / 1.
fn membership(sets: Array2<bool>) -> Array2<f64> {
    sets.mapv(|b| if b { 1.0 } else { 0.0 })
}

registry! {
    // ----------------------------------------------------------- trees
    DecisionTreeClassifier {
        config: trees::DecisionTreeClassifier = Default::default(),
        fitted: trees::FittedDecisionTreeClassifier<f64>,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| PredictProba::predict_proba(m, x),
    }
    DecisionTreeRegressor {
        config: trees::DecisionTreeRegressor = Default::default(),
        fitted: trees::FittedDecisionTreeRegressor<f64>,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
    }

    // -------------------------------------------------------- ensemble
    RandomForestClassifier {
        config: ensemble::RandomForestClassifier = Default::default(),
        fitted: ensemble::FittedRandomForestClassifier<f64>,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| PredictProba::predict_proba(m, x),
    }
    RandomForestRegressor {
        config: ensemble::RandomForestRegressor = Default::default(),
        fitted: ensemble::FittedRandomForestRegressor<f64>,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
    }
    ExtraTreesClassifier {
        config: ensemble::ExtraTreesClassifier = Default::default(),
        fitted: ensemble::FittedExtraTreesClassifier<f64>,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
    }
    ExtraTreesRegressor {
        config: ensemble::ExtraTreesRegressor = Default::default(),
        fitted: ensemble::FittedExtraTreesRegressor<f64>,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
    }
    GradientBoostingClassifier {
        config: ensemble::GradientBoostingClassifier = Default::default(),
        fitted: ensemble::FittedGradientBoostingClassifier<f64>,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| PredictProba::predict_proba(m, x),
        decision_function: |m, x| m.decision_function(x),
    }
    GradientBoostingRegressor {
        config: ensemble::GradientBoostingRegressor = Default::default(),
        fitted: ensemble::FittedGradientBoostingRegressor<f64>,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
    }
    AdaBoostClassifier {
        config: ensemble::AdaBoostClassifier = Default::default(),
        fitted: ensemble::FittedAdaBoostClassifier<f64>,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| PredictProba::predict_proba(m, x),
        decision_function: |m, x| m.decision_function(x),
    }
    AdaBoostRegressor {
        config: ensemble::AdaBoostRegressor = Default::default(),
        fitted: ensemble::FittedAdaBoostRegressor<f64>,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
    }
    BaggingClassifier {
        config: ensemble::BaggingClassifier = Default::default(),
        fitted: ensemble::FittedBaggingClassifier<f64>,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| PredictProba::predict_proba(m, x),
    }
    BaggingRegressor {
        config: ensemble::BaggingRegressor = Default::default(),
        fitted: ensemble::FittedBaggingRegressor<f64>,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
    }
    HistGradientBoostingClassifier {
        config: ensemble::HistGradientBoostingClassifier = Default::default(),
        fitted: ensemble::FittedHistGradientBoostingClassifier,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| m.predict_proba(x),
    }
    HistGradientBoostingRegressor {
        config: ensemble::HistGradientBoostingRegressor = Default::default(),
        fitted: ensemble::FittedHistGradientBoostingRegressor,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
    }
    LgbmClassifier {
        config: ensemble::LgbmClassifier = Default::default(),
        fitted: ensemble::FittedLgbmClassifier,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| m.predict_proba(x),
    }
    LgbmRegressor {
        config: ensemble::LgbmRegressor = Default::default(),
        fitted: ensemble::FittedLgbmRegressor,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
    }
    #[cfg(feature = "perpetual")]
    PerpetualRegressor {
        config: ensemble::PerpetualRegressor = Default::default(),
        fitted: ensemble::FittedPerpetualRegressor,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
    }
    #[cfg(feature = "perpetual")]
    PerpetualClassifier {
        config: ensemble::PerpetualClassifier = Default::default(),
        fitted: ensemble::FittedPerpetualClassifier,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| PredictProba::predict_proba(m, x),
    }
    IsolationForest {
        config: ensemble::IsolationForest = Default::default(),
        fitted: ensemble::FittedIsolationForest,
        fit: |c, x, y, w| unsup(K, c, x, w),
        predict: |m, x| m.predict(x),
        score_samples: |m, x| Ok::<_, anofox_ml_core::RustMlError>(m.score_samples(x)),
        decision_function: |m, x| Ok::<_, anofox_ml_core::RustMlError>(m.decision_function(x)),
    }

    // ------------------------------------------------------- conformal
    SplitConformalRegressor {
        config: SplitConformalRegressorConfig = Default::default(),
        fitted: conformal::FittedSplitConformalRegressor<RegistryFitted, f64>,
        fit: |c, x, y, w| {
            no_weight(K, w)?;
            c.build().fit_view(x, need_y(K, y)?)
        },
        predict: |m, x| m.predict(x),
        predict_interval: |m, x, alpha| m.predict_interval(x, alpha).map(interval),
    }
    CrossConformalRegressor {
        config: CrossConformalRegressorConfig = Default::default(),
        fitted: conformal::FittedCrossConformalRegressor<RegistryFitted, f64>,
        fit: |c, x, y, w| {
            no_weight(K, w)?;
            c.build().fit_view(x, need_y(K, y)?)
        },
        predict: |m, x| m.predict(x),
        predict_interval: |m, x, alpha| m.predict_interval(x, alpha).map(interval),
    }
    NormalizedConformalRegressor {
        config: NormalizedConformalRegressorConfig = Default::default(),
        fitted: conformal::FittedNormalizedConformalRegressor<RegistryFitted, RegistryFitted, f64>,
        fit: |c, x, y, w| {
            no_weight(K, w)?;
            c.build().fit_view(x, need_y(K, y)?)
        },
        predict: |m, x| m.predict(x),
        predict_interval: |m, x, alpha| m.predict_interval(x, alpha).map(interval),
    }
    SplitConformalClassifier {
        config: SplitConformalClassifierConfig = Default::default(),
        fitted: conformal::FittedSplitConformalClassifier<RegistryFitted, f64>,
        fit: |c, x, y, w| {
            no_weight(K, w)?;
            c.build().fit_view(x, need_y(K, y)?)
        },
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| PredictProba::predict_proba(m.model(), x),
        predict_set: |m, x, alpha| m.predict_set(x, alpha).map(membership),
    }

    // ------------------------------------------------------------- svm
    Svc {
        config: svm::Svc = Default::default(),
        fitted: svm::FittedSvc<f64>,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| PredictProba::predict_proba(m, x),
        decision_function: |m, x| DecisionFunction::decision_function(m, x),
    }
    NuSvc {
        config: svm::NuSvc = Default::default(),
        fitted: svm::FittedNuSvc<f64>,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| PredictProba::predict_proba(m, x),
        decision_function: |m, x| DecisionFunction::decision_function(m, x),
    }
    Svr {
        config: svm::Svr = Default::default(),
        fitted: svm::FittedSvr<f64>,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
    }
    NuSvr {
        config: svm::NuSvr = Default::default(),
        fitted: svm::FittedNuSvr<f64>,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
    }
    LinearSvc {
        config: svm::LinearSvc = Default::default(),
        fitted: svm::FittedLinearSvc<f64>,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        decision_function: |m, x| m.decision_function(x),
    }
    LinearSvr {
        config: svm::LinearSvr = Default::default(),
        fitted: svm::FittedLinearSvr<f64>,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
    }
    OneClassSvm {
        config: svm::OneClassSvm = Default::default(),
        fitted: svm::FittedOneClassSvm<f64>,
        fit: |c, x, y, w| unsup(K, c, x, w),
        predict: |m, x| m.predict(x),
        decision_function: |m, x| m.decision_function(x),
        score_samples: |m, x| m.score_samples(x),
    }

    // ---------------------------------------------------------- linear
    SgdClassifier {
        config: linear::SgdClassifier = Default::default(),
        fitted: linear::FittedSgdClassifier<f64>,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| PredictProba::predict_proba(m, x),
        decision_function: |m, x| m.decision_function(x),
    }
    SgdRegressor {
        config: linear::SgdRegressor = Default::default(),
        fitted: linear::FittedSgdRegressor<f64>,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
    }
    PassiveAggressiveClassifier {
        config: linear::PassiveAggressiveClassifier = Default::default(),
        fitted: linear::FittedPassiveAggressiveClassifier,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        decision_function: |m, x| m.decision_function(x),
    }
    PassiveAggressiveRegressor {
        config: linear::PassiveAggressiveRegressor = Default::default(),
        fitted: linear::FittedPassiveAggressiveRegressor,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
    }

    // ------------------------------------------------------ regression
    OlsRegressor {
        config: reg::OlsRegressor = Default::default(),
        fitted: LinearFit,
        fit: |c, x, y, w| wsup(K, c, x, y, w)
            .map(|f| LinearFit::new(f.coefficients(), f.intercept())),
        predict: |m, x| m.predict(x),
    }
    RidgeRegressor {
        config: reg::RidgeRegressor = Default::default(),
        fitted: LinearFit,
        fit: |c, x, y, w| wsup(K, c, x, y, w)
            .map(|f| LinearFit::new(f.coefficients(), f.intercept())),
        predict: |m, x| m.predict(x),
    }
    LassoRegressor {
        config: reg::LassoRegressor = Default::default(),
        fitted: LinearFit,
        fit: |c, x, y, w| sup(K, c, x, y, w)
            .map(|f| LinearFit::new(f.coefficients(), f.intercept())),
        predict: |m, x| m.predict(x),
    }
    ElasticNetRegressor {
        config: reg::ElasticNetRegressor = Default::default(),
        fitted: LinearFit,
        fit: |c, x, y, w| sup(K, c, x, y, w)
            .map(|f| LinearFit::new(f.coefficients(), f.intercept())),
        predict: |m, x| m.predict(x),
    }
    HuberRegressor {
        config: reg::HuberRegressor = Default::default(),
        fitted: LinearFit,
        fit: |c, x, y, w| sup(K, c, x, y, w)
            .map(|f| LinearFit::new(f.coefficients(), f.intercept())),
        predict: |m, x| m.predict(x),
    }
    QuantileRegressor {
        config: reg::QuantileRegressor = reg::QuantileRegressor::new(0.5),
        fitted: LinearFit,
        fit: |c, x, y, w| wsup(K, c, x, y, w)
            .map(|f| LinearFit::new(f.coefficients(), f.intercept())),
        predict: |m, x| m.predict(x),
    }
    RidgeCrossValidated {
        config: reg::RidgeCrossValidated = Default::default(),
        fitted: LinearFit,
        fit: |c, x, y, w| sup(K, c, x, y, w)
            .map(|f| LinearFit::new(f.coefficients(), f.intercept())),
        predict: |m, x| m.predict(x),
    }
    LassoCrossValidated {
        config: reg::LassoCrossValidated = Default::default(),
        fitted: LinearFit,
        fit: |c, x, y, w| sup(K, c, x, y, w)
            .map(|f| LinearFit::new(f.coefficients(), f.intercept())),
        predict: |m, x| m.predict(x),
    }
    ElasticNetCrossValidated {
        config: reg::ElasticNetCrossValidated = Default::default(),
        fitted: LinearFit,
        fit: |c, x, y, w| sup(K, c, x, y, w)
            .map(|f| LinearFit::new(f.coefficients(), f.intercept())),
        predict: |m, x| m.predict(x),
    }
    LogisticRegressor {
        config: reg::LogisticRegressor = Default::default(),
        fitted: reg::FittedLogisticRegressor,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| PredictProba::predict_proba(m, x),
        decision_function: |m, x| DecisionFunction::decision_function(m, x),
    }
    PoissonRegressor {
        config: reg::PoissonRegressor = Default::default(),
        fitted: reg::FittedPoissonRegressor,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
    }
    GammaRegressor {
        config: reg::GammaRegressor = Default::default(),
        fitted: reg::FittedTweedieRegressor,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
    }
    TweedieRegressor {
        config: reg::TweedieRegressor = reg::TweedieRegressor::new(0.0),
        fitted: reg::FittedTweedieRegressor,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
    }
    BayesianRidge {
        config: reg::BayesianRidge = Default::default(),
        fitted: reg::FittedBayesianRidge,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
    }
    ARDRegression {
        config: reg::ARDRegression = Default::default(),
        fitted: reg::FittedARDRegression,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
    }
    KernelRidge {
        config: reg::KernelRidge = Default::default(),
        fitted: reg::FittedKernelRidge,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
    }
    Lars {
        config: reg::Lars = reg::Lars::new(500),
        fitted: reg::FittedLars,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
    }
    OrthogonalMatchingPursuit {
        config: reg::OrthogonalMatchingPursuit = Default::default(),
        fitted: reg::FittedOrthogonalMatchingPursuit,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
    }
    RansacRegressor {
        config: reg::RansacRegressor = Default::default(),
        fitted: reg::FittedRansacRegressor,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
    }
    TheilSenRegressor {
        config: reg::TheilSenRegressor = Default::default(),
        fitted: reg::FittedTheilSenRegressor,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
    }

    // ----------------------------------------------------- naive bayes
    GaussianNB {
        config: nb::GaussianNB = Default::default(),
        fitted: nb::FittedGaussianNB<f64>,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| PredictProba::predict_proba(m, x),
    }
    MultinomialNB {
        config: nb::MultinomialNB = Default::default(),
        fitted: nb::FittedMultinomialNB<f64>,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| PredictProba::predict_proba(m, x),
    }
    BernoulliNB {
        config: nb::BernoulliNB = Default::default(),
        fitted: nb::FittedBernoulliNB<f64>,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| PredictProba::predict_proba(m, x),
    }
    ComplementNB {
        config: nb::ComplementNB = Default::default(),
        fitted: nb::FittedComplementNB<f64>,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| PredictProba::predict_proba(m, x),
    }
    CategoricalNB {
        config: nb::CategoricalNB = Default::default(),
        fitted: nb::FittedCategoricalNB<f64>,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| PredictProba::predict_proba(m, x),
    }

    // ------------------------------------------------------- neighbors
    KnnClassifier {
        config: neighbors::KnnClassifier = Default::default(),
        fitted: neighbors::FittedKnnClassifier<f64>,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| PredictProba::predict_proba(m, x),
    }
    KnnRegressor {
        config: neighbors::KnnRegressor = Default::default(),
        fitted: neighbors::FittedKnnRegressor<f64>,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
    }
    RadiusNeighborsClassifier {
        config: neighbors::RadiusNeighborsClassifier = Default::default(),
        fitted: neighbors::FittedRadiusNeighborsClassifier<f64>,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| PredictProba::predict_proba(m, x),
    }
    RadiusNeighborsRegressor {
        config: neighbors::RadiusNeighborsRegressor = Default::default(),
        fitted: neighbors::FittedRadiusNeighborsRegressor<f64>,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
    }
    LocalOutlierFactor {
        config: neighbors::LocalOutlierFactor = neighbors::LocalOutlierFactor::new(20),
        fitted: neighbors::FittedLocalOutlierFactor,
        fit: |c, x, y, w| unsup(K, c, x, w),
        labels: |m| m.predictions.clone(),
        score_samples: |m, _x| Ok::<_, anofox_ml_core::RustMlError>(m.negative_outlier_factor.clone()),
    }

    // ---------------------------------------------------- discriminant
    LinearDiscriminantAnalysis {
        config: discriminant::LinearDiscriminantAnalysis = Default::default(),
        fitted: discriminant::FittedLinearDiscriminantAnalysis,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| PredictProba::predict_proba(m, x),
        decision_function: |m, x| DecisionFunction::decision_function(m, x),
        transform: |m, x| Transform::transform(m, x),
    }
    QuadraticDiscriminantAnalysis {
        config: discriminant::QuadraticDiscriminantAnalysis = Default::default(),
        fitted: discriminant::FittedQuadraticDiscriminantAnalysis,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| PredictProba::predict_proba(m, x),
        decision_function: |m, x| DecisionFunction::decision_function(m, x),
    }

    // -------------------------------------------------- neural network
    MlpClassifier {
        config: nn::MlpClassifier = Default::default(),
        fitted: nn::FittedMlpClassifier<f64>,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| m.predict_proba(x),
    }
    MlpRegressor {
        config: nn::MlpRegressor = Default::default(),
        fitted: nn::FittedMlpRegressor<f64>,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
    }

    // --------------------------------------------------- preprocessing
    StandardScaler {
        config: pre::StandardScaler = Default::default(),
        fitted: pre::FittedStandardScaler<f64>,
        fit: |c, x, y, w| FitUnsupervisedWeighted::fit_unsupervised_weighted_view(c, x, w),
        transform: |m, x| Transform::transform(m, x),
        inverse_transform: |m, x| InverseTransform::inverse_transform(m, x),
    }
    MinMaxScaler {
        config: pre::MinMaxScaler<f64> = Default::default(),
        fitted: pre::FittedMinMaxScaler<f64>,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
        inverse_transform: |m, x| InverseTransform::inverse_transform(m, x),
    }
    MaxAbsScaler {
        config: pre::MaxAbsScaler = Default::default(),
        fitted: pre::FittedMaxAbsScaler<f64>,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
        inverse_transform: |m, x| InverseTransform::inverse_transform(m, x),
    }
    RobustScaler {
        config: pre::RobustScaler = Default::default(),
        fitted: pre::FittedRobustScaler<f64>,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
        inverse_transform: |m, x| InverseTransform::inverse_transform(m, x),
    }
    Normalizer {
        config: pre::Normalizer = Default::default(),
        fitted: pre::FittedNormalizer<f64>,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
    }
    Binarizer {
        config: pre::Binarizer<f64> = Default::default(),
        fitted: pre::FittedBinarizer<f64>,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
    }
    SimpleImputer {
        config: pre::SimpleImputer<f64> = Default::default(),
        fitted: pre::FittedSimpleImputer<f64>,
        fit: |c, x, y, w| FitUnsupervisedWeighted::fit_unsupervised_weighted_view(c, x, w),
        transform: |m, x| Transform::transform(m, x),
    }
    MissingIndicator {
        config: pre::MissingIndicator = Default::default(),
        fitted: pre::FittedMissingIndicator,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
    }
    KnnImputer {
        config: pre::KnnImputer = Default::default(),
        fitted: pre::FittedKnnImputer<f64>,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
    }
    IterativeImputer {
        config: pre::IterativeImputer = Default::default(),
        fitted: pre::FittedIterativeImputer<reg::FittedBayesianRidge>,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
    }
    TargetEncoder {
        config: pre::TargetEncoder = Default::default(),
        fitted: TargetEncoderFit,
        fit: |c, x, y, w| {
            no_weight(K, w)?;
            c.fit_transform(x, need_y(K, y)?)
                .map(|(model, train)| TargetEncoderFit { model, train })
        },
        transform: |m, x| Transform::transform(&m.model, x),
        embedding: |m| m.train.clone(),
    }
    RcsTransformer {
        config: pre::RcsTransformer = Default::default(),
        fitted: pre::FittedRcsTransformer<f64>,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
    }
    SplineTransformer {
        config: pre::SplineTransformer = Default::default(),
        fitted: pre::FittedSplineTransformer<f64>,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
    }
    PolynomialFeatures {
        config: pre::PolynomialFeatures = Default::default(),
        fitted: pre::FittedPolynomialFeatures<f64>,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
    }
    PowerTransformer {
        config: pre::PowerTransformer = Default::default(),
        fitted: pre::FittedPowerTransformer<f64>,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
        inverse_transform: |m, x| InverseTransform::inverse_transform(m, x),
    }
    QuantileTransformer {
        config: pre::QuantileTransformer = Default::default(),
        fitted: pre::FittedQuantileTransformer<f64>,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
        inverse_transform: |m, x| InverseTransform::inverse_transform(m, x),
    }
    KBinsDiscretizer {
        config: pre::KBinsDiscretizer = Default::default(),
        fitted: pre::FittedKBinsDiscretizer<f64>,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
        inverse_transform: |m, x| InverseTransform::inverse_transform(m, x),
    }
    VarianceThreshold {
        config: pre::VarianceThreshold = Default::default(),
        fitted: pre::FittedVarianceThreshold<f64>,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
    }
    Pca {
        config: pre::Pca = pre::Pca::new(2),
        fitted: pre::FittedPca<f64>,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
        inverse_transform: |m, x| InverseTransform::inverse_transform(m, x),
    }
    IncrementalPca {
        config: pre::IncrementalPca = Default::default(),
        fitted: pre::FittedIncrementalPca<f64>,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
        inverse_transform: |m, x| InverseTransform::inverse_transform(m, x),
    }
    KernelPca {
        config: pre::KernelPca = pre::KernelPca::new(2, pre::KpcaKernel::Linear),
        fitted: pre::FittedKernelPca,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
        inverse_transform: |m, x| InverseTransform::inverse_transform(m, x),
    }
    TruncatedSvd {
        config: pre::TruncatedSvd = pre::TruncatedSvd::new(2),
        fitted: pre::FittedTruncatedSvd,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
        inverse_transform: |m, x| InverseTransform::inverse_transform(m, x),
    }
    FastIca {
        config: pre::FastIca = pre::FastIca::new(2),
        fitted: pre::FittedFastIca,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
        inverse_transform: |m, x| InverseTransform::inverse_transform(m, x),
    }
    Nmf {
        config: pre::Nmf = pre::Nmf::new(2),
        fitted: pre::FittedNmf,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
    }
    SelectKBest {
        config: pre::SelectKBest = pre::SelectKBest::new(10, pre::select_k_best::ScoringFunction::FClassif),
        fitted: pre::FittedSelectKBest<f64>,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        transform: |m, x| Transform::transform(m, x),
    }
    MutualInformationSelector {
        config: pre::MutualInformationSelector = pre::MutualInformationSelector::new(10),
        fitted: pre::FittedMutualInformationSelector<f64>,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        transform: |m, x| Transform::transform(m, x),
    }

    // --------------------------------------------------------- cluster
    KMeans {
        config: cluster::KMeans = Default::default(),
        fitted: cluster::FittedKMeans<f64>,
        fit: |c, x, y, w| FitUnsupervisedWeighted::fit_unsupervised_weighted_view(c, x, w),
        predict: |m, x| m.predict(x),
        labels: |m| m.labels().clone(),
    }
    MiniBatchKMeans {
        config: cluster::MiniBatchKMeans = Default::default(),
        fitted: cluster::FittedMiniBatchKMeans<f64>,
        fit: |c, x, y, w| FitUnsupervisedWeighted::fit_unsupervised_weighted_view(c, x, w),
        predict: |m, x| m.predict(x),
        labels: |m| m.labels().clone(),
    }
    Dbscan {
        config: cluster::Dbscan = Default::default(),
        fitted: cluster::FittedDbscan<f64>,
        fit: |c, x, y, w| unsup(K, c, x, w),
        predict: |m, x| m.predict(x),
        labels: |m| m.labels().clone(),
    }
    Hdbscan {
        config: cluster::Hdbscan = Default::default(),
        fitted: cluster::FittedHdbscan,
        fit: |c, x, y, w| unsup(K, c, x, w),
        labels: |m| m.labels.clone(),
    }
    Optics {
        config: cluster::Optics = cluster::Optics::new(5, f64::INFINITY),
        fitted: cluster::FittedOptics,
        fit: |c, x, y, w| unsup(K, c, x, w),
        labels: |m| m.labels.clone(),
    }
    MeanShift {
        config: cluster::MeanShift = cluster::MeanShift::new(1.0),
        fitted: cluster::FittedMeanShift,
        fit: |c, x, y, w| unsup(K, c, x, w),
        predict: |m, x| m.predict(x),
        labels: |m| m.labels.clone(),
    }
    Birch {
        config: cluster::Birch = cluster::Birch::new(0.5),
        fitted: cluster::FittedBirch,
        fit: |c, x, y, w| unsup(K, c, x, w),
        predict: |m, x| m.predict(x),
        labels: |m| m.labels.clone(),
    }
    AgglomerativeClustering {
        config: cluster::AgglomerativeClustering = cluster::AgglomerativeClustering::new(2),
        fitted: cluster::FittedAgglomerativeClustering,
        fit: |c, x, y, w| unsup(K, c, x, w),
        labels: |m| m.labels.clone(),
    }
    SpectralClustering {
        config: cluster::SpectralClustering = cluster::SpectralClustering::new(8),
        fitted: cluster::FittedSpectralClustering,
        fit: |c, x, y, w| unsup(K, c, x, w),
        labels: |m| m.labels.clone(),
    }
    AffinityPropagation {
        config: cluster::AffinityPropagation = Default::default(),
        fitted: cluster::FittedAffinityPropagation,
        fit: |c, x, y, w| unsup(K, c, x, w),
        labels: |m| m.labels.clone(),
    }
    GaussianMixture {
        config: cluster::GaussianMixture = cluster::GaussianMixture::new(1),
        fitted: cluster::FittedGaussianMixture,
        fit: |c, x, y, w| unsup(K, c, x, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| PredictProba::predict_proba(m, x),
        score_samples: |m, x| m.score_samples(x),
    }
    BayesianGaussianMixture {
        config: cluster::BayesianGaussianMixture = cluster::BayesianGaussianMixture::new(1),
        fitted: cluster::FittedBayesianGaussianMixture,
        fit: |c, x, y, w| unsup(K, c, x, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| PredictProba::predict_proba(m, x),
    }

    // -------------------------------------------------------- manifold
    TSne {
        config: manifold::TSne = manifold::TSne::new(2),
        fitted: manifold::FittedTSne,
        fit: |c, x, y, w| unsup(K, c, x, w),
        embedding: |m| m.embedding.clone(),
    }
    Isomap {
        config: manifold::Isomap = manifold::Isomap::new(2, 5),
        fitted: manifold::FittedIsomap,
        fit: |c, x, y, w| unsup(K, c, x, w),
        embedding: |m| m.embedding.clone(),
    }
    LocallyLinearEmbedding {
        config: manifold::LocallyLinearEmbedding = manifold::LocallyLinearEmbedding::new(2, 5),
        fitted: manifold::FittedLocallyLinearEmbedding,
        fit: |c, x, y, w| unsup(K, c, x, w),
        embedding: |m| m.embedding.clone(),
    }
    ClassicalMds {
        config: manifold::ClassicalMds = manifold::ClassicalMds::new(2),
        fitted: manifold::FittedClassicalMds,
        fit: |c, x, y, w| unsup(K, c, x, w),
        embedding: |m| m.embedding.clone(),
    }
}
