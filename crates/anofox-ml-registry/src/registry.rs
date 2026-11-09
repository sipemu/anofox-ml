//! The estimator registry behind `_native.Model` (see [`crate::model`]).
//!
//! Kind names are the Rust type names; the Python package maps sklearn
//! class names and parameters onto them.

use anofox_ml_core::{
    DecisionFunction, Fit, FitUnsupervised, FitUnsupervisedWeighted, FitWeighted, InverseTransform,
    PartialFit, PartialFitUnsupervised, Predict, PredictLogProba, PredictProba, Result, Transform,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

use anofox_ml_cluster as cluster;
use anofox_ml_discriminant as discriminant;
use anofox_ml_ensemble as ensemble;
use anofox_ml_gaussian_process as gp;
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
use crate::model::{need_y, no_weight, registry, stages_to_columns};

/// The text vectorizers fit on documents ([`crate::DynModel::fit_text`]).
fn text_only(kind: &str) -> anofox_ml_core::RustMlError {
    anofox_ml_core::RustMlError::InvalidParameter(format!(
        "{kind} is fitted on documents: use fit_text / transform_text"
    ))
}

/// Fitted form of the linear models: coefficients and intercept (enough to
/// predict), plus the fit statistics. Aliased (`NaN`) coefficients count as
/// 0, as in the Rust `predict`. The optional fields default when absent, so
/// models stored before they existed still load.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct LinearFit {
    pub coef: Array1<f64>,
    pub intercept: f64,
    /// Training R².
    #[serde(default)]
    pub r_squared: Option<f64>,
    /// The selected regularisation strength of the CV kinds (sklearn's
    /// `alpha_`).
    #[serde(default)]
    pub alpha: Option<f64>,
    /// The coefficient table (OLS / Ridge fitted with `inference: true`).
    #[serde(default)]
    pub summary: Option<reg::RegressionSummary>,
    /// The full OLS / Ridge fit, behind `predict_interval`.
    #[serde(default)]
    full: Option<FullLinear>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum FullLinear {
    Ols(reg::FittedOlsRegressor),
    Ridge(reg::FittedRidgeRegressor),
}

impl LinearFit {
    fn new(coef: Array1<f64>, intercept: Option<f64>) -> Self {
        Self {
            coef: coef.mapv(|c| if c.is_nan() { 0.0 } else { c }),
            intercept: intercept.unwrap_or(0.0),
            r_squared: None,
            alpha: None,
            summary: None,
            full: None,
        }
    }

    fn with_r2(mut self, r2: f64) -> Self {
        self.r_squared = Some(r2);
        self
    }

    fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = Some(alpha);
        self
    }

    fn ols(f: reg::FittedOlsRegressor) -> Self {
        let mut l = Self::new(f.coefficients(), f.intercept()).with_r2(f.r_squared());
        l.summary = f.summary().ok();
        l.full = Some(FullLinear::Ols(f));
        l
    }

    fn ridge(f: reg::FittedRidgeRegressor) -> Self {
        let mut l = Self::new(f.coefficients(), f.intercept()).with_r2(f.r_squared());
        l.summary = f.summary().ok();
        l.full = Some(FullLinear::Ridge(f));
        l
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

    /// Prediction intervals at miscoverage `alpha` (`n × 2`: lower,
    /// upper), from the OLS / Ridge fit with `inference: true`.
    fn interval(&self, x: ArrayView2<f64>, alpha: f64) -> Result<Array2<f64>> {
        if !(alpha > 0.0 && alpha < 1.0) {
            return Err(anofox_ml_core::RustMlError::InvalidParameter(format!(
                "alpha must be in (0, 1), got {alpha}"
            )));
        }
        let x = x.to_owned();
        let kind = reg::IntervalKind::Prediction;
        let needs_inference = |e: anofox_ml_core::RustMlError| match e {
            anofox_ml_core::RustMlError::InvalidParameter(m) if m.contains("inference") => {
                anofox_ml_core::RustMlError::InvalidParameter(
                    "predict_interval is only available for a model fitted with inference: true"
                        .into(),
                )
            }
            other => other,
        };
        let pi = match &self.full {
            Some(FullLinear::Ols(m)) => m
                .predict_interval(&x, kind, 1.0 - alpha)
                .map_err(needs_inference)?,
            Some(FullLinear::Ridge(m)) => m
                .predict_interval(&x, kind, 1.0 - alpha)
                .map_err(needs_inference)?,
            None => {
                return Err(anofox_ml_core::RustMlError::InvalidParameter(
                    "predict_interval is not available for a model stored without its full fit"
                        .into(),
                ))
            }
        };
        let mut out = Array2::<f64>::zeros((x.nrows(), 2));
        out.column_mut(0).assign(&pi.lower);
        out.column_mut(1).assign(&pi.upper);
        Ok(out)
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

/// `ConformalizedQuantileRegressor` (CQR) over a registered estimator that
/// can fit a quantile. `alpha` is fixed at fit time: the quantile models
/// are trained for `alpha / 2`, `1 − alpha / 2` and the median.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct ConformalizedQuantileRegressorConfig {
    estimator: String,
    estimator_params: serde_json::Value,
    alpha: f64,
    calibration_size: f64,
    seed: u64,
    symmetric_correction: bool,
    allow_infinite_bounds: bool,
}

impl Default for ConformalizedQuantileRegressorConfig {
    fn default() -> Self {
        let d = conformal::ConformalizedQuantileRegressor::new(0.1, |_| ());
        Self {
            estimator: "QuantileRegressor".into(),
            estimator_params: empty_params(),
            alpha: d.alpha,
            calibration_size: d.calibration_size,
            seed: d.seed,
            symmetric_correction: d.symmetric_correction,
            allow_infinite_bounds: d.allow_infinite_bounds,
        }
    }
}

/// Estimator kinds CQR can use, with the JSON that sets their quantile.
const QUANTILE_KINDS: &[&str] = &[
    "QuantileRegressor",
    "GradientBoostingRegressor",
    "HistGradientBoostingRegressor",
    #[cfg(feature = "perpetual")]
    "PerpetualRegressor",
];

/// `params` with the quantile of `kind` set to `q`.
fn with_quantile(kind: &str, params: &serde_json::Value, q: f64) -> Result<serde_json::Value> {
    let overrides = match kind {
        "QuantileRegressor" => serde_json::json!({ "quantile": q }),
        "GradientBoostingRegressor" => serde_json::json!({ "loss": "Quantile", "alpha": q }),
        "HistGradientBoostingRegressor" => {
            serde_json::json!({ "loss": "Quantile", "quantile": q })
        }
        #[cfg(feature = "perpetual")]
        "PerpetualRegressor" => {
            serde_json::json!({ "objective": { "QuantileLoss": { "quantile": q } } })
        }
        other => {
            return Err(anofox_ml_core::RustMlError::InvalidParameter(format!(
                "ConformalizedQuantileRegressor cannot fit quantiles with '{other}' \
                 (supported: {QUANTILE_KINDS:?})"
            )))
        }
    };
    let mut merged = match params {
        serde_json::Value::Null => serde_json::Map::new(),
        serde_json::Value::Object(m) => m.clone(),
        _ => {
            return Err(anofox_ml_core::RustMlError::InvalidParameter(
                "estimator_params must be a JSON object".into(),
            ))
        }
    };
    for (k, v) in overrides.as_object().expect("object literal") {
        merged.insert(k.clone(), v.clone());
    }
    Ok(serde_json::Value::Object(merged))
}

impl ConformalizedQuantileRegressorConfig {
    fn fit(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
    ) -> Result<conformal::FittedConformalizedQuantileRegressor<RegistryFitted, f64>> {
        // Check the kind and the parameters once, so `make` cannot fail.
        with_quantile(&self.estimator, &self.estimator_params, 0.5)?;
        let make = |q: f64| {
            let params =
                with_quantile(&self.estimator, &self.estimator_params, q).expect("checked above");
            RegistryEstimator::new(&self.estimator, params)
        };
        conformal::ConformalizedQuantileRegressor::new(self.alpha, make)
            .with_calibration_size(self.calibration_size)
            .with_seed(self.seed)
            .with_symmetric_correction(self.symmetric_correction)
            .with_allow_infinite_bounds(self.allow_infinite_bounds)
            .fit_view(x, y)
    }
}

/// CQR intervals, for the `alpha` the model was fitted for only.
fn cqr_interval(
    m: &conformal::FittedConformalizedQuantileRegressor<RegistryFitted, f64>,
    x: ArrayView2<f64>,
    alpha: f64,
) -> Result<Array2<f64>> {
    if (alpha - m.alpha()).abs() > 1e-12 {
        return Err(anofox_ml_core::RustMlError::InvalidParameter(format!(
            "ConformalizedQuantileRegressor was fitted for alpha = {}; refit it for alpha = {alpha}",
            m.alpha()
        )));
    }
    m.predict_interval(x).map(interval)
}

// ----------------------------------------------------- Mondrian
//
// The registry `fit` has no group argument, so the Mondrian kinds read the
// group id of each row from column `group_column` of `x` (a non-negative
// integer), at fit and at predict time. The estimator sees the other
// columns.

/// `x` without column `col`, and that column as group ids.
fn split_groups(x: ArrayView2<f64>, col: usize) -> Result<(Array2<f64>, Vec<usize>)> {
    if col >= x.ncols() {
        return Err(anofox_ml_core::RustMlError::InvalidParameter(format!(
            "group_column {col} is out of range for {} columns",
            x.ncols()
        )));
    }
    let groups = x
        .column(col)
        .iter()
        .map(|&g| {
            if g.is_finite() && g >= 0.0 && g.fract() == 0.0 {
                Ok(g as usize)
            } else {
                Err(anofox_ml_core::RustMlError::InvalidParameter(format!(
                    "group ids must be non-negative integers, got {g}"
                )))
            }
        })
        .collect::<Result<Vec<usize>>>()?;
    let keep: Vec<usize> = (0..x.ncols()).filter(|&j| j != col).collect();
    Ok((x.select(ndarray::Axis(1), &keep), groups))
}

/// A Mondrian model plus the column its group ids come from.
#[derive(Serialize, Deserialize)]
pub(crate) struct Grouped<M> {
    model: M,
    group_column: usize,
}

/// `MondrianConformalRegressor` over a registered estimator.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct MondrianConformalRegressorConfig {
    estimator: String,
    estimator_params: serde_json::Value,
    group_column: usize,
    calibration_size: f64,
    seed: u64,
    allow_infinite_bounds: bool,
}

impl Default for MondrianConformalRegressorConfig {
    fn default() -> Self {
        let d = conformal::MondrianConformalRegressor::new(());
        Self {
            estimator: default_regressor(),
            estimator_params: empty_params(),
            group_column: 0,
            calibration_size: d.calibration_size,
            seed: d.seed,
            allow_infinite_bounds: d.allow_infinite_bounds,
        }
    }
}

type MondrianRegFit = Grouped<conformal::FittedMondrianRegressor<RegistryFitted, f64>>;

impl MondrianConformalRegressorConfig {
    fn fit(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<MondrianRegFit> {
        let (x, groups) = split_groups(x, self.group_column)?;
        let model = conformal::MondrianConformalRegressor::new(RegistryEstimator::new(
            &self.estimator,
            self.estimator_params.clone(),
        ))
        .with_calibration_size(self.calibration_size)
        .with_seed(self.seed)
        .with_allow_infinite_bounds(self.allow_infinite_bounds)
        .fit_with_groups(x.view(), y, &groups)?;
        Ok(Grouped {
            model,
            group_column: self.group_column,
        })
    }
}

impl MondrianRegFit {
    fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        let (x, _) = split_groups(x, self.group_column)?;
        self.model.predict(&x)
    }

    fn predict_interval(&self, x: ArrayView2<f64>, alpha: f64) -> Result<Array2<f64>> {
        let (x, groups) = split_groups(x, self.group_column)?;
        self.model
            .predict_interval(x.view(), &groups, alpha)
            .map(interval)
    }
}

/// Bagging over a registered kind (sklearn's `BaggingRegressor(estimator=)`
/// / `BaggingClassifier(estimator=)`): the sampling settings of the tree
/// bagging kinds around `{estimator, estimator_params}`.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct BaggingEstimatorConfig {
    estimator: String,
    estimator_params: serde_json::Value,
    n_estimators: usize,
    max_samples: Option<f64>,
    bootstrap: bool,
    max_features: Option<trees::MaxFeatures>,
    bootstrap_features: bool,
    oob_score: bool,
    seed: u64,
}

impl BaggingEstimatorConfig {
    /// sklearn's defaults around the base kind `estimator`.
    fn new(estimator: &str) -> Self {
        let d = ensemble::BaggingRegressor::default();
        Self {
            estimator: estimator.into(),
            estimator_params: serde_json::json!({}),
            n_estimators: d.n_estimators,
            max_samples: d.max_samples,
            bootstrap: d.bootstrap,
            max_features: d.max_features,
            bootstrap_features: d.bootstrap_features,
            oob_score: d.oob_score,
            seed: d.seed,
        }
    }

    fn base(&self) -> RegistryEstimator {
        RegistryEstimator::new(&self.estimator, self.estimator_params.clone())
    }

    fn regressor(&self) -> ensemble::GenericBaggingRegressor<RegistryEstimator> {
        ensemble::BaggingRegressor {
            n_estimators: self.n_estimators,
            max_samples: self.max_samples,
            bootstrap: self.bootstrap,
            max_features: self.max_features,
            bootstrap_features: self.bootstrap_features,
            oob_score: self.oob_score,
            seed: self.seed,
            ..Default::default()
        }
        .with_estimator(self.base())
    }

    fn classifier(&self) -> ensemble::GenericBaggingClassifier<RegistryEstimator> {
        ensemble::BaggingClassifier {
            n_estimators: self.n_estimators,
            max_samples: self.max_samples,
            bootstrap: self.bootstrap,
            max_features: self.max_features,
            bootstrap_features: self.bootstrap_features,
            oob_score: self.oob_score,
            seed: self.seed,
            ..Default::default()
        }
        .with_estimator(self.base())
    }
}

impl Default for BaggingEstimatorConfig {
    fn default() -> Self {
        Self::new("DecisionTreeRegressor")
    }
}

type BaggingEstimatorFit = ensemble::FittedGenericBagging<RegistryFitted, f64>;

/// `Cca` fitted on one matrix `[X | Y]`: its last `y_columns` columns are
/// `Y`.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct CcaConfig {
    n_components: usize,
    scale: bool,
    max_iter: usize,
    tol: f64,
    y_columns: usize,
}

impl Default for CcaConfig {
    fn default() -> Self {
        let d = pre::Cca::new(2);
        Self {
            n_components: d.n_components,
            scale: d.scale,
            max_iter: d.max_iter,
            tol: d.tol,
            y_columns: 1,
        }
    }
}

impl CcaConfig {
    /// `[X | Y]` split at its last `y_columns` columns.
    fn split(&self, kind: &str, xy: ArrayView2<f64>) -> Result<(Array2<f64>, Array2<f64>)> {
        let p = xy.ncols();
        if self.y_columns == 0 || self.y_columns >= p {
            return Err(anofox_ml_core::RustMlError::InvalidParameter(format!(
                "{kind}: y_columns must be in 1..{p} for {p} columns, got {}",
                self.y_columns
            )));
        }
        let split = p - self.y_columns;
        Ok((
            xy.slice(ndarray::s![.., ..split]).to_owned(),
            xy.slice(ndarray::s![.., split..]).to_owned(),
        ))
    }

    fn fit(&self, xy: ArrayView2<f64>) -> Result<pre::FittedCca> {
        let (x, y) = self.split("Cca", xy)?;
        pre::Cca::new(self.n_components)
            .with_scale(self.scale)
            .with_max_iter(self.max_iter)
            .with_tol(self.tol)
            .fit(&x, &y)
    }

    fn fit_pls_canonical(&self, xy: ArrayView2<f64>) -> Result<pre::FittedCca> {
        let (x, y) = self.split("PlsCanonical", xy)?;
        let mut m = pre::PlsCanonical::new(self.n_components).with_scale(self.scale);
        (m.max_iter, m.tol) = (self.max_iter, self.tol);
        m.fit(&x, &y)
    }

    fn fit_pls_svd(&self, xy: ArrayView2<f64>) -> Result<pre::FittedPlsSvd> {
        let (x, y) = self.split("PlsSvd", xy)?;
        pre::PlsSvd::new(self.n_components)
            .with_scale(self.scale)
            .fit(&x, &y)
    }
}

/// Mondrian `SplitConformalClassifier` over a registered classifier.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct MondrianConformalClassifierConfig {
    estimator: String,
    estimator_params: serde_json::Value,
    group_column: usize,
    score: conformal::ClassificationScore,
    calibration_size: f64,
    seed: u64,
    allow_full_sets: bool,
}

impl Default for MondrianConformalClassifierConfig {
    fn default() -> Self {
        let d = SplitConformalClassifierConfig::default();
        Self {
            estimator: d.estimator,
            estimator_params: d.estimator_params,
            group_column: 0,
            score: d.score,
            calibration_size: d.calibration_size,
            seed: d.seed,
            allow_full_sets: d.allow_full_sets,
        }
    }
}

type MondrianClfFit = Grouped<conformal::FittedSplitConformalClassifier<RegistryFitted, f64>>;

impl MondrianConformalClassifierConfig {
    fn fit(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<MondrianClfFit> {
        let (x, groups) = split_groups(x, self.group_column)?;
        let model = conformal::SplitConformalClassifier::new(RegistryEstimator::new(
            &self.estimator,
            self.estimator_params.clone(),
        ))
        .with_score(self.score)
        .with_calibration_size(self.calibration_size)
        .with_seed(self.seed)
        .with_allow_full_sets(self.allow_full_sets)
        .fit_with_groups(x.view(), y, &groups)?;
        Ok(Grouped {
            model,
            group_column: self.group_column,
        })
    }
}

impl MondrianClfFit {
    fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        let (x, _) = split_groups(x, self.group_column)?;
        self.model.predict(&x)
    }

    fn predict_proba(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        let (x, _) = split_groups(x, self.group_column)?;
        PredictProba::predict_proba(self.model.model(), &x)
    }

    fn predict_set(&self, x: ArrayView2<f64>, alpha: f64) -> Result<Array2<f64>> {
        let (x, groups) = split_groups(x, self.group_column)?;
        self.model
            .predict_set_mondrian(x.view(), &groups, alpha)
            .map(membership)
    }
}

/// scikit-learn's default Gaussian process kernel, `1.0 * RBF(1.0)` with
/// both hyperparameters fixed.
fn sklearn_gp_kernel() -> gp::GpKernel {
    gp::GpKernel::Bounded {
        kernel: Box::new(gp::GpKernel::Rbf {
            length_scale: 1.0,
            signal_var: 1.0,
        }),
        bounds: vec![gp::ParamBounds::Fixed; 2],
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
        feature_importances: |m| m.feature_importances(),
    }
    DecisionTreeRegressor {
        config: trees::DecisionTreeRegressor = Default::default(),
        fitted: trees::FittedDecisionTreeRegressor<f64>,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        feature_importances: |m| m.feature_importances(),
    }

    ExtraTreeRegressor {
        config: trees::ExtraTreeRegressor = Default::default(),
        fitted: trees::FittedExtraTreeRegressor,
        fit: |c, x, y, w| c.fit(x, need_y(K, y)?, w),
        predict: |m, x| m.predict(x),
    }
    ExtraTreeClassifier {
        config: trees::ExtraTreeClassifier = Default::default(),
        fitted: trees::FittedExtraTreeClassifier,
        fit: |c, x, y, w| c.fit(x, need_y(K, y)?, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| m.predict_proba(x),
    }
    RandomTreesEmbedding {
        config: ensemble::RandomTreesEmbedding = Default::default(),
        fitted: ensemble::FittedRandomTreesEmbedding,
        fit: |c, x, y, w| FitUnsupervisedWeighted::fit_unsupervised_weighted_view(c, x, w),
        transform: |m, x| Transform::transform(m, x),
    }
    // -------------------------------------------------------- ensemble
    RandomForestClassifier {
        config: ensemble::RandomForestClassifier = Default::default(),
        fitted: ensemble::FittedRandomForestClassifier<f64>,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| PredictProba::predict_proba(m, x),
        feature_importances: |m| m.feature_importances(),
    }
    RandomForestRegressor {
        config: ensemble::RandomForestRegressor = Default::default(),
        fitted: ensemble::FittedRandomForestRegressor<f64>,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        feature_importances: |m| m.feature_importances(),
    }
    ExtraTreesClassifier {
        config: ensemble::ExtraTreesClassifier = Default::default(),
        fitted: ensemble::FittedExtraTreesClassifier<f64>,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| m.predict_proba(x),
        feature_importances: |m| m.feature_importances(),
    }
    ExtraTreesRegressor {
        config: ensemble::ExtraTreesRegressor = Default::default(),
        fitted: ensemble::FittedExtraTreesRegressor<f64>,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        feature_importances: |m| m.feature_importances(),
    }
    GradientBoostingClassifier {
        config: ensemble::GradientBoostingClassifier = Default::default(),
        fitted: ensemble::FittedGradientBoostingClassifier<f64>,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| PredictProba::predict_proba(m, x),
        decision_function: |m, x| m.decision_function(x),
        feature_importances: |m| m.feature_importances(),
        staged_predict: |m, x| stages_to_columns(m.staged_predict(x)?),
        warm_start: |c, prev, x, y, w| c.fit_warm_start(prev, x, need_y(K, y)?, w),
    }
    GradientBoostingRegressor {
        config: ensemble::GradientBoostingRegressor = Default::default(),
        fitted: ensemble::FittedGradientBoostingRegressor<f64>,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        feature_importances: |m| m.feature_importances(),
        staged_predict: |m, x| stages_to_columns(m.staged_predict(x)?),
        warm_start: |c, prev, x, y, w| c.fit_warm_start(prev, x, need_y(K, y)?, w),
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
        feature_importances: |m| m.feature_importances(),
    }
    BaggingRegressor {
        config: ensemble::BaggingRegressor = Default::default(),
        fitted: ensemble::FittedBaggingRegressor<f64>,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        feature_importances: |m| m.feature_importances(),
    }
    BaggingEstimatorRegressor {
        config: BaggingEstimatorConfig = BaggingEstimatorConfig::new("DecisionTreeRegressor"),
        fitted: BaggingEstimatorFit,
        fit: |c, x, y, w| {
            no_weight(K, w)?;
            c.regressor().fit(x, need_y(K, y)?)
        },
        predict: |m, x| m.predict(x),
    }
    BaggingEstimatorClassifier {
        config: BaggingEstimatorConfig = BaggingEstimatorConfig::new("DecisionTreeClassifier"),
        fitted: BaggingEstimatorFit,
        fit: |c, x, y, w| {
            no_weight(K, w)?;
            c.classifier().fit(x, need_y(K, y)?)
        },
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| m.predict_proba(x),
    }
    HistGradientBoostingClassifier {
        config: ensemble::HistGradientBoostingClassifier = Default::default(),
        fitted: ensemble::FittedHistGradientBoostingClassifier,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        fit_with_validation: |c, x, y, w, xv, yv, wv| {
            c.fit_with_validation(x, need_y(K, y)?, w, xv, need_y(K, yv)?, wv)
        },
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| m.predict_proba(x),
        feature_importances: |m| m.feature_importances(),
        staged_predict: |m, x| stages_to_columns(m.staged_predict(x)?),
        warm_start: |c, prev, x, y, w| c.fit_warm_start(&prev, x, need_y(K, y)?, w),
    }
    HistGradientBoostingRegressor {
        config: ensemble::HistGradientBoostingRegressor = Default::default(),
        fitted: ensemble::FittedHistGradientBoostingRegressor,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        fit_with_validation: |c, x, y, w, xv, yv, wv| {
            c.fit_with_validation(x, need_y(K, y)?, w, xv, need_y(K, yv)?, wv)
        },
        predict: |m, x| m.predict(x),
        feature_importances: |m| m.feature_importances(),
        staged_predict: |m, x| stages_to_columns(m.staged_predict(x)?),
        warm_start: |c, prev, x, y, w| c.fit_warm_start(&prev, x, need_y(K, y)?, w),
    }
    LgbmClassifier {
        config: ensemble::LgbmClassifier = Default::default(),
        fitted: ensemble::FittedLgbmClassifier,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| m.predict_proba(x),
        feature_importances: |m| m.feature_importances(),
        staged_predict: |m, x| stages_to_columns(m.staged_predict(x)?),
        warm_start: |c, prev, x, y, w| {
            let opts = ensemble::LgbmFitOptions {
                sample_weight: w,
                ..Default::default()
            };
            c.fit_warm_start(&prev, x, need_y(K, y)?, &opts)
        },
    }
    LgbmRegressor {
        config: ensemble::LgbmRegressor = Default::default(),
        fitted: ensemble::FittedLgbmRegressor,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        feature_importances: |m| m.feature_importances(),
        staged_predict: |m, x| stages_to_columns(m.staged_predict(x)?),
        warm_start: |c, prev, x, y, w| {
            let opts = ensemble::LgbmFitOptions {
                sample_weight: w,
                ..Default::default()
            };
            c.fit_warm_start(&prev, x, need_y(K, y)?, &opts)
        },
    }
    CatBoostRegressor {
        config: ensemble::CatBoostRegressor = Default::default(),
        fitted: ensemble::FittedCatBoostRegressor,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        fit_with_validation: |c, x, y, w, xv, yv, wv| {
            let eval = ensemble::CatBoostEvalSet {
                x: xv,
                y: need_y(K, yv)?,
                sample_weight: wv,
            };
            c.fit_with_eval(x, need_y(K, y)?, w, eval)
        },
        predict: |m, x| m.predict(x),
    }
    CatBoostClassifier {
        config: ensemble::CatBoostClassifier = Default::default(),
        fitted: ensemble::FittedCatBoostClassifier,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        fit_with_validation: |c, x, y, w, xv, yv, wv| {
            let eval = ensemble::CatBoostEvalSet {
                x: xv,
                y: need_y(K, yv)?,
                sample_weight: wv,
            };
            c.fit_with_eval(x, need_y(K, y)?, w, eval)
        },
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| m.predict_proba(x),
        // Two classes: the log-odds; more: one column per class.
        decision_function: |m, x| m.raw_predictions(x).map(|r| match r.ncols() {
            1 => crate::model::Out::from(r.column(0).to_owned()),
            _ => crate::model::Out::from(r),
        }),
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

    // -------------------------------------------------- meta-estimators
    VotingRegressor {
        config: crate::meta::VotingRegressorConfig = Default::default(),
        fitted: crate::meta::VotingRegressorFit,
        fit: |c, x, y, w| {
            no_weight(K, w)?;
            c.fit(x, need_y(K, y)?)
        },
        predict: |m, x| m.predict(x),
    }
    VotingClassifier {
        config: crate::meta::VotingClassifierConfig = Default::default(),
        fitted: crate::meta::VotingClassifierFit,
        fit: |c, x, y, w| {
            no_weight(K, w)?;
            c.fit(x, need_y(K, y)?)
        },
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| m.predict_proba(x),
    }
    StackingRegressor {
        config: crate::meta::StackingRegressorConfig = Default::default(),
        fitted: crate::meta::StackingRegressorFit,
        fit: |c, x, y, w| {
            no_weight(K, w)?;
            c.fit(x, need_y(K, y)?)
        },
        predict: |m, x| m.predict(x),
    }
    StackingClassifier {
        config: crate::meta::StackingClassifierConfig = Default::default(),
        fitted: crate::meta::StackingClassifierFit,
        fit: |c, x, y, w| {
            no_weight(K, w)?;
            c.fit(x, need_y(K, y)?)
        },
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| m.predict_proba(x),
    }
    CalibratedClassifierCV {
        config: crate::meta::CalibratedClassifierConfig = Default::default(),
        fitted: crate::meta::CalibratedClassifierFit,
        fit: |c, x, y, w| {
            no_weight(K, w)?;
            c.fit(x, need_y(K, y)?)
        },
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| m.predict_proba(x),
    }
    TransformedTargetRegressor {
        config: crate::meta::TransformedTargetConfig = Default::default(),
        fitted: crate::meta::TransformedTargetFit,
        fit: |c, x, y, w| {
            no_weight(K, w)?;
            c.fit(x, need_y(K, y)?)
        },
        predict: |m, x| m.predict(x),
    }
    MultiTaskLasso {
        config: crate::meta::MultiTaskConfig = crate::meta::MultiTaskConfig::lasso(),
        fitted: reg::FittedMultiTask,
        fit: |c, x, y, w| {
            no_weight(K, w)?;
            c.fit(K, x)
        },
        predict: |m, x| m.predict(x),
        coef: |m| m.coef.clone(),
    }
    MultiTaskElasticNet {
        config: crate::meta::MultiTaskConfig = Default::default(),
        fitted: reg::FittedMultiTask,
        fit: |c, x, y, w| {
            no_weight(K, w)?;
            c.fit(K, x)
        },
        predict: |m, x| m.predict(x),
        coef: |m| m.coef.clone(),
    }
    MultiTaskLassoCV {
        config: crate::meta::MultiTaskCvConfig = crate::meta::MultiTaskCvConfig::lasso(),
        fitted: reg::FittedMultiTaskCV,
        fit: |c, x, y, w| {
            no_weight(K, w)?;
            c.fit(K, x)
        },
        predict: |m, x| m.predict(x),
        coef: |m| m.fitted.coef.clone(),
    }
    MultiTaskElasticNetCV {
        config: crate::meta::MultiTaskCvConfig = Default::default(),
        fitted: reg::FittedMultiTaskCV,
        fit: |c, x, y, w| {
            no_weight(K, w)?;
            c.fit(K, x)
        },
        predict: |m, x| m.predict(x),
        coef: |m| m.fitted.coef.clone(),
    }
    MultiOutputRegressor {
        config: crate::meta::MultiOutputConfig = Default::default(),
        fitted: crate::meta::MultiOutputFit,
        fit: |c, x, y, w| {
            no_weight(K, w)?;
            c.fit(x)
        },
        predict: |m, x| m.predict(x),
    }
    RegressorChain {
        config: crate::meta::RegressorChainConfig = Default::default(),
        fitted: crate::meta::RegressorChainFit,
        fit: |c, x, y, w| {
            no_weight(K, w)?;
            c.fit(x)
        },
        predict: |m, x| m.predict(x),
    }
    Rfe {
        config: crate::meta::RfeConfig = Default::default(),
        fitted: crate::meta::RfeFit,
        fit: |c, x, y, w| {
            no_weight(K, w)?;
            c.fit(x, need_y(K, y)?)
        },
        predict: |m, x| m.predict(x),
        transform: |m, x| m.transform(x),
    }
    SequentialFeatureSelector {
        config: crate::meta::SfsConfig = Default::default(),
        fitted: crate::meta::SfsFit,
        fit: |c, x, y, w| {
            no_weight(K, w)?;
            c.fit(x, need_y(K, y)?)
        },
        transform: |m, x| m.transform(x),
    }
    Rfecv {
        config: crate::meta::RfecvConfig = Default::default(),
        fitted: crate::meta::RfecvFit,
        fit: |c, x, y, w| {
            no_weight(K, w)?;
            c.fit(x, need_y(K, y)?)
        },
        predict: |m, x| m.predict(x),
        transform: |m, x| m.transform(x),
    }
    OneVsRestClassifier {
        config: crate::meta::OneVsRestConfig = Default::default(),
        fitted: crate::meta::OneVsRestFit,
        fit: |c, x, y, w| {
            no_weight(K, w)?;
            c.fit(x, need_y(K, y)?)
        },
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| m.predict_proba(x),
        decision_function: |m, x| m.decision_function(x),
    }
    OneVsOneClassifier {
        config: crate::meta::OneVsOneConfig = Default::default(),
        fitted: crate::meta::OneVsOneFit,
        fit: |c, x, y, w| {
            no_weight(K, w)?;
            c.fit(x, need_y(K, y)?)
        },
        predict: |m, x| m.predict(x),
        decision_function: |m, x| m.decision_function(x),
    }
    FixedThresholdClassifier {
        config: crate::meta::FixedThresholdConfig = Default::default(),
        fitted: crate::meta::ThresholdFit,
        fit: |c, x, y, w| {
            no_weight(K, w)?;
            c.fit(x, need_y(K, y)?)
        },
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| m.predict_proba(x),
        decision_function: |m, x| m.decision_function(x),
    }
    TunedThresholdClassifierCV {
        config: crate::meta::TunedThresholdConfig = Default::default(),
        fitted: crate::meta::ThresholdFit,
        fit: |c, x, y, w| {
            no_weight(K, w)?;
            c.fit(x, need_y(K, y)?)
        },
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| m.predict_proba(x),
        decision_function: |m, x| m.decision_function(x),
    }

    SelfTrainingClassifier {
        config: crate::meta::SelfTrainingConfig = Default::default(),
        fitted: crate::meta::SelfTrainingFit,
        fit: |c, x, y, w| {
            no_weight(K, w)?;
            c.fit(x, need_y(K, y)?)
        },
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| m.predict_proba(x),
    }

    // ------------------------------------------------------- baselines
    DummyClassifier {
        config: anofox_ml_core::DummyClassifier = Default::default(),
        fitted: anofox_ml_core::FittedDummyClassifier,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| PredictProba::predict_proba(m, x),
    }
    DummyRegressor {
        config: anofox_ml_core::DummyRegressor = Default::default(),
        fitted: anofox_ml_core::FittedDummyRegressor,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
    }
    KNeighborsTransformer {
        config: neighbors::KNeighborsTransformer = Default::default(),
        fitted: neighbors::FittedKNeighborsTransformer,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
    }
    NeighborhoodComponentsAnalysis {
        config: neighbors::NeighborhoodComponentsAnalysis = Default::default(),
        fitted: neighbors::FittedNeighborhoodComponentsAnalysis,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        transform: |m, x| Transform::transform(m, x),
    }
    LabelPropagation {
        config: neighbors::LabelPropagation = Default::default(),
        fitted: neighbors::FittedLabelPropagation,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| PredictProba::predict_proba(m, x),
    }
    LabelSpreading {
        config: neighbors::LabelSpreading = Default::default(),
        fitted: neighbors::FittedLabelPropagation,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| PredictProba::predict_proba(m, x),
    }
    NearestCentroid {
        config: neighbors::NearestCentroid = Default::default(),
        fitted: neighbors::FittedNearestCentroid,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| PredictProba::predict_proba(m, x),
        decision_function: |m, x| DecisionFunction::decision_function(m, x),
    }
    RidgeClassifier {
        config: reg::RidgeClassifier = Default::default(),
        fitted: reg::FittedRidgeClassifier,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        decision_function: |m, x| DecisionFunction::decision_function(m, x),
    }
    Perceptron {
        config: linear::Perceptron = Default::default(),
        fitted: linear::FittedSgdClassifier<f64>,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        decision_function: |m, x| m.decision_function(x),
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
    ConformalizedQuantileRegressor {
        config: ConformalizedQuantileRegressorConfig = Default::default(),
        fitted: conformal::FittedConformalizedQuantileRegressor<RegistryFitted, f64>,
        fit: |c, x, y, w| {
            no_weight(K, w)?;
            c.fit(x, need_y(K, y)?)
        },
        predict: |m, x| m.predict(x),
        predict_interval: |m, x, alpha| cqr_interval(m, x, alpha),
    }
    MondrianConformalRegressor {
        config: MondrianConformalRegressorConfig = Default::default(),
        fitted: MondrianRegFit,
        fit: |c, x, y, w| {
            no_weight(K, w)?;
            c.fit(x, need_y(K, y)?)
        },
        predict: |m, x| m.predict(x),
        predict_interval: |m, x, alpha| m.predict_interval(x, alpha),
    }
    MondrianConformalClassifier {
        config: MondrianConformalClassifierConfig = Default::default(),
        fitted: MondrianClfFit,
        fit: |c, x, y, w| {
            no_weight(K, w)?;
            c.fit(x, need_y(K, y)?)
        },
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| m.predict_proba(x),
        predict_set: |m, x, alpha| m.predict_set(x, alpha),
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
        predict_log_proba: |m, x| PredictLogProba::predict_log_proba(m, x),
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
        partial_fit: |c, prev, x, y, w, cl| no_weight(K, w)
            .and_then(|_| PartialFit::partial_fit_view(c, prev, x, need_y(K, y)?, cl)),
    }
    SgdRegressor {
        config: linear::SgdRegressor = Default::default(),
        fitted: linear::FittedSgdRegressor<f64>,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        partial_fit: |c, prev, x, y, w, cl| no_weight(K, w)
            .and_then(|_| PartialFit::partial_fit_view(c, prev, x, need_y(K, y)?, cl)),
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
        partial_fit: |c, prev, x, y, w, cl| no_weight(K, w)
            .and_then(|_| PartialFit::partial_fit_view(c, prev, x, need_y(K, y)?, cl)),
    }

    // ------------------------------------------------------ regression
    OlsRegressor {
        config: reg::OlsRegressor = Default::default(),
        fitted: LinearFit,
        fit: |c, x, y, w| wsup(K, c, x, y, w).map(LinearFit::ols),
        predict: |m, x| m.predict(x),
        predict_interval: |m, x, alpha| m.interval(x, alpha),
        coef: |m| m.coef.clone(),
    }
    RidgeRegressor {
        config: reg::RidgeRegressor = Default::default(),
        fitted: LinearFit,
        fit: |c, x, y, w| wsup(K, c, x, y, w).map(LinearFit::ridge),
        predict: |m, x| m.predict(x),
        predict_interval: |m, x, alpha| m.interval(x, alpha),
        coef: |m| m.coef.clone(),
    }
    LassoRegressor {
        config: reg::LassoRegressor = Default::default(),
        fitted: LinearFit,
        fit: |c, x, y, w| sup(K, c, x, y, w)
            .map(|f| LinearFit::new(f.coefficients(), f.intercept()).with_r2(f.r_squared())),
        predict: |m, x| m.predict(x),
        coef: |m| m.coef.clone(),
    }
    ElasticNetRegressor {
        config: reg::ElasticNetRegressor = Default::default(),
        fitted: LinearFit,
        fit: |c, x, y, w| sup(K, c, x, y, w)
            .map(|f| LinearFit::new(f.coefficients(), f.intercept()).with_r2(f.r_squared())),
        predict: |m, x| m.predict(x),
        coef: |m| m.coef.clone(),
    }
    HuberRegressor {
        config: reg::HuberRegressor = Default::default(),
        fitted: LinearFit,
        fit: |c, x, y, w| sup(K, c, x, y, w)
            .map(|f| LinearFit::new(f.coefficients(), f.intercept()).with_r2(f.r_squared())),
        predict: |m, x| m.predict(x),
    }
    QuantileRegressor {
        config: reg::QuantileRegressor = reg::QuantileRegressor::new(0.5),
        fitted: LinearFit,
        fit: |c, x, y, w| wsup(K, c, x, y, w)
            .map(|f| LinearFit::new(f.coefficients(), f.intercept()).with_r2(f.r_squared())),
        predict: |m, x| m.predict(x),
    }
    RidgeCrossValidated {
        config: reg::RidgeCrossValidated = Default::default(),
        fitted: LinearFit,
        fit: |c, x, y, w| sup(K, c, x, y, w)
            .map(|f| {
                LinearFit::new(f.coefficients(), f.intercept())
                    .with_r2(f.r_squared())
                    .with_alpha(f.best_alpha())
            }),
        predict: |m, x| m.predict(x),
    }
    LassoCrossValidated {
        config: reg::LassoCrossValidated = Default::default(),
        fitted: LinearFit,
        fit: |c, x, y, w| sup(K, c, x, y, w)
            .map(|f| {
                LinearFit::new(f.coefficients(), f.intercept())
                    .with_r2(f.r_squared())
                    .with_alpha(f.best_alpha())
            }),
        predict: |m, x| m.predict(x),
    }
    ElasticNetCrossValidated {
        config: reg::ElasticNetCrossValidated = Default::default(),
        fitted: LinearFit,
        fit: |c, x, y, w| sup(K, c, x, y, w)
            .map(|f| {
                LinearFit::new(f.coefficients(), f.intercept())
                    .with_r2(f.r_squared())
                    .with_alpha(f.best_alpha())
            }),
        predict: |m, x| m.predict(x),
    }
    LogisticRegressor {
        config: reg::LogisticRegressor = Default::default(),
        fitted: reg::FittedLogisticRegressor,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| PredictProba::predict_proba(m, x),
        predict_log_proba: |m, x| PredictLogProba::predict_log_proba(m, x),
        decision_function: |m, x| DecisionFunction::decision_function(m, x),
        coef: |m| m.coef().clone(),
    }
    LogisticRegressionCV {
        config: reg::LogisticRegressionCV = Default::default(),
        fitted: reg::FittedLogisticRegressionCV,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| PredictProba::predict_proba(m, x),
        decision_function: |m, x| DecisionFunction::decision_function(m, x),
    }
    OrthogonalMatchingPursuitCV {
        config: reg::OrthogonalMatchingPursuitCV = Default::default(),
        fitted: reg::FittedOrthogonalMatchingPursuitCV,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
    }
    PoissonRegressor {
        config: reg::PoissonRegressor = Default::default(),
        fitted: reg::FittedPoissonRegressor,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        fit_with_offset: |c, x, y, w, off| c.fit_weighted_with_offset(
            x,
            need_y(K, y)?,
            w.map(|w| w.to_owned()).as_ref(),
            Some(&off.to_owned()),
        ),
        predict_with_offset: |m, x, off| m.predict_with_offset(&x.to_owned(), &off.to_owned()),
    }
    BinomialRegressor {
        config: reg::BinomialRegressor = Default::default(),
        fitted: reg::FittedBinomialRegressor,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        fit_with_offset: |c, x, y, w, off| {
            c.fit_with_offset(x, need_y(K, y)?, w, Some(&off.to_owned()))
        },
        predict_with_offset: |m, x, off| m.predict_with_offset(&x.to_owned(), &off.to_owned()),
    }
    GammaRegressor {
        config: reg::GammaRegressor = Default::default(),
        fitted: reg::FittedTweedieRegressor,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        fit_with_offset: |c, x, y, w, off| c.fit_weighted_with_offset(
            &x.to_owned(),
            &need_y(K, y)?.to_owned(),
            w.map(|w| w.to_owned()).as_ref(),
            Some(&off.to_owned()),
        ),
        predict_with_offset: |m, x, off| m.predict_with_offset(&x.to_owned(), &off.to_owned()),
    }
    TweedieRegressor {
        config: reg::TweedieRegressor = reg::TweedieRegressor::new(0.0),
        fitted: reg::FittedTweedieRegressor,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        fit_with_offset: |c, x, y, w, off| c.fit_weighted_with_offset(
            x,
            need_y(K, y)?,
            w.map(|w| w.to_owned()).as_ref(),
            Some(&off.to_owned()),
        ),
        predict_with_offset: |m, x, off| m.predict_with_offset(&x.to_owned(), &off.to_owned()),
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
    IsotonicRegressor {
        config: reg::IsotonicRegressor = Default::default(),
        fitted: reg::FittedIsotonicRegressor,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
    }
    LassoLarsIC {
        config: reg::LassoLarsIC = reg::LassoLarsIC::new(reg::IcCriterion::Aic),
        fitted: reg::FittedLassoLarsIC,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
    }
    LarsCV {
        config: reg::LarsCV = Default::default(),
        fitted: reg::FittedLarsCV,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        coef: |m| m.fitted.coef.clone(),
    }
    LassoLarsCV {
        config: reg::LarsCV = reg::LarsCV::lasso(),
        fitted: reg::FittedLarsCV,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        coef: |m| m.fitted.coef.clone(),
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

    // ------------------------------------------------ gaussian process
    GaussianProcessRegressor {
        config: gp::GaussianProcessRegressor = gp::GaussianProcessRegressor::new(sklearn_gp_kernel()),
        fitted: gp::FittedGaussianProcessRegressor,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_std: |m, x| m.predict_std(x),
    }
    GaussianProcessClassifier {
        config: gp::MulticlassGaussianProcessClassifier =
            gp::MulticlassGaussianProcessClassifier::new(sklearn_gp_kernel()),
        fitted: gp::FittedMulticlassGaussianProcessClassifier,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| PredictProba::predict_proba(m, x),
    }

    // ----------------------------------------------------- naive bayes
    GaussianNB {
        config: nb::GaussianNB = Default::default(),
        fitted: nb::FittedGaussianNB<f64>,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| PredictProba::predict_proba(m, x),
        predict_log_proba: |m, x| PredictLogProba::predict_log_proba(m, x),
        partial_fit: |c, prev, x, y, w, cl| c.partial_fit_weighted(prev, x, need_y(K, y)?, cl, w),
    }
    MultinomialNB {
        config: nb::MultinomialNB = Default::default(),
        fitted: nb::FittedMultinomialNB<f64>,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| PredictProba::predict_proba(m, x),
        predict_log_proba: |m, x| PredictLogProba::predict_log_proba(m, x),
        partial_fit: |c, prev, x, y, w, cl| c.partial_fit_weighted(prev, x, need_y(K, y)?, cl, w),
    }
    BernoulliNB {
        config: nb::BernoulliNB = Default::default(),
        fitted: nb::FittedBernoulliNB<f64>,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| PredictProba::predict_proba(m, x),
        predict_log_proba: |m, x| PredictLogProba::predict_log_proba(m, x),
        partial_fit: |c, prev, x, y, w, cl| c.partial_fit_weighted(prev, x, need_y(K, y)?, cl, w),
    }
    ComplementNB {
        config: nb::ComplementNB = Default::default(),
        fitted: nb::FittedComplementNB<f64>,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| PredictProba::predict_proba(m, x),
        predict_log_proba: |m, x| PredictLogProba::predict_log_proba(m, x),
        partial_fit: |c, prev, x, y, w, cl| c.partial_fit_weighted(prev, x, need_y(K, y)?, cl, w),
    }
    CategoricalNB {
        config: nb::CategoricalNB = Default::default(),
        fitted: nb::FittedCategoricalNB<f64>,
        fit: |c, x, y, w| wsup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| PredictProba::predict_proba(m, x),
        predict_log_proba: |m, x| PredictLogProba::predict_log_proba(m, x),
        partial_fit: |c, prev, x, y, w, cl| c.partial_fit_weighted(prev, x, need_y(K, y)?, cl, w),
    }

    // ------------------------------------------------------- neighbors
    KnnClassifier {
        config: neighbors::KnnClassifier = Default::default(),
        fitted: neighbors::FittedKnnClassifier<f64>,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| PredictProba::predict_proba(m, x),
        kneighbors: |m, x, k| m.kneighbors(&x.to_owned(), k),
    }
    KnnRegressor {
        config: neighbors::KnnRegressor = Default::default(),
        fitted: neighbors::FittedKnnRegressor<f64>,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        kneighbors: |m, x, k| m.kneighbors(&x.to_owned(), k),
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
        predict_log_proba: |m, x| PredictLogProba::predict_log_proba(m, x),
        decision_function: |m, x| DecisionFunction::decision_function(m, x),
        transform: |m, x| Transform::transform(m, x),
    }
    QuadraticDiscriminantAnalysis {
        config: discriminant::QuadraticDiscriminantAnalysis = Default::default(),
        fitted: discriminant::FittedQuadraticDiscriminantAnalysis,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| PredictProba::predict_proba(m, x),
        predict_log_proba: |m, x| PredictLogProba::predict_log_proba(m, x),
        decision_function: |m, x| DecisionFunction::decision_function(m, x),
    }

    // -------------------------------------------------- neural network
    MlpClassifier {
        config: nn::MlpClassifier = Default::default(),
        fitted: nn::FittedMlpClassifier<f64>,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        predict_proba: |m, x| m.predict_proba(x),
        partial_fit: |c, prev, x, y, w, cl| no_weight(K, w)
            .and_then(|_| PartialFit::partial_fit_view(c, prev, x, need_y(K, y)?, cl)),
        warm_start: |c, prev, x, y, w| no_weight(K, w)
            .and_then(|_| c.fit_warm_start(prev, &x.to_owned(), &need_y(K, y)?.to_owned())),
    }
    BernoulliRbm {
        config: nn::BernoulliRbm = Default::default(),
        fitted: nn::FittedBernoulliRbm,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
        score_samples: |m, x| m.score_samples(x),
        partial_fit: |c, prev, x, y, w, cl| {
            no_weight(K, w)?;
            match prev {
                Some(mut m) => m.partial_fit(x).map(|_| m),
                None => c.partial_fit_first(x),
            }
        },
    }
    MlpRegressor {
        config: nn::MlpRegressor = Default::default(),
        fitted: nn::FittedMlpRegressor<f64>,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        partial_fit: |c, prev, x, y, w, cl| no_weight(K, w)
            .and_then(|_| PartialFit::partial_fit_view(c, prev, x, need_y(K, y)?, cl)),
        warm_start: |c, prev, x, y, w| no_weight(K, w)
            .and_then(|_| c.fit_warm_start(prev, &x.to_owned(), &need_y(K, y)?.to_owned())),
    }

    // --------------------------------------------------- preprocessing
    StandardScaler {
        config: pre::StandardScaler = Default::default(),
        fitted: pre::FittedStandardScaler<f64>,
        fit: |c, x, y, w| FitUnsupervisedWeighted::fit_unsupervised_weighted_view(c, x, w),
        transform: |m, x| Transform::transform(m, x),
        inverse_transform: |m, x| InverseTransform::inverse_transform(m, x),
        partial_fit: |c, prev, x, y, w, cl| PartialFitUnsupervised::partial_fit_weighted_view(c, prev, x, w),
    }
    MinMaxScaler {
        config: pre::MinMaxScaler<f64> = Default::default(),
        fitted: pre::FittedMinMaxScaler<f64>,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
        inverse_transform: |m, x| InverseTransform::inverse_transform(m, x),
        partial_fit: |c, prev, x, y, w, cl| PartialFitUnsupervised::partial_fit_weighted_view(c, prev, x, w),
    }
    MaxAbsScaler {
        config: pre::MaxAbsScaler = Default::default(),
        fitted: pre::FittedMaxAbsScaler<f64>,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
        inverse_transform: |m, x| InverseTransform::inverse_transform(m, x),
        partial_fit: |c, prev, x, y, w, cl| PartialFitUnsupervised::partial_fit_weighted_view(c, prev, x, w),
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
    Cca {
        config: CcaConfig = Default::default(),
        fitted: pre::FittedCca,
        fit: |c, x, y, w| {
            no_weight(K, w)?;
            c.fit(x)
        },
        transform: |m, x| m.transform_x(x),
        transform_y: |m, y| m.transform_y(y),
    }
    PlsCanonical {
        config: CcaConfig = Default::default(),
        fitted: pre::FittedCca,
        fit: |c, x, y, w| {
            no_weight(K, w)?;
            c.fit_pls_canonical(x)
        },
        transform: |m, x| m.transform_x(x),
        transform_y: |m, y| m.transform_y(y),
    }
    PlsSvd {
        config: CcaConfig = Default::default(),
        fitted: pre::FittedPlsSvd,
        fit: |c, x, y, w| {
            no_weight(K, w)?;
            c.fit_pls_svd(x)
        },
        transform: |m, x| m.transform_x(x),
        transform_y: |m, y| m.transform_y(y),
    }
    PlsRegression {
        config: pre::PlsRegression = pre::PlsRegression::new(2),
        fitted: pre::FittedPlsRegression,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        predict: |m, x| m.predict(x),
        transform: |m, x| Transform::transform(m, x),
    }
    IncrementalPca {
        config: pre::IncrementalPca = Default::default(),
        fitted: pre::FittedIncrementalPca<f64>,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
        inverse_transform: |m, x| InverseTransform::inverse_transform(m, x),
        partial_fit: |c, prev, x, y, w, cl| PartialFitUnsupervised::partial_fit_weighted_view(c, prev, x, w),
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
    LatentDirichletAllocation {
        config: pre::LatentDirichletAllocation = Default::default(),
        fitted: pre::FittedLatentDirichletAllocation,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
    }
    SparsePca {
        config: pre::SparsePca = Default::default(),
        fitted: pre::FittedSparsePca,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
    }
    MiniBatchNmf {
        config: pre::MiniBatchNmf = Default::default(),
        fitted: pre::FittedMiniBatchNmf,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
        embedding: |m| m.w.clone(),
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
    FactorAnalysis {
        config: pre::FactorAnalysis = Default::default(),
        fitted: pre::FittedFactorAnalysis,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
        score_samples: |m, x| m.score_samples(x),
    }
    KernelDensity {
        config: neighbors::KernelDensity = Default::default(),
        fitted: neighbors::FittedKernelDensity,
        fit: |c, x, y, w| c.fit_weighted(x, w),
        score_samples: |m, x| m.score_samples(x),
    }
    Nystroem {
        config: pre::Nystroem = Default::default(),
        fitted: pre::FittedNystroem,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
    }
    RbfSampler {
        config: pre::RbfSampler = Default::default(),
        fitted: pre::FittedRbfSampler,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
    }
    KernelCenterer {
        config: pre::KernelCenterer = Default::default(),
        fitted: pre::FittedKernelCenterer,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
    }
    GaussianRandomProjection {
        config: pre::GaussianRandomProjection = Default::default(),
        fitted: pre::FittedRandomProjection,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
    }
    SparseRandomProjection {
        config: pre::SparseRandomProjection = Default::default(),
        fitted: pre::FittedRandomProjection,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
    }
    EmpiricalCovariance {
        config: pre::EmpiricalCovariance = Default::default(),
        fitted: pre::FittedCovariance,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
    }
    ShrunkCovariance {
        config: pre::ShrunkCovariance = Default::default(),
        fitted: pre::FittedCovariance,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
    }
    LedoitWolf {
        config: pre::LedoitWolf = Default::default(),
        fitted: pre::FittedCovariance,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
    }
    Oas {
        config: pre::Oas = Default::default(),
        fitted: pre::FittedCovariance,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
    }
    MinCovDet {
        config: pre::MinCovDet = Default::default(),
        fitted: pre::FittedCovariance,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
    }
    EllipticEnvelope {
        config: pre::EllipticEnvelope = Default::default(),
        fitted: pre::FittedEllipticEnvelope,
        fit: |c, x, y, w| unsup(K, c, x, w),
        predict: |m, x| m.predict(x),
        decision_function: |m, x| m.decision_function(x),
        score_samples: |m, x| m.score_samples(x),
    }
    GenericUnivariateSelect {
        config: pre::GenericUnivariateSelect = Default::default(),
        fitted: pre::FittedSelectKBest<f64>,
        fit: |c, x, y, w| sup(K, c, x, y, w),
        transform: |m, x| Transform::transform(m, x),
    }
    CountVectorizer {
        config: anofox_ml_text::CountVectorizer = Default::default(),
        fitted: anofox_ml_text::FittedCountVectorizer,
        fit: |c, x, y, w| Err(text_only(K)),
        fit_text: |c, docs| c.fit(docs),
        transform_text: |m, docs| m.transform(docs),
    }
    TfidfVectorizer {
        config: anofox_ml_text::TfidfVectorizer = Default::default(),
        fitted: anofox_ml_text::FittedTfidfVectorizer,
        fit: |c, x, y, w| Err(text_only(K)),
        fit_text: |c, docs| c.fit(docs),
        transform_text: |m, docs| m.transform(docs),
    }
    HashingVectorizer {
        config: anofox_ml_text::HashingVectorizer = Default::default(),
        fitted: anofox_ml_text::HashingVectorizer,
        fit: |c, x, y, w| Err(text_only(K)),
        fit_text: |c, docs| Ok::<_, anofox_ml_core::RustMlError>(c.clone()),
        transform_text: |m, docs| Ok::<_, anofox_ml_core::RustMlError>(m.transform(docs)),
    }
    TfidfTransformer {
        config: anofox_ml_text::TfidfTransformer = Default::default(),
        fitted: anofox_ml_text::FittedTfidfTransformer,
        fit: |c, x, y, w| unsup(K, c, x, w),
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
    BisectingKMeans {
        config: cluster::BisectingKMeans = Default::default(),
        fitted: cluster::FittedBisectingKMeans,
        fit: |c, x, y, w| FitUnsupervisedWeighted::fit_unsupervised_weighted_view(c, x, w),
        predict: |m, x| m.predict(x).map(|l| l.mapv(|v| v as f64)),
        labels: |m| m.labels.mapv(|v| v as f64),
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
    FeatureAgglomeration {
        config: cluster::FeatureAgglomeration = Default::default(),
        fitted: cluster::FittedFeatureAgglomeration,
        fit: |c, x, y, w| unsup(K, c, x, w),
        transform: |m, x| Transform::transform(m, x),
        inverse_transform: |m, x| m.inverse_transform(x),
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
        predict_log_proba: |m, x| PredictLogProba::predict_log_proba(m, x),
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
    SpectralEmbedding {
        config: manifold::SpectralEmbedding = Default::default(),
        fitted: manifold::FittedSpectralEmbedding,
        fit: |c, x, y, w| unsup(K, c, x, w),
        embedding: |m| m.embedding.clone(),
    }
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
