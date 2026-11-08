//! The object-safe model interface and the `registry!` macro.
//!
//! The registry lists, per estimator, its config and fitted types, how it is
//! fitted and which prediction methods it has. Unsupported methods return an
//! error naming the estimator and the method.

use anofox_ml_core::persistence::{self, ModelBundle};
use anofox_ml_core::{Result, RustMlError};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{de::DeserializeOwned, Serialize};

/// Output of a prediction method that may be 1-D or 2-D.
#[derive(Debug)]
pub enum Out {
    A1(Array1<f64>),
    A2(Array2<f64>),
}

impl From<Array1<f64>> for Out {
    fn from(a: Array1<f64>) -> Self {
        Out::A1(a)
    }
}

impl From<Array2<f64>> for Out {
    fn from(a: Array2<f64>) -> Self {
        Out::A2(a)
    }
}

/// Error for a method an estimator does not provide.
pub(crate) fn unsupported(kind: &str, method: &str) -> RustMlError {
    RustMlError::InvalidParameter(format!("{kind} does not support {method}"))
}

pub(crate) fn not_fitted(kind: &str) -> RustMlError {
    RustMlError::NotFitted(format!("{kind} is not fitted; call fit first"))
}

/// Object-safe view of a registered estimator (config + optional fitted
/// state).
pub trait DynModel: Send + Sync {
    fn kind(&self) -> &'static str;
    fn capabilities(&self) -> Vec<&'static str>;
    fn params(&self) -> Result<serde_json::Value>;
    fn is_fitted(&self) -> bool;
    fn fit(
        &mut self,
        x: ArrayView2<f64>,
        y: Option<ArrayView1<f64>>,
        sample_weight: Option<ArrayView1<f64>>,
    ) -> Result<()>;
    fn to_bytes(&self) -> Result<Vec<u8>>;
    fn fitted_json(&self) -> Result<String>;
    /// Merge JSON overrides into the configuration (unknown keys are an
    /// error). A fitted state is kept: it is what `warm_start` continues,
    /// and new parameters take effect at the next fit.
    fn set_params(&mut self, params: &serde_json::Value) -> Result<()>;

    fn predict(&self, _x: ArrayView2<f64>) -> Result<Out> {
        Err(unsupported(self.kind(), "predict"))
    }
    fn predict_proba(&self, _x: ArrayView2<f64>) -> Result<Out> {
        Err(unsupported(self.kind(), "predict_proba"))
    }
    /// Log-probabilities of the classes: the estimator's own where it has
    /// them (naive Bayes from its joint log-likelihoods, which stay finite
    /// where `predict_proba` underflows), else `ln(predict_proba)`.
    fn predict_log_proba(&self, _x: ArrayView2<f64>) -> Result<Out> {
        Err(unsupported(self.kind(), "predict_log_proba"))
    }
    /// Standard deviation of the predictive distribution (the Gaussian
    /// process regressor; sklearn's `predict(return_std=True)`).
    fn predict_std(&self, _x: ArrayView2<f64>) -> Result<Out> {
        Err(unsupported(self.kind(), "predict_std"))
    }
    fn decision_function(&self, _x: ArrayView2<f64>) -> Result<Out> {
        Err(unsupported(self.kind(), "decision_function"))
    }
    fn transform(&self, _x: ArrayView2<f64>) -> Result<Out> {
        Err(unsupported(self.kind(), "transform"))
    }
    fn inverse_transform(&self, _x: ArrayView2<f64>) -> Result<Out> {
        Err(unsupported(self.kind(), "inverse_transform"))
    }
    fn score_samples(&self, _x: ArrayView2<f64>) -> Result<Out> {
        Err(unsupported(self.kind(), "score_samples"))
    }
    /// Per-sample result of the training fit (cluster labels, outlier
    /// labels).
    fn labels(&self) -> Result<Out> {
        Err(unsupported(self.kind(), "labels"))
    }
    /// Transformed training data from the fit: manifold embeddings, the
    /// cross-fitted TargetEncoder output.
    fn embedding(&self) -> Result<Out> {
        Err(unsupported(self.kind(), "embedding"))
    }
    /// Prediction intervals at miscoverage `alpha`: an `n × 2` array of
    /// lower and upper bounds (conformal regressors).
    fn predict_interval(&self, _x: ArrayView2<f64>, _alpha: f64) -> Result<Out> {
        Err(unsupported(self.kind(), "predict_interval"))
    }
    /// Prediction sets at miscoverage `alpha`: an `n × classes` array, 1 for
    /// the classes in each row's set and 0 otherwise (conformal
    /// classifiers).
    fn predict_set(&self, _x: ArrayView2<f64>, _alpha: f64) -> Result<Out> {
        Err(unsupported(self.kind(), "predict_set"))
    }
    /// Distances and training-row indices of the `k` nearest training rows
    /// of every row of `x` (`None`: the model's `n_neighbors`), each
    /// `n × k`, nearest first (the kNN models).
    fn kneighbors(
        &self,
        _x: ArrayView2<f64>,
        _k: Option<usize>,
    ) -> Result<(Array2<f64>, Array2<usize>)> {
        Err(unsupported(self.kind(), "kneighbors"))
    }
    /// Per-feature importances of the fitted model (trees, forests,
    /// boosting, bagging).
    fn feature_importances(&self) -> Result<Out> {
        Err(unsupported(self.kind(), "feature_importances"))
    }
    /// Predictions after each boosting stage: an `n × stages` array whose
    /// column `k` is the prediction of the first `k + 1` stages (sklearn's
    /// `staged_predict`; classes for classifiers).
    fn staged_predict(&self, _x: ArrayView2<f64>) -> Result<Out> {
        Err(unsupported(self.kind(), "staged_predict"))
    }
    /// Continue training the fitted model on `(x, y)` up to the configured
    /// number of stages (sklearn's `warm_start=True` refit; raise
    /// `n_estimators` with [`DynModel::set_params`] first). On error the
    /// fitted model is unchanged.
    fn warm_start<'a>(
        &mut self,
        _x: ArrayView2<'a, f64>,
        _y: Option<ArrayView1<'a, f64>>,
        _sample_weight: Option<ArrayView1<'a, f64>>,
    ) -> Result<()> {
        Err(unsupported(self.kind(), "warm_start"))
    }
    /// Fold the batch `(x, y)` into the model (sklearn's `partial_fit`): an
    /// unfitted model starts from the batch, a fitted one continues, and
    /// [`DynModel::fit`] starts over. `y` is `None` for the unsupervised
    /// kinds; `classes` names a classifier's classes on its first batch. On
    /// error the fitted model is unchanged.
    fn partial_fit<'a>(
        &mut self,
        _x: ArrayView2<'a, f64>,
        _y: Option<ArrayView1<'a, f64>>,
        _sample_weight: Option<ArrayView1<'a, f64>>,
        _classes: Option<&[f64]>,
    ) -> Result<()> {
        Err(unsupported(self.kind(), "partial_fit"))
    }
    /// Fit with an offset on the link scale (the GLMs; `ln(exposure)` for a
    /// log link). [`DynModel::predict`] then predicts at offset 0.
    fn fit_with_offset<'a>(
        &mut self,
        _x: ArrayView2<'a, f64>,
        _y: Option<ArrayView1<'a, f64>>,
        _sample_weight: Option<ArrayView1<'a, f64>>,
        _offset: ArrayView1<'a, f64>,
    ) -> Result<()> {
        Err(unsupported(self.kind(), "fit_with_offset"))
    }
    /// Predictions with an offset on the link scale (the GLMs).
    fn predict_with_offset(&self, _x: ArrayView2<f64>, _offset: ArrayView1<f64>) -> Result<Out> {
        Err(unsupported(self.kind(), "predict_with_offset"))
    }
    /// Fit with an explicit validation set for early stopping (sklearn's
    /// HistGradientBoosting `fit(X, y, sample_weight, X_val=, y_val=,
    /// sample_weight_val=)`).
    fn fit_with_validation<'a>(
        &mut self,
        _x: ArrayView2<'a, f64>,
        _y: Option<ArrayView1<'a, f64>>,
        _sample_weight: Option<ArrayView1<'a, f64>>,
        _x_val: ArrayView2<'a, f64>,
        _y_val: Option<ArrayView1<'a, f64>>,
        _sample_weight_val: Option<ArrayView1<'a, f64>>,
    ) -> Result<()> {
        Err(unsupported(self.kind(), "fit_with_validation"))
    }
}

/// Stack per-stage predictions as the columns of an `n × stages` array.
pub(crate) fn stages_to_columns(stages: impl Iterator<Item = Array1<f64>>) -> Result<Array2<f64>> {
    let stages: Vec<Array1<f64>> = stages.collect();
    let n = stages.first().map_or(0, |s| s.len());
    Ok(Array2::from_shape_fn((n, stages.len()), |(i, k)| {
        stages[k][i]
    }))
}

/// Build a config from its default plus the JSON overrides in `params`.
/// Unknown keys are an error.
pub(crate) fn config_from_params<C: Serialize + DeserializeOwned>(
    kind: &str,
    default: C,
    params: &serde_json::Value,
) -> Result<C> {
    let mut base =
        serde_json::to_value(&default).map_err(|e| RustMlError::Serialization(e.to_string()))?;
    match (base.as_object_mut(), params.as_object()) {
        (Some(b), Some(p)) => {
            for (k, v) in p {
                if !b.contains_key(k) {
                    let mut known: Vec<&String> = b.keys().collect();
                    known.sort();
                    return Err(RustMlError::InvalidParameter(format!(
                        "{kind} has no parameter '{k}' (known: {known:?})"
                    )));
                }
                b.insert(k.clone(), v.clone());
            }
        }
        (_, None) if params.is_null() => {}
        (None, Some(p)) if p.is_empty() => {}
        _ => {
            return Err(RustMlError::InvalidParameter(format!(
                "{kind} parameters must be a JSON object"
            )))
        }
    }
    serde_json::from_value(base)
        .map_err(|e| RustMlError::InvalidParameter(format!("invalid {kind} parameters: {e}")))
}

/// Concrete registry entry: config plus optional fitted model.
pub(crate) struct Est<C, M> {
    pub config: C,
    pub fitted: Option<M>,
}

impl<C: Serialize + DeserializeOwned, M: Serialize + DeserializeOwned> Est<C, M> {
    pub fn bytes(&self, kind: &str) -> Result<Vec<u8>> {
        let fitted = self.fitted.as_ref().ok_or_else(|| not_fitted(kind))?;
        persistence::to_binary_bytes(&ModelBundle {
            config: &self.config,
            fitted,
        })
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let b: ModelBundle<C, M> = persistence::from_binary_bytes(bytes)?;
        Ok(Self {
            config: b.config,
            fitted: Some(b.fitted),
        })
    }
}

/// Require `y` for a supervised fit.
pub(crate) fn need_y<'a>(
    kind: &str,
    y: Option<ArrayView1<'a, f64>>,
) -> Result<ArrayView1<'a, f64>> {
    y.ok_or_else(|| RustMlError::InvalidParameter(format!("{kind}.fit needs y")))
}

/// Reject `sample_weight` for estimators without weighted fitting.
pub(crate) fn no_weight(kind: &str, w: Option<ArrayView1<f64>>) -> Result<()> {
    match w {
        Some(_) => Err(RustMlError::InvalidParameter(format!(
            "{kind} does not accept sample_weight"
        ))),
        None => Ok(()),
    }
}

/// Generate one [`DynModel`] implementation per estimator plus the
/// constructors `make` / `load` / `default_params` / `kinds`.
///
/// ```ignore
/// registry! {
///     RandomForestClassifier {
///         config: ensemble::RandomForestClassifier = Default::default(),
///         fitted: ensemble::FittedRandomForestClassifier<f64>,
///         fit: |c, x, y, w| c.fit_weighted(x, need_y(K, y)?, w),
///         predict: |m, x| m.predict(x),
///         predict_proba: |m, x| m.predict_proba(x),
///     }
/// }
/// ```
///
/// An entry may carry attributes such as `#[cfg(feature = "...")]`; they
/// apply to everything generated for it.
///
/// `fit` receives the config `c`, `x`, `y: Option<&Array1>` and
/// `w: Option<&Array1>` and returns the fitted model; inside it `K` is the
/// estimator's kind. Prediction methods receive the fitted model `m` and
/// `x` and return `Result<Array1 | Array2>`; `predict_interval` /
/// `predict_set` also receive `alpha` (`|m, x, alpha|`), `kneighbors`
/// receives `|m, x, k|` and returns `(distances, indices)`; `labels` /
/// `embedding` / `feature_importances` receive only `m` and return the
/// array; `warm_start` receives `|c, prev, x, y, w|` (a copy of the fitted
/// model as `prev`) and returns the new fitted model; `partial_fit`
/// receives `|c, prev, x, y, w, classes|` (`prev`: the fitted model, if any);
/// `fit_with_offset` receives `|c, x, y, w, offset|`, `predict_with_offset`
/// `|m, x, offset|`;
/// `fit_with_validation` receives `|c, x, y, w, x_val, y_val, w_val|`.
/// Every entry gets `set_params`.
macro_rules! registry {
    ($(
        $(#[$meta:meta])*
        $kind:ident {
            config: $cfg:ty = $default:expr,
            fitted: $fitted:ty,
            fit: |$c:ident, $x:ident, $y:ident, $w:ident| $fit:expr,
            $( $cap:ident : |$m:ident $(, $cx:ident $(, $ca:ident $(, $cy:ident, $cw:ident $(, $cvx:ident $(, $cvy:ident)?)?)?)?)?| $body:expr ),* $(,)?
        }
    )*) => {
        $(
            $(#[$meta])*
            #[allow(non_camel_case_types)]
            struct $kind($crate::model::Est<$cfg, $fitted>);

            $(#[$meta])*
            impl $crate::model::DynModel for $kind {
                fn kind(&self) -> &'static str {
                    stringify!($kind)
                }
                fn capabilities(&self) -> Vec<&'static str> {
                    vec![$(stringify!($cap)),*]
                }
                fn params(&self) -> anofox_ml_core::Result<serde_json::Value> {
                    serde_json::to_value(&self.0.config)
                        .map_err(|e| anofox_ml_core::RustMlError::Serialization(e.to_string()))
                }
                fn is_fitted(&self) -> bool {
                    self.0.fitted.is_some()
                }
                #[allow(unused_variables)]
                fn fit(
                    &mut self,
                    $x: ndarray::ArrayView2<f64>,
                    $y: Option<ndarray::ArrayView1<f64>>,
                    $w: Option<ndarray::ArrayView1<f64>>,
                ) -> anofox_ml_core::Result<()> {
                    #[allow(non_snake_case, dead_code)]
                    const K: &str = stringify!($kind);
                    let $c = &self.0.config;
                    let fitted: $fitted = $fit?;
                    self.0.fitted = Some(fitted);
                    Ok(())
                }
                fn to_bytes(&self) -> anofox_ml_core::Result<Vec<u8>> {
                    self.0.bytes(stringify!($kind))
                }
                fn fitted_json(&self) -> anofox_ml_core::Result<String> {
                    let f = self
                        .0
                        .fitted
                        .as_ref()
                        .ok_or_else(|| $crate::model::not_fitted(stringify!($kind)))?;
                    serde_json::to_string(f)
                        .map_err(|e| anofox_ml_core::RustMlError::Serialization(e.to_string()))
                }
                fn set_params(&mut self, params: &serde_json::Value) -> anofox_ml_core::Result<()> {
                    let current = serde_json::to_value(&self.0.config)
                        .and_then(serde_json::from_value::<$cfg>)
                        .map_err(|e| anofox_ml_core::RustMlError::Serialization(e.to_string()))?;
                    self.0.config =
                        $crate::model::config_from_params(stringify!($kind), current, params)?;
                    Ok(())
                }
                $( registry!(@cap $kind, $cap, $m $(, $cx $(, $ca $(, $cy, $cw $(, $cvx $(, $cvy)?)?)?)?)?, $body); )*
            }
        )*

        /// Create an unfitted estimator of `kind` from JSON parameters.
        pub fn make(
            kind: &str,
            params: &serde_json::Value,
        ) -> anofox_ml_core::Result<Box<dyn $crate::model::DynModel>> {
            match kind {
                $(
                    $(#[$meta])*
                    stringify!($kind) => {
                        let default: $cfg = $default;
                        let config = $crate::model::config_from_params(kind, default, params)?;
                        Ok(Box::new($kind($crate::model::Est { config, fitted: None })))
                    }
                )*
                other => Err(anofox_ml_core::RustMlError::InvalidParameter(format!(
                    "unknown estimator kind '{other}'"
                ))),
            }
        }

        /// Restore a fitted estimator from `to_bytes` output.
        pub fn load(
            kind: &str,
            bytes: &[u8],
        ) -> anofox_ml_core::Result<Box<dyn $crate::model::DynModel>> {
            match kind {
                $(
                    $(#[$meta])*
                    stringify!($kind) => Ok(Box::new($kind(
                        $crate::model::Est::<$cfg, $fitted>::from_bytes(bytes)?,
                    ))),
                )*
                other => Err(anofox_ml_core::RustMlError::InvalidParameter(format!(
                    "unknown estimator kind '{other}'"
                ))),
            }
        }

        /// Default parameters of `kind` as JSON.
        pub fn default_params(kind: &str) -> anofox_ml_core::Result<serde_json::Value> {
            match kind {
                $(
                    $(#[$meta])*
                    stringify!($kind) => {
                        let default: $cfg = $default;
                        serde_json::to_value(&default).map_err(|e| {
                            anofox_ml_core::RustMlError::Serialization(e.to_string())
                        })
                    }
                )*
                other => Err(anofox_ml_core::RustMlError::InvalidParameter(format!(
                    "unknown estimator kind '{other}'"
                ))),
            }
        }

        /// All registered kinds.
        pub fn kinds() -> Vec<&'static str> {
            let mut kinds = Vec::new();
            $(
                $(#[$meta])*
                kinds.push(stringify!($kind));
            )*
            kinds
        }
    };

    (@cap $kind:ident, labels, $m:ident, $body:expr) => {
        fn labels(&self) -> anofox_ml_core::Result<$crate::model::Out> {
            let $m = self.0.fitted.as_ref().ok_or_else(|| $crate::model::not_fitted(stringify!($kind)))?;
            Ok($crate::model::Out::from($body))
        }
    };
    (@cap $kind:ident, feature_importances, $m:ident, $body:expr) => {
        fn feature_importances(&self) -> anofox_ml_core::Result<$crate::model::Out> {
            let $m = self.0.fitted.as_ref().ok_or_else(|| $crate::model::not_fitted(stringify!($kind)))?;
            Ok($crate::model::Out::from($body))
        }
    };
    (@cap $kind:ident, warm_start, $c:ident, $prev:ident, $x:ident, $y:ident, $w:ident, $body:expr) => {
        #[allow(unused_variables)]
        fn warm_start<'a>(
            &mut self,
            $x: ndarray::ArrayView2<'a, f64>,
            $y: Option<ndarray::ArrayView1<'a, f64>>,
            $w: Option<ndarray::ArrayView1<'a, f64>>,
        ) -> anofox_ml_core::Result<()> {
            #[allow(non_snake_case, dead_code)]
            const K: &str = stringify!($kind);
            let $prev = self
                .0
                .fitted
                .clone()
                .ok_or_else(|| $crate::model::not_fitted(stringify!($kind)))?;
            let $c = &self.0.config;
            self.0.fitted = Some($body?);
            Ok(())
        }
    };
    (@cap $kind:ident, partial_fit, $c:ident, $prev:ident, $x:ident, $y:ident, $w:ident,
     $cl:ident, $body:expr) => {
        #[allow(unused_variables)]
        fn partial_fit<'a>(
            &mut self,
            $x: ndarray::ArrayView2<'a, f64>,
            $y: Option<ndarray::ArrayView1<'a, f64>>,
            $w: Option<ndarray::ArrayView1<'a, f64>>,
            $cl: Option<&[f64]>,
        ) -> anofox_ml_core::Result<()> {
            #[allow(non_snake_case, dead_code)]
            const K: &str = stringify!($kind);
            let $prev = self.0.fitted.clone();
            let $c = &self.0.config;
            self.0.fitted = Some($body?);
            Ok(())
        }
    };
    (@cap $kind:ident, fit_with_offset, $c:ident, $x:ident, $y:ident, $w:ident, $off:ident,
     $body:expr) => {
        #[allow(unused_variables)]
        fn fit_with_offset<'a>(
            &mut self,
            $x: ndarray::ArrayView2<'a, f64>,
            $y: Option<ndarray::ArrayView1<'a, f64>>,
            $w: Option<ndarray::ArrayView1<'a, f64>>,
            $off: ndarray::ArrayView1<'a, f64>,
        ) -> anofox_ml_core::Result<()> {
            #[allow(non_snake_case, dead_code)]
            const K: &str = stringify!($kind);
            let $c = &self.0.config;
            self.0.fitted = Some($body?);
            Ok(())
        }
    };
    (@cap $kind:ident, predict_with_offset, $m:ident, $x:ident, $off:ident, $body:expr) => {
        fn predict_with_offset(
            &self,
            $x: ndarray::ArrayView2<f64>,
            $off: ndarray::ArrayView1<f64>,
        ) -> anofox_ml_core::Result<$crate::model::Out> {
            let $m = self.0.fitted.as_ref().ok_or_else(|| $crate::model::not_fitted(stringify!($kind)))?;
            Ok($crate::model::Out::from($body?))
        }
    };
    (@cap $kind:ident, fit_with_validation, $c:ident, $x:ident, $y:ident, $w:ident,
     $xv:ident, $yv:ident, $wv:ident, $body:expr) => {
        #[allow(unused_variables)]
        fn fit_with_validation<'a>(
            &mut self,
            $x: ndarray::ArrayView2<'a, f64>,
            $y: Option<ndarray::ArrayView1<'a, f64>>,
            $w: Option<ndarray::ArrayView1<'a, f64>>,
            $xv: ndarray::ArrayView2<'a, f64>,
            $yv: Option<ndarray::ArrayView1<'a, f64>>,
            $wv: Option<ndarray::ArrayView1<'a, f64>>,
        ) -> anofox_ml_core::Result<()> {
            #[allow(non_snake_case, dead_code)]
            const K: &str = stringify!($kind);
            let $c = &self.0.config;
            self.0.fitted = Some($body?);
            Ok(())
        }
    };
    (@cap $kind:ident, embedding, $m:ident, $body:expr) => {
        fn embedding(&self) -> anofox_ml_core::Result<$crate::model::Out> {
            let $m = self.0.fitted.as_ref().ok_or_else(|| $crate::model::not_fitted(stringify!($kind)))?;
            Ok($crate::model::Out::from($body))
        }
    };
    (@cap $kind:ident, kneighbors, $m:ident, $x:ident, $k:ident, $body:expr) => {
        fn kneighbors(
            &self,
            $x: ndarray::ArrayView2<f64>,
            $k: Option<usize>,
        ) -> anofox_ml_core::Result<(ndarray::Array2<f64>, ndarray::Array2<usize>)> {
            let $m = self.0.fitted.as_ref().ok_or_else(|| $crate::model::not_fitted(stringify!($kind)))?;
            $body
        }
    };
    (@cap $kind:ident, $cap:ident, $m:ident, $x:ident, $a:ident, $body:expr) => {
        fn $cap(
            &self,
            $x: ndarray::ArrayView2<f64>,
            $a: f64,
        ) -> anofox_ml_core::Result<$crate::model::Out> {
            let $m = self.0.fitted.as_ref().ok_or_else(|| $crate::model::not_fitted(stringify!($kind)))?;
            Ok($crate::model::Out::from($body?))
        }
    };
    (@cap $kind:ident, $cap:ident, $m:ident, $x:ident, $body:expr) => {
        fn $cap(&self, $x: ndarray::ArrayView2<f64>) -> anofox_ml_core::Result<$crate::model::Out> {
            let $m = self.0.fitted.as_ref().ok_or_else(|| $crate::model::not_fitted(stringify!($kind)))?;
            Ok($crate::model::Out::from($body?))
        }
    };
}

pub(crate) use registry;
