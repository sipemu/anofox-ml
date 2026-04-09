//! Classical regression models wrapping the `anofox-regression` crate.
//!
//! Provides OLS, Ridge, Lasso, Elastic Net, WLS, Quantile, Isotonic, and GLM
//! (Poisson, Binomial) regressors that implement the rustml [`Fit`] / [`Predict`]
//! type-state pattern.
//!
//! Cross-validated variants (`RidgeCrossValidated`, `LassoCrossValidated`,
//! `ElasticNetCrossValidated`) automatically select the best regularization
//! parameters using k-fold cross-validation.
//!
//! All models operate on `f64` (not generic over `Float`) because the
//! underlying `anofox-regression` crate only supports `f64`.

pub mod convert;
pub mod elastic_net;
pub mod elastic_net_cv;
pub mod glm;
pub mod huber;
pub mod isotonic;
pub mod lasso;
pub mod lasso_cv;
pub mod logistic;
pub mod ols;
pub mod quantile;
pub mod ridge;
pub mod ridge_cv;
pub mod wls;

pub use elastic_net::{ElasticNetRegressor, FittedElasticNetRegressor};
pub use elastic_net_cv::{ElasticNetCrossValidated, FittedElasticNetCrossValidated};
pub use glm::{BinomialRegressor, FittedBinomialRegressor, FittedPoissonRegressor, PoissonRegressor};
pub use huber::{FittedHuberRegressor, HuberRegressor};
pub use isotonic::{FittedIsotonicRegressor, IsotonicRegressor};
pub use lasso::{FittedLassoRegressor, LassoRegressor};
pub use lasso_cv::{FittedLassoCrossValidated, LassoCrossValidated};
pub use logistic::{FittedLogisticRegressor, LogisticRegressor};
pub use ols::{FittedOlsRegressor, OlsRegressor};
pub use quantile::{FittedQuantileRegressor, QuantileRegressor};
pub use ridge::{FittedRidgeRegressor, RidgeRegressor};
pub use ridge_cv::{FittedRidgeCrossValidated, RidgeCrossValidated};
pub use wls::{FittedWlsRegressor, WlsRegressor};
