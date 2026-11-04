//! Classical regression models wrapping the `anofox-regression` crate.
//!
//! Provides OLS, Ridge, Lasso, Elastic Net, WLS, Quantile, Isotonic, and GLM
//! (Poisson, Binomial) regressors that implement the anofox-ml `Fit` / `Predict`
//! type-state pattern.
//!
//! Cross-validated variants select the regularization strength like sklearn:
//! `RidgeCrossValidated` by efficient leave-one-out from one SVD (or k-fold),
//! `LassoCrossValidated` / `ElasticNetCrossValidated` along a warm-started
//! coordinate-descent path over a data-driven alpha grid, scored by pooled
//! k-fold MSE.
//!
//! All models operate on `f64` (not generic over `Float`) because the
//! underlying `anofox-regression` crate only supports `f64`.

pub mod bayesian_ridge;
pub mod convert;
pub mod elastic_net;
pub mod elastic_net_cv;
pub mod glm;
pub mod huber;
pub mod isotonic;
pub mod kernel_ridge;
pub mod lars;
pub mod lasso;
pub mod lasso_cv;
pub mod logistic;
pub mod ols;
pub mod omp;
pub mod quantile;
pub mod ridge;
pub mod ridge_cv;
pub mod robust;
pub mod summary;
pub mod transformed_target;
pub mod tweedie;
pub mod wls;

pub use bayesian_ridge::{ARDRegression, BayesianRidge, FittedARDRegression, FittedBayesianRidge};
pub use elastic_net::{ElasticNetRegressor, FittedElasticNetRegressor};
pub use elastic_net_cv::{ElasticNetCrossValidated, FittedElasticNetCrossValidated};
pub use glm::{
    BinomialLink, BinomialRegressor, FittedBinomialRegressor, FittedPoissonRegressor,
    PoissonRegressor,
};
pub use huber::{FittedHuberRegressor, HuberRegressor, HuberScale};
pub use isotonic::{FittedIsotonicRegressor, IsotonicRegressor, OutOfBounds};
pub use kernel_ridge::{FittedKernelRidge, KernelRidge};
pub use lars::{FittedLars, FittedLassoLarsIC, IcCriterion, Lars, LassoLarsIC};
pub use lasso::{FittedLassoRegressor, LassoRegressor};
pub use lasso_cv::{FittedLassoCrossValidated, LassoCrossValidated};
pub use logistic::{ClassWeight, FittedLogisticRegressor, LogisticRegressor, MultiClass};
pub use ols::{FittedOlsRegressor, OlsRegressor};
pub use omp::{FittedOrthogonalMatchingPursuit, OrthogonalMatchingPursuit};
pub use quantile::{FittedQuantileRegressor, QuantileRegressor};
pub use ridge::{FittedRidgeRegressor, RidgeRegressor};
pub use ridge_cv::{FittedRidgeCrossValidated, RidgeCrossValidated};
pub use robust::{
    FittedRansacRegressor, FittedTheilSenRegressor, RansacRegressor, TheilSenRegressor,
};
pub use summary::{
    InterceptSummary, IntervalKind, PredictionInterval, RegressionSummary, StatisticKind,
};
pub use transformed_target::{
    FittedTransformedTargetRegressor, TargetTransform, TransformedTargetRegressor,
};
pub use tweedie::{
    gamma_regressor, FittedGammaRegressor, FittedTweedieRegressor, GammaRegressor, TweedieRegressor,
};
pub use wls::{FittedWlsRegressor, WlsRegressor};
