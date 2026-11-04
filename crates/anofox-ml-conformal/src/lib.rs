//! Conformal prediction intervals for anofox-ml regressors.
//!
//! Wraps any regressor (`Fit` + `Predict`) and turns its point predictions
//! into intervals with a finite-sample coverage guarantee under
//! exchangeability, without distributional assumptions:
//!
//! - [`SplitConformalRegressor`] / [`conformalize`]: calibrate on held-out
//!   rows (one fit).
//! - [`CrossConformalRegressor`]: CV+, jackknife+ (leave-one-out splits),
//!   CV-minmax and the cheaper "base" variant (K + 1 fits, uses every row
//!   for both fitting and calibration).
//! - [`ConformalizedQuantileRegressor`] / [`conformalize_quantiles`]:
//!   conformalized quantile regression (CQR) on quantile models, for
//!   intervals that adapt to heteroscedastic noise.
//! - [`NormalizedConformalRegressor`] / [`conformalize_normalized`]: split
//!   conformal with residual-normalised scores (a second model learns the
//!   error scale).
//! - [`SplitConformalClassifier`] / [`conformalize_classifier`]: prediction
//!   sets for classifiers with the LAC, APS or RAPS score.
//!
//! The miscoverage `alpha` is chosen at prediction time (CQR: at fit time,
//! since the quantile models are trained for it). Results match MAPIE
//! (`mapie.regression`) to rounding.
//!
//! ```
//! use anofox_ml_conformal::{CrossConformalRegressor, CrossMethod};
//! use anofox_ml_core::Fit;
//! use anofox_ml_trees::DecisionTreeRegressor;
//! use ndarray::{Array1, Array2};
//!
//! let x = Array2::from_shape_fn((200, 1), |(i, _)| i as f64 / 20.0);
//! let y = Array1::from_shape_fn(200, |i| (i as f64 / 20.0).sin() + 0.1 * ((i * 37 % 11) as f64 - 5.0) / 5.0);
//! let tree = DecisionTreeRegressor { max_depth: Some(4), ..Default::default() };
//! let model = CrossConformalRegressor::new(tree)
//!     .with_cv(5)
//!     .with_method(CrossMethod::Plus)
//!     .fit(&x, &y)
//!     .unwrap();
//! let (lower, upper) = model.predict_interval(x.view(), 0.1).unwrap();
//! assert!(lower.iter().zip(&upper).all(|(l, u)| l <= u));
//! ```

mod classification;
mod cqr;
mod cross;
mod normalized;
mod quantile;
mod split;

pub use classification::{
    conformalize_classifier, conformalize_raps, ApsMode, ClassificationScore,
    FittedSplitConformalClassifier, RapsParams, SplitConformalClassifier,
};
pub use cqr::{
    conformalize_quantiles, ConformalizedQuantileRegressor, FittedConformalizedQuantileRegressor,
};
pub use cross::{
    CrossConformalRegressor, CrossMethod, FittedCrossConformalRegressor, PointPrediction,
};
pub use normalized::{
    conformalize_normalized, fit_residual_model, FittedNormalizedConformalRegressor,
    NormalizedConformalRegressor,
};
pub use quantile::conformal_quantile;
pub use split::{conformalize, FittedSplitConformalRegressor, SplitConformalRegressor};
