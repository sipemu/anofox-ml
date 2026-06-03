//! Bagging, random forest, Extra-Trees, gradient boosting, and AdaBoost ensemble methods.
//!
//! This crate provides ensemble learning algorithms that combine multiple
//! decision trees for improved accuracy and robustness:
//!
//! - [`BaggingClassifier`] / [`BaggingRegressor`] -- bootstrap aggregating with
//!   the full feature set (no random feature subsampling).
//! - [`RandomForestClassifier`] / [`RandomForestRegressor`] -- bagging with
//!   bootstrap samples and optional random feature subsets.
//! - [`ExtraTreesClassifier`] / [`ExtraTreesRegressor`] -- extremely randomized
//!   trees that use random split thresholds and train on the full dataset
//!   (no bootstrap).
//! - [`GradientBoostingClassifier`] / [`GradientBoostingRegressor`] -- sequential
//!   boosting that fits each tree to the residuals of the ensemble.
//! - [`AdaBoostClassifier`] / [`AdaBoostRegressor`] -- adaptive boosting that
//!   re-weights samples to focus on hard examples.
//!
//! # Examples
//!
//! ```
//! use ndarray::array;
//! use rustml_core::{Fit, Predict};
//! use rustml_ensemble::RandomForestClassifier;
//!
//! let x = array![
//!     [1.0, 0.0],
//!     [2.0, 0.0],
//!     [3.0, 0.0],
//!     [10.0, 1.0],
//!     [11.0, 1.0],
//!     [12.0, 1.0]
//! ];
//! let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
//!
//! let rf = RandomForestClassifier::new(20)
//!     .with_max_depth(Some(3))
//!     .with_seed(42);
//! let fitted = Fit::fit(&rf, &x, &y).unwrap();
//!
//! let preds = fitted.predict(&x).unwrap();
//! assert!((preds[0] - 0.0_f64).abs() < 1e-10);
//! assert!((preds[5] - 1.0_f64).abs() < 1e-10);
//! ```

pub mod adaboost_classifier;
pub mod adaboost_regressor;
pub mod bagging_classifier;
pub mod bagging_regressor;
pub mod calibrated_classifier;
pub mod extra_trees_classifier;
pub mod extra_trees_regressor;
pub mod gradient_boosting_classifier;
pub mod hist_gradient_boosting;
pub mod lgbm;
pub mod gradient_boosting_regressor;
pub mod random_forest_classifier;
pub mod random_forest_regressor;
pub mod stacking_classifier;
pub mod stacking_regressor;
pub mod voting_classifier;
pub mod voting_regressor;

pub use adaboost_classifier::{AdaBoostClassifier, FittedAdaBoostClassifier};
pub use adaboost_regressor::{
    AdaBoostLoss, AdaBoostRegressor, FittedAdaBoostRegressor,
};
pub use bagging_classifier::{BaggingClassifier, FittedBaggingClassifier};
pub use bagging_regressor::{BaggingRegressor, FittedBaggingRegressor};
pub use calibrated_classifier::{CalibratedClassifierCV, CalibrationMethod};
pub use extra_trees_classifier::{ExtraTreesClassifier, FittedExtraTreesClassifier};
pub use extra_trees_regressor::{ExtraTreesRegressor, FittedExtraTreesRegressor};
pub use gradient_boosting_classifier::{
    FittedGradientBoostingClassifier, GradientBoostingClassifier,
};
pub use hist_gradient_boosting::{
    FittedHistGradientBoostingClassifier, FittedHistGradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
};
pub use lgbm::{
    BoostingType, FittedLgbmClassifier, FittedLgbmRegressor, LgbmClassWeight, LgbmClassifier,
    LgbmFitOptions, LgbmObjective, LgbmRegressor,
};
pub use gradient_boosting_regressor::{
    FittedGradientBoostingRegressor, GradientBoostingRegressor,
};
pub use random_forest_classifier::{FittedRandomForestClassifier, RandomForestClassifier};
pub use random_forest_regressor::{FittedRandomForestRegressor, RandomForestRegressor};
pub use stacking_classifier::{FittedStackingClassifier, StackingClassifier};
pub use stacking_regressor::StackingRegressor;
pub use voting_classifier::VotingClassifier;
pub use voting_regressor::VotingRegressor;
