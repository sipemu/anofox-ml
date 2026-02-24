//! Random forest and gradient boosting ensemble methods.
//!
//! This crate provides ensemble learning algorithms that combine multiple
//! decision trees for improved accuracy and robustness:
//!
//! - [`RandomForestClassifier`] / [`RandomForestRegressor`] -- bagging with
//!   bootstrap samples and optional random feature subsets.
//! - [`GradientBoostingClassifier`] / [`GradientBoostingRegressor`] -- sequential
//!   boosting that fits each tree to the residuals of the ensemble.
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

pub mod gradient_boosting_classifier;
pub mod gradient_boosting_regressor;
pub mod random_forest_classifier;
pub mod random_forest_regressor;

pub use gradient_boosting_classifier::{
    FittedGradientBoostingClassifier, GradientBoostingClassifier,
};
pub use gradient_boosting_regressor::{
    FittedGradientBoostingRegressor, GradientBoostingRegressor,
};
pub use random_forest_classifier::{FittedRandomForestClassifier, RandomForestClassifier};
pub use random_forest_regressor::{FittedRandomForestRegressor, RandomForestRegressor};
