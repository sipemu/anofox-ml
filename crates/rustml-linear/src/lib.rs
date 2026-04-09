//! SGD-based linear models.
//!
//! Provides [`SgdClassifier`] and [`SgdRegressor`] — linear models trained with
//! stochastic gradient descent. These scale well to large datasets and support
//! multiple loss functions and regularization options.

pub mod sgd_classifier;
pub mod sgd_common;
pub mod sgd_regressor;

pub use sgd_classifier::{ClassifierLoss, FittedSgdClassifier, SgdClassifier};
pub use sgd_common::{LearningRate, Penalty as SgdPenalty};
pub use sgd_regressor::{FittedSgdRegressor, RegressorLoss, SgdRegressor};
