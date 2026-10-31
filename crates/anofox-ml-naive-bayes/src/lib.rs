//! Naive Bayes classifiers.
//!
//! This crate implements several Naive Bayes algorithms:
//!
//! - [`GaussianNB`] — assumes features follow a normal distribution within each class.
//! - [`MultinomialNB`] — for count-based or TF-IDF features (non-negative values).
//! - [`BernoulliNB`] — for binary/boolean features, with automatic binarization.
//! - [`ComplementNB`] — complement-class weights for imbalanced text data.
//! - [`CategoricalNB`] — for integer-encoded categorical features.
//!
//! All models mirror scikit-learn: `predict_log_proba` is computed in log
//! space as `jll − logsumexp(jll)`, every model supports `partial_fit` and
//! per-sample weights, and labels outside the declared classes are an error.
//!
//! # Examples
//!
//! ```
//! use ndarray::array;
//! use anofox_ml_core::{Fit, Predict};
//! use anofox_ml_naive_bayes::GaussianNB;
//!
//! // Two well-separated classes
//! let x_train = array![
//!     [1.0, 1.0],
//!     [1.1, 0.9],
//!     [0.9, 1.1],
//!     [10.0, 10.0],
//!     [10.1, 9.9],
//!     [9.9, 10.1]
//! ];
//! let y_train = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
//!
//! let nb = GaussianNB::new();
//! let fitted = Fit::fit(&nb, &x_train, &y_train).unwrap();
//!
//! let x_test = array![[1.0, 1.0], [10.0, 10.0]];
//! let preds = fitted.predict(&x_test).unwrap();
//! assert!((preds[0] - 0.0_f64).abs() < 1e-10);
//! assert!((preds[1] - 1.0_f64).abs() < 1e-10);
//! ```

mod bernoulli_nb;
mod categorical_nb;
mod common;
mod complement_nb;
mod discrete;
mod gaussian_nb;
mod multinomial_nb;

pub use bernoulli_nb::{BernoulliNB, FittedBernoulliNB};
pub use categorical_nb::{CategoricalNB, FittedCategoricalNB, MinCategories};
pub use complement_nb::{ComplementNB, FittedComplementNB};
pub use gaussian_nb::{FittedGaussianNB, GaussianNB};
pub use multinomial_nb::{FittedMultinomialNB, MultinomialNB};
