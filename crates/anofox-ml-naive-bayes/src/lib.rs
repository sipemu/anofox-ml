//! Naive Bayes classifiers.
//!
//! This crate implements several Naive Bayes algorithms:
//!
//! - [`GaussianNB`] — assumes features follow a normal distribution within each class.
//! - [`MultinomialNB`] — for count-based or TF-IDF features (non-negative values).
//! - [`BernoulliNB`] — for binary/boolean features, with automatic binarization.
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
mod gaussian_nb;
mod multinomial_nb;

pub use bernoulli_nb::{BernoulliNB, FittedBernoulliNB};
pub use gaussian_nb::{FittedGaussianNB, GaussianNB};
pub use multinomial_nb::{FittedMultinomialNB, MultinomialNB};
