//! Gaussian Naive Bayes classifier.
//!
//! This crate implements the Gaussian Naive Bayes algorithm, which assumes
//! that features within each class follow a normal distribution. It is
//! particularly effective for high-dimensional data and serves as a fast,
//! probabilistic baseline classifier.
//!
//! # Examples
//!
//! ```
//! use ndarray::array;
//! use rustml_core::{Fit, Predict};
//! use rustml_naive_bayes::GaussianNB;
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

mod gaussian_nb;

pub use gaussian_nb::{FittedGaussianNB, GaussianNB};
