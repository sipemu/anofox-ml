//! K-nearest neighbors classifiers and regressors with KD-tree acceleration.
//!
//! This crate provides [`KnnClassifier`] and [`KnnRegressor`], both of which
//! support configurable distance metrics ([`DistanceMetric`]) and weighting
//! strategies ([`WeightFunction`]). When Euclidean distance is selected, a
//! KD-tree is built at fit time for efficient O(log n) neighbor lookups;
//! other metrics use brute-force search.
//!
//! # Examples
//!
//! ```
//! use ndarray::array;
//! use rustml_core::{Fit, Predict};
//! use rustml_neighbors::KnnClassifier;
//!
//! let x_train = array![
//!     [0.0, 0.0],
//!     [1.0, 1.0],
//!     [2.0, 2.0],
//!     [10.0, 10.0],
//!     [11.0, 11.0],
//!     [12.0, 12.0]
//! ];
//! let y_train = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
//!
//! let knn = KnnClassifier::new(3);
//! let fitted = Fit::fit(&knn, &x_train, &y_train).unwrap();
//!
//! let x_test = array![[0.5, 0.5], [10.5, 10.5]];
//! let preds = fitted.predict(&x_test).unwrap();
//! assert!((preds[0] - 0.0_f64).abs() < 1e-10);
//! assert!((preds[1] - 1.0_f64).abs() < 1e-10);
//! ```

pub mod distance;
pub mod kdtree;
pub mod knn_classifier;
pub mod knn_regressor;
pub mod lof;

pub use distance::DistanceMetric;
pub use knn_classifier::{FittedKnnClassifier, KnnClassifier, WeightFunction};
pub use knn_regressor::{FittedKnnRegressor, KnnRegressor};
pub use lof::{FittedLocalOutlierFactor, LocalOutlierFactor};
