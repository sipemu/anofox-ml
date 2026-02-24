//! Classification and regression evaluation metrics.
//!
//! This crate provides functions for evaluating machine learning model
//! performance, including classification metrics ([`accuracy_score`],
//! [`precision_score`], [`recall_score`], [`f1_score_avg`],
//! [`confusion_matrix`]) and regression metrics ([`mse`], [`mae`],
//! [`r2_score`]).
//!
//! # Examples
//!
//! ```
//! use ndarray::array;
//! use rustml_metrics::{accuracy_score, mse};
//!
//! // Classification: compute accuracy
//! let y_true = array![0.0, 1.0, 1.0, 0.0];
//! let y_pred = array![0.0, 1.0, 0.0, 0.0];
//! let acc: f64 = accuracy_score(&y_true, &y_pred).unwrap();
//! assert!((acc - 0.75).abs() < 1e-10);
//!
//! // Regression: compute mean squared error
//! let actual = array![1.0, 2.0, 3.0];
//! let predicted = array![1.5, 2.5, 3.5];
//! let error: f64 = mse(&actual, &predicted).unwrap();
//! assert!((error - 0.25).abs() < 1e-10);
//! ```

pub mod classification;
pub mod regression;

pub use classification::{
    accuracy_score, confusion_matrix, f1_score, f1_score_avg, precision, precision_score, recall,
    recall_score, Average,
};
pub use regression::{mae, mse, r2_score};
