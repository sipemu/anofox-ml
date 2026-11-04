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
//! use anofox_ml_metrics::{accuracy_score, mse};
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
pub mod classification_extended;
pub mod classification_extra;
pub mod clustering;
pub mod clustering_extra;
pub mod curves;
pub mod intervals;
mod labels;
pub mod regression;
pub mod regression_extended;
pub mod regression_extra;

pub use classification::{
    accuracy_score, accuracy_score_with, confusion_matrix, confusion_matrix_with, f1_per_class,
    f1_score, f1_score_avg, f1_score_with, fbeta_score_with, precision, precision_per_class,
    precision_recall_fscore_support, precision_recall_fscore_support_per_class, precision_score,
    precision_score_with, recall, recall_per_class, recall_score, recall_score_with, Average,
    ClassificationOptions, PerClassScores, ZeroDivision,
};
pub use classification_extended::{
    average_precision_score, matthews_corrcoef, roc_auc_score, roc_auc_score_multiclass,
    roc_auc_score_ovr_per_class, roc_auc_score_with, MultiClass,
};
pub use classification_extra::{
    balanced_accuracy_score, cohen_kappa_score, log_loss, log_loss_multiclass, log_loss_with,
};
pub use clustering::silhouette_score;
pub use clustering_extra::{adjusted_rand_score, normalized_mutual_info_score};
pub use intervals::{coverage_score, interval_score, mean_interval_width};
pub use curves::{brier_score_loss, brier_score_loss_with, precision_recall_curve, roc_curve};
pub use regression::{mae, mse, r2_score};
pub use regression_extended::{
    explained_variance_score, max_error, mean_absolute_percentage_error,
};
pub use regression_extra::{mean_squared_log_error, median_absolute_error};
