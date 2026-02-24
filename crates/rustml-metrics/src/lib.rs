pub mod classification;
pub mod regression;

pub use classification::{
    accuracy_score, confusion_matrix, f1_score, f1_score_avg, precision, precision_score, recall,
    recall_score, Average,
};
pub use regression::{mae, mse, r2_score};
