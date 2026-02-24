pub mod distance;
pub mod knn_classifier;
pub mod knn_regressor;

pub use distance::DistanceMetric;
pub use knn_classifier::{FittedKnnClassifier, KnnClassifier, WeightFunction};
pub use knn_regressor::{FittedKnnRegressor, KnnRegressor};
