//! # RustML
//!
//! A scikit-learn-style machine learning library for Rust.
//!
//! ## Quick Start
//!
//! ```rust
//! use rustml::prelude::*;
//! use ndarray::array;
//!
//! // Fit a KNN classifier
//! let x_train = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
//! let y_train = array![0.0, 0.0, 1.0, 1.0];
//!
//! let knn = KnnClassifier { n_neighbors: 3, ..Default::default() };
//! let model = knn.fit(&x_train, &y_train).unwrap();
//!
//! let x_test = array![[0.5, 0.5], [2.5, 2.5]];
//! let predictions = model.predict(&x_test).unwrap();
//! ```

/// Core traits and types.
pub mod core {
    pub use rustml_core::*;
}

/// Evaluation metrics.
pub mod metrics {
    pub use rustml_metrics::*;
}

/// Feature preprocessing (scalers).
pub mod preprocessing {
    pub use rustml_preprocessing::*;
}

/// K-nearest neighbors algorithms.
pub mod neighbors {
    pub use rustml_neighbors::*;
}

/// Decision tree algorithms.
pub mod trees {
    pub use rustml_trees::*;
}

/// Convenient prelude importing the most commonly used items.
pub mod prelude {
    pub use rustml_core::{Fit, FitUnsupervised, Float, InverseTransform, Predict, Transform};

    pub use rustml_metrics::{
        accuracy_score, confusion_matrix, f1_score, mae, mse, precision, r2_score, recall,
    };

    pub use rustml_preprocessing::{MinMaxScaler, StandardScaler};

    pub use rustml_neighbors::{DistanceMetric, KnnClassifier, KnnRegressor, WeightFunction};

    pub use rustml_trees::{DecisionTreeClassifier, DecisionTreeRegressor, SplitCriterion};
}
