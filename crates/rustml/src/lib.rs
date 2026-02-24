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

/// Feature preprocessing (scalers, PCA).
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

/// Ensemble methods (Random Forest).
pub mod ensemble {
    pub use rustml_ensemble::*;
}

/// Clustering algorithms (KMeans).
pub mod cluster {
    pub use rustml_cluster::*;
}

/// Naive Bayes classifiers.
pub mod naive_bayes {
    pub use rustml_naive_bayes::*;
}

/// Convenient prelude importing the most commonly used items.
pub mod prelude {
    pub use rustml_core::{
        cross_val_score, train_test_split, Fit, FitUnsupervised, FittedPipeline, Float,
        InverseTransform, Pipeline, Predict, Transform,
    };

    pub use rustml_metrics::{
        accuracy_score, confusion_matrix, f1_score, mae, mse, precision, r2_score, recall,
    };

    pub use rustml_preprocessing::{MinMaxScaler, Pca, StandardScaler};

    pub use rustml_neighbors::{DistanceMetric, KnnClassifier, KnnRegressor, WeightFunction};

    pub use rustml_trees::{DecisionTreeClassifier, DecisionTreeRegressor, SplitCriterion};

    pub use rustml_ensemble::{RandomForestClassifier, RandomForestRegressor};

    pub use rustml_cluster::KMeans;

    pub use rustml_naive_bayes::GaussianNB;
}
