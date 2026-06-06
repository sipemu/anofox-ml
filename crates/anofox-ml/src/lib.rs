//! # anofox-ml
//!
//! A scikit-learn-style machine learning library for Rust.
//!
//! ## Quick Start
//!
//! ```rust
//! use anofox_ml::prelude::*;
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
    pub use anofox_ml_core::*;
}

/// Evaluation metrics.
pub mod metrics {
    pub use anofox_ml_metrics::*;
}

/// Feature preprocessing (scalers, PCA).
pub mod preprocessing {
    pub use anofox_ml_preprocessing::*;
}

/// K-nearest neighbors algorithms.
pub mod neighbors {
    pub use anofox_ml_neighbors::*;
}

/// Decision tree algorithms.
pub mod trees {
    pub use anofox_ml_trees::*;
}

/// Ensemble methods (Random Forest, Gradient Boosting, AdaBoost, ExtraTrees).
pub mod ensemble {
    pub use anofox_ml_ensemble::*;
}

/// Clustering algorithms (KMeans, DBSCAN).
pub mod cluster {
    pub use anofox_ml_cluster::*;
}

/// Naive Bayes classifiers.
pub mod naive_bayes {
    pub use anofox_ml_naive_bayes::*;
}

/// Linear and Quadratic Discriminant Analysis.
pub mod discriminant {
    pub use anofox_ml_discriminant::*;
}

/// Support Vector Machine classifiers.
pub mod svm {
    pub use anofox_ml_svm::*;
}

/// Neural network models (MLP).
pub mod neural_networks {
    pub use anofox_ml_neural_networks::*;
}

/// Classical regression models (OLS, Ridge, Elastic Net, GLM).
pub mod regression {
    pub use anofox_ml_regression::*;
}

/// SGD-based linear models (SGDClassifier, SGDRegressor).
pub mod linear {
    pub use anofox_ml_linear::*;
}

/// Data I/O utilities (CSV reading).
pub mod io {
    pub use anofox_ml_io::*;
}

/// Convenient prelude importing the most commonly used items.
pub mod prelude {
    pub use anofox_ml_core::{
        cross_val_predict, cross_val_score, cross_val_score_stratified, cross_validate,
        grid_search_cv, group_k_fold, k_fold, learning_curve, leave_one_out, leave_p_out,
        randomized_search_cv, repeated_k_fold, repeated_stratified_k_fold, shuffle_split,
        stratified_k_fold, stratified_shuffle_split, time_series_split, train_test_split,
        validation_curve, ColumnSelector, ColumnTransformer, CrossValidateResult, FeatureUnion,
        Fit, FitUnsupervised, FittedPipeline, Float, FunctionTransformer, GridSearchResult,
        InverseTransform, Pipeline, Predict, Remainder, Transform,
    };

    pub use anofox_ml_metrics::{
        accuracy_score, adjusted_rand_score, average_precision_score, balanced_accuracy_score,
        brier_score_loss, cohen_kappa_score, confusion_matrix, explained_variance_score, f1_score,
        f1_score_avg, log_loss, mae, matthews_corrcoef, max_error, mean_absolute_percentage_error,
        mean_squared_log_error, median_absolute_error, mse, normalized_mutual_info_score,
        precision, precision_recall_curve, precision_score, r2_score, recall, recall_score,
        roc_auc_score, roc_curve, silhouette_score, Average,
    };

    pub use anofox_ml_preprocessing::{
        BinStrategy, Binarizer, EncodeStrategy, ImputeStrategy, KBinsDiscretizer, KernelPca,
        KpcaKernel, LabelEncoder, MaxAbsScaler, MinMaxScaler, MutualInformationSelector, Nmf,
        NormType, Normalizer, OneHotEncoder, OrdinalEncoder, OutputDistribution, Pca,
        PolynomialFeatures, PowerTransformer, QuantileTransformer, Rfe, RobustScaler,
        SelectFromModel, SelectKBest, SequentialFeatureSelector, SimpleImputer, StandardScaler,
        TruncatedSvd, VarianceThreshold,
    };

    pub use anofox_ml_neighbors::{
        DistanceMetric, KnnClassifier, KnnRegressor, LocalOutlierFactor, WeightFunction,
    };

    pub use anofox_ml_trees::{
        ClassWeight, DecisionTreeClassifier, DecisionTreeRegressor, MaxFeatures, SplitCriterion,
    };

    pub use anofox_ml_ensemble::{
        AdaBoostClassifier, AdaBoostRegressor, BaggingClassifier, BaggingRegressor, BoostingType,
        CalibratedClassifierCV, CalibrationMethod, ExtraTreesClassifier, ExtraTreesRegressor,
        GradientBoostingClassifier, GradientBoostingRegressor, HistGradientBoostingClassifier,
        HistGradientBoostingRegressor, IsolationForest, LgbmClassWeight, LgbmClassifier,
        LgbmFitOptions, LgbmObjective, LgbmRegressor, RandomForestClassifier,
        RandomForestRegressor, StackingClassifier, StackingRegressor, VotingClassifier,
        VotingRegressor,
    };

    pub use anofox_ml_cluster::{
        AgglomerativeClustering, CovarianceType, Dbscan, GaussianMixture, KMeans, Linkage,
        MiniBatchKMeans,
    };

    pub use anofox_ml_naive_bayes::{BernoulliNB, GaussianNB, MultinomialNB};

    pub use anofox_ml_discriminant::{
        FittedLinearDiscriminantAnalysis, FittedQuadraticDiscriminantAnalysis,
        LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis,
    };

    pub use anofox_ml_svm::{LinearSvc, LinearSvr, NuSvc, NuSvr, OneClassSvm, Svc, SvmKernel, Svr};

    pub use anofox_ml_neural_networks::{MlpClassifier, MlpRegressor};

    pub use anofox_ml_regression::{
        ARDRegression, BayesianRidge, BinomialRegressor, ElasticNetCrossValidated,
        ElasticNetRegressor, GammaRegressor, HuberRegressor, IsotonicRegressor, KernelRidge,
        LassoCrossValidated, LassoRegressor, LogisticRegressor, OlsRegressor,
        OrthogonalMatchingPursuit, PoissonRegressor, QuantileRegressor, RansacRegressor,
        RidgeCrossValidated, RidgeRegressor, TheilSenRegressor, TransformedTargetRegressor,
        TweedieRegressor, WlsRegressor,
    };

    pub use anofox_ml_linear::{
        PassiveAggressiveClassifier, PassiveAggressiveRegressor, SgdClassifier, SgdRegressor,
    };

    pub use anofox_ml_text::{CountVectorizer, HashingVectorizer, TfidfVectorizer};

    pub use anofox_ml_core::persistence::{load_bincode, load_json, save_bincode, save_json};
}
