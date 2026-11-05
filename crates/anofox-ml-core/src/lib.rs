//! Core traits and types for the anofox-ml machine learning library.
//!
//! This crate defines the foundational traits that all anofox-ml estimators and
//! transformers implement. It uses a **type-state pattern**: calling [`Fit::fit`]
//! or [`FitUnsupervised::fit`] on an unfitted configuration struct returns a
//! distinct *fitted* type that implements [`Predict`] or [`Transform`]. This
//! makes it a compile-time error to call `predict` on an unfitted model.
//!
//! The crate also provides the [`Float`] trait (a unified bound for `f32`/`f64`),
//! error types, train/test splitting utilities, and a [`Pipeline`] for chaining
//! transformers with an estimator.
//!
//! # Examples
//!
//! ```
//! use anofox_ml_core::{Fit, Predict, FitUnsupervised, Transform, Float};
//!
//! // The type-state pattern in action:
//! // 1. `Fit` takes an unfitted config and returns a `Fitted` type.
//! // 2. Only the `Fitted` type implements `Predict`.
//! fn example_trait_bounds<F, M, FM>(model: &M, x: &ndarray::Array2<F>, y: &ndarray::Array1<F>)
//! where
//!     F: Float,
//!     M: Fit<F, Fitted = FM>,
//!     FM: Predict<F>,
//! {
//!     let fitted = model.fit(x, y).unwrap();
//!     let _predictions = fitted.predict(x).unwrap();
//! }
//! ```

pub mod class_weight;
pub mod column_transformer;
pub mod dummy;
pub mod error;
pub mod feature_union;
pub mod float;
pub mod function_transformer;
pub mod halving;
pub mod inspection;
pub mod labels;
pub mod multi_output;
pub mod optim;
pub mod persistence;
pub mod pipeline;
pub mod sparse;
pub mod splitters;
pub mod traits;
pub mod utils;
pub mod validation;

pub use class_weight::{
    check_sample_weight, compute_sample_weight, effective_sample_weight, ClassWeight,
};
pub use column_transformer::{ColumnSelector, ColumnTransformer, Remainder};
pub use dummy::{
    DummyClassifier, DummyClassifierStrategy, DummyRegressor, DummyRegressorStrategy,
    FittedDummyClassifier, FittedDummyRegressor,
};
pub use error::{Result, RustMlError};
pub use feature_union::{FeatureUnion, FittedFeatureUnion};
pub use float::{as_f64, from_f64, Float};
pub use function_transformer::FunctionTransformer;
pub use halving::{
    halving_grid_search_cv, halving_grid_search_cv_with_splitter, halving_random_search_cv,
    HalvingResult, DEFAULT_HALVING_CV,
};
pub use inspection::{permutation_importance, PermutationImportance};
pub use labels::{encode_labels, label_index, sorted_unique_labels};
pub use multi_output::{
    ClassifierChain, FittedClassifierChain, FittedMultiOutputClassifier,
    FittedMultiOutputRegressor, FittedRegressorChain, MultiOutputClassifier, MultiOutputRegressor,
    RegressorChain,
};
pub use optim::{minimize_lbfgsb, LbfgsbOptions, LbfgsbResult, LbfgsbStatus};
pub use pipeline::{
    FitPredict, FitTransform, FittedPipeline, Pipeline, PredictStep, Supervised, TransformStep,
};
pub use sparse::CsrMatrix;
pub use splitters::{
    buffered_leave_one_out, combinatorial_purged_cv, group_k_fold, group_k_fold_shuffled,
    group_shuffle_split, group_time_series_split, leave_one_group_out, leave_p_groups_out,
    purged_k_fold, quantile_strata, spatial_block_k_fold, spatial_buffer, stratified_group_k_fold,
    stratified_k_fold_continuous, time_series_split_with, CombinatorialPurgedCv, SpatialMetric,
    TimeSeriesSplitOptions, MAX_LEAVE_P_GROUPS_OUT_SPLITS,
};
pub use traits::{
    ClassifierScore, DecisionFunction, Fit, FitUnsupervised, FitUnsupervisedWeighted, FitWeighted,
    InverseTransform, PartialFit, PartialFitUnsupervised, Predict, PredictLogProba, PredictProba,
    RegressorScore, Transform,
};
pub use utils::{
    cross_val_predict, cross_val_predict_with_splits, cross_val_score, cross_val_score_stratified,
    cross_val_score_weighted_with_splits, cross_val_score_with_splits,
    cross_val_score_with_splits_par, cross_validate, cross_validate_weighted_with_splits,
    cross_validate_with_splits, default_cv_splits, grid_search_cv, grid_search_cv_with_splits,
    grid_search_cv_with_splits_par, k_fold, learning_curve, learning_curve_with_splits,
    leave_one_out, leave_p_out, looks_like_class_labels, randomized_search_cv,
    randomized_search_cv_with_splits, repeated_k_fold, repeated_stratified_k_fold, shuffle_split,
    shuffled_k_fold, stratified_k_fold, stratified_k_fold_ordered, stratified_shuffle_split,
    time_series_split, train_test_split, validation_curve, validation_curve_with_splits,
    CrossValidateResult, CvSplit, GridSearchResult, LearningCurveResult, ScorerFn,
    ValidationCurveResult, WeightedScorerFn,
};
pub use validation::{
    bootstrap_splits, bootstrap_splits_stratified, bootstrap_validate, nested_cross_validate,
    permutation_test_score, BootstrapValidation, NestedCvResult, PermutationTestResult,
    MAX_NO_INFORMATION_PAIRS,
};
