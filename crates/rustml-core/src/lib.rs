//! Core traits and types for the RustML machine learning library.
//!
//! This crate defines the foundational traits that all RustML estimators and
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
//! use rustml_core::{Fit, Predict, FitUnsupervised, Transform, Float};
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

pub mod column_transformer;
pub mod error;
pub mod feature_union;
pub mod float;
pub mod function_transformer;
pub mod halving;
pub mod inspection;
pub mod multi_output;
pub mod persistence;
pub mod pipeline;
pub mod traits;
pub mod utils;

pub use column_transformer::{ColumnSelector, ColumnTransformer, Remainder};
pub use error::{Result, RustMlError};
pub use float::Float;
pub use feature_union::{FeatureUnion, FittedFeatureUnion};
pub use function_transformer::FunctionTransformer;
pub use halving::{halving_grid_search_cv, halving_random_search_cv, HalvingResult};
pub use inspection::{permutation_importance, PermutationImportance};
pub use multi_output::{FittedMultiOutputRegressor, MultiOutputRegressor};
pub use pipeline::{FitPredict, FitTransform, FittedPipeline, Pipeline, PredictStep, TransformStep};
pub use traits::{
    Fit, FitUnsupervised, FitWeighted, InverseTransform, Predict, PredictProba, Transform,
};
pub use utils::{
    cross_val_predict, cross_val_score, cross_val_score_stratified, cross_validate, grid_search_cv,
    group_k_fold, k_fold, learning_curve, leave_one_out, leave_p_out, randomized_search_cv,
    repeated_k_fold, repeated_stratified_k_fold, shuffle_split, stratified_k_fold,
    stratified_shuffle_split, time_series_split, train_test_split, validation_curve,
    CrossValidateResult, GridSearchResult,
};
