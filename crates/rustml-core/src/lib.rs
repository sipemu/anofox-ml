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

pub mod error;
pub mod float;
pub mod pipeline;
pub mod traits;
pub mod utils;

pub use error::{Result, RustMlError};
pub use float::Float;
pub use pipeline::{FitPredict, FitTransform, FittedPipeline, Pipeline, PredictStep, TransformStep};
pub use traits::{Fit, FitUnsupervised, InverseTransform, Predict, Transform};
pub use utils::{
    cross_val_score, cross_val_score_stratified, grid_search_cv, stratified_k_fold,
    train_test_split, GridSearchResult,
};
