//! Optuna-style hyperparameter search for anofox-ml estimators, built on
//! [optuna/rustuna](https://github.com/optuna/rustuna).
//!
//! An estimator is named by its registry `kind` (see `anofox-ml-registry`);
//! the [`SearchSpace`] maps its JSON parameters to ranges, and every trial is
//! scored by cross-validation with a named [`Metric`]. Samplers: TPE
//! (default) and random, both seeded.
//!
//! ```
//! use ndarray::{Array1, Array2};
//! use serde_json::json;
//!
//! let x = Array2::from_shape_fn((80, 3), |(i, j)| ((i * (j + 2)) % 13) as f64);
//! let y = Array1::from_shape_fn(80, |i| 2.0 * x[[i, 0]] - x[[i, 1]]);
//! let options = anofox_ml_tune::TuneOptions {
//!     n_trials: 10,
//!     cv: 3,
//!     metric: Some("rmse".into()),
//!     params: json!({"n_estimators": 20}),
//!     ..Default::default()
//! };
//! let space = json!({
//!     "max_depth": {"int": [2, 8]},
//!     // enum parameters take their serde form; null = all features
//!     "max_features": {"categorical": [{"Fraction": 0.5}, "Sqrt", null]},
//! });
//! let result =
//!     anofox_ml_tune::tune("RandomForestRegressor", &space, x.view(), y.view(), &options).unwrap();
//! println!("{} = {:.3} with {}", result.metric, result.best_score, result.best_params);
//! assert_eq!(result.trials.len(), 10);
//! assert!(result.trials.iter().all(|t| t.error.is_none()));
//! ```
//!
//! Multi-objective searches add `objectives` (further metrics, `fit_time`,
//! `model_size`) and report the Pareto front; the NSGA-II sampler is
//! available for them. Not yet available: pruning (rustuna has no pruners
//! yet). This crate is not published while rustuna is only available from
//! git.

mod metric;
mod space;
mod tune;

pub use metric::{is_classifier, Metric};
pub use space::{Param, SearchSpace};
pub use tune::{tune, tune_with_splits, Objective, Sampler, TrialRecord, TuneOptions, TuneResult};
