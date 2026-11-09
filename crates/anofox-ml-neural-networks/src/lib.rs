//! Multi-layer perceptron (MLP) neural networks for classification and regression.
//!
//! This crate provides [`MlpClassifier`] and [`MlpRegressor`], feedforward neural
//! networks trained with backpropagation. They mirror
//! `sklearn.neural_network.MLPClassifier` / `MLPRegressor`: Glorot-uniform
//! initialisation, L2 penalty `alpha/(2·n_batch)·Σ||W||²`, solvers `Adam`,
//! `Sgd` (momentum / Nesterov, constant / invscaling / adaptive learning
//! rate) and `Lbfgs`, the `n_iter_no_change`/`tol` stopping rule, optional
//! validation-based early stopping, `partial_fit` and warm starts, and
//! training diagnostics (`loss_curve`, `n_iter`, `best_loss`).
//!
//! # Examples
//!
//! ```
//! use anofox_ml_neural_networks::MlpClassifier;
//! use anofox_ml_core::{Fit, Predict};
//! use ndarray::array;
//!
//! let x = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
//! let y = array![0.0, 1.0, 1.0, 0.0]; // XOR
//!
//! let mlp = MlpClassifier {
//!     hidden_layer_sizes: vec![10, 10],
//!     learning_rate: 0.01,
//!     max_iter: 500,
//!     seed: 42,
//!     batch_size: None,
//!     alpha: 0.0,
//!     ..Default::default()
//! };
//!
//! let model = mlp.fit(&x, &y).unwrap();
//! let preds = model.predict(&x).unwrap();
//! assert_eq!(preds.len(), 4);
//! ```

pub mod activation;
pub mod mlp_classifier;
pub mod mlp_regressor;
pub mod network;
pub mod rbm;
pub mod solver;
mod trainer;
pub mod utils;

#[cfg(test)]
mod parity_tests;

pub use activation::Activation;
pub use mlp_classifier::{FittedMlpClassifier, MlpClassifier};
pub use mlp_regressor::{FittedMlpRegressor, MlpRegressor};
pub use rbm::{BernoulliRbm, FittedBernoulliRbm};
pub use solver::{LearningRateSchedule, Solver};
