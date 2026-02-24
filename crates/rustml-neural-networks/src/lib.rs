//! Multi-layer perceptron (MLP) neural networks for classification and regression.
//!
//! This crate provides [`MlpClassifier`] and [`MlpRegressor`], feedforward neural
//! networks trained with backpropagation and gradient descent.
//!
//! # Examples
//!
//! ```
//! use rustml_neural_networks::MlpClassifier;
//! use rustml_core::{Fit, Predict};
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
pub mod solver;
pub mod utils;

pub use activation::Activation;
pub use mlp_classifier::{FittedMlpClassifier, MlpClassifier};
pub use mlp_regressor::{FittedMlpRegressor, MlpRegressor};
pub use solver::Solver;
