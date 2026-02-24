//! CART decision tree classifiers and regressors.
//!
//! This crate provides [`DecisionTreeClassifier`] and [`DecisionTreeRegressor`],
//! implementing the Classification and Regression Trees (CART) algorithm. Trees
//! support configurable split criteria ([`SplitCriterion`]: Gini, Entropy, MSE),
//! maximum depth, minimum samples per split, and minimum samples per leaf.
//!
//! # Examples
//!
//! ```
//! use ndarray::array;
//! use rustml_core::{Fit, Predict};
//! use rustml_trees::DecisionTreeClassifier;
//!
//! let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
//! let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
//!
//! let tree = DecisionTreeClassifier::new().with_max_depth(Some(3));
//! let fitted = Fit::fit(&tree, &x, &y).unwrap();
//!
//! let preds = fitted.predict(&array![[1.5], [5.5]]).unwrap();
//! assert!((preds[0] - 0.0_f64).abs() < 1e-10);
//! assert!((preds[1] - 1.0_f64).abs() < 1e-10);
//! ```

pub mod classifier;
pub mod node;
pub mod regressor;
pub mod split;

pub use classifier::{DecisionTreeClassifier, FittedDecisionTreeClassifier};
pub use node::TreeNode;
pub use regressor::{DecisionTreeRegressor, FittedDecisionTreeRegressor};
pub use split::SplitCriterion;
