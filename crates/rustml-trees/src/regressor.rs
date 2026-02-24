use ndarray::{Array1, Array2};
use rustml_core::{Fit, Float, Predict, Result, RustMlError};

use crate::node::TreeNode;
use crate::split::{compute_impurity, find_best_split, leaf_value, SplitCriterion};

/// Decision tree regressor parameters (unfitted state).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DecisionTreeRegressor {
    pub max_depth: Option<usize>,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
}

impl DecisionTreeRegressor {
    /// Create a new `DecisionTreeRegressor` with sensible defaults.
    pub fn new() -> Self {
        Self {
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
        }
    }

    /// Set the maximum depth of the tree.
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set the minimum number of samples required to split a node.
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split;
        self
    }

    /// Set the minimum number of samples required in a leaf node.
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }
}

impl Default for DecisionTreeRegressor {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted decision tree regressor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedDecisionTreeRegressor<F: Float> {
    tree: TreeNode<F>,
    n_features: usize,
}

impl<F: Float> Fit<F> for DecisionTreeRegressor {
    type Fitted = FittedDecisionTreeRegressor<F>;

    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Self::Fitted> {
        if x.nrows() != y.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but y has {} elements",
                x.nrows(),
                y.len()
            )));
        }
        if x.is_empty() {
            return Err(RustMlError::EmptyInput("training data is empty".into()));
        }

        let indices: Vec<usize> = (0..x.nrows()).collect();
        let tree = build_tree(
            x,
            y,
            &indices,
            0,
            self.max_depth,
            self.min_samples_split,
            self.min_samples_leaf,
        );

        Ok(FittedDecisionTreeRegressor {
            tree,
            n_features: x.ncols(),
        })
    }
}

impl<F: Float> Predict<F> for FittedDecisionTreeRegressor<F> {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }

        let predictions: Vec<F> = x
            .rows()
            .into_iter()
            .map(|row| self.tree.predict_one(row.as_slice().unwrap()))
            .collect();

        Ok(Array1::from_vec(predictions))
    }
}

impl<F: Float> FittedDecisionTreeRegressor<F> {
    /// Feature importances (normalized to sum to 1).
    pub fn feature_importances(&self) -> Array1<F> {
        let n_samples = tree_n_samples(&self.tree);
        let raw = self.tree.feature_importances(self.n_features, n_samples);
        let sum: F = raw.iter().copied().fold(F::zero(), |a, b| a + b);
        if sum > F::zero() {
            Array1::from_vec(raw.into_iter().map(|v| v / sum).collect())
        } else {
            Array1::zeros(self.n_features)
        }
    }

    pub fn tree(&self) -> &TreeNode<F> {
        &self.tree
    }
}

fn build_tree<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    indices: &[usize],
    depth: usize,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
) -> TreeNode<F> {
    let n_samples = indices.len();
    let impurity = compute_impurity(y, indices, SplitCriterion::Mse);

    let should_stop = n_samples < min_samples_split
        || max_depth.is_some_and(|d| depth >= d)
        || impurity < F::from_f64(1e-15).unwrap();

    if should_stop {
        let value = leaf_value(y, indices, SplitCriterion::Mse);
        return TreeNode::Leaf {
            value,
            n_samples,
            class_counts: None,
        };
    }

    match find_best_split(x, y, indices, SplitCriterion::Mse, min_samples_leaf) {
        Some(split) => {
            let left = build_tree(
                x,
                y,
                &split.left_indices,
                depth + 1,
                max_depth,
                min_samples_split,
                min_samples_leaf,
            );
            let right = build_tree(
                x,
                y,
                &split.right_indices,
                depth + 1,
                max_depth,
                min_samples_split,
                min_samples_leaf,
            );

            TreeNode::Split {
                feature_index: split.feature_index,
                threshold: split.threshold,
                left: Box::new(left),
                right: Box::new(right),
                n_samples,
                impurity,
            }
        }
        None => {
            let value = leaf_value(y, indices, SplitCriterion::Mse);
            TreeNode::Leaf {
                value,
                n_samples,
                class_counts: None,
            }
        }
    }
}

fn tree_n_samples<F: Float>(node: &TreeNode<F>) -> usize {
    match node {
        TreeNode::Leaf { n_samples, .. } => *n_samples,
        TreeNode::Split { n_samples, .. } => *n_samples,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_simple_regression() {
        // y = 2*x, tree should learn perfect piecewise approximation
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let tree = DecisionTreeRegressor::default();
        let fitted = Fit::fit(&tree, &x, &y).unwrap();

        // Predict on training data — should be perfect with unlimited depth
        let preds = fitted.predict(&x).unwrap();
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(p, t, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_max_depth_regression() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![1.0, 2.0, 3.0, 4.0];

        let tree = DecisionTreeRegressor {
            max_depth: Some(1),
            ..Default::default()
        };
        let fitted = Fit::fit(&tree, &x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        // With depth 1: left leaf gets mean(1,2)=1.5, right gets mean(3,4)=3.5
        assert_abs_diff_eq!(preds[0], 1.5, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[1], 1.5, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[2], 3.5, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[3], 3.5, epsilon = 1e-10);
    }
}
