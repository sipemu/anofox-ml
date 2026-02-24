use ndarray::{Array1, Array2};
use rustml_core::{Fit, Float, Predict, Result, RustMlError};

use crate::node::TreeNode;
use crate::split::{
    compute_impurity, count_classes, find_best_split, leaf_value, SplitCriterion,
};

/// Decision tree classifier parameters (unfitted state).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DecisionTreeClassifier {
    pub max_depth: Option<usize>,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub criterion: SplitCriterion,
}

impl DecisionTreeClassifier {
    /// Create a new `DecisionTreeClassifier` with sensible defaults.
    pub fn new() -> Self {
        Self {
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            criterion: SplitCriterion::Gini,
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

    /// Set the split quality criterion.
    pub fn with_criterion(mut self, criterion: SplitCriterion) -> Self {
        self.criterion = criterion;
        self
    }
}

impl Default for DecisionTreeClassifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted decision tree classifier.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedDecisionTreeClassifier<F: Float> {
    tree: TreeNode<F>,
    n_features: usize,
}

impl<F: Float> Fit<F> for DecisionTreeClassifier {
    type Fitted = FittedDecisionTreeClassifier<F>;

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
        let params = TreeBuildParams {
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            criterion: self.criterion,
        };
        let tree = build_tree(x, y, &indices, 0, &params);

        Ok(FittedDecisionTreeClassifier {
            tree,
            n_features: x.ncols(),
        })
    }
}

impl<F: Float> Predict<F> for FittedDecisionTreeClassifier<F> {
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

impl<F: Float> FittedDecisionTreeClassifier<F> {
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

/// Bundled parameters for recursive tree building (avoids too many function args).
struct TreeBuildParams {
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    criterion: SplitCriterion,
}

fn build_tree<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    indices: &[usize],
    depth: usize,
    params: &TreeBuildParams,
) -> TreeNode<F> {
    let n_samples = indices.len();
    let impurity = compute_impurity(y, indices, params.criterion);

    // Check stopping criteria
    let should_stop = n_samples < params.min_samples_split
        || params.max_depth.is_some_and(|d| depth >= d)
        || impurity < F::from_f64(1e-15).unwrap();

    if should_stop {
        return make_leaf(y, indices, params.criterion);
    }

    match find_best_split(x, y, indices, params.criterion, params.min_samples_leaf) {
        Some(split) => {
            let left = build_tree(
                x,
                y,
                &split.left_indices,
                depth + 1,
                params,
            );
            let right = build_tree(
                x,
                y,
                &split.right_indices,
                depth + 1,
                params,
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
        None => make_leaf(y, indices, params.criterion),
    }
}

fn make_leaf<F: Float>(y: &Array1<F>, indices: &[usize], criterion: SplitCriterion) -> TreeNode<F> {
    let value = leaf_value(y, indices, criterion);
    let class_counts = match criterion {
        SplitCriterion::Gini | SplitCriterion::Entropy => Some(count_classes(y, indices)),
        SplitCriterion::Mse => None,
    };
    TreeNode::Leaf {
        value,
        n_samples: indices.len(),
        class_counts,
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
    fn test_simple_classification() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let tree = DecisionTreeClassifier::default();
        let fitted = Fit::fit(&tree, &x, &y).unwrap();

        let preds = fitted.predict(&array![[1.5], [5.5]]).unwrap();
        assert_abs_diff_eq!(preds[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_max_depth() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let tree = DecisionTreeClassifier {
            max_depth: Some(1),
            ..Default::default()
        };
        let fitted = Fit::fit(&tree, &x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        // With max_depth=1, should still separate the two classes
        assert_abs_diff_eq!(preds[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[3], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_feature_importances() {
        let x = array![[1.0, 100.0], [2.0, 200.0], [3.0, 300.0], [4.0, 400.0]];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let tree = DecisionTreeClassifier::default();
        let fitted = Fit::fit(&tree, &x, &y).unwrap();

        let importances = fitted.feature_importances();
        // Sum should be 1.0
        let sum: f64 = importances.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);
    }

    mod prop_tests {
        use super::*;
        use proptest::prelude::*;
        use std::collections::HashSet;

        /// Generate deterministic training data for classification.
        fn make_classification_data(
            n_samples: usize,
            n_features: usize,
            n_classes: usize,
            seed: u64,
        ) -> (Array2<f64>, Array1<f64>) {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let mut x_data = Vec::with_capacity(n_samples * n_features);
            let mut y_data = Vec::with_capacity(n_samples);

            for i in 0..n_samples {
                for j in 0..n_features {
                    let mut h = DefaultHasher::new();
                    seed.hash(&mut h);
                    (i as u64).hash(&mut h);
                    (j as u64).hash(&mut h);
                    let bits = h.finish();
                    let v = (bits as f64 / u64::MAX as f64) * 20.0 - 10.0;
                    x_data.push(v);
                }
                let mut h = DefaultHasher::new();
                seed.hash(&mut h);
                (i as u64).hash(&mut h);
                0xDEAD_BEEFu64.hash(&mut h);
                let label = (h.finish() % n_classes as u64) as f64;
                y_data.push(label);
            }

            let x = Array2::from_shape_vec((n_samples, n_features), x_data).unwrap();
            let y = Array1::from_vec(y_data);
            (x, y)
        }

        proptest! {
            #[test]
            fn tree_predictions_are_valid_labels(
                n_samples in 4..30usize,
                n_features in 1..5usize,
                seed in 0u64..1000,
            ) {
                let n_classes = 3;
                let (x, y) = make_classification_data(n_samples, n_features, n_classes, seed);

                // Collect unique training labels
                let train_labels: HashSet<u64> = y.iter()
                    .map(|&v| v.to_bits())
                    .collect();

                let tree = DecisionTreeClassifier::new()
                    .with_max_depth(Some(5));
                let fitted = Fit::fit(&tree, &x, &y).unwrap();
                let preds = fitted.predict(&x).unwrap();

                for (i, &p) in preds.iter().enumerate() {
                    prop_assert!(
                        train_labels.contains(&p.to_bits()),
                        "prediction {} at index {} is not a valid training label",
                        p, i
                    );
                }
            }

            #[test]
            fn tree_deterministic(seed in 0u64..1000) {
                let (x, y) = make_classification_data(20, 3, 3, seed);

                let tree = DecisionTreeClassifier::new()
                    .with_max_depth(Some(4));

                let fitted1 = Fit::fit(&tree, &x, &y).unwrap();
                let fitted2 = Fit::fit(&tree, &x, &y).unwrap();

                let preds1 = fitted1.predict(&x).unwrap();
                let preds2 = fitted2.predict(&x).unwrap();

                for (i, (&a, &b)) in preds1.iter().zip(preds2.iter()).enumerate() {
                    prop_assert!((a - b).abs() < 1e-15,
                        "non-deterministic prediction at index {}: {} vs {}", i, a, b);
                }
            }
        }
    }
}
