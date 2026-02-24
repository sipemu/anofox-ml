use rustml_core::Float;

/// A node in a decision tree.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub enum TreeNode<F: Float> {
    /// Internal node: split on a feature at a threshold.
    Split {
        feature_index: usize,
        threshold: F,
        left: Box<TreeNode<F>>,
        right: Box<TreeNode<F>>,
        /// Number of training samples that reached this node.
        n_samples: usize,
        /// Impurity at this node before splitting.
        impurity: F,
    },
    /// Leaf node: predict a value.
    Leaf {
        value: F,
        /// Number of training samples in this leaf.
        n_samples: usize,
        /// For classifiers: class distribution counts.
        class_counts: Option<Vec<(F, usize)>>,
    },
}

impl<F: Float> TreeNode<F> {
    /// Predict a single sample by traversing the tree.
    #[inline]
    pub fn predict_one(&self, features: &[F]) -> F {
        match self {
            TreeNode::Leaf { value, .. } => *value,
            TreeNode::Split {
                feature_index,
                threshold,
                left,
                right,
                ..
            } => {
                if features[*feature_index] <= *threshold {
                    left.predict_one(features)
                } else {
                    right.predict_one(features)
                }
            }
        }
    }

    /// Compute feature importances by accumulating weighted impurity decreases.
    pub fn feature_importances(&self, n_features: usize, total_samples: usize) -> Vec<F> {
        let mut importances = vec![F::zero(); n_features];
        self.accumulate_importances(&mut importances, total_samples);
        importances
    }

    fn accumulate_importances(&self, importances: &mut [F], total_samples: usize) {
        if let TreeNode::Split {
            feature_index,
            left,
            right,
            n_samples,
            impurity,
            ..
        } = self
        {
            let left_samples = node_samples(left);
            let right_samples = node_samples(right);
            let left_impurity = node_impurity(left);
            let right_impurity = node_impurity(right);

            let n = num_traits::FromPrimitive::from_usize(*n_samples).unwrap_or(F::one());
            let nl = num_traits::FromPrimitive::from_usize(left_samples).unwrap_or(F::zero());
            let nr = num_traits::FromPrimitive::from_usize(right_samples).unwrap_or(F::zero());
            let total =
                num_traits::FromPrimitive::from_usize(total_samples).unwrap_or(F::one());

            // Weighted impurity decrease
            let decrease = (n / total)
                * (*impurity - (nl / n) * left_impurity - (nr / n) * right_impurity);

            importances[*feature_index] += decrease;

            left.accumulate_importances(importances, total_samples);
            right.accumulate_importances(importances, total_samples);
        }
    }
}

fn node_samples<F: Float>(node: &TreeNode<F>) -> usize {
    match node {
        TreeNode::Leaf { n_samples, .. } => *n_samples,
        TreeNode::Split { n_samples, .. } => *n_samples,
    }
}

fn node_impurity<F: Float>(node: &TreeNode<F>) -> F {
    match node {
        TreeNode::Leaf { .. } => F::zero(),
        TreeNode::Split { impurity, .. } => *impurity,
    }
}
