use anofox_ml_core::Float;
use ndarray::ArrayView1;

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

    /// Predict a single sample given as an ndarray view of any layout/stride
    /// (e.g. a row of a Fortran-order matrix).
    #[inline]
    pub fn predict_view(&self, features: ArrayView1<'_, F>) -> F {
        match self.leaf_for_view(features) {
            TreeNode::Leaf { value, .. } => *value,
            TreeNode::Split { .. } => unreachable!("leaf_for_view always returns a leaf"),
        }
    }

    /// Return the leaf node reached by `features` (any layout/stride).
    #[inline]
    pub fn leaf_for_view(&self, features: ArrayView1<'_, F>) -> &TreeNode<F> {
        let mut node = self;
        loop {
            match node {
                TreeNode::Leaf { .. } => return node,
                TreeNode::Split {
                    feature_index,
                    threshold,
                    left,
                    right,
                    ..
                } => {
                    node = if features[*feature_index] <= *threshold {
                        left
                    } else {
                        right
                    };
                }
            }
        }
    }

    /// Number of leaves in this (sub)tree.
    pub fn n_leaves(&self) -> usize {
        match self {
            TreeNode::Leaf { .. } => 1,
            TreeNode::Split { left, right, .. } => left.n_leaves() + right.n_leaves(),
        }
    }

    /// Leaf index (pre-order numbering, left subtree before right, starting
    /// at 0) of the leaf reached by every row of `x` (any layout/stride).
    ///
    /// The numbering matches [`TreeNode::leaf_values_mut`], so the two can be
    /// combined to re-estimate leaf values after fitting (e.g. the Newton
    /// step of gradient boosting).
    pub fn apply(&self, x: &ndarray::Array2<F>) -> Vec<usize> {
        let mut out = vec![0usize; x.nrows()];
        let rows: Vec<usize> = (0..x.nrows()).collect();
        let mut next_leaf = 0usize;
        self.apply_rec(x, rows, &mut next_leaf, &mut out);
        out
    }

    fn apply_rec(
        &self,
        x: &ndarray::Array2<F>,
        rows: Vec<usize>,
        next_leaf: &mut usize,
        out: &mut [usize],
    ) {
        match self {
            TreeNode::Leaf { .. } => {
                for r in rows {
                    out[r] = *next_leaf;
                }
                *next_leaf += 1;
            }
            TreeNode::Split {
                feature_index,
                threshold,
                left,
                right,
                ..
            } => {
                let (l, r): (Vec<usize>, Vec<usize>) = rows
                    .into_iter()
                    .partition(|&i| x[[i, *feature_index]] <= *threshold);
                left.apply_rec(x, l, next_leaf, out);
                right.apply_rec(x, r, next_leaf, out);
            }
        }
    }

    /// Mutable references to all leaf values in pre-order (left before
    /// right), matching the numbering of [`TreeNode::apply`].
    pub fn leaf_values_mut(&mut self) -> Vec<&mut F> {
        let mut out = Vec::new();
        self.collect_leaf_values_mut(&mut out);
        out
    }

    fn collect_leaf_values_mut<'a>(&'a mut self, out: &mut Vec<&'a mut F>) {
        match self {
            TreeNode::Leaf { value, .. } => out.push(value),
            TreeNode::Split { left, right, .. } => {
                left.collect_leaf_values_mut(out);
                right.collect_leaf_values_mut(out);
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
            let total = num_traits::FromPrimitive::from_usize(total_samples).unwrap_or(F::one());

            // Weighted impurity decrease
            let decrease =
                (n / total) * (*impurity - (nl / n) * left_impurity - (nr / n) * right_impurity);

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
