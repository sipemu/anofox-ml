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
        /// Total sample weight at this node (weighted fits only).
        #[serde(default)]
        weighted_n_samples: Option<F>,
    },
    /// Leaf node: predict a value.
    Leaf {
        value: F,
        /// Number of training samples in this leaf.
        n_samples: usize,
        /// For classifiers: class distribution counts.
        class_counts: Option<Vec<(F, usize)>>,
        /// Impurity of the leaf (0 in models saved before it was stored).
        #[serde(default)]
        impurity: F,
        /// Total sample weight in this leaf (weighted fits only).
        #[serde(default)]
        weighted_n_samples: Option<F>,
        /// For weighted classifiers: per-class sample weight.
        #[serde(default)]
        class_weights: Option<Vec<(F, F)>>,
    },
}

impl<F: Float> TreeNode<F> {
    /// Class distribution of a leaf as `(class, mass)`: the per-class sample
    /// weight for weighted fits, the counts otherwise. `None` for
    /// regression leaves and split nodes.
    pub fn class_distribution(&self) -> Option<Vec<(F, F)>> {
        match self {
            TreeNode::Leaf {
                class_weights: Some(w),
                ..
            } => Some(w.clone()),
            TreeNode::Leaf {
                class_counts: Some(c),
                ..
            } => Some(
                c.iter()
                    .map(|&(k, n)| (k, F::from_usize(n).unwrap()))
                    .collect(),
            ),
            _ => None,
        }
    }

    /// Weighted number of samples at this node (the count when unweighted).
    pub fn weighted_samples(&self) -> F {
        match self {
            TreeNode::Leaf {
                weighted_n_samples: Some(w),
                ..
            }
            | TreeNode::Split {
                weighted_n_samples: Some(w),
                ..
            } => *w,
            TreeNode::Leaf { n_samples, .. } | TreeNode::Split { n_samples, .. } => {
                F::from_usize(*n_samples).unwrap()
            }
        }
    }

    /// Impurity at this node.
    pub fn impurity(&self) -> F {
        match self {
            TreeNode::Leaf { impurity, .. } | TreeNode::Split { impurity, .. } => *impurity,
        }
    }

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

    fn accumulate_importances(&self, importances: &mut [F], _total_samples: usize) {
        let total = self.weighted_samples();
        self.accumulate_importances_weighted(importances, total);
    }

    /// Mean decrease in impurity weighted by the (weighted) node size, as
    /// sklearn's `feature_importances_` (before normalisation).
    fn accumulate_importances_weighted(&self, importances: &mut [F], total: F) {
        if let TreeNode::Split {
            feature_index,
            left,
            right,
            impurity,
            ..
        } = self
        {
            let n = self.weighted_samples();
            let nl = left.weighted_samples();
            let nr = right.weighted_samples();
            importances[*feature_index] +=
                (n * *impurity - nl * left.impurity() - nr * right.impurity()) / total;
            left.accumulate_importances_weighted(importances, total);
            right.accumulate_importances_weighted(importances, total);
        }
    }
}

/// Minimal cost-complexity pruning (sklearn's `ccp_alpha`): repeatedly
/// collapse the weakest link — the split with the smallest effective alpha
/// `(R(t) - R(T_t)) / (|leaves(T_t)| - 1)`, `R(t) = w_t / W · impurity_t`
/// — while that alpha is at most `ccp_alpha`.
pub(crate) fn prune_ccp<F: Float>(
    mut tree: TreeNode<F>,
    ccp_alpha: f64,
    classification: bool,
) -> TreeNode<F> {
    if ccp_alpha <= 0.0 {
        return tree;
    }
    let total = tree.weighted_samples();
    let alpha = F::from_f64(ccp_alpha).unwrap();
    loop {
        // (effective alpha, preorder index) of the weakest link.
        let mut best: Option<(F, usize)> = None;
        let mut counter = 0;
        weakest_link(&tree, total, &mut counter, &mut best);
        match best {
            Some((g, idx)) if g <= alpha => {
                let mut counter = 0;
                collapse_at(&mut tree, idx, &mut counter, classification);
            }
            _ => return tree,
        }
    }
}

/// Returns `(R(T_t), n_leaves)` of the subtree and records the weakest link.
fn weakest_link<F: Float>(
    node: &TreeNode<F>,
    total: F,
    counter: &mut usize,
    best: &mut Option<(F, usize)>,
) -> (F, usize) {
    let idx = *counter;
    *counter += 1;
    let r_node = node.weighted_samples() / total * node.impurity();
    match node {
        TreeNode::Leaf { .. } => (r_node, 1),
        TreeNode::Split { left, right, .. } => {
            let (rl, nl) = weakest_link(left, total, counter, best);
            let (rr, nr) = weakest_link(right, total, counter, best);
            let (r_sub, leaves) = (rl + rr, nl + nr);
            let g = (r_node - r_sub) / F::from_usize(leaves - 1).unwrap();
            if best.is_none_or(|(b, _)| g < b) {
                *best = Some((g, idx));
            }
            (r_sub, leaves)
        }
    }
}

fn collapse_at<F: Float>(
    node: &mut TreeNode<F>,
    target: usize,
    counter: &mut usize,
    classification: bool,
) -> bool {
    if *counter == target {
        let owned = std::mem::replace(
            node,
            TreeNode::Leaf {
                value: F::zero(),
                n_samples: 0,
                class_counts: None,
                impurity: F::zero(),
                weighted_n_samples: None,
                class_weights: None,
            },
        );
        *node = collapse(owned, classification);
        return true;
    }
    *counter += 1;
    if let TreeNode::Split { left, right, .. } = node {
        // Pre-order: skip the left subtree's indices when the target lies right.
        if collapse_at(left, target, counter, classification) {
            return true;
        }
        return collapse_at(right, target, counter, classification);
    }
    false
}

/// Turn a subtree into one leaf holding the merged training statistics.
fn collapse<F: Float>(node: TreeNode<F>, classification: bool) -> TreeNode<F> {
    let TreeNode::Split {
        left,
        right,
        n_samples,
        impurity,
        weighted_n_samples,
        ..
    } = node
    else {
        return node;
    };
    let (l, r) = (
        collapse(*left, classification),
        collapse(*right, classification),
    );
    let merge_counts = |a: &Option<Vec<(F, usize)>>, b: &Option<Vec<(F, usize)>>| {
        let mut out: Vec<(F, usize)> = a.iter().chain(b.iter()).flatten().copied().collect();
        out.sort_by(|x, y| crate::split::cmp_nan_last(&x.0, &y.0));
        out.dedup_by(|x, y| {
            if x.0 == y.0 {
                y.1 += x.1;
                true
            } else {
                false
            }
        });
        out
    };
    let merge_weights = |a: &Option<Vec<(F, F)>>, b: &Option<Vec<(F, F)>>| {
        let mut out: Vec<(F, F)> = a.iter().chain(b.iter()).flatten().copied().collect();
        out.sort_by(|x, y| crate::split::cmp_nan_last(&x.0, &y.0));
        out.dedup_by(|x, y| {
            if x.0 == y.0 {
                y.1 += x.1;
                true
            } else {
                false
            }
        });
        out
    };
    let (lw, rw) = (l.weighted_samples(), r.weighted_samples());
    let (
        TreeNode::Leaf {
            value: lv,
            class_counts: lc,
            class_weights: lcw,
            ..
        },
        TreeNode::Leaf {
            value: rv,
            class_counts: rc,
            class_weights: rcw,
            ..
        },
    ) = (&l, &r)
    else {
        unreachable!("collapsed children are leaves")
    };
    let (class_counts, class_weights, value) = if classification {
        let counts = merge_counts(lc, rc);
        let weights = (lcw.is_some() || rcw.is_some()).then(|| merge_weights(lcw, rcw));
        let dist: Vec<(F, F)> = match &weights {
            Some(w) => w.clone(),
            None => counts
                .iter()
                .map(|&(k, c)| (k, F::from_usize(c).unwrap()))
                .collect(),
        };
        // Majority class; ties go to the smallest label.
        let mut best = dist[0];
        for &d in &dist[1..] {
            if d.1 > best.1 {
                best = d;
            }
        }
        (Some(counts), weights, best.0)
    } else {
        let w = lw + rw;
        let v = if w > F::zero() {
            (lw * *lv + rw * *rv) / w
        } else {
            *lv
        };
        (None, None, v)
    };
    TreeNode::Leaf {
        value,
        n_samples,
        class_counts,
        impurity,
        weighted_n_samples,
        class_weights,
    }
}
