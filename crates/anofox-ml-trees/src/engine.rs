//! Presorted CART tree growing.
//!
//! The tree is grown over a list of *sample slots*: `rows[s]` is the row of
//! `x` / `y` behind slot `s`. Rows may repeat (bootstrap draws), so forests
//! and boosting can pass their draws directly instead of copying `X`.
//!
//! Each feature's slots are sorted once per tree (stably, NaN last). A node
//! is a contiguous range of every per-feature order; a split partitions each
//! range stably into its left and right part, so no node re-sorts. Split
//! search, tie-breaking, node impurities and leaves are the same as in the
//! per-node implementation in [`crate::split`], so the trees are identical.
//!
//! For classification the labels are encoded once per tree; each node maps
//! them to classes in order of first appearance (as the per-node class map
//! did) with plain arrays instead of a hash map.

use anofox_ml_core::Float;
use ndarray::{Array1, Array2};

use crate::node::TreeNode;
use crate::split::{
    child_seed, cmp_nan_last, complement_features, compute_impurity, compute_weighted_impurity,
    evaluate_candidate_split, float_key, impurity_from_counts, impurity_from_weights, make_leaf,
    select_feature_subset, try_update_best_split, CandidateSplit, RegressionAccumulator,
    SplitAccumulator, SplitCriterion, WeightedRegressionAccumulator,
};

/// Stopping rules and feature subsampling of one tree.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GrowParams {
    pub max_depth: Option<usize>,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub criterion: SplitCriterion,
    /// Features drawn per split (`None` = all).
    pub max_features_k: Option<usize>,
    pub n_features: usize,
    /// Minimum weighted impurity decrease of a split (sklearn's
    /// `min_impurity_decrease`).
    pub min_impurity_decrease: f64,
    /// Minimum fraction of the total sample weight in each child.
    pub min_weight_fraction_leaf: f64,
    /// Grow best-first up to this many leaves (`None` = depth-first).
    pub max_leaf_nodes: Option<usize>,
}

/// Grow a tree on the sample slots `rows` (row indices into `x` / `y`,
/// repeats allowed). `weights` are per row of `x`.
pub(crate) fn grow<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    rows: &[usize],
    weights: Option<&Array1<F>>,
    params: &GrowParams,
    seed: u64,
) -> TreeNode<F> {
    let m = rows.len();
    let d = x.ncols();
    // Column-major copy of the feature values per slot.
    let cols: Vec<Vec<F>> = (0..d)
        .map(|j| {
            let c = x.column(j);
            rows.iter().map(|&r| c[r]).collect()
        })
        .collect();
    let sorted: Vec<Vec<u32>> = cols
        .iter()
        .map(|col| {
            let mut order: Vec<u32> = (0..m as u32).collect();
            order.sort_by(|&a, &b| cmp_nan_last(&col[a as usize], &col[b as usize]));
            order
        })
        .collect();

    let classes = match params.criterion {
        SplitCriterion::Mse => None,
        SplitCriterion::Gini | SplitCriterion::Entropy => {
            let mut keys: Vec<u64> = rows.iter().map(|&r| float_key(y[r])).collect();
            keys.sort_unstable();
            keys.dedup();
            // Class index per row of `y` (rows outside the sample are unused).
            let mut cls = vec![u32::MAX; y.len()];
            for &r in rows {
                cls[r] = keys.binary_search(&float_key(y[r])).unwrap() as u32;
            }
            Some((cls, keys.len()))
        }
    };

    let total_weight = match weights {
        Some(w) => rows.iter().fold(F::zero(), |a, &r| a + w[r]),
        None => F::from_usize(m).unwrap(),
    };
    let mut g = Grower {
        y,
        rows,
        weights,
        params,
        total_weight,
        cols,
        sorted,
        order: (0..m as u32).collect(),
        goes_left: vec![false; m],
        buf: Vec::with_capacity(m),
        classes,
    };
    match params.max_leaf_nodes {
        Some(max_leaves) => g.grow_best_first(m, seed, max_leaves),
        None => g.node(0, m, 0, seed),
    }
}

/// A node evaluated for splitting.
struct Eval<F: Float> {
    leaf: TreeNode<F>,
    n_samples: usize,
    impurity: F,
    weighted_n_samples: Option<F>,
    /// The chosen split and its weighted impurity decrease (sklearn's
    /// `impurity_improvement`), if the node may split.
    split: Option<(CandidateSplit<F>, F)>,
}

/// Arena node of best-first growth.
enum Slot<F: Float> {
    Leaf(TreeNode<F>),
    Split {
        feature: usize,
        threshold: F,
        left: usize,
        right: usize,
        n_samples: usize,
        impurity: F,
        weighted_n_samples: Option<F>,
    },
}

fn from_arena<F: Float>(arena: &mut Vec<Option<Slot<F>>>, id: usize) -> TreeNode<F> {
    match arena[id].take().expect("arena node used once") {
        Slot::Leaf(leaf) => leaf,
        Slot::Split {
            feature,
            threshold,
            left,
            right,
            n_samples,
            impurity,
            weighted_n_samples,
        } => TreeNode::Split {
            feature_index: feature,
            threshold,
            left: Box::new(from_arena(arena, left)),
            right: Box::new(from_arena(arena, right)),
            n_samples,
            impurity,
            weighted_n_samples,
        },
    }
}

struct Grower<'a, F: Float> {
    y: &'a Array1<F>,
    rows: &'a [usize],
    weights: Option<&'a Array1<F>>,
    params: &'a GrowParams,
    /// Total (weighted) number of samples of the tree.
    total_weight: F,
    /// Feature values per slot, one vector per feature.
    cols: Vec<Vec<F>>,
    /// Per feature: slots sorted by value within every node range.
    sorted: Vec<Vec<u32>>,
    /// Slots in ascending order within every node range.
    order: Vec<u32>,
    goes_left: Vec<bool>,
    buf: Vec<u32>,
    /// Classification: class index per row of `y` and number of classes.
    classes: Option<(Vec<u32>, usize)>,
}

impl<F: Float> Grower<'_, F> {
    /// Evaluate the node `[start, end)`: its leaf, impurity and best
    /// admissible split.
    fn evaluate(
        &self,
        start: usize,
        end: usize,
        depth: usize,
        seed: u64,
        need_leaf: bool,
    ) -> Eval<F> {
        let p = *self.params;
        let indices: Vec<usize> = self.order[start..end]
            .iter()
            .map(|&s| self.rows[s as usize])
            .collect();
        let n_samples = indices.len();
        let weighted_n_samples = self
            .weights
            .map(|w| indices.iter().fold(F::zero(), |a, &i| a + w[i]));
        let impurity = match self.weights {
            Some(w) => compute_weighted_impurity(self.y, &indices, w, p.criterion),
            None => compute_impurity(self.y, &indices, p.criterion),
        };
        let node_weight = weighted_n_samples.unwrap_or_else(|| F::from_usize(n_samples).unwrap());
        let min_weight_leaf = F::from_f64(p.min_weight_fraction_leaf).unwrap() * self.total_weight;

        let mut split = None;
        if n_samples >= p.min_samples_split
            && !p.max_depth.is_some_and(|md| depth >= md)
            && impurity >= F::from_f64(1e-15).unwrap()
            && node_weight >= min_weight_leaf + min_weight_leaf
        {
            let found = match p.max_features_k {
                Some(k) if k < p.n_features => {
                    let subset = select_feature_subset(p.n_features, k, seed);
                    self.best_split(start, end, &indices, impurity, &subset)
                        .or_else(|| {
                            let rest = complement_features(p.n_features, &subset);
                            self.best_split(start, end, &indices, impurity, &rest)
                        })
                }
                _ => {
                    let all: Vec<usize> = (0..p.n_features).collect();
                    self.best_split(start, end, &indices, impurity, &all)
                }
            };
            if let Some(c) = found {
                let decrease = node_weight / self.total_weight * c.improvement;
                // sklearn: split unless improvement + EPSILON < min_impurity_decrease.
                if decrease + F::epsilon() >= F::from_f64(p.min_impurity_decrease).unwrap() {
                    split = Some((c, decrease));
                }
            }
        }
        let leaf = if need_leaf || split.is_none() {
            make_leaf(self.y, &indices, p.criterion, self.weights)
        } else {
            TreeNode::Leaf {
                value: F::zero(),
                n_samples,
                class_counts: None,
                impurity,
                weighted_n_samples,
                class_weights: None,
            }
        };
        Eval {
            leaf,
            n_samples,
            impurity,
            weighted_n_samples,
            split,
        }
    }

    /// Depth-first growth.
    fn node(&mut self, start: usize, end: usize, depth: usize, seed: u64) -> TreeNode<F> {
        let e = self.evaluate(start, end, depth, seed, false);
        let Some((split, _)) = e.split else {
            return e.leaf;
        };
        let n_left = self.partition(start, end, split.feature, split.threshold);
        let left = self.node(start, start + n_left, depth + 1, child_seed(seed, 0));
        let right = self.node(start + n_left, end, depth + 1, child_seed(seed, 1));
        TreeNode::Split {
            feature_index: split.feature,
            threshold: split.threshold,
            left: Box::new(left),
            right: Box::new(right),
            n_samples: e.n_samples,
            impurity: e.impurity,
            weighted_n_samples: e.weighted_n_samples,
        }
    }

    /// Best-first growth (sklearn's `BestFirstTreeBuilder`): always expand
    /// the open node with the largest weighted impurity decrease, until
    /// `max_leaves` leaves exist.
    fn grow_best_first(&mut self, m: usize, seed: u64, max_leaves: usize) -> TreeNode<F> {
        struct Open<F: Float> {
            priority: F,
            id: usize,
            start: usize,
            end: usize,
            depth: usize,
            seed: u64,
            split: CandidateSplit<F>,
            n_samples: usize,
            impurity: F,
            weighted_n_samples: Option<F>,
        }
        let mut arena: Vec<Option<Slot<F>>> = Vec::new();
        let mut open: Vec<Open<F>> = Vec::new();
        let push = |arena: &mut Vec<Option<Slot<F>>>,
                    open: &mut Vec<Open<F>>,
                    e: Eval<F>,
                    start: usize,
                    end: usize,
                    depth: usize,
                    seed: u64| {
            let id = arena.len();
            arena.push(Some(Slot::Leaf(e.leaf)));
            if let Some((split, priority)) = e.split {
                open.push(Open {
                    priority,
                    id,
                    start,
                    end,
                    depth,
                    seed,
                    split,
                    n_samples: e.n_samples,
                    impurity: e.impurity,
                    weighted_n_samples: e.weighted_n_samples,
                });
            }
            id
        };
        let root = self.evaluate(0, m, 0, seed, true);
        push(&mut arena, &mut open, root, 0, m, 0, seed);
        let mut leaves = 1;
        while leaves < max_leaves && !open.is_empty() {
            // Highest priority; ties go to the earliest node (stable).
            let best = (0..open.len())
                .max_by(|&a, &b| {
                    open[a]
                        .priority
                        .partial_cmp(&open[b].priority)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(open[b].id.cmp(&open[a].id))
                })
                .unwrap();
            let o = open.swap_remove(best);
            let n_left = self.partition(o.start, o.end, o.split.feature, o.split.threshold);
            let mid = o.start + n_left;
            let le = self.evaluate(o.start, mid, o.depth + 1, child_seed(o.seed, 0), true);
            let left = push(
                &mut arena,
                &mut open,
                le,
                o.start,
                mid,
                o.depth + 1,
                child_seed(o.seed, 0),
            );
            let re = self.evaluate(mid, o.end, o.depth + 1, child_seed(o.seed, 1), true);
            let right = push(
                &mut arena,
                &mut open,
                re,
                mid,
                o.end,
                o.depth + 1,
                child_seed(o.seed, 1),
            );
            arena[o.id] = Some(Slot::Split {
                feature: o.split.feature,
                threshold: o.split.threshold,
                left,
                right,
                n_samples: o.n_samples,
                impurity: o.impurity,
                weighted_n_samples: o.weighted_n_samples,
            });
            leaves += 1;
        }
        from_arena(&mut arena, 0)
    }

    /// Best split of the node `[start, end)` over `features`.
    fn best_split(
        &self,
        start: usize,
        end: usize,
        indices: &[usize],
        parent_impurity: F,
        features: &[usize],
    ) -> Option<CandidateSplit<F>> {
        let n = end - start;
        if n < 2 * self.params.min_samples_leaf {
            return None;
        }
        match (&self.classes, self.weights) {
            (None, None) => self.scan(
                start,
                end,
                indices,
                parent_impurity,
                features,
                RegressionAccumulator::new(self.y, indices),
            ),
            (None, Some(w)) => self.scan(
                start,
                end,
                indices,
                parent_impurity,
                features,
                WeightedRegressionAccumulator::new(self.y, indices, w),
            ),
            (Some((cls, k)), w) => {
                // Classes in order of first appearance in the node.
                let mut local = vec![usize::MAX; *k];
                let mut n_local = 0;
                for &r in indices {
                    let c = cls[r] as usize;
                    if local[c] == usize::MAX {
                        local[c] = n_local;
                        n_local += 1;
                    }
                }
                let acc = EncodedClassAccumulator {
                    cls,
                    local,
                    weights: w,
                    left: vec![F::zero(); n_local],
                    right: vec![F::zero(); n_local],
                    left_counts: vec![0; n_local],
                    right_counts: vec![0; n_local],
                    left_weight: F::zero(),
                    right_weight: F::zero(),
                    n_left: 0,
                    n_right: 0,
                    criterion: self.params.criterion,
                };
                self.scan(start, end, indices, parent_impurity, features, acc)
            }
        }
    }

    /// The split search of `find_best_split_inner` over presorted slots.
    fn scan<A: SplitAccumulator<F>>(
        &self,
        start: usize,
        end: usize,
        indices: &[usize],
        parent_impurity: F,
        features: &[usize],
        mut acc: A,
    ) -> Option<CandidateSplit<F>> {
        let n = end - start;
        let min_weight_leaf =
            F::from_f64(self.params.min_weight_fraction_leaf).unwrap() * self.total_weight;
        let mut best: Option<CandidateSplit<F>> = None;
        let mut best_improvement = F::neg_infinity();
        for &f in features {
            let col = &self.cols[f];
            let order = &self.sorted[f][start..end];
            acc.reset(self.y, indices);
            for pos in 0..n - 1 {
                let s = order[pos] as usize;
                acc.move_to_left(self.y, self.rows[s]);
                let cur = col[s];
                let next = col[order[pos + 1] as usize];
                if let Some((threshold, improvement)) = evaluate_candidate_split(
                    &acc,
                    n,
                    self.params.min_samples_leaf,
                    cur,
                    next,
                    parent_impurity,
                ) {
                    if min_weight_leaf > F::zero() {
                        let (ml, mr) = acc.masses();
                        if ml < min_weight_leaf || mr < min_weight_leaf {
                            continue;
                        }
                    }
                    try_update_best_split(
                        improvement,
                        &mut best_improvement,
                        &mut best,
                        f,
                        threshold,
                    );
                }
            }
        }
        best
    }

    /// Stable partition of every per-feature order (and of `order`) of the
    /// node into `x[f] <= threshold` (left) and the rest. Returns the left
    /// size.
    fn partition(&mut self, start: usize, end: usize, feature: usize, threshold: F) -> usize {
        let col = &self.cols[feature];
        let mut n_left = 0;
        for &s in &self.order[start..end] {
            let left = col[s as usize] <= threshold;
            self.goes_left[s as usize] = left;
            n_left += usize::from(left);
        }
        let goes_left = &self.goes_left;
        let buf = &mut self.buf;
        let mut split_one = |slice: &mut [u32]| {
            buf.clear();
            let mut w = 0;
            for i in 0..slice.len() {
                let s = slice[i];
                if goes_left[s as usize] {
                    slice[w] = s;
                    w += 1;
                } else {
                    buf.push(s);
                }
            }
            slice[w..].copy_from_slice(buf);
        };
        for order in &mut self.sorted {
            split_one(&mut order[start..end]);
        }
        split_one(&mut self.order[start..end]);
        n_left
    }
}

/// Classification accumulator over encoded classes. Unweighted it does the
/// arithmetic of `ClassificationAccumulator`, weighted that of
/// `WeightedClassificationAccumulator`, with the same (first-appearance)
/// class order.
struct EncodedClassAccumulator<'a, F: Float> {
    cls: &'a [u32],
    local: Vec<usize>,
    weights: Option<&'a Array1<F>>,
    left: Vec<F>,
    right: Vec<F>,
    left_counts: Vec<usize>,
    right_counts: Vec<usize>,
    left_weight: F,
    right_weight: F,
    n_left: usize,
    n_right: usize,
    criterion: SplitCriterion,
}

impl<F: Float> EncodedClassAccumulator<'_, F> {
    #[inline]
    fn class_of_row(&self, row: usize) -> usize {
        self.local[self.cls[row] as usize]
    }
}

impl<F: Float> SplitAccumulator<F> for EncodedClassAccumulator<'_, F> {
    fn reset(&mut self, _y: &Array1<F>, indices: &[usize]) {
        self.left.fill(F::zero());
        self.right.fill(F::zero());
        self.left_counts.fill(0);
        self.right_counts.fill(0);
        self.left_weight = F::zero();
        self.right_weight = F::zero();
        for &r in indices {
            let c = self.class_of_row(r);
            self.right_counts[c] += 1;
            if let Some(w) = self.weights {
                self.right[c] += w[r];
                self.right_weight += w[r];
            }
        }
        self.n_left = 0;
        self.n_right = indices.len();
    }

    #[inline]
    fn move_to_left(&mut self, _y: &Array1<F>, row: usize) {
        let c = self.class_of_row(row);
        self.left_counts[c] += 1;
        self.right_counts[c] -= 1;
        if let Some(w) = self.weights {
            let wr = w[row];
            self.left[c] += wr;
            self.right[c] -= wr;
            self.left_weight += wr;
            self.right_weight -= wr;
        }
        self.n_left += 1;
        self.n_right -= 1;
    }

    fn masses(&self) -> (F, F) {
        match self.weights {
            None => (
                F::from_usize(self.n_left).unwrap(),
                F::from_usize(self.n_right).unwrap(),
            ),
            Some(_) => (self.left_weight, self.right_weight),
        }
    }

    fn weighted_impurity(&self, n: usize) -> F {
        match self.weights {
            None => {
                let n_f = F::from_usize(n).unwrap();
                let nl = F::from_usize(self.n_left).unwrap();
                let nr = F::from_usize(self.n_right).unwrap();
                let li = impurity_from_counts(&self.left_counts, self.n_left, self.criterion);
                let ri = impurity_from_counts(&self.right_counts, self.n_right, self.criterion);
                (nl / n_f) * li + (nr / n_f) * ri
            }
            Some(_) => {
                let total = self.left_weight + self.right_weight;
                if total <= F::zero() {
                    return F::zero();
                }
                let li = impurity_from_weights(
                    self.left.iter().copied(),
                    self.left_weight,
                    self.criterion,
                );
                let ri = impurity_from_weights(
                    self.right.iter().copied(),
                    self.right_weight,
                    self.criterion,
                );
                (self.left_weight / total) * li + (self.right_weight / total) * ri
            }
        }
    }

    fn n_left(&self) -> usize {
        self.n_left
    }

    fn n_right(&self) -> usize {
        self.n_right
    }
}
