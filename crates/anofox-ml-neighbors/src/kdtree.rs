//! KD-tree over a flat, permuted copy of the training rows (sklearn's
//! `KDTree` design): leaves hold up to `leaf_size` points, nodes split at the
//! median of their widest dimension, and every node keeps its bounding box,
//! whose distance to a query is a lower bound for any Minkowski metric.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use anofox_ml_core::Float;
use ndarray::Array2;

use crate::distance::DistanceMetric;

/// Default number of points per leaf (sklearn's `leaf_size=30`).
pub const DEFAULT_LEAF_SIZE: usize = 30;

const NO_CHILD: usize = usize::MAX;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct Node {
    start: usize,
    end: usize,
    left: usize,
    right: usize,
}

/// A KD-tree for Minkowski-metric neighbour queries.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub(crate) struct KdTree<F: Float> {
    /// Training rows in tree order, row-major.
    data: Vec<F>,
    /// Original row index of every tree-ordered row.
    idx: Vec<usize>,
    nodes: Vec<Node>,
    /// Per node: `d` lower then `d` upper bounds.
    bounds: Vec<F>,
    d: usize,
}

/// A candidate neighbour: max-heap by (reduced distance, index), so the top
/// is the worst of the current k best and ties keep the lower index.
#[derive(Clone, Copy)]
pub(crate) struct Cand<F: Float>(pub F, pub usize);

impl<F: Float> PartialEq for Cand<F> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}
impl<F: Float> Eq for Cand<F> {}
impl<F: Float> PartialOrd for Cand<F> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<F: Float> Ord for Cand<F> {
    fn cmp(&self, other: &Self) -> Ordering {
        crate::total_cmp(&self.0, &other.0).then(self.1.cmp(&other.1))
    }
}

/// Bounded max-heap of the k best candidates.
pub(crate) struct KBest<F: Float> {
    k: usize,
    heap: BinaryHeap<Cand<F>>,
}

impl<F: Float> KBest<F> {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            heap: BinaryHeap::with_capacity(k + 1),
        }
    }

    /// Reduced distance a new candidate must not exceed (∞ until full).
    #[inline]
    pub fn bound(&self) -> F {
        if self.heap.len() < self.k {
            F::infinity()
        } else {
            self.heap.peek().map_or(F::infinity(), |c| c.0)
        }
    }

    #[inline]
    pub fn push(&mut self, c: Cand<F>) {
        if self.heap.len() < self.k {
            self.heap.push(c);
        } else if let Some(top) = self.heap.peek() {
            if c < *top {
                self.heap.pop();
                self.heap.push(c);
            }
        }
    }

    /// The candidates sorted by (reduced distance, index).
    pub fn into_sorted(self) -> Vec<Cand<F>> {
        self.heap.into_sorted_vec()
    }
}

impl<F: Float> KdTree<F> {
    /// Build over the rows of `x` (any layout).
    pub fn build(x: &Array2<F>, leaf_size: usize) -> Self {
        let (n, d) = x.dim();
        let leaf_size = leaf_size.max(1);
        let mut tree = KdTree {
            data: Vec::new(),
            idx: (0..n).collect(),
            nodes: Vec::new(),
            bounds: Vec::new(),
            d,
        };
        if n > 0 {
            tree.build_node(x, 0, n, leaf_size);
        }
        let mut data = Vec::with_capacity(n * d);
        for &i in &tree.idx {
            data.extend(x.row(i).iter().copied());
        }
        tree.data = data;
        tree
    }

    fn build_node(&mut self, x: &Array2<F>, start: usize, end: usize, leaf_size: usize) -> usize {
        let d = self.d;
        let id = self.nodes.len();
        self.nodes.push(Node {
            start,
            end,
            left: NO_CHILD,
            right: NO_CHILD,
        });
        // Bounding box (NaN coordinates are ignored by max/min).
        let mut lo = vec![F::infinity(); d];
        let mut hi = vec![F::neg_infinity(); d];
        for &i in &self.idx[start..end] {
            for (j, &v) in x.row(i).iter().enumerate() {
                lo[j] = lo[j].min(v);
                hi[j] = hi[j].max(v);
            }
        }
        let (dim, spread) = (0..d)
            .map(|j| (j, hi[j] - lo[j]))
            .fold((0, F::neg_infinity()), |b, c| if c.1 > b.1 { c } else { b });
        self.bounds.extend_from_slice(&lo);
        self.bounds.extend_from_slice(&hi);

        if end - start > leaf_size && spread > F::zero() {
            let mid = start + (end - start) / 2;
            self.idx[start..end].select_nth_unstable_by(mid - start, |&a, &b| {
                crate::total_cmp(&x[[a, dim]], &x[[b, dim]])
            });
            let left = self.build_node(x, start, mid, leaf_size);
            let right = self.build_node(x, mid, end, leaf_size);
            self.nodes[id].left = left;
            self.nodes[id].right = right;
        }
        id
    }

    #[inline]
    fn row(&self, p: usize) -> &[F] {
        &self.data[p * self.d..(p + 1) * self.d]
    }

    /// Reduced distance from `q` to the bounding box of `node`.
    #[inline]
    fn min_rdist(&self, node: usize, q: &[F], metric: DistanceMetric) -> F {
        let b = &self.bounds[node * 2 * self.d..(node + 1) * 2 * self.d];
        let (lo, hi) = b.split_at(self.d);
        q.iter()
            .zip(lo.iter().zip(hi))
            .fold(F::zero(), |acc, (&v, (&l, &h))| {
                let diff = (l - v).max(v - h).max(F::zero());
                metric.rdist_add(acc, diff)
            })
    }

    /// The k nearest rows of `q` as (reduced distance, original index),
    /// ties broken by the lower index.
    pub fn knn(&self, q: &[F], k: usize, metric: DistanceMetric) -> Vec<Cand<F>> {
        let mut best = KBest::new(k);
        if !self.nodes.is_empty() && k > 0 {
            let r = self.min_rdist(0, q, metric);
            self.knn_node(0, r, q, metric, &mut best);
        }
        best.into_sorted()
    }

    fn knn_node(
        &self,
        node: usize,
        min_r: F,
        q: &[F],
        metric: DistanceMetric,
        best: &mut KBest<F>,
    ) {
        // `>` (not `>=`): an equally distant point with a lower index may
        // still be inside.
        if min_r > best.bound() {
            return;
        }
        let n = &self.nodes[node];
        if n.left == NO_CHILD {
            for p in n.start..n.end {
                let r = metric.rdist(self.row(p), q);
                best.push(Cand(r, self.idx[p]));
            }
            return;
        }
        let rl = self.min_rdist(n.left, q, metric);
        let rr = self.min_rdist(n.right, q, metric);
        let (a, ra, b, rb) = if rl <= rr {
            (n.left, rl, n.right, rr)
        } else {
            (n.right, rr, n.left, rl)
        };
        self.knn_node(a, ra, q, metric, best);
        self.knn_node(b, rb, q, metric, best);
    }

    /// All rows within reduced distance `r` of `q` (inclusive), unsorted.
    pub fn radius(&self, q: &[F], r: F, metric: DistanceMetric, out: &mut Vec<Cand<F>>) {
        if !self.nodes.is_empty() {
            self.radius_node(0, q, r, metric, out);
        }
    }

    fn radius_node(
        &self,
        node: usize,
        q: &[F],
        r: F,
        metric: DistanceMetric,
        out: &mut Vec<Cand<F>>,
    ) {
        if self.min_rdist(node, q, metric) > r {
            return;
        }
        let n = &self.nodes[node];
        if n.left == NO_CHILD {
            for p in n.start..n.end {
                let dr = metric.rdist(self.row(p), q);
                if dr <= r {
                    out.push(Cand(dr, self.idx[p]));
                }
            }
            return;
        }
        self.radius_node(n.left, q, r, metric, out);
        self.radius_node(n.right, q, r, metric, out);
    }
}
