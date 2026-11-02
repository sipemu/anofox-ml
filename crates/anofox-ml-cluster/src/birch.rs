//! BIRCH — Balanced Iterative Reducing and Clustering using Hierarchies.
//!
//! Mirrors `sklearn.cluster.Birch`: samples are inserted one by one into a
//! CF-tree of clustering features (count, linear sum, squared sum). A
//! sample descends to the closest subcluster at every level and is
//! absorbed by the closest leaf subcluster if the merged subcluster's
//! radius stays within `threshold`; otherwise it starts a new subcluster.
//! Nodes with more than `branching_factor` subclusters split around their
//! two farthest subclusters, possibly up to a new root. The leaf
//! subcluster centroids are then clustered globally with
//! AgglomerativeClustering (Ward) into `n_clusters`, and every sample takes
//! the label of its nearest subcluster.

use anofox_ml_core::Float;
use anofox_ml_core::{FitUnsupervised, Predict, Result, RustMlError};
use ndarray::{Array1, Array2, ArrayView2};

use crate::agglomerative::AgglomerativeClustering;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Birch {
    /// Maximum radius of a leaf subcluster after absorbing a sample.
    pub threshold: f64,
    /// Number of clusters of the global step (`None`: every subcluster is a
    /// cluster).
    pub n_clusters: Option<usize>,
    /// Maximum subclusters per node (default 50).
    #[serde(default = "default_branching_factor")]
    pub branching_factor: usize,
}

fn default_branching_factor() -> usize {
    50
}

impl Birch {
    /// sklearn's defaults apart from `threshold`: 50 branches, 3 clusters.
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            n_clusters: Some(3),
            branching_factor: 50,
        }
    }
    pub fn with_n_clusters(mut self, n: Option<usize>) -> Self {
        self.n_clusters = n;
        self
    }
    pub fn with_branching_factor(mut self, b: usize) -> Self {
        self.branching_factor = b;
        self
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedBirch {
    /// Leaf subcluster centroids, in leaf order (sklearn's
    /// `subcluster_centers_`).
    pub subcluster_centers: Array2<f64>,
    /// Global cluster of every subcluster.
    pub subcluster_labels: Array1<f64>,
    /// Per-sample cluster labels.
    pub labels: Array1<f64>,
}

/// A clustering feature.
#[derive(Clone)]
struct Sub {
    n: usize,
    ls: Vec<f64>,
    ss: f64,
    centroid: Vec<f64>,
    sq_norm: f64,
    child: Option<usize>,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

impl Sub {
    fn sample(x: &[f64]) -> Self {
        let sq = dot(x, x);
        Sub {
            n: 1,
            ls: x.to_vec(),
            ss: sq,
            centroid: x.to_vec(),
            sq_norm: sq,
            child: None,
        }
    }

    fn empty(d: usize) -> Self {
        Sub {
            n: 0,
            ls: vec![0.0; d],
            ss: 0.0,
            centroid: vec![0.0; d],
            sq_norm: 0.0,
            child: None,
        }
    }

    /// Absorb `other` (sklearn's `update`).
    fn update(&mut self, other: &Sub) {
        self.n += other.n;
        for (a, b) in self.ls.iter_mut().zip(&other.ls) {
            *a += b;
        }
        self.ss += other.ss;
        let n = self.n as f64;
        self.centroid = self.ls.iter().map(|v| v / n).collect();
        self.sq_norm = dot(&self.centroid, &self.centroid);
    }

    /// Absorb `other` if the merged radius stays within `threshold`
    /// (sklearn's `merge_subcluster`).
    fn try_merge(&mut self, other: &Sub, threshold: f64) -> bool {
        let new_ss = self.ss + other.ss;
        let new_ls: Vec<f64> = self.ls.iter().zip(&other.ls).map(|(a, b)| a + b).collect();
        let new_n = self.n + other.n;
        let inv = 1.0 / new_n as f64;
        let new_centroid: Vec<f64> = new_ls.iter().map(|v| inv * v).collect();
        let new_sq_norm = dot(&new_centroid, &new_centroid);
        let sq_radius = new_ss / new_n as f64 - new_sq_norm;
        if sq_radius <= threshold * threshold {
            self.n = new_n;
            self.ls = new_ls;
            self.ss = new_ss;
            self.centroid = new_centroid;
            self.sq_norm = new_sq_norm;
            true
        } else {
            false
        }
    }
}

struct Node {
    is_leaf: bool,
    subs: Vec<Sub>,
    prev_leaf: Option<usize>,
    next_leaf: Option<usize>,
}

struct Tree {
    nodes: Vec<Node>,
    threshold: f64,
    branching_factor: usize,
    d: usize,
}

impl Tree {
    fn new_node(&mut self, is_leaf: bool) -> usize {
        self.nodes.push(Node {
            is_leaf,
            subs: Vec::new(),
            prev_leaf: None,
            next_leaf: None,
        });
        self.nodes.len() - 1
    }

    /// sklearn's `_split_node`: two new nodes (and their parent
    /// subclusters) around the farthest pair of subclusters.
    fn split(&mut self, node: usize) -> (Sub, Sub) {
        let is_leaf = self.nodes[node].is_leaf;
        let n1 = self.new_node(is_leaf);
        let n2 = self.new_node(is_leaf);
        let mut s1 = Sub::empty(self.d);
        let mut s2 = Sub::empty(self.d);
        s1.child = Some(n1);
        s2.child = Some(n2);
        if is_leaf {
            let (prev, next) = (self.nodes[node].prev_leaf, self.nodes[node].next_leaf);
            if let Some(p) = prev {
                self.nodes[p].next_leaf = Some(n1);
            }
            self.nodes[n1].prev_leaf = prev;
            self.nodes[n1].next_leaf = Some(n2);
            self.nodes[n2].prev_leaf = Some(n1);
            self.nodes[n2].next_leaf = next;
            if let Some(nx) = next {
                self.nodes[nx].prev_leaf = Some(n2);
            }
        }
        let subs = std::mem::take(&mut self.nodes[node].subs);
        let m = subs.len();
        // sklearn's `euclidean_distances(..., squared=True)` evaluated in its
        // order, (-2 a·b + |a|²) + |b|², which is not exactly symmetric: the
        // row-major argmax can pick (j, i) over (i, j).
        let dist = |i: usize, j: usize| -> f64 {
            if i == j {
                0.0
            } else {
                ((-2.0 * dot(&subs[i].centroid, &subs[j].centroid) + subs[i].sq_norm)
                    + subs[j].sq_norm)
                    .max(0.0)
            }
        };
        let (mut fa, mut fb, mut best) = (0, 0, f64::NEG_INFINITY);
        for i in 0..m {
            for j in 0..m {
                let v = dist(i, j);
                if v > best {
                    best = v;
                    fa = i;
                    fb = j;
                }
            }
        }
        let to_first: Vec<bool> = (0..m)
            .map(|idx| idx == fa || dist(fa, idx) < dist(fb, idx))
            .collect();
        for (sub, closer_to_first) in subs.into_iter().zip(to_first) {
            if closer_to_first {
                s1.update(&sub);
                self.nodes[n1].subs.push(sub);
            } else {
                s2.update(&sub);
                self.nodes[n2].subs.push(sub);
            }
        }
        (s1, s2)
    }

    /// sklearn's `insert_cf_subcluster`; returns whether `node` overflows.
    fn insert(&mut self, node: usize, sub: Sub) -> bool {
        if self.nodes[node].subs.is_empty() {
            self.nodes[node].subs.push(sub);
            return false;
        }
        let closest = {
            let subs = &self.nodes[node].subs;
            let mut best = (0, f64::INFINITY);
            for (i, c) in subs.iter().enumerate() {
                let v = -2.0 * dot(&c.centroid, &sub.centroid) + c.sq_norm;
                if v < best.1 {
                    best = (i, v);
                }
            }
            best.0
        };
        if let Some(child) = self.nodes[node].subs[closest].child {
            if !self.insert(child, sub.clone()) {
                self.nodes[node].subs[closest].update(&sub);
                return false;
            }
            let (s1, s2) = self.split(child);
            self.nodes[node].subs[closest] = s1;
            self.nodes[node].subs.push(s2);
            return self.nodes[node].subs.len() > self.branching_factor;
        }
        if self.nodes[node].subs[closest].try_merge(&sub, self.threshold) {
            return false;
        }
        self.nodes[node].subs.push(sub);
        self.nodes[node].subs.len() > self.branching_factor
    }
}

/// Index of the nearest row of `centers` (first on ties).
fn nearest(row: ndarray::ArrayView1<'_, f64>, centers: &Array2<f64>) -> usize {
    let mut best = (0, f64::INFINITY);
    for (k, c) in centers.rows().into_iter().enumerate() {
        let d: f64 = row.iter().zip(c).map(|(a, b)| (a - b) * (a - b)).sum();
        if d < best.1 {
            best = (k, d);
        }
    }
    best.0
}

impl<F: Float> FitUnsupervised<F> for Birch {
    type Fitted = FittedBirch;

    /// Computed in `f64` whatever the input float type.
    fn fit_view(&self, x: ArrayView2<F>) -> Result<Self::Fitted> {
        let x_f64 = anofox_ml_core::as_f64(x);
        let x = x_f64.view();
        let (n, d) = x.dim();
        if n == 0 {
            return Err(RustMlError::EmptyInput("empty input".into()));
        }
        if self.threshold <= 0.0 || self.branching_factor < 2 {
            return Err(RustMlError::InvalidParameter(
                "threshold must be > 0 and branching_factor >= 2".into(),
            ));
        }
        let mut tree = Tree {
            nodes: Vec::new(),
            threshold: self.threshold,
            branching_factor: self.branching_factor,
            d,
        };
        let dummy = tree.new_node(true);
        let mut root = tree.new_node(true);
        tree.nodes[dummy].next_leaf = Some(root);
        tree.nodes[root].prev_leaf = Some(dummy);
        for row in x.rows() {
            let sample: Vec<f64> = row.to_vec();
            if tree.insert(root, Sub::sample(&sample)) {
                let (s1, s2) = tree.split(root);
                root = tree.new_node(false);
                tree.nodes[root].subs.push(s1);
                tree.nodes[root].subs.push(s2);
            }
        }
        // Leaf centroids in linked-list order.
        let mut centers: Vec<Vec<f64>> = Vec::new();
        let mut leaf = tree.nodes[dummy].next_leaf;
        while let Some(l) = leaf {
            centers.extend(tree.nodes[l].subs.iter().map(|s| s.centroid.clone()));
            leaf = tree.nodes[l].next_leaf;
        }
        let subcluster_centers = Array2::from_shape_fn((centers.len(), d), |(i, j)| centers[i][j]);
        let m = centers.len();
        let subcluster_labels = match self.n_clusters {
            Some(k) if k <= m => {
                AgglomerativeClustering::new(k)
                    .fit(&subcluster_centers)?
                    .labels
            }
            _ => Array1::from_shape_fn(m, |i| i as f64),
        };
        let labels = Array1::from_shape_fn(n, |i| {
            subcluster_labels[nearest(x.row(i), &subcluster_centers)]
        });
        Ok(FittedBirch {
            subcluster_centers,
            subcluster_labels,
            labels,
        })
    }
}

impl FittedBirch {
    fn predict_f64(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        if x.ncols() != self.subcluster_centers.ncols() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.subcluster_centers.ncols(),
                x.ncols()
            )));
        }
        Ok(Array1::from_shape_fn(x.nrows(), |i| {
            self.subcluster_labels[nearest(x.row(i), &self.subcluster_centers)]
        }))
    }
}

impl<F: Float> Predict<F> for FittedBirch {
    /// Label of the nearest subcluster.
    fn predict_view(&self, x: ArrayView2<F>) -> Result<Array1<F>> {
        Ok(anofox_ml_core::from_f64(
            self.predict_f64(anofox_ml_core::as_f64(x).view())?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_birch_two_blobs() {
        let x = array![
            [0.0_f64, 0.0],
            [0.1, 0.1],
            [-0.1, 0.2],
            [0.1, -0.1],
            [10.0, 10.0],
            [10.1, 9.9],
            [9.8, 10.2],
            [10.2, 9.8],
        ];
        let fitted = Birch::new(1.0).with_n_clusters(Some(2)).fit(&x).unwrap();
        let labels = &fitted.labels;
        let l0 = labels[0];
        for i in 1..4 {
            assert_eq!(labels[i], l0);
        }
        for i in 4..8 {
            assert_ne!(labels[i], l0);
        }
    }

    #[test]
    fn test_birch_no_global_step() {
        // Without a final clustering step, each CF is its own cluster.
        let x = array![[0.0_f64, 0.0], [10.0, 10.0]];
        let fitted = Birch::new(1.0).with_n_clusters(None).fit(&x).unwrap();
        assert_eq!(fitted.subcluster_centers.nrows(), 2);
        assert_ne!(fitted.labels[0], fitted.labels[1]);
    }
}

impl<F: Float> anofox_ml_core::ClassifierScore<F> for FittedBirch {}
