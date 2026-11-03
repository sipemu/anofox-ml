//! Agglomerative (hierarchical) clustering.
//!
//! Mirrors `sklearn.cluster.AgglomerativeClustering` without connectivity
//! constraints: single linkage from a minimum spanning tree (Prim, O(n)
//! memory, as sklearn's `mst_linkage_core`), complete / average / Ward
//! from scipy's implementation of Müllner's
//! nearest-neighbour chain with Lance–Williams updates (O(n²) time, O(n²)
//! memory for the condensed distances). Merges are sorted by distance and
//! relabelled like scipy's `linkage`, giving sklearn's `children_` and
//! `distances_`; flat labels come from sklearn's `_hc_cut` (including its
//! label order), with either a fixed `n_clusters` or a `distance_threshold`.

use anofox_ml_core::Float;
use anofox_ml_core::{FitUnsupervised, Result, RustMlError};
use ndarray::{Array1, ArrayView2};

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Linkage {
    Single,
    Complete,
    Average,
    Ward,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AgglomerativeClustering {
    /// Number of clusters (ignored when `distance_threshold` is set).
    pub n_clusters: usize,
    pub linkage: Linkage,
    /// Cut the tree where the merge distance reaches this value instead of
    /// at a fixed number of clusters (sklearn's `distance_threshold`).
    #[serde(default)]
    pub distance_threshold: Option<f64>,
}

impl AgglomerativeClustering {
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            linkage: Linkage::Ward,
            distance_threshold: None,
        }
    }
    pub fn with_linkage(mut self, l: Linkage) -> Self {
        self.linkage = l;
        self
    }
    /// Cut at merge distance `t` (every merge at or above it is undone).
    pub fn with_distance_threshold(mut self, t: Option<f64>) -> Self {
        self.distance_threshold = t;
        self
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedAgglomerativeClustering {
    pub labels: Array1<f64>,
    pub n_clusters: usize,
    /// Merged nodes of every step (sklearn's `children_`): node ids below
    /// `n_samples` are samples, `n_samples + i` the node from step `i`.
    #[serde(default)]
    pub children: Vec<[usize; 2]>,
    /// Merge distance of every step (sklearn's `distances_`; Ward
    /// distances as scipy defines them).
    #[serde(default)]
    pub distances: Vec<f64>,
}

/// Index of pair (i, j), i != j, in a condensed distance vector.
#[inline]
fn condensed(n: usize, i: usize, j: usize) -> usize {
    let (i, j) = if i < j { (i, j) } else { (j, i) };
    n * i - i * (i + 1) / 2 + (j - i - 1)
}

fn euclid(x: ArrayView2<f64>, i: usize, j: usize) -> f64 {
    x.row(i)
        .iter()
        .zip(x.row(j))
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt()
}

/// One unsorted merge (x, y, distance).
type Merge = (usize, usize, f64);

/// sklearn's `mst_linkage_core` (Prim on the complete graph, distances on
/// the fly).
fn single_linkage(x: ArrayView2<f64>) -> Vec<Merge> {
    let n = x.nrows();
    let mut merged = vec![false; n];
    let mut d = vec![f64::INFINITY; n];
    let mut out = Vec::with_capacity(n - 1);
    let mut y = 0;
    for _ in 0..n - 1 {
        let cur = y;
        merged[cur] = true;
        let mut best = f64::INFINITY;
        y = usize::MAX;
        for i in 0..n {
            if merged[i] {
                continue;
            }
            let dist = euclid(x, cur, i);
            if d[i] > dist {
                d[i] = dist;
            }
            if d[i] < best {
                best = d[i];
                y = i;
            }
        }
        if y == usize::MAX {
            // Only NaN distances left: take the first unmerged sample.
            y = (0..n).find(|&i| !merged[i]).expect("unmerged sample");
            best = d[y];
        }
        out.push((cur, y, best));
    }
    out
}

/// scipy's `nn_chain` with the Lance–Williams update of `linkage`.
fn nn_chain(x: ArrayView2<f64>, linkage: Linkage) -> Vec<Merge> {
    let n = x.nrows();
    let mut dist = vec![0.0_f64; n * (n - 1) / 2];
    for i in 0..n {
        for j in (i + 1)..n {
            dist[condensed(n, i, j)] = euclid(x, i, j);
        }
    }
    let new_dist = |dxi: f64, dyi: f64, dxy: f64, nx: f64, ny: f64, ni: f64| match linkage {
        Linkage::Single => dxi.min(dyi),
        Linkage::Complete => dxi.max(dyi),
        Linkage::Average => (nx * dxi + ny * dyi) / (nx + ny),
        Linkage::Ward => (((ni + nx) * dxi * dxi + (ni + ny) * dyi * dyi - ni * dxy * dxy)
            / (nx + ny + ni))
            .sqrt(),
    };
    let mut size = vec![1usize; n];
    let mut chain: Vec<usize> = Vec::with_capacity(n);
    let mut out = Vec::with_capacity(n - 1);
    for _ in 0..n - 1 {
        if chain.is_empty() {
            chain.push((0..n).find(|&i| size[i] > 0).expect("active cluster"));
        }
        let (mut x_c, mut y_c, mut current_min);
        loop {
            x_c = *chain.last().expect("non-empty chain");
            // Prefer the previous chain element on ties (avoids cycles).
            if chain.len() > 1 {
                y_c = chain[chain.len() - 2];
                current_min = dist[condensed(n, x_c, y_c)];
            } else {
                y_c = usize::MAX;
                current_min = f64::INFINITY;
            }
            for i in 0..n {
                if size[i] == 0 || i == x_c {
                    continue;
                }
                let d = dist[condensed(n, x_c, i)];
                if d < current_min {
                    current_min = d;
                    y_c = i;
                }
            }
            if y_c == usize::MAX {
                // Only NaN distances: merge with the first other cluster.
                y_c = (0..n)
                    .find(|&i| size[i] > 0 && i != x_c)
                    .expect("two clusters");
                current_min = dist[condensed(n, x_c, y_c)];
            }
            if chain.len() > 1 && y_c == chain[chain.len() - 2] {
                break;
            }
            chain.push(y_c);
        }
        chain.truncate(chain.len() - 2);
        let (x_c, y_c) = (x_c.min(y_c), x_c.max(y_c));
        let (nx, ny) = (size[x_c], size[y_c]);
        out.push((x_c, y_c, current_min));
        size[x_c] = 0;
        size[y_c] = nx + ny;
        for i in 0..n {
            let ni = size[i];
            if ni == 0 || i == y_c {
                continue;
            }
            let d = new_dist(
                dist[condensed(n, i, x_c)],
                dist[condensed(n, i, y_c)],
                current_min,
                nx as f64,
                ny as f64,
                ni as f64,
            );
            dist[condensed(n, i, y_c)] = d;
        }
    }
    out
}

/// Post-processing: stable sort by distance, then relabel through
/// union-find so step `i` creates node `n + i`. scipy's `label` (nn-chain
/// linkages) orders each pair ascending; sklearn's `single_linkage_label`
/// keeps (tree side, new node).
fn relabel(mut merges: Vec<Merge>, n: usize, ascending: bool) -> (Vec<[usize; 2]>, Vec<f64>) {
    merges.sort_by(|a, b| a.2.total_cmp(&b.2));
    let mut parent: Vec<usize> = (0..2 * n - 1).collect();
    let mut next = n;
    let find = |parent: &mut Vec<usize>, mut v: usize| {
        let mut root = v;
        while parent[root] != root {
            root = parent[root];
        }
        while parent[v] != root {
            let nxt = parent[v];
            parent[v] = root;
            v = nxt;
        }
        root
    };
    let mut children = Vec::with_capacity(n - 1);
    let mut distances = Vec::with_capacity(n - 1);
    for (a, b, d) in merges {
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        children.push(if ascending {
            [ra.min(rb), ra.max(rb)]
        } else {
            [ra, rb]
        });
        distances.push(d);
        parent[ra] = next;
        parent[rb] = next;
        next += 1;
    }
    (children, distances)
}

/// CPython's `heapq` on a list (min-heap), enough for `_hc_cut`.
struct PyHeap(Vec<i64>);

impl PyHeap {
    fn sift_down(&mut self, start: usize, mut pos: usize) {
        let item = self.0[pos];
        while pos > start {
            let parent = (pos - 1) >> 1;
            if item < self.0[parent] {
                self.0[pos] = self.0[parent];
                pos = parent;
            } else {
                break;
            }
        }
        self.0[pos] = item;
    }
    fn sift_up(&mut self, mut pos: usize) {
        let end = self.0.len();
        let start = pos;
        let item = self.0[pos];
        let mut child = 2 * pos + 1;
        while child < end {
            let right = child + 1;
            if right < end && self.0[child] >= self.0[right] {
                child = right;
            }
            self.0[pos] = self.0[child];
            pos = child;
            child = 2 * pos + 1;
        }
        self.0[pos] = item;
        self.sift_down(start, pos);
    }
    fn push(&mut self, v: i64) {
        self.0.push(v);
        let last = self.0.len() - 1;
        self.sift_down(0, last);
    }
    fn pushpop(&mut self, v: i64) -> i64 {
        if !self.0.is_empty() && self.0[0] < v {
            let top = std::mem::replace(&mut self.0[0], v);
            self.sift_up(0);
            top
        } else {
            v
        }
    }
}

/// sklearn's `_hc_cut`: split the top `n_clusters - 1` merges; labels
/// follow the order of the cut nodes in the heap list.
fn hc_cut(n_clusters: usize, children: &[[usize; 2]], n_leaves: usize) -> Array1<f64> {
    let mut labels = Array1::<f64>::zeros(n_leaves);
    if children.is_empty() {
        return labels;
    }
    let root =
        children.last().expect("non-empty")[1].max(children.last().expect("non-empty")[0]) + 1;
    let mut heap = PyHeap(vec![-(root as i64)]);
    for _ in 0..n_clusters.saturating_sub(1) {
        let node = (-heap.0[0]) as usize;
        let these = children[node - n_leaves];
        heap.push(-(these[0] as i64));
        heap.pushpop(-(these[1] as i64));
    }
    for (label, &node) in heap.0.iter().enumerate() {
        // Descendant leaves of the node.
        let mut stack = vec![(-node) as usize];
        while let Some(v) = stack.pop() {
            if v < n_leaves {
                labels[v] = label as f64;
            } else {
                let c = children[v - n_leaves];
                stack.push(c[0]);
                stack.push(c[1]);
            }
        }
    }
    labels
}

impl<F: Float> FitUnsupervised<F> for AgglomerativeClustering {
    type Fitted = FittedAgglomerativeClustering;

    /// Computed in `f64` whatever the input float type.
    fn fit_view(&self, x: ArrayView2<F>) -> Result<Self::Fitted> {
        let x_f64 = anofox_ml_core::as_f64(x);
        let x = x_f64.view();
        let n = x.nrows();
        if n == 0 {
            return Err(RustMlError::EmptyInput("empty input".into()));
        }
        if self.distance_threshold.is_none() && (self.n_clusters == 0 || self.n_clusters > n) {
            return Err(RustMlError::InvalidParameter("invalid n_clusters".into()));
        }
        if n == 1 {
            return Ok(FittedAgglomerativeClustering {
                labels: Array1::zeros(1),
                n_clusters: 1,
                children: Vec::new(),
                distances: Vec::new(),
            });
        }
        let merges = match self.linkage {
            Linkage::Single => single_linkage(x),
            l => nn_chain(x, l),
        };
        let (children, distances) = relabel(merges, n, self.linkage != Linkage::Single);
        let n_clusters = match self.distance_threshold {
            Some(t) => distances.iter().filter(|&&d| d >= t).count() + 1,
            None => self.n_clusters,
        };
        Ok(FittedAgglomerativeClustering {
            labels: hc_cut(n_clusters, &children, n),
            n_clusters,
            children,
            distances,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_agglomerative_two_groups_ward() {
        let x = array![
            [0.0, 0.0],
            [0.5, 0.1],
            [-0.3, 0.2],
            [0.1, -0.2],
            [10.0, 10.0],
            [10.5, 10.1],
            [9.9, 9.8],
            [10.1, 9.9],
        ];
        let fitted = AgglomerativeClustering::new(2)
            .with_linkage(Linkage::Ward)
            .fit(&x)
            .unwrap();
        let labels = &fitted.labels;
        // All first 4 in the same cluster, all last 4 in the other.
        let l0 = labels[0];
        for i in 1..4 {
            assert_eq!(labels[i], l0);
        }
        for i in 4..8 {
            assert_ne!(labels[i], l0);
        }
    }

    #[test]
    fn test_agglomerative_single_complete_average() {
        let x = array![[0.0], [0.1], [10.0], [10.1], [100.0],];
        for lk in [Linkage::Single, Linkage::Complete, Linkage::Average] {
            let fitted = AgglomerativeClustering::new(3)
                .with_linkage(lk)
                .fit(&x)
                .unwrap();
            // Three distinct clusters, last point should be its own.
            let mut labs: Vec<f64> = fitted.labels.iter().copied().collect();
            labs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            labs.dedup();
            assert_eq!(labs.len(), 3);
        }
    }
}
