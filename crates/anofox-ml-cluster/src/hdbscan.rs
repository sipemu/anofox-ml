//! HDBSCAN — Hierarchical Density-Based Spatial Clustering.
//!
//! Mirrors `sklearn.cluster.HDBSCAN` (brute-force / dense variant):
//!
//! 1. **Core distance**: `core_dist(x_i)` = distance to its `min_samples`-th
//!    nearest neighbour, counting the point itself (so `min_samples = 1`
//!    gives core distance 0, exactly as in sklearn).
//! 2. **Mutual reachability**: `d_mr(a, b) = max(core_dist(a), core_dist(b), d(a, b))`.
//! 3. **Minimum spanning tree** of the complete mutual-reachability graph via
//!    dense Prim (O(n²) time, O(n²) memory for the pairwise distances).
//! 4. **Single-linkage tree** (scipy `linkage` format) from the MST edges
//!    sorted by weight.
//! 5. **Condensed tree**: walk the hierarchy top-down; a split where one side
//!    has fewer than `min_cluster_size` points is treated as those points
//!    "falling out" of the parent cluster at `λ = 1 / distance`.
//! 6. **Stability**: `S(C) = Σ_{x ∈ C} (λ_x − λ_birth(C))`.
//! 7. **Flat extraction**: Excess-of-Mass (default) or leaf selection.
//!    The root cluster is only eligible when `allow_single_cluster = true`.
//!
//! Points not belonging to a selected cluster get label `-1`.
//! [`FittedHdbscan::probabilities`] holds sklearn's membership strengths
//! (`λ_point / λ_max(cluster)`, clipped to `[0, 1]`; 0 for noise).

use anofox_ml_core::{FitUnsupervised, Result, RustMlError};
use ndarray::{Array1, Array2};

/// Flat-cluster extraction strategy (sklearn `cluster_selection_method`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum ClusterSelectionMethod {
    /// Excess of Mass: choose the antichain of condensed-tree clusters with
    /// maximal total stability.
    #[default]
    Eom,
    /// Select the leaves of the condensed cluster tree (finer, more
    /// homogeneous clusters).
    Leaf,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Hdbscan {
    /// Number of samples (including the point itself) in a neighbourhood
    /// for a point to be considered a core point. Values larger than
    /// `n_samples` are clamped to `n_samples`.
    pub min_samples: usize,
    /// Minimum number of samples in a group for it to be considered a cluster.
    pub min_cluster_size: usize,
    /// Allow the root of the condensed tree to be selected (i.e. return a
    /// single cluster). Default `false`, as in sklearn.
    pub allow_single_cluster: bool,
    /// Excess-of-Mass (default) or leaf extraction.
    pub cluster_selection_method: ClusterSelectionMethod,
}

impl Hdbscan {
    pub fn new(min_samples: usize, min_cluster_size: usize) -> Self {
        Self {
            min_samples,
            min_cluster_size,
            allow_single_cluster: false,
            cluster_selection_method: ClusterSelectionMethod::Eom,
        }
    }

    pub fn with_min_samples(mut self, min_samples: usize) -> Self {
        self.min_samples = min_samples;
        self
    }

    pub fn with_min_cluster_size(mut self, min_cluster_size: usize) -> Self {
        self.min_cluster_size = min_cluster_size;
        self
    }

    pub fn with_allow_single_cluster(mut self, allow: bool) -> Self {
        self.allow_single_cluster = allow;
        self
    }

    pub fn with_cluster_selection_method(mut self, method: ClusterSelectionMethod) -> Self {
        self.cluster_selection_method = method;
        self
    }
}

impl Default for Hdbscan {
    /// sklearn defaults: `min_cluster_size = 5`, `min_samples = min_cluster_size`.
    fn default() -> Self {
        Self::new(5, 5)
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedHdbscan {
    /// Cluster label per sample (`0..n_clusters`), `-1` for noise.
    pub labels: Array1<f64>,
    /// Number of clusters found (excluding noise).
    pub n_clusters: usize,
    /// Cluster-membership strength per sample in `[0, 1]` (0 for noise),
    /// as sklearn's `probabilities_`.
    #[serde(default)]
    pub probabilities: Array1<f64>,
}

/// One row of the condensed tree (sklearn `CONDENSED_dtype`).
#[derive(Debug, Clone, Copy)]
struct CondensedRow {
    parent: usize,
    child: usize,
    lambda: f64,
    size: usize,
}

/// Single-linkage merge (scipy linkage row).
#[derive(Debug, Clone, Copy)]
struct Merge {
    left: usize,
    right: usize,
    dist: f64,
    size: usize,
}

fn euclid(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Dense Prim MST over the mutual-reachability graph. Returns `n - 1` edges
/// `(u, v, weight)` in insertion order.
fn mst_prim(d: &[f64], core: &[f64], n: usize) -> Vec<(usize, usize, f64)> {
    let mr = |i: usize, j: usize| d[i * n + j].max(core[i]).max(core[j]);
    let mut in_tree = vec![false; n];
    let mut min_dist = vec![f64::INFINITY; n];
    let mut source = vec![0_usize; n];
    let mut edges = Vec::with_capacity(n.saturating_sub(1));
    let mut current = 0usize;
    in_tree[0] = true;
    for _ in 1..n {
        let mut best = f64::INFINITY;
        let mut best_j = usize::MAX;
        for j in 0..n {
            if in_tree[j] {
                continue;
            }
            let r = mr(current, j);
            if r < min_dist[j] {
                min_dist[j] = r;
                source[j] = current;
            }
            if best_j == usize::MAX || min_dist[j] < best {
                best = min_dist[j];
                best_j = j;
            }
        }
        edges.push((source[best_j], best_j, best));
        in_tree[best_j] = true;
        current = best_j;
    }
    edges
}

/// scipy-format single-linkage tree from (stably) sorted MST edges.
fn single_linkage(edges: &[(usize, usize, f64)], n: usize) -> Vec<Merge> {
    let total = 2 * n - 1;
    let mut parent: Vec<usize> = (0..total).collect();
    let mut size = vec![1usize; total];
    let find = |parent: &mut Vec<usize>, mut x: usize| -> usize {
        let mut root = x;
        while parent[root] != root {
            root = parent[root];
        }
        while parent[x] != root {
            let next = parent[x];
            parent[x] = root;
            x = next;
        }
        root
    };
    let mut merges = Vec::with_capacity(n - 1);
    for (step, &(a, b, w)) in edges.iter().enumerate() {
        let next = n + step;
        let ra = find(&mut parent, a);
        let rb = find(&mut parent, b);
        let s = size[ra] + size[rb];
        merges.push(Merge {
            left: ra,
            right: rb,
            dist: w,
            size: s,
        });
        parent[ra] = next;
        parent[rb] = next;
        size[next] = s;
    }
    merges
}

/// BFS over the scipy hierarchy starting at `root` (sklearn `bfs_from_hierarchy`).
fn bfs_hierarchy(merges: &[Merge], root: usize, n: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut queue = vec![root];
    while !queue.is_empty() {
        out.extend_from_slice(&queue);
        let mut next = Vec::new();
        for &node in &queue {
            if node >= n {
                let m = merges[node - n];
                next.push(m.left);
                next.push(m.right);
            }
        }
        queue = next;
    }
    out
}

/// sklearn `_condense_tree`. Cluster ids start at `n` (the root).
fn condense_tree(merges: &[Merge], n: usize, min_cluster_size: usize) -> Vec<CondensedRow> {
    let root = 2 * n - 2;
    let node_count = |node: usize| -> usize {
        if node >= n {
            merges[node - n].size
        } else {
            1
        }
    };
    let mut relabel = vec![0usize; root + 1];
    relabel[root] = n;
    let mut next_label = n + 1;
    let mut ignore = vec![false; root + 1];
    let mut rows = Vec::with_capacity(2 * n);

    for node in bfs_hierarchy(merges, root, n) {
        if ignore[node] || node < n {
            continue;
        }
        let m = merges[node - n];
        let lambda = if m.dist > 0.0 {
            1.0 / m.dist
        } else {
            f64::INFINITY
        };
        let lc = node_count(m.left);
        let rc = node_count(m.right);
        let parent = relabel[node];

        let fall_out = |sub: usize, rows: &mut Vec<CondensedRow>, ignore: &mut Vec<bool>| {
            for s in bfs_hierarchy(merges, sub, n) {
                if s < n {
                    rows.push(CondensedRow {
                        parent,
                        child: s,
                        lambda,
                        size: 1,
                    });
                }
                ignore[s] = true;
            }
        };

        if lc >= min_cluster_size && rc >= min_cluster_size {
            relabel[m.left] = next_label;
            next_label += 1;
            rows.push(CondensedRow {
                parent,
                child: relabel[m.left],
                lambda,
                size: lc,
            });
            relabel[m.right] = next_label;
            next_label += 1;
            rows.push(CondensedRow {
                parent,
                child: relabel[m.right],
                lambda,
                size: rc,
            });
        } else if lc < min_cluster_size && rc < min_cluster_size {
            fall_out(m.left, &mut rows, &mut ignore);
            fall_out(m.right, &mut rows, &mut ignore);
        } else if lc < min_cluster_size {
            relabel[m.right] = parent;
            fall_out(m.left, &mut rows, &mut ignore);
        } else {
            relabel[m.left] = parent;
            fall_out(m.right, &mut rows, &mut ignore);
        }
    }
    rows
}

/// Labels + probabilities from a condensed tree (sklearn `_get_clusters`,
/// `_do_labelling`, `get_probabilities`; `cluster_selection_epsilon = 0`,
/// `max_cluster_size = None`).
fn extract_clusters(
    rows: &[CondensedRow],
    n: usize,
    method: ClusterSelectionMethod,
    allow_single_cluster: bool,
) -> (Vec<i64>, Vec<f64>) {
    let root = n;
    let max_id = rows
        .iter()
        .map(|r| r.parent.max(r.child))
        .max()
        .unwrap_or(root);
    let n_ids = max_id.max(root) + 1;

    // Births, stability, deaths (max λ of direct children), tree links.
    let mut birth = vec![0.0_f64; n_ids];
    let mut parent_of = vec![usize::MAX; n_ids];
    let mut lambda_of = vec![0.0_f64; n_ids];
    for r in rows {
        birth[r.child] = r.lambda;
        parent_of[r.child] = r.parent;
        lambda_of[r.child] = r.lambda;
    }
    birth[root] = 0.0;
    let mut stability = vec![0.0_f64; n_ids];
    let mut death = vec![0.0_f64; n_ids];
    let mut is_parent = vec![false; n_ids];
    for r in rows {
        stability[r.parent] += (r.lambda - birth[r.parent]) * r.size as f64;
        if !is_parent[r.parent] || r.lambda > death[r.parent] {
            death[r.parent] = r.lambda;
        }
        is_parent[r.parent] = true;
    }
    // Cluster-tree children (rows with size > 1).
    let mut cluster_children: Vec<Vec<usize>> = vec![Vec::new(); n_ids];
    for r in rows.iter().filter(|r| r.size > 1) {
        cluster_children[r.parent].push(r.child);
    }

    // Candidate clusters: every id ≥ root appearing as a parent.
    let mut node_list: Vec<usize> = (root..n_ids).filter(|&c| is_parent[c]).collect();
    node_list.sort_unstable_by(|a, b| b.cmp(a)); // descending: children first
    if !allow_single_cluster {
        node_list.retain(|&c| c != root);
    }
    let mut is_cluster = vec![false; n_ids];
    for &c in &node_list {
        is_cluster[c] = true;
    }

    let descendants = |c: usize| -> Vec<usize> {
        let mut out = Vec::new();
        let mut stack = cluster_children[c].clone();
        while let Some(s) = stack.pop() {
            out.push(s);
            stack.extend_from_slice(&cluster_children[s]);
        }
        out
    };

    match method {
        ClusterSelectionMethod::Eom => {
            for &node in &node_list {
                let subtree: f64 = cluster_children[node].iter().map(|&c| stability[c]).sum();
                if subtree > stability[node] {
                    is_cluster[node] = false;
                    stability[node] = subtree;
                } else {
                    for s in descendants(node) {
                        is_cluster[s] = false;
                    }
                }
            }
        }
        ClusterSelectionMethod::Leaf => {
            let has_cluster_tree = rows.iter().any(|r| r.size > 1);
            for c in is_cluster.iter_mut() {
                *c = false;
            }
            if has_cluster_tree {
                let mut stack = vec![root];
                while let Some(c) = stack.pop() {
                    if cluster_children[c].is_empty() {
                        is_cluster[c] = true;
                    } else {
                        stack.extend_from_slice(&cluster_children[c]);
                    }
                }
            } else {
                is_cluster[root] = true;
            }
        }
    }

    let selected: Vec<usize> = (root..n_ids).filter(|&c| is_cluster[c]).collect();
    let mut label_of_cluster = vec![-1_i64; n_ids];
    for (i, &c) in selected.iter().enumerate() {
        label_of_cluster[c] = i as i64;
    }

    // Labelling: nearest selected ancestor; root handling as in sklearn.
    let root_threshold = death[root];
    let mut labels = vec![-1_i64; n];
    for (p, label) in labels.iter_mut().enumerate() {
        let mut c = parent_of[p];
        if c == usize::MAX {
            continue;
        }
        while c != root && !is_cluster[c] {
            c = parent_of[c];
        }
        if c != root {
            *label = label_of_cluster[c];
        } else if selected.len() == 1
            && allow_single_cluster
            && is_cluster[root]
            && lambda_of[p] >= root_threshold
        {
            *label = label_of_cluster[root];
        }
    }

    // Probabilities.
    let mut probs = vec![0.0_f64; n];
    for p in 0..n {
        if labels[p] < 0 || parent_of[p] == usize::MAX {
            continue;
        }
        let cluster = selected[labels[p] as usize];
        let max_lambda = death[cluster];
        let lam = lambda_of[p];
        probs[p] = if max_lambda == 0.0 || lam.is_infinite() {
            1.0
        } else {
            lam.min(max_lambda) / max_lambda
        };
    }
    (labels, probs)
}

impl FitUnsupervised<f64> for Hdbscan {
    type Fitted = FittedHdbscan;

    fn fit(&self, x: &Array2<f64>) -> Result<Self::Fitted> {
        let n = x.nrows();
        if n == 0 {
            return Err(RustMlError::EmptyInput("empty input".into()));
        }
        if self.min_samples < 1 || self.min_cluster_size < 2 {
            return Err(RustMlError::InvalidParameter(
                "min_samples >= 1, min_cluster_size >= 2".into(),
            ));
        }
        if n == 1 {
            return Ok(FittedHdbscan {
                labels: Array1::from_elem(1, -1.0),
                n_clusters: 0,
                probabilities: Array1::zeros(1),
            });
        }
        let x_std = crate::standard_layout(x);
        let x: &Array2<f64> = &x_std;

        // Pairwise distances (dense, row-major n×n).
        let mut d = vec![0.0_f64; n * n];
        for i in 0..n {
            let xi = x.row(i);
            let xi = xi.as_slice().expect("standard layout");
            for j in (i + 1)..n {
                let xj = x.row(j);
                let val = euclid(xi, xj.as_slice().expect("standard layout"));
                d[i * n + j] = val;
                d[j * n + i] = val;
            }
        }

        // Core distances: k-th smallest distance counting the point itself
        // (d[i][i] = 0), k = min_samples — identical to sklearn.
        let k = self.min_samples.min(n);
        let mut core = vec![0.0_f64; n];
        let mut buf = vec![0.0_f64; n];
        for i in 0..n {
            buf.copy_from_slice(&d[i * n..(i + 1) * n]);
            let (_, kth, _) = buf.select_nth_unstable_by(k - 1, |a, b| a.total_cmp(b));
            core[i] = *kth;
        }

        let mut edges = mst_prim(&d, &core, n);
        drop(d);
        edges.sort_by(|a, b| a.2.total_cmp(&b.2)); // stable, like mergesort
        let merges = single_linkage(&edges, n);
        let rows = condense_tree(&merges, n, self.min_cluster_size);
        let (labels, probs) = extract_clusters(
            &rows,
            n,
            self.cluster_selection_method,
            self.allow_single_cluster,
        );

        let mut seen: Vec<i64> = labels.iter().copied().filter(|&l| l >= 0).collect();
        seen.sort_unstable();
        seen.dedup();
        Ok(FittedHdbscan {
            labels: labels.iter().map(|&l| l as f64).collect(),
            n_clusters: seen.len(),
            probabilities: Array1::from_vec(probs),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, ShapeBuilder};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_hdbscan_two_blobs_with_noise() {
        // Two dense clusters plus a wild outlier.
        let x = array![
            [0.0_f64, 0.0],
            [0.1, 0.1],
            [-0.1, 0.2],
            [0.05, -0.1],
            [0.0, 0.15],
            [10.0, 10.0],
            [10.1, 9.9],
            [9.8, 10.2],
            [10.05, 9.95],
            [10.0, 10.1],
            [50.0, 50.0],
        ];
        let fitted = Hdbscan::new(2, 3).fit(&x).unwrap();
        // Cluster A points all share a label; cluster B's all share another.
        let l0 = fitted.labels[0];
        for i in 1..5 {
            assert_eq!(fitted.labels[i], l0);
        }
        let l5 = fitted.labels[5];
        for i in 6..10 {
            assert_eq!(fitted.labels[i], l5);
        }
        assert_ne!(l0, l5);
        // Outlier should be marked noise (label -1).
        assert_eq!(
            fitted.labels[10], -1.0,
            "outlier should be noise, got {}",
            fitted.labels[10]
        );
        assert_eq!(fitted.n_clusters, 2);
        assert_eq!(fitted.probabilities[10], 0.0);
    }

    /// Three Gaussian blobs (σ = 0.5, 50 points each) at well-separated
    /// centres plus 10 uniform noise points. Deterministic (seeded).
    pub(crate) fn three_blobs_with_noise(seed: u64) -> Array2<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let centers = [(0.0, 0.0), (10.0, 0.0), (5.0, 9.0)];
        let mut v = Vec::new();
        for &(cx, cy) in &centers {
            for _ in 0..50 {
                // Box–Muller.
                let u1: f64 = rng.gen_range(1e-12..1.0);
                let u2: f64 = rng.gen_range(0.0..1.0);
                let r = (-2.0 * u1.ln()).sqrt() * 0.5;
                let t = 2.0 * std::f64::consts::PI * u2;
                v.push(cx + r * t.cos());
                v.push(cy + r * t.sin());
            }
        }
        for _ in 0..10 {
            v.push(rng.gen_range(-10.0..20.0));
            v.push(rng.gen_range(20.0..30.0));
        }
        Array2::from_shape_vec((160, 2), v).unwrap()
    }

    /// Regression for #57: the third well-separated cluster used to come
    /// back as all noise because the first two components to meet were
    /// "frozen".
    #[test]
    fn test_hdbscan_three_blobs_plus_noise() {
        let x = three_blobs_with_noise(7);
        let fitted = Hdbscan::new(5, 10).fit(&x).unwrap();
        assert_eq!(fitted.n_clusters, 3, "labels: {:?}", fitted.labels);
        for b in 0..3 {
            let blob = fitted.labels.slice(ndarray::s![b * 50..(b + 1) * 50]);
            // Majority of each blob shares one non-noise label.
            let mut counts = std::collections::BTreeMap::new();
            for &l in blob.iter() {
                *counts.entry(l as i64).or_insert(0usize) += 1;
            }
            let (&lab, &cnt) = counts.iter().max_by_key(|(_, &c)| c).unwrap();
            assert!(lab >= 0, "blob {b} is mostly noise: {counts:?}");
            assert!(cnt >= 45, "blob {b} fragmented: {counts:?}");
        }
        let l: Vec<i64> = (0..3).map(|b| fitted.labels[b * 50] as i64).collect();
        assert!(l[0] != l[1] && l[1] != l[2] && l[0] != l[2]);
        for &p in fitted.probabilities.iter() {
            assert!((0.0..=1.0).contains(&p));
        }
    }

    /// Noise points far from all blobs must be labelled -1 with probability 0.
    #[test]
    fn test_hdbscan_far_noise_is_noise() {
        let mut x = three_blobs_with_noise(3);
        x.row_mut(150).assign(&array![500.0, 500.0]);
        let fitted = Hdbscan::new(5, 10).fit(&x).unwrap();
        assert_eq!(fitted.labels[150], -1.0);
        assert_eq!(fitted.probabilities[150], 0.0);
    }

    /// Single tight blob: default (allow_single_cluster = false) yields no
    /// cluster, allow_single_cluster = true yields one.
    #[test]
    fn test_hdbscan_allow_single_cluster() {
        let mut rng = StdRng::seed_from_u64(1);
        let v: Vec<f64> = (0..80).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let x = Array2::from_shape_vec((40, 2), v).unwrap();
        let single = Hdbscan::new(5, 20)
            .with_allow_single_cluster(true)
            .fit(&x)
            .unwrap();
        assert_eq!(single.n_clusters, 1);
        let default = Hdbscan::new(5, 20).fit(&x).unwrap();
        // Without the root being eligible, a single blob never yields exactly one
        // cluster (either 0, or >= 2 sibling sub-clusters).
        assert_ne!(default.n_clusters, 1);
    }

    #[test]
    fn test_hdbscan_leaf_selection() {
        let x = three_blobs_with_noise(7);
        let fitted = Hdbscan::new(5, 10)
            .with_cluster_selection_method(ClusterSelectionMethod::Leaf)
            .fit(&x)
            .unwrap();
        assert!(fitted.n_clusters >= 3);
    }

    #[test]
    fn test_hdbscan_fortran_order_matches_c_order() {
        let x = three_blobs_with_noise(11);
        let v: Vec<f64> = x.t().iter().copied().collect();
        let xf = Array2::from_shape_vec((x.nrows(), x.ncols()).f(), v).unwrap();
        assert!(!xf.is_standard_layout());
        let a = Hdbscan::new(5, 10).fit(&x).unwrap();
        let b = Hdbscan::new(5, 10).fit(&xf).unwrap();
        assert_eq!(a.labels, b.labels);
    }

    #[test]
    fn test_hdbscan_builders_and_tiny_inputs() {
        let h = Hdbscan::default()
            .with_min_samples(3)
            .with_min_cluster_size(4);
        assert_eq!((h.min_samples, h.min_cluster_size), (3, 4));
        let f = h.fit(&array![[1.0, 2.0]]).unwrap();
        assert_eq!(f.labels.len(), 1);
        assert_eq!(f.labels[0], -1.0);
        // min_samples larger than n is clamped.
        let f = Hdbscan::new(50, 2)
            .fit(&array![[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]])
            .unwrap();
        assert_eq!(f.labels.len(), 4);
    }
}
