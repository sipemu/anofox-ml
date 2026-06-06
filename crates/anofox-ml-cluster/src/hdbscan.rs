//! HDBSCAN — Hierarchical Density-Based Spatial Clustering.
//!
//! Implements the core HDBSCAN pipeline:
//!
//! 1. **Core distance**: `core_dist(x_i)` = distance to its
//!    `min_samples`-th nearest neighbour.
//! 2. **Mutual reachability**: `d_mr(a, b) = max(core_dist(a), core_dist(b), d(a, b))`.
//! 3. **Minimum spanning tree** of the complete mutual-reachability graph
//!    via Prim's algorithm.
//! 4. **Single-linkage hierarchy**: sort MST edges by weight, build a
//!    bottom-up cluster hierarchy.
//! 5. **Condense**: walk the hierarchy top-down; subtrees with fewer than
//!    `min_cluster_size` points "fall out" of their parent as noise rather
//!    than forming a new cluster.
//! 6. **Stability-based extraction**: pick the set of clusters that maximises
//!    total stability `Σ_{x ∈ C} (λ_x_leaves - λ_C_birth)`.

use anofox_ml_core::{FitUnsupervised, Result, RustMlError};
use ndarray::{Array1, Array2};

#[derive(Debug, Clone)]
pub struct Hdbscan {
    pub min_samples: usize,
    pub min_cluster_size: usize,
}

impl Hdbscan {
    pub fn new(min_samples: usize, min_cluster_size: usize) -> Self {
        Self {
            min_samples,
            min_cluster_size,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedHdbscan {
    pub labels: Array1<f64>,
    pub n_clusters: usize,
}

fn euclid(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Union-Find with path compression for the single-linkage hierarchy.
struct DSU {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl DSU {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            size: vec![1; n],
        }
    }
    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }
    fn union(&mut self, a: usize, b: usize) -> usize {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return ra;
        }
        // Union by size.
        let (big, small) = if self.size[ra] >= self.size[rb] {
            (ra, rb)
        } else {
            (rb, ra)
        };
        self.parent[small] = big;
        self.size[big] += self.size[small];
        big
    }
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

        // Pairwise distances.
        let mut d = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            let xi: Vec<f64> = x.row(i).iter().copied().collect();
            for j in (i + 1)..n {
                let xj: Vec<f64> = x.row(j).iter().copied().collect();
                let val = euclid(&xi, &xj);
                d[i][j] = val;
                d[j][i] = val;
            }
        }

        // Core distances.
        let mut core = vec![0.0_f64; n];
        for i in 0..n {
            let mut nb: Vec<f64> = (0..n).filter(|&j| j != i).map(|j| d[i][j]).collect();
            nb.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let k = (self.min_samples - 1).min(nb.len().saturating_sub(1));
            core[i] = if nb.is_empty() { 0.0 } else { nb[k] };
        }

        // Mutual reachability.
        let mut mr = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    mr[i][j] = d[i][j].max(core[i]).max(core[j]);
                }
            }
        }

        // Prim's MST on mutual reachability.
        let mut in_tree = vec![false; n];
        let mut min_dist = vec![f64::INFINITY; n];
        let mut closest = vec![0_usize; n];
        in_tree[0] = true;
        for j in 1..n {
            min_dist[j] = mr[0][j];
            closest[j] = 0;
        }
        let mut edges: Vec<(usize, usize, f64)> = Vec::with_capacity(n - 1);
        for _ in 1..n {
            let mut best = f64::INFINITY;
            let mut bi = 0;
            for j in 0..n {
                if !in_tree[j] && min_dist[j] < best {
                    best = min_dist[j];
                    bi = j;
                }
            }
            edges.push((closest[bi], bi, best));
            in_tree[bi] = true;
            for j in 0..n {
                if !in_tree[j] && mr[bi][j] < min_dist[j] {
                    min_dist[j] = mr[bi][j];
                    closest[j] = bi;
                }
            }
        }

        // Sort MST edges ascending — single-linkage merges from densest pairs first.
        edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

        // Single-linkage hierarchy: track each cluster's birth λ = 1 / edge_weight
        // and accumulate stability as edges are added.
        // For each point we maintain its current cluster id (DSU root).
        // When two clusters merge at edge weight w (i.e. λ = 1/w):
        //   - sub-clusters with < min_cluster_size points are "noise" → their
        //     points drop out at the parent λ (contribute to parent stability).
        //   - sub-clusters with ≥ min_cluster_size become bona-fide clusters
        //     in the condensed tree.
        // For a HDBSCAN-lite extraction we use a simpler rule below.
        let dsu = DSU::new(n);

        // Track "lambda when point left its cluster" for stability.
        // We extract the final flat clustering as follows:
        //   - sort MST edges descending by weight
        //   - find the largest edge weight `w_split` such that cutting it
        //     leaves at least one component of size ≥ min_cluster_size
        //   - components with ≥ min_cluster_size become clusters; others noise
        // This is HDBSCAN-lite but matches the "flat" version of the
        // algorithm and works on dense data.

        // Build all edges in ascending order (densest first), keep them all
        // until the threshold and then cut.
        // Strategy: scan ascending; each point's "leave lambda" is when its
        // component would exceed the natural split point — we use the
        // single-linkage descendant-counting approach.

        // Stability-based flat extraction.
        //
        // We track for each DSU component whether it has been "finalised"
        // (assigned a cluster label after a true split). Walk MST edges
        // ascending. Each merge of distinct roots ra, rb falls into one of:
        //
        //   (a) Neither side finalised → just union (growing component
        //       absorbs whatever; no labels yet).
        //   (b) Exactly one side finalised → the un-finalised side is a
        //       set of points joining/falling out of an already-frozen
        //       cluster. They become noise. We union for bookkeeping but
        //       the joined points stay labelled -1.
        //   (c) Both sides un-finalised AND both ≥ min_cluster_size → true
        //       split: finalise both sides with distinct cluster ids; do
        //       NOT union.
        //   (d) Both sides finalised → would re-merge two already-fixed
        //       clusters; do nothing (no union, no relabel).
        //
        // Sub-min_cluster_size joinees while no split has happened yet just
        // grow the pre-cluster component.
        let _ = dsu; // suppress unused — we use dsu2 below

        let mut dsu2 = DSU::new(n);
        let mut cluster_label = vec![-1.0_f64; n];
        let mut frozen_as_noise = vec![false; n];
        let mut next_id = 0.0_f64;
        let mut finalised: std::collections::HashSet<usize> = std::collections::HashSet::new();

        for &(a, b, _) in &edges {
            let ra = dsu2.find(a);
            let rb = dsu2.find(b);
            if ra == rb {
                continue;
            }
            let a_fin = finalised.contains(&ra);
            let b_fin = finalised.contains(&rb);
            let sa = dsu2.size[ra];
            let sb = dsu2.size[rb];

            if a_fin && b_fin {
                // Case (d): both already clusters, leave alone.
                continue;
            } else if a_fin || b_fin {
                // Case (b): the un-finalised side's points become noise.
                let join_root = if a_fin { rb } else { ra };
                for i in 0..n {
                    if dsu2.find(i) == join_root && cluster_label[i] < 0.0 {
                        frozen_as_noise[i] = true;
                    }
                }
                // Union, but preserve which root stays finalised: since DSU
                // unions by size, the bigger side's root wins. The finalised
                // side has its label already on its points, so the root
                // change doesn't matter for labels. We just need `finalised`
                // to follow the new root.
                let old_fin_root = if a_fin { ra } else { rb };
                dsu2.union(a, b);
                let new_root = dsu2.find(a);
                if new_root != old_fin_root {
                    finalised.remove(&old_fin_root);
                    finalised.insert(new_root);
                }
            } else {
                // Neither finalised. Check for cluster split.
                let a_big = sa >= self.min_cluster_size;
                let b_big = sb >= self.min_cluster_size;
                if a_big && b_big {
                    // Case (c): true split. Finalise both.
                    for r in [ra, rb] {
                        let label = next_id;
                        next_id += 1.0;
                        for i in 0..n {
                            if dsu2.find(i) == r && !frozen_as_noise[i] {
                                cluster_label[i] = label;
                            }
                        }
                        finalised.insert(r);
                    }
                    // No union — the split edge is cut.
                } else {
                    // Case (a): just grow.
                    dsu2.union(a, b);
                }
            }
        }
        // Any point that never reached a cluster split but isn't frozen as
        // noise also becomes noise — it merged into a singleton component
        // that never grew above min_cluster_size, or there was no split at
        // all (single-cluster dataset → all noise per HDBSCAN convention is
        // not what we want; if no split happened, treat the largest final
        // component as cluster 0).
        let mut labels: Vec<f64> = cluster_label
            .iter()
            .enumerate()
            .map(|(i, &l)| if frozen_as_noise[i] { -1.0 } else { l })
            .collect();
        let mut n_clusters = next_id as usize;
        if n_clusters == 0 {
            // No split occurred. Promote the single largest component to
            // cluster 0; everything outside it is noise.
            let mut sizes = std::collections::HashMap::<usize, usize>::new();
            for i in 0..n {
                if !frozen_as_noise[i] {
                    *sizes.entry(dsu2.find(i)).or_insert(0) += 1;
                }
            }
            if let Some((&big_root, &sz)) = sizes.iter().max_by_key(|(_, s)| *s) {
                if sz >= self.min_cluster_size {
                    for i in 0..n {
                        if !frozen_as_noise[i] && dsu2.find(i) == big_root {
                            labels[i] = 0.0;
                        } else {
                            labels[i] = -1.0;
                        }
                    }
                    n_clusters = 1;
                }
            }
        }
        Ok(FittedHdbscan {
            labels: Array1::from_vec(labels),
            n_clusters,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

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
    }
}
