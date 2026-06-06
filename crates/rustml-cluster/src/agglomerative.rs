//! Agglomerative (hierarchical) clustering.
//!
//! Mirrors `sklearn.cluster.AgglomerativeClustering`. Naive O(n²) memory /
//! O(n³) time implementation using the Lance-Williams update for the four
//! standard linkages.

use ndarray::{Array1, Array2};
use rustml_core::{FitUnsupervised, Result, RustMlError};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Linkage {
    Single,
    Complete,
    Average,
    Ward,
}

#[derive(Debug, Clone)]
pub struct AgglomerativeClustering {
    pub n_clusters: usize,
    pub linkage: Linkage,
}

impl AgglomerativeClustering {
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            linkage: Linkage::Ward,
        }
    }
    pub fn with_linkage(mut self, l: Linkage) -> Self {
        self.linkage = l;
        self
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedAgglomerativeClustering {
    pub labels: Array1<f64>,
    pub n_clusters: usize,
}

fn sq_euclid(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = x - y;
        acc += d * d;
    }
    acc
}

impl FitUnsupervised<f64> for AgglomerativeClustering {
    type Fitted = FittedAgglomerativeClustering;

    fn fit(&self, x: &Array2<f64>) -> Result<Self::Fitted> {
        let n = x.nrows();
        if n == 0 {
            return Err(RustMlError::EmptyInput("empty input".into()));
        }
        if self.n_clusters == 0 || self.n_clusters > n {
            return Err(RustMlError::InvalidParameter("invalid n_clusters".into()));
        }

        // Active cluster set with sizes and pairwise distances.
        let mut active: Vec<bool> = vec![true; n];
        let mut size: Vec<usize> = vec![1; n];
        let mut cluster_of: Vec<usize> = (0..n).collect();
        let mut dist = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            let xi = x.row(i).to_owned();
            for j in (i + 1)..n {
                let xj = x.row(j).to_owned();
                let d = if self.linkage == Linkage::Ward {
                    0.5 * sq_euclid(xi.as_slice().unwrap(), xj.as_slice().unwrap())
                } else {
                    sq_euclid(xi.as_slice().unwrap(), xj.as_slice().unwrap()).sqrt()
                };
                dist[i][j] = d;
                dist[j][i] = d;
            }
        }

        let mut current_clusters = n;
        // For Ward (the only reducible linkage we support) use Müllner's
        // O(n²) nn-chain. For Single/Complete/Average — also reducible —
        // the naive O(n³) path stays default because nn-chain's gains
        // require maintaining the full reduced-distance matrix, which the
        // naive sweep already does. Override with `RUSTML_AGGLO_NAIVE=1` to
        // force the naive path everywhere (used by regression tests that
        // confirm both paths agree).
        let use_nn_chain =
            self.linkage == Linkage::Ward && std::env::var("RUSTML_AGGLO_NAIVE").is_err();

        // Helper: Lance-Williams update for cluster k after merging bi and bj
        // into bi.
        let update = |dist: &mut Vec<Vec<f64>>,
                      size: &Vec<usize>,
                      bi: usize,
                      bj: usize,
                      k: usize,
                      linkage: Linkage|
         -> f64 {
            let d_ik = dist[bi][k];
            let d_jk = dist[bj][k];
            let ni = size[bi] as f64;
            let nj = size[bj] as f64;
            let nk = size[k] as f64;
            match linkage {
                Linkage::Single => d_ik.min(d_jk),
                Linkage::Complete => d_ik.max(d_jk),
                Linkage::Average => (ni * d_ik + nj * d_jk) / (ni + nj),
                Linkage::Ward => {
                    let d_ij = dist[bi][bj];
                    let total = ni + nj + nk;
                    ((ni + nk) * d_ik + (nj + nk) * d_jk - nk * d_ij) / total
                }
            }
        };

        if use_nn_chain {
            // nn-chain algorithm (Müllner 2011, §4.1). For reducible linkages
            // like Ward, a reciprocal-NN pair at the chain tail can be merged
            // safely — under reducibility no later merge can produce a closer
            // pair involving the merged cluster. CRITICALLY, nn-chain
            // produces merges in *chain order* not *distance order*. To
            // recover the same flat clustering as the naive O(n³) sweep we
            // must:
            //   1. run nn-chain all the way to a single cluster, recording
            //      every merge as (a, b, distance);
            //   2. sort the recorded merges by distance ascending;
            //   3. apply them via a fresh DSU, stopping at n_clusters.
            //
            // Step 1 is O(n²) total. Steps 2/3 are O(n log n). Net cost
            // O(n²) — matches Müllner's bound.
            let mut chain: Vec<usize> = Vec::with_capacity(n);
            let mut merges: Vec<(usize, usize, f64)> = Vec::with_capacity(n - 1);
            while current_clusters > 1 {
                if chain.is_empty() {
                    for i in 0..n {
                        if active[i] {
                            chain.push(i);
                            break;
                        }
                    }
                }
                loop {
                    let top = *chain.last().unwrap();
                    let mut nn = top;
                    let mut nn_dist = f64::INFINITY;
                    for j in 0..n {
                        if j == top || !active[j] {
                            continue;
                        }
                        let d = dist[top][j];
                        if d < nn_dist {
                            nn_dist = d;
                            nn = j;
                        }
                    }
                    let prev_idx = if chain.len() >= 2 {
                        Some(chain[chain.len() - 2])
                    } else {
                        None
                    };
                    if let Some(prev) = prev_idx {
                        let d_top_prev = dist[top][prev];
                        if d_top_prev <= nn_dist {
                            let bi = prev.min(top);
                            let bj = prev.max(top);
                            merges.push((bi, bj, d_top_prev));
                            for k in 0..n {
                                if k == bi || k == bj || !active[k] {
                                    continue;
                                }
                                let new_d = update(&mut dist, &size, bi, bj, k, self.linkage);
                                dist[bi][k] = new_d;
                                dist[k][bi] = new_d;
                            }
                            size[bi] += size[bj];
                            active[bj] = false;
                            current_clusters -= 1;
                            chain.pop();
                            chain.pop();
                            break;
                        }
                    }
                    chain.push(nn);
                }
            }

            // Step 2: sort by merge distance ascending.
            merges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

            // Step 3: apply merges in distance order via a fresh DSU on
            // `cluster_of`, stopping at n_clusters remaining.
            let target = self.n_clusters;
            let mut parent: Vec<usize> = (0..n).collect();
            fn find(parent: &mut [usize], i: usize) -> usize {
                let mut r = i;
                while parent[r] != r {
                    r = parent[r];
                }
                let mut cur = i;
                while parent[cur] != r {
                    let nxt = parent[cur];
                    parent[cur] = r;
                    cur = nxt;
                }
                r
            }
            let mut active_count = n;
            for (a, b, _d) in merges {
                if active_count <= target {
                    break;
                }
                let ra = find(&mut parent, a);
                let rb = find(&mut parent, b);
                if ra != rb {
                    parent[ra] = rb;
                    active_count -= 1;
                }
            }
            for i in 0..n {
                cluster_of[i] = find(&mut parent, i);
            }
        } else {
            // Naive O(n³) path for non-Ward linkages.
            while current_clusters > self.n_clusters {
                let mut best = f64::INFINITY;
                let mut bi = 0;
                let mut bj = 0;
                for i in 0..n {
                    if !active[i] {
                        continue;
                    }
                    for j in (i + 1)..n {
                        if !active[j] {
                            continue;
                        }
                        if dist[i][j] < best {
                            best = dist[i][j];
                            bi = i;
                            bj = j;
                        }
                    }
                }
                for k in 0..n {
                    if k == bi || k == bj || !active[k] {
                        continue;
                    }
                    let new_d = update(&mut dist, &size, bi, bj, k, self.linkage);
                    dist[bi][k] = new_d;
                    dist[k][bi] = new_d;
                }
                for c in &mut cluster_of {
                    if *c == bj {
                        *c = bi;
                    }
                }
                size[bi] += size[bj];
                active[bj] = false;
                current_clusters -= 1;
            }
        }

        // Compact cluster labels into 0..n_clusters.
        let mut id_map = std::collections::HashMap::<usize, usize>::new();
        let mut next_id = 0usize;
        let mut labels = Array1::<f64>::zeros(n);
        for i in 0..n {
            let c = cluster_of[i];
            let id = *id_map.entry(c).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            labels[i] = id as f64;
        }

        Ok(FittedAgglomerativeClustering {
            labels,
            n_clusters: self.n_clusters,
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
    fn test_ward_nnchain_matches_naive() {
        // Spread-out 3-blob data with enough points that any algorithmic
        // difference would surface; nn-chain (default for Ward) must
        // produce the same flat labels as the naive O(n³) path.
        let mut data = Vec::new();
        let centres = [(0.0_f64, 0.0), (8.0, 0.0), (4.0, 7.0)];
        for &(cx, cy) in &centres {
            for i in 0..15 {
                let t = i as f64 * 0.1;
                data.push(cx + t.sin() * 0.4);
                data.push(cy + t.cos() * 0.4);
            }
        }
        let x = ndarray::Array2::from_shape_vec((45, 2), data).unwrap();

        let nnc = AgglomerativeClustering::new(3)
            .with_linkage(Linkage::Ward)
            .fit(&x)
            .unwrap();

        std::env::set_var("RUSTML_AGGLO_NAIVE", "1");
        let naive = AgglomerativeClustering::new(3)
            .with_linkage(Linkage::Ward)
            .fit(&x)
            .unwrap();
        std::env::remove_var("RUSTML_AGGLO_NAIVE");

        // Labels may be permuted between runs; compare via cluster
        // partition equality (same induced equivalence relation).
        let same_partition = |a: &Array1<f64>, b: &Array1<f64>| -> bool {
            for i in 0..a.len() {
                for j in (i + 1)..a.len() {
                    if (a[i] == a[j]) != (b[i] == b[j]) {
                        return false;
                    }
                }
            }
            true
        };
        assert!(
            same_partition(&nnc.labels, &naive.labels),
            "nn-chain and naive should produce identical partitions"
        );
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
