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
        Self { n_clusters, linkage: Linkage::Ward }
    }
    pub fn with_linkage(mut self, l: Linkage) -> Self { self.linkage = l; self }
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
        // nn-chain disabled: the chain-popping logic has a known correctness
        // bug when intermediate merges invalidate the chain invariant; reverts
        // to the naive O(n³) loop until a tested implementation lands.
        let use_nn_chain = false;

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
            // nn-chain algorithm (Müllner 2011). For Ward (reducible),
            // walking a nearest-neighbour chain finds reciprocal NN pairs
            // that are guaranteed to be the globally-closest active pair.
            // Total cost O(n²) instead of O(n³).
            let mut chain: Vec<usize> = Vec::with_capacity(n);
            let mut chain_dists: Vec<f64> = Vec::with_capacity(n);
            while current_clusters > self.n_clusters {
                if chain.is_empty() {
                    // Start chain from any active cluster.
                    for i in 0..n {
                        if active[i] {
                            chain.push(i);
                            chain_dists.push(f64::INFINITY); // sentinel
                            break;
                        }
                    }
                }
                let top = *chain.last().unwrap();
                // Find nearest active neighbour of `top`.
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
                // Reciprocal NN: chain tail's NN is the previous chain element.
                let prev = if chain.len() >= 2 {
                    Some(chain[chain.len() - 2])
                } else {
                    None
                };
                if Some(nn) == prev {
                    // Merge top and prev.
                    let bi = prev.unwrap().min(top);
                    let bj = prev.unwrap().max(top);
                    // Apply Lance-Williams on bi's row (merge bj into bi).
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
                    // Pop top + prev from chain.
                    chain.pop();
                    chain.pop();
                    chain_dists.pop();
                    chain_dists.pop();
                } else {
                    chain.push(nn);
                    chain_dists.push(nn_dist);
                }
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
            [0.0, 0.0], [0.5, 0.1], [-0.3, 0.2], [0.1, -0.2],
            [10.0, 10.0], [10.5, 10.1], [9.9, 9.8], [10.1, 9.9],
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
        let x = array![
            [0.0], [0.1], [10.0], [10.1], [100.0],
        ];
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
