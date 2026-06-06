//! OPTICS — Ordering Points To Identify the Clustering Structure.
//!
//! Mirrors `sklearn.cluster.OPTICS` with brute-force neighbour lookups and
//! the DBSCAN-like cluster extraction (`extract_dbscan` mode with a single
//! ε threshold). Returns the reachability ordering plus per-sample labels.
//!
//! Algorithm (Ankerst et al. 1999):
//! 1. For each point compute `core_distance(p)` = distance to its
//!    `min_samples`-th nearest neighbour within `eps_inf` (or ∞ if fewer
//!    than `min_samples - 1` neighbours within that radius).
//! 2. Process points in a priority-queue order: from a seed, repeatedly
//!    extract the point with the smallest `reachability_distance` and update
//!    its neighbours' reachability distances.
//! 3. Output: `ordering` (the visit order) and `reachability` (per-point
//!    reachability distance).
//! 4. Cluster extraction (`extract_dbscan`): walk `ordering`, points with
//!    `reachability ≤ eps` join the current cluster; otherwise start a new
//!    cluster or label as noise.

use ndarray::{Array1, Array2};
use rustml_core::{FitUnsupervised, Result, RustMlError};

#[derive(Debug, Clone)]
pub struct Optics {
    pub min_samples: usize,
    pub eps: f64,
    pub eps_max: f64,
}

impl Optics {
    pub fn new(min_samples: usize, eps: f64) -> Self {
        Self {
            min_samples,
            eps,
            eps_max: f64::INFINITY,
        }
    }
    pub fn with_eps_max(mut self, e: f64) -> Self {
        self.eps_max = e;
        self
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedOptics {
    /// Visit order of samples.
    pub ordering: Vec<usize>,
    /// Reachability distance per sample (in original sample-index order).
    pub reachability: Array1<f64>,
    /// Cluster label per sample (-1 = noise).
    pub labels: Array1<f64>,
}

fn euclid(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

impl FitUnsupervised<f64> for Optics {
    type Fitted = FittedOptics;

    fn fit(&self, x: &Array2<f64>) -> Result<Self::Fitted> {
        let n = x.nrows();
        if n == 0 {
            return Err(RustMlError::EmptyInput("empty input".into()));
        }
        if self.min_samples < 1 {
            return Err(RustMlError::InvalidParameter("min_samples >= 1".into()));
        }

        // Pre-compute pairwise distances.
        let mut dist = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            let xi: Vec<f64> = x.row(i).iter().copied().collect();
            for j in (i + 1)..n {
                let xj: Vec<f64> = x.row(j).iter().copied().collect();
                let d = euclid(&xi, &xj);
                dist[i][j] = d;
                dist[j][i] = d;
            }
        }
        // Core distance per point.
        let mut core_dist = vec![f64::INFINITY; n];
        for i in 0..n {
            let mut neigh: Vec<f64> = (0..n).filter(|&j| j != i).map(|j| dist[i][j]).collect();
            neigh.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if neigh.len() >= self.min_samples - 1 + 1 {
                let kth = neigh[self.min_samples - 1];
                if kth <= self.eps_max {
                    core_dist[i] = kth;
                }
            }
        }

        let mut processed = vec![false; n];
        let mut reach = Array1::<f64>::from_elem(n, f64::INFINITY);
        let mut order: Vec<usize> = Vec::with_capacity(n);

        for seed in 0..n {
            if processed[seed] {
                continue;
            }
            // Seed start: own reachability stays INF.
            // Process the connected reachability ordering rooted at seed.
            let mut stack: Vec<usize> = vec![seed];
            while let Some(p) = stack
                .iter()
                .filter(|&&q| !processed[q])
                .min_by(|&&a, &&b| reach[a].partial_cmp(&reach[b]).unwrap())
                .copied()
            {
                processed[p] = true;
                order.push(p);
                // Remove p from stack.
                stack.retain(|&q| q != p);

                if !core_dist[p].is_finite() {
                    continue;
                }
                // Update reachability of p's neighbours within eps_max.
                for q in 0..n {
                    if q == p || processed[q] {
                        continue;
                    }
                    let d = dist[p][q];
                    if d > self.eps_max {
                        continue;
                    }
                    let new_r = core_dist[p].max(d);
                    if new_r < reach[q] {
                        reach[q] = new_r;
                        if !stack.contains(&q) {
                            stack.push(q);
                        }
                    }
                }
            }
        }

        // DBSCAN-like extraction: scan `order`. Point with `reach > eps` starts a
        // new cluster only if its `core_dist <= eps` (otherwise noise).
        let mut labels = Array1::<f64>::from_elem(n, -1.0);
        let mut current_cluster = -1.0_f64;
        for &p in &order {
            if reach[p] > self.eps {
                // Possible cluster start.
                if core_dist[p] <= self.eps {
                    current_cluster += 1.0;
                    labels[p] = current_cluster;
                } else {
                    labels[p] = -1.0;
                }
            } else if current_cluster >= 0.0 {
                labels[p] = current_cluster;
            }
        }

        Ok(FittedOptics {
            ordering: order,
            reachability: reach,
            labels,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_optics_two_blobs() {
        let x = array![
            [0.0_f64, 0.0],
            [0.1, 0.1],
            [-0.1, 0.2],
            [0.1, -0.1],
            [10.0, 10.0],
            [10.1, 9.9],
            [9.8, 10.2],
            [10.2, 9.8],
            [50.0, 50.0], // far outlier → noise (-1)
        ];
        let fitted = Optics::new(2, 1.0).fit(&x).unwrap();
        let labels = &fitted.labels;
        // First 4 should share a label; next 4 share a different label; the
        // outlier should be noise (-1) or its own.
        let l0 = labels[0];
        for i in 1..4 {
            assert_eq!(labels[i], l0);
        }
        let l4 = labels[4];
        for i in 5..8 {
            assert_eq!(labels[i], l4);
        }
        assert_ne!(l0, l4);
        // Outlier:
        assert!(
            labels[8] == -1.0 || labels[8] != l0 && labels[8] != l4,
            "outlier label = {}",
            labels[8]
        );
    }
}
