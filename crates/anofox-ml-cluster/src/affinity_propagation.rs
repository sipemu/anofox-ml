//! Affinity Propagation.
//!
//! Mirrors `sklearn.cluster.AffinityPropagation`. Message-passing on a
//! similarity matrix `s_{i,k} = -||x_i - x_k||²` until exemplars stabilise.

use anofox_ml_core::{FitUnsupervised, Result, RustMlError};
use ndarray::{Array1, Array2};

#[derive(Debug, Clone)]
pub struct AffinityPropagation {
    pub damping: f64,
    pub max_iter: usize,
    pub convergence_iter: usize,
    pub preference: Option<f64>,
    /// If `Some(k)`, sparsify the similarity matrix to each point's k
    /// nearest neighbours (symmetrised). Mirrors sklearn's
    /// `affinity='precomputed_nearest_neighbors'`. Cuts memory from O(n²)
    /// to O(n·k) and per-iteration cost from O(n²) to O(n·k).
    pub n_neighbors: Option<usize>,
}

impl AffinityPropagation {
    pub fn new() -> Self {
        Self {
            damping: 0.9,
            max_iter: 200,
            convergence_iter: 15,
            preference: None,
            n_neighbors: None,
        }
    }
    pub fn with_damping(mut self, d: f64) -> Self {
        self.damping = d;
        self
    }
    pub fn with_preference(mut self, p: f64) -> Self {
        self.preference = Some(p);
        self
    }
    pub fn with_n_neighbors(mut self, k: usize) -> Self {
        self.n_neighbors = Some(k);
        self
    }
}

impl Default for AffinityPropagation {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedAffinityPropagation {
    pub labels: Array1<f64>,
    pub cluster_centers_indices: Vec<usize>,
}

impl FitUnsupervised<f64> for AffinityPropagation {
    type Fitted = FittedAffinityPropagation;

    fn fit(&self, x: &Array2<f64>) -> Result<Self::Fitted> {
        let n = x.nrows();
        if n == 0 {
            return Err(RustMlError::EmptyInput("empty input".into()));
        }
        if let Some(k) = self.n_neighbors {
            if k == 0 {
                return Err(RustMlError::InvalidParameter(
                    "n_neighbors must be ≥ 1".into(),
                ));
            }
            return self.fit_sparse(x, k);
        }

        // Similarity matrix s_{i,k} = -||x_i - x_k||²
        let mut s = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let mut sd = 0.0;
                for k in 0..x.ncols() {
                    let d = x[[i, k]] - x[[j, k]];
                    sd += d * d;
                }
                s[[i, j]] = -sd;
            }
        }
        // Diagonal: preference = median of off-diagonals (sklearn default).
        let pref = self.preference.unwrap_or_else(|| {
            let mut offdiag = Vec::with_capacity(n * (n - 1));
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        offdiag.push(s[[i, j]]);
                    }
                }
            }
            offdiag.sort_by(|a, b| a.partial_cmp(b).unwrap());
            offdiag[offdiag.len() / 2]
        });
        for i in 0..n {
            s[[i, i]] = pref;
        }
        // Tie-breaking noise. sklearn adds tiny deterministic positive noise
        // to break degeneracy. We use a hash-style scrambler that's non-negative
        // and stable across runs.
        for i in 0..n {
            for j in 0..n {
                let mix = ((i.wrapping_mul(2654435761) ^ j.wrapping_mul(40503)) & 0xFFFF) as f64;
                s[[i, j]] += 1e-12 * (mix / 65536.0);
            }
        }

        let mut r = Array2::<f64>::zeros((n, n));
        let mut a = Array2::<f64>::zeros((n, n));
        let mut last_exemplars: Option<Vec<bool>> = None;
        let mut converge_count = 0usize;

        for _iter in 0..self.max_iter {
            // Update responsibilities r_{i,k} = s_{i,k} - max_{k' != k}(a_{i,k'} + s_{i,k'})
            let mut new_r = Array2::<f64>::zeros((n, n));
            for i in 0..n {
                // Find top two indices by (a + s) over k.
                let mut top1 = f64::NEG_INFINITY;
                let mut top1_k = 0usize;
                let mut top2 = f64::NEG_INFINITY;
                for k in 0..n {
                    let v = a[[i, k]] + s[[i, k]];
                    if v > top1 {
                        top2 = top1;
                        top1 = v;
                        top1_k = k;
                    } else if v > top2 {
                        top2 = v;
                    }
                }
                for k in 0..n {
                    let other_max = if k == top1_k { top2 } else { top1 };
                    new_r[[i, k]] = s[[i, k]] - other_max;
                }
            }
            // Damping.
            for i in 0..n {
                for k in 0..n {
                    r[[i, k]] = self.damping * r[[i, k]] + (1.0 - self.damping) * new_r[[i, k]];
                }
            }

            // Update availabilities.
            let mut new_a = Array2::<f64>::zeros((n, n));
            for k in 0..n {
                // For i != k: a_{i,k} = min(0, r_{k,k} + sum_{i' != i, k} max(0, r_{i', k}))
                let mut sum_pos = 0.0;
                for ip in 0..n {
                    if ip == k {
                        continue;
                    }
                    sum_pos += r[[ip, k]].max(0.0);
                }
                for i in 0..n {
                    if i == k {
                        // a_{k,k} = sum_{i' != k} max(0, r_{i', k})
                        new_a[[i, k]] = sum_pos;
                    } else {
                        let excl = r[[i, k]].max(0.0);
                        let v = r[[k, k]] + (sum_pos - excl);
                        new_a[[i, k]] = v.min(0.0);
                    }
                }
            }
            for i in 0..n {
                for k in 0..n {
                    a[[i, k]] = self.damping * a[[i, k]] + (1.0 - self.damping) * new_a[[i, k]];
                }
            }

            // Exemplars: indices k where a_{k,k} + r_{k,k} > 0.
            let exemplars: Vec<bool> = (0..n).map(|k| a[[k, k]] + r[[k, k]] > 0.0).collect();
            if let Some(prev) = &last_exemplars {
                if prev == &exemplars {
                    converge_count += 1;
                    if converge_count >= self.convergence_iter {
                        break;
                    }
                } else {
                    converge_count = 0;
                }
            }
            last_exemplars = Some(exemplars);
        }

        // Final cluster centers and labels.
        let exemplars = last_exemplars.unwrap_or_else(|| vec![true; n]);
        let cluster_centers_indices: Vec<usize> = exemplars
            .iter()
            .enumerate()
            .filter(|(_, &b)| b)
            .map(|(i, _)| i)
            .collect();
        if cluster_centers_indices.is_empty() {
            // Pathological — return single cluster.
            return Ok(FittedAffinityPropagation {
                labels: Array1::<f64>::zeros(n),
                cluster_centers_indices: vec![0],
            });
        }

        let mut labels = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut best = f64::NEG_INFINITY;
            let mut best_c = 0;
            for (c, &k) in cluster_centers_indices.iter().enumerate() {
                let sim = s[[i, k]];
                if sim > best {
                    best = sim;
                    best_c = c;
                }
            }
            labels[i] = best_c as f64;
        }
        Ok(FittedAffinityPropagation {
            labels,
            cluster_centers_indices,
        })
    }
}

impl AffinityPropagation {
    /// Sparse k-NN affinity propagation. Each point keeps:
    ///   - a self-loop with similarity = preference
    ///   - edges to its `k` nearest neighbours (symmetrised)
    ///
    /// The r and a matrices are stored only on these edges. The
    /// max-over-N(i) and sum-over-N⁻¹(k) updates run in O(degree) per row.
    fn fit_sparse(&self, x: &Array2<f64>, k: usize) -> Result<FittedAffinityPropagation> {
        let n = x.nrows();
        let d = x.ncols();
        let k = k.min(n.saturating_sub(1));
        if k == 0 {
            return Err(RustMlError::InvalidParameter(
                "n_neighbors must be ≥ 1 and < n".into(),
            ));
        }

        // ─── Build symmetric k-NN graph of similarities ─────────────────
        // s_{i,j} = -||x_i - x_j||² + tiny tie-breaking noise.
        let sq = |i: usize, j: usize| -> f64 {
            let mut s = 0.0;
            for c in 0..d {
                let dv = x[[i, c]] - x[[j, c]];
                s += dv * dv;
            }
            s
        };
        let noise = |i: usize, j: usize| -> f64 {
            let mix = ((i.wrapping_mul(2654435761) ^ j.wrapping_mul(40503)) & 0xFFFF) as f64;
            1e-12 * (mix / 65536.0)
        };

        // Per-point k-NN via bounded heap.
        use std::cmp::Ordering;
        #[derive(Clone, Copy)]
        struct DPair(usize, f64);
        impl Ord for DPair {
            fn cmp(&self, other: &Self) -> Ordering {
                self.1.partial_cmp(&other.1).unwrap_or(Ordering::Equal)
            }
        }
        impl PartialOrd for DPair {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Eq for DPair {}
        impl PartialEq for DPair {
            fn eq(&self, other: &Self) -> bool {
                self.1 == other.1
            }
        }

        // out_nbrs[i] = sorted list of (j, s_{i,j}) for j in NN(i) ∪ {i}.
        // Use a HashSet of edges to enforce symmetry, then materialise.
        let mut edge_set: std::collections::HashSet<(usize, usize)> =
            std::collections::HashSet::new();
        for i in 0..n {
            let mut heap: std::collections::BinaryHeap<DPair> =
                std::collections::BinaryHeap::with_capacity(k);
            for j in 0..n {
                if j == i {
                    continue;
                }
                let dd = sq(i, j);
                if heap.len() < k {
                    heap.push(DPair(j, dd));
                } else if let Some(top) = heap.peek() {
                    if dd < top.1 {
                        heap.pop();
                        heap.push(DPair(j, dd));
                    }
                }
            }
            for p in heap.into_iter() {
                edge_set.insert((i, p.0));
                edge_set.insert((p.0, i)); // symmetrise
            }
        }

        // Compute preference (median of off-diagonal similarities on the
        // sparse graph if not provided).
        let pref = self.preference.unwrap_or_else(|| {
            let mut vals: Vec<f64> = edge_set.iter().map(|&(i, j)| -sq(i, j)).collect();
            if vals.is_empty() {
                return 0.0;
            }
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            vals[vals.len() / 2]
        });

        // Materialise per-row neighbours including the self-loop. Edges are
        // sorted by neighbour index for deterministic iteration. Each row
        // entry carries (k, s, r, a) co-located for cache locality.
        let mut row_nbrs: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(i, j) in &edge_set {
            row_nbrs[i].push(j);
        }
        for i in 0..n {
            row_nbrs[i].push(i);
            row_nbrs[i].sort_unstable();
            row_nbrs[i].dedup();
        }

        // sim[i] / r[i] / a[i] are parallel arrays over row_nbrs[i].
        let mut sim: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                row_nbrs[i]
                    .iter()
                    .map(|&j| {
                        if i == j {
                            pref + noise(i, j)
                        } else {
                            -sq(i, j) + noise(i, j)
                        }
                    })
                    .collect()
            })
            .collect();
        let _ = &mut sim;
        let sim = sim;

        let mut r: Vec<Vec<f64>> = row_nbrs.iter().map(|v| vec![0.0_f64; v.len()]).collect();
        let mut a: Vec<Vec<f64>> = row_nbrs.iter().map(|v| vec![0.0_f64; v.len()]).collect();

        // Build column index: col_entries[k] = list of (i, position-in-row-i)
        // so a-update can sweep through points sending positive r to k.
        let mut col_entries: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
        for i in 0..n {
            for (pos, &j) in row_nbrs[i].iter().enumerate() {
                col_entries[j].push((i, pos));
            }
        }
        // For each k we need the (i, position) of the self-loop row (i == k).
        let mut self_idx_in_col: Vec<Option<usize>> = vec![None; n];
        for k in 0..n {
            for (idx, &(i, _)) in col_entries[k].iter().enumerate() {
                if i == k {
                    self_idx_in_col[k] = Some(idx);
                    break;
                }
            }
        }
        // For each i, position of i in row_nbrs[i] (self entry).
        let mut self_pos_in_row: Vec<usize> = vec![0; n];
        for i in 0..n {
            self_pos_in_row[i] = row_nbrs[i]
                .iter()
                .position(|&j| j == i)
                .expect("row must contain self-loop");
        }

        let mut last_exemplars: Option<Vec<bool>> = None;
        let mut converge_count = 0usize;

        for _iter in 0..self.max_iter {
            // ─── r-update ──────────────────────────────────────────────
            // r_{i,k} = s_{i,k} - max_{k' ∈ N(i), k' ≠ k}(a_{i,k'} + s_{i,k'})
            for i in 0..n {
                let row = &row_nbrs[i];
                let m = row.len();
                let mut top1 = f64::NEG_INFINITY;
                let mut top1_idx = 0usize;
                let mut top2 = f64::NEG_INFINITY;
                for p in 0..m {
                    let v = a[i][p] + sim[i][p];
                    if v > top1 {
                        top2 = top1;
                        top1 = v;
                        top1_idx = p;
                    } else if v > top2 {
                        top2 = v;
                    }
                }
                for p in 0..m {
                    let other_max = if p == top1_idx { top2 } else { top1 };
                    let new_r = sim[i][p] - other_max;
                    r[i][p] = self.damping * r[i][p] + (1.0 - self.damping) * new_r;
                }
            }

            // ─── a-update ──────────────────────────────────────────────
            // For each k:
            //   sum_pos = Σ_{i' ∈ N⁻¹(k), i' ≠ k} max(0, r_{i', k})
            //   For i ≠ k: a_{i,k} = min(0, r_{k,k} + (sum_pos - max(0, r_{i,k})))
            //   For i = k: a_{k,k} = sum_pos
            for k in 0..n {
                let col = &col_entries[k];
                // r_{k,k} lookup.
                let r_kk = match self_idx_in_col[k] {
                    Some(idx) => {
                        let (ii, pos) = col[idx];
                        debug_assert_eq!(ii, k);
                        r[ii][pos]
                    }
                    None => 0.0,
                };
                let mut sum_pos = 0.0;
                for &(ip, pos) in col {
                    if ip == k {
                        continue;
                    }
                    sum_pos += r[ip][pos].max(0.0);
                }
                for &(i, pos) in col {
                    let new_a = if i == k {
                        sum_pos
                    } else {
                        let excl = r[i][pos].max(0.0);
                        let v = r_kk + (sum_pos - excl);
                        v.min(0.0)
                    };
                    a[i][pos] = self.damping * a[i][pos] + (1.0 - self.damping) * new_a;
                }
            }

            // ─── exemplar check ────────────────────────────────────────
            let exemplars: Vec<bool> = (0..n)
                .map(|k| {
                    let p = self_pos_in_row[k];
                    a[k][p] + r[k][p] > 0.0
                })
                .collect();
            if let Some(prev) = &last_exemplars {
                if prev == &exemplars {
                    converge_count += 1;
                    if converge_count >= self.convergence_iter {
                        break;
                    }
                } else {
                    converge_count = 0;
                }
            }
            last_exemplars = Some(exemplars);
        }

        let exemplars = last_exemplars.unwrap_or_else(|| vec![true; n]);
        let cluster_centers_indices: Vec<usize> = exemplars
            .iter()
            .enumerate()
            .filter(|(_, &b)| b)
            .map(|(i, _)| i)
            .collect();
        if cluster_centers_indices.is_empty() {
            return Ok(FittedAffinityPropagation {
                labels: Array1::<f64>::zeros(n),
                cluster_centers_indices: vec![0],
            });
        }

        // Assign each point to nearest exemplar (using dense distance since
        // exemplars are O(n_clusters) and matter for label assignment).
        let mut labels = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut best = f64::NEG_INFINITY;
            let mut best_c = 0;
            for (c, &kk) in cluster_centers_indices.iter().enumerate() {
                let sim_ik = -sq(i, kk);
                if sim_ik > best {
                    best = sim_ik;
                    best_c = c;
                }
            }
            labels[i] = best_c as f64;
        }
        Ok(FittedAffinityPropagation {
            labels,
            cluster_centers_indices,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_ap_two_clusters() {
        let x = array![
            [0.0_f64, 0.0],
            [0.2, 0.1],
            [-0.1, 0.2],
            [10.0, 10.0],
            [10.2, 9.9],
            [9.8, 10.1],
        ];
        let ap = AffinityPropagation::new()
            .with_damping(0.7)
            .with_preference(-1.0);
        let fitted = ap.fit(&x).unwrap();
        assert!(
            fitted.cluster_centers_indices.len() >= 2,
            "expected ≥2 clusters, got {}",
            fitted.cluster_centers_indices.len()
        );
        let l0 = fitted.labels[0];
        for i in 1..3 {
            assert_eq!(fitted.labels[i], l0);
        }
        for i in 3..6 {
            assert_ne!(fitted.labels[i], l0);
        }
    }

    #[test]
    fn test_ap_sparse_separates_two_blobs() {
        // Two well-separated tight blobs of 15 points each. Sparse AP
        // with k=10 should never produce a label assignment that mixes
        // points across the two blobs.
        let mut data = Vec::new();
        for i in 0..15 {
            let t = i as f64 * 0.05;
            data.push(t.sin() * 0.05);
            data.push(t.cos() * 0.05);
        }
        for i in 0..15 {
            let t = i as f64 * 0.05;
            data.push(20.0 + t.sin() * 0.05);
            data.push(20.0 + t.cos() * 0.05);
        }
        let x = ndarray::Array2::from_shape_vec((30, 2), data).unwrap();
        let mut ap = AffinityPropagation::new()
            .with_damping(0.5)
            .with_preference(-0.001) // small negative — encourage exemplars
            .with_n_neighbors(10);
        ap.max_iter = 500;
        ap.convergence_iter = 30;
        let fitted = ap.fit(&x).unwrap();
        // Diagnostic: must have at least 2 exemplars.
        assert!(
            fitted.cluster_centers_indices.len() >= 2,
            "expected ≥2 exemplars, got {} ({:?})",
            fitted.cluster_centers_indices.len(),
            fitted.cluster_centers_indices
        );
        // Within each blob, all assigned exemplars must lie within the blob
        // — i.e. no two rows from different blobs share a label.
        let a_labels: std::collections::HashSet<i64> =
            (0..15).map(|i| fitted.labels[i] as i64).collect();
        let b_labels: std::collections::HashSet<i64> =
            (15..30).map(|i| fitted.labels[i] as i64).collect();
        assert!(
            a_labels.is_disjoint(&b_labels),
            "A and B share labels: A={:?}, B={:?}",
            a_labels,
            b_labels
        );
        // Also verify the sparse path produced at least 2 exemplars.
        assert!(fitted.cluster_centers_indices.len() >= 2);
    }
}
