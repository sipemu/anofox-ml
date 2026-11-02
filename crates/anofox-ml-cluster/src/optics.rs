//! OPTICS — Ordering Points To Identify the Clustering Structure.
//!
//! Mirrors `sklearn.cluster.OPTICS` (neighbour lookups through the shared
//! [`NeighborIndex`]) with the DBSCAN-like cluster extraction (`extract_dbscan` mode with a single
//! ε threshold). Returns the reachability ordering plus per-sample labels.
//!
//! Algorithm (Ankerst et al. 1999):
//! 1. For each point compute `core_distance(p)` = distance to its
//!    `min_samples`-th nearest neighbour, counting `p` itself as in sklearn
//!    (∞ if that distance exceeds `eps_max`).
//! 2. Repeatedly process the unprocessed point with the smallest
//!    reachability distance (ties: lowest index; sklearn's global argmin,
//!    here a heap with lazy deletion) and update its neighbours'
//!    reachability distances.
//! 3. Output: `ordering` (the visit order) and `reachability` (per-point
//!    reachability distance).
//! 4. Cluster extraction (`extract_dbscan`): walk `ordering`, points with
//!    `reachability ≤ eps` join the current cluster; otherwise start a new
//!    cluster or label as noise.

use anofox_ml_core::Float;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use anofox_ml_core::{FitUnsupervised, Result, RustMlError};
use anofox_ml_neighbors::{DistanceMetric, NeighborAlgorithm, NeighborIndex};
use ndarray::{Array1, ArrayView2};

/// How clusters are extracted from the reachability plot.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub enum OpticsClusterMethod {
    /// DBSCAN-equivalent cut at `eps` (sklearn's `cluster_method="dbscan"`).
    #[default]
    Dbscan,
    /// Steep-area extraction (sklearn's default `cluster_method="xi"`):
    /// clusters start / end where the reachability drops / rises by at
    /// least a factor `1 - xi`.
    Xi {
        xi: f64,
        /// Minimum cluster size (`None` = `min_samples`).
        min_cluster_size: Option<usize>,
        /// Correct cluster ends from predecessors (sklearn's default).
        predecessor_correction: bool,
    },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Optics {
    pub min_samples: usize,
    /// Cut level of the DBSCAN extraction.
    pub eps: f64,
    pub eps_max: f64,
    #[serde(default)]
    pub cluster_method: OpticsClusterMethod,
}

impl Optics {
    /// DBSCAN extraction at `eps`.
    pub fn new(min_samples: usize, eps: f64) -> Self {
        Self {
            min_samples,
            eps,
            eps_max: f64::INFINITY,
            cluster_method: OpticsClusterMethod::Dbscan,
        }
    }
    /// ξ extraction (sklearn's default method; sklearn's default xi is 0.05).
    pub fn xi(min_samples: usize, xi: f64) -> Self {
        Self {
            cluster_method: OpticsClusterMethod::Xi {
                xi,
                min_cluster_size: None,
                predecessor_correction: true,
            },
            ..Self::new(min_samples, f64::INFINITY)
        }
    }
    pub fn with_eps_max(mut self, e: f64) -> Self {
        self.eps_max = e;
        self
    }
    pub fn with_cluster_method(mut self, m: OpticsClusterMethod) -> Self {
        self.cluster_method = m;
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
    /// Core distance per sample (∞ when above `eps_max`).
    #[serde(default)]
    pub core_distances: Array1<f64>,
    /// ξ clusters as `[start, end]` positions in `ordering` (sklearn's
    /// `cluster_hierarchy_`); empty for the DBSCAN extraction.
    #[serde(default)]
    pub cluster_hierarchy: Vec<[usize; 2]>,
    /// The point from which each sample was reached (-1 for none).
    #[serde(default)]
    pub predecessor: Vec<i64>,
}

/// Heap key: (reachability, index), smallest first.
#[derive(PartialEq)]
struct Key(f64, usize);
impl Eq for Key {}
impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Key {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0).then(self.1.cmp(&other.1))
    }
}

impl<F: Float> FitUnsupervised<F> for Optics {
    type Fitted = FittedOptics;

    /// Computed in `f64` whatever the input float type.
    fn fit_view(&self, x: ArrayView2<F>) -> Result<Self::Fitted> {
        let x_f64 = anofox_ml_core::as_f64(x);
        let x = x_f64.view();
        let n = x.nrows();
        if n == 0 {
            return Err(RustMlError::EmptyInput("empty input".into()));
        }
        if self.min_samples < 1 {
            return Err(RustMlError::InvalidParameter("min_samples >= 1".into()));
        }

        let index =
            NeighborIndex::new(x, DistanceMetric::Euclidean, NeighborAlgorithm::Auto, None)?;
        // Core distance per point: distance to the `min_samples`-th nearest
        // neighbour *counting the point itself* (sklearn semantics, so
        // `min_samples = 1` gives 0 and `min_samples = 2` the nearest other
        // point). Infinite when that distance exceeds `eps_max` or there are
        // fewer than `min_samples` points.
        let mut core_dist = vec![f64::INFINITY; n];
        if self.min_samples <= n {
            let (d, _) = index.kneighbors(x, self.min_samples)?;
            for (c, row) in core_dist.iter_mut().zip(d.rows()) {
                let kth = row[self.min_samples - 1];
                if kth <= self.eps_max {
                    *c = kth;
                }
            }
        }

        let mut processed = vec![false; n];
        let mut reach = Array1::<f64>::from_elem(n, f64::INFINITY);
        let mut predecessor = vec![-1_i64; n];
        let mut order: Vec<usize> = Vec::with_capacity(n);
        // Unprocessed points with a finite reachability; stale entries are
        // skipped when popped. With none left, the next point is the lowest
        // unprocessed index (all remaining reachabilities are ∞).
        let mut heap: BinaryHeap<Reverse<Key>> = BinaryHeap::new();
        let mut next_unreached = 0usize;
        while order.len() < n {
            let p = loop {
                match heap.pop() {
                    Some(Reverse(Key(r, q))) if !processed[q] && r == reach[q] => break q,
                    Some(_) => continue,
                    None => {
                        while processed[next_unreached] {
                            next_unreached += 1;
                        }
                        break next_unreached;
                    }
                }
            };
            processed[p] = true;
            order.push(p);
            if !core_dist[p].is_finite() {
                continue;
            }
            // Update the reachability of p's unprocessed neighbours within
            // eps_max.
            let row = x.row(p).to_owned().insert_axis(ndarray::Axis(0));
            for &(q, d) in &index.radius_neighbors(row.view(), self.eps_max)?[0] {
                if processed[q] {
                    continue;
                }
                let new_r = core_dist[p].max(d);
                if new_r < reach[q] {
                    reach[q] = new_r;
                    predecessor[q] = p as i64;
                    heap.push(Reverse(Key(new_r, q)));
                }
            }
        }

        let (labels, cluster_hierarchy) = match self.cluster_method {
            OpticsClusterMethod::Dbscan => {
                // DBSCAN-like extraction: scan `order`. A point with
                // `reach > eps` starts a new cluster only if its
                // `core_dist <= eps` (otherwise noise).
                let mut labels = Array1::<f64>::from_elem(n, -1.0);
                let mut current_cluster = -1.0_f64;
                for &p in &order {
                    if reach[p] > self.eps {
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
                (labels, Vec::new())
            }
            OpticsClusterMethod::Xi {
                xi,
                min_cluster_size,
                predecessor_correction,
            } => {
                if !(0.0..=1.0).contains(&xi) {
                    return Err(RustMlError::InvalidParameter(format!(
                        "xi must be in [0, 1], got {xi}"
                    )));
                }
                let min_samples = self.min_samples.max(2);
                let min_cluster_size = min_cluster_size.unwrap_or(min_samples).max(2);
                let plot: Vec<f64> = order.iter().map(|&p| reach[p]).collect();
                let pred_plot: Vec<i64> = order.iter().map(|&p| predecessor[p]).collect();
                let clusters = xi_cluster(
                    &plot,
                    &pred_plot,
                    &order,
                    xi,
                    min_samples,
                    min_cluster_size,
                    predecessor_correction,
                );
                (extract_xi_labels(&order, &clusters), clusters)
            }
        };

        Ok(FittedOptics {
            ordering: order,
            reachability: reach,
            labels,
            core_distances: Array1::from(core_dist),
            cluster_hierarchy,
            predecessor,
        })
    }
}

// ───── ξ extraction (sklearn's `cluster_optics_xi`) ──────────────────────

/// End of a steep region starting at `start`: extends over steep points
/// and at most `min_samples` consecutive non-steep, non-counter-direction
/// points.
fn extend_region(steep: &[bool], xward: &[bool], start: usize, min_samples: usize) -> usize {
    let mut non_xward = 0;
    let mut end = start;
    for index in start..steep.len() {
        if steep[index] {
            non_xward = 0;
            end = index;
        } else if !xward[index] {
            non_xward += 1;
            if non_xward > min_samples {
                break;
            }
        } else {
            return end;
        }
    }
    end
}

/// A steep down area with its maximum-in-between.
struct Sda {
    start: usize,
    end: usize,
    mib: f64,
}

fn update_filter_sdas(sdas: Vec<Sda>, mib: f64, xi_c: f64, plot: &[f64]) -> Vec<Sda> {
    if mib.is_infinite() {
        return Vec::new();
    }
    sdas.into_iter()
        .filter(|d| mib <= plot[d.start] * xi_c)
        .map(|mut d| {
            d.mib = d.mib.max(mib);
            d
        })
        .collect()
}

fn correct_predecessor(
    plot: &[f64],
    pred_plot: &[i64],
    ordering: &[usize],
    s: usize,
    mut e: usize,
) -> Option<(usize, usize)> {
    while s < e {
        if plot[s] > plot[e] {
            return Some((s, e));
        }
        let p_e = pred_plot[e];
        if (s..e).any(|i| p_e == ordering[i] as i64) {
            return Some((s, e));
        }
        e -= 1;
    }
    None
}

/// sklearn's `_xi_cluster` on the reachability plot (reachabilities in
/// visit order).
fn xi_cluster(
    plot: &[f64],
    pred_plot: &[i64],
    ordering: &[usize],
    xi: f64,
    min_samples: usize,
    min_cluster_size: usize,
    predecessor_correction: bool,
) -> Vec<[usize; 2]> {
    let n = plot.len();
    let mut plot: Vec<f64> = plot.to_vec();
    plot.push(f64::INFINITY);
    let xi_c = 1.0 - xi;
    let ratio: Vec<f64> = (0..n).map(|i| plot[i] / plot[i + 1]).collect();
    let steep_up: Vec<bool> = ratio.iter().map(|&r| r <= xi_c).collect();
    let steep_down: Vec<bool> = ratio.iter().map(|&r| r >= 1.0 / xi_c).collect();
    let down: Vec<bool> = ratio.iter().map(|&r| r > 1.0).collect();
    let up: Vec<bool> = ratio.iter().map(|&r| r < 1.0).collect();

    let mut sdas: Vec<Sda> = Vec::new();
    let mut clusters: Vec<[usize; 2]> = Vec::new();
    let mut index = 0usize;
    let mut mib = 0.0_f64;
    for steep_index in (0..n).filter(|&i| steep_up[i] || steep_down[i]) {
        if steep_index < index {
            continue;
        }
        mib = plot[index..=steep_index]
            .iter()
            .copied()
            .fold(mib, f64::max);
        sdas = update_filter_sdas(sdas, mib, xi_c, &plot);
        if steep_down[steep_index] {
            let d_end = extend_region(&steep_down, &up, steep_index, min_samples);
            sdas.push(Sda {
                start: steep_index,
                end: d_end,
                mib: 0.0,
            });
            index = d_end + 1;
            mib = plot[index];
        } else {
            let u_start = steep_index;
            let u_end = extend_region(&steep_up, &down, u_start, min_samples);
            index = u_end + 1;
            mib = plot[index];
            let mut u_clusters = Vec::new();
            for d in &sdas {
                let mut c_start = d.start;
                let mut c_end = u_end;
                // Line (**), sc2*.
                if plot[c_end + 1] * xi_c < d.mib {
                    continue;
                }
                // Definition 11: criterion 4.
                let d_max = plot[d.start];
                if d_max * xi_c >= plot[c_end + 1] {
                    while plot[c_start + 1] > plot[c_end + 1] && c_start < d.end {
                        c_start += 1;
                    }
                } else if plot[c_end + 1] * xi_c >= d_max {
                    while c_end > u_start && plot[c_end - 1] > d_max {
                        c_end -= 1;
                    }
                }
                if predecessor_correction {
                    match correct_predecessor(&plot, pred_plot, ordering, c_start, c_end) {
                        Some((s, e)) => {
                            c_start = s;
                            c_end = e;
                        }
                        None => continue,
                    }
                }
                // Criteria 1–3.
                if c_end - c_start + 1 < min_cluster_size || c_start > d.end || c_end < u_start {
                    continue;
                }
                u_clusters.push([c_start, c_end]);
            }
            u_clusters.reverse();
            clusters.extend(u_clusters);
        }
    }
    clusters
}

/// sklearn's `_extract_xi_labels`: leaf clusters (no labelled point inside
/// yet) in hierarchy order, mapped back from the ordering.
fn extract_xi_labels(ordering: &[usize], clusters: &[[usize; 2]]) -> Array1<f64> {
    let n = ordering.len();
    let mut in_order = vec![-1.0_f64; n];
    let mut label = 0.0;
    for c in clusters {
        if in_order[c[0]..=c[1]].iter().all(|&l| l == -1.0) {
            in_order[c[0]..=c[1]].fill(label);
            label += 1.0;
        }
    }
    let mut labels = Array1::from_elem(n, -1.0);
    for (pos, &p) in ordering.iter().enumerate() {
        labels[p] = in_order[pos];
    }
    labels
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
    /// Parity with `sklearn.cluster.OPTICS(min_samples, max_eps=inf,
    /// cluster_method="dbscan", eps=1.0)` (sklearn 1.8.0). Regression for the
    /// core-distance off-by-one (#57): `min_samples` counts the point itself.
    #[test]
    fn test_optics_matches_sklearn() {
        let x = array![
            [0.537, 0.131],
            [0.029, -0.559],
            [-0.083, -0.106],
            [-0.025, -0.188],
            [-0.013, -0.143],
            [-0.394, 0.265],
            [0.264, 0.513],
            [0.015, -0.121],
            [3.836, 3.536],
            [4.295, 3.67],
            [3.644, 3.938],
            [4.446, 4.071],
            [3.693, 3.786],
            [4.188, 3.952],
            [3.769, 3.931],
            [4.224, 4.593],
            [10.0, -6.0]
        ];
        let inf = f64::INFINITY;
        let cases: [(usize, Vec<f64>, Vec<usize>); 2] = [
            (
                2,
                vec![
                    inf,
                    0.3749093224,
                    0.0791770169,
                    0.0465725241,
                    0.0356089876,
                    0.4841094917,
                    0.4695242273,
                    0.5796447188,
                    4.6794992253,
                    0.3016173072,
                    0.1597028491,
                    0.2841214529,
                    0.2880086804,
                    0.4195259229,
                    0.1251958466,
                    0.5672459784,
                    11.2274629815,
                ],
                vec![0, 6, 7, 4, 3, 2, 1, 5, 8, 12, 10, 14, 13, 11, 9, 15, 16],
            ),
            (
                4,
                vec![
                    inf,
                    0.3749093224,
                    0.0791770169,
                    0.0791770169,
                    0.6144721312,
                    0.4841094917,
                    0.6144721312,
                    0.0791770169,
                    4.6794992253,
                    0.4195259229,
                    0.4454974747,
                    0.4195259229,
                    0.4454974747,
                    0.4195259229,
                    0.2880086804,
                    0.5672459784,
                    11.2274629815,
                ],
                vec![0, 4, 2, 3, 7, 1, 5, 6, 8, 10, 12, 14, 13, 9, 11, 15, 16],
            ),
        ];
        let labels: Vec<f64> = [vec![0.0; 8], vec![1.0; 8], vec![-1.0]].concat();
        for (ms, reach, order) in cases {
            let f = Optics::new(ms, 1.0).fit(&x).unwrap();
            assert_eq!(f.ordering, order, "min_samples={ms}");
            for (i, (&a, &b)) in f.reachability.iter().zip(reach.iter()).enumerate() {
                assert!(
                    (a.is_infinite() && b.is_infinite()) || (a - b).abs() < 1e-8,
                    "min_samples={ms} reach[{i}] {a} vs {b}"
                );
            }
            assert_eq!(f.labels.to_vec(), labels, "min_samples={ms}");
        }
    }
}
