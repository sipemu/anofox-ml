//! Local Outlier Factor.
//!
//! Mirrors `sklearn.neighbors.LocalOutlierFactor`. For each training point:
//! 1. Find its `k` nearest neighbors and the `k`-distance.
//! 2. `reach_dist(a, b) = max(k_dist(b), dist(a, b))`.
//! 3. Local reachability density `lrd(a) = 1 / mean_b reach_dist(a, b)`.
//! 4. `LOF(a) = mean_b lrd(b) / lrd(a)`.
//!
//! LOF >> 1 indicates an outlier.

use ndarray::{Array1, Array2};
use rustml_core::{FitUnsupervised, Result, RustMlError};

#[derive(Debug, Clone)]
pub struct LocalOutlierFactor {
    pub n_neighbors: usize,
    pub contamination: f64,
}

impl LocalOutlierFactor {
    pub fn new(n_neighbors: usize) -> Self {
        Self {
            n_neighbors,
            contamination: 0.1,
        }
    }
    pub fn with_contamination(mut self, c: f64) -> Self {
        self.contamination = c;
        self
    }
}

#[derive(Debug, Clone)]
pub struct FittedLocalOutlierFactor {
    /// Negative LOF per training point; higher = more normal (sklearn sign
    /// convention).
    pub negative_outlier_factor: Array1<f64>,
    /// Threshold (in `negative_outlier_factor` space) — points below are
    /// labelled outliers.
    pub threshold: f64,
    /// 1.0 for inlier, -1.0 for outlier.
    pub predictions: Array1<f64>,
}

fn euclidean(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = x - y;
        acc += d * d;
    }
    acc.sqrt()
}

impl FitUnsupervised<f64> for LocalOutlierFactor {
    type Fitted = FittedLocalOutlierFactor;

    fn fit(&self, x: &Array2<f64>) -> Result<Self::Fitted> {
        let n = x.nrows();
        if n == 0 {
            return Err(RustMlError::EmptyInput("empty input".into()));
        }
        let k = self.n_neighbors.min(n.saturating_sub(1));
        if k == 0 {
            return Err(RustMlError::InvalidParameter("n_neighbors >= 1".into()));
        }

        // 1. For each point find k-NN using a bounded max-heap; store
        //    (neighbor_index, distance) per point. O(n*k) memory total.
        use std::cmp::Ordering;
        #[derive(Clone, Copy)]
        struct DPair(usize, f64);
        impl Ord for DPair {
            fn cmp(&self, other: &Self) -> Ordering {
                self.1.partial_cmp(&other.1).unwrap_or(Ordering::Equal)
            }
        }
        impl PartialOrd for DPair {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
        }
        impl Eq for DPair {}
        impl PartialEq for DPair {
            fn eq(&self, other: &Self) -> bool { self.1 == other.1 }
        }

        let mut neighbors: Vec<Vec<(usize, f64)>> = vec![Vec::with_capacity(k); n];
        let mut k_dist = vec![0.0_f64; n];
        for i in 0..n {
            let row_i = x.row(i).to_owned();
            let ri = row_i.as_slice().unwrap();
            let mut heap: std::collections::BinaryHeap<DPair> = std::collections::BinaryHeap::with_capacity(k);
            for j in 0..n {
                if j == i { continue; }
                let row_j = x.row(j).to_owned();
                let rj = row_j.as_slice().unwrap();
                let d = euclidean(ri, rj);
                if heap.len() < k {
                    heap.push(DPair(j, d));
                } else if let Some(top) = heap.peek() {
                    if d < top.1 {
                        heap.pop();
                        heap.push(DPair(j, d));
                    }
                }
            }
            // Drain into sorted-ascending order.
            let mut v: Vec<(usize, f64)> = heap.into_iter().map(|p| (p.0, p.1)).collect();
            v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            k_dist[i] = v.last().map(|p| p.1).unwrap_or(0.0);
            neighbors[i] = v;
        }

        // 2. LRD per point — reach_dist(i,j) = max(dist(i,j), k_dist(j)).
        let mut lrd = vec![0.0_f64; n];
        for i in 0..n {
            let mut s = 0.0;
            for &(j, d_ij) in &neighbors[i] {
                let rd = d_ij.max(k_dist[j]);
                s += rd;
            }
            let mean_rd = s / k as f64;
            lrd[i] = if mean_rd > 0.0 { 1.0 / mean_rd } else { 1.0 };
        }
        // 3. LOF per point = mean(lrd(neighbors) / lrd(i)).
        let mut lof = vec![0.0_f64; n];
        for i in 0..n {
            let mut s = 0.0;
            for &(j, _) in &neighbors[i] {
                s += lrd[j];
            }
            lof[i] = (s / k as f64) / lrd[i];
        }
        let neg_of: Vec<f64> = lof.iter().map(|v| -v).collect();
        let neg_of = Array1::from(neg_of);

        // Threshold: contamination-th percentile of negative LOF (sklearn).
        let mut sorted: Vec<f64> = neg_of.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = (self.contamination * (n as f64 - 1.0)).round() as usize;
        let q = q.min(n - 1);
        let threshold = sorted[q];

        let predictions: Array1<f64> = neg_of.mapv(|v| if v > threshold { 1.0 } else { -1.0 });

        Ok(FittedLocalOutlierFactor {
            negative_outlier_factor: neg_of,
            threshold,
            predictions,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_lof_flags_obvious_outlier() {
        // Cluster + 1 outlier.
        let mut data = Vec::new();
        for i in 0..30 {
            data.push((i as f64) * 0.05);
            data.push((i as f64) * -0.05);
        }
        data.push(100.0); data.push(100.0);
        let x = Array2::from_shape_vec((31, 2), data).unwrap();
        let lof = LocalOutlierFactor::new(5).with_contamination(1.0 / 31.0);
        let fitted = lof.fit(&x).unwrap();
        // Last point should be outlier.
        assert_eq!(fitted.predictions[30], -1.0);
        // Other points mostly inliers.
        let inliers = fitted.predictions.iter().take(30).filter(|&&p| p > 0.0).count();
        assert!(inliers >= 28, "too few inliers: {inliers}");
    }
}
