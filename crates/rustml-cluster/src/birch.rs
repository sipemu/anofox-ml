//! Birch-lite — single-pass online sub-clustering, final global KMeans.
//!
//! Mirrors the high-level behaviour of `sklearn.cluster.Birch` without the
//! full CF-tree: each new point joins the nearest existing Cluster Feature
//! (CF) whose centroid is within `threshold` Euclidean distance, otherwise
//! starts a fresh CF. After the pass, the CFs' centroids are clustered into
//! the requested `n_clusters` by KMeans. Points inherit the cluster label of
//! their CF.

use ndarray::{Array1, Array2};
use rustml_core::{FitUnsupervised, Predict, Result, RustMlError};

use crate::kmeans::KMeans;

#[derive(Debug, Clone)]
pub struct Birch {
    pub threshold: f64,
    pub n_clusters: Option<usize>,
    pub seed: u64,
}

impl Birch {
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            n_clusters: Some(3),
            seed: 0,
        }
    }
    pub fn with_n_clusters(mut self, n: Option<usize>) -> Self {
        self.n_clusters = n;
        self
    }
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedBirch {
    /// One sub-cluster centroid per row (after the online pass).
    pub subcluster_centers: Array2<f64>,
    /// For each sub-cluster, the cluster label assigned by the global step.
    pub subcluster_labels: Array1<f64>,
    /// Per-sample cluster labels.
    pub labels: Array1<f64>,
}

fn euclid_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

impl FitUnsupervised<f64> for Birch {
    type Fitted = FittedBirch;

    fn fit(&self, x: &Array2<f64>) -> Result<Self::Fitted> {
        let n = x.nrows();
        let d = x.ncols();
        if n == 0 {
            return Err(RustMlError::EmptyInput("empty input".into()));
        }
        if self.threshold <= 0.0 {
            return Err(RustMlError::InvalidParameter("threshold > 0".into()));
        }

        let t_sq = self.threshold * self.threshold;
        // Each CF stores (sum, n). Centroid = sum / n.
        let mut sums: Vec<Vec<f64>> = Vec::new();
        let mut counts: Vec<usize> = Vec::new();
        let mut sub_of: Vec<usize> = vec![0; n];

        for i in 0..n {
            let xi: Vec<f64> = x.row(i).iter().copied().collect();
            // Find nearest CF centroid.
            let mut best = f64::INFINITY;
            let mut best_k = None;
            for k in 0..sums.len() {
                let nk = counts[k] as f64;
                let centroid: Vec<f64> = sums[k].iter().map(|s| s / nk).collect();
                let d2 = euclid_sq(&xi, &centroid);
                if d2 < best {
                    best = d2;
                    best_k = Some(k);
                }
            }
            match best_k {
                Some(k) if best <= t_sq => {
                    for j in 0..d {
                        sums[k][j] += xi[j];
                    }
                    counts[k] += 1;
                    sub_of[i] = k;
                }
                _ => {
                    sub_of[i] = sums.len();
                    sums.push(xi);
                    counts.push(1);
                }
            }
        }

        let m = sums.len();
        let mut subcluster_centers = Array2::<f64>::zeros((m, d));
        for k in 0..m {
            let nk = counts[k] as f64;
            for j in 0..d {
                subcluster_centers[[k, j]] = sums[k][j] / nk;
            }
        }

        // Final global clustering.
        let n_clusters = self.n_clusters.unwrap_or(m).min(m);
        let subcluster_labels = if n_clusters == m {
            // No further clustering: each CF is its own cluster.
            Array1::from_vec((0..m).map(|k| k as f64).collect())
        } else {
            let km = KMeans::new(n_clusters).with_seed(self.seed);
            let fitted: crate::kmeans::FittedKMeans<f64> =
                FitUnsupervised::fit(&km, &subcluster_centers)?;
            fitted.predict(&subcluster_centers)?
        };
        // Per-sample labels.
        let labels: Vec<f64> = sub_of.iter().map(|&k| subcluster_labels[k]).collect();
        Ok(FittedBirch {
            subcluster_centers,
            subcluster_labels,
            labels: Array1::from_vec(labels),
        })
    }
}

impl Predict<f64> for FittedBirch {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let d = self.subcluster_centers.ncols();
        if x.ncols() != d {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}", d, x.ncols()
            )));
        }
        let m = self.subcluster_centers.nrows();
        let n = x.nrows();
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let xi: Vec<f64> = x.row(i).iter().copied().collect();
            let mut best = f64::INFINITY;
            let mut best_k = 0;
            for k in 0..m {
                let centroid: Vec<f64> =
                    self.subcluster_centers.row(k).iter().copied().collect();
                let d2 = euclid_sq(&xi, &centroid);
                if d2 < best {
                    best = d2;
                    best_k = k;
                }
            }
            out[i] = self.subcluster_labels[best_k];
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_birch_two_blobs() {
        let x = array![
            [0.0_f64, 0.0], [0.1, 0.1], [-0.1, 0.2], [0.1, -0.1],
            [10.0, 10.0], [10.1, 9.9], [9.8, 10.2], [10.2, 9.8],
        ];
        let fitted = Birch::new(1.0).with_n_clusters(Some(2)).with_seed(0).fit(&x).unwrap();
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

impl rustml_core::ClassifierScore<f64> for FittedBirch {}
