//! Local Outlier Factor.
//!
//! Mirrors `sklearn.neighbors.LocalOutlierFactor`. For each training point:
//! 1. Find its `k` nearest neighbors and the `k`-distance.
//! 2. `reach_dist(a, b) = max(k_dist(b), dist(a, b))`.
//! 3. Local reachability density `lrd(a) = 1 / (mean_b reach_dist(a, b) + 1e-10)`
//!    (the `1e-10` guard is sklearn's: duplicated points get a huge `lrd`,
//!    so a point whose neighbours are duplicates gets a huge LOF).
//! 4. `LOF(a) = mean_b lrd(b) / lrd(a)`.
//!
//! LOF >> 1 indicates an outlier.

use anofox_ml_core::Float;
use anofox_ml_core::{FitUnsupervised, Result, RustMlError};
use ndarray::{Array1, Array2};

use crate::distance::DistanceMetric;
use crate::index::{NeighborAlgorithm, NeighborIndex};

/// Algorithm used for k-NN lookup. `Auto` follows sklearn's rule (see
/// [`NeighborAlgorithm::Auto`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LofAlgorithm {
    Auto,
    KdTree,
    BruteForce,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LocalOutlierFactor {
    pub n_neighbors: usize,
    pub contamination: f64,
    pub algorithm: LofAlgorithm,
}

impl LocalOutlierFactor {
    pub fn new(n_neighbors: usize) -> Self {
        Self {
            n_neighbors,
            contamination: 0.1,
            algorithm: LofAlgorithm::Auto,
        }
    }
    pub fn with_contamination(mut self, c: f64) -> Self {
        self.contamination = c;
        self
    }
    pub fn with_algorithm(mut self, a: LofAlgorithm) -> Self {
        self.algorithm = a;
        self
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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

impl<F: Float> FitUnsupervised<F> for LocalOutlierFactor {
    type Fitted = FittedLocalOutlierFactor;

    /// Computed in `f64` whatever the input float type.
    fn fit(&self, x: &Array2<F>) -> Result<Self::Fitted> {
        let x_f64 = anofox_ml_core::as_f64(x);
        let x: &Array2<f64> = &x_f64;
        let n = x.nrows();
        if n == 0 {
            return Err(RustMlError::EmptyInput("empty input".into()));
        }
        let k = self.n_neighbors.min(n.saturating_sub(1));
        if k == 0 {
            return Err(RustMlError::InvalidParameter("n_neighbors >= 1".into()));
        }

        // 1. k-NN of every sample among the others.
        let algorithm = match self.algorithm {
            LofAlgorithm::KdTree => NeighborAlgorithm::KdTree,
            LofAlgorithm::BruteForce => NeighborAlgorithm::BruteForce,
            LofAlgorithm::Auto => NeighborAlgorithm::Auto,
        };
        let index = NeighborIndex::new(x, DistanceMetric::Euclidean, algorithm, Some(k))?;
        let (dist, idx) = index.kneighbors_self(k)?;
        let neighbors: Vec<Vec<(usize, f64)>> = (0..n)
            .map(|i| {
                idx.row(i)
                    .iter()
                    .copied()
                    .zip(dist.row(i).iter().copied())
                    .collect()
            })
            .collect();
        let k_dist: Vec<f64> = neighbors
            .iter()
            .map(|nbrs| nbrs.last().map(|p| p.1).unwrap_or(0.0))
            .collect();

        // 2. LRD per point — reach_dist(i,j) = max(dist(i,j), k_dist(j)).
        let mut lrd = vec![0.0_f64; n];
        for i in 0..n {
            let mut s = 0.0;
            for &(j, d_ij) in &neighbors[i] {
                let rd = d_ij.max(k_dist[j]);
                s += rd;
            }
            let mean_rd = s / k as f64;
            // sklearn `_local_reachability_density`: 1 / (mean + 1e-10).
            lrd[i] = 1.0 / (mean_rd + 1e-10);
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
        sorted.sort_by(|a, b| a.total_cmp(b));
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
        data.push(100.0);
        data.push(100.0);
        let x = Array2::from_shape_vec((31, 2), data).unwrap();
        let lof = LocalOutlierFactor::new(5).with_contamination(1.0 / 31.0);
        let fitted = lof.fit(&x).unwrap();
        // Last point should be outlier.
        assert_eq!(fitted.predictions[30], -1.0);
        // Other points mostly inliers.
        let inliers = fitted
            .predictions
            .iter()
            .take(30)
            .filter(|&&p| p > 0.0)
            .count();
        assert!(inliers >= 28, "too few inliers: {inliers}");
    }

    #[test]
    fn test_lof_kdtree_matches_brute() {
        // KD-tree path must produce identical LOF scores to brute force on
        // a small low-dim dataset.
        let mut data = Vec::new();
        for i in 0..50 {
            let t = i as f64 * 0.1;
            data.push(t.sin());
            data.push(t.cos());
        }
        data.push(5.0);
        data.push(5.0);
        let x = Array2::from_shape_vec((51, 2), data).unwrap();

        let kd = LocalOutlierFactor::new(7)
            .with_algorithm(LofAlgorithm::KdTree)
            .with_contamination(0.1);
        let bf = LocalOutlierFactor::new(7)
            .with_algorithm(LofAlgorithm::BruteForce)
            .with_contamination(0.1);
        let f_kd = kd.fit(&x).unwrap();
        let f_bf = bf.fit(&x).unwrap();
        for (a, b) in f_kd
            .negative_outlier_factor
            .iter()
            .zip(f_bf.negative_outlier_factor.iter())
        {
            assert!(
                (a - b).abs() < 1e-9,
                "KD-tree vs brute LOF mismatch: {} vs {}",
                a,
                b
            );
        }
    }
    /// Regression for #57: sklearn's `lrd = 1 / (mean_reach + 1e-10)`.
    /// Six duplicated points have reach distance 0 -> huge lrd, so the
    /// point next to them gets LOF ~ 5e9 (it used to get ~ 0.5, i.e. look
    /// like the most normal point). Reference values: sklearn 1.8.0.
    #[test]
    fn test_lof_duplicates_match_sklearn() {
        let x = ndarray::array![
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.5, 0.0],
            [3.0, 3.0],
            [3.2, 3.1],
            [2.9, 3.3],
            [3.1, 2.8],
            [3.3, 3.2],
            [2.8, 2.9]
        ];
        let expected = [
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -5000000001.0,
            -1.0710583821777968,
            -1.030117830630578,
            -0.9869830906938422,
            -0.9869830906938422,
            -0.9659225184560674,
            -0.9659225184560674,
        ];
        for algo in [LofAlgorithm::KdTree, LofAlgorithm::BruteForce] {
            let f = LocalOutlierFactor::new(5)
                .with_algorithm(algo)
                .fit(&x)
                .unwrap();
            for (i, (&a, &b)) in f.negative_outlier_factor.iter().zip(&expected).enumerate() {
                assert!(
                    ((a - b) / b).abs() < 1e-6,
                    "{algo:?}: nof[{i}] = {a}, sklearn {b}"
                );
            }
            assert_eq!(f.predictions[6], -1.0);
        }
    }

    #[test]
    fn test_lof_fortran_order_input() {
        use ndarray::ShapeBuilder;
        let mut v = Vec::new();
        for i in 0..20 {
            let t = i as f64 * 0.3;
            v.push(t.sin());
            v.push(t.cos());
        }
        let x = Array2::from_shape_vec((20, 2), v).unwrap();
        let col: Vec<f64> = x.t().iter().copied().collect();
        let xf = Array2::from_shape_vec((20, 2).f(), col).unwrap();
        assert!(!xf.is_standard_layout());
        for algo in [LofAlgorithm::KdTree, LofAlgorithm::BruteForce] {
            let a = LocalOutlierFactor::new(4)
                .with_algorithm(algo)
                .fit(&x)
                .unwrap();
            let b = LocalOutlierFactor::new(4)
                .with_algorithm(algo)
                .fit(&xf)
                .unwrap();
            assert_eq!(a.negative_outlier_factor, b.negative_outlier_factor);
        }
    }

    #[test]
    fn test_f32_input_matches_f64() {
        let x = Array2::from_shape_fn((25, 2), |(i, j)| ((i * (j + 2)) % 9) as f64 / 4.0);
        let x32 = x.mapv(|v| v as f32);
        let lof = LocalOutlierFactor::new(5);
        let a = FitUnsupervised::<f64>::fit(&lof, &x).unwrap();
        let b = FitUnsupervised::<f32>::fit(&lof, &x32).unwrap();
        assert_eq!(a.negative_outlier_factor, b.negative_outlier_factor);
        assert_eq!(a.predictions, b.predictions);
    }
}
