//! Mean-Shift clustering.
//!
//! Mirrors `sklearn.cluster.MeanShift`. From each seed point, iteratively
//! shifts toward the weighted mean of points within `bandwidth`. Points that
//! converge to the same mode are merged into a cluster. Mode-merging
//! collapses centroids that are within `bandwidth / 2` of each other.

use anofox_ml_core::{FitUnsupervised, Predict, Result, RustMlError};
use ndarray::{Array1, Array2};

#[derive(Debug, Clone)]
pub struct MeanShift {
    pub bandwidth: f64,
    pub max_iter: usize,
    pub tol: f64,
}

impl MeanShift {
    pub fn new(bandwidth: f64) -> Self {
        Self {
            bandwidth,
            max_iter: 300,
            tol: 1e-3,
        }
    }
    pub fn with_max_iter(mut self, m: usize) -> Self {
        self.max_iter = m;
        self
    }
    pub fn with_tol(mut self, t: f64) -> Self {
        self.tol = t;
        self
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedMeanShift {
    pub cluster_centers: Array2<f64>,
    pub labels: Array1<f64>,
}

fn sq_euclid(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

fn shift_one(seed: &[f64], x: &Array2<f64>, bw_sq: f64) -> Vec<f64> {
    let d = seed.len();
    let mut num = vec![0.0_f64; d];
    let mut den = 0.0_f64;
    for i in 0..x.nrows() {
        let xi: Vec<f64> = x.row(i).iter().copied().collect();
        let sd = sq_euclid(seed, &xi);
        if sd <= bw_sq {
            // Flat kernel: weight = 1 within bandwidth.
            for j in 0..d {
                num[j] += xi[j];
            }
            den += 1.0;
        }
    }
    if den == 0.0 {
        seed.to_vec()
    } else {
        num.iter().map(|v| v / den).collect()
    }
}

impl FitUnsupervised<f64> for MeanShift {
    type Fitted = FittedMeanShift;

    fn fit(&self, x: &Array2<f64>) -> Result<Self::Fitted> {
        let n = x.nrows();
        if n == 0 {
            return Err(RustMlError::EmptyInput("empty input".into()));
        }
        if self.bandwidth <= 0.0 {
            return Err(RustMlError::InvalidParameter(
                "bandwidth must be > 0".into(),
            ));
        }
        let d = x.ncols();
        let bw_sq = self.bandwidth * self.bandwidth;

        // Run mean-shift from each point.
        let mut modes: Vec<Vec<f64>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut cur: Vec<f64> = x.row(i).iter().copied().collect();
            for _ in 0..self.max_iter {
                let next = shift_one(&cur, x, bw_sq);
                let shift = sq_euclid(&cur, &next).sqrt();
                cur = next;
                if shift < self.tol {
                    break;
                }
            }
            modes.push(cur);
        }
        // Mode-merging: collapse modes within bandwidth/2 of each other.
        let merge_radius_sq = (self.bandwidth / 2.0).powi(2);
        let mut centers: Vec<Vec<f64>> = Vec::new();
        let mut labels = Array1::<f64>::zeros(n);
        for (i, m) in modes.iter().enumerate() {
            let mut assigned = None;
            for (k, c) in centers.iter().enumerate() {
                if sq_euclid(m, c) < merge_radius_sq {
                    assigned = Some(k);
                    break;
                }
            }
            match assigned {
                Some(k) => labels[i] = k as f64,
                None => {
                    labels[i] = centers.len() as f64;
                    centers.push(m.clone());
                }
            }
        }
        let n_clusters = centers.len();
        let mut cc = Array2::<f64>::zeros((n_clusters, d));
        for (k, c) in centers.iter().enumerate() {
            for j in 0..d {
                cc[[k, j]] = c[j];
            }
        }
        Ok(FittedMeanShift {
            cluster_centers: cc,
            labels,
        })
    }
}

impl Predict<f64> for FittedMeanShift {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let d = self.cluster_centers.ncols();
        if x.ncols() != d {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                d,
                x.ncols()
            )));
        }
        let n = x.nrows();
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut best = f64::INFINITY;
            let mut best_k = 0;
            let xi: Vec<f64> = x.row(i).iter().copied().collect();
            for k in 0..self.cluster_centers.nrows() {
                let ck: Vec<f64> = self.cluster_centers.row(k).iter().copied().collect();
                let d2 = sq_euclid(&xi, &ck);
                if d2 < best {
                    best = d2;
                    best_k = k;
                }
            }
            out[i] = best_k as f64;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_mean_shift_two_blobs() {
        let x = array![
            [0.0_f64, 0.0],
            [0.1, 0.1],
            [-0.1, 0.2],
            [0.1, -0.1],
            [10.0, 10.0],
            [10.1, 9.9],
            [9.9, 10.2],
            [10.0, 10.1],
        ];
        let ms = MeanShift::new(2.0);
        let fitted = ms.fit(&x).unwrap();
        assert_eq!(fitted.cluster_centers.nrows(), 2);
        let l0 = fitted.labels[0];
        for i in 1..4 {
            assert_eq!(fitted.labels[i], l0);
        }
        for i in 4..8 {
            assert_ne!(fitted.labels[i], l0);
        }
    }

    #[test]
    fn test_mean_shift_centers_near_blob_means() {
        let x = array![
            [0.0_f64, 0.0],
            [0.1, 0.1],
            [-0.1, 0.2],
            [0.1, -0.1],
            [10.0, 10.0],
            [10.1, 9.9],
            [9.9, 10.2],
            [10.0, 10.1],
        ];
        let fitted = MeanShift::new(2.0).fit(&x).unwrap();
        // Centroids should be near (0.025, 0.05) and (10.0, 10.05).
        let mut has_low = false;
        let mut has_high = false;
        for k in 0..fitted.cluster_centers.nrows() {
            let cx = fitted.cluster_centers[[k, 0]];
            if cx.abs() < 1.0 {
                has_low = true;
            }
            if (cx - 10.0).abs() < 1.0 {
                has_high = true;
            }
        }
        assert!(has_low && has_high);
    }
}

impl anofox_ml_core::ClassifierScore<f64> for FittedMeanShift {}
