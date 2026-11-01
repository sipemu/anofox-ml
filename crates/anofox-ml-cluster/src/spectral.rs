//! Spectral clustering.
//!
//! Mirrors `sklearn.cluster.SpectralClustering`. Builds an affinity matrix
//! (RBF or k-NN graph), forms the normalised Laplacian, takes the bottom-`k`
//! eigenvectors, and runs KMeans on the resulting embedding.

use anofox_ml_core::{FitUnsupervised, Predict, Result, RustMlError};
use anofox_ml_neighbors::{DistanceMetric, NeighborAlgorithm, NeighborIndex};
use faer::linalg::solvers::SelfAdjointEigen;
use faer::{Mat, Side};
use ndarray::{Array1, Array2};

use crate::kmeans::KMeans;

/// How to build the n×n affinity matrix.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum Affinity {
    /// `exp(-γ ||x_i - x_j||²)` with the given gamma.
    Rbf(f64),
    /// Symmetric k-NN graph: A_{ij} = 1 if either i is in j's k-NN or vice versa.
    KNearest(usize),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpectralClustering {
    pub n_clusters: usize,
    pub affinity: Affinity,
    pub seed: u64,
}

impl SpectralClustering {
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            affinity: Affinity::Rbf(1.0),
            seed: 0,
        }
    }
    pub fn with_affinity(mut self, a: Affinity) -> Self {
        self.affinity = a;
        self
    }
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedSpectralClustering {
    pub labels: Array1<f64>,
}

fn build_affinity(x: &Array2<f64>, affinity: &Affinity) -> Array2<f64> {
    let n = x.nrows();
    let mut a = Array2::<f64>::zeros((n, n));
    match affinity {
        Affinity::Rbf(gamma) => {
            for i in 0..n {
                for j in i..n {
                    let mut sd = 0.0;
                    for k in 0..x.ncols() {
                        let d = x[[i, k]] - x[[j, k]];
                        sd += d * d;
                    }
                    let v = (-gamma * sd).exp();
                    a[[i, j]] = v;
                    a[[j, i]] = v;
                }
            }
        }
        Affinity::KNearest(k) => {
            // k nearest other samples of every sample (shared index).
            let kk = (*k).min(n - 1);
            if kk > 0 {
                let index = NeighborIndex::new(
                    x,
                    DistanceMetric::Euclidean,
                    NeighborAlgorithm::Auto,
                    Some(kk),
                )
                .expect("non-empty input");
                let (_, nbrs) = index.kneighbors_self(kk).expect("valid k");
                for (i, row) in nbrs.rows().into_iter().enumerate() {
                    for &j in row {
                        a[[i, j]] = 1.0;
                    }
                }
            }
            // Symmetrize.
            for i in 0..n {
                for j in 0..n {
                    if a[[i, j]] > 0.0 || a[[j, i]] > 0.0 {
                        a[[i, j]] = 1.0;
                        a[[j, i]] = 1.0;
                    }
                }
            }
        }
    }
    // Zero out the diagonal — points are not their own neighbours.
    for i in 0..n {
        a[[i, i]] = 0.0;
    }
    a
}

impl FitUnsupervised<f64> for SpectralClustering {
    type Fitted = FittedSpectralClustering;

    fn fit(&self, x: &Array2<f64>) -> Result<Self::Fitted> {
        let n = x.nrows();
        if n == 0 {
            return Err(RustMlError::EmptyInput("empty input".into()));
        }
        if self.n_clusters == 0 || self.n_clusters > n {
            return Err(RustMlError::InvalidParameter("invalid n_clusters".into()));
        }

        let a = build_affinity(x, &self.affinity);
        // Degree matrix D (diagonal). Normalised symmetric Laplacian:
        //   L_sym = I - D^{-1/2} A D^{-1/2}
        // The eigenvectors of L_sym corresponding to its smallest eigenvalues
        // give the spectral embedding.
        let mut d_sqrt_inv = vec![0.0_f64; n];
        for i in 0..n {
            let deg: f64 = (0..n).map(|j| a[[i, j]]).sum::<f64>().max(1e-12);
            d_sqrt_inv[i] = 1.0 / deg.sqrt();
        }
        let mut l = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let off = -d_sqrt_inv[i] * a[[i, j]] * d_sqrt_inv[j];
                if i == j {
                    l[[i, j]] = 1.0 + off;
                } else {
                    l[[i, j]] = off;
                }
            }
        }
        // Symmetric eigendecomposition.
        let lm = Mat::from_fn(n, n, |i, j| l[[i, j]]);
        let eig = SelfAdjointEigen::new(lm.as_ref(), Side::Lower)
            .map_err(|e| RustMlError::InvalidParameter(format!("eigen failed: {e:?}")))?;
        let s = eig.S();
        let u = eig.U();
        // Pick bottom-k eigenvectors. SelfAdjointEigen returns ascending.
        let k = self.n_clusters;
        let mut embedding = Array2::<f64>::zeros((n, k));
        for c in 0..k {
            for i in 0..n {
                embedding[[i, c]] = u[(i, c)];
            }
        }
        // Row-normalise the embedding (sklearn does this for `assign_labels='kmeans'`).
        for i in 0..n {
            let nrm: f64 = (0..k)
                .map(|c| embedding[[i, c]].powi(2))
                .sum::<f64>()
                .sqrt()
                .max(1e-12);
            for c in 0..k {
                embedding[[i, c]] /= nrm;
            }
        }
        let _ = s; // unused
                   // KMeans on the embedding.
        let km = KMeans::new(k).with_seed(self.seed);
        let fitted: crate::kmeans::FittedKMeans<f64> = FitUnsupervised::fit(&km, &embedding)?;
        let labels = fitted.predict(&embedding)?;
        Ok(FittedSpectralClustering { labels })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_spectral_two_well_separated_blobs() {
        let x = array![
            [0.0_f64, 0.0],
            [0.1, 0.1],
            [-0.1, 0.2],
            [0.1, -0.2],
            [10.0, 10.0],
            [10.1, 9.9],
            [9.8, 10.2],
            [10.2, 9.8],
        ];
        let sc = SpectralClustering::new(2)
            .with_affinity(Affinity::Rbf(0.1))
            .with_seed(0);
        let fitted = sc.fit(&x).unwrap();
        let l0 = fitted.labels[0];
        for i in 1..4 {
            assert_eq!(fitted.labels[i], l0);
        }
        for i in 4..8 {
            assert_ne!(fitted.labels[i], l0);
        }
    }

    #[test]
    fn test_spectral_knn_graph() {
        let x = array![
            [0.0_f64, 0.0],
            [0.1, 0.1],
            [-0.1, 0.2],
            [0.1, -0.2],
            [10.0, 10.0],
            [10.1, 9.9],
            [9.8, 10.2],
            [10.2, 9.8],
        ];
        let sc = SpectralClustering::new(2)
            .with_affinity(Affinity::KNearest(3))
            .with_seed(0);
        let fitted = sc.fit(&x).unwrap();
        let l0 = fitted.labels[0];
        for i in 1..4 {
            assert_eq!(fitted.labels[i], l0);
        }
    }
}
