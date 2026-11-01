//! Spectral clustering.
//!
//! Mirrors `sklearn.cluster.SpectralClustering` with
//! `assign_labels="kmeans"`: an affinity graph (RBF kernel, or the k-NN
//! connectivity `0.5·(C + Cᵀ)` with each sample among its own neighbours),
//! its spectral embedding (bottom eigenvectors of the normalised Laplacian
//! divided by `√degree`, from a sparse eigensolver for large graphs), and
//! k-means with `n_init` restarts on that embedding.

use anofox_ml_core::Float;
use anofox_ml_core::{FitUnsupervised, Result, RustMlError};
use anofox_ml_manifold::spectral_embedding;
use anofox_ml_neighbors::{DistanceMetric, NeighborAlgorithm, NeighborIndex};
use ndarray::{Array1, Array2};

use crate::kmeans::KMeans;

/// How to build the affinity graph.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum Affinity {
    /// `exp(-γ ||x_i - x_j||²)` with the given gamma (dense).
    Rbf(f64),
    /// k-NN connectivity including the sample itself, symmetrised as
    /// `0.5·(C + Cᵀ)` (sparse).
    KNearest(usize),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpectralClustering {
    pub n_clusters: usize,
    pub affinity: Affinity,
    pub seed: u64,
    /// k-means restarts on the embedding (sklearn's `n_init`, default 10).
    #[serde(default = "default_n_init")]
    pub n_init: usize,
}

fn default_n_init() -> usize {
    10
}

impl SpectralClustering {
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            affinity: Affinity::Rbf(1.0),
            seed: 0,
            n_init: 10,
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
    pub fn with_n_init(mut self, n: usize) -> Self {
        self.n_init = n;
        self
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedSpectralClustering {
    pub labels: Array1<f64>,
}

/// Affinity graph entries `(i, j, w)` (both triangles).
fn build_affinity(x: &Array2<f64>, affinity: &Affinity) -> Result<Vec<(usize, usize, f64)>> {
    let n = x.nrows();
    match *affinity {
        Affinity::Rbf(gamma) => {
            let mut e = Vec::with_capacity(n * n);
            for i in 0..n {
                for j in 0..n {
                    let sd: f64 = x
                        .row(i)
                        .iter()
                        .zip(x.row(j))
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum();
                    e.push((i, j, (-gamma * sd).exp()));
                }
            }
            Ok(e)
        }
        Affinity::KNearest(k) => {
            // kneighbors_graph(X, k, include_self=True): each sample's k
            // nearest samples, itself first.
            let k = k.clamp(1, n);
            let index = NeighborIndex::new(
                x,
                DistanceMetric::Euclidean,
                NeighborAlgorithm::Auto,
                Some(k),
            )?;
            let (_, nbrs) = index.kneighbors(x, k)?;
            let mut e = Vec::with_capacity(2 * n * k);
            for (i, row) in nbrs.rows().into_iter().enumerate() {
                for &j in row {
                    e.push((i, j, 0.5));
                    e.push((j, i, 0.5));
                }
            }
            Ok(e)
        }
    }
}

impl<F: Float> FitUnsupervised<F> for SpectralClustering {
    type Fitted = FittedSpectralClustering;

    /// Computed in `f64` whatever the input float type.
    fn fit(&self, x: &Array2<F>) -> Result<Self::Fitted> {
        let x_f64 = anofox_ml_core::as_f64(x);
        let x: &Array2<f64> = &x_f64;
        let n = x.nrows();
        if n == 0 {
            return Err(RustMlError::EmptyInput("empty input".into()));
        }
        if self.n_clusters == 0 || self.n_clusters > n {
            return Err(RustMlError::InvalidParameter("invalid n_clusters".into()));
        }
        let graph = build_affinity(x, &self.affinity)?;
        let maps = spectral_embedding(n, &graph, self.n_clusters, false)?;
        let km = KMeans::new(self.n_clusters)
            .with_seed(self.seed)
            .with_n_init(self.n_init.max(1));
        let fitted: crate::kmeans::FittedKMeans<f64> = FitUnsupervised::fit(&km, &maps)?;
        Ok(FittedSpectralClustering {
            labels: fitted.labels().clone(),
        })
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
