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
pub use anofox_ml_manifold::Affinity;
use anofox_ml_manifold::{build_affinity, spectral_embedding};
use ndarray::{Array1, ArrayView2};

use crate::kmeans::KMeans;

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

impl<F: Float> FitUnsupervised<F> for SpectralClustering {
    type Fitted = FittedSpectralClustering;

    /// Computed in `f64` whatever the input float type.
    fn fit_view(&self, x: ArrayView2<F>) -> Result<Self::Fitted> {
        let x_f64 = anofox_ml_core::as_f64(x);
        let x = x_f64.view();
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
