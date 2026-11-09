//! Spectral embedding of a graph (sklearn's `manifold.spectral_embedding`):
//! the bottom eigenvectors of the normalised Laplacian, scaled back by the
//! square-root degrees.

use anofox_ml_core::{Result, RustMlError};
use anofox_ml_neighbors::{DistanceMetric, NeighborAlgorithm, NeighborIndex};
use ndarray::{Array2, ArrayView2};

use crate::sparse_eigen::smallest_eigenpairs;

/// Embed the `n` nodes of the symmetric weighted graph `adjacency` (entries
/// `(i, j, w)` of both triangles; the diagonal is ignored) in
/// `n_components` dimensions.
///
/// As sklearn: `L = I − D^{-1/2} A D^{-1/2}` with unit diagonal (degrees
/// without self loops; isolated nodes keep degree 1), its `n_components`
/// smallest eigenvectors divided by `√degree`, each flipped so its largest
/// absolute entry is positive. With `drop_first` the first (trivial)
/// eigenvector is dropped and one more is computed.
pub fn spectral_embedding(
    n: usize,
    adjacency: &[(usize, usize, f64)],
    n_components: usize,
    drop_first: bool,
) -> Result<Array2<f64>> {
    let k = n_components + usize::from(drop_first);
    if n_components == 0 || k > n {
        return Err(RustMlError::InvalidParameter(format!(
            "n_components ({n_components}) must be in [1, {}]",
            n - usize::from(drop_first)
        )));
    }
    let mut deg = vec![0.0_f64; n];
    for &(i, j, w) in adjacency {
        if i != j {
            deg[i] += w;
        }
    }
    let dd: Vec<f64> = deg
        .iter()
        .map(|&d| if d == 0.0 { 1.0 } else { d.sqrt() })
        .collect();
    let entries: Vec<(usize, usize, f64)> = adjacency
        .iter()
        .filter(|e| e.0 != e.1)
        .map(|&(i, j, w)| (i, j, -w / (dd[i] * dd[j])))
        .chain((0..n).map(|i| (i, i, 1.0)))
        .collect();
    let (_, vecs) = smallest_eigenpairs(n, &entries, k)?;
    let first = usize::from(drop_first);
    let mut emb = Array2::from_shape_fn((n, n_components), |(i, c)| vecs[[i, c + first]] / dd[i]);
    // sklearn's `_deterministic_vector_sign_flip`.
    for mut col in emb.columns_mut() {
        let big = col
            .iter()
            .copied()
            .fold(0.0_f64, |b, v| if v.abs() > b.abs() { v } else { b });
        if big < 0.0 {
            col.mapv_inplace(|v| -v);
        }
    }
    Ok(emb)
}

/// How to build the affinity graph.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum Affinity {
    /// `exp(-γ ||x_i - x_j||²)` with the given gamma (dense).
    Rbf(f64),
    /// k-NN connectivity including the sample itself, symmetrised as
    /// `0.5·(C + Cᵀ)` (sparse).
    KNearest(usize),
}

/// Affinity graph entries `(i, j, w)` (both triangles).
pub fn build_affinity(x: ArrayView2<f64>, affinity: &Affinity) -> Result<Vec<(usize, usize, f64)>> {
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
                x.view(),
                DistanceMetric::Euclidean,
                NeighborAlgorithm::Auto,
                Some(k),
            )?;
            let (_, nbrs) = index.kneighbors(x.view(), k)?;
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

/// The affinity of [`SpectralEmbedding`] (sklearn's `affinity`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SpectralAffinity {
    /// k-NN connectivity (`kneighbors_graph(include_self=True)`),
    /// symmetrised as `0.5·(C + Cᵀ)`.
    #[default]
    NearestNeighbors,
    /// `exp(-γ ||x_i - x_j||²)`.
    Rbf,
}

/// sklearn's `manifold.SpectralEmbedding` (Laplacian eigenmaps): the
/// [`spectral_embedding`] of the training rows' affinity graph, the first
/// (constant) eigenvector dropped. There is no out-of-sample transform.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpectralEmbedding {
    pub n_components: usize,
    pub affinity: SpectralAffinity,
    /// RBF width (`None`: `1 / n_features`).
    pub gamma: Option<f64>,
    /// Neighbours of the k-NN graph (`None`: `max(n / 10, 1)`).
    pub n_neighbors: Option<usize>,
}

impl SpectralEmbedding {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            affinity: SpectralAffinity::NearestNeighbors,
            gamma: None,
            n_neighbors: None,
        }
    }
}

impl Default for SpectralEmbedding {
    fn default() -> Self {
        Self::new(2)
    }
}

/// A fitted [`SpectralEmbedding`]: the training rows' coordinates.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedSpectralEmbedding {
    /// `(n_samples, n_components)` (sklearn `embedding_`).
    pub embedding: Array2<f64>,
}

impl<F: anofox_ml_core::Float> anofox_ml_core::FitUnsupervised<F> for SpectralEmbedding {
    type Fitted = FittedSpectralEmbedding;

    /// Computed in `f64` whatever the input float type.
    fn fit_view(&self, x: ArrayView2<F>) -> Result<FittedSpectralEmbedding> {
        let x = anofox_ml_core::as_f64(x);
        let (n, p) = x.dim();
        if n < 2 {
            return Err(RustMlError::EmptyInput("need at least 2 samples".into()));
        }
        let affinity = match self.affinity {
            SpectralAffinity::NearestNeighbors => {
                Affinity::KNearest(self.n_neighbors.unwrap_or((n / 10).max(1)))
            }
            SpectralAffinity::Rbf => Affinity::Rbf(self.gamma.unwrap_or(1.0 / p as f64)),
        };
        let graph = build_affinity(x.view(), &affinity)?;
        Ok(FittedSpectralEmbedding {
            embedding: spectral_embedding(n, &graph, self.n_components, true)?,
        })
    }
}
