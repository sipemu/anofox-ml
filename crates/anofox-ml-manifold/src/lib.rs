//! Manifold learning algorithms.
//!
//! Provides:
//! - **Classical MDS** — `sklearn.manifold.ClassicalMDS`.
//! - **Isomap** — `sklearn.manifold.Isomap`: k-NN graph + geodesic shortest
//!   paths (Dijkstra) + classical MDS on the geodesic distances.
//!
//! - **LocallyLinearEmbedding** — `sklearn.manifold.LocallyLinearEmbedding`:
//!   local reconstruction weights + bottom-k eigenvectors of `(I − W)ᵀ(I − W)`.
//!
//! - **t-SNE** — `sklearn.manifold.TSNE`, exact or Barnes–Hut.
//! - **spectral_embedding** — `sklearn.manifold.spectral_embedding`.

use anofox_ml_core::Float;
pub mod isomap;
pub mod lle;
pub mod sparse_eigen;
pub mod spectral_embedding;
pub mod tsne;

pub use isomap::{FittedIsomap, Isomap};
pub use lle::{FittedLocallyLinearEmbedding, LocallyLinearEmbedding};
pub use spectral_embedding::{
    build_affinity, spectral_embedding, Affinity, FittedSpectralEmbedding, SpectralAffinity,
    SpectralEmbedding,
};
pub use tsne::{FittedTSne, TSne, TSneInit, TSneMethod};

use anofox_ml_core::{FitUnsupervised, Result, RustMlError};
use anofox_ml_neighbors::distance::compute_distance;
use anofox_ml_neighbors::DistanceMetric;
use faer::linalg::solvers::SelfAdjointEigen;
use faer::{Mat, Side};
use ndarray::{Array1, Array2, ArrayView2, Axis};

/// Classical (Torgerson) multidimensional scaling, as sklearn's
/// `ClassicalMDS`: the top eigenvectors of the double-centred squared
/// dissimilarities, scaled by the square roots of their eigenvalues, each
/// flipped so its largest absolute entry is positive.
///
/// With the Euclidean metric on raw features this equals PCA of the centred
/// data and is computed from the `d × d` covariance (O(n·d²)). Other metrics
/// build the `n × n` dissimilarity matrix; `Precomputed` takes it as input.
/// Negative eigenvalues (non-Euclidean dissimilarities) give zero
/// coordinates instead of NaN.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ClassicalMds {
    pub n_components: usize,
    /// Dissimilarity metric; `Precomputed` means `fit` receives the square
    /// symmetric dissimilarity matrix.
    #[serde(default)]
    pub metric: DistanceMetric,
}

impl ClassicalMds {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            metric: DistanceMetric::Euclidean,
        }
    }
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedClassicalMds {
    pub embedding: Array2<f64>,
    pub eigenvalues: Array1<f64>,
}

/// sklearn's `svd_flip` on columns: largest |entry| of each column positive.
fn flip_columns(m: &mut Array2<f64>) {
    for mut col in m.columns_mut() {
        let big = col
            .iter()
            .copied()
            .fold(0.0_f64, |b, v| if v.abs() > b.abs() { v } else { b });
        if big < 0.0 {
            col.mapv_inplace(|v| -v);
        }
    }
}

/// Top-`k` eigenpairs (descending) of a symmetric matrix.
fn top_eigen(b: &Array2<f64>, k: usize) -> Result<(Array1<f64>, Array2<f64>)> {
    let n = b.nrows();
    let m = Mat::<f64>::from_fn(n, n, |i, j| b[[i, j]]);
    let eig = SelfAdjointEigen::new(m.as_ref(), Side::Lower)
        .map_err(|e| RustMlError::InvalidParameter(format!("eigen failed: {e:?}")))?;
    let (s, u) = (eig.S(), eig.U());
    let vals = Array1::from_shape_fn(k, |c| s.column_vector()[n - 1 - c]);
    let vecs = Array2::from_shape_fn((n, k), |(i, c)| u[(i, n - 1 - c)]);
    Ok((vals, vecs))
}

impl<F: Float> FitUnsupervised<F> for ClassicalMds {
    type Fitted = FittedClassicalMds;

    /// Computed in `f64` whatever the input float type.
    fn fit_view(&self, x: ArrayView2<F>) -> Result<Self::Fitted> {
        let x_f64 = anofox_ml_core::as_f64(x);
        let x = x_f64.view();
        let n = x.nrows();
        let k = self.n_components;
        if n == 0 {
            return Err(RustMlError::EmptyInput("empty input".into()));
        }
        if k == 0 || k > n {
            return Err(RustMlError::InvalidParameter(format!(
                "n_components must be in [1, {n}], got {k}"
            )));
        }
        let (eigenvalues, mut u) = if self.metric.canonical_is_euclidean() {
            // B = Xc Xcᵀ: its nonzero eigenpairs come from Xcᵀ Xc = V S² Vᵀ
            // with eigenvectors Xc V / S.
            let d = x.ncols();
            let xc = &x - &x.mean_axis(Axis(0)).expect("non-empty");
            let kk = k.min(d);
            let (vals, v) = top_eigen(&xc.t().dot(&xc), kk)?;
            let proj = xc.dot(&v);
            let mut u = Array2::<f64>::zeros((n, k));
            let mut w = Array1::<f64>::zeros(k);
            for c in 0..kk {
                w[c] = vals[c];
                let s = vals[c].max(0.0).sqrt();
                if s > 0.0 {
                    u.column_mut(c).assign(&(&proj.column(c) / s));
                }
            }
            (w, u)
        } else {
            let d2 = if self.metric == DistanceMetric::Precomputed {
                if x.ncols() != n {
                    return Err(RustMlError::ShapeMismatch(format!(
                        "precomputed dissimilarities must be square, got {n}x{}",
                        x.ncols()
                    )));
                }
                if (0..n).any(|i| {
                    (0..i).any(|j| (x[[i, j]] - x[[j, i]]).abs() > 1e-10 * x[[i, j]].abs().max(1.0))
                }) {
                    return Err(RustMlError::InvalidParameter(
                        "precomputed dissimilarities must be symmetric".into(),
                    ));
                }
                x.mapv(|v| v * v)
            } else {
                Array2::from_shape_fn((n, n), |(i, j)| {
                    compute_distance(&x.row(i), &x.row(j), self.metric).powi(2)
                })
            };
            // B = -½ J D² J.
            let mut b = d2;
            let col_mean = b.mean_axis(Axis(0)).expect("non-empty");
            b -= &col_mean;
            let row_mean = b.mean_axis(Axis(1)).expect("non-empty");
            b -= &row_mean.insert_axis(Axis(1));
            b *= -0.5;
            top_eigen(&b, k)?
        };
        flip_columns(&mut u);
        let mut embedding = u;
        for (mut col, &w) in embedding.columns_mut().into_iter().zip(&eigenvalues) {
            col *= w.max(0.0).sqrt();
        }
        Ok(FittedClassicalMds {
            embedding,
            eigenvalues,
        })
    }
}

#[cfg(test)]
fn pairwise_dist(x: &Array2<f64>) -> Array2<f64> {
    let n = x.nrows();
    Array2::from_shape_fn((n, n), |(i, j)| {
        compute_distance(&x.row(i), &x.row(j), DistanceMetric::Euclidean)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_mds_recovers_planar_distances() {
        // Points in 2D — MDS to 2 components should preserve pairwise distances.
        let x = array![
            [0.0_f64, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 2.0],
            [-1.0, 3.0],
        ];
        let mds = ClassicalMds::new(2);
        let fitted = mds.fit(&x).unwrap();
        // Pairwise distances in the embedding should match the originals.
        let orig = pairwise_dist(&x);
        let emb = pairwise_dist(&fitted.embedding);
        for i in 0..5 {
            for j in 0..5 {
                assert!(
                    (orig[[i, j]] - emb[[i, j]]).abs() < 1e-6,
                    "dist[{i},{j}]: orig {} vs emb {}",
                    orig[[i, j]],
                    emb[[i, j]]
                );
            }
        }
    }
}
