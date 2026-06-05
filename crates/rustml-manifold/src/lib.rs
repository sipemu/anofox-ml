//! Manifold learning algorithms.
//!
//! Provides:
//! - **Classical MDS** — eigendecomposition variant of `sklearn.manifold.MDS`.
//! - **Isomap** — `sklearn.manifold.Isomap`: k-NN graph + geodesic shortest
//!   paths (Floyd-Warshall) + classical MDS on the geodesic distances.
//!
//! - **LocallyLinearEmbedding** — `sklearn.manifold.LocallyLinearEmbedding`:
//!   local reconstruction weights + bottom-k eigenvectors of `(I − W)ᵀ(I − W)`.
//!
//! Future: t-SNE.

pub mod isomap;
pub mod lle;

pub use isomap::{FittedIsomap, Isomap};
pub use lle::{FittedLocallyLinearEmbedding, LocallyLinearEmbedding};

use faer::linalg::solvers::SelfAdjointEigen;
use faer::{Mat, Side};
use ndarray::{Array1, Array2};
use rustml_core::{FitUnsupervised, Result, RustMlError};

#[derive(Debug, Clone)]
pub struct ClassicalMds {
    pub n_components: usize,
}

impl ClassicalMds {
    pub fn new(n_components: usize) -> Self {
        Self { n_components }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedClassicalMds {
    pub embedding: Array2<f64>,
    pub eigenvalues: Array1<f64>,
}

fn pairwise_dist(x: &Array2<f64>) -> Array2<f64> {
    let n = x.nrows();
    let mut d = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        let xi = x.row(i).to_owned();
        for j in (i + 1)..n {
            let xj = x.row(j).to_owned();
            let mut sd = 0.0;
            for k in 0..x.ncols() {
                sd += (xi[k] - xj[k]).powi(2);
            }
            let v = sd.sqrt();
            d[[i, j]] = v;
            d[[j, i]] = v;
        }
    }
    d
}

impl FitUnsupervised<f64> for ClassicalMds {
    type Fitted = FittedClassicalMds;

    fn fit(&self, x: &Array2<f64>) -> Result<Self::Fitted> {
        let n = x.nrows();
        let k = self.n_components.min(n);
        if k == 0 {
            return Err(RustMlError::InvalidParameter("n_components >= 1".into()));
        }
        let d = pairwise_dist(x);

        // Double-centered squared distance: B = -1/2 * (I - 1/n J) * D² * (I - 1/n J)
        let mut d2 = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                d2[[i, j]] = d[[i, j]] * d[[i, j]];
            }
        }
        let mut row_mean = vec![0.0_f64; n];
        let mut col_mean = vec![0.0_f64; n];
        let mut global = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                row_mean[i] += d2[[i, j]];
                col_mean[j] += d2[[i, j]];
                global += d2[[i, j]];
            }
        }
        let n_f = n as f64;
        for i in 0..n {
            row_mean[i] /= n_f;
            col_mean[i] /= n_f;
        }
        let global = global / (n_f * n_f);

        let mut b = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                b[[i, j]] = -0.5 * (d2[[i, j]] - row_mean[i] - col_mean[j] + global);
            }
        }

        // Symmetric eigendecomposition; eigenvalues come back ascending.
        let m = Mat::<f64>::from_fn(n, n, |i, j| b[[i, j]]);
        let eig = SelfAdjointEigen::new(m.as_ref(), Side::Lower)
            .map_err(|e| RustMlError::InvalidParameter(format!("eigen failed: {e:?}")))?;
        let s = eig.S();
        let v = eig.U();

        let mut embedding = Array2::<f64>::zeros((n, k));
        let mut eigenvalues = Array1::<f64>::zeros(k);
        for c in 0..k {
            let src = n - 1 - c;
            let lam = s.column_vector()[src];
            eigenvalues[c] = lam;
            let scale = lam.max(0.0).sqrt();
            for i in 0..n {
                embedding[[i, c]] = v[(i, src)] * scale;
            }
        }
        Ok(FittedClassicalMds { embedding, eigenvalues })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_mds_recovers_planar_distances() {
        // Points in 2D — MDS to 2 components should preserve pairwise distances.
        let x = array![
            [0.0_f64, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0], [-1.0, 3.0],
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
                    orig[[i, j]], emb[[i, j]]
                );
            }
        }
    }
}
