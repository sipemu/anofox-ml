//! Locally Linear Embedding (LLE).
//!
//! Mirrors `sklearn.manifold.LocallyLinearEmbedding` with the standard
//! algorithm (no Hessian or modified LLE variants):
//!
//! 1. For each `x_i`, find its `k` nearest neighbours.
//! 2. Compute local reconstruction weights `W_{ij}` minimising
//!    `||x_i - Σⱼ W_{ij} x_j||²` subject to `Σⱼ W_{ij} = 1`.
//!    Solved from the local Gram matrix regularised by `reg · trace` (as
//!    sklearn's `barycenter_weights`).
//! 3. Form the sparse `M = (I − W)ᵀ (I − W)` (O(n·k²) entries) and take its
//!    bottom `n_components + 1` eigenvectors (dense for small n, sparse
//!    shift-invert Lanczos otherwise), dropping the very smallest (the
//!    constant direction).

use anofox_ml_core::Float;
use anofox_ml_core::{FitUnsupervised, Result, RustMlError};
use anofox_ml_neighbors::{DistanceMetric, NeighborAlgorithm, NeighborIndex};
use faer::linalg::solvers::Solve;
use faer::{Mat, Side};
use ndarray::{Array1, Array2};
use rayon::prelude::*;

use crate::sparse_eigen::smallest_eigenpairs;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LocallyLinearEmbedding {
    pub n_components: usize,
    pub n_neighbors: usize,
    pub reg: f64,
}

impl LocallyLinearEmbedding {
    pub fn new(n_components: usize, n_neighbors: usize) -> Self {
        Self {
            n_components,
            n_neighbors,
            reg: 1e-3,
        }
    }
    pub fn with_reg(mut self, r: f64) -> Self {
        self.reg = r;
        self
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedLocallyLinearEmbedding {
    pub embedding: Array2<f64>,
    pub eigenvalues: Array1<f64>,
    /// Sum of the kept eigenvalues (sklearn's `reconstruction_error_`).
    #[serde(default)]
    pub reconstruction_error: f64,
}

impl<F: Float> FitUnsupervised<F> for LocallyLinearEmbedding {
    type Fitted = FittedLocallyLinearEmbedding;

    /// Computed in `f64` whatever the input float type.
    fn fit(&self, x: &Array2<F>) -> Result<Self::Fitted> {
        let x_f64 = anofox_ml_core::as_f64(x);
        let x: &Array2<f64> = &x_f64;
        let n = x.nrows();
        let d = x.ncols();
        if n == 0 {
            return Err(RustMlError::EmptyInput("empty input".into()));
        }
        let k = self.n_neighbors.min(n.saturating_sub(1));
        if k == 0 || self.n_components == 0 {
            return Err(RustMlError::InvalidParameter(
                "n_neighbors and n_components must be >= 1".into(),
            ));
        }
        if self.n_components >= n {
            return Err(RustMlError::InvalidParameter(format!(
                "n_components ({}) must be < n_samples ({})",
                self.n_components, n
            )));
        }

        // 1. For each point find its k-NN (shared neighbour index).
        let index = NeighborIndex::new(
            x,
            DistanceMetric::Euclidean,
            NeighborAlgorithm::Auto,
            Some(k),
        )?;
        let (_, knn) = index.kneighbors_self(k)?;
        let neighbours: Vec<Vec<usize>> = knn.rows().into_iter().map(|r| r.to_vec()).collect();

        // 2. Reconstruction weights (sklearn's `barycenter_weights`): solve
        //    (G + R I) w = 1 with G_ab = (x_a - x_i)·(x_b - x_i) and
        //    R = reg·trace(G) (reg if the trace is 0), then normalise.
        let weights: Vec<Vec<f64>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let nbrs = &neighbours[i];
                let m = nbrs.len();
                let diffs: Vec<Vec<f64>> = nbrs
                    .iter()
                    .map(|&j| (0..d).map(|q| x[[j, q]] - x[[i, q]]).collect())
                    .collect();
                let mut g = Mat::<f64>::from_fn(m, m, |a, b| {
                    diffs[a].iter().zip(&diffs[b]).map(|(u, v)| u * v).sum()
                });
                let trace: f64 = (0..m).map(|q| g[(q, q)]).sum();
                let r = if trace > 0.0 {
                    self.reg * trace
                } else {
                    self.reg
                };
                for q in 0..m {
                    g[(q, q)] += r;
                }
                let ones = Mat::<f64>::from_fn(m, 1, |_, _| 1.0);
                let w = match faer::linalg::solvers::Llt::new(g.as_ref(), Side::Lower) {
                    Ok(llt) => llt.solve(&ones),
                    Err(_) => g.as_ref().full_piv_lu().solve(&ones),
                };
                let sum: f64 = (0..m).map(|q| w[(q, 0)]).sum();
                (0..m).map(|q| w[(q, 0)] / sum).collect()
            })
            .collect();

        // 3. M = (I − W)ᵀ (I − W) = Σ_i r_i r_iᵀ with r_i = e_i − W_iᵀ, each
        //    with k + 1 non-zeros: O(n·k²) entries.
        let mut entries: Vec<(usize, usize, f64)> = Vec::with_capacity(n * (k + 1) * (k + 1));
        for i in 0..n {
            let row: Vec<(usize, f64)> = std::iter::once((i, 1.0))
                .chain(
                    neighbours[i]
                        .iter()
                        .zip(&weights[i])
                        .map(|(&j, &w)| (j, -w)),
                )
                .collect();
            for &(a, va) in &row {
                for &(b, vb) in &row {
                    entries.push((a, b, va * vb));
                }
            }
        }

        // 4. Bottom (n_components + 1) eigenvectors of M; drop the smallest
        //    (the constant direction).
        let (vals, vecs) = smallest_eigenpairs(n, &entries, self.n_components + 1)?;
        let embedding = vecs.slice(ndarray::s![.., 1..]).to_owned();
        let eigenvalues = vals.slice(ndarray::s![1..]).to_owned();
        let reconstruction_error = eigenvalues.sum();
        Ok(FittedLocallyLinearEmbedding {
            embedding,
            eigenvalues,
            reconstruction_error,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_lle_runs_2d_grid() {
        let x = array![
            [0.0_f64, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [0.0, 2.0],
            [1.0, 2.0],
            [2.0, 2.0],
        ];
        let fitted = LocallyLinearEmbedding::new(2, 3).fit(&x).unwrap();
        assert_eq!(fitted.embedding.shape(), &[9, 2]);
        for v in fitted.embedding.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_lle_unrolls_arc() {
        let n = 25;
        let mut x = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let t = i as f64 * 0.15;
            x[[i, 0]] = t.cos() * 5.0;
            x[[i, 1]] = t.sin() * 5.0;
        }
        let fitted = LocallyLinearEmbedding::new(1, 3).fit(&x).unwrap();
        assert_eq!(fitted.embedding.shape(), &[n, 1]);
        // Embedding should be monotone in i (the arc parameter), modulo sign.
        let e: Vec<f64> = fitted.embedding.column(0).iter().copied().collect();
        let inc = e.windows(2).all(|w| w[1] >= w[0] - 1e-3);
        let dec = e.windows(2).all(|w| w[1] <= w[0] + 1e-3);
        assert!(inc || dec, "non-monotone: {:?}", e);
    }
}
