//! Locally Linear Embedding (LLE).
//!
//! Mirrors `sklearn.manifold.LocallyLinearEmbedding` with the standard
//! algorithm (no Hessian or modified LLE variants):
//!
//! 1. For each `x_i`, find its `k` nearest neighbours.
//! 2. Compute local reconstruction weights `W_{ij}` minimising
//!    `||x_i - Σⱼ W_{ij} x_j||²` subject to `Σⱼ W_{ij} = 1`.
//!    Solved by inverting the local Gram matrix.
//! 3. Form `M = (I − W)ᵀ (I − W)` and take its bottom `n_components + 1`
//!    eigenvectors, dropping the very smallest (corresponds to the constant
//!    direction).

use faer::linalg::solvers::{SelfAdjointEigen, Solve};
use faer::{Mat, Side};
use ndarray::{Array1, Array2};
use rustml_core::{FitUnsupervised, Result, RustMlError};

#[derive(Debug, Clone)]
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
}

fn euclid_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

impl FitUnsupervised<f64> for LocallyLinearEmbedding {
    type Fitted = FittedLocallyLinearEmbedding;

    fn fit(&self, x: &Array2<f64>) -> Result<Self::Fitted> {
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

        // 1. For each point find its k-NN.
        let mut neighbours: Vec<Vec<usize>> = vec![Vec::with_capacity(k); n];
        for i in 0..n {
            let xi: Vec<f64> = x.row(i).iter().copied().collect();
            let mut others: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let xj: Vec<f64> = x.row(j).iter().copied().collect();
                    (j, euclid_sq(&xi, &xj))
                })
                .collect();
            others.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            for &(j, _) in others.iter().take(k) {
                neighbours[i].push(j);
            }
        }

        // 2. Reconstruction weights. Solve C w = 1 (then normalise), where
        //    C_{ab} = (x_i - x_a) · (x_i - x_b) for a, b in N(i).
        let mut weights = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            let nbrs = &neighbours[i];
            let m = nbrs.len();
            let xi: Vec<f64> = x.row(i).iter().copied().collect();
            // Build C.
            let mut c = Array2::<f64>::zeros((m, m));
            let mut diffs: Vec<Vec<f64>> = Vec::with_capacity(m);
            for &j in nbrs {
                let xj: Vec<f64> = x.row(j).iter().copied().collect();
                let dij: Vec<f64> = xi.iter().zip(xj.iter()).map(|(a, b)| a - b).collect();
                diffs.push(dij);
            }
            for a in 0..m {
                for b in 0..m {
                    let mut s = 0.0;
                    for q in 0..d {
                        s += diffs[a][q] * diffs[b][q];
                    }
                    c[[a, b]] = s;
                }
            }
            // Regularise diagonal: add reg * trace(C) / m.
            let trace: f64 = (0..m).map(|q| c[[q, q]]).sum();
            let lambda = self.reg * trace.max(1e-12) / m as f64;
            for a in 0..m {
                c[[a, a]] += lambda;
            }
            // Solve C w = 1.
            let cm = Mat::<f64>::from_fn(m, m, |i, j| c[[i, j]]);
            let llt = faer::linalg::solvers::Llt::new(cm.as_ref(), Side::Lower)
                .map_err(|e| RustMlError::InvalidParameter(format!("Cholesky failed: {e:?}")))?;
            let ones = Mat::<f64>::from_fn(m, 1, |_, _| 1.0);
            let w_raw = llt.solve(&ones);
            let sum: f64 = (0..m).map(|q| w_raw[(q, 0)]).sum();
            let sum = sum.max(1e-12);
            for (q, &j) in nbrs.iter().enumerate() {
                weights[i][j] = w_raw[(q, 0)] / sum;
            }
        }

        // 3. Build M = (I − W)ᵀ (I − W). Compute directly as
        //    M = I − W − Wᵀ + WᵀW.
        let mut m = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            m[[i, i]] += 1.0;
            for j in 0..n {
                m[[i, j]] -= weights[i][j];
                m[[j, i]] -= weights[i][j];
            }
        }
        // Add Wᵀ W.
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for q in 0..n {
                    s += weights[q][i] * weights[q][j];
                }
                m[[i, j]] += s;
            }
        }

        // 4. Bottom (n_components + 1) eigenvectors of M; drop the smallest.
        let mat = Mat::<f64>::from_fn(n, n, |i, j| m[[i, j]]);
        let eig = SelfAdjointEigen::new(mat.as_ref(), Side::Lower)
            .map_err(|e| RustMlError::InvalidParameter(format!("eigen failed: {e:?}")))?;
        let s = eig.S();
        let v = eig.U();
        // Ascending eigenvalues: skip the very smallest (constant direction).
        let mut embedding = Array2::<f64>::zeros((n, self.n_components));
        let mut eigenvalues = Array1::<f64>::zeros(self.n_components);
        for c in 0..self.n_components {
            let src = c + 1;
            eigenvalues[c] = s.column_vector()[src];
            for i in 0..n {
                embedding[[i, c]] = v[(i, src)];
            }
        }
        Ok(FittedLocallyLinearEmbedding {
            embedding,
            eigenvalues,
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
