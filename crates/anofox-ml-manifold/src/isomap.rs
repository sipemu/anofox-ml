//! Isomap — non-linear dimensionality reduction via geodesic distances.
//!
//! Mirrors `sklearn.manifold.Isomap` with a brute-force k-NN graph and
//! Floyd-Warshall all-pairs shortest paths:
//!
//! 1. Build the k-nearest-neighbor graph G(V, E).
//! 2. Compute geodesic distances d_g(i, j) = shortest-path(i, j) in G.
//! 3. Apply classical MDS to the geodesic distance matrix.

use anofox_ml_core::{FitUnsupervised, Result, RustMlError};
use faer::linalg::solvers::SelfAdjointEigen;
use faer::{Mat, Side};
use ndarray::{Array1, Array2};

#[derive(Debug, Clone)]
pub struct Isomap {
    pub n_components: usize,
    pub n_neighbors: usize,
}

impl Isomap {
    pub fn new(n_components: usize, n_neighbors: usize) -> Self {
        Self {
            n_components,
            n_neighbors,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedIsomap {
    pub embedding: Array2<f64>,
    pub eigenvalues: Array1<f64>,
}

fn euclid(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

impl FitUnsupervised<f64> for Isomap {
    type Fitted = FittedIsomap;

    fn fit(&self, x: &Array2<f64>) -> Result<Self::Fitted> {
        let n = x.nrows();
        if n == 0 {
            return Err(RustMlError::EmptyInput("empty input".into()));
        }
        let k = self.n_neighbors.min(n.saturating_sub(1));
        if k == 0 {
            return Err(RustMlError::InvalidParameter("n_neighbors >= 1".into()));
        }
        let n_components = self.n_components.min(n);
        if n_components == 0 {
            return Err(RustMlError::InvalidParameter("n_components >= 1".into()));
        }

        // 1. Pairwise distances.
        let mut pd = vec![vec![f64::INFINITY; n]; n];
        for i in 0..n {
            let xi: Vec<f64> = x.row(i).iter().copied().collect();
            pd[i][i] = 0.0;
            for j in (i + 1)..n {
                let xj: Vec<f64> = x.row(j).iter().copied().collect();
                let d = euclid(&xi, &xj);
                pd[i][j] = d;
                pd[j][i] = d;
            }
        }

        // 2. Build k-NN graph: keep d(i,j) only if j is in i's k-NN OR vice versa
        //    (mutual k-NN), and set others to INFINITY.
        let mut g = vec![vec![f64::INFINITY; n]; n];
        for i in 0..n {
            let mut others: Vec<(usize, f64)> =
                (0..n).filter(|&j| j != i).map(|j| (j, pd[i][j])).collect();
            others.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            for &(j, d) in others.iter().take(k) {
                g[i][j] = d;
            }
            g[i][i] = 0.0;
        }
        // Symmetrize: take min of g[i][j] and g[j][i].
        for i in 0..n {
            for j in (i + 1)..n {
                let mn = g[i][j].min(g[j][i]);
                g[i][j] = mn;
                g[j][i] = mn;
            }
        }

        // 3. Floyd-Warshall.
        let mut d = g;
        for kk in 0..n {
            for i in 0..n {
                if !d[i][kk].is_finite() {
                    continue;
                }
                for j in 0..n {
                    let alt = d[i][kk] + d[kk][j];
                    if alt < d[i][j] {
                        d[i][j] = alt;
                    }
                }
            }
        }

        // 4. Classical MDS on the geodesic distance matrix.
        let mut d2 = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                d2[i][j] = if d[i][j].is_finite() {
                    d[i][j] * d[i][j]
                } else {
                    0.0
                };
            }
        }
        let mut row_mean = vec![0.0_f64; n];
        let mut col_mean = vec![0.0_f64; n];
        let mut global = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                row_mean[i] += d2[i][j];
                col_mean[j] += d2[i][j];
                global += d2[i][j];
            }
        }
        let n_f = n as f64;
        for v in &mut row_mean {
            *v /= n_f;
        }
        for v in &mut col_mean {
            *v /= n_f;
        }
        let global = global / (n_f * n_f);

        let mut b = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                b[[i, j]] = -0.5 * (d2[i][j] - row_mean[i] - col_mean[j] + global);
            }
        }
        let m = Mat::<f64>::from_fn(n, n, |i, j| b[[i, j]]);
        let eig = SelfAdjointEigen::new(m.as_ref(), Side::Lower)
            .map_err(|e| RustMlError::InvalidParameter(format!("eigen failed: {e:?}")))?;
        let s = eig.S();
        let v = eig.U();
        let mut embedding = Array2::<f64>::zeros((n, n_components));
        let mut eigenvalues = Array1::<f64>::zeros(n_components);
        for c in 0..n_components {
            let src = n - 1 - c;
            let lam = s.column_vector()[src];
            eigenvalues[c] = lam;
            let scale = lam.max(0.0).sqrt();
            for i in 0..n {
                embedding[[i, c]] = v[(i, src)] * scale;
            }
        }
        Ok(FittedIsomap {
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
    fn test_isomap_unrolls_swiss_strip() {
        // 1D manifold embedded in 2D: a curved arc. Geodesic distances along
        // the arc should be near-linearly related to a 1D embedding.
        let n = 20;
        let mut x = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let t = i as f64 * 0.2;
            x[[i, 0]] = t.cos() * 5.0;
            x[[i, 1]] = t.sin() * 5.0;
        }
        let fitted = Isomap::new(1, 3).fit(&x).unwrap();
        assert_eq!(fitted.embedding.shape(), &[n, 1]);
        // Embedding should be monotone in i (the arc parameter).
        let e: Vec<f64> = fitted.embedding.column(0).iter().copied().collect();
        let monotone_inc = e.windows(2).all(|w| w[1] >= w[0] - 1e-6);
        let monotone_dec = e.windows(2).all(|w| w[1] <= w[0] + 1e-6);
        assert!(monotone_inc || monotone_dec, "embedding={:?}", e);
    }

    #[test]
    fn test_isomap_runs_2d_to_2d() {
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
        let fitted = Isomap::new(2, 3).fit(&x).unwrap();
        assert_eq!(fitted.embedding.shape(), &[9, 2]);
        for v in fitted.embedding.iter() {
            assert!(v.is_finite());
        }
    }
}
