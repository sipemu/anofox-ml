//! Isomap — non-linear dimensionality reduction via geodesic distances.
//!
//! Mirrors `sklearn.manifold.Isomap` (k-NN graph from the shared neighbour
//! index, Dijkstra shortest paths as `path_method="D"`):
//!
//! 1. Build the k-nearest-neighbor graph G(V, E), symmetrised as the
//!    **union** of the directed k-NN relations (edge i–j if j is among i's
//!    k nearest neighbours *or* i is among j's), like sklearn's
//!    `kneighbors_graph` + `shortest_path(directed=False)`.
//! 2. Compute geodesic distances d_g(i, j) = shortest-path(i, j) in G.
//! 3. Apply classical MDS to the geodesic distance matrix.
//!
//! If the k-NN graph is disconnected, geodesic distances between components
//! are undefined and `fit` returns [`RustMlError::InvalidParameter`] asking
//! for a larger `n_neighbors` (instead of silently treating unreachable pairs
//! as distance 0, which collapsed unrelated points onto each other).

use anofox_ml_core::Float;
use std::collections::BinaryHeap;

use anofox_ml_core::{FitUnsupervised, Result, RustMlError};
use anofox_ml_neighbors::{DistanceMetric, NeighborAlgorithm, NeighborIndex};
use faer::linalg::solvers::SelfAdjointEigen;
use faer::{Mat, Side};
use ndarray::{Array1, Array2};
use rayon::prelude::*;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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

/// Shortest-path distances from `source` (∞ for unreachable nodes).
fn dijkstra(adj: &[Vec<(usize, f64)>], source: usize) -> Vec<f64> {
    #[derive(PartialEq)]
    struct Item(f64, usize);
    impl Eq for Item {}
    impl PartialOrd for Item {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Item {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            // Reversed: BinaryHeap pops the smallest distance first.
            other.0.total_cmp(&self.0).then(other.1.cmp(&self.1))
        }
    }
    let mut dist = vec![f64::INFINITY; adj.len()];
    dist[source] = 0.0;
    let mut heap = BinaryHeap::from([Item(0.0, source)]);
    while let Some(Item(du, u)) = heap.pop() {
        if du > dist[u] {
            continue;
        }
        for &(v, w) in &adj[u] {
            let alt = du + w;
            if alt < dist[v] {
                dist[v] = alt;
                heap.push(Item(alt, v));
            }
        }
    }
    dist
}

impl<F: Float> FitUnsupervised<F> for Isomap {
    type Fitted = FittedIsomap;

    /// Computed in `f64` whatever the input float type.
    fn fit(&self, x: &Array2<F>) -> Result<Self::Fitted> {
        let x_f64 = anofox_ml_core::as_f64(x);
        let x: &Array2<f64> = &x_f64;
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

        // 1-2. k-NN graph, symmetrised as the union of both directions
        //      (sklearn), as adjacency lists.
        let index = NeighborIndex::new(
            x,
            DistanceMetric::Euclidean,
            NeighborAlgorithm::Auto,
            Some(k),
        )?;
        let (knn_d, knn_i) = index.kneighbors_self(k)?;
        let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::with_capacity(2 * k); n];
        for i in 0..n {
            for c in 0..k {
                let (j, w) = (knn_i[[i, c]], knn_d[[i, c]]);
                adj[i].push((j, w));
                adj[j].push((i, w));
            }
        }

        // 3. Geodesic distances: Dijkstra from every source over the sparse
        //    graph (sklearn's `path_method="D"`), in parallel.
        let d: Vec<Vec<f64>> = (0..n).into_par_iter().map(|s| dijkstra(&adj, s)).collect();

        // Disconnected k-NN graph: geodesics between components are infinite.
        if d.iter().any(|row| row.iter().any(|v| !v.is_finite())) {
            // Count connected components for a helpful message.
            let mut comp = vec![usize::MAX; n];
            let mut n_comp = 0;
            for s0 in 0..n {
                if comp[s0] != usize::MAX {
                    continue;
                }
                for j in 0..n {
                    if d[s0][j].is_finite() {
                        comp[j] = n_comp;
                    }
                }
                n_comp += 1;
            }
            return Err(RustMlError::InvalidParameter(format!(
                "Isomap: the k-nearest-neighbour graph (n_neighbors = {k}) has {n_comp} \
                 connected components, so geodesic distances between them are undefined; \
                 increase n_neighbors"
            )));
        }

        // 4. Classical MDS on the geodesic distance matrix.
        let mut d2 = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                d2[i][j] = d[i][j] * d[i][j];
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
    /// Regression for #57: two far-apart groups with a small `n_neighbors`
    /// give a disconnected k-NN graph. This used to "succeed" with the
    /// infinite geodesics set to 0; it must now be an error.
    #[test]
    fn test_isomap_disconnected_graph_errors() {
        let x = array![
            [0.0_f64, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [100.0, 0.0],
            [100.1, 0.0],
            [100.2, 0.0],
        ];
        match Isomap::new(1, 2).fit(&x) {
            Err(RustMlError::InvalidParameter(msg)) => {
                assert!(msg.contains("2 connected components"), "{msg}");
                assert!(msg.contains("n_neighbors"), "{msg}");
            }
            other => panic!("expected InvalidParameter, got {other:?}"),
        }
        // Enough neighbours to connect the groups -> fine.
        let f = Isomap::new(1, 3).fit(&x).unwrap();
        assert!(f.embedding.iter().all(|v| v.is_finite()));
    }

    /// The graph is the union of directed k-NN relations: the outlier's
    /// nearest neighbour does not list it, but the edge still exists.
    #[test]
    fn test_isomap_union_knn_graph_connects_outlier() {
        let x = array![[0.0_f64], [1.0], [2.0], [3.0], [10.0]];
        assert!(Isomap::new(1, 1).fit(&x).is_ok());
    }
}
