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
//! If the k-NN graph is disconnected, every pair of components is joined by
//! an edge between its closest pair of points, as sklearn's
//! `_fix_connected_components` does (sklearn also warns; increasing
//! `n_neighbors` is usually the better fix).

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
/// Component label of every node, numbered in order of their smallest node
/// (as `scipy.sparse.csgraph.connected_components`).
fn connected_components(adj: &[Vec<(usize, f64)>]) -> Vec<usize> {
    let n = adj.len();
    let mut label = vec![usize::MAX; n];
    let mut next = 0;
    for start in 0..n {
        if label[start] != usize::MAX {
            continue;
        }
        let mut stack = vec![start];
        label[start] = next;
        while let Some(u) = stack.pop() {
            for &(v, _) in &adj[u] {
                if label[v] == usize::MAX {
                    label[v] = next;
                    stack.push(v);
                }
            }
        }
        next += 1;
    }
    label
}

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

        // Disconnected k-NN graph: like sklearn (`_fix_connected_components`),
        // connect every pair of components by an edge between their closest
        // pair of points.
        let labels = connected_components(&adj);
        let n_comp = labels.iter().max().map_or(0, |&m| m + 1);
        if n_comp > 1 {
            let members: Vec<Vec<usize>> = (0..n_comp)
                .map(|c| (0..n).filter(|&i| labels[i] == c).collect())
                .collect();
            let dist = |a: usize, b: usize| {
                x.row(a)
                    .iter()
                    .zip(x.row(b).iter())
                    .map(|(p, q)| (p - q) * (p - q))
                    .sum::<f64>()
                    .sqrt()
            };
            for ci in 0..n_comp {
                for cj in 0..ci {
                    // First minimum in row-major order over (members[ci], members[cj]).
                    let mut best = (f64::INFINITY, 0, 0);
                    for &a in &members[ci] {
                        for &b in &members[cj] {
                            let dd = dist(a, b);
                            if dd < best.0 {
                                best = (dd, a, b);
                            }
                        }
                    }
                    let (w, a, b) = best;
                    adj[a].push((b, w));
                    adj[b].push((a, w));
                }
            }
        }

        // 3. Geodesic distances: Dijkstra from every source over the sparse
        //    graph (sklearn's `path_method="D"`), in parallel.
        let d: Vec<Vec<f64>> = (0..n).into_par_iter().map(|s| dijkstra(&adj, s)).collect();

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
        // As sklearn's KernelPCA on the top `n_components`:
        // `_check_psd_eigenvalues` (significant negatives are an error, small
        // ones and positives below 1e-12 · max become 0) and `svd_flip`
        // signs (largest-|·| entry of each eigenvector positive).
        let sv = s.column_vector();
        let top: Vec<f64> = (0..n_components).map(|c| sv[n - 1 - c]).collect();
        let max_eig = top[0];
        let min_eig = top.iter().copied().fold(f64::INFINITY, f64::min);
        if max_eig > 0.0 && min_eig < -1e-5 * max_eig && min_eig < -1e-10 {
            return Err(RustMlError::InvalidParameter(format!(
                "Isomap: the geodesic kernel has significant negative eigenvalues \
                 ({min_eig:e}, largest {max_eig:e}); reduce n_components"
            )));
        }
        for (c, &raw) in top.iter().enumerate() {
            let src = n - 1 - c;
            let lam = if raw <= 0.0 || raw < 1e-12 * max_eig {
                0.0
            } else {
                raw
            };
            eigenvalues[c] = lam;
            let mut best = 0;
            for i in 1..n {
                if v[(i, src)].abs() > v[(best, src)].abs() {
                    best = i;
                }
            }
            let sign = if v[(best, src)] < 0.0 { -1.0 } else { 1.0 };
            let scale = lam.sqrt();
            for i in 0..n {
                embedding[[i, c]] = sign * v[(i, src)] * scale;
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
    /// A disconnected k-NN graph is completed like sklearn's
    /// `_fix_connected_components` (one edge between the closest pair of
    /// every two components). sklearn 1.8:
    /// `Isomap(n_neighbors=2, n_components=2).fit_transform(X)`.
    #[test]
    fn test_isomap_disconnected_graph_matches_sklearn() {
        let x = array![
            [0.0_f64, 0.0],
            [0.1, 0.0],
            [0.2, 0.05],
            [100.0, 0.0],
            [100.1, 0.02],
            [100.3, 0.0],
            [50.0, 40.0],
            [50.2, 40.1],
            [50.1, 39.8],
        ];
        let want = [
            [-50.104_054_958_360_85, -13.416_570_398_743_627],
            [-50.009_711_635_069_09, -13.383_370_874_972_231],
            [-49.897_920_214_377_095, -13.344_032_995_374_707],
            [49.902_087_715_493_096, -13.308_429_270_771_782],
            [50.004_075_358_662_725, -13.344_215_403_409_031],
            [50.202_112_917_331_384, -13.413_707_664_162_363],
            [-0.032_240_289_567_270_673, 26.806_514_233_030_352],
            [-0.032_333_436_700_541_12, 26.954_576_682_271_16],
            [-0.032_015_457_412_341_91, 26.449_235_692_132_15],
        ];
        let f = Isomap::new(2, 2).fit(&x).unwrap();
        for i in 0..9 {
            for c in 0..2 {
                assert!(
                    (f.embedding[[i, c]] - want[i][c]).abs() < 1e-8,
                    "[{i},{c}] {} vs {}",
                    f.embedding[[i, c]],
                    want[i][c]
                );
            }
        }
    }

    /// The graph is the union of directed k-NN relations: the outlier's
    /// nearest neighbour does not list it, but the edge still exists.
    #[test]
    fn test_isomap_union_knn_graph_connects_outlier() {
        let x = array![[0.0_f64], [1.0], [2.0], [3.0], [10.0]];
        assert!(Isomap::new(1, 1).fit(&x).is_ok());
    }
}
