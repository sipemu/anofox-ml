//! Shared nearest-neighbour index (sklearn's `NearestNeighbors`): batched,
//! parallel k-nearest and radius queries by brute force or a KD-tree, for
//! the neighbour estimators here and the clustering / manifold crates.

use anofox_ml_core::{Float, Result, RustMlError};
use ndarray::{Array2, ArrayView1, ArrayView2, AsArray, Ix2};
use rayon::prelude::*;

use crate::distance::DistanceMetric;
use crate::kdtree::{Cand, KdTree, DEFAULT_LEAF_SIZE};

/// Search algorithm of a [`NeighborIndex`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum NeighborAlgorithm {
    /// sklearn's rule: brute force for precomputed or non-Minkowski
    /// metrics, more than 15 features, or `n_neighbors >= n_samples / 2`;
    /// otherwise a KD-tree.
    #[default]
    Auto,
    /// Compare every query with every training sample.
    BruteForce,
    /// KD-tree (Minkowski metrics only).
    KdTree,
}

impl NeighborAlgorithm {
    /// Brute force or KD-tree for `n` samples with `d` features.
    fn use_tree(
        self,
        metric: DistanceMetric,
        n: usize,
        d: usize,
        n_neighbors: Option<usize>,
    ) -> Result<bool> {
        match self {
            NeighborAlgorithm::BruteForce => Ok(false),
            NeighborAlgorithm::KdTree if !metric.is_minkowski() => Err(
                RustMlError::InvalidParameter(format!("KD-tree does not support {metric:?}")),
            ),
            NeighborAlgorithm::KdTree => Ok(true),
            NeighborAlgorithm::Auto => {
                Ok(metric.is_minkowski() && d <= 15 && n_neighbors.is_none_or(|k| k < n / 2))
            }
        }
    }
}

/// Neighbour lists of a radius query: per query, (index, distance) pairs
/// sorted by distance, then index.
pub type RadiusNeighbors<F> = Vec<Vec<(usize, F)>>;

/// k-nearest / radius neighbour index over a set of training samples.
///
/// Results are exact for every algorithm and identical between them: ties
/// in distance are broken by the lower training index.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct NeighborIndex<F: Float> {
    /// Training samples (row-major), or the training distance matrix for
    /// `Precomputed`.
    data: Array2<F>,
    metric: DistanceMetric,
    tree: Option<KdTree<F>>,
}

/// A neighbour query: the k nearest, or all within a reduced radius.
#[derive(Clone, Copy)]
enum Query<F> {
    Knn(usize),
    Radius(F),
}

/// Below this many queries the work is done on the calling thread.
const PARALLEL_QUERIES: usize = 32;

impl<F: Float> NeighborIndex<F> {
    /// Build an index over the rows of `x` with sklearn's leaf size (30).
    ///
    /// `n_neighbors` is only a hint for [`NeighborAlgorithm::Auto`].
    pub fn new<'a>(
        x: impl AsArray<'a, F, Ix2>,
        metric: DistanceMetric,
        algorithm: NeighborAlgorithm,
        n_neighbors: Option<usize>,
    ) -> Result<Self> {
        let x: ArrayView2<F> = x.into();
        Self::with_leaf_size(x, metric, algorithm, n_neighbors, DEFAULT_LEAF_SIZE)
    }

    /// [`new`](Self::new) with an explicit KD-tree leaf size.
    pub fn with_leaf_size<'a>(
        x: impl AsArray<'a, F, Ix2>,
        metric: DistanceMetric,
        algorithm: NeighborAlgorithm,
        n_neighbors: Option<usize>,
        leaf_size: usize,
    ) -> Result<Self> {
        let x: ArrayView2<F> = x.into();
        metric.validate()?;
        let metric = metric.canonical();
        let (n, d) = x.dim();
        if n == 0 {
            return Err(RustMlError::EmptyInput("training data is empty".into()));
        }
        if metric == DistanceMetric::Precomputed && n != d {
            return Err(RustMlError::ShapeMismatch(format!(
                "precomputed distances must be square, got {n}x{d}"
            )));
        }
        let tree = algorithm
            .use_tree(metric, n, d, n_neighbors)?
            .then(|| KdTree::build(x, leaf_size));
        Ok(Self {
            data: x.as_standard_layout().into_owned(),
            metric,
            tree,
        })
    }

    /// Number of training samples.
    pub fn n_samples(&self) -> usize {
        self.data.nrows()
    }

    /// Number of columns a query must have (features, or training samples
    /// for `Precomputed`).
    pub fn n_query_columns(&self) -> usize {
        self.data.ncols()
    }

    /// The (canonical) metric.
    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    /// Whether queries use the KD-tree.
    pub fn uses_tree(&self) -> bool {
        self.tree.is_some()
    }

    /// Training samples (or training distances for `Precomputed`).
    pub fn data(&self) -> &Array2<F> {
        &self.data
    }

    fn check_query(&self, q: ArrayView2<F>) -> Result<()> {
        if q.ncols() != self.data.ncols() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} columns, got {}",
                self.data.ncols(),
                q.ncols()
            )));
        }
        Ok(())
    }

    fn check_k(&self, k: usize) -> Result<()> {
        if k == 0 || k > self.n_samples() {
            return Err(RustMlError::InvalidParameter(format!(
                "n_neighbors must be in [1, {}], got {k}",
                self.n_samples()
            )));
        }
        Ok(())
    }

    /// The `k` nearest training samples of one query row, as (reduced
    /// distance, index) sorted ascending.
    fn knn_one(&self, q: &[F], k: usize) -> Vec<Cand<F>> {
        if let Some(tree) = &self.tree {
            return tree.knn(q, k, self.metric);
        }
        let mut all: Vec<Cand<F>> = if self.metric == DistanceMetric::Precomputed {
            q.iter().enumerate().map(|(j, &v)| Cand(v, j)).collect()
        } else {
            self.data
                .rows()
                .into_iter()
                .enumerate()
                .map(|(j, r)| Cand(self.metric.rdist(r.as_slice().expect("row-major"), q), j))
                .collect()
        };
        if k < all.len() {
            all.select_nth_unstable(k - 1);
            all.truncate(k);
        }
        all.sort_unstable();
        all
    }

    fn radius_one(&self, q: &[F], r_red: F) -> Vec<Cand<F>> {
        let mut out = Vec::new();
        if let Some(tree) = &self.tree {
            tree.radius(q, r_red, self.metric, &mut out);
        } else if self.metric == DistanceMetric::Precomputed {
            out.extend(
                q.iter()
                    .enumerate()
                    .filter(|(_, &v)| v <= r_red)
                    .map(|(j, &v)| Cand(v, j)),
            );
        } else {
            for (j, row) in self.data.rows().into_iter().enumerate() {
                let dr = self.metric.rdist(row.as_slice().expect("row-major"), q);
                if dr <= r_red {
                    out.push(Cand(dr, j));
                }
            }
        }
        out.sort_unstable();
        out
    }

    /// Neighbour candidates of every query row: the `Knn(k)` nearest or all
    /// within reduced radius `Radius(r)`, sorted by (distance, index).
    fn query_rows(&self, q: ArrayView2<F>, query: Query<F>) -> Vec<Vec<Cand<F>>> {
        if self.tree.is_none() && self.metric == DistanceMetric::Euclidean {
            return self.euclidean_brute_rows(q, query);
        }
        self.per_row(q, |_, r| match query {
            Query::Knn(k) => self.knn_one(r, k),
            Query::Radius(rr) => self.radius_one(r, rr),
        })
    }

    /// Euclidean brute force through a matrix product, as sklearn does,
    /// but exact: `|q|² + |x|² - 2 q·x` only pre-selects candidates (up to
    /// the k-th smallest, or the radius, plus a bound on its rounding
    /// error), which are then ranked by exact distances. Results equal
    /// [`knn_one`](Self::knn_one) / [`radius_one`](Self::radius_one).
    fn euclidean_brute_rows(&self, q: ArrayView2<F>, query: Query<F>) -> Vec<Vec<Cand<F>>> {
        const BLOCK: usize = 64;
        let q = q.as_standard_layout();
        let x = &self.data;
        let (n, d) = x.dim();
        let xn: Vec<F> = x
            .rows()
            .into_iter()
            .map(|r| r.iter().fold(F::zero(), |a, &v| a + v * v))
            .collect();
        let max_xn = xn.iter().copied().fold(F::zero(), F::max);
        let two = F::from_f64(2.0).unwrap();
        let slack = F::from_usize(2 * (d + 4)).unwrap() * F::epsilon();

        let block = |s: usize| -> Vec<Vec<Cand<F>>> {
            let e = (s + BLOCK).min(q.nrows());
            let g = q.slice(ndarray::s![s..e, ..]).dot(&x.t());
            (s..e)
                .map(|i| {
                    let qi = q.row(i).to_slice().expect("row-major");
                    let qn = qi.iter().fold(F::zero(), |a, &v| a + v * v);
                    let gi = g.row(i - s);
                    let approx = |j: usize| qn + xn[j] - two * gi[j];
                    let tol = slack * (qn + max_xn);
                    let threshold = match query {
                        Query::Knn(k) => {
                            let mut a: Vec<F> = (0..n).map(approx).collect();
                            let (_, kth, _) = a.select_nth_unstable_by(k - 1, crate::total_cmp);
                            *kth + tol
                        }
                        Query::Radius(rr) => rr + tol,
                    };
                    let mut c: Vec<Cand<F>> = (0..n)
                        .filter(|&j| approx(j) <= threshold)
                        .map(|j| {
                            Cand(
                                self.metric
                                    .rdist(x.row(j).to_slice().expect("row-major"), qi),
                                j,
                            )
                        })
                        .collect();
                    match query {
                        Query::Knn(k) if c.len() < k => self.knn_one(qi, k),
                        Query::Knn(k) => {
                            c.sort_unstable();
                            c.truncate(k);
                            c
                        }
                        Query::Radius(rr) => {
                            c.retain(|c| c.0 <= rr);
                            c.sort_unstable();
                            c
                        }
                    }
                })
                .collect()
        };
        let starts: Vec<usize> = (0..q.nrows()).step_by(BLOCK).collect();
        if q.nrows() >= PARALLEL_QUERIES {
            starts
                .into_par_iter()
                .map(block)
                .collect::<Vec<_>>()
                .into_iter()
                .flatten()
                .collect()
        } else {
            starts.into_iter().flat_map(block).collect()
        }
    }

    /// Run `f` on every query row (in parallel for many queries).
    fn per_row<T: Send>(
        &self,
        q: ArrayView2<F>,
        f: impl Fn(usize, &[F]) -> T + Sync + Send,
    ) -> Vec<T> {
        let q = q.as_standard_layout();
        let rows = |i: usize| f(i, q.row(i).to_slice().expect("row-major"));
        if q.nrows() >= PARALLEL_QUERIES {
            (0..q.nrows()).into_par_iter().map(rows).collect()
        } else {
            (0..q.nrows()).map(rows).collect()
        }
    }

    fn knn_rows(&self, rows: Vec<Vec<Cand<F>>>, k: usize) -> (Array2<F>, Array2<usize>) {
        let n = rows.len();
        let mut dist = Array2::zeros((n, k));
        let mut idx = Array2::zeros((n, k));
        for (i, row) in rows.into_iter().enumerate() {
            for (j, c) in row.into_iter().enumerate() {
                dist[[i, j]] = self.metric.rdist_to_dist(c.0);
                idx[[i, j]] = c.1;
            }
        }
        (dist, idx)
    }

    /// Distances and indices of the `k` nearest training samples of every
    /// row of `q`, each row sorted by distance (sklearn's `kneighbors`).
    pub fn kneighbors<'a>(
        &self,
        q: impl AsArray<'a, F, Ix2>,
        k: usize,
    ) -> Result<(Array2<F>, Array2<usize>)> {
        let q: ArrayView2<F> = q.into();
        self.check_query(q)?;
        self.check_k(k)?;
        let rows = self.query_rows(q, Query::Knn(k));
        Ok(self.knn_rows(rows, k))
    }

    /// The `k` nearest *other* training samples of every training sample
    /// (sklearn's `kneighbors()` without `X`: each sample is not its own
    /// neighbour).
    pub fn kneighbors_self(&self, k: usize) -> Result<(Array2<F>, Array2<usize>)> {
        if k == 0 || k >= self.n_samples() {
            return Err(RustMlError::InvalidParameter(format!(
                "n_neighbors must be in [1, {}), got {k}",
                self.n_samples()
            )));
        }
        let mut rows = self.query_rows(self.data.view(), Query::Knn(k + 1));
        for (i, c) in rows.iter_mut().enumerate() {
            // Drop the sample itself, or the farthest candidate when
            // duplicates with lower indices pushed it out.
            match c.iter().position(|c| c.1 == i) {
                Some(p) => {
                    c.remove(p);
                }
                None => {
                    c.pop();
                }
            }
        }
        Ok(self.knn_rows(rows, k))
    }

    fn radius_rows(rows: Vec<Vec<Cand<F>>>, metric: DistanceMetric) -> RadiusNeighbors<F> {
        rows.into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|c| (c.1, metric.rdist_to_dist(c.0)))
                    .collect()
            })
            .collect()
    }

    /// Training samples within distance `radius` (inclusive) of every row
    /// of `q` (sklearn's `radius_neighbors(sort_results=True)`).
    pub fn radius_neighbors<'a>(
        &self,
        q: impl AsArray<'a, F, Ix2>,
        radius: F,
    ) -> Result<RadiusNeighbors<F>> {
        let q: ArrayView2<F> = q.into();
        self.check_query(q)?;
        let r_red = self.metric.dist_to_rdist(radius);
        let rows = self.query_rows(q, Query::Radius(r_red));
        Ok(Self::radius_rows(rows, self.metric))
    }

    /// Other training samples within `radius` of every training sample.
    pub fn radius_neighbors_self(&self, radius: F) -> RadiusNeighbors<F> {
        let r_red = self.metric.dist_to_rdist(radius);
        let mut rows = self.query_rows(self.data.view(), Query::Radius(r_red));
        for (i, c) in rows.iter_mut().enumerate() {
            c.retain(|c| c.1 != i);
        }
        Self::radius_rows(rows, self.metric)
    }

    /// Distance between query row `q` and training sample `j`.
    pub fn distance_to(&self, q: ArrayView1<'_, F>, j: usize) -> F {
        if self.metric == DistanceMetric::Precomputed {
            return q[j];
        }
        let q = q.to_vec();
        let r = self.data.row(j);
        self.metric
            .rdist_to_dist(self.metric.rdist(r.as_slice().expect("row-major"), &q))
    }
}
