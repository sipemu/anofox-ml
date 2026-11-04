use anofox_ml_core::{
    FitUnsupervised, FitUnsupervisedWeighted, Float, Predict, Result, RustMlError,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::rngs::StdRng;
#[cfg(test)]
use rand::Rng;
use rand::SeedableRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[cfg(test)]
use crate::util::squared_euclidean;
use crate::util::{
    assign_nearest, kmeans_plusplus, row_norms_sq, sample_without_replacement, sq_dist_rows,
};

/// Centroid initialisation strategy for [`KMeans`] (sklearn's `init`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum KMeansInit {
    /// Greedy k-means++ (`2 + ⌊ln k⌋` candidates per step), sklearn's default.
    #[default]
    KMeansPlusPlus,
    /// `k` distinct training rows drawn at random.
    Random,
    /// Fixed initial centroids (`n_clusters × n_features`). Implies a single
    /// run (`n_init = 1`), as in sklearn.
    Array(Array2<f64>),
}

/// Parameters for K-Means clustering (unfitted state).
///
/// Groups data into `n_clusters` clusters using Lloyd's algorithm. Mirrors
/// `sklearn.cluster.KMeans(algorithm="lloyd")`:
///
/// - `n_init` independent runs (default `"auto"`: 1 for k-means++ or array
///   initialisation, 10 for random initialisation); the run with the lowest
///   inertia is kept;
/// - convergence when the labels stop changing, or when the total squared
///   centroid shift `Σ_k ‖c_k^new − c_k‖²` is `≤ tol · mean(var(X, axis=0))`;
/// - clusters that become empty are relocated to the points farthest from
///   their current centroid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KMeans {
    /// Number of clusters to form.
    pub n_clusters: usize,
    /// Maximum number of Lloyd's iterations per run.
    pub max_iter: usize,
    /// Relative convergence tolerance (scaled by the mean feature variance).
    pub tol: f64,
    /// Random seed for reproducible centroid initialization.
    pub seed: u64,
    /// Number of runs with different seeds; `None` = sklearn's `"auto"`.
    #[serde(default)]
    pub n_init: Option<usize>,
    /// Centroid initialisation strategy.
    #[serde(default)]
    pub init: KMeansInit,
}

impl KMeans {
    /// Create a new `KMeans` with the given number of clusters and default parameters.
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            max_iter: 300,
            tol: 1e-4,
            seed: 42,
            n_init: None,
            init: KMeansInit::KMeansPlusPlus,
        }
    }

    /// Set the maximum number of Lloyd's iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the relative convergence tolerance (multiplied by the mean
    /// per-feature variance of the training data, as in sklearn).
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the random seed for reproducible centroid initialization.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set the number of independent runs (best inertia is kept).
    pub fn with_n_init(mut self, n_init: usize) -> Self {
        self.n_init = Some(n_init.max(1));
        self
    }

    /// Set the centroid initialisation strategy.
    pub fn with_init(mut self, init: KMeansInit) -> Self {
        self.init = init;
        self
    }

    /// Effective number of runs (resolves sklearn's `n_init="auto"`).
    pub fn effective_n_init(&self) -> usize {
        match (&self.init, self.n_init) {
            (KMeansInit::Array(_), _) => 1,
            (_, Some(n)) => n.max(1),
            (KMeansInit::KMeansPlusPlus, None) => 1,
            (KMeansInit::Random, None) => 10,
        }
    }
}

impl Default for KMeans {
    fn default() -> Self {
        Self::new(3)
    }
}

/// Fitted K-Means model containing learned centroids and training metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedKMeans<F: Float> {
    centroids: Array2<F>,
    labels: Array1<F>,
    inertia: F,
    n_iter: usize,
}

impl<F: Float> FittedKMeans<F> {
    /// Returns a reference to the cluster centroids (n_clusters x n_features).
    pub fn centroids(&self) -> &Array2<F> {
        &self.centroids
    }

    /// Returns a reference to the cluster labels assigned to training data.
    pub fn labels(&self) -> &Array1<F> {
        &self.labels
    }

    /// Returns the inertia (sum of squared distances to closest centroid).
    pub fn inertia(&self) -> F {
        self.inertia
    }

    /// Returns the number of iterations performed by the best run.
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }
}

/// Result of a single Lloyd run.
struct LloydRun<F> {
    centroids: Array2<F>,
    labels: Vec<usize>,
    inertia: F,
    n_iter: usize,
}

/// Rows per block of the parallel reductions. The block partials are summed
/// in block order, so results do not depend on the rayon thread count
/// (rayon's `reduce` combines in an order that does).
const BLOCK: usize = 1024;

/// Weighted per-cluster coordinate sums and weight totals, in parallel over
/// row blocks.
fn cluster_sums<F: Float>(
    x: ArrayView2<F>,
    labels: &[usize],
    sample_weight: Option<ArrayView1<F>>,
    k: usize,
) -> (Array2<F>, Vec<F>) {
    let n = x.nrows();
    let d = x.ncols();
    let parts: Vec<(Array2<F>, Vec<F>)> = (0..n.div_ceil(BLOCK))
        .into_par_iter()
        .map(|b| {
            let mut sums = Array2::<F>::zeros((k, d));
            let mut wsum = vec![F::zero(); k];
            for i in b * BLOCK..((b + 1) * BLOCK).min(n) {
                let w = sample_weight.map_or(F::one(), |sw| sw[i]);
                if w == F::zero() {
                    continue;
                }
                let c = labels[i];
                wsum[c] += w;
                let mut row = sums.row_mut(c);
                for (s, &v) in row.iter_mut().zip(x.row(i).iter()) {
                    *s += v * w;
                }
            }
            (sums, wsum)
        })
        .collect();
    parts.into_iter().fold(
        (Array2::<F>::zeros((k, d)), vec![F::zero(); k]),
        |(mut sa, mut wa), (sb, wb)| {
            sa += &sb;
            for (a, b) in wa.iter_mut().zip(wb) {
                *a += b;
            }
            (sa, wa)
        },
    )
}

/// Relocate empty clusters (zero total weight) to the points farthest from
/// their currently assigned centroid — sklearn's
/// `_relocate_empty_clusters_dense`. Operates on the un-normalised sums.
fn relocate_empty_clusters<F: Float>(
    x: ArrayView2<F>,
    labels: &[usize],
    sample_weight: Option<ArrayView1<F>>,
    centers_old: &Array2<F>,
    sums: &mut Array2<F>,
    wsum: &mut [F],
) {
    let empty: Vec<usize> = (0..wsum.len()).filter(|&c| wsum[c] == F::zero()).collect();
    if empty.is_empty() {
        return;
    }
    let n = x.nrows();
    let dist: Vec<F> = (0..n)
        .into_par_iter()
        .with_min_len(256)
        .map(|i| sq_dist_rows(x.row(i), centers_old.row(labels[i])))
        .collect();
    let mut order: Vec<usize> = (0..n).collect();
    // Descending distance; stable on index for determinism.
    order.sort_by(|&a, &b| {
        dist[b]
            .partial_cmp(&dist[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (&new_c, &far) in empty.iter().zip(order.iter()) {
        let w = sample_weight.map_or(F::one(), |sw| sw[far]);
        let old_c = labels[far];
        for j in 0..x.ncols() {
            let v = x[[far, j]] * w;
            sums[[old_c, j]] -= v;
            sums[[new_c, j]] = v;
        }
        wsum[new_c] = w;
        wsum[old_c] -= w;
    }
}

/// One Lloyd run from the given initial centroids (sklearn's
/// `_kmeans_single_lloyd`).
fn lloyd<F: Float>(
    x: ArrayView2<F>,
    x_sq: &Array1<F>,
    sample_weight: Option<ArrayView1<F>>,
    mut centers: Array2<F>,
    max_iter: usize,
    tol: F,
) -> LloydRun<F> {
    let n = x.nrows();
    let k = centers.nrows();
    let mut labels_old = vec![usize::MAX; n];
    let mut labels = labels_old.clone();
    let mut strict_convergence = false;
    let mut n_iter = 0;

    for it in 0..max_iter {
        n_iter = it + 1;
        labels = assign_nearest(x, x_sq, &centers).0;
        let (mut sums, mut wsum) = cluster_sums(x, &labels, sample_weight, k);
        relocate_empty_clusters(x, &labels, sample_weight, &centers, &mut sums, &mut wsum);

        let mut new_centers = sums;
        for (c, &wc) in wsum.iter().enumerate() {
            if wc > F::zero() {
                let inv = F::one() / wc;
                new_centers.row_mut(c).mapv_inplace(|v| v * inv);
            } else {
                new_centers.row_mut(c).assign(&centers.row(c));
            }
        }
        let mut shift_tot = F::zero();
        for c in 0..k {
            shift_tot += sq_dist_rows(centers.row(c), new_centers.row(c));
        }
        centers = new_centers;

        if labels == labels_old {
            strict_convergence = true;
            break;
        }
        if shift_tot <= tol {
            break;
        }
        labels_old.clone_from(&labels);
    }
    if !strict_convergence {
        // Labels consistent with the final centroids.
        labels = assign_nearest(x, x_sq, &centers).0;
    }
    let inertia = inertia(x, &labels, &centers, sample_weight);
    LloydRun {
        centroids: centers,
        labels,
        inertia,
        n_iter,
    }
}

/// Exact (weighted) inertia `Σ w_i ‖x_i − c_{z_i}‖²`.
fn inertia<F: Float>(
    x: ArrayView2<F>,
    labels: &[usize],
    centers: &Array2<F>,
    sample_weight: Option<ArrayView1<F>>,
) -> F {
    let n = x.nrows();
    let parts: Vec<F> = (0..n.div_ceil(BLOCK))
        .into_par_iter()
        .map(|b| {
            (b * BLOCK..((b + 1) * BLOCK).min(n)).fold(F::zero(), |acc, i| {
                let w = sample_weight.map_or(F::one(), |sw| sw[i]);
                acc + sq_dist_rows(x.row(i), centers.row(labels[i])) * w
            })
        })
        .collect();
    parts.into_iter().fold(F::zero(), |a, b| a + b)
}

/// sklearn's `_tolerance`: `tol · mean(var(X, axis=0))` (unweighted, ddof 0).
fn scaled_tolerance<F: Float>(x: ArrayView2<F>, tol: f64) -> F {
    if tol == 0.0 {
        return F::zero();
    }
    let n = F::from_usize(x.nrows()).unwrap();
    let d = x.ncols();
    let mut total = F::zero();
    for j in 0..d {
        let col = x.column(j);
        let mean = col.iter().fold(F::zero(), |a, &v| a + v) / n;
        let var = col
            .iter()
            .fold(F::zero(), |a, &v| a + (v - mean) * (v - mean))
            / n;
        total += var;
    }
    total / F::from_usize(d.max(1)).unwrap() * F::from_f64(tol).unwrap()
}

impl KMeans {
    /// Shared fit used by both unweighted and weighted fits.
    fn fit_inner<F: Float>(
        &self,
        x: ArrayView2<F>,
        sample_weight: Option<ArrayView1<F>>,
    ) -> Result<FittedKMeans<F>> {
        let x_std = crate::standard_layout(x);
        let x = x_std.view();
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples == 0 {
            return Err(RustMlError::EmptyInput("input array is empty".into()));
        }
        if self.n_clusters == 0 {
            return Err(RustMlError::InvalidParameter(
                "n_clusters must be at least 1".into(),
            ));
        }
        if self.n_clusters > n_samples {
            return Err(RustMlError::InvalidParameter(format!(
                "n_clusters ({}) must not exceed n_samples ({})",
                self.n_clusters, n_samples
            )));
        }
        if let Some(w) = sample_weight {
            if w.len() != n_samples {
                return Err(RustMlError::ShapeMismatch(format!(
                    "sample_weight length {} does not match n_samples {}",
                    w.len(),
                    n_samples
                )));
            }
            for &v in w.iter() {
                if v < F::zero() {
                    return Err(RustMlError::InvalidParameter(
                        "sample_weight must be non-negative".into(),
                    ));
                }
            }
        }
        if let KMeansInit::Array(c) = &self.init {
            if c.dim() != (self.n_clusters, n_features) {
                return Err(RustMlError::ShapeMismatch(format!(
                    "init array has shape {:?}, expected ({}, {})",
                    c.dim(),
                    self.n_clusters,
                    n_features
                )));
            }
        }

        let x_sq = row_norms_sq(x);
        let tol = scaled_tolerance(x, self.tol);
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut best: Option<LloydRun<F>> = None;

        for _ in 0..self.effective_n_init() {
            let init = match &self.init {
                KMeansInit::KMeansPlusPlus => {
                    kmeans_plusplus(x.view(), self.n_clusters, &mut rng, sample_weight).0
                }
                KMeansInit::Random => {
                    let idx = sample_without_replacement(
                        n_samples,
                        self.n_clusters,
                        &mut rng,
                        sample_weight,
                    );
                    let mut c = Array2::<F>::zeros((self.n_clusters, n_features));
                    for (r, &i) in idx.iter().enumerate() {
                        c.row_mut(r).assign(&x.row(i));
                    }
                    c
                }
                KMeansInit::Array(c) => c.mapv(|v| F::from_f64(v).unwrap()),
            };
            let run = lloyd(x, &x_sq, sample_weight, init, self.max_iter, tol);
            if best.as_ref().is_none_or(|b| run.inertia < b.inertia) {
                best = Some(run);
            }
        }
        let best = best.unwrap();
        Ok(FittedKMeans {
            centroids: best.centroids,
            labels: best
                .labels
                .iter()
                .map(|&l| F::from_usize(l).unwrap())
                .collect(),
            inertia: best.inertia,
            n_iter: best.n_iter,
        })
    }
}

impl<F: Float> FitUnsupervised<F> for KMeans {
    type Fitted = FittedKMeans<F>;

    fn fit_view(&self, x: ArrayView2<F>) -> Result<Self::Fitted> {
        self.fit_inner(x, None)
    }
}

impl<F: Float> FitUnsupervisedWeighted<F> for KMeans {
    type Fitted = FittedKMeans<F>;

    fn fit_unsupervised_weighted_view(
        &self,
        x: ArrayView2<F>,
        sample_weight: Option<ArrayView1<F>>,
    ) -> Result<Self::Fitted> {
        self.fit_inner(x, sample_weight)
    }
}

impl<F: Float> Predict<F> for FittedKMeans<F> {
    fn predict_view(&self, x: ArrayView2<F>) -> Result<Array1<F>> {
        if x.ncols() != self.centroids.ncols() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.centroids.ncols(),
                x.ncols()
            )));
        }
        let x_std = crate::standard_layout(x);
        let x = x_std.view();
        let (labels, _) = assign_nearest(x, &row_norms_sq(x), &self.centroids);
        Ok(labels.iter().map(|&l| F::from_usize(l).unwrap()).collect())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn fit_does_not_depend_on_the_thread_count() {
        // Large enough for several reduction blocks (#108).
        let x = Array2::from_shape_fn((5000, 4), |(i, j)| {
            ((i * 7919 + j * 104729) % 1000) as f64 / 100.0
        });
        let fit = |threads| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| KMeans::new(5).with_seed(3).fit(&x).unwrap())
        };
        let (a, b) = (fit(1), fit(8));
        assert_eq!(a.centroids(), b.centroids());
        assert_eq!(a.inertia().to_bits(), b.inertia().to_bits());
        assert_eq!(a.labels(), b.labels());
    }

    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array2, Axis};

    /// Build a dataset with 3 well-separated clusters centred at
    /// (0, 0), (10, 10), and (20, 0), each with 30 points.
    fn make_blobs() -> Array2<f64> {
        let mut rng = StdRng::seed_from_u64(123);
        let centers = [(0.0, 0.0), (10.0, 10.0), (20.0, 0.0)];
        let mut data = Array2::<f64>::zeros((90, 2));
        for (c, &(cx, cy)) in centers.iter().enumerate() {
            for i in 0..30 {
                let row = c * 30 + i;
                data[[row, 0]] = cx + (rng.gen_range(-1.0..1.0));
                data[[row, 1]] = cy + (rng.gen_range(-1.0..1.0));
            }
        }
        data
    }

    #[test]
    fn test_finds_three_clusters() {
        let x = make_blobs();
        let kmeans = KMeans::new(3);
        let fitted = FitUnsupervised::<f64>::fit(&kmeans, &x).unwrap();

        // Each of the 3 original groups should have a single unique label.
        let labels = fitted.labels();
        assert_eq!(labels.len(), 90);

        // Cluster 0 points (rows 0..30) should all share one label.
        let label_a = labels[0] as usize;
        for i in 1..30 {
            assert_eq!(labels[i] as usize, label_a, "row {i} has wrong label");
        }

        // Cluster 1 points (rows 30..60) should all share one label.
        let label_b = labels[30] as usize;
        for i in 31..60 {
            assert_eq!(labels[i] as usize, label_b, "row {i} has wrong label");
        }

        // Cluster 2 points (rows 60..90) should all share one label.
        let label_c = labels[60] as usize;
        for i in 61..90 {
            assert_eq!(labels[i] as usize, label_c, "row {i} has wrong label");
        }

        // The three labels should be distinct.
        assert_ne!(label_a, label_b);
        assert_ne!(label_a, label_c);
        assert_ne!(label_b, label_c);
    }

    #[test]
    fn test_predict_assigns_correct_clusters() {
        let x = make_blobs();
        let kmeans = KMeans::new(3);
        let fitted = FitUnsupervised::<f64>::fit(&kmeans, &x).unwrap();

        // New points near each cluster centre should get the same label as
        // the training points from that cluster.
        let new_points = array![[0.5, 0.5], [10.2, 9.8], [19.5, 0.3]];
        let predicted = fitted.predict(&new_points).unwrap();

        // The predicted label for each new point should match the label of
        // a training point from the same original cluster.
        assert_abs_diff_eq!(predicted[0], fitted.labels()[0], epsilon = 1e-10);
        assert_abs_diff_eq!(predicted[1], fitted.labels()[30], epsilon = 1e-10);
        assert_abs_diff_eq!(predicted[2], fitted.labels()[60], epsilon = 1e-10);
    }

    #[test]
    fn test_convergence_before_max_iter() {
        let x = make_blobs();
        let kmeans = KMeans {
            n_clusters: 3,
            max_iter: 300,
            tol: 1e-4,
            seed: 42,
            ..KMeans::new(3)
        };
        let fitted = FitUnsupervised::<f64>::fit(&kmeans, &x).unwrap();

        // Well-separated data should converge well before 300 iterations.
        assert!(
            fitted.n_iter() < 300,
            "expected convergence before max_iter, got n_iter={}",
            fitted.n_iter()
        );
    }

    #[test]
    fn test_inertia_lower_than_random() {
        let x = make_blobs();
        let kmeans = KMeans::new(3);
        let fitted = FitUnsupervised::<f64>::fit(&kmeans, &x).unwrap();

        // Compute inertia of a "random" assignment: assign everything to a
        // single centroid at the data mean.
        let n = x.nrows() as f64;
        let mean = x.sum_axis(Axis(0)) / n;
        let random_inertia: f64 = x
            .rows()
            .into_iter()
            .map(|row| squared_euclidean(row.as_slice().unwrap(), mean.as_slice().unwrap()))
            .sum();

        assert!(
            fitted.inertia() < random_inertia,
            "k-means inertia ({}) should be less than single-centroid inertia ({})",
            fitted.inertia(),
            random_inertia
        );
    }

    #[test]
    fn test_error_n_clusters_exceeds_n_samples() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let kmeans = KMeans::new(5);
        let result = FitUnsupervised::<f64>::fit(&kmeans, &x);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("n_clusters"),
            "error should mention n_clusters: {err}"
        );
    }

    #[test]
    fn test_predict_shape_mismatch() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let kmeans = KMeans::new(2);
        let fitted = FitUnsupervised::<f64>::fit(&kmeans, &x).unwrap();

        let bad_input = array![[1.0, 2.0, 3.0]];
        let result = fitted.predict(&bad_input);
        assert!(result.is_err());
    }

    #[test]
    fn test_centroids_accessor() {
        let x = array![[0.0, 0.0], [1.0, 0.0], [10.0, 10.0], [11.0, 10.0]];
        let kmeans = KMeans::new(2);
        let fitted = FitUnsupervised::<f64>::fit(&kmeans, &x).unwrap();

        let centroids = fitted.centroids();
        assert_eq!(centroids.nrows(), 2);
        assert_eq!(centroids.ncols(), 2);
    }

    #[test]
    fn test_weighted_equiv_to_duplication() {
        // Duplicating point i k times should be equivalent to weighting it
        // by k. We verify on a small dataset that fitted centroids match.
        let base = array![
            [0.0_f64, 0.0],
            [0.2, 0.1],
            [10.0, 10.0],
            [10.1, 9.9],
            [10.2, 10.1],
        ];
        // Weight: blob B has 3 points (rows 2..5), but assign them weights
        // (1, 1, 3) so total weight 5; compare against duplicating row 4 3×.
        let weights = array![1.0_f64, 1.0, 1.0, 1.0, 3.0];

        let mut dup = Vec::new();
        for i in 0..base.nrows() {
            let times = weights[i] as usize;
            for _ in 0..times {
                for j in 0..base.ncols() {
                    dup.push(base[[i, j]]);
                }
            }
        }
        let dup = Array2::from_shape_vec((dup.len() / 2, 2), dup).unwrap();

        let km = KMeans::new(2).with_seed(7);
        let f_w = km.fit_unsupervised_weighted(&base, Some(&weights)).unwrap();
        let f_d = FitUnsupervised::<f64>::fit(&km, &dup).unwrap();

        // Centroids may permute label order. Match by nearest centroid.
        let cw = f_w.centroids();
        let cd = f_d.centroids();
        for i in 0..2 {
            let mut best = f64::INFINITY;
            for j in 0..2 {
                let d =
                    squared_euclidean(cw.row(i).as_slice().unwrap(), cd.row(j).as_slice().unwrap());
                if d < best {
                    best = d;
                }
            }
            assert!(
                best < 1e-9,
                "weighted centroid {} not close to any duplication centroid (best d² = {})",
                i,
                best
            );
        }
    }

    #[test]
    fn test_weighted_unweighted_equals_uniform() {
        let x = make_blobs();
        let km = KMeans::new(3).with_seed(42);
        let unweighted = FitUnsupervised::<f64>::fit(&km, &x).unwrap();
        let uniform = Array1::<f64>::from_elem(x.nrows(), 1.0);
        let weighted = km.fit_unsupervised_weighted(&x, Some(&uniform)).unwrap();
        for (a, b) in unweighted.labels().iter().zip(weighted.labels().iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-15);
        }
        assert_abs_diff_eq!(unweighted.inertia(), weighted.inertia(), epsilon = 1e-9);
    }

    #[test]
    fn test_reproducibility() {
        let x = make_blobs();
        let kmeans = KMeans::new(3);
        let fitted1 = FitUnsupervised::<f64>::fit(&kmeans, &x).unwrap();
        let fitted2 = FitUnsupervised::<f64>::fit(&kmeans, &x).unwrap();

        // Same seed should produce identical results.
        for (a, b) in fitted1.labels().iter().zip(fitted2.labels().iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-15);
        }
        assert_abs_diff_eq!(fitted1.inertia(), fitted2.inertia(), epsilon = 1e-15);
    }

    mod prop_tests {
        use super::*;
        use proptest::prelude::*;

        /// Generate well-separated cluster data for k clusters.
        fn make_cluster_data(k: usize, seed: u64) -> Array2<f64> {
            let mut rng = StdRng::seed_from_u64(seed);
            let points_per_cluster = 15;
            let n_samples = k * points_per_cluster;
            let mut data = Array2::<f64>::zeros((n_samples, 2));

            for c in 0..k {
                let cx = (c as f64) * 100.0;
                let cy = (c as f64) * 100.0;
                for i in 0..points_per_cluster {
                    let row = c * points_per_cluster + i;
                    data[[row, 0]] = cx + <StdRng as rand::Rng>::gen_range(&mut rng, -1.0..1.0);
                    data[[row, 1]] = cy + <StdRng as rand::Rng>::gen_range(&mut rng, -1.0..1.0);
                }
            }
            data
        }

        proptest! {
            #[test]
            fn kmeans_labels_in_range(k in 2..5usize, seed in 0u64..1000) {
                let x = make_cluster_data(k, seed);
                let kmeans = KMeans::new(k).with_seed(seed);
                let fitted = FitUnsupervised::<f64>::fit(&kmeans, &x).unwrap();

                let labels = fitted.labels();
                for (i, &label) in labels.iter().enumerate() {
                    let l = label as usize;
                    prop_assert!(l < k,
                        "label {} at index {} is out of range [0, {})", l, i, k);
                }

                // Also check predict on the same data
                let predicted = fitted.predict(&x).unwrap();
                for (i, &label) in predicted.iter().enumerate() {
                    let l = label as usize;
                    prop_assert!(l < k,
                        "predicted label {} at index {} is out of range [0, {})", l, i, k);
                }
            }

            #[test]
            fn kmeans_deterministic(seed in 0u64..1000) {
                let x = make_cluster_data(3, seed);
                let kmeans = KMeans::new(3).with_seed(seed);

                let fitted1 = FitUnsupervised::<f64>::fit(&kmeans, &x).unwrap();
                let fitted2 = FitUnsupervised::<f64>::fit(&kmeans, &x).unwrap();

                for (a, b) in fitted1.labels().iter().zip(fitted2.labels().iter()) {
                    prop_assert!((a - b).abs() < 1e-15,
                        "non-deterministic labels: {} vs {}", a, b);
                }
                prop_assert!((fitted1.inertia() - fitted2.inertia()).abs() < 1e-15,
                    "non-deterministic inertia: {} vs {}", fitted1.inertia(), fitted2.inertia());
            }
        }
    }
}

impl<F: anofox_ml_core::Float> anofox_ml_core::ClassifierScore<F> for FittedKMeans<F> {}
