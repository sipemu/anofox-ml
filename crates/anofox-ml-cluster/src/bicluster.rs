//! Spectral co-clustering (sklearn's `SpectralCoclustering`, Dhillon
//! 2001): rows and columns of a non-negative matrix clustered together;
//! spectral biclustering (sklearn's `SpectralBiclustering`, Kluger et al.
//! 2003): a checkerboard of row and column clusters.
//!
//! sklearn's `_fit`: the scale-normalized matrix `D_r^{-1/2} A D_c^{-1/2}`,
//! its `1 + ceil(log2 k)` leading singular vectors by `randomized_svd`
//! (the first discarded), stacked rows and columns scaled back, then
//! `KMeans(k, k-means++, n_init)`. Both the SVD and KMeans draw from
//! their own numpy `RandomState(seed)`, as sklearn does with an int
//! `random_state`, so the labels are sklearn's.

use anofox_ml_core::numpy_random::RandomState;
use anofox_ml_core::{FitUnsupervised, Result, RustMlError};
use ndarray::{concatenate, Array1, Array2, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

/// sklearn's `SpectralCoclustering` (`svd_method="randomized"`, KMeans).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SpectralCoclustering {
    pub n_clusters: usize,
    /// `randomized_svd`'s `n_oversamples` (sklearn's `n_svd_vecs`; `None`:
    /// 10).
    pub n_svd_vecs: Option<usize>,
    pub n_init: usize,
    pub seed: u32,
}

impl Default for SpectralCoclustering {
    fn default() -> Self {
        Self {
            n_clusters: 3,
            n_svd_vecs: None,
            n_init: 10,
            seed: 0,
        }
    }
}

/// A fitted [`SpectralCoclustering`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedSpectralCoclustering {
    /// The bicluster of each row (sklearn's `row_labels_`).
    pub row_labels: Vec<usize>,
    /// The bicluster of each column (sklearn's `column_labels_`).
    pub column_labels: Vec<usize>,
    pub n_clusters: usize,
}

/// sklearn's `_scale_normalize` (after `make_nonnegative`): the scaled
/// matrix and the row / column scalings (0 for an all-zero row / column).
fn scale_normalize(x: ArrayView2<f64>) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let min = x.iter().copied().fold(f64::INFINITY, f64::min);
    let a = if min < 0.0 { &x - min } else { x.to_owned() };
    let inv_sqrt = |s: f64| {
        let v = 1.0 / s.sqrt();
        if v.is_nan() {
            0.0
        } else {
            v
        }
    };
    let row = a.sum_axis(Axis(1)).mapv(inv_sqrt);
    let col = a.sum_axis(Axis(0)).mapv(inv_sqrt);
    let an = &(&a * &row.view().insert_axis(Axis(1))) * &col;
    (an, row, col)
}

impl FitUnsupervised<f64> for SpectralCoclustering {
    type Fitted = FittedSpectralCoclustering;

    fn fit_view(&self, x: ArrayView2<f64>) -> Result<FittedSpectralCoclustering> {
        let (n, p) = x.dim();
        let k = self.n_clusters;
        if k == 0 || k > n {
            return Err(RustMlError::InvalidParameter(format!(
                "n_clusters should be <= n_samples={n}. Got {k} instead."
            )));
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(RustMlError::InvalidParameter("x must be finite".into()));
        }
        let (an, row, col) = scale_normalize(x);
        let n_sv = 1 + (k as f64).log2().ceil() as usize;
        let mut rng = RandomState::new(self.seed);
        let (u, _, vt) = anofox_ml_preprocessing::randomized_svd::randomized_svd_oversampled(
            &an,
            n_sv,
            self.n_svd_vecs.unwrap_or(10),
            &mut rng,
        )?;
        let u = u.slice(ndarray::s![.., 1..]).to_owned();
        let v = vt.slice(ndarray::s![1.., ..]).t().to_owned();
        let z = concatenate(
            Axis(0),
            &[
                (&u * &row.view().insert_axis(Axis(1))).view(),
                (&v * &col.view().insert_axis(Axis(1))).view(),
            ],
        )
        .map_err(|e| RustMlError::InvalidParameter(e.to_string()))?;
        let (labels, _, _) =
            crate::bisecting_kmeans::sklearn_kmeans(z.view(), k, self.n_init, self.seed)?;
        Ok(FittedSpectralCoclustering {
            row_labels: labels[..n].to_vec(),
            column_labels: labels[n..n + p].to_vec(),
            n_clusters: k,
        })
    }
}

/// The normalization of [`SpectralBiclustering`] (sklearn's `method`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum BiclusterMethod {
    /// Repeated scale normalization until rows and columns each sum to a
    /// constant (sklearn's `_bistochastic_normalize`).
    #[default]
    Bistochastic,
    /// One scale normalization `D_r^{-1/2} A D_c^{-1/2}`.
    Scale,
    /// Kluger's log interactions: `log A` minus row and column means plus
    /// the grand mean.
    Log,
}

/// sklearn's `SpectralBiclustering` (`svd_method="randomized"`, KMeans):
/// the `n_components` leading singular vectors (after the trivial one,
/// unless `Log`), the `n_best` of them best fit by a piecewise-constant
/// vector (a 1-D KMeans), then KMeans of the data projected onto them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SpectralBiclustering {
    pub n_row_clusters: usize,
    pub n_column_clusters: usize,
    pub method: BiclusterMethod,
    pub n_components: usize,
    pub n_best: usize,
    /// `randomized_svd`'s `n_oversamples` (sklearn's `n_svd_vecs`; `None`:
    /// 10).
    pub n_svd_vecs: Option<usize>,
    pub n_init: usize,
    pub seed: u32,
}

impl Default for SpectralBiclustering {
    fn default() -> Self {
        Self {
            n_row_clusters: 3,
            n_column_clusters: 3,
            method: BiclusterMethod::Bistochastic,
            n_components: 6,
            n_best: 3,
            n_svd_vecs: None,
            n_init: 10,
            seed: 0,
        }
    }
}

/// A fitted [`SpectralBiclustering`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedSpectralBiclustering {
    /// The row cluster of each row (sklearn's `row_labels_`).
    pub row_labels: Vec<usize>,
    /// The column cluster of each column (sklearn's `column_labels_`).
    pub column_labels: Vec<usize>,
    pub n_row_clusters: usize,
    pub n_column_clusters: usize,
}

/// sklearn's `_bistochastic_normalize` (`max_iter=1000`, `tol=1e-5`).
fn bistochastic_normalize(x: ArrayView2<f64>) -> Array2<f64> {
    let min = x.iter().copied().fold(f64::INFINITY, f64::min);
    let mut scaled = if min < 0.0 { &x - min } else { x.to_owned() };
    for _ in 0..1000 {
        let next = scale_normalize(scaled.view()).0;
        let dist = (&scaled - &next).mapv(|v| v * v).sum().sqrt();
        scaled = next;
        if dist < 1e-5 {
            break;
        }
    }
    scaled
}

/// sklearn's `_log_normalize`: shifted to a minimum of at least 1, then
/// the double-centred log.
fn log_normalize(x: ArrayView2<f64>) -> Array2<f64> {
    let min = x.iter().copied().fold(f64::INFINITY, f64::min);
    let l = if min < 1.0 {
        x.mapv(|v| (v + (1.0 - min)).ln())
    } else {
        x.mapv(f64::ln)
    };
    let row = l.mean_axis(Axis(1)).expect("non-empty");
    let col = l.mean_axis(Axis(0)).expect("non-empty");
    let avg = l.mean().expect("non-empty");
    &(&l - &row.insert_axis(Axis(1))) - &col + avg
}

impl SpectralBiclustering {
    /// sklearn's `_fit_best_piecewise`: the `n_best` rows of `vectors`
    /// closest (Euclidean) to their 1-D KMeans approximation.
    fn best_piecewise(&self, vectors: &Array2<f64>, k: usize) -> Result<Array2<f64>> {
        let mut dists = Vec::with_capacity(vectors.nrows());
        for v in vectors.rows() {
            let col = v.to_owned().insert_axis(Axis(1));
            let (labels, centres, _) =
                crate::bisecting_kmeans::sklearn_kmeans(col.view(), k, self.n_init, self.seed)?;
            let d: f64 = v
                .iter()
                .zip(&labels)
                .map(|(a, &l)| (a - centres[[l, 0]]).powi(2))
                .sum();
            dists.push(d.sqrt());
        }
        let mut order: Vec<usize> = (0..dists.len()).collect();
        order.sort_by(|&a, &b| dists[a].total_cmp(&dists[b]));
        Ok(vectors.select(Axis(0), &order[..self.n_best]))
    }
}

impl FitUnsupervised<f64> for SpectralBiclustering {
    type Fitted = FittedSpectralBiclustering;

    fn fit_view(&self, x: ArrayView2<f64>) -> Result<FittedSpectralBiclustering> {
        let n = x.nrows();
        let (kr, kc) = (self.n_row_clusters, self.n_column_clusters);
        if kr == 0 || kc == 0 || kr > n || kc > n {
            return Err(RustMlError::InvalidParameter(format!(
                "n_clusters should be in 1..=n_samples={n}. Got ({kr}, {kc}) instead."
            )));
        }
        if self.n_best == 0 || self.n_best > self.n_components {
            return Err(RustMlError::InvalidParameter(format!(
                "n_best={} must be in 1..=n_components={}.",
                self.n_best, self.n_components
            )));
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(RustMlError::InvalidParameter("x must be finite".into()));
        }
        let (normalized, n_discard) = match self.method {
            BiclusterMethod::Bistochastic => (bistochastic_normalize(x), 1),
            BiclusterMethod::Scale => (scale_normalize(x).0, 1),
            BiclusterMethod::Log => (log_normalize(x), 0),
        };
        let mut rng = RandomState::new(self.seed);
        let (u, _, vt) = anofox_ml_preprocessing::randomized_svd::randomized_svd_oversampled(
            &normalized,
            self.n_components + n_discard,
            self.n_svd_vecs.unwrap_or(10),
            &mut rng,
        )?;
        let ut = u.slice(ndarray::s![.., n_discard..]).t().to_owned();
        let vt = vt.slice(ndarray::s![n_discard.., ..]).to_owned();
        let best_ut = self.best_piecewise(&ut, kr)?;
        let best_vt = self.best_piecewise(&vt, kc)?;
        let kmeans = |data: Array2<f64>, k| {
            crate::bisecting_kmeans::sklearn_kmeans(data.view(), k, self.n_init, self.seed)
                .map(|r| r.0)
        };
        Ok(FittedSpectralBiclustering {
            row_labels: kmeans(x.dot(&best_vt.t()), kr)?,
            column_labels: kmeans(x.t().dot(&best_ut.t()), kc)?,
            n_row_clusters: kr,
            n_column_clusters: kc,
        })
    }
}
