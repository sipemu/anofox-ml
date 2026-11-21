//! Spectral co-clustering (sklearn's `SpectralCoclustering`, Dhillon
//! 2001): rows and columns of a non-negative matrix clustered together.
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
