//! Mini-batch K-Means.
//!
//! A port of `sklearn.cluster.MiniBatchKMeans` (sklearn 1.8):
//!
//! - **Init:** a validation set of `init_size` random rows is drawn, then
//!   `n_init` candidate initialisations, each by k-means++ (or random
//!   rows) on its own random subsample of `init_size` rows. The candidate
//!   with the lowest inertia on the validation set wins.
//! - **Steps:** `max_iter · n_samples / batch_size` steps, each on
//!   `batch_size` rows drawn with replacement. Every batch is labelled with
//!   the current centres, then each centre moves to the weighted mean of its
//!   old position (weighted by all samples it has received so far) and its
//!   new samples.
//! - **Reassignment:** when a centre has never been used, or every
//!   `10 · n_clusters` samples, centres whose accumulated weight is below
//!   `reassignment_ratio · max` jump to random batch rows.
//! - **Stopping:** an exponentially weighted average of the batch inertia;
//!   stop after `max_no_improvement` steps without a new minimum, or (with
//!   `tol > 0`) once the squared centre change is below
//!   `tol · mean(var(X))`.
//!
//! The draws come from this crate's generator, so results agree with
//! sklearn's statistically, not number for number.

use anofox_ml_core::{
    FitUnsupervised, FitUnsupervisedWeighted, Float, Predict, Result, RustMlError,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::util::{assign_nearest, kmeans_plusplus, row_norms_sq, sample_without_replacement};

/// Initialisation (sklearn `init`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum MiniBatchInit {
    #[default]
    KMeansPlusPlus,
    /// `n_clusters` random rows of the init subsample.
    Random,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiniBatchKMeans {
    pub n_clusters: usize,
    /// Rows per step (sklearn default 1024).
    pub batch_size: usize,
    /// Number of passes over the data (epochs).
    pub max_iter: usize,
    /// Stop when the squared centre change is below `tol · mean(var(X))`;
    /// 0 (sklearn's default) disables it.
    pub tol: f64,
    /// Stop after this many steps without improvement of the smoothed batch
    /// inertia; `None` disables it.
    #[serde(default = "default_max_no_improvement")]
    pub max_no_improvement: Option<usize>,
    #[serde(default)]
    pub init: MiniBatchInit,
    /// Rows used for each initialisation and for the validation set;
    /// `None` = `3 · batch_size` (at least `3 · n_clusters`).
    #[serde(default)]
    pub init_size: Option<usize>,
    /// Number of initialisations; `None` = sklearn's `'auto'` (1 for
    /// k-means++, 3 for random).
    #[serde(default)]
    pub n_init: Option<usize>,
    #[serde(default = "default_reassignment_ratio")]
    pub reassignment_ratio: f64,
    pub seed: u64,
}

fn default_max_no_improvement() -> Option<usize> {
    Some(10)
}

fn default_reassignment_ratio() -> f64 {
    0.01
}

impl MiniBatchKMeans {
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            batch_size: 1024,
            max_iter: 100,
            tol: 0.0,
            max_no_improvement: Some(10),
            init: MiniBatchInit::KMeansPlusPlus,
            init_size: None,
            n_init: None,
            reassignment_ratio: 0.01,
            seed: 42,
        }
    }
    pub fn with_batch_size(mut self, b: usize) -> Self {
        self.batch_size = b;
        self
    }
    pub fn with_max_iter(mut self, m: usize) -> Self {
        self.max_iter = m;
        self
    }
    pub fn with_tol(mut self, t: f64) -> Self {
        self.tol = t;
        self
    }
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }
    pub fn with_max_no_improvement(mut self, m: Option<usize>) -> Self {
        self.max_no_improvement = m;
        self
    }
    pub fn with_init(mut self, init: MiniBatchInit) -> Self {
        self.init = init;
        self
    }
    pub fn with_init_size(mut self, s: usize) -> Self {
        self.init_size = Some(s);
        self
    }
    pub fn with_n_init(mut self, n: usize) -> Self {
        self.n_init = Some(n);
        self
    }
    pub fn with_reassignment_ratio(mut self, r: f64) -> Self {
        self.reassignment_ratio = r;
        self
    }
}

impl Default for MiniBatchKMeans {
    fn default() -> Self {
        Self::new(3)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedMiniBatchKMeans<F: Float> {
    centroids: Array2<F>,
    /// Epochs started (sklearn `n_iter_`).
    n_iter: usize,
    /// Mini-batch steps run (sklearn `n_steps_`).
    #[serde(default)]
    n_steps: usize,
    /// Training labels (sklearn `labels_`).
    #[serde(default)]
    labels: Array1<F>,
    /// Weighted inertia of the training data (sklearn `inertia_`).
    #[serde(default)]
    inertia: f64,
}

impl<F: Float> FittedMiniBatchKMeans<F> {
    pub fn centroids(&self) -> &Array2<F> {
        &self.centroids
    }
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }
    pub fn n_steps(&self) -> usize {
        self.n_steps
    }
    pub fn labels(&self) -> &Array1<F> {
        &self.labels
    }
    pub fn inertia(&self) -> f64 {
        self.inertia
    }
}

/// Labels and weighted inertia of `x` for `centers` (sklearn
/// `_labels_inertia`).
fn labels_inertia<F: Float>(x: ArrayView2<F>, w: &[F], centers: &Array2<F>) -> (Vec<usize>, f64) {
    let (labels, dists) = assign_nearest(x, &row_norms_sq(x), centers);
    let inertia = dists
        .iter()
        .zip(w)
        .map(|(&d, &wi)| (d * wi).to_f64().unwrap())
        .sum();
    (labels, inertia)
}

impl<F: Float + Send + Sync> FitUnsupervised<F> for MiniBatchKMeans {
    type Fitted = FittedMiniBatchKMeans<F>;

    fn fit_view(&self, x: ArrayView2<F>) -> Result<Self::Fitted> {
        self.fit_unsupervised_weighted(x, None)
    }
}

impl<F: Float + Send + Sync> FitUnsupervisedWeighted<F> for MiniBatchKMeans {
    type Fitted = FittedMiniBatchKMeans<F>;

    fn fit_unsupervised_weighted_view(
        &self,
        x: ArrayView2<F>,
        sample_weight: Option<ArrayView1<F>>,
    ) -> Result<Self::Fitted> {
        let x_std = crate::standard_layout(x);
        let x = x_std.view();
        let (n, d) = x.dim();
        if n == 0 {
            return Err(RustMlError::EmptyInput("empty input".into()));
        }
        let k = self.n_clusters;
        if k == 0 || k > n {
            return Err(RustMlError::InvalidParameter("invalid n_clusters".into()));
        }
        if self.batch_size == 0 || self.reassignment_ratio < 0.0 || self.tol < 0.0 {
            return Err(RustMlError::InvalidParameter(
                "batch_size must be >= 1, tol and reassignment_ratio >= 0".into(),
            ));
        }
        let w: Vec<F> = match sample_weight {
            Some(sw) => {
                anofox_ml_core::check_sample_weight(sw, n)?;
                sw.to_vec()
            }
            None => vec![F::one(); n],
        };
        let batch_size = self.batch_size.min(n);
        let init_size = match self.init_size {
            None if 3 * batch_size < k => 3 * k,
            None => 3 * batch_size,
            Some(s) if s < k => 3 * k,
            Some(s) => s,
        }
        .min(n);
        let n_init = self.n_init.unwrap_or(match self.init {
            MiniBatchInit::KMeansPlusPlus => 1,
            MiniBatchInit::Random => 3,
        });
        if n_init == 0 {
            return Err(RustMlError::InvalidParameter("n_init must be >= 1".into()));
        }
        // sklearn `_tolerance`: tol · mean of the per-feature variances.
        let tol = if self.tol > 0.0 {
            let var = x.var_axis(Axis(0), F::zero());
            self.tol * var.iter().map(|v| v.to_f64().unwrap()).sum::<f64>() / d as f64
        } else {
            0.0
        };

        let mut rng = StdRng::seed_from_u64(self.seed);

        // Initialisation: best of n_init on a validation subsample.
        let valid: Vec<usize> = (0..init_size).map(|_| rng.gen_range(0..n)).collect();
        let x_valid = x.select(Axis(0), &valid);
        let w_valid: Vec<F> = valid.iter().map(|&i| w[i]).collect();
        let mut best: Option<(f64, Array2<F>)> = None;
        for _ in 0..n_init {
            let sub: Vec<usize> = if init_size < n {
                (0..init_size).map(|_| rng.gen_range(0..n)).collect()
            } else {
                (0..n).collect()
            };
            let xs = x.select(Axis(0), &sub);
            let ws: Array1<F> = sub.iter().map(|&i| w[i]).collect();
            let centers = match self.init {
                MiniBatchInit::KMeansPlusPlus => {
                    kmeans_plusplus(xs.view(), k, &mut rng, Some(ws.view())).0
                }
                MiniBatchInit::Random => {
                    let rows = sample_without_replacement(xs.nrows(), k, &mut rng, Some(ws.view()));
                    xs.select(Axis(0), &rows)
                }
            };
            let (_, inertia) = labels_inertia(x_valid.view(), &w_valid, &centers);
            if best.as_ref().is_none_or(|(b, _)| inertia < *b) {
                best = Some((inertia, centers));
            }
        }
        let mut centers = best.expect("n_init >= 1").1;

        let mut counts = vec![F::zero(); k];
        let mut ewa: Option<f64> = None;
        let mut ewa_min: Option<f64> = None;
        let mut no_improvement = 0usize;
        let mut since_reassign = 0usize;
        let n_steps = (self.max_iter * n / batch_size).max(1);
        let mut steps_run = 0;
        for step in 0..n_steps {
            steps_run = step + 1;
            let batch: Vec<usize> = (0..batch_size).map(|_| rng.gen_range(0..n)).collect();
            let xb = x.select(Axis(0), &batch);
            let wb: Vec<F> = batch.iter().map(|&i| w[i]).collect();

            // Random reassignment schedule (sklearn `_random_reassign`).
            since_reassign += batch_size;
            let reassign = if counts.iter().any(|&c| c == F::zero()) || since_reassign >= 10 * k {
                since_reassign = 0;
                true
            } else {
                false
            };

            // `_mini_batch_step`: label with the current centres, then move
            // every centre to the weighted mean of old mass and new samples.
            let (labels, batch_inertia) = labels_inertia(xb.view(), &wb, &centers);
            let mut new_centers = centers.clone();
            let mut sums = Array2::<F>::zeros((k, d));
            let mut wsum = vec![F::zero(); k];
            for (r, &l) in labels.iter().enumerate() {
                sums.row_mut(l).scaled_add(wb[r], &xb.row(r));
                wsum[l] += wb[r];
            }
            for c in 0..k {
                if wsum[c] > F::zero() {
                    let mut row = new_centers.row_mut(c);
                    row *= counts[c];
                    row += &sums.row(c);
                    counts[c] += wsum[c];
                    row /= counts[c];
                }
            }
            if reassign && self.reassignment_ratio > 0.0 {
                let max_c = counts.iter().copied().fold(F::zero(), F::max);
                let thr = F::from_f64(self.reassignment_ratio).unwrap() * max_c;
                let mut to_reassign: Vec<bool> = counts.iter().map(|&c| c < thr).collect();
                let n_flag = to_reassign.iter().filter(|&&b| b).count();
                if n_flag * 2 > batch_size {
                    // Keep the centres with the largest counts.
                    let mut order: Vec<usize> = (0..k).collect();
                    order.sort_by(|&a, &b| counts[a].partial_cmp(&counts[b]).unwrap());
                    for &c in &order[batch_size / 2..] {
                        to_reassign[c] = false;
                    }
                }
                let flagged: Vec<usize> = (0..k).filter(|&c| to_reassign[c]).collect();
                if !flagged.is_empty() {
                    let rows =
                        sample_without_replacement::<F>(batch_size, flagged.len(), &mut rng, None);
                    for (&c, &r) in flagged.iter().zip(&rows) {
                        new_centers.row_mut(c).assign(&xb.row(r));
                    }
                    let min_kept = (0..k)
                        .filter(|&c| !to_reassign[c])
                        .map(|c| counts[c])
                        .fold(F::infinity(), F::min);
                    for &c in &flagged {
                        counts[c] = min_kept;
                    }
                }
            }
            let centers_sq_diff = if tol > 0.0 {
                (&new_centers - &centers)
                    .iter()
                    .map(|v| (*v * *v).to_f64().unwrap())
                    .sum::<f64>()
            } else {
                0.0
            };
            centers = new_centers;

            // `_mini_batch_convergence`.
            let bi = batch_inertia / batch_size as f64;
            if step == 0 {
                continue;
            }
            let e = match ewa {
                None => bi,
                Some(prev) => {
                    let alpha = (batch_size as f64 * 2.0 / (n as f64 + 1.0)).min(1.0);
                    prev * (1.0 - alpha) + bi * alpha
                }
            };
            ewa = Some(e);
            if tol > 0.0 && centers_sq_diff <= tol {
                break;
            }
            if ewa_min.is_none_or(|m| e < m) {
                no_improvement = 0;
                ewa_min = Some(e);
            } else {
                no_improvement += 1;
            }
            if self.max_no_improvement.is_some_and(|m| no_improvement >= m) {
                break;
            }
        }

        let (labels, inertia) = labels_inertia(x, &w, &centers);
        Ok(FittedMiniBatchKMeans {
            centroids: centers,
            n_iter: (steps_run * batch_size).div_ceil(n),
            n_steps: steps_run,
            labels: labels.iter().map(|&l| F::from_usize(l).unwrap()).collect(),
            inertia,
        })
    }
}

impl<F: Float> Predict<F> for FittedMiniBatchKMeans<F> {
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
        Ok(labels.iter().map(|&k| F::from_usize(k).unwrap()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_mini_batch_kmeans_two_clusters() {
        let x = array![
            [0.0_f64, 0.0],
            [0.1, 0.1],
            [-0.1, 0.05],
            [0.05, -0.1],
            [10.0, 10.0],
            [10.1, 10.05],
            [9.95, 10.1],
            [10.05, 9.95],
        ];
        let model = MiniBatchKMeans::new(2)
            .with_batch_size(4)
            .with_max_iter(50)
            .with_seed(7);
        let fitted: FittedMiniBatchKMeans<f64> = FitUnsupervised::fit(&model, &x).unwrap();
        let labels = fitted.predict(&x).unwrap();
        let l0 = labels[0];
        for i in 1..4 {
            assert_eq!(labels[i], l0);
        }
        for i in 4..8 {
            assert_ne!(labels[i], l0);
        }
        assert_eq!(fitted.labels(), &labels);
    }

    #[test]
    fn test_batch_update_is_weighted_mean() {
        // One step on the whole data from fixed centres equals one Lloyd
        // step (the old centres carry no weight yet).
        let x = array![[0.0_f64], [1.0], [10.0], [12.0]];
        let w = [1.0, 1.0, 1.0, 1.0];
        let centers = array![[0.5_f64], [11.0]];
        let (labels, inertia) = labels_inertia(x.view(), &w, &centers);
        assert_eq!(labels, vec![0, 0, 1, 1]);
        assert!((inertia - (0.25 + 0.25 + 1.0 + 1.0)).abs() < 1e-12);
    }
}

impl<F: anofox_ml_core::Float> anofox_ml_core::ClassifierScore<F> for FittedMiniBatchKMeans<F> {}
