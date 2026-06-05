//! Mini-batch K-Means.
//!
//! Mirrors `sklearn.cluster.MiniBatchKMeans`. Samples a batch of size
//! `batch_size` per iteration and applies the per-sample learning-rate
//! update of Sculley (2010):
//!
//!   cₖ ← (1 − 1/Nₖ) cₖ + (1/Nₖ) x       — for the cluster `k` a sample
//!                                          is assigned to.
//!
//! where `Nₖ` is the running count of samples ever assigned to cluster `k`.

use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rustml_core::{FitUnsupervised, Float, Predict, Result, RustMlError};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiniBatchKMeans {
    pub n_clusters: usize,
    pub batch_size: usize,
    pub max_iter: usize,
    pub tol: f64,
    pub seed: u64,
}

impl MiniBatchKMeans {
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            batch_size: 256,
            max_iter: 100,
            tol: 1e-4,
            seed: 42,
        }
    }
    pub fn with_batch_size(mut self, b: usize) -> Self { self.batch_size = b; self }
    pub fn with_max_iter(mut self, m: usize) -> Self { self.max_iter = m; self }
    pub fn with_tol(mut self, t: f64) -> Self { self.tol = t; self }
    pub fn with_seed(mut self, s: u64) -> Self { self.seed = s; self }
}

impl Default for MiniBatchKMeans {
    fn default() -> Self { Self::new(3) }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedMiniBatchKMeans<F: Float> {
    centroids: Array2<F>,
    n_iter: usize,
}

impl<F: Float> FittedMiniBatchKMeans<F> {
    pub fn centroids(&self) -> &Array2<F> { &self.centroids }
    pub fn n_iter(&self) -> usize { self.n_iter }
}

fn sq_dist<F: Float>(a: &[F], b: &[F]) -> F {
    let mut acc = F::zero();
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = x - y;
        acc = acc + d * d;
    }
    acc
}

fn nearest<F: Float>(point: &[F], centroids: &Array2<F>) -> usize {
    let mut best = 0;
    let mut best_d = F::infinity();
    for k in 0..centroids.nrows() {
        let row = centroids.row(k);
        let d = sq_dist(point, row.as_slice().unwrap());
        if d < best_d {
            best_d = d;
            best = k;
        }
    }
    best
}

fn weighted_choice<F: Float>(weights: &Array1<F>, rng: &mut StdRng) -> usize {
    let total: F = weights.iter().copied().fold(F::zero(), |acc, v| acc + v);
    let r = F::from_f64(rng.gen::<f64>()).unwrap() * total;
    let mut cum = F::zero();
    for (i, &w) in weights.iter().enumerate() {
        cum = cum + w;
        if cum >= r {
            return i;
        }
    }
    weights.len() - 1
}

fn init_centroids<F: Float>(x: &Array2<F>, k: usize, rng: &mut StdRng) -> Array2<F> {
    let n = x.nrows();
    let d = x.ncols();
    let mut centroids = Array2::<F>::zeros((k, d));
    let first = rng.gen_range(0..n);
    centroids.row_mut(0).assign(&x.row(first));

    let mut min_d = Array1::<F>::from_elem(n, F::infinity());
    for ci in 1..k {
        for i in 0..n {
            let d2 = sq_dist(
                x.row(i).as_slice().unwrap(),
                centroids.row(ci - 1).as_slice().unwrap(),
            );
            if d2 < min_d[i] {
                min_d[i] = d2;
            }
        }
        let total: F = min_d.iter().copied().fold(F::zero(), |a, v| a + v);
        if total == F::zero() {
            centroids.row_mut(ci).assign(&x.row(rng.gen_range(0..n)));
            continue;
        }
        let pick = weighted_choice(&min_d, rng);
        centroids.row_mut(ci).assign(&x.row(pick));
    }
    centroids
}

impl<F: Float + Send + Sync> FitUnsupervised<F> for MiniBatchKMeans {
    type Fitted = FittedMiniBatchKMeans<F>;

    fn fit(&self, x: &Array2<F>) -> Result<Self::Fitted> {
        let n = x.nrows();
        let d = x.ncols();
        if n == 0 {
            return Err(RustMlError::EmptyInput("empty input".into()));
        }
        if self.n_clusters == 0 || self.n_clusters > n {
            return Err(RustMlError::InvalidParameter("invalid n_clusters".into()));
        }
        let batch_size = self.batch_size.min(n);

        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut centroids = init_centroids(x, self.n_clusters, &mut rng);
        let mut counts = vec![0usize; self.n_clusters];

        let tol = F::from_f64(self.tol).unwrap();
        let mut n_iter = 0;

        // Reused per iteration: partial Fisher-Yates shuffles the first
        // `batch_size` slots of `idx`, no per-iter allocation.
        let mut idx: Vec<usize> = (0..n).collect();

        for iter in 0..self.max_iter {
            n_iter = iter + 1;

            // Partial Fisher-Yates: pick batch_size distinct indices without
            // shuffling the whole tail.
            use rand::Rng;
            for k in 0..batch_size {
                let pick = k + rng.gen_range(0..(n - k));
                idx.swap(k, pick);
            }
            let batch: &[usize] = &idx[..batch_size];

            // Save old centroids to measure shift.
            let prev = centroids.clone();

            for &i in batch {
                let row_slice = x.row(i).into_owned();
                let k = nearest(row_slice.as_slice().unwrap(), &centroids);
                counts[k] += 1;
                let lr = F::one() / F::from_usize(counts[k]).unwrap();
                let one = F::one();
                for j in 0..d {
                    centroids[[k, j]] = (one - lr) * centroids[[k, j]] + lr * row_slice[j];
                }
            }

            // Convergence check: max centroid shift.
            let mut max_shift = F::zero();
            for kk in 0..self.n_clusters {
                let s = sq_dist(
                    prev.row(kk).as_slice().unwrap(),
                    centroids.row(kk).as_slice().unwrap(),
                );
                if s > max_shift {
                    max_shift = s;
                }
            }
            if max_shift < tol * tol {
                break;
            }
        }

        Ok(FittedMiniBatchKMeans { centroids, n_iter })
    }
}

impl<F: Float> Predict<F> for FittedMiniBatchKMeans<F> {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        if x.ncols() != self.centroids.ncols() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.centroids.ncols(),
                x.ncols()
            )));
        }
        let mut out = Array1::<F>::zeros(x.nrows());
        for i in 0..x.nrows() {
            let row = x.row(i).into_owned();
            let k = nearest(row.as_slice().unwrap(), &self.centroids);
            out[i] = F::from_usize(k).unwrap();
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_mini_batch_kmeans_two_clusters() {
        let x = array![
            [0.0_f64, 0.0], [0.1, 0.1], [-0.1, 0.05], [0.05, -0.1],
            [10.0, 10.0], [10.1, 10.05], [9.95, 10.1], [10.05, 9.95],
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
    }
}

impl<F: rustml_core::Float> rustml_core::ClassifierScore<F> for FittedMiniBatchKMeans<F> {}
