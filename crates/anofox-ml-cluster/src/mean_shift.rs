//! Mean-Shift clustering.
//!
//! Mirrors `sklearn.cluster.MeanShift` with a flat kernel:
//!
//! 1. Seeds are the samples, the given `seeds`, or with `bin_seeding` the
//!    centres of a grid of `bandwidth`-sized bins holding at least
//!    `min_bin_freq` samples.
//! 2. Every seed moves to the mean of the samples within `bandwidth` until
//!    it moves less than `1e-3 · bandwidth` or `max_iter` is reached (seeds
//!    in parallel, neighbourhoods from the shared neighbour index).
//! 3. The resulting modes are sorted by the number of samples in their
//!    window (then by coordinates, descending); every mode within
//!    `bandwidth` of a stronger one is dropped.
//! 4. Each sample joins its nearest centre (or, without `cluster_all`, gets
//!    -1 when no centre is within `bandwidth`).
//!
//! Without a bandwidth, [`estimate_bandwidth`] picks one.

use anofox_ml_core::Float;
use anofox_ml_core::{FitUnsupervised, Predict, Result, RustMlError};
use anofox_ml_neighbors::{DistanceMetric, NeighborAlgorithm, NeighborIndex};
use ndarray::{Array1, Array2, ArrayView2, AsArray, Axis, Ix2};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MeanShift {
    /// Kernel radius; `None` estimates it with [`estimate_bandwidth`].
    pub bandwidth: Option<f64>,
    /// Starting points; `None` uses the samples (or bin seeds).
    #[serde(default)]
    pub seeds: Option<Array2<f64>>,
    /// Seed from a grid of bandwidth-sized bins instead of every sample.
    #[serde(default)]
    pub bin_seeding: bool,
    /// Minimum samples in a bin for it to become a seed.
    #[serde(default = "one")]
    pub min_bin_freq: usize,
    /// Assign every sample to a cluster (else far samples get -1).
    #[serde(default = "yes")]
    pub cluster_all: bool,
    pub max_iter: usize,
}

fn one() -> usize {
    1
}
fn yes() -> bool {
    true
}

impl MeanShift {
    pub fn new(bandwidth: f64) -> Self {
        Self {
            bandwidth: Some(bandwidth),
            seeds: None,
            bin_seeding: false,
            min_bin_freq: 1,
            cluster_all: true,
            max_iter: 300,
        }
    }
    /// sklearn's defaults: estimated bandwidth, every sample a seed.
    pub fn auto() -> Self {
        Self {
            bandwidth: None,
            ..Self::new(1.0)
        }
    }
    pub fn with_max_iter(mut self, m: usize) -> Self {
        self.max_iter = m;
        self
    }
    pub fn with_bin_seeding(mut self, bin_seeding: bool, min_bin_freq: usize) -> Self {
        self.bin_seeding = bin_seeding;
        self.min_bin_freq = min_bin_freq;
        self
    }
    pub fn with_seeds(mut self, seeds: Option<Array2<f64>>) -> Self {
        self.seeds = seeds;
        self
    }
    pub fn with_cluster_all(mut self, cluster_all: bool) -> Self {
        self.cluster_all = cluster_all;
        self
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedMeanShift {
    pub cluster_centers: Array2<f64>,
    pub labels: Array1<f64>,
    /// Bandwidth used (estimated or given).
    #[serde(default)]
    pub bandwidth: f64,
    /// Most iterations any seed needed (sklearn's `n_iter_`).
    #[serde(default)]
    pub n_iter: usize,
}

/// sklearn's `estimate_bandwidth`: the mean distance of every sample to its
/// `⌊quantile · n⌋`-th nearest sample (itself included), computed on a
/// random subset of `n_samples` rows if given.
pub fn estimate_bandwidth<'a>(
    x: impl AsArray<'a, f64, Ix2>,
    quantile: f64,
    n_samples: Option<usize>,
    seed: u64,
) -> Result<f64> {
    let x: ArrayView2<f64> = x.into();
    if x.nrows() == 0 {
        return Err(RustMlError::EmptyInput("empty input".into()));
    }
    if !(0.0..=1.0).contains(&quantile) {
        return Err(RustMlError::InvalidParameter(format!(
            "quantile must be in [0, 1], got {quantile}"
        )));
    }
    let sub;
    let x = match n_samples {
        Some(m) if m < x.nrows() => {
            let mut idx: Vec<usize> = (0..x.nrows()).collect();
            idx.shuffle(&mut StdRng::seed_from_u64(seed));
            idx.truncate(m.max(1));
            sub = x.select(Axis(0), &idx);
            sub.view()
        }
        _ => x.view(),
    };
    let k = ((x.nrows() as f64 * quantile) as usize).max(1);
    let index = NeighborIndex::new(
        x.view(),
        DistanceMetric::Euclidean,
        NeighborAlgorithm::Auto,
        Some(k),
    )?;
    let (d, _) = index.kneighbors(x.view(), k)?;
    Ok(d.column(k - 1).sum() / x.nrows() as f64)
}

/// sklearn's `get_bin_seeds`: grid cells (rounded half to even) holding at
/// least `min_bin_freq` samples, in first-seen order, as single precision.
fn bin_seeds(x: ArrayView2<f64>, bin_size: f64, min_bin_freq: usize) -> Array2<f64> {
    if bin_size == 0.0 {
        return x.to_owned();
    }
    let mut cells: Vec<(Vec<f64>, usize)> = Vec::new();
    let mut lookup: std::collections::HashMap<Vec<u64>, usize> = std::collections::HashMap::new();
    for row in x.rows() {
        let cell: Vec<f64> = row
            .iter()
            .map(|v| (v / bin_size).round_ties_even())
            .collect();
        let key: Vec<u64> = cell.iter().map(|v| (v + 0.0).to_bits()).collect();
        match lookup.get(&key) {
            Some(&i) => cells[i].1 += 1,
            None => {
                lookup.insert(key, cells.len());
                cells.push((cell, 1));
            }
        }
    }
    let kept: Vec<&Vec<f64>> = cells
        .iter()
        .filter(|c| c.1 >= min_bin_freq)
        .map(|c| &c.0)
        .collect();
    if kept.len() == x.nrows() {
        return x.to_owned();
    }
    let d = x.ncols();
    Array2::from_shape_fn((kept.len(), d), |(i, j)| {
        ((kept[i][j] as f32) * (bin_size as f32)) as f64
    })
}

impl<F: Float> FitUnsupervised<F> for MeanShift {
    type Fitted = FittedMeanShift;

    /// Computed in `f64` whatever the input float type.
    fn fit_view(&self, x: ArrayView2<F>) -> Result<Self::Fitted> {
        let x_f64 = anofox_ml_core::as_f64(x);
        let x = x_f64.view();
        let n = x.nrows();
        if n == 0 {
            return Err(RustMlError::EmptyInput("empty input".into()));
        }
        let bandwidth = match self.bandwidth {
            Some(b) if b > 0.0 => b,
            Some(_) => {
                return Err(RustMlError::InvalidParameter(
                    "bandwidth must be > 0".into(),
                ))
            }
            None => estimate_bandwidth(x, 0.3, None, 0)?,
        };
        let seeds = match &self.seeds {
            Some(s) if s.ncols() != x.ncols() => {
                return Err(RustMlError::ShapeMismatch(format!(
                    "seeds have {} features, X has {}",
                    s.ncols(),
                    x.ncols()
                )))
            }
            Some(s) => s.clone(),
            None if self.bin_seeding => bin_seeds(x, bandwidth, self.min_bin_freq),
            None => x.to_owned(),
        };

        let index =
            NeighborIndex::new(x, DistanceMetric::Euclidean, NeighborAlgorithm::Auto, None)?;
        let stop = 1e-3 * bandwidth;
        // (mode, points in its final window, iterations) per seed.
        let results: Vec<(Vec<f64>, usize, usize)> = (0..seeds.nrows())
            .into_par_iter()
            .map(|s| -> Result<(Vec<f64>, usize, usize)> {
                let mut mean = seeds.row(s).to_owned();
                let mut done = 0usize;
                loop {
                    let q = mean.view().insert_axis(Axis(0)).to_owned();
                    let nb = &index.radius_neighbors(q.view(), bandwidth)?[0];
                    if nb.is_empty() {
                        return Ok((mean.to_vec(), 0, done));
                    }
                    let idx: Vec<usize> = nb.iter().map(|p| p.0).collect();
                    let new = x
                        .select(Axis(0), &idx)
                        .mean_axis(Axis(0))
                        .expect("non-empty");
                    let moved = (&new - &mean).mapv(|v| v * v).sum().sqrt();
                    mean = new;
                    if moved <= stop || done == self.max_iter {
                        return Ok((mean.to_vec(), idx.len(), done));
                    }
                    done += 1;
                }
            })
            .collect::<Result<_>>()?;
        let n_iter = results.iter().map(|r| r.2).max().unwrap_or(0);

        // Distinct modes with their intensity (a later identical mode
        // overwrites the count, as sklearn's dict does).
        let mut modes: Vec<(Vec<f64>, usize)> = Vec::new();
        for (m, count, _) in results.into_iter().filter(|r| r.1 > 0) {
            match modes.iter_mut().find(|e| e.0 == m) {
                Some(e) => e.1 = count,
                None => modes.push((m, count)),
            }
        }
        if modes.is_empty() {
            return Err(RustMlError::InvalidParameter(format!(
                "no sample within bandwidth = {bandwidth} of any seed"
            )));
        }
        // Strongest first; ties by coordinates, descending.
        modes.sort_by(|a, b| {
            b.1.cmp(&a.1).then_with(|| {
                b.0.iter()
                    .zip(&a.0)
                    .map(|(u, v)| u.total_cmp(v))
                    .find(|o| o.is_ne())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        });
        let d = x.ncols();
        let sorted = Array2::from_shape_fn((modes.len(), d), |(i, j)| modes[i].0[j]);
        let center_index = NeighborIndex::new(
            sorted.view(),
            DistanceMetric::Euclidean,
            NeighborAlgorithm::Auto,
            None,
        )?;
        let near = center_index.radius_neighbors(sorted.view(), bandwidth)?;
        let mut unique = vec![true; modes.len()];
        for i in 0..modes.len() {
            if unique[i] {
                for &(j, _) in &near[i] {
                    unique[j] = false;
                }
                unique[i] = true;
            }
        }
        let keep: Vec<usize> = (0..modes.len()).filter(|&i| unique[i]).collect();
        let cluster_centers = sorted.select(Axis(0), &keep);

        let (dist, idx) = NeighborIndex::new(
            cluster_centers.view(),
            DistanceMetric::Euclidean,
            NeighborAlgorithm::Auto,
            Some(1),
        )?
        .kneighbors(x, 1)?;
        let labels = Array1::from_shape_fn(n, |i| {
            if self.cluster_all || dist[[i, 0]] <= bandwidth {
                idx[[i, 0]] as f64
            } else {
                -1.0
            }
        });
        Ok(FittedMeanShift {
            cluster_centers,
            labels,
            bandwidth,
            n_iter,
        })
    }
}

impl FittedMeanShift {
    fn predict_f64(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        let d = self.cluster_centers.ncols();
        if x.ncols() != d {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                d,
                x.ncols()
            )));
        }
        let (_, idx) = NeighborIndex::new(
            self.cluster_centers.view(),
            DistanceMetric::Euclidean,
            NeighborAlgorithm::Auto,
            Some(1),
        )?
        .kneighbors(x.view(), 1)?;
        Ok(idx.column(0).mapv(|v| v as f64))
    }
}

impl<F: Float> Predict<F> for FittedMeanShift {
    /// Index of the nearest cluster centre.
    fn predict_view(&self, x: ArrayView2<F>) -> Result<Array1<F>> {
        Ok(anofox_ml_core::from_f64(
            self.predict_f64(anofox_ml_core::as_f64(x).view())?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_mean_shift_two_blobs() {
        let x = array![
            [0.0_f64, 0.0],
            [0.1, 0.1],
            [-0.1, 0.2],
            [0.1, -0.1],
            [10.0, 10.0],
            [10.1, 9.9],
            [9.9, 10.2],
            [10.0, 10.1],
        ];
        let ms = MeanShift::new(2.0);
        let fitted = ms.fit(&x).unwrap();
        assert_eq!(fitted.cluster_centers.nrows(), 2);
        let l0 = fitted.labels[0];
        for i in 1..4 {
            assert_eq!(fitted.labels[i], l0);
        }
        for i in 4..8 {
            assert_ne!(fitted.labels[i], l0);
        }
    }

    #[test]
    fn test_mean_shift_centers_near_blob_means() {
        let x = array![
            [0.0_f64, 0.0],
            [0.1, 0.1],
            [-0.1, 0.2],
            [0.1, -0.1],
            [10.0, 10.0],
            [10.1, 9.9],
            [9.9, 10.2],
            [10.0, 10.1],
        ];
        let fitted = MeanShift::new(2.0).fit(&x).unwrap();
        // Centroids should be near (0.025, 0.05) and (10.0, 10.05).
        let mut has_low = false;
        let mut has_high = false;
        for k in 0..fitted.cluster_centers.nrows() {
            let cx = fitted.cluster_centers[[k, 0]];
            if cx.abs() < 1.0 {
                has_low = true;
            }
            if (cx - 10.0).abs() < 1.0 {
                has_high = true;
            }
        }
        assert!(has_low && has_high);
    }
}

impl<F: Float> anofox_ml_core::ClassifierScore<F> for FittedMeanShift {}
