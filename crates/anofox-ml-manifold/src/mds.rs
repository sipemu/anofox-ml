//! Multidimensional scaling by SMACOF (sklearn's `MDS`): metric or
//! non-metric stress majorization from random starts (numpy's
//! `RandomState`, so the starts are sklearn's for the same seed) or from
//! [`ClassicalMds`](crate::ClassicalMds).

use anofox_ml_core::numpy_random::RandomState;
use anofox_ml_core::{Fit, FitUnsupervised, Predict, Result, RustMlError};
use anofox_ml_neighbors::DistanceMetric;
use anofox_ml_regression::{IsotonicRegressor, OutOfBounds};
use ndarray::{Array1, Array2, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

/// The starting configuration of [`Mds`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MdsInit {
    /// `n_init` uniform random starts (sklearn 1.8's default).
    #[default]
    Random,
    /// One start from classical MDS (in 2 dimensions, as sklearn 1.8:
    /// the embedding then has 2 columns whatever `n_components` says).
    ClassicalMds,
}

/// sklearn's `MDS` (SMACOF): an `n_components`-dimensional embedding whose
/// distances fit the dissimilarities (euclidean distances of `x`, or `x`
/// itself with `precomputed`). `metric_mds = false` fits only their order
/// (isotonic regression). The best of `n_init` starts by stress is kept.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Mds {
    pub n_components: usize,
    pub metric_mds: bool,
    pub n_init: usize,
    pub init: MdsInit,
    pub max_iter: usize,
    pub eps: f64,
    pub precomputed: bool,
    /// `None`: sklearn's "auto" (normalized for non-metric MDS).
    pub normalized_stress: Option<bool>,
    pub seed: u32,
}

impl Default for Mds {
    fn default() -> Self {
        Self {
            n_components: 2,
            metric_mds: true,
            n_init: 4,
            init: MdsInit::Random,
            max_iter: 300,
            eps: 1e-6,
            precomputed: false,
            normalized_stress: None,
            seed: 0,
        }
    }
}

/// A fitted [`Mds`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedMds {
    /// `n × n_components` (sklearn's `embedding_`).
    pub embedding: Array2<f64>,
    /// The final stress (sklearn's `stress_`).
    pub stress: f64,
    /// Iterations of the best start (sklearn's `n_iter_`).
    pub n_iter: usize,
    /// sklearn's `dissimilarity_matrix_`.
    pub dissimilarity: Array2<f64>,
}

/// sklearn's `euclidean_distances(x)`: `‖a‖² + ‖b‖² − 2 a·b`, clipped at 0,
/// zero diagonal, square-rooted.
fn euclidean_distances(x: &Array2<f64>) -> Array2<f64> {
    let sq: Array1<f64> = x.rows().into_iter().map(|r| r.dot(&r)).collect();
    let mut d = x.dot(&x.t()) * -2.0;
    let n = x.nrows();
    for i in 0..n {
        for j in 0..n {
            d[[i, j]] = if i == j {
                0.0
            } else {
                (d[[i, j]] + sq[i] + sq[j]).max(0.0).sqrt()
            };
        }
    }
    d
}

/// The result of one SMACOF run.
struct Run {
    x: Array2<f64>,
    stress: f64,
    n_iter: usize,
}

/// sklearn's `_smacof_single`.
#[allow(clippy::too_many_arguments)]
fn smacof_single(
    d: &Array2<f64>,
    metric: bool,
    k: usize,
    init: Option<&Array2<f64>>,
    max_iter: usize,
    eps: f64,
    rng: &mut RandomState,
    normalized: bool,
) -> Result<Run> {
    let n = d.nrows();
    // The strictly upper triangle's non-zero dissimilarities.
    let mut upper: Vec<(usize, usize)> = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            if d[[i, j]] != 0.0 {
                upper.push((i, j));
            }
        }
    }
    let dis_w = Array2::from_shape_fn((upper.len(), 1), |(r, _)| d[[upper[r].0, upper[r].1]]);
    let mut x = match init {
        Some(x0) => x0.clone(),
        None => Array2::from_shape_simple_fn((n, k), || rng.random_sample()),
    };
    let mut distances = euclidean_distances(&x);
    let iso = IsotonicRegressor::new().with_out_of_bounds(OutOfBounds::Clip);
    let mut old_stress: Option<f64> = None;
    let mut stress = 0.0;
    let mut it_done = 0;
    for it in 0..max_iter {
        it_done = it + 1;
        let disparities = if metric {
            d.clone()
        } else {
            let fitted_vals: Vec<f64> = if it < 1 {
                dis_w.column(0).to_vec()
            } else {
                let dist_w: Array1<f64> = upper.iter().map(|&(i, j)| distances[[i, j]]).collect();
                let model = iso.fit(&dis_w, &dist_w)?;
                model.predict(&dis_w)?.to_vec()
            };
            let mut disp = Array2::<f64>::zeros((n, n));
            for (&(i, j), &v) in upper.iter().zip(&fitted_vals) {
                disp[[i, j]] = v;
            }
            let total = disp.mapv(|v| v * v).sum();
            disp *= ((n * (n - 1)) as f64 / 2.0 / total).sqrt();
            &disp + &disp.t()
        };
        distances.mapv_inplace(|v| if v == 0.0 { 1e-5 } else { v });
        let ratio = &disparities / &distances;
        let mut b = -&ratio;
        let sums = ratio.sum_axis(Axis(1));
        for i in 0..n {
            b[[i, i]] += sums[i];
        }
        x = b.dot(&x) / n as f64;
        distances = euclidean_distances(&x);
        stress = (&distances - &disparities).mapv(|v| v * v).sum() / 2.0;
        if let Some(old) = old_stress {
            let ssd = distances.mapv(|v| v * v).sum();
            if (old - stress) / (ssd / 2.0) < eps {
                break;
            }
        }
        old_stress = Some(stress);
    }
    if normalized {
        let ssd = distances.mapv(|v| v * v).sum();
        stress = (stress / (ssd / 2.0)).sqrt();
    }
    Ok(Run {
        x,
        stress,
        n_iter: it_done,
    })
}

impl FitUnsupervised<f64> for Mds {
    type Fitted = FittedMds;

    fn fit_view(&self, x: ArrayView2<f64>) -> Result<FittedMds> {
        let n = x.nrows();
        if n < 2 || x.ncols() == 0 {
            return Err(RustMlError::InvalidParameter(
                "MDS needs at least 2 samples".into(),
            ));
        }
        if self.n_components == 0 || self.n_init == 0 || self.max_iter == 0 {
            return Err(RustMlError::InvalidParameter(
                "n_components, n_init and max_iter must be at least 1".into(),
            ));
        }
        let d = if self.precomputed {
            if x.ncols() != n {
                return Err(RustMlError::ShapeMismatch(
                    "a precomputed dissimilarity matrix must be square".into(),
                ));
            }
            for i in 0..n {
                for j in 0..n {
                    if (x[[i, j]] - x[[j, i]]).abs() > 1e-10 * x[[i, j]].abs().max(1.0) {
                        return Err(RustMlError::InvalidParameter(
                            "the dissimilarity matrix must be symmetric".into(),
                        ));
                    }
                }
            }
            x.to_owned()
        } else {
            euclidean_distances(&x.to_owned())
        };
        let (init, n_init) = match self.init {
            MdsInit::Random => (None, self.n_init),
            MdsInit::ClassicalMds => {
                // sklearn 1.8 builds ClassicalMDS with its default 2
                // components, and SMACOF takes the start's dimension: the
                // embedding is 2-D whatever `n_components` says.
                let cmds = crate::ClassicalMds::new(2)
                    .with_metric(DistanceMetric::Precomputed)
                    .fit(&d)?;
                (Some(cmds.embedding), 1)
            }
        };
        let normalized = self.normalized_stress.unwrap_or(!self.metric_mds);
        let mut rng = RandomState::new(self.seed);
        let mut best: Option<Run> = None;
        for _ in 0..n_init {
            let run = smacof_single(
                &d,
                self.metric_mds,
                self.n_components,
                init.as_ref(),
                self.max_iter,
                self.eps,
                &mut rng,
                normalized,
            )?;
            if best.as_ref().is_none_or(|b| run.stress < b.stress) {
                best = Some(run);
            }
        }
        let best = best.expect("n_init >= 1");
        Ok(FittedMds {
            embedding: best.x,
            stress: best.stress,
            n_iter: best.n_iter,
            dissimilarity: d,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn recovers_a_planar_configuration() {
        // Points in the plane: metric MDS in 2-D reproduces their distances.
        let x = array![[0.0, 0.0], [3.0, 0.0], [0.0, 4.0], [3.0, 4.0], [1.0, 2.0]];
        let fit = Mds {
            max_iter: 3000,
            eps: 1e-12,
            ..Default::default()
        }
        .fit(&x)
        .unwrap();
        let d0 = euclidean_distances(&x);
        let d1 = euclidean_distances(&fit.embedding);
        assert!((&d0 - &d1).mapv(f64::abs).iter().all(|&v| v < 1e-4));
        assert!(fit.stress < 1e-6);
    }
}
