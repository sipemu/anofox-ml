//! Shared sampling for [`BaggingClassifier`](crate::BaggingClassifier) and
//! [`BaggingRegressor`](crate::BaggingRegressor) (sklearn's
//! `_generate_bagging_indices`).

use std::borrow::Cow;

use anofox_ml_core::{Float, Result, RustMlError};
use anofox_ml_trees::MaxFeatures;
use ndarray::{Array2, Axis};
use rand::rngs::StdRng;
use rand::seq::index::sample;
use rand::{Rng, SeedableRng};

/// What one estimator of the ensemble is trained on.
pub(crate) struct BagPlan {
    /// Training rows; repeated rows when bootstrapping.
    pub rows: Vec<usize>,
    /// Sorted feature subset; empty means all features.
    pub features: Vec<usize>,
    /// Seed of the estimator itself.
    pub seed: u64,
}

/// Sampling settings shared by both bagging estimators.
pub(crate) struct BagSampling {
    pub n_estimators: usize,
    pub max_samples: Option<f64>,
    pub bootstrap: bool,
    pub max_features: Option<MaxFeatures>,
    pub bootstrap_features: bool,
    pub oob_score: bool,
    pub seed: u64,
}

impl BagSampling {
    /// Validate the settings and draw every estimator's rows and features
    /// (sequentially, so the draws do not depend on thread scheduling).
    pub fn plans(&self, n_samples: usize, n_features: usize) -> Result<Vec<BagPlan>> {
        if self.n_estimators == 0 {
            return Err(RustMlError::InvalidParameter(
                "n_estimators must be > 0".into(),
            ));
        }
        if self.oob_score && !self.bootstrap {
            return Err(RustMlError::InvalidParameter(
                "out-of-bag estimation needs bootstrap = true".into(),
            ));
        }
        let draw_size = match self.max_samples {
            Some(frac) if frac <= 0.0 || frac > 1.0 => {
                return Err(RustMlError::InvalidParameter(
                    "max_samples must be in (0, 1]".into(),
                ))
            }
            Some(frac) => ((n_samples as f64 * frac).ceil() as usize).max(1),
            None => n_samples,
        };
        let n_feat_draw = match self.max_features {
            None => n_features,
            Some(MaxFeatures::Fraction(f)) if !(f > 0.0 && f <= 1.0) => {
                return Err(RustMlError::InvalidParameter(
                    "max_features fraction must be in (0, 1]".into(),
                ))
            }
            Some(MaxFeatures::Fixed(k)) if k == 0 || k > n_features => {
                return Err(RustMlError::InvalidParameter(format!(
                    "max_features must be in [1, {n_features}], got {k}"
                )))
            }
            Some(mf) => mf.resolve(n_features).clamp(1, n_features),
        };
        let subset_features = n_feat_draw < n_features || self.bootstrap_features;

        let mut rng = StdRng::seed_from_u64(self.seed);
        Ok((0..self.n_estimators)
            .map(|_| {
                let mut features: Vec<usize> = if !subset_features {
                    Vec::new()
                } else if self.bootstrap_features {
                    (0..n_feat_draw)
                        .map(|_| rng.gen_range(0..n_features))
                        .collect()
                } else {
                    sample(&mut rng, n_features, n_feat_draw).into_vec()
                };
                features.sort_unstable();
                let rows = if self.bootstrap {
                    (0..draw_size)
                        .map(|_| rng.gen_range(0..n_samples))
                        .collect()
                } else if draw_size < n_samples {
                    let mut r = sample(&mut rng, n_samples, draw_size).into_vec();
                    r.sort_unstable();
                    r
                } else {
                    (0..n_samples).collect()
                };
                BagPlan {
                    rows,
                    features,
                    seed: rng.gen(),
                }
            })
            .collect())
    }
}

/// The columns `features` of `x` (all of `x` when `features` is empty).
pub(crate) fn select_columns<'a, F: Float>(
    x: &'a Array2<F>,
    features: &[usize],
) -> Cow<'a, Array2<F>> {
    if features.is_empty() {
        Cow::Borrowed(x)
    } else {
        Cow::Owned(x.select(Axis(1), features))
    }
}

/// Rows not drawn into `rows` (the estimator's out-of-bag samples).
pub(crate) fn out_of_bag(n_samples: usize, rows: &[usize]) -> Vec<usize> {
    let mut in_bag = vec![false; n_samples];
    for &r in rows {
        in_bag[r] = true;
    }
    (0..n_samples).filter(|&i| !in_bag[i]).collect()
}

/// Add an estimator's feature importances (over `features`, or all features
/// when empty) into the full-width `acc`.
pub(crate) fn add_importances<F: Float>(
    acc: &mut [F],
    imp: &ndarray::Array1<F>,
    features: &[usize],
) {
    if features.is_empty() {
        for (a, &v) in acc.iter_mut().zip(imp) {
            *a += v;
        }
    } else {
        for (&f, &v) in features.iter().zip(imp) {
            acc[f] += v;
        }
    }
}
