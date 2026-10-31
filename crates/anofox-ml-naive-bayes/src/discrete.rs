//! Count accumulation shared by the discrete Naive Bayes models
//! (Multinomial, Bernoulli, Complement), mirroring sklearn's
//! `_BaseDiscreteNB.fit` / `partial_fit` + `_count`.

use anofox_ml_core::{Float, Result, RustMlError};
use ndarray::{Array1, Array2, Axis};

use crate::common::{
    check_n_features, check_partial_fit_classes, check_xy, encode_labels, sorted_unique,
    weighted_one_hot,
};

/// Raw (weighted) class and feature counts.
#[derive(Debug, Clone)]
pub(crate) struct Counts<F: Float> {
    pub classes: Vec<F>,
    pub class_count: Array1<F>,
    pub feature_count: Array2<F>,
}

impl<F: Float> Counts<F> {
    /// Rebuild the counts of a previously fitted model. Fails for models
    /// serialised before raw counts were stored.
    pub fn from_parts(
        classes: Vec<F>,
        class_count: Array1<F>,
        feature_count: Array2<F>,
    ) -> Result<Self> {
        if class_count.len() != classes.len() || feature_count.nrows() != classes.len() {
            return Err(RustMlError::InvalidParameter(
                "model state has no raw counts (serialised by an older version); \
                 cannot continue partial_fit"
                    .into(),
            ));
        }
        Ok(Self {
            classes,
            class_count,
            feature_count,
        })
    }
}

/// Add a batch to `state` (or start fresh when `state` is `None`).
///
/// `x` must already be transformed into the counted representation (e.g.
/// binarised for BernoulliNB). Labels not in the class set are an error.
pub(crate) fn accumulate<F: Float>(
    state: Option<Counts<F>>,
    x: &Array2<F>,
    y: &Array1<F>,
    classes: Option<&[F]>,
    sample_weight: Option<&Array1<F>>,
) -> Result<Counts<F>> {
    check_xy(x, y, sample_weight)?;
    let mut counts = match state {
        Some(s) => {
            check_n_features(x, s.feature_count.ncols())?;
            check_partial_fit_classes(&s.classes, classes)?;
            s
        }
        None => {
            let classes = match classes {
                Some(c) => sorted_unique(c),
                None => sorted_unique(&y.to_vec()),
            };
            let k = classes.len();
            Counts {
                classes,
                class_count: Array1::zeros(k),
                feature_count: Array2::zeros((k, x.ncols())),
            }
        }
    };
    let idx = encode_labels(y, &counts.classes)?;
    let y1h = weighted_one_hot(&idx, counts.classes.len(), sample_weight);
    counts.feature_count = &counts.feature_count + &y1h.t().dot(x);
    counts.class_count = &counts.class_count + &y1h.sum_axis(Axis(0));
    Ok(counts)
}

/// Error if any entry of `x` is negative (sklearn's `check_non_negative`).
pub(crate) fn check_non_negative<F: Float>(x: &Array2<F>, model: &str) -> Result<()> {
    if x.iter().any(|v| *v < F::zero()) {
        return Err(RustMlError::InvalidParameter(format!(
            "{model} requires all feature values to be non-negative"
        )));
    }
    Ok(())
}
