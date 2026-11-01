//! Binary indicators for missing values.
//!
//! Mirrors `sklearn.impute.MissingIndicator` (sklearn 1.8) with
//! `missing_values = NaN` and dense output.

use anofox_ml_core::{FitUnsupervised, Float, Result, RustMlError, Transform};
use ndarray::Array2;

/// Which features get an indicator column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum MissingFeatures {
    /// Only the features that had missing values during fit (sklearn's
    /// `'missing-only'`).
    #[default]
    MissingOnly,
    /// Every feature (`'all'`).
    All,
}

/// Parameters for `MissingIndicator` (unfitted state).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MissingIndicator {
    /// Which features get an indicator column.
    #[serde(default)]
    pub features: MissingFeatures,
    /// With [`MissingFeatures::MissingOnly`], error at transform time when a
    /// feature without missing values during fit has some.
    #[serde(default = "default_true")]
    pub error_on_new: bool,
}

fn default_true() -> bool {
    true
}

impl Default for MissingIndicator {
    fn default() -> Self {
        Self {
            features: MissingFeatures::MissingOnly,
            error_on_new: true,
        }
    }
}

impl MissingIndicator {
    /// Create a `MissingIndicator` with sklearn's defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set which features get an indicator column.
    pub fn with_features(mut self, features: MissingFeatures) -> Self {
        self.features = features;
        self
    }

    /// Set whether new missing features raise an error at transform time.
    pub fn with_error_on_new(mut self, error_on_new: bool) -> Self {
        self.error_on_new = error_on_new;
        self
    }
}

/// Fitted `MissingIndicator`: the indices of the indicated features.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedMissingIndicator {
    features: Vec<usize>,
    n_features_in: usize,
    missing_only: bool,
    error_on_new: bool,
}

impl FittedMissingIndicator {
    /// Indices of the features that get an indicator column (`features_`).
    pub fn features(&self) -> &[usize] {
        &self.features
    }
}

impl<F: Float> FitUnsupervised<F> for MissingIndicator {
    type Fitted = FittedMissingIndicator;

    fn fit(&self, x: &Array2<F>) -> Result<Self::Fitted> {
        if x.is_empty() {
            return Err(RustMlError::EmptyInput("input array is empty".into()));
        }
        let features = match self.features {
            MissingFeatures::All => (0..x.ncols()).collect(),
            MissingFeatures::MissingOnly => (0..x.ncols())
                .filter(|&j| x.column(j).iter().any(|v| v.is_nan()))
                .collect(),
        };
        Ok(FittedMissingIndicator {
            features,
            n_features_in: x.ncols(),
            missing_only: self.features == MissingFeatures::MissingOnly,
            error_on_new: self.error_on_new,
        })
    }
}

impl<F: Float> Transform<F> for FittedMissingIndicator {
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>> {
        if x.ncols() != self.n_features_in {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features_in,
                x.ncols()
            )));
        }
        if self.missing_only && self.error_on_new {
            let new: Vec<usize> = (0..x.ncols())
                .filter(|j| self.features.binary_search(j).is_err())
                .filter(|&j| x.column(j).iter().any(|v| v.is_nan()))
                .collect();
            if !new.is_empty() {
                return Err(RustMlError::InvalidParameter(format!(
                    "the features {new:?} have missing values in transform but have no \
                     missing values in fit"
                )));
            }
        }
        Ok(Array2::from_shape_fn(
            (x.nrows(), self.features.len()),
            |(i, c)| {
                if x[[i, self.features[c]]].is_nan() {
                    F::one()
                } else {
                    F::zero()
                }
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_matches_sklearn() {
        // sklearn: MissingIndicator().fit(X).features_ -> [0, 2]
        let x = array![[f64::NAN, 1.0, 3.0], [4.0, 0.0, f64::NAN], [8.0, 1.0, 0.0]];
        let f = FitUnsupervised::<f64>::fit(&MissingIndicator::new(), &x).unwrap();
        assert_eq!(f.features(), &[0, 2]);
        let t = f.transform(&x).unwrap();
        assert_eq!(t, array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]);
        // A new missing feature is an error unless error_on_new = false.
        let x2 = array![[1.0, f64::NAN, 2.0]];
        assert!(f.transform(&x2).is_err());
        let f = FitUnsupervised::<f64>::fit(&MissingIndicator::new().with_error_on_new(false), &x)
            .unwrap();
        assert_eq!(f.transform(&x2).unwrap(), array![[0.0, 0.0]]);
        // features='all'
        let f = FitUnsupervised::<f64>::fit(
            &MissingIndicator::new().with_features(MissingFeatures::All),
            &x,
        )
        .unwrap();
        assert_eq!(f.transform(&x2).unwrap(), array![[0.0, 1.0, 0.0]]);
    }
}
