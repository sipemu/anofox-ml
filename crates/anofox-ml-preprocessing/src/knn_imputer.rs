//! k-nearest-neighbour imputation.
//!
//! Mirrors `sklearn.impute.KNNImputer` (sklearn 1.8) with
//! `missing_values = NaN` and `metric = 'nan_euclidean'`: a missing entry of
//! feature `j` is the (optionally distance-weighted) mean of feature `j` over
//! the `n_neighbors` closest training rows that have feature `j`. Distances
//! between rows with missing coordinates use only the coordinates present in
//! both, scaled up by `n_features / n_present` (sklearn's
//! `nan_euclidean_distances`). A row with no usable distance to any donor
//! gets the feature's training mean.
//!
//! The distances are computed by brute force, `O(n_missing_rows · n_train ·
//! n_features)`, in parallel over the rows to impute.

use anofox_ml_core::{FitUnsupervised, Float, Result, RustMlError, Transform};
use ndarray::{Array2, Axis};
use rayon::prelude::*;

use crate::missing_indicator::{FittedMissingIndicator, MissingIndicator};

/// Weighting of the donors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum KnnImputeWeights {
    /// Plain mean of the donors (`'uniform'`).
    #[default]
    Uniform,
    /// Weight donors by inverse distance (`'distance'`); donors at distance 0
    /// take all the weight.
    Distance,
}

/// Parameters for `KnnImputer` (unfitted state).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KnnImputer {
    /// Number of donors per missing entry.
    pub n_neighbors: usize,
    /// Donor weighting.
    #[serde(default)]
    pub weights: KnnImputeWeights,
    /// Append a [`MissingIndicator`] to the output.
    #[serde(default)]
    pub add_indicator: bool,
    /// Keep features that are entirely missing at fit time (imputed with 0).
    #[serde(default)]
    pub keep_empty_features: bool,
}

impl Default for KnnImputer {
    fn default() -> Self {
        Self {
            n_neighbors: 5,
            weights: KnnImputeWeights::Uniform,
            add_indicator: false,
            keep_empty_features: false,
        }
    }
}

impl KnnImputer {
    /// Impute from `n_neighbors` donors.
    pub fn new(n_neighbors: usize) -> Self {
        Self {
            n_neighbors,
            ..Self::default()
        }
    }

    /// Set the donor weighting.
    pub fn with_weights(mut self, weights: KnnImputeWeights) -> Self {
        self.weights = weights;
        self
    }

    /// Append missing-value indicator columns.
    pub fn with_add_indicator(mut self, add_indicator: bool) -> Self {
        self.add_indicator = add_indicator;
        self
    }

    /// Keep all-missing features (imputed with 0).
    pub fn with_keep_empty_features(mut self, keep: bool) -> Self {
        self.keep_empty_features = keep;
        self
    }
}

/// Fitted `KnnImputer`: the training data are the donor pool.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedKnnImputer<F: Float> {
    fit_x: Array2<F>,
    /// Feature has at least one observed value at fit time.
    valid: Vec<bool>,
    /// Training mean of each feature (NaN-ignoring), the fallback value.
    col_mean: Vec<F>,
    n_neighbors: usize,
    weights: KnnImputeWeights,
    keep_empty_features: bool,
    indicator: Option<FittedMissingIndicator>,
}

impl<F: Float> FitUnsupervised<F> for KnnImputer {
    type Fitted = FittedKnnImputer<F>;

    fn fit(&self, x: &Array2<F>) -> Result<Self::Fitted> {
        if x.is_empty() {
            return Err(RustMlError::EmptyInput("input array is empty".into()));
        }
        if self.n_neighbors == 0 {
            return Err(RustMlError::InvalidParameter(
                "n_neighbors must be at least 1".into(),
            ));
        }
        if x.iter().any(|v| v.is_infinite()) {
            return Err(RustMlError::InvalidParameter(
                "KnnImputer input contains infinite values".into(),
            ));
        }
        let d = x.ncols();
        let mut valid = vec![false; d];
        let mut col_mean = vec![F::nan(); d];
        for j in 0..d {
            let (mut sum, mut n) = (F::zero(), 0usize);
            for &v in x.column(j) {
                if !v.is_nan() {
                    sum += v;
                    n += 1;
                }
            }
            if n > 0 {
                valid[j] = true;
                col_mean[j] = sum / F::from_usize(n).unwrap();
            }
        }
        let indicator = if self.add_indicator {
            let mi = MissingIndicator::new().with_error_on_new(false);
            Some(FitUnsupervised::<F>::fit(&mi, x)?)
        } else {
            None
        };
        Ok(FittedKnnImputer {
            fit_x: x.to_owned(),
            valid,
            col_mean,
            n_neighbors: self.n_neighbors,
            weights: self.weights,
            keep_empty_features: self.keep_empty_features,
            indicator,
        })
    }
}

/// sklearn's `nan_euclidean_distances` for one pair of rows: `NaN` when no
/// coordinate is present in both.
fn nan_euclidean<F: Float>(a: ndarray::ArrayView1<F>, b: ndarray::ArrayView1<F>) -> F {
    let mut sq = F::zero();
    let mut present = 0usize;
    for (&u, &v) in a.iter().zip(b.iter()) {
        if !u.is_nan() && !v.is_nan() {
            let t = u - v;
            sq += t * t;
            present += 1;
        }
    }
    if present == 0 {
        return F::nan();
    }
    let scale = F::from_usize(a.len()).unwrap() / F::from_usize(present).unwrap();
    (sq * scale).sqrt()
}

impl<F: Float> FittedKnnImputer<F> {
    /// sklearn's `_calc_impute`: the `k` donors with the smallest distance
    /// (NaN distances last; ties by donor order), averaged with uniform or
    /// inverse-distance weights. NaN distances get weight 0.
    fn calc_impute(&self, cand: &mut [(F, usize)], j: usize) -> F {
        let k = self.n_neighbors.min(cand.len());
        cand.sort_by(|a, b| {
            a.0.is_nan()
                .cmp(&b.0.is_nan())
                .then_with(|| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| a.1.cmp(&b.1))
        });
        let donors = &cand[..k];
        let w: Vec<F> = match self.weights {
            KnnImputeWeights::Uniform => donors
                .iter()
                .map(|p| if p.0.is_nan() { F::zero() } else { F::one() })
                .collect(),
            KnnImputeWeights::Distance => {
                if donors.iter().any(|p| p.0 == F::zero()) {
                    donors
                        .iter()
                        .map(|p| {
                            if p.0 == F::zero() {
                                F::one()
                            } else {
                                F::zero()
                            }
                        })
                        .collect()
                } else {
                    donors
                        .iter()
                        .map(|p| {
                            if p.0.is_nan() {
                                F::zero()
                            } else {
                                F::one() / p.0
                            }
                        })
                        .collect()
                }
            }
        };
        let (mut num, mut den) = (F::zero(), F::zero());
        for (p, &wi) in donors.iter().zip(&w) {
            num += wi * self.fit_x[[p.1, j]];
            den += wi;
        }
        num / den
    }

    /// Indices of the features kept in the output.
    fn kept(&self) -> Vec<usize> {
        (0..self.valid.len())
            .filter(|&j| self.valid[j] || self.keep_empty_features)
            .collect()
    }
}

impl<F: Float> Transform<F> for FittedKnnImputer<F> {
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>> {
        let d = self.valid.len();
        if x.ncols() != d {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {d} features, got {}",
                x.ncols()
            )));
        }
        let n_fit = self.fit_x.nrows();
        let mut out = x.to_owned();
        let rows: Vec<usize> = (0..x.nrows())
            .filter(|&i| (0..d).any(|j| self.valid[j] && x[[i, j]].is_nan()))
            .collect();

        let imputed: Vec<Vec<(usize, F)>> = rows
            .par_iter()
            .map(|&i| {
                let xi = x.row(i);
                let dist: Vec<F> = (0..n_fit)
                    .map(|r| nan_euclidean(xi, self.fit_x.row(r)))
                    .collect();
                let mut vals = Vec::new();
                for j in (0..d).filter(|&j| self.valid[j] && xi[j].is_nan()) {
                    let mut cand: Vec<(F, usize)> = (0..n_fit)
                        .filter(|&r| !self.fit_x[[r, j]].is_nan())
                        .map(|r| (dist[r], r))
                        .collect();
                    let v = if cand.iter().all(|p| p.0.is_nan()) {
                        self.col_mean[j]
                    } else {
                        self.calc_impute(&mut cand, j)
                    };
                    vals.push((j, v));
                }
                vals
            })
            .collect();
        for (&i, vals) in rows.iter().zip(imputed) {
            for (j, v) in vals {
                out[[i, j]] = v;
            }
        }

        let kept = self.kept();
        let mut out = if kept.len() < d {
            out.select(Axis(1), &kept)
        } else {
            out
        };
        if self.keep_empty_features {
            // All-missing features are imputed with 0.
            for (c, &j) in kept.iter().enumerate() {
                if !self.valid[j] {
                    out.column_mut(c).fill(F::zero());
                }
            }
        }
        match &self.indicator {
            Some(ind) => {
                let mask = ind.transform(x)?;
                ndarray::concatenate(Axis(1), &[out.view(), mask.view()])
                    .map_err(|e| RustMlError::ShapeMismatch(e.to_string()))
            }
            None => Ok(out),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    fn assert_close(a: &Array2<f64>, b: &Array2<f64>) {
        assert_eq!(a.dim(), b.dim());
        for (u, v) in a.iter().zip(b.iter()) {
            assert_abs_diff_eq!(*u, *v, epsilon = 1e-12);
        }
    }

    fn data() -> Array2<f64> {
        let n = f64::NAN;
        array![
            [1.0, 2.0, n, 4.0],
            [3.0, 4.0, 3.0, n],
            [n, 6.0, 5.0, 8.0],
            [8.0, 8.0, 7.0, 1.0],
            [2.5, n, 1.0, 0.5],
            [6.0, 1.5, 2.0, 3.5],
            [0.0, 9.0, n, n],
            [4.0, 3.0, 6.5, 2.0]
        ]
    }

    #[test]
    fn test_nan_euclidean_matches_sklearn() {
        // sklearn: nan_euclidean_distances([[3, nan, nan, 6]], [[1, nan, 4, 5]])
        //   -> sqrt(4/2 * ((3-1)^2 + (6-5)^2)) = sqrt(10)
        let a = array![3.0, f64::NAN, f64::NAN, 6.0];
        let b = array![1.0, f64::NAN, 4.0, 5.0];
        assert_abs_diff_eq!(
            nan_euclidean(a.view(), b.view()),
            10f64.sqrt(),
            epsilon = 1e-15
        );
        let c = array![f64::NAN, 1.0, 2.0, f64::NAN];
        assert!(nan_euclidean(a.view(), c.view()).is_nan());
    }

    #[test]
    fn test_uniform_matches_sklearn() {
        // sklearn 1.8: KNNImputer(n_neighbors=2).fit_transform(X)
        let x = data();
        let f = FitUnsupervised::<f64>::fit(&KnnImputer::new(2), &x).unwrap();
        let expected = sklearn_uniform();
        assert_close(&f.transform(&x).unwrap(), &expected);
    }

    #[test]
    fn test_distance_matches_sklearn() {
        // sklearn 1.8: KNNImputer(n_neighbors=3, weights='distance')
        let x = data();
        let imp = KnnImputer::new(3).with_weights(KnnImputeWeights::Distance);
        let f = FitUnsupervised::<f64>::fit(&imp, &x).unwrap();
        assert_close(&f.transform(&x).unwrap(), &sklearn_distance());
        // New data with its own missing pattern.
        let q = array![[f64::NAN, 5.0, f64::NAN, 2.0]];
        assert_close(&f.transform(&q).unwrap(), &sklearn_distance_new());
    }

    #[test]
    fn test_empty_feature_and_indicator() {
        // sklearn: KNNImputer(add_indicator=True) drops the all-NaN column
        // and appends indicators for the features with missing values.
        let n = f64::NAN;
        let x = array![[1.0, n, 2.0], [n, n, 4.0], [3.0, n, 6.0]];
        let f =
            FitUnsupervised::<f64>::fit(&KnnImputer::new(1).with_add_indicator(true), &x).unwrap();
        assert_close(
            &f.transform(&x).unwrap(),
            &array![
                [1.0, 2.0, 0.0, 1.0],
                [1.0, 4.0, 1.0, 1.0],
                [3.0, 6.0, 0.0, 1.0]
            ],
        );
        let f = FitUnsupervised::<f64>::fit(&KnnImputer::new(1).with_keep_empty_features(true), &x)
            .unwrap();
        assert_close(
            &f.transform(&x).unwrap(),
            &array![[1.0, 0.0, 2.0], [1.0, 0.0, 4.0], [3.0, 0.0, 6.0]],
        );
    }

    fn sklearn_uniform() -> Array2<f64> {
        array![
            [1.0, 2.0, 4.75, 4.0],
            [3.0, 4.0, 3.0, 2.25],
            [1.5, 6.0, 5.0, 8.0],
            [8.0, 8.0, 7.0, 1.0],
            [2.5, 6.5, 1.0, 0.5],
            [6.0, 1.5, 2.0, 3.5],
            [0.0, 9.0, 3.0, 4.25],
            [4.0, 3.0, 6.5, 2.0]
        ]
    }

    fn sklearn_distance() -> Array2<f64> {
        array![
            [1.0, 2.0, 3.657_575_765_821_542, 4.0],
            [3.0, 4.0, 3.0, 3.762_169_906_300_188_3],
            [2.310_773_947_715_987, 6.0, 5.0, 8.0],
            [8.0, 8.0, 7.0, 1.0],
            [2.5, 4.862_653_748_111_94, 1.0, 0.5],
            [6.0, 1.5, 2.0, 3.5],
            [0.0, 9.0, 2.863_369_633_607_275_6, 3.928_571_428_571_429],
            [4.0, 3.0, 6.5, 2.0]
        ]
    }

    fn sklearn_distance_new() -> Array2<f64> {
        array![[3.157_459_612_749_163_6, 5.0, 3.480_896_945_684_503, 2.0]]
    }
}
