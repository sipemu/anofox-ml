use anofox_ml_core::{
    FitUnsupervised, FitUnsupervisedWeighted, Float, Result, RustMlError, Transform,
};
use ndarray::{Array1, Array2, Axis};

use crate::missing_indicator::{FittedMissingIndicator, MissingIndicator};
use std::collections::HashMap;

/// Strategy used to compute the fill value for missing (NaN) entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ImputeStrategy {
    /// Replace NaN with the column mean (computed from non-NaN values).
    Mean,
    /// Replace NaN with the column median (computed from non-NaN values).
    Median,
    /// Replace NaN with the most frequent value in the column.
    MostFrequent,
    /// Replace NaN with a fixed `fill_value`.
    Constant,
}

/// Parameters for `SimpleImputer` (unfitted state).
///
/// Fills missing values (`NaN`) in each column with a statistic computed from
/// the non-missing values during [`FitUnsupervised::fit`]. Mirrors
/// `sklearn.impute.SimpleImputer` (sklearn 1.8) with `missing_values = NaN`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct SimpleImputer<F: Float> {
    strategy: ImputeStrategy,
    fill_value: Option<F>,
    /// Append a [`MissingIndicator`] (features with missing values at fit
    /// time) to the output.
    #[serde(default)]
    add_indicator: bool,
    /// Keep features that are entirely missing at fit time (imputed with 0
    /// unless the strategy is `Constant`). By default they are dropped.
    #[serde(default)]
    keep_empty_features: bool,
}

impl<F: Float> SimpleImputer<F> {
    /// Create a new `SimpleImputer` with the default [`ImputeStrategy::Mean`].
    pub fn new() -> Self {
        Self {
            strategy: ImputeStrategy::Mean,
            fill_value: None,
            add_indicator: false,
            keep_empty_features: false,
        }
    }

    /// Set the imputation strategy.
    pub fn with_strategy(mut self, strategy: ImputeStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set the fill value used when strategy is [`ImputeStrategy::Constant`].
    pub fn with_fill_value(mut self, value: F) -> Self {
        self.fill_value = Some(value);
        self
    }

    /// Append missing-value indicator columns (sklearn's `add_indicator`).
    pub fn with_add_indicator(mut self, add_indicator: bool) -> Self {
        self.add_indicator = add_indicator;
        self
    }

    /// Keep all-missing features (sklearn's `keep_empty_features`).
    pub fn with_keep_empty_features(mut self, keep: bool) -> Self {
        self.keep_empty_features = keep;
        self
    }
}

impl<F: Float> Default for SimpleImputer<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted `SimpleImputer` — holds one fill value per column.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedSimpleImputer<F: Float> {
    fill_values: Array1<F>,
    /// Columns kept in the output (all but the dropped empty features).
    #[serde(default)]
    keep: Option<Vec<usize>>,
    #[serde(default)]
    indicator: Option<FittedMissingIndicator>,
}

impl<F: Float> FittedSimpleImputer<F> {
    /// Return the per-column fill values learned during fit (sklearn's
    /// `statistics_`; `NaN` for a dropped all-missing feature).
    pub fn fill_values(&self) -> &Array1<F> {
        &self.fill_values
    }

    /// The fitted missing-value indicator when `add_indicator` is set.
    pub fn indicator(&self) -> Option<&FittedMissingIndicator> {
        self.indicator.as_ref()
    }
}

/// Weighted mean of the non-NaN values. `None` if all are NaN (or carry no
/// weight).
fn column_mean<F: Float>(values: &[F], w: &[F]) -> Option<F> {
    let mut sum = F::zero();
    let mut total = F::zero();
    for (&v, &wi) in values.iter().zip(w) {
        if !v.is_nan() {
            sum += v * wi;
            total += wi;
        }
    }
    (total > F::zero()).then(|| sum / total)
}

/// Weighted median of the non-NaN values: the average of the lower and upper
/// weighted medians (smallest value whose cumulative weight reaches half the
/// total, from below and from above). With unit weights this is the ordinary
/// median, the mean of the two middle values for even counts.
fn column_median<F: Float>(values: &[F], w: &[F]) -> Option<F> {
    let mut valid: Vec<(F, F)> = values
        .iter()
        .zip(w)
        .filter(|(v, wi)| !v.is_nan() && **wi > F::zero())
        .map(|(&v, &wi)| (v, wi))
        .collect();
    if valid.is_empty() {
        return None;
    }
    valid.sort_by(|a, b| crate::quantile::total_cmp(&a.0, &b.0));
    let half = valid.iter().fold(F::zero(), |acc, p| acc + p.1) / F::from_f64(2.0).unwrap();
    let mut cum = F::zero();
    let lower = valid
        .iter()
        .find(|p| {
            cum += p.1;
            cum >= half
        })
        .map(|p| p.0)
        .unwrap_or(valid[valid.len() - 1].0);
    let mut cum = F::zero();
    let upper = valid
        .iter()
        .rev()
        .find(|p| {
            cum += p.1;
            cum >= half
        })
        .map(|p| p.0)
        .unwrap_or(valid[0].0);
    Some((lower + upper) / F::from_f64(2.0).unwrap())
}

/// Most frequent non-NaN value (largest total weight). `None` if all are NaN.
/// Ties are broken by taking the smallest value, as sklearn.
fn column_most_frequent<F: Float>(values: &[F], w: &[F]) -> Option<F> {
    let mut counts: HashMap<u64, (F, F)> = HashMap::new();
    for (&v, &wi) in values.iter().zip(w) {
        if v.is_nan() || wi <= F::zero() {
            continue;
        }
        // Use bit representation as hash key for exact equality.
        let bits = v.to_f64().unwrap().to_bits();
        counts
            .entry(bits)
            .and_modify(|e| e.1 += wi)
            .or_insert((v, wi));
    }
    // Pick highest count, break ties by smallest value.
    counts
        .values()
        .max_by(|a, b| {
            crate::quantile::total_cmp(&a.1, &b.1)
                .then_with(|| crate::quantile::total_cmp(&b.0, &a.0))
        })
        .map(|&(v, _)| v)
}

impl<F: Float> FitUnsupervised<F> for SimpleImputer<F> {
    type Fitted = FittedSimpleImputer<F>;

    fn fit(&self, x: &Array2<F>) -> Result<Self::Fitted> {
        self.fit_unsupervised_weighted(x, None)
    }
}

impl<F: Float> FitUnsupervisedWeighted<F> for SimpleImputer<F> {
    type Fitted = FittedSimpleImputer<F>;

    /// Fit with per-sample weights: weighted mean, weighted median and
    /// weighted mode (an extension; sklearn's `SimpleImputer` has no
    /// `sample_weight`). Unit weights give the unweighted statistics.
    fn fit_unsupervised_weighted(
        &self,
        x: &Array2<F>,
        sample_weight: Option<&Array1<F>>,
    ) -> Result<Self::Fitted> {
        if x.is_empty() {
            return Err(RustMlError::EmptyInput("input array is empty".into()));
        }
        let ones;
        let w: &[F] = match sample_weight {
            Some(w) => {
                if w.len() != x.nrows() {
                    return Err(RustMlError::ShapeMismatch(format!(
                        "sample_weight has {} entries, X has {} rows",
                        w.len(),
                        x.nrows()
                    )));
                }
                if w.iter().any(|&v| v < F::zero() || !v.is_finite()) {
                    return Err(RustMlError::InvalidParameter(
                        "sample_weight must be finite and non-negative".into(),
                    ));
                }
                w.as_slice().expect("1-D array is contiguous")
            }
            None => {
                ones = vec![F::one(); x.nrows()];
                &ones
            }
        };

        let ncols = x.ncols();
        let mut fill_values = Array1::<F>::zeros(ncols);
        let mut keep = Vec::with_capacity(ncols);
        for j in 0..ncols {
            let col: Vec<F> = x.column(j).to_vec();
            let computed = match self.strategy {
                ImputeStrategy::Mean => column_mean(&col, w),
                ImputeStrategy::Median => column_median(&col, w),
                ImputeStrategy::MostFrequent => column_most_frequent(&col, w),
                ImputeStrategy::Constant => Some(self.fill_value.unwrap_or_else(F::zero)),
            };
            match computed {
                Some(v) => {
                    fill_values[j] = v;
                    keep.push(j);
                }
                // All missing: sklearn drops the feature, or imputes 0 with
                // keep_empty_features=True.
                None if self.keep_empty_features => keep.push(j),
                None => fill_values[j] = F::nan(),
            }
        }

        let indicator = if self.add_indicator {
            let mi = MissingIndicator::new().with_error_on_new(false);
            Some(FitUnsupervised::<F>::fit(&mi, x)?)
        } else {
            None
        };
        Ok(FittedSimpleImputer {
            fill_values,
            keep: (keep.len() < ncols).then_some(keep),
            indicator,
        })
    }
}

impl<F: Float> Transform<F> for FittedSimpleImputer<F> {
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>> {
        if x.ncols() != self.fill_values.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.fill_values.len(),
                x.ncols()
            )));
        }

        let mut result = match &self.keep {
            Some(keep) => x.select(Axis(1), keep),
            None => x.to_owned(),
        };
        let cols: Vec<usize> = match &self.keep {
            Some(keep) => keep.clone(),
            None => (0..x.ncols()).collect(),
        };
        for mut row in result.rows_mut() {
            for (val, &j) in row.iter_mut().zip(&cols) {
                if val.is_nan() {
                    *val = self.fill_values[j];
                }
            }
        }
        match &self.indicator {
            Some(ind) => {
                let mask = ind.transform(x)?;
                ndarray::concatenate(Axis(1), &[result.view(), mask.view()])
                    .map_err(|e| RustMlError::ShapeMismatch(e.to_string()))
            }
            None => Ok(result),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_mean_strategy_basic() {
        let x = array![[1.0, f64::NAN], [2.0, 4.0], [3.0, 6.0],];
        let imputer = SimpleImputer::<f64>::new();
        let fitted = FitUnsupervised::<f64>::fit(&imputer, &x).unwrap();
        let result = fitted.transform(&x).unwrap();

        // Column 0: no NaN, unchanged
        assert_abs_diff_eq!(result[[0, 0]], 1.0);
        assert_abs_diff_eq!(result[[1, 0]], 2.0);
        assert_abs_diff_eq!(result[[2, 0]], 3.0);
        // Column 1: NaN replaced with mean of 4.0 and 6.0 = 5.0
        assert_abs_diff_eq!(result[[0, 1]], 5.0);
        assert_abs_diff_eq!(result[[1, 1]], 4.0);
        assert_abs_diff_eq!(result[[2, 1]], 6.0);
    }

    #[test]
    fn test_median_strategy() {
        let x = array![[f64::NAN, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0],];
        let imputer = SimpleImputer::<f64>::new().with_strategy(ImputeStrategy::Median);
        let fitted = FitUnsupervised::<f64>::fit(&imputer, &x).unwrap();
        let result = fitted.transform(&x).unwrap();

        // Column 0: valid = [2, 4, 6], median = 4.0
        assert_abs_diff_eq!(result[[0, 0]], 4.0);
        // Column 1: no NaN
        assert_abs_diff_eq!(result[[0, 1]], 1.0);
    }

    #[test]
    fn test_most_frequent_strategy() {
        let x = array![[1.0, f64::NAN], [2.0, 3.0], [2.0, 3.0], [3.0, 5.0],];
        let imputer = SimpleImputer::<f64>::new().with_strategy(ImputeStrategy::MostFrequent);
        let fitted = FitUnsupervised::<f64>::fit(&imputer, &x).unwrap();
        let result = fitted.transform(&x).unwrap();

        // Column 0: most frequent = 2.0 (appears twice)
        // Column 1: most frequent among [3,3,5] = 3.0
        assert_abs_diff_eq!(result[[0, 0]], 1.0); // not NaN, unchanged
        assert_abs_diff_eq!(result[[0, 1]], 3.0); // NaN replaced with 3.0
    }

    #[test]
    fn test_constant_strategy() {
        let x = array![[f64::NAN, 1.0], [2.0, f64::NAN],];
        let imputer = SimpleImputer::<f64>::new()
            .with_strategy(ImputeStrategy::Constant)
            .with_fill_value(-999.0);
        let fitted = FitUnsupervised::<f64>::fit(&imputer, &x).unwrap();
        let result = fitted.transform(&x).unwrap();

        assert_abs_diff_eq!(result[[0, 0]], -999.0);
        assert_abs_diff_eq!(result[[0, 1]], 1.0);
        assert_abs_diff_eq!(result[[1, 0]], 2.0);
        assert_abs_diff_eq!(result[[1, 1]], -999.0);
    }

    #[test]
    fn test_mixed_nan_positions() {
        let x = array![
            [f64::NAN, 2.0, f64::NAN],
            [1.0, f64::NAN, 6.0],
            [3.0, 4.0, f64::NAN],
            [5.0, 6.0, 8.0],
        ];
        let imputer = SimpleImputer::<f64>::new();
        let fitted = FitUnsupervised::<f64>::fit(&imputer, &x).unwrap();
        let result = fitted.transform(&x).unwrap();

        // Column 0: valid = [1,3,5], mean = 3.0
        assert_abs_diff_eq!(result[[0, 0]], 3.0);
        // Column 1: valid = [2,4,6], mean = 4.0
        assert_abs_diff_eq!(result[[1, 1]], 4.0);
        // Column 2: valid = [6,8], mean = 7.0
        assert_abs_diff_eq!(result[[0, 2]], 7.0);
        assert_abs_diff_eq!(result[[2, 2]], 7.0);
        // Non-NaN values unchanged
        assert_abs_diff_eq!(result[[3, 0]], 5.0);
        assert_abs_diff_eq!(result[[3, 1]], 6.0);
        assert_abs_diff_eq!(result[[3, 2]], 8.0);
    }

    #[test]
    fn test_all_nan_column_dropped_like_sklearn() {
        // sklearn: SimpleImputer().fit_transform([[1, nan], [2, nan], [3, nan]])
        //   -> [[1], [2], [3]], statistics_ = [2, nan]; keep_empty_features=True
        //   -> second column imputed with 0.
        let x = array![[1.0, f64::NAN], [2.0, f64::NAN], [3.0, f64::NAN],];
        let fitted = FitUnsupervised::<f64>::fit(&SimpleImputer::<f64>::new(), &x).unwrap();
        assert_eq!(fitted.transform(&x).unwrap(), array![[1.0], [2.0], [3.0]]);
        assert!(fitted.fill_values()[1].is_nan());
        let keep = SimpleImputer::<f64>::new().with_keep_empty_features(true);
        let fitted = FitUnsupervised::<f64>::fit(&keep, &x).unwrap();
        assert_eq!(
            fitted.transform(&x).unwrap(),
            array![[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
        );
    }

    #[test]
    fn test_add_indicator_matches_sklearn() {
        // sklearn: SimpleImputer(add_indicator=True).fit(X).transform(X2)
        let x = array![[f64::NAN, 1.0, 3.0], [4.0, 0.0, f64::NAN], [8.0, 1.0, 0.0]];
        let imp = SimpleImputer::<f64>::new().with_add_indicator(true);
        let fitted = FitUnsupervised::<f64>::fit(&imp, &x).unwrap();
        assert_eq!(
            fitted.transform(&x).unwrap(),
            array![
                [6.0, 1.0, 3.0, 1.0, 0.0],
                [4.0, 0.0, 1.5, 0.0, 1.0],
                [8.0, 1.0, 0.0, 0.0, 0.0]
            ]
        );
        // A feature that was complete at fit time gets no indicator, and new
        // missing values there are simply imputed.
        let x2 = array![[1.0, f64::NAN, f64::NAN]];
        assert_eq!(
            fitted.transform(&x2).unwrap(),
            array![[1.0, 2.0 / 3.0, 1.5, 0.0, 1.0]]
        );
    }

    #[test]
    fn test_sample_weight() {
        let x = array![[1.0], [2.0], [f64::NAN], [10.0]];
        let w = array![1.0, 3.0, 5.0, 0.0];
        let fit = |s: ImputeStrategy, w: &Array1<f64>| {
            SimpleImputer::<f64>::new()
                .with_strategy(s)
                .fit_unsupervised_weighted(&x, Some(w))
                .unwrap()
                .fill_values()[0]
        };
        assert_abs_diff_eq!(fit(ImputeStrategy::Mean, &w), 1.75, epsilon = 1e-12);
        assert_abs_diff_eq!(fit(ImputeStrategy::Median, &w), 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(fit(ImputeStrategy::MostFrequent, &w), 2.0, epsilon = 1e-12);
        // Unit weights reproduce the unweighted statistics (median of
        // [1, 2, 10] is 2; even count averages the middle pair).
        let ones = Array1::ones(4);
        assert_abs_diff_eq!(fit(ImputeStrategy::Median, &ones), 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(
            fit(ImputeStrategy::Mean, &ones),
            13.0 / 3.0,
            epsilon = 1e-12
        );
        let w2 = array![1.0, 1.0, 1.0, 1.0];
        let x2 = array![[1.0], [3.0], [5.0], [7.0]];
        let m = SimpleImputer::<f64>::new()
            .with_strategy(ImputeStrategy::Median)
            .fit_unsupervised_weighted(&x2, Some(&w2))
            .unwrap();
        assert_abs_diff_eq!(m.fill_values()[0], 4.0, epsilon = 1e-12);
    }

    #[test]
    fn test_no_nan_passthrough() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],];
        let imputer = SimpleImputer::<f64>::new();
        let fitted = FitUnsupervised::<f64>::fit(&imputer, &x).unwrap();
        let result = fitted.transform(&x).unwrap();

        // Should be identical to input
        for (a, b) in x.iter().zip(result.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-15);
        }
    }

    #[test]
    fn test_shape_mismatch_on_transform() {
        let x_fit = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0],];
        let x_transform = array![[1.0, 2.0], [3.0, 4.0],];
        let imputer = SimpleImputer::<f64>::new();
        let fitted = FitUnsupervised::<f64>::fit(&imputer, &x_fit).unwrap();
        let result = fitted.transform(&x_transform);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("3") && msg.contains("2"),
            "error should mention expected and actual: {}",
            msg
        );
    }

    #[test]
    fn test_f32_support() {
        let x = array![[1.0f32, f32::NAN], [3.0f32, 4.0f32], [5.0f32, 6.0f32],];
        let imputer = SimpleImputer::<f32>::new();
        let fitted = FitUnsupervised::<f32>::fit(&imputer, &x).unwrap();
        let result = fitted.transform(&x).unwrap();

        // Column 1: mean of 4.0 and 6.0 = 5.0
        assert_abs_diff_eq!(result[[0, 1]], 5.0f32, epsilon = 1e-6);
        // Non-NaN unchanged
        assert_abs_diff_eq!(result[[0, 0]], 1.0f32, epsilon = 1e-6);
    }

    #[test]
    fn test_constant_strategy_default_fill_value() {
        // When Constant strategy is used without specifying fill_value, default to 0.
        let x = array![[f64::NAN, 1.0], [2.0, f64::NAN],];
        let imputer = SimpleImputer::<f64>::new().with_strategy(ImputeStrategy::Constant);
        let fitted = FitUnsupervised::<f64>::fit(&imputer, &x).unwrap();
        let result = fitted.transform(&x).unwrap();

        assert_abs_diff_eq!(result[[0, 0]], 0.0);
        assert_abs_diff_eq!(result[[1, 1]], 0.0);
    }

    #[test]
    fn test_median_even_count() {
        // Even number of non-NaN values: median is average of two middle values.
        let x = array![[1.0], [3.0], [5.0], [7.0],];
        let imputer = SimpleImputer::<f64>::new().with_strategy(ImputeStrategy::Median);
        let fitted = FitUnsupervised::<f64>::fit(&imputer, &x).unwrap();
        // Median of [1,3,5,7] = (3+5)/2 = 4.0
        assert_abs_diff_eq!(fitted.fill_values()[0], 4.0);
    }
}
