use ndarray::{Array1, Array2};
use rustml_core::{FitUnsupervised, Float, Result, RustMlError, Transform};
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
/// the non-missing values during [`FitUnsupervised::fit`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct SimpleImputer<F: Float> {
    strategy: ImputeStrategy,
    fill_value: Option<F>,
}

impl<F: Float> SimpleImputer<F> {
    /// Create a new `SimpleImputer` with the default [`ImputeStrategy::Mean`].
    pub fn new() -> Self {
        Self {
            strategy: ImputeStrategy::Mean,
            fill_value: None,
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
}

impl<F: Float> FittedSimpleImputer<F> {
    /// Return the per-column fill values learned during fit.
    pub fn fill_values(&self) -> &Array1<F> {
        &self.fill_values
    }
}

/// Compute the mean of non-NaN values in `values`. Returns `None` if all are NaN.
fn column_mean<F: Float>(values: &[F]) -> Option<F> {
    let mut sum = F::zero();
    let mut count = 0usize;
    for &v in values {
        if !v.is_nan() {
            sum = sum + v;
            count += 1;
        }
    }
    if count == 0 {
        None
    } else {
        Some(sum / F::from_usize(count).unwrap())
    }
}

/// Compute the median of non-NaN values in `values`. Returns `None` if all are NaN.
fn column_median<F: Float>(values: &[F]) -> Option<F> {
    let mut valid: Vec<F> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    if valid.is_empty() {
        return None;
    }
    valid.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = valid.len();
    if n % 2 == 1 {
        Some(valid[n / 2])
    } else {
        Some((valid[n / 2 - 1] + valid[n / 2]) / F::from_f64(2.0).unwrap())
    }
}

/// Compute the most frequent non-NaN value in `values`. Returns `None` if all are NaN.
/// Ties are broken by taking the smallest value.
fn column_most_frequent<F: Float>(values: &[F]) -> Option<F> {
    let mut counts: HashMap<u64, (F, usize)> = HashMap::new();
    for &v in values {
        if v.is_nan() {
            continue;
        }
        // Use bit representation as hash key for exact equality.
        let bits = v.to_f64().unwrap().to_bits();
        counts
            .entry(bits)
            .and_modify(|e| e.1 += 1)
            .or_insert((v, 1));
    }
    if counts.is_empty() {
        return None;
    }
    // Pick highest count, break ties by smallest value.
    counts
        .values()
        .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.partial_cmp(&a.0).unwrap()))
        .map(|&(v, _)| v)
}

impl<F: Float> FitUnsupervised<F> for SimpleImputer<F> {
    type Fitted = FittedSimpleImputer<F>;

    fn fit(&self, x: &Array2<F>) -> Result<Self::Fitted> {
        if x.is_empty() {
            return Err(RustMlError::EmptyInput("input array is empty".into()));
        }

        if self.strategy == ImputeStrategy::Constant {
            let fill = self.fill_value.unwrap_or_else(F::zero);
            let fill_values = Array1::from_elem(x.ncols(), fill);
            return Ok(FittedSimpleImputer { fill_values });
        }

        let ncols = x.ncols();
        let mut fill_values = Array1::<F>::zeros(ncols);

        for j in 0..ncols {
            let col: Vec<F> = x.column(j).to_vec();
            let computed = match self.strategy {
                ImputeStrategy::Mean => column_mean(&col),
                ImputeStrategy::Median => column_median(&col),
                ImputeStrategy::MostFrequent => column_most_frequent(&col),
                ImputeStrategy::Constant => unreachable!(),
            };
            match computed {
                Some(v) => fill_values[j] = v,
                None => {
                    return Err(RustMlError::InvalidParameter(format!(
                        "column {} contains only NaN values",
                        j
                    )));
                }
            }
        }

        Ok(FittedSimpleImputer { fill_values })
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

        let mut result = x.to_owned();
        for mut row in result.rows_mut() {
            for (j, val) in row.iter_mut().enumerate() {
                if val.is_nan() {
                    *val = self.fill_values[j];
                }
            }
        }
        Ok(result)
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
    fn test_all_nan_column_error() {
        let x = array![[1.0, f64::NAN], [2.0, f64::NAN], [3.0, f64::NAN],];
        let imputer = SimpleImputer::<f64>::new();
        let result = FitUnsupervised::<f64>::fit(&imputer, &x);
        assert!(result.is_err());
        let err = result.unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("column 1"),
            "error should mention column index: {}",
            msg
        );
        assert!(msg.contains("NaN"), "error should mention NaN: {}", msg);
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
