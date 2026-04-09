use ndarray::{Array1, Array2};
use rustml_core::{Float, FitUnsupervised, InverseTransform, Result, RustMlError, Transform};

/// Parameters for RobustScaler (unfitted state).
///
/// Scales features using statistics that are robust to outliers.
/// Uses the median and interquartile range (IQR = Q3 - Q1) instead of
/// mean and standard deviation:
/// `z = (x - median) / IQR`
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RobustScaler {
    /// If true, center data by subtracting the median.
    pub with_centering: bool,
    /// If true, scale data by dividing by the IQR.
    pub with_scaling: bool,
}

impl RobustScaler {
    /// Create a new `RobustScaler` with defaults (both centering and scaling enabled).
    pub fn new() -> Self {
        Self {
            with_centering: true,
            with_scaling: true,
        }
    }

    /// Set whether to center data by subtracting the median.
    pub fn with_centering(mut self, with_centering: bool) -> Self {
        self.with_centering = with_centering;
        self
    }

    /// Set whether to scale data by dividing by the IQR.
    pub fn with_scaling(mut self, with_scaling: bool) -> Self {
        self.with_scaling = with_scaling;
        self
    }
}

impl Default for RobustScaler {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted RobustScaler — holds learned median and IQR per feature.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedRobustScaler<F: Float> {
    median: Array1<F>,
    iqr: Array1<F>,
    with_centering: bool,
    with_scaling: bool,
}

/// Compute a percentile value using linear interpolation.
///
/// `sorted` must be a sorted slice of values and `p` must be in [0, 1].
fn percentile<F: Float>(sorted: &[F], p: f64) -> F {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let idx = p * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = F::from_f64(idx - lo as f64).unwrap();
        sorted[lo] * (F::one() - frac) + sorted[hi] * frac
    }
}

impl<F: Float> FitUnsupervised<F> for RobustScaler {
    type Fitted = FittedRobustScaler<F>;

    fn fit(&self, x: &Array2<F>) -> Result<Self::Fitted> {
        if x.is_empty() {
            return Err(RustMlError::EmptyInput("input array is empty".into()));
        }

        let ncols = x.ncols();
        let mut median = Array1::<F>::zeros(ncols);
        let mut iqr = Array1::<F>::ones(ncols);

        for j in 0..ncols {
            let col = x.column(j);
            let mut sorted: Vec<F> = col.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

            median[j] = percentile(&sorted, 0.5);

            if self.with_scaling {
                let q1 = percentile(&sorted, 0.25);
                let q3 = percentile(&sorted, 0.75);
                iqr[j] = q3 - q1;
            }
        }

        Ok(FittedRobustScaler {
            median,
            iqr,
            with_centering: self.with_centering,
            with_scaling: self.with_scaling,
        })
    }
}

impl<F: Float> Transform<F> for FittedRobustScaler<F> {
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>> {
        if x.ncols() != self.median.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.median.len(),
                x.ncols()
            )));
        }

        let mut result = x.to_owned();
        for mut row in result.rows_mut() {
            for (j, val) in row.iter_mut().enumerate() {
                if self.with_centering {
                    *val -= self.median[j];
                }
                if self.with_scaling && self.iqr[j] > F::from_f64(1e-15).unwrap() {
                    *val /= self.iqr[j];
                }
            }
        }
        Ok(result)
    }
}

impl<F: Float> InverseTransform<F> for FittedRobustScaler<F> {
    fn inverse_transform(&self, x: &Array2<F>) -> Result<Array2<F>> {
        if x.ncols() != self.median.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.median.len(),
                x.ncols()
            )));
        }

        let mut result = x.to_owned();
        for mut row in result.rows_mut() {
            for (j, val) in row.iter_mut().enumerate() {
                if self.with_scaling && self.iqr[j] > F::from_f64(1e-15).unwrap() {
                    *val *= self.iqr[j];
                }
                if self.with_centering {
                    *val += self.median[j];
                }
            }
        }
        Ok(result)
    }
}

impl<F: Float> FittedRobustScaler<F> {
    /// Return the median per feature.
    pub fn median(&self) -> &Array1<F> {
        &self.median
    }

    /// Return the IQR per feature.
    pub fn iqr(&self) -> &Array1<F> {
        &self.iqr
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_fit_transform() {
        let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]];
        let scaler = RobustScaler::default();
        let fitted = FitUnsupervised::<f64>::fit(&scaler, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        // Median of [1,2,3,4,5] is 3; Q1=2, Q3=4, IQR=2
        // (1 - 3)/2 = -1.0, (3-3)/2 = 0.0, (5-3)/2 = 1.0
        assert_abs_diff_eq!(fitted.median()[0], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(fitted.iqr()[0], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[2, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[0, 0]], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[4, 0]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_inverse_transform_roundtrip() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let scaler = RobustScaler::default();
        let fitted = FitUnsupervised::<f64>::fit(&scaler, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();
        let recovered = fitted.inverse_transform(&transformed).unwrap();

        for (a, b) in x.iter().zip(recovered.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_without_centering() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let scaler = RobustScaler::new().with_centering(false);
        let fitted = FitUnsupervised::<f64>::fit(&scaler, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        // Without centering, median of [1,2,3,4,5] is 3; IQR=2
        // 1/2 = 0.5, 3/2 = 1.5
        assert_abs_diff_eq!(transformed[[0, 0]], 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[2, 0]], 1.5, epsilon = 1e-10);
    }

    #[test]
    fn test_without_scaling() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let scaler = RobustScaler::new().with_scaling(false);
        let fitted = FitUnsupervised::<f64>::fit(&scaler, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        // Without scaling, just center: 1-3 = -2, 3-3 = 0, 5-3 = 2
        assert_abs_diff_eq!(transformed[[0, 0]], -2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[2, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[4, 0]], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_constant_column() {
        let x = array![[5.0, 1.0], [5.0, 2.0], [5.0, 3.0], [5.0, 4.0]];
        let scaler = RobustScaler::default();
        let fitted = FitUnsupervised::<f64>::fit(&scaler, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        for &v in transformed.iter() {
            assert!(v.is_finite(), "constant column produced non-finite: {}", v);
        }
    }

    #[test]
    fn test_empty_input() {
        let x: Array2<f64> = Array2::zeros((0, 0));
        let scaler = RobustScaler::default();
        let result = FitUnsupervised::<f64>::fit(&scaler, &x);
        assert!(result.is_err());
    }

    #[test]
    fn test_shape_mismatch() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let scaler = RobustScaler::default();
        let fitted = FitUnsupervised::<f64>::fit(&scaler, &x).unwrap();

        let x_wrong = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&x_wrong).is_err());
        assert!(fitted.inverse_transform(&x_wrong).is_err());
    }

    #[test]
    fn test_even_number_of_rows() {
        // Even number of rows: median by linear interpolation
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let scaler = RobustScaler::default();
        let fitted = FitUnsupervised::<f64>::fit(&scaler, &x).unwrap();
        // Median of [1,2,3,4] = (2+3)/2 = 2.5
        assert_abs_diff_eq!(fitted.median()[0], 2.5, epsilon = 1e-10);
    }

    #[test]
    fn test_large_values() {
        let x = array![[1e10], [2e10], [3e10], [4e10], [5e10]];
        let scaler = RobustScaler::default();
        let fitted = FitUnsupervised::<f64>::fit(&scaler, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        for &v in transformed.iter() {
            assert!(v.is_finite(), "large values produced non-finite: {}", v);
        }
    }

    #[test]
    fn test_single_row() {
        let x = array![[1.0, 2.0]];
        let scaler = RobustScaler::default();
        let fitted = FitUnsupervised::<f64>::fit(&scaler, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        // Single row: median = value, IQR = 0 -> centering gives 0, no scaling
        assert_abs_diff_eq!(transformed[[0, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[0, 1]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_f32() {
        let x = array![[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let scaler = RobustScaler::default();
        let fitted = FitUnsupervised::<f32>::fit(&scaler, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();
        let recovered = fitted.inverse_transform(&transformed).unwrap();

        for (a, b) in x.iter().zip(recovered.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-5);
        }
    }

    mod prop_tests {
        use super::*;
        use proptest::prelude::*;

        fn make_data(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut values = Vec::with_capacity(rows * cols);
            for i in 0..(rows * cols) {
                let mut h = DefaultHasher::new();
                seed.hash(&mut h);
                (i as u64).hash(&mut h);
                let bits = h.finish();
                let v = (bits as f64 / u64::MAX as f64) * 20.0 - 10.0;
                values.push(v);
            }
            Array2::from_shape_vec((rows, cols), values).unwrap()
        }

        proptest! {
            #[test]
            fn robust_scaler_roundtrip(
                rows in 2..50usize,
                cols in 1..10usize,
                seed in 0u64..10000,
            ) {
                let x = make_data(rows, cols, seed);
                let scaler = RobustScaler::default();
                let fitted = FitUnsupervised::<f64>::fit(&scaler, &x).unwrap();
                let transformed = fitted.transform(&x).unwrap();
                let recovered = fitted.inverse_transform(&transformed).unwrap();

                for (a, b) in x.iter().zip(recovered.iter()) {
                    prop_assert!((a - b).abs() < 1e-8,
                        "roundtrip failed: original={}, recovered={}", a, b);
                }
            }

            #[test]
            fn robust_scaler_median_zero(
                rows in 4..50usize,
                cols in 1..10usize,
                seed in 0u64..10000,
            ) {
                let x = make_data(rows, cols, seed);
                let scaler = RobustScaler::default();
                let fitted = FitUnsupervised::<f64>::fit(&scaler, &x).unwrap();
                let transformed = fitted.transform(&x).unwrap();

                // After centering, the median of each column should be ~0
                for col_idx in 0..cols {
                    let col = transformed.column(col_idx);
                    let mut sorted: Vec<f64> = col.to_vec();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let median = super::super::percentile(&sorted, 0.5);
                    prop_assert!(median.abs() < 1e-8,
                        "column {} median should be ~0, got {}", col_idx, median);
                }
            }
        }
    }
}
