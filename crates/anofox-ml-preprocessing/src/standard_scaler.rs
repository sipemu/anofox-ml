use anofox_ml_core::{FitUnsupervised, Float, InverseTransform, Result, RustMlError, Transform};
use ndarray::{Array1, Array2};

/// Parameters for StandardScaler (unfitted state).
///
/// Standardizes features by removing the mean and scaling to unit variance:
/// `z = (x - mean) / std`
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StandardScaler {
    /// If true, center data to zero mean before scaling.
    pub with_mean: bool,
    /// If true, scale data to unit variance.
    pub with_std: bool,
}

impl StandardScaler {
    /// Create a new `StandardScaler` with defaults (both centering and scaling enabled).
    pub fn new() -> Self {
        Self {
            with_mean: true,
            with_std: true,
        }
    }

    /// Set whether to center data to zero mean before scaling.
    pub fn with_mean(mut self, with_mean: bool) -> Self {
        self.with_mean = with_mean;
        self
    }

    /// Set whether to scale data to unit variance.
    pub fn with_std(mut self, with_std: bool) -> Self {
        self.with_std = with_std;
        self
    }
}

impl Default for StandardScaler {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted StandardScaler — holds learned mean and std per feature.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedStandardScaler<F: Float> {
    mean: Array1<F>,
    std: Array1<F>,
    with_mean: bool,
    with_std: bool,
}

impl<F: Float> FitUnsupervised<F> for StandardScaler {
    type Fitted = FittedStandardScaler<F>;

    fn fit(&self, x: &Array2<F>) -> Result<Self::Fitted> {
        if x.is_empty() {
            return Err(RustMlError::EmptyInput("input array is empty".into()));
        }

        // NaN-aware per-column mean / variance (sklearn ignores NaN at fit
        // time and passes it through at transform time).
        let ncols = x.ncols();
        let mut mean = Array1::<F>::zeros(ncols);
        let mut std = Array1::<F>::ones(ncols);
        for j in 0..ncols {
            let col = x.column(j);
            let mut count = 0usize;
            let mut sum = F::zero();
            for &v in col.iter() {
                if !v.is_nan() {
                    count += 1;
                    sum += v;
                }
            }
            if count == 0 {
                mean[j] = F::nan();
                if self.with_std {
                    std[j] = F::nan();
                }
                continue;
            }
            let n = F::from_usize(count).unwrap();
            let m = sum / n;
            mean[j] = m;
            if self.with_std {
                // Two-pass variance with the same correction term as
                // sklearn's `_incremental_mean_and_var`.
                let mut sq = F::zero();
                let mut corr = F::zero();
                for &v in col.iter() {
                    if !v.is_nan() {
                        let d = v - m;
                        sq += d * d;
                        corr += d;
                    }
                }
                let var = ((sq - corr * corr / n) / n).max(F::zero());
                std[j] = if is_constant_feature(var, m, count) {
                    F::one()
                } else {
                    var.sqrt()
                };
            }
        }

        Ok(FittedStandardScaler {
            mean,
            std,
            with_mean: self.with_mean,
            with_std: self.with_std,
        })
    }
}

impl<F: Float> Transform<F> for FittedStandardScaler<F> {
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>> {
        if x.ncols() != self.mean.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.mean.len(),
                x.ncols()
            )));
        }

        let mut result = x.to_owned();
        for mut row in result.rows_mut() {
            for (j, val) in row.iter_mut().enumerate() {
                if self.with_mean {
                    *val -= self.mean[j];
                }
                if self.with_std && self.std[j] != F::zero() {
                    *val /= self.std[j];
                }
            }
        }
        Ok(result)
    }
}

impl<F: Float> InverseTransform<F> for FittedStandardScaler<F> {
    fn inverse_transform(&self, x: &Array2<F>) -> Result<Array2<F>> {
        if x.ncols() != self.mean.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.mean.len(),
                x.ncols()
            )));
        }

        let mut result = x.to_owned();
        for mut row in result.rows_mut() {
            for (j, val) in row.iter_mut().enumerate() {
                if self.with_std && self.std[j] != F::zero() {
                    *val *= self.std[j];
                }
                if self.with_mean {
                    *val += self.mean[j];
                }
            }
        }
        Ok(result)
    }
}

/// sklearn's `_is_constant_feature`: a feature is treated as constant when
/// its variance is within the floating-point error bound of the variance
/// computation, `var <= n·eps·var + (n·mean·eps)²`. Such features get a
/// scale of 1 (`_handle_zeros_in_scale`), so they are only centred.
pub(crate) fn is_constant_feature<F: Float>(var: F, mean: F, n_samples: usize) -> bool {
    let eps = f64::EPSILON;
    let var = var.to_f64().unwrap();
    let mean = mean.to_f64().unwrap();
    let n = n_samples as f64;
    let upper = n * eps * var + (n * mean * eps).powi(2);
    var <= upper
}

impl<F: Float> FittedStandardScaler<F> {
    pub fn mean(&self) -> &Array1<F> {
        &self.mean
    }

    /// Per-feature scale (`scale_` in sklearn): the standard deviation, or
    /// `1` for (numerically) constant features and when `with_std = false`.
    pub fn std(&self) -> &Array1<F> {
        &self.std
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;
    use ndarray::Axis;

    #[test]
    fn test_fit_transform() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let scaler = StandardScaler::default();
        let fitted = FitUnsupervised::<f64>::fit(&scaler, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        // Mean of each column should be ~0
        let col_means = transformed.sum_axis(Axis(0)) / 3.0;
        assert_abs_diff_eq!(col_means[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(col_means[1], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_inverse_transform_roundtrip() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let scaler = StandardScaler::default();
        let fitted = FitUnsupervised::<f64>::fit(&scaler, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();
        let recovered = fitted.inverse_transform(&transformed).unwrap();

        for (a, b) in x.iter().zip(recovered.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_without_mean() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let scaler = StandardScaler {
            with_mean: false,
            with_std: true,
        };
        let fitted = FitUnsupervised::<f64>::fit(&scaler, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        // Without centering, values should still be positive
        assert!(transformed[[0, 0]] > 0.0);
    }

    #[test]
    fn test_large_values() {
        // Very large feature values should not produce NaN/Inf
        let x = array![[1e10, -1e10], [2e10, -2e10], [3e10, -3e10], [4e10, -4e10],];
        let scaler = StandardScaler::default();
        let fitted = FitUnsupervised::<f64>::fit(&scaler, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        for &v in transformed.iter() {
            assert!(
                v.is_finite(),
                "transformed value should be finite, got {}",
                v
            );
        }
        // Mean should be ~0
        let col_means = transformed.sum_axis(Axis(0)) / 4.0;
        assert_abs_diff_eq!(col_means[0], 0.0, epsilon = 1e-8);
    }

    #[test]
    fn test_small_values() {
        // Very small feature values should not lose precision
        let x = array![
            [1e-10, 2e-10],
            [3e-10, 4e-10],
            [5e-10, 6e-10],
            [7e-10, 8e-10],
        ];
        let scaler = StandardScaler::default();
        let fitted = FitUnsupervised::<f64>::fit(&scaler, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        for &v in transformed.iter() {
            assert!(
                v.is_finite(),
                "transformed value should be finite, got {}",
                v
            );
        }
        // Roundtrip should preserve values
        let recovered = fitted.inverse_transform(&transformed).unwrap();
        for (a, b) in x.iter().zip(recovered.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-18);
        }
    }

    #[test]
    fn test_near_zero_variance_column() {
        // One column has near-zero variance; scaler should handle without NaN
        let x = array![
            [1.0, 5.0],
            [2.0, 5.0 + 1e-15],
            [3.0, 5.0 - 1e-15],
            [4.0, 5.0],
        ];
        let scaler = StandardScaler::default();
        let fitted = FitUnsupervised::<f64>::fit(&scaler, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        for &v in transformed.iter() {
            assert!(
                v.is_finite(),
                "near-zero variance column produced non-finite: {}",
                v
            );
        }
    }

    #[test]
    fn test_constant_feature_scale_is_one_like_sklearn() {
        // sklearn: StandardScaler().fit([[1, 0.1], [2, 0.1], [3, 0.1]]).scale_
        //   -> [0.81649658, 1.0]; a column of 0.1s has var ~1e-34 from
        //   rounding, which `_is_constant_feature` treats as zero.
        let x = array![[1.0, 0.1], [2.0, 0.1], [3.0, 0.1]];
        let fitted = FitUnsupervised::<f64>::fit(&StandardScaler::new(), &x).unwrap();
        assert_abs_diff_eq!(fitted.std()[0], 0.816_496_580_927_726, epsilon = 1e-12);
        assert_eq!(fitted.std()[1], 1.0);
        let t = fitted.transform(&x).unwrap();
        for i in 0..3 {
            assert!(t[[i, 1]].abs() < 1e-15);
        }
        // Large-offset constant column: rounding noise must not survive as
        // a tiny positive scale.
        let x = array![[1e8 + 0.1, 1.0], [1e8 + 0.1, 2.0], [1e8 + 0.1, 4.0]];
        let fitted = FitUnsupervised::<f64>::fit(&StandardScaler::new(), &x).unwrap();
        assert_eq!(fitted.std()[0], 1.0);
    }

    #[test]
    fn test_nan_ignored_at_fit_and_passed_through() {
        // sklearn: StandardScaler().fit([[1],[nan],[3]]) -> mean 2, scale 1
        let x = array![[1.0], [f64::NAN], [3.0]];
        let fitted = FitUnsupervised::<f64>::fit(&StandardScaler::new(), &x).unwrap();
        assert_abs_diff_eq!(fitted.mean()[0], 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(fitted.std()[0], 1.0, epsilon = 1e-12);
        let t = fitted.transform(&x).unwrap();
        assert!(t[[1, 0]].is_nan());
        assert_abs_diff_eq!(t[[0, 0]], -1.0, epsilon = 1e-12);
    }

    mod prop_tests {
        use super::*;
        use proptest::prelude::*;

        /// Generate a deterministic Array2<f64> from dimensions and a seed.
        fn make_data(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut values = Vec::with_capacity(rows * cols);
            for i in 0..(rows * cols) {
                let mut h = DefaultHasher::new();
                seed.hash(&mut h);
                (i as u64).hash(&mut h);
                let bits = h.finish();
                // Map to a reasonable f64 range [-10, 10]
                let v = (bits as f64 / u64::MAX as f64) * 20.0 - 10.0;
                values.push(v);
            }
            Array2::from_shape_vec((rows, cols), values).unwrap()
        }

        proptest! {
            #[test]
            fn standard_scaler_roundtrip(
                rows in 2..50usize,
                cols in 1..10usize,
                seed in 0u64..10000,
            ) {
                let x = make_data(rows, cols, seed);

                let scaler = StandardScaler::default();
                let fitted = FitUnsupervised::<f64>::fit(&scaler, &x).unwrap();
                let transformed = fitted.transform(&x).unwrap();
                let recovered = fitted.inverse_transform(&transformed).unwrap();

                for (a, b) in x.iter().zip(recovered.iter()) {
                    prop_assert!((a - b).abs() < 1e-8,
                        "roundtrip failed: original={}, recovered={}", a, b);
                }
            }

            #[test]
            fn standard_scaler_mean_zero(
                rows in 2..50usize,
                cols in 1..10usize,
                seed in 0u64..10000,
            ) {
                let x = make_data(rows, cols, seed);

                let scaler = StandardScaler::default();
                let fitted = FitUnsupervised::<f64>::fit(&scaler, &x).unwrap();
                let transformed = fitted.transform(&x).unwrap();

                let n = rows as f64;
                for col_idx in 0..cols {
                    let col_mean: f64 = transformed.column(col_idx).sum() / n;
                    prop_assert!(col_mean.abs() < 1e-8,
                        "column {} mean should be ~0, got {}", col_idx, col_mean);

                    // Standard deviation should be ~1 (if original std > 0)
                    let col_std: f64 = (transformed.column(col_idx)
                        .iter()
                        .map(|&v| (v - col_mean) * (v - col_mean))
                        .sum::<f64>() / n)
                        .sqrt();
                    if fitted.std()[col_idx] > 1e-15 {
                        prop_assert!((col_std - 1.0).abs() < 1e-6,
                            "column {} std should be ~1, got {}", col_idx, col_std);
                    }
                }
            }
        }
    }
}
