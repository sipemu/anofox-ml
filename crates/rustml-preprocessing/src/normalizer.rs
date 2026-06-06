use ndarray::Array2;
use rustml_core::{FitUnsupervised, Float, Result, RustMlError, Transform};

/// The type of norm used to normalize each sample (row).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum NormType {
    /// L1 norm: sum of absolute values.
    L1,
    /// L2 norm (Euclidean): square root of sum of squares.
    L2,
    /// Max norm: maximum absolute value.
    Max,
}

/// Parameters for Normalizer (unfitted state).
///
/// Normalizes each **row** (sample) independently so that it has unit norm
/// according to the chosen [`NormType`]. Unlike most scalers this operates
/// per-sample, not per-feature.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Normalizer {
    /// The type of norm to apply.
    pub norm: NormType,
}

impl Normalizer {
    /// Create a new `Normalizer` with the default L2 norm.
    pub fn new() -> Self {
        Self { norm: NormType::L2 }
    }

    /// Set the norm type.
    pub fn with_norm(mut self, norm: NormType) -> Self {
        self.norm = norm;
        self
    }
}

impl Default for Normalizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Normalizer — stateless (fit is a validation-only no-op).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedNormalizer<F: Float> {
    norm: NormType,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> FitUnsupervised<F> for Normalizer {
    type Fitted = FittedNormalizer<F>;

    fn fit(&self, x: &Array2<F>) -> Result<Self::Fitted> {
        if x.is_empty() {
            return Err(RustMlError::EmptyInput("input array is empty".into()));
        }

        Ok(FittedNormalizer {
            norm: self.norm,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<F: Float> Transform<F> for FittedNormalizer<F> {
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>> {
        let eps = F::from_f64(1e-15).unwrap();
        let mut result = x.to_owned();

        for mut row in result.rows_mut() {
            let norm = match self.norm {
                NormType::L1 => {
                    let mut s = F::zero();
                    for &v in row.iter() {
                        s = s + v.abs();
                    }
                    s
                }
                NormType::L2 => {
                    let mut s = F::zero();
                    for &v in row.iter() {
                        s = s + v * v;
                    }
                    s.sqrt()
                }
                NormType::Max => {
                    let mut m = F::zero();
                    for &v in row.iter() {
                        let a = v.abs();
                        if a > m {
                            m = a;
                        }
                    }
                    m
                }
            };

            if norm > eps {
                for val in row.iter_mut() {
                    *val = *val / norm;
                }
            }
        }
        Ok(result)
    }
}

impl<F: Float> FittedNormalizer<F> {
    /// Return the norm type used for normalization.
    pub fn norm(&self) -> NormType {
        self.norm
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_l2_unit_norm() {
        let x = array![[3.0, 4.0], [1.0, 0.0], [0.0, 0.0,]];
        let normalizer = Normalizer::new();
        let fitted = FitUnsupervised::<f64>::fit(&normalizer, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        // Row 0: norm = 5, so [3/5, 4/5]
        assert_abs_diff_eq!(transformed[[0, 0]], 0.6, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[0, 1]], 0.8, epsilon = 1e-10);

        // Check each non-zero row has unit L2 norm
        for row_idx in 0..2 {
            let row = transformed.row(row_idx);
            let norm: f64 = row.iter().map(|&v| v * v).sum::<f64>().sqrt();
            assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_l1_norm() {
        let x = array![[1.0, -2.0, 3.0], [4.0, 0.0, -1.0]];
        let normalizer = Normalizer::new().with_norm(NormType::L1);
        let fitted = FitUnsupervised::<f64>::fit(&normalizer, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        // Each row's absolute values should sum to 1
        for row_idx in 0..transformed.nrows() {
            let row = transformed.row(row_idx);
            let l1: f64 = row.iter().map(|&v| v.abs()).sum();
            assert_abs_diff_eq!(l1, 1.0, epsilon = 1e-10);
        }

        // Row 0: L1 = |1|+|-2|+|3| = 6, so [1/6, -2/6, 3/6]
        assert_abs_diff_eq!(transformed[[0, 0]], 1.0 / 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[0, 1]], -2.0 / 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_max_norm() {
        let x = array![[1.0, -3.0, 2.0], [0.5, 0.0, -4.0]];
        let normalizer = Normalizer::new().with_norm(NormType::Max);
        let fitted = FitUnsupervised::<f64>::fit(&normalizer, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        // Each row's max absolute value should be 1
        for row_idx in 0..transformed.nrows() {
            let row = transformed.row(row_idx);
            let max_abs: f64 = row.iter().map(|&v| v.abs()).fold(0.0, f64::max);
            assert_abs_diff_eq!(max_abs, 1.0, epsilon = 1e-10);
        }

        // Row 0: max_abs = 3, so [1/3, -1, 2/3]
        assert_abs_diff_eq!(transformed[[0, 1]], -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_zero_row_handled() {
        // A row of all zeros should remain all zeros (no division by zero)
        let x = array![[0.0, 0.0], [3.0, 4.0]];
        let normalizer = Normalizer::new();
        let fitted = FitUnsupervised::<f64>::fit(&normalizer, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_abs_diff_eq!(transformed[[0, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[0, 1]], 0.0, epsilon = 1e-10);

        for &v in transformed.iter() {
            assert!(v.is_finite(), "zero row produced non-finite: {}", v);
        }
    }

    #[test]
    fn test_f32_support() {
        let x = array![[3.0f32, 4.0], [1.0, 0.0]];
        let normalizer = Normalizer::new();
        let fitted = FitUnsupervised::<f32>::fit(&normalizer, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        let row = transformed.row(0);
        let norm: f32 = row.iter().map(|&v| v * v).sum::<f32>().sqrt();
        assert_abs_diff_eq!(norm, 1.0f32, epsilon = 1e-5);
    }

    #[test]
    fn test_empty_input() {
        let x: Array2<f64> = Array2::zeros((0, 0));
        let normalizer = Normalizer::new();
        let result = FitUnsupervised::<f64>::fit(&normalizer, &x);
        assert!(result.is_err());
    }
}
