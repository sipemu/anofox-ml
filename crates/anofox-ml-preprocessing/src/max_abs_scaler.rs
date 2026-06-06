use anofox_ml_core::{FitUnsupervised, Float, InverseTransform, Result, RustMlError, Transform};
use ndarray::{Array1, Array2, Axis};

/// Parameters for MaxAbsScaler (unfitted state).
///
/// Scales each feature by its maximum absolute value so that the resulting
/// values lie in the range [-1, 1]. Unlike `StandardScaler` or
/// `RobustScaler`, this scaler does **not** center the data, which makes
/// it suitable for sparse data.
///
/// `x_scaled[i, j] = x[i, j] / max_abs[j]`
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MaxAbsScaler;

impl MaxAbsScaler {
    /// Create a new `MaxAbsScaler`.
    pub fn new() -> Self {
        Self
    }
}

impl Default for MaxAbsScaler {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted MaxAbsScaler — holds the maximum absolute value per feature.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedMaxAbsScaler<F: Float> {
    max_abs: Array1<F>,
}

impl<F: Float> FitUnsupervised<F> for MaxAbsScaler {
    type Fitted = FittedMaxAbsScaler<F>;

    fn fit(&self, x: &Array2<F>) -> Result<Self::Fitted> {
        if x.is_empty() {
            return Err(RustMlError::EmptyInput("input array is empty".into()));
        }

        let max_abs = x
            .mapv(|v| v.abs())
            .fold_axis(Axis(0), F::zero(), |&a, &b| a.max(b));

        Ok(FittedMaxAbsScaler { max_abs })
    }
}

impl<F: Float> Transform<F> for FittedMaxAbsScaler<F> {
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>> {
        if x.ncols() != self.max_abs.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.max_abs.len(),
                x.ncols()
            )));
        }

        let eps = F::from_f64(1e-15).unwrap();
        let mut result = x.to_owned();
        for mut row in result.rows_mut() {
            for (j, val) in row.iter_mut().enumerate() {
                if self.max_abs[j] > eps {
                    *val = *val / self.max_abs[j];
                }
            }
        }
        Ok(result)
    }
}

impl<F: Float> InverseTransform<F> for FittedMaxAbsScaler<F> {
    fn inverse_transform(&self, x: &Array2<F>) -> Result<Array2<F>> {
        if x.ncols() != self.max_abs.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.max_abs.len(),
                x.ncols()
            )));
        }

        let eps = F::from_f64(1e-15).unwrap();
        let mut result = x.to_owned();
        for mut row in result.rows_mut() {
            for (j, val) in row.iter_mut().enumerate() {
                if self.max_abs[j] > eps {
                    *val = *val * self.max_abs[j];
                }
            }
        }
        Ok(result)
    }
}

impl<F: Float> FittedMaxAbsScaler<F> {
    /// Return the maximum absolute value per feature.
    pub fn max_abs(&self) -> &Array1<F> {
        &self.max_abs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_basic_scaling() {
        let x = array![[1.0, -4.0], [2.0, 2.0], [-3.0, 1.0]];
        let scaler = MaxAbsScaler::new();
        let fitted = FitUnsupervised::<f64>::fit(&scaler, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        // max_abs for col 0 = 3.0, col 1 = 4.0
        assert_abs_diff_eq!(fitted.max_abs()[0], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(fitted.max_abs()[1], 4.0, epsilon = 1e-10);

        assert_abs_diff_eq!(transformed[[0, 0]], 1.0 / 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[0, 1]], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[2, 0]], -1.0, epsilon = 1e-10);

        // All values should be in [-1, 1]
        for &v in transformed.iter() {
            assert!(v >= -1.0 && v <= 1.0, "value {} not in [-1, 1]", v);
        }
    }

    #[test]
    fn test_inverse_transform_roundtrip() {
        let x = array![[1.0, -4.0], [2.0, 2.0], [-3.0, 1.0]];
        let scaler = MaxAbsScaler::new();
        let fitted = FitUnsupervised::<f64>::fit(&scaler, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();
        let recovered = fitted.inverse_transform(&transformed).unwrap();

        for (a, b) in x.iter().zip(recovered.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_zero_column() {
        // A column of all zeros should pass through unchanged (no division by zero)
        let x = array![[0.0, 2.0], [0.0, -4.0], [0.0, 1.0]];
        let scaler = MaxAbsScaler::new();
        let fitted = FitUnsupervised::<f64>::fit(&scaler, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        // Zero column stays zero
        assert_abs_diff_eq!(transformed[[0, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[1, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[2, 0]], 0.0, epsilon = 1e-10);

        // Non-zero column is scaled
        assert_abs_diff_eq!(transformed[[1, 1]], -1.0, epsilon = 1e-10);

        for &v in transformed.iter() {
            assert!(v.is_finite(), "zero column produced non-finite: {}", v);
        }
    }

    #[test]
    fn test_f32_support() {
        let x = array![[1.0f32, -4.0], [2.0, 2.0], [-3.0, 1.0]];
        let scaler = MaxAbsScaler::new();
        let fitted = FitUnsupervised::<f32>::fit(&scaler, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();
        let recovered = fitted.inverse_transform(&transformed).unwrap();

        for (a, b) in x.iter().zip(recovered.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_already_scaled() {
        // Data already in [-1, 1] should be unchanged when max_abs == 1
        let x = array![[-1.0, 0.5], [0.0, 1.0], [0.5, -1.0]];
        let scaler = MaxAbsScaler::new();
        let fitted = FitUnsupervised::<f64>::fit(&scaler, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        for (a, b) in x.iter().zip(transformed.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_empty_input() {
        let x: Array2<f64> = Array2::zeros((0, 0));
        let scaler = MaxAbsScaler::new();
        let result = FitUnsupervised::<f64>::fit(&scaler, &x);
        assert!(result.is_err());
    }

    #[test]
    fn test_shape_mismatch() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let scaler = MaxAbsScaler::new();
        let fitted = FitUnsupervised::<f64>::fit(&scaler, &x).unwrap();

        let x_wrong = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&x_wrong).is_err());
        assert!(fitted.inverse_transform(&x_wrong).is_err());
    }
}
