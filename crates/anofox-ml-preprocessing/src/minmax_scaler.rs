use anofox_ml_core::{
    FitUnsupervised, Float, InverseTransform, PartialFitUnsupervised, Result, RustMlError,
    Transform,
};
use ndarray::{Array1, Array2};

/// Parameters for MinMaxScaler (unfitted state).
///
/// Scales features to a given range (default [0, 1]):
/// `x_scaled = (x - min) / (max - min) * (feature_max - feature_min) + feature_min`
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct MinMaxScaler<F: Float> {
    pub feature_min: F,
    pub feature_max: F,
}

impl<F: Float> MinMaxScaler<F> {
    /// Create a new `MinMaxScaler` with the default range [0, 1].
    pub fn new() -> Self {
        Self {
            feature_min: F::zero(),
            feature_max: F::one(),
        }
    }

    /// Set the target feature range (min, max).
    pub fn with_range(mut self, min: F, max: F) -> Self {
        self.feature_min = min;
        self.feature_max = max;
        self
    }
}

impl<F: Float> Default for MinMaxScaler<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted MinMaxScaler — holds learned min/max per feature.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedMinMaxScaler<F: Float> {
    data_min: Array1<F>,
    data_max: Array1<F>,
    feature_min: F,
    feature_max: F,
}

impl<F: Float> FitUnsupervised<F> for MinMaxScaler<F> {
    type Fitted = FittedMinMaxScaler<F>;

    fn fit(&self, x: &Array2<F>) -> Result<Self::Fitted> {
        self.partial_fit(None, x)
    }
}

impl<F: Float> PartialFitUnsupervised<F> for MinMaxScaler<F> {
    type Fitted = FittedMinMaxScaler<F>;

    /// sklearn's `MinMaxScaler.partial_fit`: the running per-feature min and
    /// max are widened by each batch (NaN ignored).
    fn partial_fit(&self, state: Option<Self::Fitted>, x: &Array2<F>) -> Result<Self::Fitted> {
        if x.is_empty() {
            return Err(RustMlError::EmptyInput("input array is empty".into()));
        }
        if self.feature_min >= self.feature_max {
            return Err(RustMlError::InvalidParameter(
                "feature_min must be less than feature_max".into(),
            ));
        }
        let (init_min, init_max) = match state {
            Some(prev) => {
                if prev.data_min.len() != x.ncols() {
                    return Err(RustMlError::ShapeMismatch(format!(
                        "expected {} features, got {}",
                        prev.data_min.len(),
                        x.ncols()
                    )));
                }
                (prev.data_min, prev.data_max)
            }
            None => (
                Array1::from_elem(x.ncols(), F::infinity()),
                Array1::from_elem(x.ncols(), F::neg_infinity()),
            ),
        };
        let mut data_min = init_min;
        let mut data_max = init_max;
        for row in x.rows() {
            for (j, &v) in row.iter().enumerate() {
                if !v.is_nan() {
                    data_min[j] = data_min[j].min(v);
                    data_max[j] = data_max[j].max(v);
                }
            }
        }

        Ok(FittedMinMaxScaler {
            data_min,
            data_max,
            feature_min: self.feature_min,
            feature_max: self.feature_max,
        })
    }
}

impl<F: Float> Transform<F> for FittedMinMaxScaler<F> {
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>> {
        if x.ncols() != self.data_min.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.data_min.len(),
                x.ncols()
            )));
        }

        let range = self.feature_max - self.feature_min;
        let mut result = x.to_owned();

        for mut row in result.rows_mut() {
            for (j, val) in row.iter_mut().enumerate() {
                let data_range = self.data_max[j] - self.data_min[j];
                if data_range > F::from_f64(1e-15).unwrap() {
                    *val = (*val - self.data_min[j]) / data_range * range + self.feature_min;
                } else {
                    *val = self.feature_min;
                }
            }
        }
        Ok(result)
    }
}

impl<F: Float> InverseTransform<F> for FittedMinMaxScaler<F> {
    fn inverse_transform(&self, x: &Array2<F>) -> Result<Array2<F>> {
        if x.ncols() != self.data_min.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.data_min.len(),
                x.ncols()
            )));
        }

        let range = self.feature_max - self.feature_min;
        let mut result = x.to_owned();

        for mut row in result.rows_mut() {
            for (j, val) in row.iter_mut().enumerate() {
                let data_range = self.data_max[j] - self.data_min[j];
                if data_range > F::from_f64(1e-15).unwrap() {
                    *val = (*val - self.feature_min) / range * data_range + self.data_min[j];
                } else {
                    *val = self.data_min[j];
                }
            }
        }
        Ok(result)
    }
}

impl<F: Float> FittedMinMaxScaler<F> {
    pub fn data_min(&self) -> &Array1<F> {
        &self.data_min
    }

    pub fn data_max(&self) -> &Array1<F> {
        &self.data_max
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_fit_transform_default() {
        let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
        let scaler = MinMaxScaler::<f64>::default();
        let fitted = scaler.fit(&x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_abs_diff_eq!(transformed[[0, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[2, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[0, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[2, 1]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_custom_range() {
        let x = array![[1.0], [2.0], [3.0]];
        let scaler = MinMaxScaler {
            feature_min: -1.0,
            feature_max: 1.0,
        };
        let fitted = scaler.fit(&x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_abs_diff_eq!(transformed[[0, 0]], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[1, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[2, 0]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_inverse_transform_roundtrip() {
        let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
        let scaler = MinMaxScaler::<f64>::default();
        let fitted = scaler.fit(&x).unwrap();
        let transformed = fitted.transform(&x).unwrap();
        let recovered = fitted.inverse_transform(&transformed).unwrap();

        for (a, b) in x.iter().zip(recovered.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_large_values() {
        let x = array![[1e10], [2e10], [3e10]];
        let scaler = MinMaxScaler::<f64>::default();
        let fitted = scaler.fit(&x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_abs_diff_eq!(transformed[[0, 0]], 0.0, epsilon = 1e-8);
        assert_abs_diff_eq!(transformed[[2, 0]], 1.0, epsilon = 1e-8);
        for &v in transformed.iter() {
            assert!(v.is_finite(), "large values produced non-finite: {}", v);
        }
    }

    #[test]
    fn test_small_values() {
        let x = array![[1e-10], [2e-10], [3e-10]];
        let scaler = MinMaxScaler::<f64>::default();
        let fitted = scaler.fit(&x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_abs_diff_eq!(transformed[[0, 0]], 0.0, epsilon = 1e-8);
        assert_abs_diff_eq!(transformed[[2, 0]], 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_constant_column() {
        // Column with zero range should not produce NaN
        let x = array![[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]];
        let scaler = MinMaxScaler::<f64>::default();
        let fitted = scaler.fit(&x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        for &v in transformed.iter() {
            assert!(v.is_finite(), "constant column produced non-finite: {}", v);
        }
    }

    #[test]
    fn test_partial_fit_matches_sklearn() {
        // sklearn 1.8: MinMaxScaler().partial_fit(X[:2]).partial_fit(X[2:])
        let x = array![
            [1.0, -2.0, 5.0],
            [3.0, f64::NAN, 5.0],
            [4.5, 0.5, 5.0],
            [-1.0, 7.0, 5.0],
            [2.0, 3.0, 5.0],
            [10.0, -4.0, 5.0]
        ];
        let s = MinMaxScaler::<f64>::new();
        let st = s
            .partial_fit(None, &x.slice(ndarray::s![..2, ..]).to_owned())
            .unwrap();
        let f = s
            .partial_fit(Some(st), &x.slice(ndarray::s![2.., ..]).to_owned())
            .unwrap();
        assert_eq!(f.data_min().to_vec(), vec![-1.0, -4.0, 5.0]);
        assert_eq!(f.data_max().to_vec(), vec![10.0, 7.0, 5.0]);
    }
}
