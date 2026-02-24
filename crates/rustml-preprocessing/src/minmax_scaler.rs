use ndarray::{Array1, Array2, Axis};
use rustml_core::{Float, FitUnsupervised, InverseTransform, Result, RustMlError, Transform};

/// Parameters for MinMaxScaler (unfitted state).
///
/// Scales features to a given range (default [0, 1]):
/// `x_scaled = (x - min) / (max - min) * (feature_max - feature_min) + feature_min`
#[derive(Debug, Clone)]
pub struct MinMaxScaler<F: Float> {
    pub feature_min: F,
    pub feature_max: F,
}

impl<F: Float> Default for MinMaxScaler<F> {
    fn default() -> Self {
        Self {
            feature_min: F::zero(),
            feature_max: F::one(),
        }
    }
}

/// Fitted MinMaxScaler — holds learned min/max per feature.
#[derive(Debug, Clone)]
pub struct FittedMinMaxScaler<F: Float> {
    data_min: Array1<F>,
    data_max: Array1<F>,
    feature_min: F,
    feature_max: F,
}

impl<F: Float> FitUnsupervised<F> for MinMaxScaler<F> {
    type Fitted = FittedMinMaxScaler<F>;

    fn fit(&self, x: &Array2<F>) -> Result<Self::Fitted> {
        if x.is_empty() {
            return Err(RustMlError::EmptyInput("input array is empty".into()));
        }
        if self.feature_min >= self.feature_max {
            return Err(RustMlError::InvalidParameter(
                "feature_min must be less than feature_max".into(),
            ));
        }

        let data_min = x.fold_axis(Axis(0), F::infinity(), |&a, &b| a.min(b));
        let data_max = x.fold_axis(Axis(0), F::neg_infinity(), |&a, &b| a.max(b));

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
}
