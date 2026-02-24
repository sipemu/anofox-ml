use ndarray::{Array1, Array2, Axis};
use rustml_core::{Float, FitUnsupervised, InverseTransform, Result, RustMlError, Transform};

/// Parameters for StandardScaler (unfitted state).
///
/// Standardizes features by removing the mean and scaling to unit variance:
/// `z = (x - mean) / std`
#[derive(Debug, Clone)]
pub struct StandardScaler {
    /// If true, center data to zero mean before scaling.
    pub with_mean: bool,
    /// If true, scale data to unit variance.
    pub with_std: bool,
}

impl Default for StandardScaler {
    fn default() -> Self {
        Self {
            with_mean: true,
            with_std: true,
        }
    }
}

/// Fitted StandardScaler — holds learned mean and std per feature.
#[derive(Debug, Clone)]
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

        let n = F::from_usize(x.nrows()).unwrap();
        let mean = x.sum_axis(Axis(0)) / n;

        let std = if self.with_std {
            let mut s = Array1::<F>::zeros(x.ncols());
            for row in x.rows() {
                for (j, (&val, &m)) in row.iter().zip(mean.iter()).enumerate() {
                    s[j] += (val - m) * (val - m);
                }
            }
            s.mapv(|v| (v / n).sqrt())
        } else {
            Array1::ones(x.ncols())
        };

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
                if self.with_std && self.std[j] > F::from_f64(1e-15).unwrap() {
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
                if self.with_std && self.std[j] > F::from_f64(1e-15).unwrap() {
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

impl<F: Float> FittedStandardScaler<F> {
    pub fn mean(&self) -> &Array1<F> {
        &self.mean
    }

    pub fn std(&self) -> &Array1<F> {
        &self.std
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

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
}
