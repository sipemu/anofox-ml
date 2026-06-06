use anofox_ml_core::{FitUnsupervised, Float, Result, RustMlError, Transform};
use ndarray::Array2;

/// Parameters for Binarizer (unfitted state).
///
/// Thresholds features: values strictly greater than the threshold become 1,
/// all others become 0.
///
/// `x_out[i, j] = if x[i, j] > threshold { 1 } else { 0 }`
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct Binarizer<F: Float> {
    /// The threshold value. Features greater than this are set to 1, else 0.
    pub threshold: F,
}

impl<F: Float> Binarizer<F> {
    /// Create a new `Binarizer` with the given threshold.
    pub fn new(threshold: F) -> Self {
        Self { threshold }
    }
}

impl<F: Float> Default for Binarizer<F> {
    /// Default binarizer with threshold 0.
    fn default() -> Self {
        Self::new(F::zero())
    }
}

/// Fitted Binarizer — stateless, stores only the threshold.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedBinarizer<F: Float> {
    threshold: F,
}

impl<F: Float> FitUnsupervised<F> for Binarizer<F> {
    type Fitted = FittedBinarizer<F>;

    fn fit(&self, x: &Array2<F>) -> Result<Self::Fitted> {
        if x.is_empty() {
            return Err(RustMlError::EmptyInput("input array is empty".into()));
        }

        Ok(FittedBinarizer {
            threshold: self.threshold,
        })
    }
}

impl<F: Float> Transform<F> for FittedBinarizer<F> {
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>> {
        let one = F::one();
        let zero = F::zero();

        let result = x.mapv(|v| if v > self.threshold { one } else { zero });
        Ok(result)
    }
}

impl<F: Float> FittedBinarizer<F> {
    /// Return the threshold value.
    pub fn threshold(&self) -> F {
        self.threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_basic_thresholding() {
        let x = array![[1.0, -1.0, 2.0], [0.5, 0.0, -0.5]];
        let binarizer = Binarizer::new(0.5);
        let fitted = FitUnsupervised::<f64>::fit(&binarizer, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        // Values > 0.5 become 1, others become 0
        assert_abs_diff_eq!(transformed[[0, 0]], 1.0, epsilon = 1e-10); // 1.0 > 0.5
        assert_abs_diff_eq!(transformed[[0, 1]], 0.0, epsilon = 1e-10); // -1.0 <= 0.5
        assert_abs_diff_eq!(transformed[[0, 2]], 1.0, epsilon = 1e-10); // 2.0 > 0.5
        assert_abs_diff_eq!(transformed[[1, 0]], 0.0, epsilon = 1e-10); // 0.5 is NOT > 0.5
        assert_abs_diff_eq!(transformed[[1, 1]], 0.0, epsilon = 1e-10); // 0.0 <= 0.5
        assert_abs_diff_eq!(transformed[[1, 2]], 0.0, epsilon = 1e-10); // -0.5 <= 0.5
    }

    #[test]
    fn test_default_threshold_zero() {
        let x = array![[1.0, 0.0, -1.0], [0.1, -0.1, 0.0]];
        let binarizer = Binarizer::<f64>::default();
        let fitted = FitUnsupervised::<f64>::fit(&binarizer, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        // threshold = 0: values > 0 become 1
        assert_abs_diff_eq!(transformed[[0, 0]], 1.0, epsilon = 1e-10); // 1.0 > 0
        assert_abs_diff_eq!(transformed[[0, 1]], 0.0, epsilon = 1e-10); // 0.0 is NOT > 0
        assert_abs_diff_eq!(transformed[[0, 2]], 0.0, epsilon = 1e-10); // -1.0 <= 0
        assert_abs_diff_eq!(transformed[[1, 0]], 1.0, epsilon = 1e-10); // 0.1 > 0
        assert_abs_diff_eq!(transformed[[1, 1]], 0.0, epsilon = 1e-10); // -0.1 <= 0
        assert_abs_diff_eq!(transformed[[1, 2]], 0.0, epsilon = 1e-10); // 0.0 is NOT > 0
    }

    #[test]
    fn test_negative_threshold() {
        let x = array![[-2.0, -1.0, 0.0], [1.0, -0.5, 0.5]];
        let binarizer = Binarizer::new(-1.0);
        let fitted = FitUnsupervised::<f64>::fit(&binarizer, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        // threshold = -1.0: values > -1.0 become 1
        assert_abs_diff_eq!(transformed[[0, 0]], 0.0, epsilon = 1e-10); // -2.0 <= -1.0
        assert_abs_diff_eq!(transformed[[0, 1]], 0.0, epsilon = 1e-10); // -1.0 is NOT > -1.0
        assert_abs_diff_eq!(transformed[[0, 2]], 1.0, epsilon = 1e-10); // 0.0 > -1.0
        assert_abs_diff_eq!(transformed[[1, 0]], 1.0, epsilon = 1e-10); // 1.0 > -1.0
        assert_abs_diff_eq!(transformed[[1, 1]], 1.0, epsilon = 1e-10); // -0.5 > -1.0
    }

    #[test]
    fn test_all_ones_and_zeros() {
        // All values above threshold -> all ones
        let x = array![[10.0, 20.0], [30.0, 40.0]];
        let binarizer = Binarizer::new(0.0);
        let fitted = FitUnsupervised::<f64>::fit(&binarizer, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        for &v in transformed.iter() {
            assert_abs_diff_eq!(v, 1.0, epsilon = 1e-10);
        }

        // All values below threshold -> all zeros
        let x2 = array![[-10.0, -20.0], [-30.0, -40.0]];
        let transformed2 = fitted.transform(&x2).unwrap();

        for &v in transformed2.iter() {
            assert_abs_diff_eq!(v, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_f32_support() {
        let x = array![[1.0f32, -1.0, 0.5], [0.0, 2.0, -0.5]];
        let binarizer = Binarizer::new(0.0f32);
        let fitted = FitUnsupervised::<f32>::fit(&binarizer, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_abs_diff_eq!(transformed[[0, 0]], 1.0f32, epsilon = 1e-6);
        assert_abs_diff_eq!(transformed[[0, 1]], 0.0f32, epsilon = 1e-6);
    }

    #[test]
    fn test_empty_input() {
        let x: Array2<f64> = Array2::zeros((0, 0));
        let binarizer = Binarizer::<f64>::default();
        let result = FitUnsupervised::<f64>::fit(&binarizer, &x);
        assert!(result.is_err());
    }
}
