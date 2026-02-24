use ndarray::{Array1, Array2, Axis};
use rustml_core::{Float, FitUnsupervised, Result, RustMlError, Transform};

/// Parameters for `VarianceThreshold` feature selector (unfitted state).
///
/// Removes all features whose variance does not meet a minimum threshold.
/// By default (`threshold = 0.0`), it removes features that have zero variance,
/// i.e., features that are constant across all samples.
///
/// This is a simple baseline approach to feature selection: a feature with
/// higher variance is more likely to be informative (though this is not
/// guaranteed).
///
/// # Example
///
/// ```
/// use rustml_preprocessing::VarianceThreshold;
/// use rustml_core::{FitUnsupervised, Transform};
/// use ndarray::array;
///
/// let x = array![
///     [0.0, 2.0, 0.0],
///     [0.0, 4.0, 0.0],
///     [0.0, 6.0, 0.0],
/// ];
///
/// // Remove zero-variance features (columns 0 and 2 are constant)
/// let selector = VarianceThreshold::new(0.0);
/// let fitted = FitUnsupervised::<f64>::fit(&selector, &x).unwrap();
/// let x_selected = fitted.transform(&x).unwrap();
///
/// assert_eq!(x_selected.ncols(), 1); // only the varying column survives
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VarianceThreshold {
    /// Minimum variance required for a feature to be kept.
    /// Features with variance <= threshold are removed.
    pub threshold: f64,
}

impl VarianceThreshold {
    /// Create a new `VarianceThreshold` with the given threshold.
    ///
    /// A threshold of `0.0` removes only constant (zero-variance) features.
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }

    /// Set the variance threshold.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }
}

impl Default for VarianceThreshold {
    fn default() -> Self {
        Self::new(0.0)
    }
}

/// Fitted `VarianceThreshold` — holds learned per-feature variances and the
/// indices of features that exceeded the threshold.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedVarianceThreshold<F: Float> {
    /// Per-feature variance computed during fitting.
    variances: Array1<F>,
    /// Indices of features whose variance exceeded the threshold.
    selected_indices: Vec<usize>,
    /// Total number of input features (before selection).
    n_features_in: usize,
}

impl<F: Float> FittedVarianceThreshold<F> {
    /// Per-feature variances computed during fitting.
    pub fn variances(&self) -> &Array1<F> {
        &self.variances
    }

    /// Indices of selected features (those with variance > threshold).
    pub fn selected_indices(&self) -> &[usize] {
        &self.selected_indices
    }

    /// Number of features that survived selection.
    pub fn n_features_selected(&self) -> usize {
        self.selected_indices.len()
    }
}

impl<F: Float> FitUnsupervised<F> for VarianceThreshold {
    type Fitted = FittedVarianceThreshold<F>;

    fn fit(&self, x: &Array2<F>) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(RustMlError::EmptyInput("input array is empty".into()));
        }

        if self.threshold < 0.0 {
            return Err(RustMlError::InvalidParameter(
                "threshold must be non-negative".into(),
            ));
        }

        let n = F::from_usize(n_samples).unwrap();

        // Compute per-feature mean.
        let mean = x.sum_axis(Axis(0)) / n;

        // Compute per-feature variance: Var(X) = E[(X - mean)^2].
        let mut variances = Array1::<F>::zeros(n_features);
        for row in x.rows() {
            for (j, (&val, &m)) in row.iter().zip(mean.iter()).enumerate() {
                let diff = val - m;
                variances[j] += diff * diff;
            }
        }
        variances.mapv_inplace(|v| v / n);

        // Select features whose variance exceeds the threshold.
        let threshold_f = F::from_f64(self.threshold).unwrap();
        let selected_indices: Vec<usize> = (0..n_features)
            .filter(|&j| variances[j] > threshold_f)
            .collect();

        if selected_indices.is_empty() {
            return Err(RustMlError::InvalidParameter(
                "no features meet the variance threshold; all features have variance <= threshold"
                    .into(),
            ));
        }

        Ok(FittedVarianceThreshold {
            variances,
            selected_indices,
            n_features_in: n_features,
        })
    }
}

impl<F: Float> Transform<F> for FittedVarianceThreshold<F> {
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>> {
        if x.ncols() != self.n_features_in {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features_in,
                x.ncols()
            )));
        }

        let n_rows = x.nrows();
        let n_selected = self.selected_indices.len();
        let mut result = Array2::<F>::zeros((n_rows, n_selected));

        for (i, row) in x.rows().into_iter().enumerate() {
            for (out_j, &src_j) in self.selected_indices.iter().enumerate() {
                result[[i, out_j]] = row[src_j];
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
    fn test_removes_constant_features() {
        // Column 0 and 2 are constant, column 1 varies.
        let x = array![
            [5.0, 1.0, 3.0],
            [5.0, 2.0, 3.0],
            [5.0, 3.0, 3.0],
            [5.0, 4.0, 3.0],
        ];

        let selector = VarianceThreshold::default();
        let fitted = FitUnsupervised::<f64>::fit(&selector, &x).unwrap();

        assert_eq!(fitted.selected_indices(), &[1]);
        assert_eq!(fitted.n_features_selected(), 1);

        // Constant columns should have variance 0.
        assert_abs_diff_eq!(fitted.variances()[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(fitted.variances()[2], 0.0, epsilon = 1e-10);
        assert!(fitted.variances()[1] > 0.0);
    }

    #[test]
    fn test_higher_threshold_removes_low_variance() {
        // Col 0: values 1,2,3,4 -> var = 1.25
        // Col 1: values 10,20,30,40 -> var = 125.0
        // Col 2: values 0,0,0,1 -> var = 0.1875
        let x = array![
            [1.0, 10.0, 0.0],
            [2.0, 20.0, 0.0],
            [3.0, 30.0, 0.0],
            [4.0, 40.0, 1.0],
        ];

        // Threshold = 1.0 should remove col 2 (var=0.1875) but keep col 0 (var=1.25)
        let selector = VarianceThreshold::new(1.0);
        let fitted = FitUnsupervised::<f64>::fit(&selector, &x).unwrap();

        assert_eq!(fitted.selected_indices(), &[0, 1]);

        // Threshold = 2.0 should keep only col 1 (var=125.0)
        let selector = VarianceThreshold::new(2.0);
        let fitted = FitUnsupervised::<f64>::fit(&selector, &x).unwrap();

        assert_eq!(fitted.selected_indices(), &[1]);
    }

    #[test]
    fn test_transform_outputs_correct_shape() {
        let x = array![
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 4.0, 5.0, 6.0],
            [0.0, 7.0, 8.0, 9.0],
        ];

        let selector = VarianceThreshold::new(0.0);
        let fitted = FitUnsupervised::<f64>::fit(&selector, &x).unwrap();
        let result = fitted.transform(&x).unwrap();

        // Column 0 is constant -> removed; columns 1,2,3 survive.
        assert_eq!(result.dim(), (3, 3));

        // Verify the selected columns contain the right data.
        assert_abs_diff_eq!(result[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[0, 1]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[0, 2]], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[2, 0]], 7.0, epsilon = 1e-10);
    }

    #[test]
    fn test_keeps_all_features_when_all_vary() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let selector = VarianceThreshold::new(0.0);
        let fitted = FitUnsupervised::<f64>::fit(&selector, &x).unwrap();

        assert_eq!(fitted.selected_indices(), &[0, 1]);
        let result = fitted.transform(&x).unwrap();
        assert_eq!(result.dim(), (3, 2));
    }

    #[test]
    fn test_error_when_no_features_survive() {
        // All features are constant.
        let x = array![[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]];

        let selector = VarianceThreshold::new(0.0);
        let result = FitUnsupervised::<f64>::fit(&selector, &x);

        assert!(result.is_err());
        match result.unwrap_err() {
            RustMlError::InvalidParameter(msg) => {
                assert!(msg.contains("no features"), "unexpected message: {}", msg);
            }
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }

    #[test]
    fn test_error_on_empty_input() {
        let x = Array2::<f64>::zeros((0, 3));

        let selector = VarianceThreshold::new(0.0);
        let result = FitUnsupervised::<f64>::fit(&selector, &x);

        assert!(result.is_err());
    }

    #[test]
    fn test_shape_mismatch_on_transform() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        let selector = VarianceThreshold::new(0.0);
        let fitted = FitUnsupervised::<f64>::fit(&selector, &x).unwrap();

        let wrong = array![[1.0, 2.0]]; // 2 cols instead of 3
        assert!(fitted.transform(&wrong).is_err());
    }

    #[test]
    fn test_works_with_f32() {
        let x: Array2<f32> = array![[0.0_f32, 1.0], [0.0, 2.0], [0.0, 3.0]];

        let selector = VarianceThreshold::new(0.0);
        let fitted = FitUnsupervised::<f32>::fit(&selector, &x).unwrap();

        assert_eq!(fitted.selected_indices(), &[1]);
        let result = fitted.transform(&x).unwrap();
        assert_eq!(result.dim(), (3, 1));
    }
}
