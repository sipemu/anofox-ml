use ndarray::{Array1, Array2, Axis};
use rustml_core::{Fit, Float, Predict, Result, RustMlError};

/// Gaussian Naive Bayes classifier parameters (unfitted state).
///
/// Uses the type-state pattern: call [`Fit::fit`] to produce a
/// [`FittedGaussianNB`] that can make predictions.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GaussianNB {
    /// Portion of the largest variance of all features that is added to
    /// variances for calculation stability.
    pub var_smoothing: f64,
}

impl GaussianNB {
    /// Create a new `GaussianNB` with default parameters.
    pub fn new() -> Self {
        Self {
            var_smoothing: 1e-9,
        }
    }

    /// Set the variance smoothing factor added to variances for calculation stability.
    pub fn with_var_smoothing(mut self, var_smoothing: f64) -> Self {
        self.var_smoothing = var_smoothing;
        self
    }
}

impl Default for GaussianNB {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Gaussian Naive Bayes classifier.
///
/// Stores learned class priors, per-class feature means (theta), and
/// per-class feature variances (sigma) needed for prediction.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedGaussianNB<F: Float> {
    /// Unique sorted class labels.
    class_labels: Vec<F>,
    /// Prior probability of each class (count / total).
    class_prior: Vec<F>,
    /// Mean of each feature per class, shape `(n_classes, n_features)`.
    theta: Array2<F>,
    /// Variance of each feature per class + var_smoothing, shape `(n_classes, n_features)`.
    sigma: Array2<F>,
}

impl<F: Float> FittedGaussianNB<F> {
    /// Returns the prior probability of each class.
    pub fn class_prior(&self) -> &[F] {
        &self.class_prior
    }

    /// Returns the per-class feature means, shape `(n_classes, n_features)`.
    pub fn theta(&self) -> &Array2<F> {
        &self.theta
    }

    /// Returns the per-class feature variances (with smoothing), shape `(n_classes, n_features)`.
    pub fn sigma(&self) -> &Array2<F> {
        &self.sigma
    }

    /// Returns the unique sorted class labels.
    pub fn class_labels(&self) -> &[F] {
        &self.class_labels
    }
}

impl<F: Float> Fit<F> for GaussianNB {
    type Fitted = FittedGaussianNB<F>;

    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Self::Fitted> {
        if x.is_empty() || y.is_empty() {
            return Err(RustMlError::EmptyInput(
                "training data must not be empty".into(),
            ));
        }
        if x.nrows() != y.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but y has {} elements",
                x.nrows(),
                y.len()
            )));
        }

        let n_samples = F::from_usize(x.nrows()).unwrap();
        let n_features = x.ncols();
        let smoothing = F::from_f64(self.var_smoothing).unwrap();

        // Extract unique class labels and sort them.
        let mut class_labels: Vec<F> = y.to_vec();
        class_labels.sort_by(|a, b| a.partial_cmp(b).unwrap());
        class_labels.dedup_by(|a, b| (*a - *b).abs() < F::from_f64(1e-12).unwrap());

        let n_classes = class_labels.len();
        let mut theta = Array2::<F>::zeros((n_classes, n_features));
        let mut sigma = Array2::<F>::zeros((n_classes, n_features));
        let mut class_prior = Vec::with_capacity(n_classes);

        for (ci, &label) in class_labels.iter().enumerate() {
            // Gather row indices belonging to this class.
            let mask: Vec<usize> = y
                .iter()
                .enumerate()
                .filter(|(_, &val)| (val - label).abs() < F::from_f64(1e-12).unwrap())
                .map(|(i, _)| i)
                .collect();

            let count = F::from_usize(mask.len()).unwrap();
            class_prior.push(count / n_samples);

            // Build sub-matrix for this class.
            let class_x = x.select(Axis(0), &mask);

            // Compute per-feature mean.
            let mean = class_x.mean_axis(Axis(0)).unwrap();

            // Compute per-feature variance: E[(x - mu)^2].
            let diff = &class_x - &mean;
            let var = diff.mapv(|v| v * v).mean_axis(Axis(0)).unwrap();

            theta.row_mut(ci).assign(&mean);
            sigma.row_mut(ci).assign(&(var + smoothing));
        }

        Ok(FittedGaussianNB {
            class_labels,
            class_prior,
            theta,
            sigma,
        })
    }
}

impl<F: Float> Predict<F> for FittedGaussianNB<F> {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        if x.ncols() != self.theta.ncols() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.theta.ncols(),
                x.ncols()
            )));
        }

        let two = F::from_f64(2.0).unwrap();
        let two_pi = two * F::from_f64(std::f64::consts::PI).unwrap();
        let half = F::from_f64(0.5).unwrap();

        let mut predictions = Array1::<F>::zeros(x.nrows());

        for (i, sample) in x.rows().into_iter().enumerate() {
            let mut best_class = self.class_labels[0];
            let mut best_log_posterior = F::neg_infinity();

            for (ci, &label) in self.class_labels.iter().enumerate() {
                let log_prior = self.class_prior[ci].ln();

                // Sum of log-likelihoods over features:
                // -0.5 * log(2*pi*sigma) - 0.5 * (x - mu)^2 / sigma
                let mut log_likelihood = F::zero();
                for j in 0..x.ncols() {
                    let mu = self.theta[[ci, j]];
                    let var = self.sigma[[ci, j]];
                    let diff = sample[j] - mu;
                    log_likelihood = log_likelihood
                        - half * (two_pi * var).ln()
                        - half * diff * diff / var;
                }

                let log_posterior = log_prior + log_likelihood;
                if log_posterior > best_log_posterior {
                    best_log_posterior = log_posterior;
                    best_class = label;
                }
            }

            predictions[i] = best_class;
        }

        Ok(predictions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_two_class_well_separated() {
        // Two well-separated Gaussian clusters.
        let x_train = array![
            [1.0, 1.0],
            [1.1, 1.1],
            [0.9, 0.9],
            [1.0, 1.2],
            [10.0, 10.0],
            [10.1, 10.1],
            [9.9, 9.9],
            [10.0, 10.2]
        ];
        let y_train = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let nb = GaussianNB::default();
        let fitted: FittedGaussianNB<f64> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        // Verify class labels.
        assert_eq!(fitted.class_labels(), &[0.0, 1.0]);

        // Verify equal priors.
        assert_abs_diff_eq!(fitted.class_prior()[0], 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(fitted.class_prior()[1], 0.5, epsilon = 1e-10);

        // Predict points near each cluster center.
        let x_test = array![[1.0, 1.0], [10.0, 10.0], [0.5, 0.5], [9.5, 9.5]];
        let preds = fitted.predict(&x_test).unwrap();

        assert_abs_diff_eq!(preds[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[1], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[2], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[3], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_predictions_known_data() {
        // Simple 1-D data: class 0 centred at 0, class 1 centred at 5.
        let x_train = array![[0.0], [0.5], [-0.5], [5.0], [5.5], [4.5]];
        let y_train = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let nb = GaussianNB::default();
        let fitted: FittedGaussianNB<f64> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        // Means should be approximately 0.0 and 5.0.
        assert_abs_diff_eq!(fitted.theta()[[0, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(fitted.theta()[[1, 0]], 5.0, epsilon = 1e-10);

        // A point at 1.0 should be class 0; at 4.0 should be class 1.
        let x_test = array![[1.0], [4.0]];
        let preds = fitted.predict(&x_test).unwrap();
        assert_abs_diff_eq!(preds[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_empty_input_error() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<f64>::zeros(0);

        let nb = GaussianNB::default();
        let result: Result<FittedGaussianNB<f64>> = Fit::fit(&nb, &x, &y);
        assert!(result.is_err());

        match result {
            Err(RustMlError::EmptyInput(_)) => {}
            other => panic!("expected EmptyInput error, got {:?}", other),
        }
    }

    #[test]
    fn test_shape_mismatch_fit() {
        // X has 2 rows, y has 3 elements.
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0, 0.0];

        let nb = GaussianNB::default();
        let result: Result<FittedGaussianNB<f64>> = Fit::fit(&nb, &x, &y);
        assert!(result.is_err());

        match result {
            Err(RustMlError::ShapeMismatch(_)) => {}
            other => panic!("expected ShapeMismatch error, got {:?}", other),
        }
    }

    #[test]
    fn test_shape_mismatch_predict() {
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let y_train = array![0.0, 1.0];

        let nb = GaussianNB::default();
        let fitted: FittedGaussianNB<f64> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        // Predict with wrong number of features (3 instead of 2).
        let x_test = array![[1.0, 2.0, 3.0]];
        let result = fitted.predict(&x_test);
        assert!(result.is_err());

        match result {
            Err(RustMlError::ShapeMismatch(_)) => {}
            other => panic!("expected ShapeMismatch error, got {:?}", other),
        }
    }
}
