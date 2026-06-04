use ndarray::{Array1, Array2, Axis};
use rustml_core::{Fit, Float, Predict, PredictProba, Result, RustMlError};

/// Multinomial Naive Bayes classifier parameters (unfitted state).
///
/// Suitable for count-based or TF-IDF feature representations where all
/// feature values must be non-negative.  Uses Laplace (additive) smoothing
/// controlled by the `alpha` parameter.
///
/// Uses the type-state pattern: call [`Fit::fit`] to produce a
/// [`FittedMultinomialNB`] that can make predictions.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MultinomialNB {
    /// Additive (Laplace) smoothing parameter (>= 0).
    pub alpha: f64,
}

impl MultinomialNB {
    /// Create a new `MultinomialNB` with default parameters (`alpha = 1.0`).
    pub fn new() -> Self {
        Self { alpha: 1.0 }
    }

    /// Set the additive smoothing parameter.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }
}

impl Default for MultinomialNB {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Multinomial Naive Bayes classifier.
///
/// Stores log class priors and log feature probabilities learned from
/// training data.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedMultinomialNB<F: Float> {
    /// Unique sorted class labels.
    class_labels: Vec<F>,
    /// Log prior probability of each class, shape `(n_classes,)`.
    log_class_prior: Vec<F>,
    /// Log probability of each feature per class, shape `(n_classes, n_features)`.
    log_feature_prob: Array2<F>,
}

impl<F: Float> FittedMultinomialNB<F> {
    /// Returns the unique sorted class labels.
    pub fn classes(&self) -> &[F] {
        &self.class_labels
    }

    /// Returns the log prior probability of each class.
    pub fn log_class_prior(&self) -> &[F] {
        &self.log_class_prior
    }

    /// Returns the log feature probabilities, shape `(n_classes, n_features)`.
    pub fn log_feature_prob(&self) -> &Array2<F> {
        &self.log_feature_prob
    }

    /// Predict class probabilities for each sample in `x`.
    ///
    /// Returns an array of shape `(n_samples, n_classes)` where each row sums
    /// to 1.
    pub fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>> {
        if x.ncols() != self.log_feature_prob.ncols() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.log_feature_prob.ncols(),
                x.ncols()
            )));
        }

        let n_samples = x.nrows();
        let n_classes = self.class_labels.len();
        let mut proba = Array2::<F>::zeros((n_samples, n_classes));

        for (i, sample) in x.rows().into_iter().enumerate() {
            // Compute log posteriors (unnormalized).
            let mut log_posteriors = Vec::with_capacity(n_classes);
            for ci in 0..n_classes {
                let mut log_post = self.log_class_prior[ci];
                for j in 0..x.ncols() {
                    log_post = log_post + self.log_feature_prob[[ci, j]] * sample[j];
                }
                log_posteriors.push(log_post);
            }

            // Log-sum-exp for numerical stability.
            let max_log = log_posteriors
                .iter()
                .copied()
                .fold(F::neg_infinity(), |a, b| if a > b { a } else { b });

            let sum_exp: F = log_posteriors
                .iter()
                .map(|&lp| (lp - max_log).exp())
                .fold(F::zero(), |a, b| a + b);
            let log_norm = max_log + sum_exp.ln();

            for ci in 0..n_classes {
                proba[[i, ci]] = (log_posteriors[ci] - log_norm).exp();
            }
        }

        Ok(proba)
    }
}

impl<F: Float> Fit<F> for MultinomialNB {
    type Fitted = FittedMultinomialNB<F>;

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

        // Validate all features are non-negative.
        let zero = F::zero();
        for val in x.iter() {
            if *val < zero {
                return Err(RustMlError::InvalidParameter(
                    "MultinomialNB requires all feature values to be non-negative".into(),
                ));
            }
        }

        let alpha = F::from_f64(self.alpha).unwrap();
        let n_samples = F::from_usize(x.nrows()).unwrap();
        let n_features = x.ncols();
        let n_features_f = F::from_usize(n_features).unwrap();

        // Extract unique class labels and sort them.
        let mut class_labels: Vec<F> = y.to_vec();
        class_labels.sort_by(|a, b| a.partial_cmp(b).unwrap());
        class_labels.dedup_by(|a, b| (*a - *b).abs() < F::from_f64(1e-12).unwrap());

        let n_classes = class_labels.len();
        let mut log_class_prior = Vec::with_capacity(n_classes);
        let mut log_feature_prob = Array2::<F>::zeros((n_classes, n_features));

        for (ci, &label) in class_labels.iter().enumerate() {
            // Gather row indices belonging to this class.
            let mask: Vec<usize> = y
                .iter()
                .enumerate()
                .filter(|(_, &val)| (val - label).abs() < F::from_f64(1e-12).unwrap())
                .map(|(i, _)| i)
                .collect();

            let count = F::from_usize(mask.len()).unwrap();
            log_class_prior.push((count / n_samples).ln());

            // Build sub-matrix for this class.
            let class_x = x.select(Axis(0), &mask);

            // Sum of each feature across all samples in this class.
            let feature_counts = class_x.sum_axis(Axis(0));

            // Total count across all features for this class (with smoothing).
            let total_count: F = feature_counts.iter().copied().fold(F::zero(), |a, b| a + b)
                + alpha * n_features_f;

            // Log probability with Laplace smoothing:
            // log P(x_j | c) = log((count_j + alpha) / (total + alpha * n_features))
            for j in 0..n_features {
                log_feature_prob[[ci, j]] = ((feature_counts[j] + alpha) / total_count).ln();
            }
        }

        Ok(FittedMultinomialNB {
            class_labels,
            log_class_prior,
            log_feature_prob,
        })
    }
}

impl<F: Float> Predict<F> for FittedMultinomialNB<F> {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        if x.ncols() != self.log_feature_prob.ncols() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.log_feature_prob.ncols(),
                x.ncols()
            )));
        }

        let mut predictions = Array1::<F>::zeros(x.nrows());

        for (i, sample) in x.rows().into_iter().enumerate() {
            let mut best_class = self.class_labels[0];
            let mut best_log_posterior = F::neg_infinity();

            for (ci, &label) in self.class_labels.iter().enumerate() {
                let mut log_posterior = self.log_class_prior[ci];
                for j in 0..x.ncols() {
                    log_posterior =
                        log_posterior + self.log_feature_prob[[ci, j]] * sample[j];
                }

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
    fn test_basic_classification() {
        // Simple count data: class 0 has high counts in feature 0,
        // class 1 has high counts in feature 1.
        let x_train = array![
            [5.0, 0.0],
            [4.0, 1.0],
            [6.0, 0.0],
            [0.0, 5.0],
            [1.0, 4.0],
            [0.0, 6.0]
        ];
        let y_train = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let nb = MultinomialNB::new();
        let fitted: FittedMultinomialNB<f64> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        let x_test = array![[5.0, 0.0], [0.0, 5.0]];
        let preds = fitted.predict(&x_test).unwrap();

        assert_abs_diff_eq!(preds[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_predict_proba_sums_to_one() {
        let x_train = array![
            [3.0, 1.0],
            [4.0, 0.0],
            [1.0, 3.0],
            [0.0, 4.0]
        ];
        let y_train = array![0.0, 0.0, 1.0, 1.0];

        let nb = MultinomialNB::new();
        let fitted: FittedMultinomialNB<f64> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        let x_test = array![[2.0, 2.0], [5.0, 0.0], [0.0, 5.0]];
        let proba = fitted.predict_proba(&x_test).unwrap();

        // Each row should sum to 1.
        for i in 0..proba.nrows() {
            let row_sum: f64 = proba.row(i).iter().sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }

        // Probabilities should be in [0, 1].
        for &p in proba.iter() {
            assert!(p >= 0.0 && p <= 1.0, "probability {} out of range", p);
        }
    }

    #[test]
    fn test_alpha_smoothing_effect() {
        // With very small alpha, probabilities should be more extreme.
        let x_train = array![
            [10.0, 0.0],
            [0.0, 10.0]
        ];
        let y_train = array![0.0, 1.0];

        let nb_small = MultinomialNB::new().with_alpha(1e-10);
        let fitted_small: FittedMultinomialNB<f64> =
            Fit::fit(&nb_small, &x_train, &y_train).unwrap();

        let nb_large = MultinomialNB::new().with_alpha(100.0);
        let fitted_large: FittedMultinomialNB<f64> =
            Fit::fit(&nb_large, &x_train, &y_train).unwrap();

        let x_test = array![[10.0, 0.0]];
        let proba_small = fitted_small.predict_proba(&x_test).unwrap();
        let proba_large = fitted_large.predict_proba(&x_test).unwrap();

        // With small alpha, class 0 probability should be very high.
        assert!(
            proba_small[[0, 0]] > proba_large[[0, 0]],
            "smaller alpha should give more extreme probabilities"
        );
    }

    #[test]
    fn test_negative_feature_validation() {
        let x_train = array![[1.0, -1.0], [2.0, 3.0]];
        let y_train = array![0.0, 1.0];

        let nb = MultinomialNB::new();
        let result: Result<FittedMultinomialNB<f64>> = Fit::fit(&nb, &x_train, &y_train);
        assert!(result.is_err());

        match result {
            Err(RustMlError::InvalidParameter(msg)) => {
                assert!(
                    msg.contains("non-negative"),
                    "error should mention non-negative: {}",
                    msg
                );
            }
            other => panic!("expected InvalidParameter error, got {:?}", other),
        }
    }

    #[test]
    fn test_multiclass() {
        // Three classes with distinctive feature patterns.
        let x_train = array![
            [5.0, 0.0, 0.0],
            [4.0, 1.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 4.0, 1.0],
            [0.0, 0.0, 5.0],
            [1.0, 0.0, 4.0]
        ];
        let y_train = array![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];

        let nb = MultinomialNB::new();
        let fitted: FittedMultinomialNB<f64> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        assert_eq!(fitted.classes().len(), 3);

        let x_test = array![[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]];
        let preds = fitted.predict(&x_test).unwrap();

        assert_abs_diff_eq!(preds[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[1], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[2], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_shape_errors() {
        // Fit with mismatched X rows and y length.
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0, 0.0];

        let nb = MultinomialNB::new();
        let result: Result<FittedMultinomialNB<f64>> = Fit::fit(&nb, &x, &y);
        assert!(result.is_err());
        match result {
            Err(RustMlError::ShapeMismatch(_)) => {}
            other => panic!("expected ShapeMismatch error, got {:?}", other),
        }

        // Predict with wrong number of features.
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let y_train = array![0.0, 1.0];
        let fitted: FittedMultinomialNB<f64> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        let x_test = array![[1.0, 2.0, 3.0]];
        let result = fitted.predict(&x_test);
        assert!(result.is_err());
        match result {
            Err(RustMlError::ShapeMismatch(_)) => {}
            other => panic!("expected ShapeMismatch error, got {:?}", other),
        }
    }

    #[test]
    fn test_empty_input_error() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<f64>::zeros(0);

        let nb = MultinomialNB::new();
        let result: Result<FittedMultinomialNB<f64>> = Fit::fit(&nb, &x, &y);
        assert!(result.is_err());
        match result {
            Err(RustMlError::EmptyInput(_)) => {}
            other => panic!("expected EmptyInput error, got {:?}", other),
        }
    }

    #[test]
    fn test_f32_support() {
        let x_train: Array2<f32> = array![
            [5.0f32, 0.0],
            [4.0, 1.0],
            [0.0, 5.0],
            [1.0, 4.0]
        ];
        let y_train: Array1<f32> = array![0.0f32, 0.0, 1.0, 1.0];

        let nb = MultinomialNB::new();
        let fitted: FittedMultinomialNB<f32> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        let x_test: Array2<f32> = array![[5.0f32, 0.0], [0.0, 5.0]];
        let preds = fitted.predict(&x_test).unwrap();

        assert_abs_diff_eq!(preds[0], 0.0f32, epsilon = 1e-5);
        assert_abs_diff_eq!(preds[1], 1.0f32, epsilon = 1e-5);
    }

    #[test]
    fn test_predict_proba_shape_error() {
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let y_train = array![0.0, 1.0];

        let nb = MultinomialNB::new();
        let fitted: FittedMultinomialNB<f64> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        let x_test = array![[1.0]];
        let result = fitted.predict_proba(&x_test);
        assert!(result.is_err());
        match result {
            Err(RustMlError::ShapeMismatch(_)) => {}
            other => panic!("expected ShapeMismatch error, got {:?}", other),
        }
    }
}

impl<F: Float> PredictProba<F> for FittedMultinomialNB<F> {
    fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>> {
        Self::predict_proba(self, x)
    }
}
