use ndarray::{Array1, Array2};
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rustml_core::{Fit, Float, Predict, PredictProba, Result, RustMlError};
use rustml_trees::{DecisionTreeClassifier, FittedDecisionTreeClassifier, SplitCriterion};

/// AdaBoost classifier parameters (unfitted state).
///
/// Implements the SAMME (Stagewise Additive Modeling using a Multi-class
/// Exponential loss function) algorithm. Each boosting round fits a decision
/// tree (typically a stump with `max_depth = 1`) on a weighted bootstrap
/// sample, then adjusts sample weights so that misclassified examples receive
/// greater emphasis in subsequent rounds.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AdaBoostClassifier {
    /// Number of boosting rounds (weak learners).
    pub n_estimators: usize,
    /// Learning rate that shrinks each estimator's contribution.
    pub learning_rate: f64,
    /// Maximum depth of each decision tree. `Some(1)` yields stumps.
    pub max_depth: Option<usize>,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl AdaBoostClassifier {
    /// Create a new `AdaBoostClassifier` with default parameters.
    pub fn new() -> Self {
        Self {
            n_estimators: 50,
            learning_rate: 1.0,
            max_depth: Some(1),
            seed: 0,
        }
    }

    /// Set the number of boosting rounds.
    pub fn with_n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators;
        self
    }

    /// Set the learning rate (shrinkage).
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the maximum depth of each tree.
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set the random seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl Default for AdaBoostClassifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted AdaBoost classifier.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedAdaBoostClassifier<F: Float> {
    /// Weak learners.
    estimators: Vec<FittedDecisionTreeClassifier<F>>,
    /// Weight (alpha) for each estimator.
    estimator_weights: Vec<F>,
    /// Unique class labels sorted in ascending order.
    classes: Vec<F>,
    /// Number of features expected at prediction time.
    n_features: usize,
}

impl<F: Float> Fit<F> for AdaBoostClassifier {
    type Fitted = FittedAdaBoostClassifier<F>;

    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Self::Fitted> {
        if x.nrows() != y.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but y has {} elements",
                x.nrows(),
                y.len()
            )));
        }
        if x.is_empty() {
            return Err(RustMlError::EmptyInput("training data is empty".into()));
        }
        if self.n_estimators == 0 {
            return Err(RustMlError::InvalidParameter(
                "n_estimators must be > 0".into(),
            ));
        }
        if self.learning_rate <= 0.0 {
            return Err(RustMlError::InvalidParameter(
                "learning_rate must be > 0".into(),
            ));
        }

        let classes = unique_sorted(y);
        let n_classes = classes.len();
        if n_classes < 2 {
            return Err(RustMlError::InvalidParameter(
                "y must contain at least 2 distinct classes".into(),
            ));
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();
        let lr = F::from_f64(self.learning_rate).unwrap();

        let tree_params = DecisionTreeClassifier {
            max_depth: self.max_depth,
            min_samples_split: 2,
            min_samples_leaf: 1,
            criterion: SplitCriterion::Gini,
            max_features: None,
            sample_weight: None,
            class_weight: None,
        };

        let mut rng = StdRng::seed_from_u64(self.seed);

        // Initialize sample weights uniformly.
        let mut weights: Vec<F> = vec![F::one() / F::from_usize(n_samples).unwrap(); n_samples];

        let mut estimators = Vec::with_capacity(self.n_estimators);
        let mut estimator_weights = Vec::with_capacity(self.n_estimators);

        let n_classes_f = F::from_usize(n_classes).unwrap();
        let eps = F::from_f64(1e-15).unwrap();

        for _ in 0..self.n_estimators {
            // Create weighted bootstrap sample.
            let weights_f64: Vec<f64> = weights.iter().map(|w| w.to_f64().unwrap()).collect();
            let dist = WeightedIndex::new(&weights_f64).map_err(|e| {
                RustMlError::InvalidParameter(format!("invalid sample weights: {e}"))
            })?;

            let bootstrap_indices: Vec<usize> =
                (0..n_samples).map(|_| dist.sample(&mut rng)).collect();

            let x_bootstrap = build_sub_rows(x, &bootstrap_indices);
            let y_bootstrap =
                Array1::from_vec(bootstrap_indices.iter().map(|&i| y[i]).collect::<Vec<F>>());

            // Fit weak learner on the bootstrap sample.
            let fitted_tree: FittedDecisionTreeClassifier<F> =
                tree_params.fit(&x_bootstrap, &y_bootstrap)?;

            // Predict on the *full* training set.
            let preds = fitted_tree.predict(x)?;

            // Compute weighted error.
            let mut err = F::zero();
            let mut w_sum = F::zero();
            for i in 0..n_samples {
                w_sum = w_sum + weights[i];
                if !approx_eq(preds[i], y[i], eps) {
                    err = err + weights[i];
                }
            }
            err = err / w_sum;

            // If error is too high, stop early.
            let err_threshold = F::one() - F::one() / n_classes_f;
            if err >= err_threshold {
                // If this is the first estimator, keep it with zero weight
                // so we still have something to predict with.
                if estimators.is_empty() {
                    estimators.push(fitted_tree);
                    estimator_weights.push(F::zero());
                }
                break;
            }

            // Compute estimator weight (SAMME formula).
            let alpha =
                lr * (((F::one() - err) / (err + eps)).ln() + (n_classes_f - F::one()).ln());

            // Update sample weights.
            for i in 0..n_samples {
                if !approx_eq(preds[i], y[i], eps) {
                    weights[i] = weights[i] * alpha.exp();
                }
            }

            // Normalize weights.
            let w_total: F = weights.iter().copied().fold(F::zero(), |a, b| a + b);
            if w_total > F::zero() {
                for w in &mut weights {
                    *w = *w / w_total;
                }
            }

            estimators.push(fitted_tree);
            estimator_weights.push(alpha);
        }

        Ok(FittedAdaBoostClassifier {
            estimators,
            estimator_weights,
            classes,
            n_features,
        })
    }
}

impl<F: Float> Predict<F> for FittedAdaBoostClassifier<F> {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }

        let n_samples = x.nrows();
        let mut predictions = Vec::with_capacity(n_samples);
        let eps = F::from_f64(1e-9).unwrap();

        for i in 0..n_samples {
            let row = x.row(i);
            let row_arr = row.to_owned().insert_axis(ndarray::Axis(0));

            // Compute weighted vote for each class.
            let mut class_scores: Vec<F> = vec![F::zero(); self.classes.len()];
            for (tree, &alpha) in self.estimators.iter().zip(self.estimator_weights.iter()) {
                let pred = tree.predict(&row_arr)?;
                let pred_val = pred[0];
                // Find which class this prediction corresponds to.
                for (k, &cls) in self.classes.iter().enumerate() {
                    if (pred_val - cls).abs() < eps {
                        class_scores[k] = class_scores[k] + alpha;
                        break;
                    }
                }
            }

            // Pick class with maximum weighted vote.
            let mut best_class = 0;
            let mut best_score = class_scores[0];
            for (k, &score) in class_scores.iter().enumerate().skip(1) {
                if score > best_score {
                    best_score = score;
                    best_class = k;
                }
            }
            predictions.push(self.classes[best_class]);
        }

        Ok(Array1::from_vec(predictions))
    }
}

impl<F: Float> FittedAdaBoostClassifier<F> {
    /// Number of estimators in the ensemble.
    pub fn n_estimators(&self) -> usize {
        self.estimators.len()
    }

    /// The unique classes discovered during training.
    pub fn classes(&self) -> &[F] {
        &self.classes
    }

    /// The weight of each estimator.
    pub fn estimator_weights(&self) -> &[F] {
        &self.estimator_weights
    }

    /// Predict class probabilities for each sample.
    ///
    /// For each sample, computes the weighted vote for every class and
    /// normalizes so that the probabilities sum to 1.
    /// Returns an `Array2` of shape `(n_samples, n_classes)`.
    pub fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }

        let n_samples = x.nrows();
        let n_classes = self.classes.len();
        let eps = F::from_f64(1e-9).unwrap();

        let mut proba = Array2::<F>::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            let row = x.row(i);
            let row_arr = row.to_owned().insert_axis(ndarray::Axis(0));

            for (tree, &alpha) in self.estimators.iter().zip(self.estimator_weights.iter()) {
                let pred = tree.predict(&row_arr)?;
                let pred_val = pred[0];
                for (k, &cls) in self.classes.iter().enumerate() {
                    if (pred_val - cls).abs() < eps {
                        proba[[i, k]] = proba[[i, k]] + alpha;
                        break;
                    }
                }
            }

            // Normalize row to sum to 1.
            let row_sum: F = (0..n_classes)
                .map(|k| proba[[i, k]])
                .fold(F::zero(), |a, b| a + b);
            if row_sum > F::zero() {
                for k in 0..n_classes {
                    proba[[i, k]] = proba[[i, k]] / row_sum;
                }
            }
        }

        Ok(proba)
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Check approximate equality between two float values.
#[inline]
fn approx_eq<F: Float>(a: F, b: F, eps: F) -> bool {
    (a - b).abs() < eps
}

/// Return sorted unique values from an array.
fn unique_sorted<F: Float>(arr: &Array1<F>) -> Vec<F> {
    let eps = F::from_f64(1e-9).unwrap();
    let mut vals: Vec<F> = arr.to_vec();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    vals.dedup_by(|a, b| (*a - *b).abs() < eps);
    vals
}

/// Build a sub-matrix selecting specific rows from `x`.
fn build_sub_rows<F: Float>(x: &Array2<F>, row_indices: &[usize]) -> Array2<F> {
    x.select(ndarray::Axis(0), row_indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_basic_binary_classification() {
        let x = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [10.0, 1.0],
            [11.0, 1.0],
            [12.0, 1.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let ada = AdaBoostClassifier {
            n_estimators: 50,
            learning_rate: 1.0,
            max_depth: Some(1),
            seed: 42,
        };
        let fitted: FittedAdaBoostClassifier<f64> = ada.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(*p, *t, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_multiclass_classification() {
        let x = array![
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [5.0, 5.0],
            [5.5, 5.0],
            [6.0, 5.0],
            [10.0, 10.0],
            [10.5, 10.0],
            [11.0, 10.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];

        let ada = AdaBoostClassifier {
            n_estimators: 100,
            learning_rate: 1.0,
            max_depth: Some(2),
            seed: 42,
        };
        let fitted: FittedAdaBoostClassifier<f64> = ada.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(*p, *t, epsilon = 1e-10);
        }

        assert_eq!(fitted.classes().len(), 3);
    }

    #[test]
    fn test_reproducibility() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let ada = AdaBoostClassifier {
            n_estimators: 20,
            learning_rate: 1.0,
            max_depth: Some(1),
            seed: 123,
        };

        let fitted1: FittedAdaBoostClassifier<f64> = ada.fit(&x, &y).unwrap();
        let fitted2: FittedAdaBoostClassifier<f64> = ada.fit(&x, &y).unwrap();

        let preds1 = fitted1.predict(&x).unwrap();
        let preds2 = fitted2.predict(&x).unwrap();

        for (a, b) in preds1.iter().zip(preds2.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-15);
        }
    }

    #[test]
    fn test_predict_proba_sums_to_one() {
        let x = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [10.0, 1.0],
            [11.0, 1.0],
            [12.0, 1.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let ada = AdaBoostClassifier {
            n_estimators: 20,
            learning_rate: 1.0,
            max_depth: Some(1),
            seed: 42,
        };
        let fitted: FittedAdaBoostClassifier<f64> = ada.fit(&x, &y).unwrap();

        let proba = fitted.predict_proba(&x).unwrap();
        assert_eq!(proba.nrows(), x.nrows());
        assert_eq!(proba.ncols(), 2);

        for i in 0..proba.nrows() {
            let row_sum: f64 = (0..proba.ncols()).map(|k| proba[[i, k]]).sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_predict_proba_high_confidence() {
        let x = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [10.0, 1.0],
            [11.0, 1.0],
            [12.0, 1.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let ada = AdaBoostClassifier {
            n_estimators: 50,
            learning_rate: 1.0,
            max_depth: Some(1),
            seed: 42,
        };
        let fitted: FittedAdaBoostClassifier<f64> = ada.fit(&x, &y).unwrap();

        let proba = fitted.predict_proba(&x).unwrap();
        // Class 0 samples should have high probability for class 0.
        for i in 0..3 {
            assert!(
                proba[[i, 0]] > 0.5,
                "expected high prob for class 0 at sample {i}"
            );
        }
        // Class 1 samples should have high probability for class 1.
        for i in 3..6 {
            assert!(
                proba[[i, 1]] > 0.5,
                "expected high prob for class 1 at sample {i}"
            );
        }
    }

    #[test]
    fn test_shape_mismatch_error() {
        let x = array![[1.0], [2.0]];
        let y = array![0.0, 1.0, 2.0];

        let ada = AdaBoostClassifier::default();
        let result: std::result::Result<FittedAdaBoostClassifier<f64>, _> = ada.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_predict_wrong_features_error() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0];

        let ada = AdaBoostClassifier {
            n_estimators: 5,
            seed: 0,
            ..Default::default()
        };
        let fitted: FittedAdaBoostClassifier<f64> = ada.fit(&x, &y).unwrap();

        let x_bad = array![[1.0], [2.0]];
        let result = fitted.predict(&x_bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_single_class_error() {
        let x = array![[1.0], [2.0], [3.0]];
        let y = array![1.0, 1.0, 1.0];

        let ada = AdaBoostClassifier::default();
        let result: std::result::Result<FittedAdaBoostClassifier<f64>, _> = ada.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_parameters() {
        let x = array![[1.0], [2.0]];
        let y = array![0.0, 1.0];

        let ada = AdaBoostClassifier {
            n_estimators: 0,
            ..Default::default()
        };
        assert!(Fit::<f64>::fit(&ada, &x, &y).is_err());

        let ada = AdaBoostClassifier {
            learning_rate: -0.1,
            ..Default::default()
        };
        assert!(Fit::<f64>::fit(&ada, &x, &y).is_err());
    }

    #[test]
    fn test_n_estimators_accessor() {
        let x = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [10.0, 1.0],
            [11.0, 1.0],
            [12.0, 1.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let ada = AdaBoostClassifier {
            n_estimators: 10,
            max_depth: Some(1),
            seed: 42,
            ..Default::default()
        };
        let fitted: FittedAdaBoostClassifier<f64> = ada.fit(&x, &y).unwrap();
        // The ensemble may stop early, but should have at most n_estimators.
        assert!(fitted.n_estimators() <= 10);
        assert!(fitted.n_estimators() >= 1);
    }

    #[test]
    fn test_predictions_are_valid_labels() {
        let x = array![
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [5.0, 5.0],
            [5.5, 5.0],
            [6.0, 5.0],
            [10.0, 10.0],
            [10.5, 10.0],
            [11.0, 10.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];

        let ada = AdaBoostClassifier {
            n_estimators: 50,
            learning_rate: 1.0,
            max_depth: Some(2),
            seed: 42,
        };
        let fitted: FittedAdaBoostClassifier<f64> = ada.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        let valid_labels: std::collections::HashSet<u64> = y.iter().map(|v| v.to_bits()).collect();
        for &p in preds.iter() {
            assert!(
                valid_labels.contains(&p.to_bits()),
                "prediction {p} is not a valid training label"
            );
        }
    }

    #[test]
    fn test_with_builder_pattern() {
        let x = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [10.0, 1.0],
            [11.0, 1.0],
            [12.0, 1.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let ada = AdaBoostClassifier::new()
            .with_n_estimators(30)
            .with_learning_rate(0.5)
            .with_max_depth(Some(2))
            .with_seed(42);

        let fitted: FittedAdaBoostClassifier<f64> = ada.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), y.len());
    }

    #[test]
    fn test_estimator_weights_positive() {
        let x = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [10.0, 1.0],
            [11.0, 1.0],
            [12.0, 1.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let ada = AdaBoostClassifier {
            n_estimators: 10,
            learning_rate: 1.0,
            max_depth: Some(1),
            seed: 42,
        };
        let fitted: FittedAdaBoostClassifier<f64> = ada.fit(&x, &y).unwrap();

        for &w in fitted.estimator_weights() {
            assert!(w >= 0.0, "estimator weight must be non-negative, got {w}");
        }
    }

    #[test]
    fn test_empty_input_error() {
        let x: Array2<f64> = Array2::zeros((0, 2));
        let y: Array1<f64> = Array1::zeros(0);

        let ada = AdaBoostClassifier::default();
        let result: std::result::Result<FittedAdaBoostClassifier<f64>, _> = ada.fit(&x, &y);
        assert!(result.is_err());
    }
}

impl<F: Float> PredictProba<F> for FittedAdaBoostClassifier<F> {
    fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>> {
        Self::predict_proba(self, x)
    }
}
