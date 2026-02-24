use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rustml_core::{Fit, Float, Predict, Result, RustMlError};
use rustml_trees::{DecisionTreeRegressor, FittedDecisionTreeRegressor};

/// Gradient boosting classifier parameters (unfitted state).
///
/// For binary classification, fits trees to the negative gradient of the
/// log loss (logistic regression loss). For multi-class (>2 classes), uses
/// a one-vs-rest strategy with separate sets of trees for each class.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GradientBoostingClassifier {
    /// Number of boosting rounds (trees per class for multi-class).
    pub n_estimators: usize,
    /// Shrinkage applied to each tree's contribution.
    pub learning_rate: f64,
    /// Maximum depth of each tree.
    pub max_depth: Option<usize>,
    /// Minimum samples required to split a node.
    pub min_samples_split: usize,
    /// Minimum samples required in a leaf node.
    pub min_samples_leaf: usize,
    /// Fraction of training samples used per tree.
    pub subsample: f64,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl GradientBoostingClassifier {
    /// Create a new `GradientBoostingClassifier` with default parameters.
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: Some(3),
            min_samples_split: 2,
            min_samples_leaf: 1,
            subsample: 1.0,
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

    /// Set the minimum number of samples required to split a node.
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split;
        self
    }

    /// Set the minimum number of samples required in a leaf node.
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }

    /// Set the fraction of samples used per boosting round.
    pub fn with_subsample(mut self, subsample: f64) -> Self {
        self.subsample = subsample;
        self
    }

    /// Set the random seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl Default for GradientBoostingClassifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted gradient boosting classifier.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedGradientBoostingClassifier<F: Float> {
    /// Unique class labels sorted in ascending order.
    classes: Vec<F>,
    /// For binary: a single list of trees operating on log-odds.
    /// For multi-class OVR: one list of trees per class.
    tree_sets: Vec<Vec<FittedDecisionTreeRegressor<F>>>,
    /// Initial log-odds per class set.
    initial_values: Vec<F>,
    /// Learning rate.
    learning_rate: F,
    /// Number of features expected.
    n_features: usize,
}

impl<F: Float> Fit<F> for GradientBoostingClassifier {
    type Fitted = FittedGradientBoostingClassifier<F>;

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
        if self.subsample <= 0.0 || self.subsample > 1.0 {
            return Err(RustMlError::InvalidParameter(
                "subsample must be in (0, 1]".into(),
            ));
        }

        // Discover unique classes.
        let classes = unique_sorted(y);
        let n_classes = classes.len();
        if n_classes < 2 {
            return Err(RustMlError::InvalidParameter(
                "y must contain at least 2 distinct classes".into(),
            ));
        }

        let n_features = x.ncols();
        let lr = F::from_f64(self.learning_rate).unwrap();

        if n_classes == 2 {
            // Binary classification: single set of trees on log-odds.
            let (initial, trees) =
                self.fit_binary(x, y, &classes[1], lr)?;
            Ok(FittedGradientBoostingClassifier {
                classes,
                tree_sets: vec![trees],
                initial_values: vec![initial],
                learning_rate: lr,
                n_features,
            })
        } else {
            // Multi-class: one-vs-rest, one tree set per class.
            let mut tree_sets = Vec::with_capacity(n_classes);
            let mut initial_values = Vec::with_capacity(n_classes);

            for class in &classes {
                let (initial, trees) =
                    self.fit_binary(x, y, class, lr)?;
                tree_sets.push(trees);
                initial_values.push(initial);
            }

            Ok(FittedGradientBoostingClassifier {
                classes,
                tree_sets,
                initial_values,
                learning_rate: lr,
                n_features,
            })
        }
    }
}

impl GradientBoostingClassifier {
    /// Fit a binary gradient boosting model where the positive class is
    /// `positive_class`. Returns (initial_log_odds, fitted_trees).
    fn fit_binary<F: Float>(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        positive_class: &F,
        lr: F,
    ) -> Result<(F, Vec<FittedDecisionTreeRegressor<F>>)> {
        let n_samples = x.nrows();
        let eps = F::from_f64(1e-15).unwrap();

        // Convert labels to binary 0/1 for the positive class.
        let binary_y: Array1<F> = y.mapv(|v| {
            if (v - *positive_class).abs() < eps {
                F::one()
            } else {
                F::zero()
            }
        });

        // Initial prediction: log-odds of positive class frequency.
        let p = binary_y.sum() / F::from_usize(n_samples).unwrap();
        let p_clipped = clamp(p, eps, F::one() - eps);
        let initial_log_odds = (p_clipped / (F::one() - p_clipped)).ln();

        let mut log_odds = Array1::from_elem(n_samples, initial_log_odds);

        let tree_params = DecisionTreeRegressor {
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
        };

        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut trees = Vec::with_capacity(self.n_estimators);
        let subsample_size = ((self.subsample * n_samples as f64).round() as usize).max(1);

        for _ in 0..self.n_estimators {
            // Compute probabilities via sigmoid.
            let probs = log_odds.mapv(sigmoid);

            // Pseudo-residuals: negative gradient of log loss = y - p(x).
            let residuals = &binary_y - &probs;

            // Fit tree to (subsampled) pseudo-residuals.
            let fitted_tree: FittedDecisionTreeRegressor<F> = if subsample_size < n_samples {
                let mut indices: Vec<usize> = (0..n_samples).collect();
                indices.shuffle(&mut rng);
                indices.truncate(subsample_size);
                indices.sort_unstable();

                let x_sub = build_sub_rows(x, &indices);
                let r_sub = Array1::from_vec(indices.iter().map(|&i| residuals[i]).collect());
                tree_params.fit(&x_sub, &r_sub)?
            } else {
                tree_params.fit(x, &residuals)?
            };

            // Update log-odds on full training set.
            let tree_preds = fitted_tree.predict(x)?;
            log_odds += &(tree_preds * lr);

            trees.push(fitted_tree);
        }

        Ok((initial_log_odds, trees))
    }
}

impl<F: Float> Predict<F> for FittedGradientBoostingClassifier<F> {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }

        let n_samples = x.nrows();

        if self.classes.len() == 2 {
            // Binary: single set of trees.
            let log_odds = self.predict_log_odds(x, 0)?;
            let half = F::from_f64(0.5).unwrap();

            let predictions: Vec<F> = log_odds
                .iter()
                .map(|&lo| {
                    if sigmoid(lo) >= half {
                        self.classes[1]
                    } else {
                        self.classes[0]
                    }
                })
                .collect();

            Ok(Array1::from_vec(predictions))
        } else {
            // Multi-class: predict class with highest log-odds (OVR).
            let mut all_log_odds = Vec::with_capacity(self.classes.len());
            for k in 0..self.classes.len() {
                all_log_odds.push(self.predict_log_odds(x, k)?);
            }

            let mut predictions = Vec::with_capacity(n_samples);
            for sample_idx in 0..n_samples {
                let mut best_class = 0;
                let mut best_val = all_log_odds[0][sample_idx];
                for (k, log_odds_k) in all_log_odds.iter().enumerate().skip(1) {
                    if log_odds_k[sample_idx] > best_val {
                        best_val = log_odds_k[sample_idx];
                        best_class = k;
                    }
                }
                predictions.push(self.classes[best_class]);
            }

            Ok(Array1::from_vec(predictions))
        }
    }
}

impl<F: Float> FittedGradientBoostingClassifier<F> {
    /// Number of estimators per class set.
    pub fn n_estimators(&self) -> usize {
        self.tree_sets.first().map_or(0, |ts| ts.len())
    }

    /// The unique classes discovered during training.
    pub fn classes(&self) -> &[F] {
        &self.classes
    }

    /// Compute raw log-odds for the k-th tree set.
    fn predict_log_odds(&self, x: &Array2<F>, k: usize) -> Result<Array1<F>> {
        let n_samples = x.nrows();
        let mut log_odds = Array1::from_elem(n_samples, self.initial_values[k]);

        for tree in &self.tree_sets[k] {
            let tree_preds = tree.predict(x)?;
            log_odds += &(tree_preds * self.learning_rate);
        }

        Ok(log_odds)
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Sigmoid function: 1 / (1 + exp(-x)).
fn sigmoid<F: Float>(x: F) -> F {
    F::one() / (F::one() + (-x).exp())
}

/// Clamp value to [lo, hi].
fn clamp<F: Float>(x: F, lo: F, hi: F) -> F {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
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
    let n_rows = row_indices.len();
    let n_cols = x.ncols();
    let mut data = Vec::with_capacity(n_rows * n_cols);
    for &ri in row_indices {
        for c in 0..n_cols {
            data.push(x[[ri, c]]);
        }
    }
    Array2::from_shape_vec((n_rows, n_cols), data).expect("shape matches data length")
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

        let gb = GradientBoostingClassifier {
            n_estimators: 50,
            learning_rate: 0.1,
            max_depth: Some(3),
            seed: 42,
            ..Default::default()
        };
        let fitted: FittedGradientBoostingClassifier<f64> = gb.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(*p, *t, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_multiclass_classification() {
        // Three classes with clearly separable data.
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

        let gb = GradientBoostingClassifier {
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: Some(3),
            seed: 42,
            ..Default::default()
        };
        let fitted: FittedGradientBoostingClassifier<f64> = gb.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(*p, *t, epsilon = 1e-10);
        }

        // Verify all 3 classes were detected.
        assert_eq!(fitted.classes().len(), 3);
    }

    #[test]
    fn test_reproducibility() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let gb = GradientBoostingClassifier {
            n_estimators: 20,
            seed: 123,
            ..Default::default()
        };

        let fitted1: FittedGradientBoostingClassifier<f64> = gb.fit(&x, &y).unwrap();
        let fitted2: FittedGradientBoostingClassifier<f64> = gb.fit(&x, &y).unwrap();

        let preds1 = fitted1.predict(&x).unwrap();
        let preds2 = fitted2.predict(&x).unwrap();

        for (a, b) in preds1.iter().zip(preds2.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-15);
        }
    }

    #[test]
    fn test_subsample_binary() {
        let x = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [10.0, 1.0],
            [11.0, 1.0],
            [12.0, 1.0],
            [13.0, 1.0]
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let gb = GradientBoostingClassifier {
            n_estimators: 80,
            learning_rate: 0.1,
            max_depth: Some(3),
            subsample: 0.75,
            seed: 42,
            ..Default::default()
        };
        let fitted: FittedGradientBoostingClassifier<f64> = gb.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(*p, *t, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_shape_mismatch_error() {
        let x = array![[1.0], [2.0]];
        let y = array![0.0, 1.0, 2.0];

        let gb = GradientBoostingClassifier::default();
        let result: std::result::Result<FittedGradientBoostingClassifier<f64>, _> = gb.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_predict_wrong_features_error() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0];

        let gb = GradientBoostingClassifier {
            n_estimators: 5,
            seed: 0,
            ..Default::default()
        };
        let fitted: FittedGradientBoostingClassifier<f64> = gb.fit(&x, &y).unwrap();

        let x_bad = array![[1.0], [2.0]];
        let result = fitted.predict(&x_bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_single_class_error() {
        let x = array![[1.0], [2.0], [3.0]];
        let y = array![1.0, 1.0, 1.0];

        let gb = GradientBoostingClassifier::default();
        let result: std::result::Result<FittedGradientBoostingClassifier<f64>, _> = gb.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_parameters() {
        let x = array![[1.0], [2.0]];
        let y = array![0.0, 1.0];

        let gb = GradientBoostingClassifier {
            n_estimators: 0,
            ..Default::default()
        };
        assert!(Fit::<f64>::fit(&gb, &x, &y).is_err());

        let gb = GradientBoostingClassifier {
            learning_rate: -0.1,
            ..Default::default()
        };
        assert!(Fit::<f64>::fit(&gb, &x, &y).is_err());
    }

    #[test]
    fn test_n_estimators_one() {
        let x = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [10.0, 1.0],
            [11.0, 1.0],
            [12.0, 1.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let gb = GradientBoostingClassifier {
            n_estimators: 1,
            learning_rate: 0.1,
            max_depth: Some(3),
            seed: 42,
            ..Default::default()
        };
        let fitted: FittedGradientBoostingClassifier<f64> = gb.fit(&x, &y).unwrap();
        assert_eq!(fitted.n_estimators(), 1);

        // Even a single boosting round should produce predictions with the correct length.
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), y.len());
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

        let gb = GradientBoostingClassifier {
            n_estimators: 50,
            learning_rate: 0.1,
            max_depth: Some(3),
            seed: 42,
            ..Default::default()
        };
        let fitted: FittedGradientBoostingClassifier<f64> = gb.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        let valid_labels: std::collections::HashSet<u64> =
            y.iter().map(|v| v.to_bits()).collect();
        for &p in preds.iter() {
            assert!(
                valid_labels.contains(&p.to_bits()),
                "prediction {p} is not a valid training label"
            );
        }
    }

    #[test]
    fn test_subsample_impact() {
        // With subsample < 1.0, the model should still produce reasonable
        // predictions on clearly separable data.
        let x = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [10.0, 1.0],
            [11.0, 1.0],
            [12.0, 1.0],
            [13.0, 1.0]
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let gb = GradientBoostingClassifier {
            n_estimators: 80,
            learning_rate: 0.1,
            max_depth: Some(3),
            subsample: 0.5,
            seed: 7,
            ..Default::default()
        };
        let fitted: FittedGradientBoostingClassifier<f64> = gb.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        let correct: usize = preds
            .iter()
            .zip(y.iter())
            .filter(|(p, t)| (*p - *t).abs() < 1e-10)
            .count();
        let accuracy = correct as f64 / y.len() as f64;
        assert!(
            accuracy >= 0.75,
            "subsample=0.5 should still achieve reasonable accuracy, got {accuracy}"
        );
    }

    #[test]
    fn test_learning_rate_zero_error_or_degrades() {
        // A very small learning rate with few estimators should produce weaker
        // predictions than a normal learning rate (less overfitting / underfitting
        // with few rounds).
        let x = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [10.0, 1.0],
            [11.0, 1.0],
            [12.0, 1.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        // Normal learning rate should fit well with enough estimators.
        let gb_normal = GradientBoostingClassifier {
            n_estimators: 50,
            learning_rate: 0.1,
            max_depth: Some(3),
            seed: 42,
            ..Default::default()
        };
        let fitted_normal: FittedGradientBoostingClassifier<f64> =
            gb_normal.fit(&x, &y).unwrap();
        let preds_normal = fitted_normal.predict(&x).unwrap();
        let correct_normal: usize = preds_normal
            .iter()
            .zip(y.iter())
            .filter(|(p, t)| (*p - *t).abs() < 1e-10)
            .count();

        // Tiny learning rate with the same number of estimators should learn slower.
        let gb_tiny = GradientBoostingClassifier {
            n_estimators: 50,
            learning_rate: 0.001,
            max_depth: Some(3),
            seed: 42,
            ..Default::default()
        };
        let fitted_tiny: FittedGradientBoostingClassifier<f64> =
            gb_tiny.fit(&x, &y).unwrap();
        let preds_tiny = fitted_tiny.predict(&x).unwrap();
        let correct_tiny: usize = preds_tiny
            .iter()
            .zip(y.iter())
            .filter(|(p, t)| (*p - *t).abs() < 1e-10)
            .count();

        // The normal learning rate should be at least as accurate (likely better).
        assert!(
            correct_normal >= correct_tiny,
            "normal lr ({correct_normal} correct) should be >= tiny lr ({correct_tiny} correct)"
        );
    }
}
