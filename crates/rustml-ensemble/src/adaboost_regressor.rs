use ndarray::{Array1, Array2};
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rustml_core::{Fit, Float, Predict, Result, RustMlError};
use rustml_trees::{DecisionTreeRegressor, FittedDecisionTreeRegressor};

/// Loss function used by AdaBoost.R2 to compute sample losses.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum AdaBoostLoss {
    /// Linear loss: L_i = e_i / D.
    Linear,
    /// Square loss: L_i = (e_i / D)^2.
    Square,
    /// Exponential loss: L_i = 1 - exp(-e_i / D).
    Exponential,
}

impl Default for AdaBoostLoss {
    fn default() -> Self {
        Self::Linear
    }
}

/// AdaBoost regressor parameters (unfitted state).
///
/// Implements the AdaBoost.R2 algorithm for regression. Each boosting round
/// fits a decision tree regressor on a weighted bootstrap sample, then adjusts
/// sample weights so that samples with higher prediction error receive greater
/// emphasis in subsequent rounds.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AdaBoostRegressor {
    /// Number of boosting rounds (weak learners).
    pub n_estimators: usize,
    /// Learning rate that scales the estimator weights.
    pub learning_rate: f64,
    /// Maximum depth of each decision tree. `Some(1)` yields stumps.
    pub max_depth: Option<usize>,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Loss function for computing sample losses.
    pub loss: AdaBoostLoss,
}

impl AdaBoostRegressor {
    /// Create a new `AdaBoostRegressor` with default parameters.
    pub fn new() -> Self {
        Self {
            n_estimators: 50,
            learning_rate: 1.0,
            max_depth: Some(1),
            seed: 0,
            loss: AdaBoostLoss::Linear,
        }
    }

    /// Set the number of boosting rounds.
    pub fn with_n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators;
        self
    }

    /// Set the learning rate.
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

    /// Set the loss function.
    pub fn with_loss(mut self, loss: AdaBoostLoss) -> Self {
        self.loss = loss;
        self
    }
}

impl Default for AdaBoostRegressor {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted AdaBoost regressor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedAdaBoostRegressor<F: Float> {
    /// Weak learners.
    estimators: Vec<FittedDecisionTreeRegressor<F>>,
    /// Weight (log(1/beta)) for each estimator.
    estimator_weights: Vec<F>,
    /// Number of features expected at prediction time.
    n_features: usize,
}

impl<F: Float> Fit<F> for AdaBoostRegressor {
    type Fitted = FittedAdaBoostRegressor<F>;

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

        let n_samples = x.nrows();
        let n_features = x.ncols();
        let lr = F::from_f64(self.learning_rate).unwrap();

        let tree_params = DecisionTreeRegressor {
            max_depth: self.max_depth,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            sample_weight: None,
        };

        let mut rng = StdRng::seed_from_u64(self.seed);

        // Initialize sample weights uniformly.
        let mut weights: Vec<F> = vec![F::one() / F::from_usize(n_samples).unwrap(); n_samples];

        let mut estimators = Vec::with_capacity(self.n_estimators);
        let mut estimator_weights = Vec::with_capacity(self.n_estimators);

        let eps = F::from_f64(1e-15).unwrap();
        let half = F::from_f64(0.5).unwrap();

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
            let fitted_tree: FittedDecisionTreeRegressor<F> =
                tree_params.fit(&x_bootstrap, &y_bootstrap)?;

            // Predict on the *full* training set.
            let preds = fitted_tree.predict(x)?;

            // Compute absolute errors.
            let abs_errors: Vec<F> = (0..n_samples)
                .map(|i| (y[i] - preds[i]).abs())
                .collect();

            // Maximum error D.
            let d_max = abs_errors
                .iter()
                .copied()
                .fold(F::zero(), |a, b| if b > a { b } else { a });

            // If D is zero, the tree perfectly predicts -- keep it and stop.
            if d_max <= eps {
                estimators.push(fitted_tree);
                estimator_weights.push(lr);
                break;
            }

            // Compute individual losses.
            let losses: Vec<F> = abs_errors
                .iter()
                .map(|&e| compute_loss(e, d_max, self.loss))
                .collect();

            // Compute weighted average loss.
            let w_sum: F = weights.iter().copied().fold(F::zero(), |a, b| a + b);
            let l_bar: F = weights
                .iter()
                .zip(losses.iter())
                .map(|(&w, &l)| w * l)
                .fold(F::zero(), |a, b| a + b)
                / w_sum;

            // If average loss is >= 0.5, stop (weak learner is no better than random).
            if l_bar >= half {
                if estimators.is_empty() {
                    estimators.push(fitted_tree);
                    estimator_weights.push(F::zero());
                }
                break;
            }

            // Compute beta.
            let beta = l_bar / (F::one() - l_bar);

            // Update sample weights: w_i *= beta^(1-L_i).
            for i in 0..n_samples {
                let exponent = F::one() - losses[i];
                weights[i] = weights[i] * beta.powf(exponent);
            }

            // Normalize weights.
            let w_total: F = weights.iter().copied().fold(F::zero(), |a, b| a + b);
            if w_total > F::zero() {
                for w in &mut weights {
                    *w = *w / w_total;
                }
            }

            // Estimator weight = learning_rate * ln(1/beta).
            let alpha = lr * (F::one() / (beta + eps)).ln();

            estimators.push(fitted_tree);
            estimator_weights.push(alpha);
        }

        Ok(FittedAdaBoostRegressor {
            estimators,
            estimator_weights,
            n_features,
        })
    }
}

impl<F: Float> Predict<F> for FittedAdaBoostRegressor<F> {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }

        let n_samples = x.nrows();

        // Weighted average of estimator predictions.
        let mut weighted_sum = Array1::<F>::zeros(n_samples);
        let mut weight_total = F::zero();

        for (tree, &alpha) in self.estimators.iter().zip(self.estimator_weights.iter()) {
            let tree_preds = tree.predict(x)?;
            weighted_sum = weighted_sum + &(tree_preds * alpha);
            weight_total = weight_total + alpha;
        }

        if weight_total > F::zero() {
            weighted_sum.mapv_inplace(|v| v / weight_total);
        }

        Ok(weighted_sum)
    }
}

impl<F: Float> FittedAdaBoostRegressor<F> {
    /// Number of estimators in the ensemble.
    pub fn n_estimators(&self) -> usize {
        self.estimators.len()
    }

    /// The weight of each estimator.
    pub fn estimator_weights(&self) -> &[F] {
        &self.estimator_weights
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Compute individual sample loss for AdaBoost.R2.
#[inline]
fn compute_loss<F: Float>(error: F, d_max: F, loss: AdaBoostLoss) -> F {
    let ratio = error / d_max;
    match loss {
        AdaBoostLoss::Linear => ratio,
        AdaBoostLoss::Square => ratio * ratio,
        AdaBoostLoss::Exponential => F::one() - (-ratio).exp(),
    }
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
    fn test_basic_regression() {
        // y = 2*x, AdaBoost should learn a reasonable approximation.
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0],
            [10.0]
        ];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

        let ada = AdaBoostRegressor {
            n_estimators: 100,
            learning_rate: 1.0,
            max_depth: Some(3),
            seed: 42,
            loss: AdaBoostLoss::Linear,
        };
        let fitted: FittedAdaBoostRegressor<f64> = ada.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(*p, *t, epsilon = 3.0);
        }
    }

    #[test]
    fn test_training_error_decreases() {
        // Use a simple linear dataset where deeper trees with more estimators
        // should produce a clear improvement over a single stump.
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];

        let ada_one = AdaBoostRegressor {
            n_estimators: 1,
            learning_rate: 1.0,
            max_depth: Some(3),
            seed: 42,
            loss: AdaBoostLoss::Linear,
        };
        let ada_many = AdaBoostRegressor {
            n_estimators: 50,
            learning_rate: 1.0,
            max_depth: Some(3),
            seed: 42,
            loss: AdaBoostLoss::Linear,
        };

        let fitted_one: FittedAdaBoostRegressor<f64> = ada_one.fit(&x, &y).unwrap();
        let fitted_many: FittedAdaBoostRegressor<f64> = ada_many.fit(&x, &y).unwrap();

        let preds_one = fitted_one.predict(&x).unwrap();
        let preds_many = fitted_many.predict(&x).unwrap();

        let mse_one: f64 = (&y - &preds_one).mapv(|v| v * v).mean().unwrap();
        let mse_many: f64 = (&y - &preds_many).mapv(|v| v * v).mean().unwrap();

        assert!(
            mse_many <= mse_one,
            "more estimators should reduce training error: mse_many={mse_many} >= mse_one={mse_one}"
        );
    }

    #[test]
    fn test_reproducibility() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let ada = AdaBoostRegressor {
            n_estimators: 20,
            learning_rate: 1.0,
            max_depth: Some(1),
            seed: 123,
            loss: AdaBoostLoss::Linear,
        };

        let fitted1: FittedAdaBoostRegressor<f64> = ada.fit(&x, &y).unwrap();
        let fitted2: FittedAdaBoostRegressor<f64> = ada.fit(&x, &y).unwrap();

        let preds1 = fitted1.predict(&x).unwrap();
        let preds2 = fitted2.predict(&x).unwrap();

        for (a, b) in preds1.iter().zip(preds2.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-15);
        }
    }

    #[test]
    fn test_loss_functions() {
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0]
        ];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];

        for loss in [AdaBoostLoss::Linear, AdaBoostLoss::Square, AdaBoostLoss::Exponential] {
            let ada = AdaBoostRegressor {
                n_estimators: 50,
                learning_rate: 1.0,
                max_depth: Some(2),
                seed: 42,
                loss,
            };
            let fitted: FittedAdaBoostRegressor<f64> = ada.fit(&x, &y).unwrap();
            let preds = fitted.predict(&x).unwrap();

            // All predictions should be finite.
            for &p in preds.iter() {
                assert!(p.is_finite(), "prediction must be finite, got {p} with loss {:?}", loss);
            }
        }
    }

    #[test]
    fn test_shape_mismatch_error() {
        let x = array![[1.0], [2.0]];
        let y = array![0.0, 1.0, 2.0];

        let ada = AdaBoostRegressor::default();
        let result: std::result::Result<FittedAdaBoostRegressor<f64>, _> = ada.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_predict_wrong_features_error() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 2.0];

        let ada = AdaBoostRegressor {
            n_estimators: 5,
            seed: 0,
            ..Default::default()
        };
        let fitted: FittedAdaBoostRegressor<f64> = ada.fit(&x, &y).unwrap();

        let x_bad = array![[1.0], [2.0]];
        let result = fitted.predict(&x_bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_parameters() {
        let x = array![[1.0], [2.0]];
        let y = array![1.0, 2.0];

        // n_estimators = 0
        let ada = AdaBoostRegressor {
            n_estimators: 0,
            ..Default::default()
        };
        assert!(Fit::<f64>::fit(&ada, &x, &y).is_err());

        // learning_rate <= 0
        let ada = AdaBoostRegressor {
            learning_rate: 0.0,
            ..Default::default()
        };
        assert!(Fit::<f64>::fit(&ada, &x, &y).is_err());
    }

    #[test]
    fn test_predictions_finite() {
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0],
            [10.0]
        ];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

        let ada = AdaBoostRegressor {
            n_estimators: 50,
            learning_rate: 1.0,
            max_depth: Some(3),
            seed: 42,
            loss: AdaBoostLoss::Linear,
        };
        let fitted: FittedAdaBoostRegressor<f64> = ada.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(p.is_finite(), "prediction must be finite, got {p}");
        }
    }

    #[test]
    fn test_n_estimators_accessor() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let ada = AdaBoostRegressor {
            n_estimators: 10,
            max_depth: Some(1),
            seed: 0,
            ..Default::default()
        };
        let fitted: FittedAdaBoostRegressor<f64> = ada.fit(&x, &y).unwrap();
        assert!(fitted.n_estimators() <= 10);
        assert!(fitted.n_estimators() >= 1);
    }

    #[test]
    fn test_with_builder_pattern() {
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0]
        ];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];

        let ada = AdaBoostRegressor::new()
            .with_n_estimators(30)
            .with_learning_rate(0.5)
            .with_max_depth(Some(2))
            .with_seed(42)
            .with_loss(AdaBoostLoss::Square);

        let fitted: FittedAdaBoostRegressor<f64> = ada.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), y.len());
    }

    #[test]
    fn test_empty_input_error() {
        let x: Array2<f64> = Array2::zeros((0, 2));
        let y: Array1<f64> = Array1::zeros(0);

        let ada = AdaBoostRegressor::default();
        let result: std::result::Result<FittedAdaBoostRegressor<f64>, _> = ada.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_estimator_weights_positive() {
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0]
        ];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];

        let ada = AdaBoostRegressor {
            n_estimators: 20,
            learning_rate: 1.0,
            max_depth: Some(2),
            seed: 42,
            loss: AdaBoostLoss::Linear,
        };
        let fitted: FittedAdaBoostRegressor<f64> = ada.fit(&x, &y).unwrap();

        for &w in fitted.estimator_weights() {
            assert!(
                w >= 0.0,
                "estimator weight must be non-negative, got {w}"
            );
        }
    }
}
