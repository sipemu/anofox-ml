use anofox_ml_core::{Fit, Float, Predict, Result, RustMlError};
use anofox_ml_trees::{DecisionTreeRegressor, FittedDecisionTreeRegressor};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

/// Gradient boosting regressor parameters (unfitted state).
///
/// Builds an ensemble of decision tree regressors sequentially, where each
/// tree fits the negative gradient (residuals) of the squared-error loss.
/// Predictions are the sum of the initial value plus a learning-rate-scaled
/// contribution from each tree.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GradientBoostingRegressor {
    /// Number of boosting rounds (trees).
    pub n_estimators: usize,
    /// Shrinkage applied to each tree's contribution.
    pub learning_rate: f64,
    /// Maximum depth of each tree.
    pub max_depth: Option<usize>,
    /// Minimum samples required to split a node.
    pub min_samples_split: usize,
    /// Minimum samples required in a leaf node.
    pub min_samples_leaf: usize,
    /// Fraction of training samples used per tree (stochastic gradient boosting).
    pub subsample: f64,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl GradientBoostingRegressor {
    /// Create a new `GradientBoostingRegressor` with default parameters.
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

impl Default for GradientBoostingRegressor {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted gradient boosting regressor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedGradientBoostingRegressor<F: Float> {
    /// Initial prediction (mean of training targets).
    initial_prediction: F,
    /// Sequence of fitted regression trees.
    trees: Vec<FittedDecisionTreeRegressor<F>>,
    /// Learning rate used during training.
    learning_rate: F,
    /// Number of features expected at prediction time.
    n_features: usize,
}

impl<F: Float> Fit<F> for GradientBoostingRegressor {
    type Fitted = FittedGradientBoostingRegressor<F>;

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

        let n_samples = x.nrows();
        let n_features = x.ncols();
        let lr = F::from_f64(self.learning_rate).unwrap();

        // Step 1: Initialize with mean of y.
        let initial_prediction = y.sum() / F::from_usize(n_samples).unwrap();

        // Current predictions for every training sample.
        let mut predictions = Array1::from_elem(n_samples, initial_prediction);

        let tree_params = DecisionTreeRegressor {
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            max_features: None,
            sample_weight: None,
        };

        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut trees = Vec::with_capacity(self.n_estimators);

        let subsample_size = ((self.subsample * n_samples as f64).round() as usize).max(1);

        for _ in 0..self.n_estimators {
            // Step 2a: Compute residuals (negative gradient of squared error loss).
            let residuals = y - &predictions;

            // Step 2b+c: Fit a tree to (subsampled) residuals.
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

            // Step 2d: Update predictions on the full training set.
            let tree_preds = fitted_tree.predict(x)?;
            predictions += &(tree_preds * lr);

            trees.push(fitted_tree);
        }

        Ok(FittedGradientBoostingRegressor {
            initial_prediction,
            trees,
            learning_rate: lr,
            n_features,
        })
    }
}

impl<F: Float> Predict<F> for FittedGradientBoostingRegressor<F> {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }

        let n_samples = x.nrows();
        let mut predictions = Array1::from_elem(n_samples, self.initial_prediction);

        for tree in &self.trees {
            let tree_preds = tree.predict(x)?;
            predictions += &(tree_preds * self.learning_rate);
        }

        Ok(predictions)
    }
}

impl<F: Float> FittedGradientBoostingRegressor<F> {
    /// Number of trees in the ensemble.
    pub fn n_estimators(&self) -> usize {
        self.trees.len()
    }

    /// The initial prediction (mean of training targets).
    pub fn initial_prediction(&self) -> F {
        self.initial_prediction
    }

    /// Feature importances averaged across all trees, normalized to sum to 1.
    pub fn feature_importances(&self) -> Array1<F> {
        let mut importances = vec![F::zero(); self.n_features];
        let n_trees = self.trees.len();

        if n_trees == 0 {
            return Array1::zeros(self.n_features);
        }

        let n_trees_f = F::from_usize(n_trees).unwrap();
        for tree in &self.trees {
            let tree_imp = tree.feature_importances();
            for (j, &imp) in tree_imp.iter().enumerate() {
                importances[j] += imp / n_trees_f;
            }
        }

        // Normalize to sum to 1
        let sum: F = importances.iter().copied().fold(F::zero(), |a, b| a + b);
        if sum > F::zero() {
            Array1::from_vec(importances.into_iter().map(|v| v / sum).collect())
        } else {
            Array1::zeros(self.n_features)
        }
    }
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
    fn test_basic_regression() {
        // y = 2*x, gradient boosting should learn a good approximation.
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

        let gb = GradientBoostingRegressor {
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: Some(3),
            seed: 42,
            ..Default::default()
        };
        let fitted: FittedGradientBoostingRegressor<f64> = gb.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(*p, *t, epsilon = 1.0);
        }
    }

    #[test]
    fn test_training_error_decreases() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]];
        let y = array![1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0];

        let gb_few = GradientBoostingRegressor {
            n_estimators: 5,
            learning_rate: 0.1,
            max_depth: Some(2),
            seed: 0,
            ..Default::default()
        };
        let gb_many = GradientBoostingRegressor {
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: Some(2),
            seed: 0,
            ..Default::default()
        };

        let fitted_few: FittedGradientBoostingRegressor<f64> = gb_few.fit(&x, &y).unwrap();
        let fitted_many: FittedGradientBoostingRegressor<f64> = gb_many.fit(&x, &y).unwrap();

        let preds_few = fitted_few.predict(&x).unwrap();
        let preds_many = fitted_many.predict(&x).unwrap();

        let mse_few: f64 = (&y - &preds_few).mapv(|v| v * v).mean().unwrap();
        let mse_many: f64 = (&y - &preds_many).mapv(|v| v * v).mean().unwrap();

        assert!(
            mse_many < mse_few,
            "more estimators should reduce training error: mse_many={mse_many} >= mse_few={mse_few}"
        );
    }

    #[test]
    fn test_reproducibility() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let gb = GradientBoostingRegressor {
            n_estimators: 20,
            seed: 123,
            ..Default::default()
        };

        let fitted1: FittedGradientBoostingRegressor<f64> = gb.fit(&x, &y).unwrap();
        let fitted2: FittedGradientBoostingRegressor<f64> = gb.fit(&x, &y).unwrap();

        let preds1 = fitted1.predict(&x).unwrap();
        let preds2 = fitted2.predict(&x).unwrap();

        for (a, b) in preds1.iter().zip(preds2.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-15);
        }
    }

    #[test]
    fn test_subsample() {
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

        let gb = GradientBoostingRegressor {
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: Some(3),
            subsample: 0.8,
            seed: 42,
            ..Default::default()
        };
        let fitted: FittedGradientBoostingRegressor<f64> = gb.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        // With subsample < 1.0, predictions should still be reasonable.
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(*p, *t, epsilon = 2.0);
        }
    }

    #[test]
    fn test_shape_mismatch_error() {
        let x = array![[1.0], [2.0]];
        let y = array![0.0, 1.0, 2.0];

        let gb = GradientBoostingRegressor::default();
        let result: std::result::Result<FittedGradientBoostingRegressor<f64>, _> = gb.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_predict_wrong_features_error() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 2.0];

        let gb = GradientBoostingRegressor {
            n_estimators: 5,
            seed: 0,
            ..Default::default()
        };
        let fitted: FittedGradientBoostingRegressor<f64> = gb.fit(&x, &y).unwrap();

        let x_bad = array![[1.0], [2.0]];
        let result = fitted.predict(&x_bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_initial_prediction_is_mean() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![10.0, 20.0, 30.0, 40.0];

        let gb = GradientBoostingRegressor {
            n_estimators: 1,
            seed: 0,
            ..Default::default()
        };
        let fitted: FittedGradientBoostingRegressor<f64> = gb.fit(&x, &y).unwrap();
        assert_abs_diff_eq!(fitted.initial_prediction(), 25.0, epsilon = 1e-10);
    }

    #[test]
    fn test_invalid_parameters() {
        let x = array![[1.0], [2.0]];
        let y = array![1.0, 2.0];

        // n_estimators = 0
        let gb = GradientBoostingRegressor {
            n_estimators: 0,
            ..Default::default()
        };
        assert!(Fit::<f64>::fit(&gb, &x, &y).is_err());

        // learning_rate <= 0
        let gb = GradientBoostingRegressor {
            learning_rate: 0.0,
            ..Default::default()
        };
        assert!(Fit::<f64>::fit(&gb, &x, &y).is_err());

        // subsample out of range
        let gb = GradientBoostingRegressor {
            subsample: 0.0,
            ..Default::default()
        };
        assert!(Fit::<f64>::fit(&gb, &x, &y).is_err());

        let gb = GradientBoostingRegressor {
            subsample: 1.5,
            ..Default::default()
        };
        assert!(Fit::<f64>::fit(&gb, &x, &y).is_err());
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

        let gb = GradientBoostingRegressor {
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: Some(3),
            seed: 42,
            ..Default::default()
        };
        let fitted: FittedGradientBoostingRegressor<f64> = gb.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(p.is_finite(), "prediction must be finite, got {p}");
        }
    }

    #[test]
    fn test_n_estimators_one_regressor() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let gb = GradientBoostingRegressor {
            n_estimators: 1,
            learning_rate: 0.1,
            max_depth: Some(3),
            seed: 0,
            ..Default::default()
        };
        let fitted: FittedGradientBoostingRegressor<f64> = gb.fit(&x, &y).unwrap();
        assert_eq!(fitted.n_estimators(), 1);

        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), y.len());
        // All predictions should be finite even with a single tree.
        for &p in preds.iter() {
            assert!(p.is_finite());
        }
    }
}
