use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use rustml_core::{Fit, Float, Predict, Result, RustMlError};
use rustml_trees::{DecisionTreeRegressor, FittedDecisionTreeRegressor};

/// Random forest regressor parameters (unfitted state).
///
/// Trains an ensemble of decision tree regressors, each on a bootstrap sample
/// of the data with an optional random subset of features. Predictions are the
/// average of individual tree predictions.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RandomForestRegressor {
    /// Number of trees in the forest.
    pub n_estimators: usize,
    /// Maximum depth of each tree.
    pub max_depth: Option<usize>,
    /// Minimum samples required to split a node.
    pub min_samples_split: usize,
    /// Minimum samples required in a leaf node.
    pub min_samples_leaf: usize,
    /// Number of features to consider per tree. If `None`, all features are used.
    pub max_features: Option<usize>,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl RandomForestRegressor {
    /// Create a new `RandomForestRegressor` with the given number of trees and default parameters.
    pub fn new(n_estimators: usize) -> Self {
        Self {
            n_estimators,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            seed: 0,
        }
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

    /// Set the number of features to consider per tree.
    pub fn with_max_features(mut self, max_features: Option<usize>) -> Self {
        self.max_features = max_features;
        self
    }

    /// Set the random seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl Default for RandomForestRegressor {
    fn default() -> Self {
        Self::new(100)
    }
}

/// A single tree in the forest together with its selected feature indices.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
struct ForestTree<F: Float> {
    tree: FittedDecisionTreeRegressor<F>,
    /// Indices of the features this tree was trained on (relative to the
    /// original feature matrix). When `max_features` is `None` this contains
    /// `0..n_features`.
    feature_indices: Vec<usize>,
}

/// Fitted random forest regressor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedRandomForestRegressor<F: Float> {
    trees: Vec<ForestTree<F>>,
    n_features: usize,
}

impl<F: Float> Fit<F> for RandomForestRegressor {
    type Fitted = FittedRandomForestRegressor<F>;

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

        let n_samples = x.nrows();
        let n_features = x.ncols();

        if let Some(k) = self.max_features {
            if k == 0 || k > n_features {
                return Err(RustMlError::InvalidParameter(format!(
                    "max_features={k} is invalid for data with {n_features} features"
                )));
            }
        }

        let mut rng = StdRng::seed_from_u64(self.seed);

        let tree_params = DecisionTreeRegressor {
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
        };

        // Pre-generate bootstrap and feature indices for determinism
        let sample_plans: Vec<(Vec<usize>, Vec<usize>)> = (0..self.n_estimators)
            .map(|_| {
                let bootstrap_indices: Vec<usize> = (0..n_samples)
                    .map(|_| rng.gen_range(0..n_samples))
                    .collect();
                let feature_indices = select_features(n_features, self.max_features, &mut rng);
                (bootstrap_indices, feature_indices)
            })
            .collect();

        // Train trees in parallel
        let trees: Result<Vec<ForestTree<F>>> = sample_plans
            .into_par_iter()
            .map(|(bootstrap_indices, feature_indices)| {
                let x_bootstrap = build_sub_matrix(x, &bootstrap_indices, &feature_indices);
                let y_bootstrap = Array1::from_vec(
                    bootstrap_indices.iter().map(|&i| y[i]).collect::<Vec<F>>(),
                );
                let fitted_tree: FittedDecisionTreeRegressor<F> =
                    tree_params.fit(&x_bootstrap, &y_bootstrap)?;
                Ok(ForestTree {
                    tree: fitted_tree,
                    feature_indices,
                })
            })
            .collect();

        Ok(FittedRandomForestRegressor {
            trees: trees?,
            n_features,
        })
    }
}

impl<F: Float> Predict<F> for FittedRandomForestRegressor<F> {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }

        let n_samples = x.nrows();
        let n_trees = F::from_usize(self.trees.len()).unwrap();
        let mut predictions = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let mut sum = F::zero();
            for forest_tree in &self.trees {
                // Build a row with only the features this tree was trained on
                let sub_row: Vec<F> = forest_tree
                    .feature_indices
                    .iter()
                    .map(|&fi| x[[i, fi]])
                    .collect();
                let sub_x = Array2::from_shape_vec((1, sub_row.len()), sub_row)
                    .expect("shape matches feature count");
                let pred = forest_tree.tree.predict(&sub_x)?;
                sum += pred[0];
            }
            predictions.push(sum / n_trees);
        }

        Ok(Array1::from_vec(predictions))
    }
}

impl<F: Float> FittedRandomForestRegressor<F> {
    /// Feature importances averaged across all trees and normalized to sum to 1.
    ///
    /// Each tree's importances are computed in its own (possibly reduced)
    /// feature space, then mapped back to the original feature indices and
    /// averaged.
    pub fn feature_importances(&self) -> Array1<F> {
        let mut importances = vec![F::zero(); self.n_features];
        let n_trees = F::from_usize(self.trees.len()).unwrap();

        for forest_tree in &self.trees {
            let tree_importances = forest_tree.tree.feature_importances();
            for (local_idx, &original_idx) in forest_tree.feature_indices.iter().enumerate() {
                importances[original_idx] += tree_importances[local_idx] / n_trees;
            }
        }

        // Normalize so importances sum to 1
        let sum: F = importances.iter().copied().fold(F::zero(), |a, b| a + b);
        if sum > F::zero() {
            Array1::from_vec(importances.into_iter().map(|v| v / sum).collect())
        } else {
            Array1::zeros(self.n_features)
        }
    }

    /// Number of trees in the forest.
    pub fn n_estimators(&self) -> usize {
        self.trees.len()
    }
}

/// Select `k` distinct feature indices from `0..n_features` without replacement.
/// If `max_features` is `None`, returns all feature indices.
fn select_features(n_features: usize, max_features: Option<usize>, rng: &mut StdRng) -> Vec<usize> {
    match max_features {
        None => (0..n_features).collect(),
        Some(k) => {
            // Fisher-Yates partial shuffle
            let mut indices: Vec<usize> = (0..n_features).collect();
            for i in 0..k {
                let j = rng.gen_range(i..n_features);
                indices.swap(i, j);
            }
            indices.truncate(k);
            indices.sort_unstable();
            indices
        }
    }
}

/// Build a sub-matrix selecting specific rows and columns from `x`.
fn build_sub_matrix<F: Float>(
    x: &Array2<F>,
    row_indices: &[usize],
    col_indices: &[usize],
) -> Array2<F> {
    let n_rows = row_indices.len();
    let n_cols = col_indices.len();
    let mut data = Vec::with_capacity(n_rows * n_cols);
    for &ri in row_indices {
        for &ci in col_indices {
            data.push(x[[ri, ci]]);
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
        // y = 2*x, forest should learn a good approximation
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

        let rf = RandomForestRegressor {
            n_estimators: 50,
            max_depth: None,
            seed: 42,
            ..Default::default()
        };
        let fitted: FittedRandomForestRegressor<f64> = rf.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        // With enough trees and unlimited depth on training data, predictions
        // should be close to true values.
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(*p, *t, epsilon = 2.0);
        }
    }

    #[test]
    fn test_reproducibility() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let rf = RandomForestRegressor {
            n_estimators: 10,
            seed: 123,
            ..Default::default()
        };

        let fitted1: FittedRandomForestRegressor<f64> = rf.fit(&x, &y).unwrap();
        let fitted2: FittedRandomForestRegressor<f64> = rf.fit(&x, &y).unwrap();

        let preds1 = fitted1.predict(&x).unwrap();
        let preds2 = fitted2.predict(&x).unwrap();

        for (a, b) in preds1.iter().zip(preds2.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-15);
        }
    }

    #[test]
    fn test_max_features() {
        let x = array![
            [1.0, 100.0, 0.5],
            [2.0, 200.0, 0.6],
            [3.0, 300.0, 0.7],
            [10.0, 400.0, 0.8],
            [11.0, 500.0, 0.9],
            [12.0, 600.0, 1.0]
        ];
        let y = array![1.0, 2.0, 3.0, 10.0, 11.0, 12.0];

        let rf = RandomForestRegressor {
            n_estimators: 30,
            max_features: Some(2),
            seed: 99,
            ..Default::default()
        };
        let fitted: FittedRandomForestRegressor<f64> = rf.fit(&x, &y).unwrap();

        // Should produce reasonable predictions
        let preds = fitted.predict(&x).unwrap();
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(*p, *t, epsilon = 3.0);
        }
    }

    #[test]
    fn test_feature_importances_sum_to_one() {
        let x = array![
            [1.0, 100.0],
            [2.0, 200.0],
            [3.0, 300.0],
            [4.0, 400.0],
            [5.0, 500.0],
            [6.0, 600.0]
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let rf = RandomForestRegressor {
            n_estimators: 20,
            seed: 7,
            ..Default::default()
        };
        let fitted: FittedRandomForestRegressor<f64> = rf.fit(&x, &y).unwrap();

        let importances = fitted.feature_importances();
        let sum: f64 = importances.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_n_estimators() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![1.0, 2.0, 3.0, 4.0];

        let rf = RandomForestRegressor {
            n_estimators: 7,
            seed: 0,
            ..Default::default()
        };
        let fitted: FittedRandomForestRegressor<f64> = rf.fit(&x, &y).unwrap();
        assert_eq!(fitted.n_estimators(), 7);
    }

    #[test]
    fn test_shape_mismatch_error() {
        let x = array![[1.0], [2.0]];
        let y = array![0.0, 1.0, 2.0];

        let rf = RandomForestRegressor::default();
        let result: std::result::Result<FittedRandomForestRegressor<f64>, _> = rf.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_predict_wrong_features_error() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 2.0];

        let rf = RandomForestRegressor {
            n_estimators: 5,
            seed: 0,
            ..Default::default()
        };
        let fitted: FittedRandomForestRegressor<f64> = rf.fit(&x, &y).unwrap();

        let x_bad = array![[1.0], [2.0]];
        let result = fitted.predict(&x_bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_max_features() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 2.0];

        let rf = RandomForestRegressor {
            n_estimators: 5,
            max_features: Some(5),
            seed: 0,
            ..Default::default()
        };
        let result: std::result::Result<FittedRandomForestRegressor<f64>, _> = rf.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_average_predictions() {
        // With max_depth=0 (immediate leaf), each tree predicts the mean
        // of its bootstrap sample. The forest should predict approximately
        // the global mean.
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![10.0, 20.0, 30.0, 40.0];

        let rf = RandomForestRegressor {
            n_estimators: 200,
            max_depth: Some(0),
            seed: 42,
            ..Default::default()
        };
        let fitted: FittedRandomForestRegressor<f64> = rf.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        // All predictions from depth-0 trees are the mean of their bootstrap
        // sample, which converges to the global mean 25.0.
        let global_mean = 25.0;
        for &p in preds.iter() {
            assert_abs_diff_eq!(p, global_mean, epsilon = 3.0);
        }
    }
}
