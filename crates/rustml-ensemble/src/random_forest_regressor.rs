use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use rustml_core::{Fit, Float, Predict, Result, RustMlError};
use rustml_trees::{DecisionTreeRegressor, FittedDecisionTreeRegressor};

use rustml_trees::SplitCriterion;

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
    /// Split criterion for each tree. Default: MSE.
    pub criterion: SplitCriterion,
    /// Whether to use bootstrap sampling. Default: true.
    pub bootstrap: bool,
    /// Fraction of samples to draw for each tree (when bootstrap=true).
    /// If `None`, draws n_samples (with replacement). Value in (0, 1].
    pub max_samples: Option<f64>,
    /// Whether to compute out-of-bag score after fitting.
    pub oob_score: bool,
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
            criterion: SplitCriterion::Mse,
            bootstrap: true,
            max_samples: None,
            oob_score: false,
            seed: 0,
        }
    }

    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split;
        self
    }
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }
    pub fn with_max_features(mut self, max_features: Option<usize>) -> Self {
        self.max_features = max_features;
        self
    }
    pub fn with_criterion(mut self, criterion: SplitCriterion) -> Self {
        self.criterion = criterion;
        self
    }
    pub fn with_bootstrap(mut self, bootstrap: bool) -> Self {
        self.bootstrap = bootstrap;
        self
    }
    pub fn with_max_samples(mut self, max_samples: Option<f64>) -> Self {
        self.max_samples = max_samples;
        self
    }
    pub fn with_oob_score(mut self, oob_score: bool) -> Self {
        self.oob_score = oob_score;
        self
    }
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
    /// OOB R² score (only set when oob_score=true).
    oob_score_value: Option<f64>,
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

        let draw_size = if let Some(frac) = self.max_samples {
            if frac <= 0.0 || frac > 1.0 {
                return Err(RustMlError::InvalidParameter(
                    "max_samples must be in (0, 1]".into(),
                ));
            }
            (n_samples as f64 * frac).ceil() as usize
        } else {
            n_samples
        };

        let tree_params = DecisionTreeRegressor {
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            max_features: None,
            sample_weight: None,
        };

        // Pre-generate row and feature indices for determinism
        let sample_plans: Vec<(Vec<usize>, Vec<usize>)> = (0..self.n_estimators)
            .map(|_| {
                let row_indices: Vec<usize> = if self.bootstrap {
                    (0..draw_size)
                        .map(|_| rng.gen_range(0..n_samples))
                        .collect()
                } else {
                    (0..n_samples).collect()
                };
                let feature_indices = select_features(n_features, self.max_features, &mut rng);
                (row_indices, feature_indices)
            })
            .collect();

        let all_row_indices: Vec<Vec<usize>> = if self.oob_score {
            sample_plans.iter().map(|(ri, _)| ri.clone()).collect()
        } else {
            Vec::new()
        };

        // Train trees in parallel
        let trees: Result<Vec<ForestTree<F>>> = sample_plans
            .into_par_iter()
            .map(|(row_indices, feature_indices)| {
                let x_sub = build_sub_matrix(x, &row_indices, &feature_indices);
                let y_sub = Array1::from_vec(row_indices.iter().map(|&i| y[i]).collect::<Vec<F>>());
                let fitted_tree: FittedDecisionTreeRegressor<F> =
                    tree_params.fit(&x_sub, &y_sub)?;
                Ok(ForestTree {
                    tree: fitted_tree,
                    feature_indices,
                })
            })
            .collect();
        let trees = trees?;

        let oob_score_value = if self.oob_score && self.bootstrap {
            compute_oob_score_regression(&trees, x, y, n_samples, &all_row_indices)
        } else {
            None
        };

        Ok(FittedRandomForestRegressor {
            trees,
            n_features,
            oob_score_value,
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
        let n_trees_f = F::from_usize(self.trees.len()).unwrap();

        // Collect all tree predictions in parallel
        let all_preds: Result<Vec<Array1<F>>> = self
            .trees
            .par_iter()
            .map(|forest_tree| {
                let sub_x = build_sub_matrix_cols(x, &forest_tree.feature_indices);
                forest_tree.tree.predict(&sub_x)
            })
            .collect();
        let all_preds = all_preds?;

        // Average predictions across trees
        let mut predictions = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let mut sum = F::zero();
            for tree_pred in &all_preds {
                sum += tree_pred[i];
            }
            predictions.push(sum / n_trees_f);
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

    /// OOB R² score (only available when `oob_score=true` and `bootstrap=true`).
    pub fn oob_score(&self) -> Option<f64> {
        self.oob_score_value
    }

    /// Compute R² score on the given data.
    pub fn score(&self, x: &Array2<F>, y: &Array1<F>) -> Result<f64> {
        let preds = self.predict(x)?;
        let n = y.len();
        let y_mean = y.iter().copied().fold(F::zero(), |a, b| a + b) / F::from_usize(n).unwrap();
        let ss_res: f64 = preds
            .iter()
            .zip(y.iter())
            .map(|(&p, &t)| (p - t).to_f64().unwrap().powi(2))
            .sum();
        let ss_tot: f64 = y
            .iter()
            .map(|&t| (t - y_mean).to_f64().unwrap().powi(2))
            .sum();
        Ok(if ss_tot > 0.0 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        })
    }
}

/// Compute OOB R² score for regression.
fn compute_oob_score_regression<F: Float>(
    trees: &[ForestTree<F>],
    x: &Array2<F>,
    y: &Array1<F>,
    n_samples: usize,
    bootstrap_indices: &[Vec<usize>],
) -> Option<f64> {
    use std::collections::HashSet;

    let mut oob_preds = vec![0.0f64; n_samples];
    let mut oob_counts = vec![0usize; n_samples];

    for (t_idx, forest_tree) in trees.iter().enumerate() {
        let in_bag: HashSet<usize> = bootstrap_indices[t_idx].iter().copied().collect();
        for i in 0..n_samples {
            if in_bag.contains(&i) {
                continue;
            }
            let sub_x = build_sub_matrix_cols_single(x, i, &forest_tree.feature_indices);
            if let Ok(pred) = forest_tree.tree.predict(&sub_x) {
                oob_preds[i] += pred[0].to_f64().unwrap();
                oob_counts[i] += 1;
            }
        }
    }

    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    let mut y_sum = 0.0;
    let mut n_eval = 0usize;

    for i in 0..n_samples {
        if oob_counts[i] > 0 {
            oob_preds[i] /= oob_counts[i] as f64;
            y_sum += y[i].to_f64().unwrap();
            n_eval += 1;
        }
    }

    if n_eval == 0 {
        return None;
    }

    let y_mean = y_sum / n_eval as f64;
    for i in 0..n_samples {
        if oob_counts[i] > 0 {
            let yi = y[i].to_f64().unwrap();
            ss_res += (yi - oob_preds[i]).powi(2);
            ss_tot += (yi - y_mean).powi(2);
        }
    }

    Some(if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    })
}

/// Build a single-row sub-matrix selecting specific columns for one sample.
fn build_sub_matrix_cols_single<F: Float>(
    x: &Array2<F>,
    row: usize,
    col_indices: &[usize],
) -> Array2<F> {
    let n_cols = col_indices.len();
    let data: Vec<F> = col_indices.iter().map(|&ci| x[[row, ci]]).collect();
    Array2::from_shape_vec((1, n_cols), data).expect("shape matches data length")
}

/// Build a sub-matrix selecting all rows but only specific columns from `x`.
fn build_sub_matrix_cols<F: Float>(x: &Array2<F>, col_indices: &[usize]) -> Array2<F> {
    let n_rows = x.nrows();
    let n_cols = col_indices.len();
    let mut data = Vec::with_capacity(n_rows * n_cols);
    for i in 0..n_rows {
        for &ci in col_indices {
            data.push(x[[i, ci]]);
        }
    }
    Array2::from_shape_vec((n_rows, n_cols), data).expect("shape matches data length")
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
