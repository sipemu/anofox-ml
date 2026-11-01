use anofox_ml_core::{Fit, FitWeighted, Float, Predict, Result, RustMlError};
use anofox_ml_trees::node::TreeNode;
use anofox_ml_trees::split::SplitCriterion;
use anofox_ml_trees::MaxFeatures;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::forest_common::{build_extra_tree, resolve_max_features, tree_input, ExtraTreeParams};

/// Extra-Trees (Extremely Randomized Trees) regressor parameters (unfitted state).
///
/// Trains an ensemble of decision trees using random split thresholds instead of
/// the best possible split at each node. Unlike Random Forests, Extra-Trees does
/// **not** use bootstrap sampling — each tree is trained on the full dataset.
/// If `max_features` is set, every split draws a fresh random feature subset;
/// default `None` = all features (sklearn's `max_features=1.0`).
///
/// Predictions are the average of individual tree predictions.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExtraTreesRegressor {
    /// Number of trees in the forest.
    pub n_estimators: usize,
    /// Maximum depth of each tree.
    pub max_depth: Option<usize>,
    /// Minimum samples required to split a node.
    pub min_samples_split: usize,
    /// Minimum samples required in a leaf node.
    pub min_samples_leaf: usize,
    /// Number of features to consider at each split. `None` = all features.
    pub max_features: Option<MaxFeatures>,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl ExtraTreesRegressor {
    /// Create a new `ExtraTreesRegressor` with the given number of trees and default parameters.
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

    /// Set the number of features to consider at each split (`None` = all).
    pub fn with_max_features(mut self, max_features: Option<MaxFeatures>) -> Self {
        self.max_features = max_features;
        self
    }

    /// Set the random seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl Default for ExtraTreesRegressor {
    fn default() -> Self {
        Self::new(100)
    }
}

/// A single tree in the Extra-Trees ensemble together with the feature indices it reads.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
struct ExtraForestTree<F: Float> {
    tree: TreeNode<F>,
    /// Indices of the input features this tree was trained on (`0..n_features`
    /// for trees built by this version; older serialized models may hold a
    /// per-tree subset).
    feature_indices: Vec<usize>,
    /// Number of features the tree was trained on.
    n_features_tree: usize,
}

/// Fitted Extra-Trees regressor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedExtraTreesRegressor<F: Float> {
    trees: Vec<ExtraForestTree<F>>,
    n_features: usize,
}

impl<F: Float> Fit<F> for ExtraTreesRegressor {
    type Fitted = FittedExtraTreesRegressor<F>;

    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Self::Fitted> {
        self.fit_impl(x, y, None)
    }
}

impl<F: Float> FitWeighted<F> for ExtraTreesRegressor {
    type Fitted = FittedExtraTreesRegressor<F>;

    /// Per-sample weights (sklearn's `sample_weight`).
    fn fit_weighted(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        sample_weight: Option<&Array1<F>>,
    ) -> Result<Self::Fitted> {
        self.fit_impl(x, y, sample_weight)
    }
}

impl ExtraTreesRegressor {
    fn fit_impl<F: Float>(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        sample_weight: Option<&Array1<F>>,
    ) -> Result<FittedExtraTreesRegressor<F>> {
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

        if let Some(w) = sample_weight {
            anofox_ml_core::check_sample_weight(w, y.len())?;
        }
        let n_features = x.ncols();
        let max_features_k = resolve_max_features(self.max_features, n_features)?;

        // Pre-generate per-tree seeds sequentially for determinism under rayon.
        // ExtraTrees does NOT use bootstrap — each tree trains on the full dataset.
        let mut rng = StdRng::seed_from_u64(self.seed);
        let tree_seeds: Vec<u64> = (0..self.n_estimators).map(|_| rng.gen()).collect();

        let params = ExtraTreeParams {
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            criterion: SplitCriterion::Mse,
            max_features_k,
            n_features,
        };
        let indices: Vec<usize> = (0..x.nrows()).collect();

        let trees: Vec<ExtraForestTree<F>> = tree_seeds
            .into_par_iter()
            .map(|tree_seed| ExtraForestTree {
                tree: build_extra_tree(x, y, &indices, 0, &params, tree_seed, sample_weight),
                feature_indices: (0..n_features).collect(),
                n_features_tree: n_features,
            })
            .collect();

        Ok(FittedExtraTreesRegressor { trees, n_features })
    }
}

impl<F: Float> Predict<F> for FittedExtraTreesRegressor<F> {
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

        // Collect all tree predictions in parallel (layout-agnostic traversal).
        let all_preds: Vec<Array1<F>> = self
            .trees
            .par_iter()
            .map(|forest_tree| {
                let sub_x = tree_input(x, &forest_tree.feature_indices);
                Array1::from_iter(
                    sub_x
                        .rows()
                        .into_iter()
                        .map(|row| forest_tree.tree.predict_view(row)),
                )
            })
            .collect();

        // Average predictions across trees (summed in tree order).
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

impl<F: Float> FittedExtraTreesRegressor<F> {
    /// Feature importances averaged across all trees and normalized to sum to 1.
    pub fn feature_importances(&self) -> Array1<F> {
        let mut importances = vec![F::zero(); self.n_features];
        let n_trees = F::from_usize(self.trees.len()).unwrap();

        for forest_tree in &self.trees {
            let total_samples = tree_n_samples(&forest_tree.tree);
            let tree_raw = forest_tree
                .tree
                .feature_importances(forest_tree.n_features_tree, total_samples);
            // Normalize individual tree importances
            let sum: F = tree_raw.iter().copied().fold(F::zero(), |a, b| a + b);
            for (local_idx, &original_idx) in forest_tree.feature_indices.iter().enumerate() {
                if sum > F::zero() {
                    importances[original_idx] += (tree_raw[local_idx] / sum) / n_trees;
                }
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

    /// Number of trees in the ensemble.
    pub fn n_estimators(&self) -> usize {
        self.trees.len()
    }
}

fn tree_n_samples<F: Float>(node: &TreeNode<F>) -> usize {
    match node {
        TreeNode::Leaf { n_samples, .. } => *n_samples,
        TreeNode::Split { n_samples, .. } => *n_samples,
    }
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

        let et = ExtraTreesRegressor {
            n_estimators: 50,
            max_depth: None,
            seed: 42,
            ..Default::default()
        };
        let fitted: FittedExtraTreesRegressor<f64> = et.fit(&x, &y).unwrap();

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

        let et = ExtraTreesRegressor {
            n_estimators: 10,
            seed: 123,
            ..Default::default()
        };

        let fitted1: FittedExtraTreesRegressor<f64> = et.fit(&x, &y).unwrap();
        let fitted2: FittedExtraTreesRegressor<f64> = et.fit(&x, &y).unwrap();

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

        let et = ExtraTreesRegressor {
            n_estimators: 30,
            max_features: Some(MaxFeatures::Fixed(2)),
            seed: 99,
            ..Default::default()
        };
        let fitted: FittedExtraTreesRegressor<f64> = et.fit(&x, &y).unwrap();

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

        let et = ExtraTreesRegressor {
            n_estimators: 20,
            seed: 7,
            ..Default::default()
        };
        let fitted: FittedExtraTreesRegressor<f64> = et.fit(&x, &y).unwrap();

        let importances = fitted.feature_importances();
        let sum: f64 = importances.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_feature_importances_non_negative() {
        let x = array![
            [1.0, 100.0, 0.5],
            [2.0, 200.0, 0.6],
            [3.0, 300.0, 0.7],
            [10.0, 400.0, 0.8],
            [11.0, 500.0, 0.9],
            [12.0, 600.0, 1.0]
        ];
        let y = array![1.0, 2.0, 3.0, 10.0, 11.0, 12.0];

        let et = ExtraTreesRegressor {
            n_estimators: 20,
            seed: 7,
            ..Default::default()
        };
        let fitted: FittedExtraTreesRegressor<f64> = et.fit(&x, &y).unwrap();

        let importances = fitted.feature_importances();
        for &imp in importances.iter() {
            assert!(
                imp >= 0.0,
                "feature importance must be non-negative, got {imp}"
            );
        }
    }

    #[test]
    fn test_n_estimators() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![1.0, 2.0, 3.0, 4.0];

        let et = ExtraTreesRegressor {
            n_estimators: 7,
            seed: 0,
            ..Default::default()
        };
        let fitted: FittedExtraTreesRegressor<f64> = et.fit(&x, &y).unwrap();
        assert_eq!(fitted.n_estimators(), 7);
    }

    #[test]
    fn test_shape_mismatch_error() {
        let x = array![[1.0], [2.0]];
        let y = array![0.0, 1.0, 2.0];

        let et = ExtraTreesRegressor::default();
        let result: std::result::Result<FittedExtraTreesRegressor<f64>, _> = et.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_predict_wrong_features_error() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 2.0];

        let et = ExtraTreesRegressor {
            n_estimators: 5,
            seed: 0,
            ..Default::default()
        };
        let fitted: FittedExtraTreesRegressor<f64> = et.fit(&x, &y).unwrap();

        let x_bad = array![[1.0], [2.0]];
        let result = fitted.predict(&x_bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_max_features() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 2.0];

        let et = ExtraTreesRegressor {
            n_estimators: 5,
            max_features: Some(MaxFeatures::Fixed(5)),
            seed: 0,
            ..Default::default()
        };
        let result: std::result::Result<FittedExtraTreesRegressor<f64>, _> = et.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_estimators_error() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 2.0];

        let et = ExtraTreesRegressor {
            n_estimators: 0,
            seed: 0,
            ..Default::default()
        };
        let result: std::result::Result<FittedExtraTreesRegressor<f64>, _> = et.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_input_error() {
        let x: Array2<f64> = Array2::zeros((0, 2));
        let y: Array1<f64> = Array1::zeros(0);

        let et = ExtraTreesRegressor::default();
        let result: std::result::Result<FittedExtraTreesRegressor<f64>, _> = et.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_average_predictions() {
        // With max_depth=0 (immediate leaf), each tree predicts the mean of
        // the full dataset (no bootstrap). So the forest should predict exactly
        // the global mean.
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![10.0, 20.0, 30.0, 40.0];

        let et = ExtraTreesRegressor {
            n_estimators: 200,
            max_depth: Some(0),
            seed: 42,
            ..Default::default()
        };
        let fitted: FittedExtraTreesRegressor<f64> = et.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        // All predictions should be exactly the global mean 25.0 since
        // ExtraTrees trains on the full dataset (no bootstrap variance).
        let global_mean = 25.0;
        for &p in preds.iter() {
            assert_abs_diff_eq!(p, global_mean, epsilon = 1e-10);
        }
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
        let y = array![1.0, 2.0, 3.0, 10.0, 11.0, 12.0];

        let et = ExtraTreesRegressor {
            n_estimators: 1,
            max_depth: Some(3),
            seed: 42,
            ..Default::default()
        };
        let fitted: FittedExtraTreesRegressor<f64> = et.fit(&x, &y).unwrap();
        assert_eq!(fitted.n_estimators(), 1);

        // A single tree should still produce valid predictions.
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), y.len());
        for &p in preds.iter() {
            assert!(p.is_finite(), "prediction should be finite, got {p}");
        }
    }

    mod prop_tests {
        use super::*;
        use proptest::prelude::*;

        /// Generate deterministic regression data.
        fn make_regression_data(
            n_samples: usize,
            n_features: usize,
            seed: u64,
        ) -> (Array2<f64>, Array1<f64>) {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let mut x_data = Vec::with_capacity(n_samples * n_features);
            let mut y_data = Vec::with_capacity(n_samples);

            for i in 0..n_samples {
                let mut y_val = 0.0;
                for j in 0..n_features {
                    let mut h = DefaultHasher::new();
                    seed.hash(&mut h);
                    (i as u64).hash(&mut h);
                    (j as u64).hash(&mut h);
                    let bits = h.finish();
                    let v = (bits as f64 / u64::MAX as f64) * 20.0 - 10.0;
                    x_data.push(v);
                    y_val += v * (j as f64 + 1.0);
                }
                y_data.push(y_val);
            }

            let x = Array2::from_shape_vec((n_samples, n_features), x_data).unwrap();
            let y = Array1::from_vec(y_data);
            (x, y)
        }

        proptest! {
            #[test]
            fn predictions_are_finite(
                n_samples in 6..30usize,
                n_features in 1..5usize,
                seed in 0u64..1000,
            ) {
                let (x, y) = make_regression_data(n_samples, n_features, seed);

                let et = ExtraTreesRegressor {
                    n_estimators: 10,
                    max_depth: Some(5),
                    seed: seed as u64,
                    ..Default::default()
                };
                let fitted: FittedExtraTreesRegressor<f64> = et.fit(&x, &y).unwrap();
                let preds = fitted.predict(&x).unwrap();

                for (i, &p) in preds.iter().enumerate() {
                    prop_assert!(
                        p.is_finite(),
                        "prediction at index {} is not finite: {}",
                        i, p
                    );
                }
            }

            #[test]
            fn feature_importances_sum_to_one(
                n_samples in 6..30usize,
                n_features in 1..5usize,
                seed in 0u64..1000,
            ) {
                let (x, y) = make_regression_data(n_samples, n_features, seed);

                let et = ExtraTreesRegressor {
                    n_estimators: 10,
                    max_depth: Some(5),
                    seed: seed as u64,
                    ..Default::default()
                };
                let fitted: FittedExtraTreesRegressor<f64> = et.fit(&x, &y).unwrap();
                let importances = fitted.feature_importances();
                let sum: f64 = importances.iter().sum();

                prop_assert!(
                    (sum - 1.0).abs() < 1e-10,
                    "feature importances sum to {} (expected ~1.0), n_samples={}, n_features={}, seed={}",
                    sum, n_samples, n_features, seed
                );
            }
        }
    }
}
