use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use rustml_core::{Fit, Float, Predict, Result, RustMlError};
use rustml_trees::{ClassWeight, DecisionTreeClassifier, FittedDecisionTreeClassifier, SplitCriterion};

/// Random forest classifier parameters (unfitted state).
///
/// Trains an ensemble of decision tree classifiers, each on a bootstrap sample
/// of the data with an optional random subset of features.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RandomForestClassifier {
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
    /// Split criterion for each tree. Default: Gini.
    pub criterion: SplitCriterion,
    /// Whether to use bootstrap sampling. Default: true.
    pub bootstrap: bool,
    /// Fraction of samples to draw for each tree (when bootstrap=true).
    /// If `None`, draws n_samples (with replacement). Value in (0, 1].
    pub max_samples: Option<f64>,
    /// Class weight strategy passed to each tree.
    pub class_weight: Option<ClassWeight>,
    /// Whether to compute out-of-bag score after fitting.
    pub oob_score: bool,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl RandomForestClassifier {
    /// Create a new `RandomForestClassifier` with the given number of trees and default parameters.
    pub fn new(n_estimators: usize) -> Self {
        Self {
            n_estimators,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            criterion: SplitCriterion::Gini,
            bootstrap: true,
            max_samples: None,
            class_weight: None,
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
    pub fn with_class_weight(mut self, class_weight: Option<ClassWeight>) -> Self {
        self.class_weight = class_weight;
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

impl Default for RandomForestClassifier {
    fn default() -> Self {
        Self::new(100)
    }
}

/// A single tree in the forest together with its selected feature indices.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
struct ForestTree<F: Float> {
    tree: FittedDecisionTreeClassifier<F>,
    /// Indices of the features this tree was trained on (relative to the
    /// original feature matrix). When `max_features` is `None` this contains
    /// `0..n_features`.
    feature_indices: Vec<usize>,
}

/// Fitted random forest classifier.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedRandomForestClassifier<F: Float> {
    trees: Vec<ForestTree<F>>,
    n_features: usize,
    /// OOB accuracy score (only set when oob_score=true).
    oob_score_value: Option<f64>,
}

impl<F: Float> Fit<F> for RandomForestClassifier {
    type Fitted = FittedRandomForestClassifier<F>;

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

        // Compute bootstrap sample size
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

        let tree_params = DecisionTreeClassifier {
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            criterion: self.criterion,
            max_features: None,
            sample_weight: None,
            class_weight: self.class_weight.clone(),
        };

        // Pre-generate bootstrap/sample and feature indices for determinism
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

        // Keep a copy of row indices for OOB scoring
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
                let y_sub = Array1::from_vec(
                    row_indices.iter().map(|&i| y[i]).collect::<Vec<F>>(),
                );
                let fitted_tree: FittedDecisionTreeClassifier<F> =
                    tree_params.fit(&x_sub, &y_sub)?;
                Ok(ForestTree {
                    tree: fitted_tree,
                    feature_indices,
                })
            })
            .collect();
        let trees = trees?;

        // Compute OOB score if requested
        let oob_score_value = if self.oob_score && self.bootstrap {
            compute_oob_score_classification(&trees, x, y, n_samples, &all_row_indices)
        } else {
            None
        };

        Ok(FittedRandomForestClassifier {
            trees,
            n_features,
            oob_score_value,
        })
    }
}

impl<F: Float> Predict<F> for FittedRandomForestClassifier<F> {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }

        let n_samples = x.nrows();
        let n_trees = self.trees.len();

        // Collect all tree predictions in parallel
        let all_preds: Result<Vec<Array1<F>>> = self.trees
            .par_iter()
            .map(|forest_tree| {
                let sub_x = build_sub_matrix_cols(x, &forest_tree.feature_indices);
                forest_tree.tree.predict(&sub_x)
            })
            .collect();
        let all_preds = all_preds?;

        // Aggregate votes per sample
        let mut predictions = Vec::with_capacity(n_samples);
        let mut votes = Vec::with_capacity(n_trees);
        for i in 0..n_samples {
            votes.clear();
            for tree_pred in &all_preds {
                votes.push(tree_pred[i]);
            }
            predictions.push(majority_vote(&votes));
        }

        Ok(Array1::from_vec(predictions))
    }
}

impl<F: Float> FittedRandomForestClassifier<F> {
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

    /// Predict class probabilities for each sample.
    ///
    /// Returns an `Array2<F>` of shape `(n_samples, n_classes)` where each row
    /// sums to 1.0. Probabilities are averaged across all trees in the forest.
    pub fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }

        // Collect probabilities from each tree in parallel
        let all_proba: Result<Vec<Array2<F>>> = self
            .trees
            .par_iter()
            .map(|forest_tree| {
                let sub_x = build_sub_matrix_cols(x, &forest_tree.feature_indices);
                forest_tree.tree.predict_proba(&sub_x)
            })
            .collect();
        let all_proba = all_proba?;

        // Determine global class set (union of all trees' classes)
        let mut global_classes: Vec<F> = Vec::new();
        let eps = F::from_f64(1e-9).unwrap();
        for forest_tree in &self.trees {
            for c in forest_tree.tree.classes() {
                if !global_classes
                    .iter()
                    .any(|&gc| (gc - c).abs() < eps)
                {
                    global_classes.push(c);
                }
            }
        }
        global_classes.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n_samples = x.nrows();
        let n_classes = global_classes.len();
        let n_trees_f = F::from_usize(self.trees.len()).unwrap();
        let mut avg_proba = Array2::<F>::zeros((n_samples, n_classes));

        // Map each tree's probabilities to the global class indices and average
        for (tree_idx, forest_tree) in self.trees.iter().enumerate() {
            let tree_classes = forest_tree.tree.classes();
            let tree_proba = &all_proba[tree_idx];

            for (local_ci, &tc) in tree_classes.iter().enumerate() {
                if let Some(global_ci) = global_classes
                    .iter()
                    .position(|&gc| (gc - tc).abs() < eps)
                {
                    for i in 0..n_samples {
                        avg_proba[[i, global_ci]] += tree_proba[[i, local_ci]] / n_trees_f;
                    }
                }
            }
        }

        Ok(avg_proba)
    }

    /// OOB accuracy score (only available when `oob_score=true` and `bootstrap=true`).
    pub fn oob_score(&self) -> Option<f64> {
        self.oob_score_value
    }

    /// Compute classification accuracy on the given data.
    pub fn score(&self, x: &Array2<F>, y: &Array1<F>) -> Result<f64> {
        let preds = self.predict(x)?;
        let n = y.len();
        let correct = preds
            .iter()
            .zip(y.iter())
            .filter(|(&p, &t)| (p - t).abs() < F::from_f64(1e-9).unwrap())
            .count();
        Ok(correct as f64 / n as f64)
    }

    /// Returns the unique sorted class labels across all trees.
    pub fn classes(&self) -> Vec<F> {
        let eps = F::from_f64(1e-9).unwrap();
        let mut classes: Vec<F> = Vec::new();
        for forest_tree in &self.trees {
            for c in forest_tree.tree.classes() {
                if !classes.iter().any(|&gc| (gc - c).abs() < eps) {
                    classes.push(c);
                }
            }
        }
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        classes
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
    // Select rows first (produces C-contiguous), then columns.
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

/// Build a sub-matrix selecting all rows but only specific columns from `x`.
/// Produces a guaranteed C-contiguous (standard layout) array so that
/// `row.as_slice()` works in downstream predict calls.
fn build_sub_matrix_cols<F: Float>(
    x: &Array2<F>,
    col_indices: &[usize],
) -> Array2<F> {
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

/// Compute OOB classification accuracy.
/// For each sample, only trees that did NOT include it in their bootstrap are used.
fn compute_oob_score_classification<F: Float>(
    trees: &[ForestTree<F>],
    x: &Array2<F>,
    y: &Array1<F>,
    n_samples: usize,
    bootstrap_indices: &[Vec<usize>],
) -> Option<f64> {
    use std::collections::HashSet;

    let mut correct = 0usize;
    let mut evaluated = 0usize;

    for i in 0..n_samples {
        let mut votes: Vec<F> = Vec::new();
        for (t_idx, forest_tree) in trees.iter().enumerate() {
            // Check if sample i was NOT in the bootstrap for tree t_idx
            let in_bag: HashSet<usize> = bootstrap_indices[t_idx].iter().copied().collect();
            if in_bag.contains(&i) {
                continue;
            }
            // Get prediction from this tree for sample i
            let sub_x = build_sub_matrix_cols_single(x, i, &forest_tree.feature_indices);
            if let Ok(pred) = forest_tree.tree.predict(&sub_x) {
                votes.push(pred[0]);
            }
        }
        if !votes.is_empty() {
            let oob_pred = majority_vote(&votes);
            if (oob_pred - y[i]).abs() < F::from_f64(1e-9).unwrap() {
                correct += 1;
            }
            evaluated += 1;
        }
    }

    if evaluated > 0 {
        Some(correct as f64 / evaluated as f64)
    } else {
        None
    }
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

/// Return the class that appears most frequently in `votes`.
/// Uses HashMap with f64 bit representation for O(1) lookup per vote.
#[inline]
fn majority_vote<F: Float>(votes: &[F]) -> F {
    use std::collections::HashMap;
    let mut counts: HashMap<u64, (F, usize)> = HashMap::new();
    for &v in votes {
        let key = v.to_f64().unwrap().to_bits();
        counts
            .entry(key)
            .and_modify(|e| e.1 += 1)
            .or_insert((v, 1));
    }
    counts
        .into_values()
        .max_by_key(|&(_, count)| count)
        .unwrap()
        .0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_basic_classification() {
        let x = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [10.0, 1.0],
            [11.0, 1.0],
            [12.0, 1.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let rf = RandomForestClassifier {
            n_estimators: 20,
            max_depth: Some(3),
            seed: 42,
            ..Default::default()
        };
        let fitted: FittedRandomForestClassifier<f64> = rf.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(*p, *t, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_reproducibility() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let rf = RandomForestClassifier {
            n_estimators: 10,
            seed: 123,
            ..Default::default()
        };

        let fitted1: FittedRandomForestClassifier<f64> = rf.fit(&x, &y).unwrap();
        let fitted2: FittedRandomForestClassifier<f64> = rf.fit(&x, &y).unwrap();

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
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let rf = RandomForestClassifier {
            n_estimators: 30,
            max_features: Some(2),
            seed: 99,
            ..Default::default()
        };
        let fitted: FittedRandomForestClassifier<f64> = rf.fit(&x, &y).unwrap();

        // Training accuracy should be high
        let preds = fitted.predict(&x).unwrap();
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(*p, *t, epsilon = 1e-10);
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
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let rf = RandomForestClassifier {
            n_estimators: 20,
            seed: 7,
            ..Default::default()
        };
        let fitted: FittedRandomForestClassifier<f64> = rf.fit(&x, &y).unwrap();

        let importances = fitted.feature_importances();
        let sum: f64 = importances.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_n_estimators() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let rf = RandomForestClassifier {
            n_estimators: 7,
            seed: 0,
            ..Default::default()
        };
        let fitted: FittedRandomForestClassifier<f64> = rf.fit(&x, &y).unwrap();
        assert_eq!(fitted.n_estimators(), 7);
    }

    #[test]
    fn test_shape_mismatch_error() {
        let x = array![[1.0], [2.0]];
        let y = array![0.0, 1.0, 2.0];

        let rf = RandomForestClassifier::default();
        let result: std::result::Result<FittedRandomForestClassifier<f64>, _> = rf.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_predict_wrong_features_error() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0];

        let rf = RandomForestClassifier {
            n_estimators: 5,
            seed: 0,
            ..Default::default()
        };
        let fitted: FittedRandomForestClassifier<f64> = rf.fit(&x, &y).unwrap();

        let x_bad = array![[1.0], [2.0]];
        let result = fitted.predict(&x_bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_max_features() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0];

        let rf = RandomForestClassifier {
            n_estimators: 5,
            max_features: Some(5),
            seed: 0,
            ..Default::default()
        };
        let result: std::result::Result<FittedRandomForestClassifier<f64>, _> = rf.fit(&x, &y);
        assert!(result.is_err());
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
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let rf = RandomForestClassifier {
            n_estimators: 20,
            seed: 7,
            ..Default::default()
        };
        let fitted: FittedRandomForestClassifier<f64> = rf.fit(&x, &y).unwrap();

        let importances = fitted.feature_importances();
        for &imp in importances.iter() {
            assert!(
                imp >= 0.0,
                "feature importance must be non-negative, got {imp}"
            );
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
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let rf = RandomForestClassifier {
            n_estimators: 1,
            max_depth: Some(3),
            seed: 42,
            ..Default::default()
        };
        let fitted: FittedRandomForestClassifier<f64> = rf.fit(&x, &y).unwrap();
        assert_eq!(fitted.n_estimators(), 1);

        // A single tree should still produce valid predictions.
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), y.len());
    }

    #[test]
    fn test_predictions_are_valid_labels() {
        let x = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [10.0, 1.0],
            [11.0, 1.0],
            [12.0, 1.0],
            [20.0, 2.0],
            [21.0, 2.0],
            [22.0, 2.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];

        let rf = RandomForestClassifier {
            n_estimators: 30,
            max_depth: Some(5),
            seed: 42,
            ..Default::default()
        };
        let fitted: FittedRandomForestClassifier<f64> = rf.fit(&x, &y).unwrap();

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
    fn test_empty_input_error() {
        let x: Array2<f64> = Array2::zeros((0, 2));
        let y: Array1<f64> = Array1::zeros(0);

        let rf = RandomForestClassifier::default();
        let result: std::result::Result<FittedRandomForestClassifier<f64>, _> = rf.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_estimators_error() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0];

        let rf = RandomForestClassifier {
            n_estimators: 0,
            seed: 0,
            ..Default::default()
        };
        let result: std::result::Result<FittedRandomForestClassifier<f64>, _> = rf.fit(&x, &y);
        assert!(result.is_err());
    }

    mod prop_tests {
        use super::*;
        use proptest::prelude::*;
        use std::collections::HashSet;

        /// Generate deterministic training data for classification.
        fn make_classification_data(
            n_samples: usize,
            n_features: usize,
            n_classes: usize,
            seed: u64,
        ) -> (Array2<f64>, Array1<f64>) {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let mut x_data = Vec::with_capacity(n_samples * n_features);
            let mut y_data = Vec::with_capacity(n_samples);

            for i in 0..n_samples {
                for j in 0..n_features {
                    let mut h = DefaultHasher::new();
                    seed.hash(&mut h);
                    (i as u64).hash(&mut h);
                    (j as u64).hash(&mut h);
                    let bits = h.finish();
                    let v = (bits as f64 / u64::MAX as f64) * 20.0 - 10.0;
                    x_data.push(v);
                }
                let mut h = DefaultHasher::new();
                seed.hash(&mut h);
                (i as u64).hash(&mut h);
                0xDEAD_BEEFu64.hash(&mut h);
                let label = (h.finish() % n_classes as u64) as f64;
                y_data.push(label);
            }

            let x = Array2::from_shape_vec((n_samples, n_features), x_data).unwrap();
            let y = Array1::from_vec(y_data);
            (x, y)
        }

        proptest! {
            #[test]
            fn predictions_are_valid_labels(
                n_samples in 6..30usize,
                n_features in 1..5usize,
                n_classes in 2..5usize,
                seed in 0u64..1000,
            ) {
                let (x, y) = make_classification_data(n_samples, n_features, n_classes, seed);

                let train_labels: HashSet<u64> = y.iter()
                    .map(|&v| v.to_bits())
                    .collect();

                let rf = RandomForestClassifier {
                    n_estimators: 10,
                    max_depth: Some(5),
                    seed: seed as u64,
                    ..Default::default()
                };
                let fitted: FittedRandomForestClassifier<f64> = rf.fit(&x, &y).unwrap();
                let preds = fitted.predict(&x).unwrap();

                for (i, &p) in preds.iter().enumerate() {
                    prop_assert!(
                        train_labels.contains(&p.to_bits()),
                        "prediction {} at index {} is not a valid training label",
                        p, i
                    );
                }
            }

            #[test]
            fn feature_importances_sum_to_one(
                n_samples in 6..30usize,
                n_features in 1..5usize,
                seed in 0u64..1000,
            ) {
                let n_classes = 3;
                let (x, y) = make_classification_data(n_samples, n_features, n_classes, seed);

                let rf = RandomForestClassifier {
                    n_estimators: 10,
                    max_depth: Some(5),
                    seed: seed as u64,
                    ..Default::default()
                };
                let fitted: FittedRandomForestClassifier<f64> = rf.fit(&x, &y).unwrap();
                let importances = fitted.feature_importances();
                let sum: f64 = importances.iter().sum();

                // Valid outcomes: (1) importances are a valid probability
                // distribution (sum to 1), or (2) every tree is a pure leaf
                // with no splits (e.g. degenerate tiny input produces uniform
                // labels) and all importances are zero.
                prop_assert!(
                    (sum - 1.0).abs() < 1e-10 || sum == 0.0,
                    "feature importances sum to {} (expected ~1.0 or 0.0 for no-split case), n_samples={}, n_features={}, seed={}",
                    sum, n_samples, n_features, seed
                );
                // Importances must always be non-negative.
                for (i, &imp) in importances.iter().enumerate() {
                    prop_assert!(
                        imp >= 0.0,
                        "importance[{}] = {} is negative",
                        i, imp
                    );
                }
            }
        }
    }
}
