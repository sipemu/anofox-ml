use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use rustml_core::{Fit, Float, Predict, Result, RustMlError};
use rustml_trees::{DecisionTreeClassifier, FittedDecisionTreeClassifier, SplitCriterion};

/// Bagging (Bootstrap Aggregating) classifier parameters (unfitted state).
///
/// Trains an ensemble of decision tree classifiers, each on a bootstrap sample
/// of the data using the **full** feature set. Unlike [`RandomForestClassifier`],
/// bagging does not perform random feature subsampling at the tree level --
/// every tree sees all features.
///
/// Predictions are made by majority vote across all trees.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BaggingClassifier {
    /// Number of trees in the ensemble.
    pub n_estimators: usize,
    /// Maximum depth of each tree.
    pub max_depth: Option<usize>,
    /// Fraction of samples to draw for each tree (with replacement when
    /// `bootstrap=true`). If `None`, draws `n_samples`. Value in (0, 1].
    pub max_samples: Option<f64>,
    /// Whether to use bootstrap sampling. Default: true.
    pub bootstrap: bool,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl BaggingClassifier {
    /// Create a new `BaggingClassifier` with the given number of trees and default parameters.
    pub fn new(n_estimators: usize) -> Self {
        Self {
            n_estimators,
            max_depth: None,
            max_samples: None,
            bootstrap: true,
            seed: 0,
        }
    }

    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }
    pub fn with_max_samples(mut self, max_samples: Option<f64>) -> Self {
        self.max_samples = max_samples;
        self
    }
    pub fn with_bootstrap(mut self, bootstrap: bool) -> Self {
        self.bootstrap = bootstrap;
        self
    }
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl Default for BaggingClassifier {
    fn default() -> Self {
        Self::new(10)
    }
}

/// Fitted bagging classifier.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedBaggingClassifier<F: Float> {
    trees: Vec<FittedDecisionTreeClassifier<F>>,
    n_features: usize,
}

impl<F: Float> Fit<F> for BaggingClassifier {
    type Fitted = FittedBaggingClassifier<F>;

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
            min_samples_split: 2,
            min_samples_leaf: 1,
            criterion: SplitCriterion::Gini,
            max_features: None,
            sample_weight: None,
            class_weight: None,
        };

        // Pre-generate bootstrap row indices for determinism
        let sample_plans: Vec<Vec<usize>> = (0..self.n_estimators)
            .map(|_| {
                if self.bootstrap {
                    (0..draw_size)
                        .map(|_| rng.gen_range(0..n_samples))
                        .collect()
                } else {
                    (0..n_samples).collect()
                }
            })
            .collect();

        // Train trees in parallel -- no feature subsampling
        let trees: Result<Vec<FittedDecisionTreeClassifier<F>>> = sample_plans
            .into_par_iter()
            .map(|row_indices| {
                let x_sub = build_sub_matrix_rows(x, &row_indices);
                let y_sub = Array1::from_vec(
                    row_indices.iter().map(|&i| y[i]).collect::<Vec<F>>(),
                );
                tree_params.fit(&x_sub, &y_sub)
            })
            .collect();
        let trees = trees?;

        Ok(FittedBaggingClassifier { trees, n_features })
    }
}

impl<F: Float> Predict<F> for FittedBaggingClassifier<F> {
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
        let all_preds: Result<Vec<Array1<F>>> = self
            .trees
            .par_iter()
            .map(|tree| tree.predict(x))
            .collect();
        let all_preds = all_preds?;

        // Aggregate votes per sample (majority vote)
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

impl<F: Float> FittedBaggingClassifier<F> {
    /// Feature importances averaged across all trees and normalized to sum to 1.
    pub fn feature_importances(&self) -> Array1<F> {
        let mut importances = vec![F::zero(); self.n_features];
        let n_trees = F::from_usize(self.trees.len()).unwrap();

        for tree in &self.trees {
            let tree_importances = tree.feature_importances();
            for (idx, &imp) in tree_importances.iter().enumerate() {
                importances[idx] += imp / n_trees;
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

    /// Predict class probabilities for each sample.
    ///
    /// Returns an `Array2<F>` of shape `(n_samples, n_classes)` where each row
    /// sums to 1.0. Probabilities are averaged across all trees in the ensemble.
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
            .map(|tree| tree.predict_proba(x))
            .collect();
        let all_proba = all_proba?;

        // Determine global class set (union of all trees' classes)
        let mut global_classes: Vec<F> = Vec::new();
        let eps = F::from_f64(1e-9).unwrap();
        for tree in &self.trees {
            for c in tree.classes() {
                if !global_classes.iter().any(|&gc| (gc - c).abs() < eps) {
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
        for (tree_idx, tree) in self.trees.iter().enumerate() {
            let tree_classes = tree.classes();
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
        for tree in &self.trees {
            for c in tree.classes() {
                if !classes.iter().any(|&gc| (gc - c).abs() < eps) {
                    classes.push(c);
                }
            }
        }
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        classes
    }
}

/// Build a sub-matrix selecting specific rows (all columns) from `x`.
fn build_sub_matrix_rows<F: Float>(x: &Array2<F>, row_indices: &[usize]) -> Array2<F> {
    let n_rows = row_indices.len();
    let n_cols = x.ncols();
    let mut data = Vec::with_capacity(n_rows * n_cols);
    for &ri in row_indices {
        for ci in 0..n_cols {
            data.push(x[[ri, ci]]);
        }
    }
    Array2::from_shape_vec((n_rows, n_cols), data).expect("shape matches data length")
}

/// Return the class that appears most frequently in `votes`.
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

        let bc = BaggingClassifier::new(20)
            .with_max_depth(Some(3))
            .with_seed(42);
        let fitted: FittedBaggingClassifier<f64> = bc.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(*p, *t, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_reproducibility() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let bc = BaggingClassifier::new(10).with_seed(123);

        let fitted1: FittedBaggingClassifier<f64> = bc.fit(&x, &y).unwrap();
        let fitted2: FittedBaggingClassifier<f64> = bc.fit(&x, &y).unwrap();

        let preds1 = fitted1.predict(&x).unwrap();
        let preds2 = fitted2.predict(&x).unwrap();

        for (a, b) in preds1.iter().zip(preds2.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-15);
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

        let bc = BaggingClassifier::new(20).with_seed(7);
        let fitted: FittedBaggingClassifier<f64> = bc.fit(&x, &y).unwrap();

        let importances = fitted.feature_importances();
        let sum: f64 = importances.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_predict_proba_rows_sum_to_one() {
        let x = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [10.0, 1.0],
            [11.0, 1.0],
            [12.0, 1.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let bc = BaggingClassifier::new(20)
            .with_max_depth(Some(3))
            .with_seed(42);
        let fitted: FittedBaggingClassifier<f64> = bc.fit(&x, &y).unwrap();

        let proba = fitted.predict_proba(&x).unwrap();
        assert_eq!(proba.nrows(), x.nrows());
        for i in 0..proba.nrows() {
            let row_sum: f64 = proba.row(i).iter().sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_score() {
        let x = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [10.0, 1.0],
            [11.0, 1.0],
            [12.0, 1.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let bc = BaggingClassifier::new(20)
            .with_max_depth(Some(3))
            .with_seed(42);
        let fitted: FittedBaggingClassifier<f64> = bc.fit(&x, &y).unwrap();

        let acc = fitted.score(&x, &y).unwrap();
        assert_abs_diff_eq!(acc, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_n_estimators() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let bc = BaggingClassifier::new(7).with_seed(0);
        let fitted: FittedBaggingClassifier<f64> = bc.fit(&x, &y).unwrap();
        assert_eq!(fitted.n_estimators(), 7);
    }

    #[test]
    fn test_shape_mismatch_error() {
        let x = array![[1.0], [2.0]];
        let y = array![0.0, 1.0, 2.0];

        let bc = BaggingClassifier::default();
        let result: std::result::Result<FittedBaggingClassifier<f64>, _> = bc.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_predict_wrong_features_error() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0];

        let bc = BaggingClassifier::new(5).with_seed(0);
        let fitted: FittedBaggingClassifier<f64> = bc.fit(&x, &y).unwrap();

        let x_bad = array![[1.0], [2.0]];
        let result = fitted.predict(&x_bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_input_error() {
        let x: Array2<f64> = Array2::zeros((0, 2));
        let y: Array1<f64> = Array1::zeros(0);

        let bc = BaggingClassifier::default();
        let result: std::result::Result<FittedBaggingClassifier<f64>, _> = bc.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_estimators_error() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0];

        let bc = BaggingClassifier::new(0);
        let result: std::result::Result<FittedBaggingClassifier<f64>, _> = bc.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_multiclass() {
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

        let bc = BaggingClassifier::new(30)
            .with_max_depth(Some(5))
            .with_seed(42);
        let fitted: FittedBaggingClassifier<f64> = bc.fit(&x, &y).unwrap();

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
    fn test_max_samples() {
        let x = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [10.0, 1.0],
            [11.0, 1.0],
            [12.0, 1.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let bc = BaggingClassifier::new(30)
            .with_max_depth(Some(3))
            .with_max_samples(Some(0.5))
            .with_seed(42);
        let fitted: FittedBaggingClassifier<f64> = bc.fit(&x, &y).unwrap();

        // Should still produce valid predictions
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), y.len());
    }

    #[test]
    fn test_default() {
        let bc = BaggingClassifier::default();
        assert_eq!(bc.n_estimators, 10);
        assert!(bc.bootstrap);
        assert!(bc.max_depth.is_none());
        assert!(bc.max_samples.is_none());
        assert_eq!(bc.seed, 0);
    }
}
