use anofox_ml_core::{Fit, FitWeighted, Float, Predict, PredictProba, Result, RustMlError};
use anofox_ml_trees::{
    ClassWeight, DecisionTreeClassifier, FittedDecisionTreeClassifier, MaxFeatures, SplitCriterion,
};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::forest_common::{
    argmax_first, class_index, fit_tree_on_rows, resolve_max_features, select_rows, sorted_classes,
    tree_input,
};

/// Random forest classifier parameters (unfitted state).
///
/// Trains an ensemble of decision tree classifiers, each on a bootstrap sample
/// of the data. At **every split** each tree considers a fresh random subset
/// of `max_features` features (sklearn semantics); the default is
/// `MaxFeatures::Sqrt`, like `sklearn.ensemble.RandomForestClassifier`.
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
    /// Number of features to consider at each split. `None` = all features
    /// (plain bagged trees). Default: `Some(MaxFeatures::Sqrt)`.
    pub max_features: Option<MaxFeatures>,
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
            max_features: Some(MaxFeatures::Sqrt),
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
    /// Number of features considered per split (`None` = all features).
    pub fn with_max_features(mut self, max_features: Option<MaxFeatures>) -> Self {
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

/// A single tree in the forest together with the feature indices it reads.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
struct ForestTree<F: Float> {
    tree: FittedDecisionTreeClassifier<F>,
    /// Indices of the input features this tree was trained on. Trees are
    /// trained on all features (feature subsampling happens per split inside
    /// the tree), so this is `0..n_features`; models serialized by older
    /// versions may hold a per-tree subset.
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
        self.fit_impl(x, y, None)
    }
}

impl<F: Float> FitWeighted<F> for RandomForestClassifier {
    type Fitted = FittedRandomForestClassifier<F>;

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

impl RandomForestClassifier {
    fn fit_impl<F: Float>(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        sample_weight: Option<&Array1<F>>,
    ) -> Result<FittedRandomForestClassifier<F>> {
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

        // sklearn: class weights come from the full y (not each bootstrap
        // draw) and multiply the sample weights.
        let effective_weight =
            anofox_ml_core::effective_sample_weight(y, sample_weight, self.class_weight.as_ref())?;
        let n_samples = x.nrows();
        let n_features = x.ncols();
        resolve_max_features(self.max_features, n_features)?;

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
            max_features: self.max_features,
            sample_weight: None,
            class_weight: None,
            seed: 0,
        };

        // Pre-generate bootstrap rows and per-tree seeds sequentially so the
        // result does not depend on rayon's scheduling.
        let sample_plans: Vec<(Vec<usize>, u64)> = (0..self.n_estimators)
            .map(|_| {
                let row_indices: Vec<usize> = if self.bootstrap {
                    (0..draw_size)
                        .map(|_| rng.gen_range(0..n_samples))
                        .collect()
                } else {
                    (0..n_samples).collect()
                };
                let tree_seed: u64 = rng.gen();
                (row_indices, tree_seed)
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
            .map(|(row_indices, tree_seed)| {
                let fitted_tree: FittedDecisionTreeClassifier<F> = fit_tree_on_rows(
                    &tree_params.clone().with_seed(tree_seed),
                    x,
                    y,
                    &row_indices,
                    effective_weight.as_ref(),
                )?;
                Ok(ForestTree {
                    tree: fitted_tree,
                    feature_indices: (0..n_features).collect(),
                })
            })
            .collect();
        let trees = trees?;

        let mut fitted = FittedRandomForestClassifier {
            trees,
            n_features,
            oob_score_value: None,
        };

        // Compute OOB score if requested
        if self.oob_score && self.bootstrap {
            fitted.oob_score_value =
                compute_oob_score_classification(&fitted, x, y, &all_row_indices)?;
        }

        Ok(fitted)
    }
}

impl<F: Float> Predict<F> for FittedRandomForestClassifier<F> {
    /// Predict the class with the highest mean leaf probability across trees
    /// (sklearn semantics). Ties go to the smallest class label.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        let proba = self.predict_proba(x)?;
        let classes = self.classes();
        Ok(Array1::from_iter(
            proba
                .rows()
                .into_iter()
                .map(|row| classes[argmax_first(row.iter().copied())]),
        ))
    }
}

impl<F: Float> FittedRandomForestClassifier<F> {
    /// Feature importances averaged across all trees and normalized to sum to 1.
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

    /// Add `weight * tree_proba` of tree `t` into `acc` (columns = `classes`).
    fn accumulate_tree_proba(
        &self,
        t: usize,
        x: &Array2<F>,
        classes: &[F],
        weight: F,
        acc: &mut Array2<F>,
    ) -> Result<()> {
        let forest_tree = &self.trees[t];
        let sub_x = tree_input(x, &forest_tree.feature_indices);
        let tree_proba = forest_tree.tree.predict_proba(&sub_x)?;
        for (local_ci, tc) in forest_tree.tree.classes().into_iter().enumerate() {
            if let Some(global_ci) = class_index(classes, tc) {
                let mut col = acc.column_mut(global_ci);
                col.scaled_add(weight, &tree_proba.column(local_ci));
            }
        }
        Ok(())
    }

    /// Predict class probabilities for each sample.
    ///
    /// Returns an `Array2<F>` of shape `(n_samples, n_classes)` where each row
    /// sums to 1.0. Probabilities are averaged across all trees in the forest.
    /// Columns follow [`classes`](Self::classes) (sorted ascending).
    pub fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }

        let classes = self.classes();
        let n_samples = x.nrows();
        let n_classes = classes.len();
        let w = F::one() / F::from_usize(self.trees.len()).unwrap();

        // Per-tree probabilities in parallel, reduced in tree order so the
        // floating-point sum is deterministic.
        let per_tree: Result<Vec<Array2<F>>> = (0..self.trees.len())
            .into_par_iter()
            .map(|t| {
                let mut acc = Array2::<F>::zeros((n_samples, n_classes));
                self.accumulate_tree_proba(t, x, &classes, w, &mut acc)?;
                Ok(acc)
            })
            .collect();
        let mut avg_proba = Array2::<F>::zeros((n_samples, n_classes));
        for p in per_tree? {
            avg_proba += &p;
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
        let mut all: Vec<F> = Vec::new();
        for forest_tree in &self.trees {
            all.extend(forest_tree.tree.classes());
        }
        sorted_classes(&Array1::from_vec(all))
    }
}

/// Compute OOB classification accuracy (sklearn-style): each sample's OOB
/// prediction is the argmax of the probabilities averaged over the trees that
/// did *not* see it in their bootstrap sample.
fn compute_oob_score_classification<F: Float>(
    forest: &FittedRandomForestClassifier<F>,
    x: &Array2<F>,
    y: &Array1<F>,
    bootstrap_indices: &[Vec<usize>],
) -> Result<Option<f64>> {
    let n_samples = x.nrows();
    let classes = forest.classes();
    let mut oob_proba = Array2::<F>::zeros((n_samples, classes.len()));
    let mut oob_counts = vec![0usize; n_samples];

    for (t, in_bag_rows) in bootstrap_indices.iter().enumerate() {
        let mut in_bag = vec![false; n_samples];
        for &r in in_bag_rows {
            in_bag[r] = true;
        }
        let oob_rows: Vec<usize> = (0..n_samples).filter(|&i| !in_bag[i]).collect();
        if oob_rows.is_empty() {
            continue;
        }
        let x_oob = select_rows(x, &oob_rows);
        let mut acc = Array2::<F>::zeros((oob_rows.len(), classes.len()));
        forest.accumulate_tree_proba(t, &x_oob, &classes, F::one(), &mut acc)?;
        for (k, &i) in oob_rows.iter().enumerate() {
            let mut row = oob_proba.row_mut(i);
            row += &acc.row(k);
            oob_counts[i] += 1;
        }
    }

    let mut correct = 0usize;
    let mut evaluated = 0usize;
    for i in 0..n_samples {
        if oob_counts[i] == 0 {
            continue;
        }
        let pred = classes[argmax_first(oob_proba.row(i).iter().copied())];
        if pred == y[i] {
            correct += 1;
        }
        evaluated += 1;
    }

    Ok(if evaluated > 0 {
        Some(correct as f64 / evaluated as f64)
    } else {
        None
    })
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
            max_features: Some(MaxFeatures::Fixed(2)),
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
            max_features: Some(MaxFeatures::Fixed(5)),
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
        let valid_labels: std::collections::HashSet<u64> = y.iter().map(|v| v.to_bits()).collect();
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

impl<F: Float> PredictProba<F> for FittedRandomForestClassifier<F> {
    fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>> {
        Self::predict_proba(self, x)
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;
    use crate::{
        BaggingClassifier, ExtraTreesClassifier, ExtraTreesRegressor, RandomForestRegressor,
        VotingClassifier,
    };
    use anofox_ml_trees::TreeNode;
    use ndarray::{array, ShapeBuilder};

    fn data(n: usize, d: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
        let mut s = seed;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let x = Array2::from_shape_fn((n, d), |_| next());
        let y = Array1::from_iter(x.rows().into_iter().map(|r| {
            if r.iter().take(4).sum::<f64>() > 2.0 {
                1.0
            } else {
                0.0
            }
        }));
        (x, y)
    }

    fn to_fortran(x: &Array2<f64>) -> Array2<f64> {
        let mut f = Array2::<f64>::zeros(x.raw_dim().f());
        f.assign(x);
        assert!(!f.is_standard_layout());
        f
    }

    fn split_features(node: &TreeNode<f64>, out: &mut Vec<usize>) {
        if let TreeNode::Split {
            feature_index,
            left,
            right,
            ..
        } = node
        {
            out.push(*feature_index);
            split_features(left, out);
            split_features(right, out);
        }
    }

    /// #45: the default classifier forest samples sqrt(d) features at every
    /// split (not once per tree), and trees get distinct seeds.
    #[test]
    fn rf_samples_features_per_split_and_trees_differ() {
        let (x, y) = data(400, 16, 3);
        let rf = RandomForestClassifier::new(5).with_seed(11);
        assert_eq!(rf.max_features, Some(MaxFeatures::Sqrt));
        let fitted: FittedRandomForestClassifier<f64> = rf.fit(&x, &y).unwrap();

        let mut structures = Vec::new();
        for t in &fitted.trees {
            assert_eq!(t.feature_indices, (0..16).collect::<Vec<_>>());
            let mut feats = Vec::new();
            split_features(t.tree.tree(), &mut feats);
            let mut distinct = feats.clone();
            distinct.sort_unstable();
            distinct.dedup();
            assert!(
                distinct.len() > 4,
                "per-split sampling of 4/16 features should reach more than 4 features, got {distinct:?}"
            );
            structures.push(feats);
        }
        for i in 0..structures.len() {
            for j in i + 1..structures.len() {
                assert_ne!(structures[i], structures[j], "trees {i} and {j} identical");
            }
        }

        // Without bootstrap, trees only differ through the per-split RNG.
        let rf_nb = RandomForestClassifier::new(2)
            .with_bootstrap(false)
            .with_seed(5);
        let f_nb: FittedRandomForestClassifier<f64> = rf_nb.fit(&x, &y).unwrap();
        let mut a = Vec::new();
        let mut b = Vec::new();
        split_features(f_nb.trees[0].tree.tree(), &mut a);
        split_features(f_nb.trees[1].tree.tree(), &mut b);
        assert_ne!(a, b, "two trees with different seeds must differ");
    }

    #[test]
    fn rf_is_reproducible_under_rayon() {
        let (x, y) = data(200, 9, 4);
        let rf = RandomForestClassifier::new(16).with_seed(3);
        let a: FittedRandomForestClassifier<f64> = rf.fit(&x, &y).unwrap();
        for _ in 0..3 {
            let b: FittedRandomForestClassifier<f64> = rf.fit(&x, &y).unwrap();
            assert_eq!(a.predict_proba(&x).unwrap(), b.predict_proba(&x).unwrap());
        }
        let r = RandomForestRegressor::new(16)
            .with_max_features(Some(MaxFeatures::Sqrt))
            .with_seed(3);
        let ra = r.fit(&x, &y).unwrap();
        let rb = r.fit(&x, &y).unwrap();
        assert_eq!(ra.predict(&x).unwrap(), rb.predict(&x).unwrap());
    }

    /// #42: Fortran-order input used to panic.
    #[test]
    fn forests_accept_fortran_order_input() {
        let (x, y) = data(120, 5, 8);
        let xf = to_fortran(&x);

        let rf = RandomForestClassifier::new(8)
            .with_seed(1)
            .with_oob_score(true);
        let a: FittedRandomForestClassifier<f64> = rf.fit(&x, &y).unwrap();
        let b: FittedRandomForestClassifier<f64> = rf.fit(&xf, &y).unwrap();
        assert_eq!(a.predict(&x).unwrap(), b.predict(&xf).unwrap());
        assert_eq!(a.oob_score(), b.oob_score());

        let rr = RandomForestRegressor::new(8).with_seed(1);
        let ra = rr.fit(&x, &y).unwrap();
        let rb = rr.fit(&xf, &y).unwrap();
        assert_eq!(ra.predict(&x).unwrap(), rb.predict(&xf).unwrap());

        let et = ExtraTreesClassifier::new(8).with_seed(1);
        let ea = et.fit(&x, &y).unwrap();
        let eb = et.fit(&xf, &y).unwrap();
        assert_eq!(ea.predict(&x).unwrap(), eb.predict(&xf).unwrap());

        let etr = ExtraTreesRegressor::new(8).with_seed(1);
        let era = etr.fit(&x, &y).unwrap();
        let erb = etr.fit(&xf, &y).unwrap();
        let x_tt = x.t().to_owned();
        let x_tt = x_tt.t().to_owned();
        assert_eq!(era.predict(&x).unwrap(), erb.predict(&x_tt).unwrap());

        let bag = BaggingClassifier::new(5).with_seed(1);
        let ba = bag.fit(&x, &y).unwrap();
        let bb = bag.fit(&xf, &y).unwrap();
        assert_eq!(ba.predict(&x).unwrap(), bb.predict(&xf).unwrap());
    }

    /// #42: tied votes must be resolved deterministically (smallest label).
    #[test]
    fn tied_votes_are_deterministic() {
        // Identical rows with balanced labels -> every leaf is a 50/50 tie.
        let x = array![[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]];
        let y = array![9.0, 4.0, 9.0, 4.0];
        for _ in 0..20 {
            let rf = RandomForestClassifier::new(6)
                .with_bootstrap(false)
                .with_seed(0);
            let f: FittedRandomForestClassifier<f64> = rf.fit(&x, &y).unwrap();
            assert_eq!(f.predict(&x).unwrap(), array![4.0, 4.0, 4.0, 4.0]);

            let et = ExtraTreesClassifier::new(6).with_seed(0);
            assert_eq!(
                et.fit(&x, &y).unwrap().predict(&x).unwrap(),
                array![4.0, 4.0, 4.0, 4.0]
            );

            let bag = BaggingClassifier::new(4).with_bootstrap(false);
            assert_eq!(
                bag.fit(&x, &y).unwrap().predict(&x).unwrap(),
                array![4.0, 4.0, 4.0, 4.0]
            );
        }

        // Voting: two members that disagree -> 1:1 tie -> smallest label.
        let xs = array![[0.0], [1.0], [2.0], [3.0]];
        let ys = array![0.0, 0.0, 1.0, 1.0];
        for _ in 0..20 {
            let vc = VotingClassifier::new()
                .push("tree", anofox_ml_trees::DecisionTreeClassifier::new())
                .push("const", ConstClassifier(5.0));
            let f = vc.fit(&xs, &ys).unwrap();
            let p = f.predict(&xs).unwrap();
            assert_eq!(p, array![0.0, 0.0, 1.0, 1.0]);
        }
    }

    /// Always predicts a fixed label.
    struct ConstClassifier(f64);
    struct FittedConst(f64);
    impl Fit<f64> for ConstClassifier {
        type Fitted = FittedConst;
        fn fit(&self, _x: &Array2<f64>, _y: &Array1<f64>) -> Result<FittedConst> {
            Ok(FittedConst(self.0))
        }
    }
    impl Predict<f64> for FittedConst {
        fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
            Ok(Array1::from_elem(x.nrows(), self.0))
        }
    }

    /// #45: ExtraTrees also draws a fresh feature subset per split.
    #[test]
    fn extra_trees_sample_features_per_split() {
        let (x, y) = data(300, 16, 21);
        let et = ExtraTreesClassifier::new(3).with_seed(2);
        assert_eq!(et.max_features, Some(MaxFeatures::Sqrt));
        let fitted = et.fit(&x, &y).unwrap();
        let imp = fitted.feature_importances();
        let used = imp.iter().filter(|&&v| v > 0.0).count();
        assert!(used > 4, "only {used} features used across the forest");
    }

    #[test]
    fn invalid_max_features_fraction_is_rejected() {
        let (x, y) = data(20, 4, 1);
        let rf = RandomForestClassifier::new(2).with_max_features(Some(MaxFeatures::Fraction(1.5)));
        assert!(Fit::<f64>::fit(&rf, &x, &y).is_err());
    }
}
