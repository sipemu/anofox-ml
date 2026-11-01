use anofox_ml_core::{Fit, FitWeighted, Float, Predict, Result, RustMlError};
use anofox_ml_trees::node::TreeNode;
use anofox_ml_trees::split::SplitCriterion;
use anofox_ml_trees::MaxFeatures;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::forest_common::{
    argmax_first, build_extra_tree, class_index, resolve_max_features, sorted_classes, tree_input,
    ExtraTreeParams,
};

/// Extra-Trees (Extremely Randomized Trees) classifier parameters (unfitted state).
///
/// Trains an ensemble of decision trees using random split thresholds instead of
/// the best possible split at each node. Unlike Random Forests, Extra-Trees does
/// **not** use bootstrap sampling — each tree is trained on the full dataset.
/// At every split a fresh random subset of `max_features` features is drawn
/// (default `MaxFeatures::Sqrt`, like `sklearn.ensemble.ExtraTreesClassifier`).
///
/// The randomization in split thresholds reduces variance further than Random
/// Forests and can lead to smoother decision boundaries.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExtraTreesClassifier {
    /// Number of trees in the forest.
    pub n_estimators: usize,
    /// Maximum depth of each tree.
    pub max_depth: Option<usize>,
    /// Minimum samples required to split a node.
    pub min_samples_split: usize,
    /// Minimum samples required in a leaf node.
    pub min_samples_leaf: usize,
    /// Number of features to consider at each split. `None` = all features.
    /// Default: `Some(MaxFeatures::Sqrt)`.
    pub max_features: Option<MaxFeatures>,
    /// Split criterion: `Gini` (default) or `Entropy`.
    #[serde(default = "default_criterion")]
    pub criterion: SplitCriterion,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl ExtraTreesClassifier {
    /// Create a new `ExtraTreesClassifier` with the given number of trees and default parameters.
    pub fn new(n_estimators: usize) -> Self {
        Self {
            n_estimators,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: Some(MaxFeatures::Sqrt),
            criterion: SplitCriterion::Gini,
            seed: 0,
        }
    }

    /// Set the split criterion (`Gini` or `Entropy`).
    pub fn with_criterion(mut self, criterion: SplitCriterion) -> Self {
        self.criterion = criterion;
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

impl Default for ExtraTreesClassifier {
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

/// Fitted Extra-Trees classifier.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedExtraTreesClassifier<F: Float> {
    trees: Vec<ExtraForestTree<F>>,
    n_features: usize,
    /// Sorted training class labels.
    #[serde(default)]
    classes: Vec<F>,
}

impl<F: Float> Fit<F> for ExtraTreesClassifier {
    type Fitted = FittedExtraTreesClassifier<F>;

    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Self::Fitted> {
        self.fit_impl(x, y, None)
    }
}

impl<F: Float> FitWeighted<F> for ExtraTreesClassifier {
    type Fitted = FittedExtraTreesClassifier<F>;

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

impl ExtraTreesClassifier {
    fn fit_impl<F: Float>(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        sample_weight: Option<&Array1<F>>,
    ) -> Result<FittedExtraTreesClassifier<F>> {
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
            criterion: self.criterion,
            max_features_k,
            n_features,
        };
        let indices: Vec<usize> = (0..x.nrows()).collect();

        // Train trees in parallel
        let trees: Vec<ExtraForestTree<F>> = tree_seeds
            .into_par_iter()
            .map(|tree_seed| ExtraForestTree {
                tree: build_extra_tree(x, y, &indices, 0, &params, tree_seed, sample_weight),
                feature_indices: (0..n_features).collect(),
                n_features_tree: n_features,
            })
            .collect();

        Ok(FittedExtraTreesClassifier {
            trees,
            n_features,
            classes: sorted_classes(y),
        })
    }
}

impl<F: Float> Predict<F> for FittedExtraTreesClassifier<F> {
    /// Predict the class with the highest mean leaf probability across trees
    /// (sklearn semantics). Ties go to the smallest class label.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        let proba = self.proba_matrix(x)?;
        let classes = self.classes();
        Ok(Array1::from_iter(
            proba
                .rows()
                .into_iter()
                .map(|row| classes[argmax_first(row.iter().copied())]),
        ))
    }
}

fn default_criterion() -> SplitCriterion {
    SplitCriterion::Gini
}

impl<F: Float> anofox_ml_core::PredictProba<F> for FittedExtraTreesClassifier<F> {
    fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>> {
        self.proba_matrix(x)
    }
}

impl<F: Float> FittedExtraTreesClassifier<F> {
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

    /// Sorted class labels seen during training.
    pub fn classes(&self) -> Vec<F> {
        if !self.classes.is_empty() {
            return self.classes.clone();
        }
        // Models serialized before `classes` was stored: recover from leaves.
        let mut all = Vec::new();
        for t in &self.trees {
            collect_leaf_classes(&t.tree, &mut all);
        }
        sorted_classes(&Array1::from_vec(all))
    }

    /// Mean leaf class-probabilities, shape `(n_samples, n_classes)`.
    fn proba_matrix(&self, x: &Array2<F>) -> Result<Array2<F>> {
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

        let per_tree: Vec<Array2<F>> = self
            .trees
            .par_iter()
            .map(|forest_tree| {
                let sub_x = tree_input(x, &forest_tree.feature_indices);
                let mut acc = Array2::<F>::zeros((n_samples, n_classes));
                for (i, row) in sub_x.rows().into_iter().enumerate() {
                    let leaf = forest_tree.tree.leaf_for_view(row);
                    match (leaf.class_distribution(), leaf) {
                        (Some(dist), _) => {
                            let total = dist.iter().fold(F::zero(), |a, &(_, m)| a + m);
                            if total > F::zero() {
                                for &(label, m) in &dist {
                                    if let Some(ci) = class_index(&classes, label) {
                                        acc[[i, ci]] += w * m / total;
                                    }
                                }
                            }
                        }
                        (None, TreeNode::Leaf { value, .. }) => {
                            if let Some(ci) = class_index(&classes, *value) {
                                acc[[i, ci]] += w;
                            }
                        }
                        (None, TreeNode::Split { .. }) => unreachable!(),
                    }
                }
                acc
            })
            .collect();

        // Reduce in tree order for a deterministic floating-point sum.
        let mut proba = Array2::<F>::zeros((n_samples, n_classes));
        for p in &per_tree {
            proba += p;
        }
        Ok(proba)
    }

    /// Class probabilities (mean leaf probabilities over the trees), shape
    /// `(n_samples, n_classes)`; columns follow [`classes`](Self::classes).
    pub fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>> {
        self.proba_matrix(x)
    }

    /// Number of trees in the ensemble.
    pub fn n_estimators(&self) -> usize {
        self.trees.len()
    }
}

fn collect_leaf_classes<F: Float>(node: &TreeNode<F>, out: &mut Vec<F>) {
    match node {
        TreeNode::Leaf {
            class_counts: Some(counts),
            ..
        } => out.extend(counts.iter().map(|&(c, _)| c)),
        TreeNode::Leaf { value, .. } => out.push(*value),
        TreeNode::Split { left, right, .. } => {
            collect_leaf_classes(left, out);
            collect_leaf_classes(right, out);
        }
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

        let et = ExtraTreesClassifier {
            n_estimators: 20,
            max_depth: Some(3),
            seed: 42,
            ..Default::default()
        };
        let fitted: FittedExtraTreesClassifier<f64> = et.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(*p, *t, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_reproducibility() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let et = ExtraTreesClassifier {
            n_estimators: 10,
            seed: 123,
            ..Default::default()
        };

        let fitted1: FittedExtraTreesClassifier<f64> = et.fit(&x, &y).unwrap();
        let fitted2: FittedExtraTreesClassifier<f64> = et.fit(&x, &y).unwrap();

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

        let et = ExtraTreesClassifier {
            n_estimators: 30,
            max_features: Some(MaxFeatures::Fixed(2)),
            seed: 99,
            ..Default::default()
        };
        let fitted: FittedExtraTreesClassifier<f64> = et.fit(&x, &y).unwrap();

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

        let et = ExtraTreesClassifier {
            n_estimators: 20,
            seed: 7,
            ..Default::default()
        };
        let fitted: FittedExtraTreesClassifier<f64> = et.fit(&x, &y).unwrap();

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
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let et = ExtraTreesClassifier {
            n_estimators: 20,
            seed: 7,
            ..Default::default()
        };
        let fitted: FittedExtraTreesClassifier<f64> = et.fit(&x, &y).unwrap();

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
        let y = array![0.0, 0.0, 1.0, 1.0];

        let et = ExtraTreesClassifier {
            n_estimators: 7,
            seed: 0,
            ..Default::default()
        };
        let fitted: FittedExtraTreesClassifier<f64> = et.fit(&x, &y).unwrap();
        assert_eq!(fitted.n_estimators(), 7);
    }

    #[test]
    fn test_shape_mismatch_error() {
        let x = array![[1.0], [2.0]];
        let y = array![0.0, 1.0, 2.0];

        let et = ExtraTreesClassifier::default();
        let result: std::result::Result<FittedExtraTreesClassifier<f64>, _> = et.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_predict_wrong_features_error() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0];

        let et = ExtraTreesClassifier {
            n_estimators: 5,
            seed: 0,
            ..Default::default()
        };
        let fitted: FittedExtraTreesClassifier<f64> = et.fit(&x, &y).unwrap();

        let x_bad = array![[1.0], [2.0]];
        let result = fitted.predict(&x_bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_max_features() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0];

        let et = ExtraTreesClassifier {
            n_estimators: 5,
            max_features: Some(MaxFeatures::Fixed(5)),
            seed: 0,
            ..Default::default()
        };
        let result: std::result::Result<FittedExtraTreesClassifier<f64>, _> = et.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_estimators_error() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0];

        let et = ExtraTreesClassifier {
            n_estimators: 0,
            seed: 0,
            ..Default::default()
        };
        let result: std::result::Result<FittedExtraTreesClassifier<f64>, _> = et.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_input_error() {
        let x: Array2<f64> = Array2::zeros((0, 2));
        let y: Array1<f64> = Array1::zeros(0);

        let et = ExtraTreesClassifier::default();
        let result: std::result::Result<FittedExtraTreesClassifier<f64>, _> = et.fit(&x, &y);
        assert!(result.is_err());
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

        let et = ExtraTreesClassifier {
            n_estimators: 1,
            max_depth: Some(3),
            seed: 42,
            ..Default::default()
        };
        let fitted: FittedExtraTreesClassifier<f64> = et.fit(&x, &y).unwrap();
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

        let et = ExtraTreesClassifier {
            n_estimators: 30,
            max_depth: Some(5),
            seed: 42,
            ..Default::default()
        };
        let fitted: FittedExtraTreesClassifier<f64> = et.fit(&x, &y).unwrap();

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
    fn test_predict_proba() {
        let x = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [10.0, 1.0],
            [11.0, 1.0],
            [12.0, 1.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let et = ExtraTreesClassifier {
            n_estimators: 20,
            max_depth: Some(3),
            seed: 42,
            ..Default::default()
        };
        let fitted: FittedExtraTreesClassifier<f64> = et.fit(&x, &y).unwrap();

        let proba = fitted.predict_proba(&x).unwrap();
        assert_eq!(proba.dim(), (x.nrows(), 2));

        // Each sample's probabilities should sum to 1
        for sample_probs in proba.rows() {
            assert_abs_diff_eq!(sample_probs.sum(), 1.0, epsilon = 1e-10);
        }

        // For clearly separable data, class 0 samples should have high P(class=0)
        for i in 0..3 {
            let p_class0 = proba[[i, 0]];
            assert!(p_class0 > 0.5, "expected P(class=0) > 0.5, got {p_class0}");
        }
    }

    #[test]
    fn test_predict_proba_wrong_features_error() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0];

        let et = ExtraTreesClassifier {
            n_estimators: 5,
            seed: 0,
            ..Default::default()
        };
        let fitted: FittedExtraTreesClassifier<f64> = et.fit(&x, &y).unwrap();

        let x_bad = array![[1.0], [2.0]];
        let result = fitted.predict_proba(&x_bad);
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

                let et = ExtraTreesClassifier {
                    n_estimators: 10,
                    max_depth: Some(5),
                    seed: seed as u64,
                    ..Default::default()
                };
                let fitted: FittedExtraTreesClassifier<f64> = et.fit(&x, &y).unwrap();
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

                let et = ExtraTreesClassifier {
                    n_estimators: 10,
                    max_depth: Some(5),
                    seed: seed as u64,
                    ..Default::default()
                };
                let fitted: FittedExtraTreesClassifier<f64> = et.fit(&x, &y).unwrap();
                let importances = fitted.feature_importances();
                let sum: f64 = importances.iter().sum();

                // Valid outcomes: (1) importances form a valid probability
                // distribution (sum to 1), or (2) every tree in the forest
                // is a pure leaf with no splits and all importances are zero.
                prop_assert!(
                    (sum - 1.0).abs() < 1e-10 || sum == 0.0,
                    "feature importances sum to {} (expected ~1.0 or 0.0 for no-split case), n_samples={}, n_features={}, seed={}",
                    sum, n_samples, n_features, seed
                );
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
