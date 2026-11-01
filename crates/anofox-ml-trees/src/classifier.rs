use anofox_ml_core::{Fit, FitWeighted, Float, Predict, PredictProba, Result, RustMlError};
use ndarray::{Array1, Array2};

use crate::engine::{grow, GrowParams};
use crate::node::TreeNode;
use crate::split::{cmp_nan_last, ClassWeight, MaxFeatures, SplitCriterion};

/// Decision tree classifier parameters (unfitted state).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DecisionTreeClassifier {
    pub max_depth: Option<usize>,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub criterion: SplitCriterion,
    /// Maximum number of features to consider at each split.
    pub max_features: Option<MaxFeatures>,
    /// Per-sample weights.
    #[serde(skip)]
    pub sample_weight: Option<Array1<f64>>,
    /// Class weighting strategy.
    pub class_weight: Option<ClassWeight>,
    /// Seed for the per-split feature subsampling RNG (only used when
    /// `max_features` is set). Different seeds give different trees.
    #[serde(default)]
    pub seed: u64,
}

impl DecisionTreeClassifier {
    /// Create a new `DecisionTreeClassifier` with sensible defaults.
    pub fn new() -> Self {
        Self {
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            criterion: SplitCriterion::Gini,
            max_features: None,
            sample_weight: None,
            class_weight: None,
            seed: 0,
        }
    }

    /// Set the maximum depth of the tree.
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

    /// Set the split quality criterion.
    pub fn with_criterion(mut self, criterion: SplitCriterion) -> Self {
        self.criterion = criterion;
        self
    }

    /// Set the maximum number of features to consider at each split.
    pub fn with_max_features(mut self, max_features: Option<MaxFeatures>) -> Self {
        self.max_features = max_features;
        self
    }

    /// Set per-sample weights.
    pub fn with_sample_weight(mut self, sample_weight: Option<Array1<f64>>) -> Self {
        self.sample_weight = sample_weight;
        self
    }

    /// Set class weighting strategy.
    pub fn with_class_weight(mut self, class_weight: Option<ClassWeight>) -> Self {
        self.class_weight = class_weight;
        self
    }

    /// Set the seed of the per-split feature subsampling RNG.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl Default for DecisionTreeClassifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted decision tree classifier.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedDecisionTreeClassifier<F: Float> {
    tree: TreeNode<F>,
    n_features: usize,
}

impl<F: Float> Fit<F> for DecisionTreeClassifier {
    type Fitted = FittedDecisionTreeClassifier<F>;

    /// Fits with the configured `sample_weight` (if any).
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Self::Fitted> {
        let sw = self
            .sample_weight
            .as_ref()
            .map(|sw| sw.mapv(|v| F::from_f64(v).unwrap()));
        self.fit_impl(x, y, sw.as_ref())
    }
}

impl<F: Float> FitWeighted<F> for DecisionTreeClassifier {
    type Fitted = FittedDecisionTreeClassifier<F>;

    /// `sample_weight` replaces the configured one; it is multiplied by the
    /// `class_weight`, as in sklearn.
    fn fit_weighted(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        sample_weight: Option<&Array1<F>>,
    ) -> Result<Self::Fitted> {
        match sample_weight {
            Some(sw) => self.fit_impl(x, y, Some(sw)),
            None => Fit::fit(self, x, y),
        }
    }
}

impl DecisionTreeClassifier {
    fn fit_impl<F: Float>(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        sample_weight: Option<&Array1<F>>,
    ) -> Result<FittedDecisionTreeClassifier<F>> {
        check_xy(x, y)?;
        let rows: Vec<usize> = (0..x.nrows()).collect();
        self.fit_on_rows(x, y, &rows, sample_weight)
    }

    /// Fit on the sample `rows` of `(x, y)` without copying `X`. Rows may
    /// repeat (a bootstrap draw counts each copy as a sample, exactly as if
    /// the rows had been copied). `sample_weight` is indexed by row of `x`;
    /// `class_weight` is computed over the drawn samples.
    pub fn fit_on_rows<F: Float>(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        rows: &[usize],
        sample_weight: Option<&Array1<F>>,
    ) -> Result<FittedDecisionTreeClassifier<F>> {
        check_xy(x, y)?;
        check_rows(rows, x.nrows())?;
        if let Some(sw) = sample_weight {
            if sw.len() != y.len() {
                return Err(RustMlError::ShapeMismatch(format!(
                    "sample_weight has {} elements but X has {} rows",
                    sw.len(),
                    y.len()
                )));
            }
        }
        // Effective sample weights: sample_weight × class_weight, per row.
        let effective_weights = match &self.class_weight {
            None => {
                if let Some(sw) = sample_weight {
                    let drawn = Array1::from_iter(rows.iter().map(|&r| sw[r]));
                    anofox_ml_core::check_sample_weight(&drawn, drawn.len())?;
                }
                sample_weight.cloned()
            }
            Some(cw) => {
                let y_drawn = Array1::from_iter(rows.iter().map(|&r| y[r]));
                let w_drawn =
                    sample_weight.map(|sw| Array1::from_iter(rows.iter().map(|&r| sw[r])));
                let eff =
                    anofox_ml_core::effective_sample_weight(&y_drawn, w_drawn.as_ref(), Some(cw))?
                        .expect("class_weight is set");
                let mut per_row = Array1::<F>::zeros(y.len());
                for (&r, &w) in rows.iter().zip(eff.iter()) {
                    per_row[r] = w;
                }
                Some(per_row)
            }
        };

        let n_features = x.ncols();
        let params = GrowParams {
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            criterion: self.criterion,
            max_features_k: self.max_features.map(|mf| mf.resolve(n_features)),
            n_features,
        };
        let tree = grow(x, y, rows, effective_weights.as_ref(), &params, self.seed);
        Ok(FittedDecisionTreeClassifier { tree, n_features })
    }
}

/// Matching row counts and non-empty data.
pub(crate) fn check_xy<F: Float>(x: &Array2<F>, y: &Array1<F>) -> Result<()> {
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
    Ok(())
}

/// Non-empty and in range.
pub(crate) fn check_rows(rows: &[usize], n: usize) -> Result<()> {
    if rows.is_empty() {
        return Err(RustMlError::EmptyInput("no sample rows".into()));
    }
    if let Some(&r) = rows.iter().find(|&&r| r >= n) {
        return Err(RustMlError::InvalidParameter(format!(
            "sample row {r} out of range for {n} rows"
        )));
    }
    Ok(())
}

impl<F: Float> Predict<F> for FittedDecisionTreeClassifier<F> {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }

        let predictions: Vec<F> = x
            .rows()
            .into_iter()
            .map(|row| self.tree.predict_view(row))
            .collect();

        Ok(Array1::from_vec(predictions))
    }
}

impl<F: Float> FittedDecisionTreeClassifier<F> {
    /// Feature importances (normalized to sum to 1).
    pub fn feature_importances(&self) -> Array1<F> {
        let n_samples = tree_n_samples(&self.tree);
        let raw = self.tree.feature_importances(self.n_features, n_samples);
        let sum: F = raw.iter().copied().fold(F::zero(), |a, b| a + b);
        if sum > F::zero() {
            Array1::from_vec(raw.into_iter().map(|v| v / sum).collect())
        } else {
            Array1::zeros(self.n_features)
        }
    }

    pub fn tree(&self) -> &TreeNode<F> {
        &self.tree
    }

    /// Predict class probabilities for each sample.
    ///
    /// Returns an `Array2<F>` of shape `(n_samples, n_classes)` where each row
    /// sums to 1.0. Classes are sorted in ascending order.
    pub fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }

        // Collect all unique classes from the tree
        let classes = collect_classes(&self.tree);

        let n_samples = x.nrows();
        let n_classes = classes.len();
        let mut proba = Array2::<F>::zeros((n_samples, n_classes));

        for (i, row) in x.rows().into_iter().enumerate() {
            let leaf = self.tree.leaf_for_view(row);
            if let TreeNode::Leaf {
                class_counts: Some(counts),
                ..
            } = leaf
            {
                let total: usize = counts.iter().map(|&(_, c)| c).sum();
                let total_f = F::from_usize(total).unwrap();
                for &(class_val, count) in counts {
                    if let Some(ci) = classes
                        .iter()
                        .position(|&c| (c - class_val).abs() < F::from_f64(1e-9).unwrap())
                    {
                        proba[[i, ci]] = F::from_usize(count).unwrap() / total_f;
                    }
                }
            } else {
                // Regression leaf or no counts — put all weight on predicted class
                let pred = self.tree.predict_view(row);
                if let Some(ci) = classes
                    .iter()
                    .position(|&c| (c - pred).abs() < F::from_f64(1e-9).unwrap())
                {
                    proba[[i, ci]] = F::one();
                }
            }
        }

        Ok(proba)
    }

    /// Returns the unique sorted class labels learned during fitting.
    pub fn classes(&self) -> Vec<F> {
        collect_classes(&self.tree)
    }

    /// Number of features expected at prediction time.
    pub fn n_features(&self) -> usize {
        self.n_features
    }
}

impl<F: Float> PredictProba<F> for FittedDecisionTreeClassifier<F> {
    fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>> {
        // Delegate to the inherent method.
        Self::predict_proba(self, x)
    }
}

/// Collect all unique sorted class labels from the tree's leaf nodes.
fn collect_classes<F: Float>(node: &TreeNode<F>) -> Vec<F> {
    let mut classes = Vec::new();
    collect_classes_recursive(node, &mut classes);
    classes.sort_by(cmp_nan_last);
    classes.dedup_by(|a, b| (*a - *b).abs() < F::from_f64(1e-9).unwrap());
    classes
}

fn collect_classes_recursive<F: Float>(node: &TreeNode<F>, classes: &mut Vec<F>) {
    match node {
        TreeNode::Leaf {
            class_counts: Some(counts),
            ..
        } => {
            for &(class_val, _) in counts {
                classes.push(class_val);
            }
        }
        TreeNode::Leaf { value, .. } => {
            classes.push(*value);
        }
        TreeNode::Split { left, right, .. } => {
            collect_classes_recursive(left, classes);
            collect_classes_recursive(right, classes);
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
    fn test_simple_classification() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let tree = DecisionTreeClassifier::default();
        let fitted = Fit::fit(&tree, &x, &y).unwrap();

        let preds = fitted.predict(&array![[1.5], [5.5]]).unwrap();
        assert_abs_diff_eq!(preds[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_max_depth() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let tree = DecisionTreeClassifier {
            max_depth: Some(1),
            ..Default::default()
        };
        let fitted = Fit::fit(&tree, &x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        // With max_depth=1, should still separate the two classes
        assert_abs_diff_eq!(preds[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[3], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_feature_importances() {
        let x = array![[1.0, 100.0], [2.0, 200.0], [3.0, 300.0], [4.0, 400.0]];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let tree = DecisionTreeClassifier::default();
        let fitted = Fit::fit(&tree, &x, &y).unwrap();

        let importances = fitted.feature_importances();
        // Sum should be 1.0
        let sum: f64 = importances.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_min_samples_split_constraint() {
        // 4 samples with min_samples_split=5 means the root can never split
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let tree = DecisionTreeClassifier::new().with_min_samples_split(5);
        let fitted = Fit::fit(&tree, &x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        // All predictions should be the same (single leaf)
        let first = preds[0];
        for &p in preds.iter() {
            assert_abs_diff_eq!(p, first, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_min_samples_leaf_constraint() {
        // 4 samples, 2 of each class. min_samples_leaf=2 means leaves need >= 2 samples.
        // A split into [0,0] and [1,1] satisfies this, but min_samples_leaf=3 would not.
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let tree = DecisionTreeClassifier::new().with_min_samples_leaf(3);
        let fitted = Fit::fit(&tree, &x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        // With min_samples_leaf=3 on 4 samples, no valid split exists (each side would
        // have at most 2 samples), so the tree degenerates to a single leaf.
        let first = preds[0];
        for &p in preds.iter() {
            assert_abs_diff_eq!(p, first, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_multiclass_three_classes() {
        // 9 data points, 3 classes separated by feature value
        let x = array![
            [1.0],
            [2.0],
            [3.0], // class 0
            [5.0],
            [6.0],
            [7.0], // class 1
            [9.0],
            [10.0],
            [11.0] // class 2
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];

        let tree = DecisionTreeClassifier::default();
        let fitted = Fit::fit(&tree, &x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        for (pred, target) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(pred, target, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_single_class_input() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![7.0, 7.0, 7.0, 7.0];

        let tree = DecisionTreeClassifier::default();
        let fitted = Fit::fit(&tree, &x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        for &p in preds.iter() {
            assert_abs_diff_eq!(p, 7.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_single_feature() {
        // Simple binary split on one feature
        let x = array![[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let tree = DecisionTreeClassifier::default();
        let fitted = Fit::fit(&tree, &x, &y).unwrap();

        let test_x = array![[0.5], [11.5]];
        let preds = fitted.predict(&test_x).unwrap();
        assert_abs_diff_eq!(preds[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_stump_depth_one() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let tree = DecisionTreeClassifier::new().with_max_depth(Some(1));
        let fitted = Fit::fit(&tree, &x, &y).unwrap();

        // The root should be a Split whose children are both Leaves
        match fitted.tree() {
            TreeNode::Split { left, right, .. } => {
                assert!(matches!(**left, TreeNode::Leaf { .. }));
                assert!(matches!(**right, TreeNode::Leaf { .. }));
            }
            TreeNode::Leaf { .. } => panic!("expected a stump (Split node), got Leaf"),
        }
    }

    #[test]
    fn test_shape_mismatch_error() {
        let x = array![[1.0], [2.0], [3.0]];
        let y = array![0.0, 1.0]; // 3 rows vs 2 labels

        let tree = DecisionTreeClassifier::default();
        let result = Fit::<f64>::fit(&tree, &x, &y);
        assert!(result.is_err());
        match result.unwrap_err() {
            RustMlError::ShapeMismatch(_) => {} // expected
            other => panic!("expected ShapeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn test_empty_input_error() {
        let x: Array2<f64> = Array2::zeros((0, 0));
        let y: Array1<f64> = array![];

        let tree = DecisionTreeClassifier::default();
        let result = Fit::<f64>::fit(&tree, &x, &y);
        assert!(result.is_err());
        match result.unwrap_err() {
            RustMlError::EmptyInput(_) => {} // expected
            other => panic!("expected EmptyInput, got {:?}", other),
        }
    }

    #[test]
    fn test_predict_wrong_features() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let tree = DecisionTreeClassifier::default();
        let fitted = Fit::fit(&tree, &x, &y).unwrap();

        // Predict with 3 features instead of 2
        let bad_x = array![[1.0, 2.0, 3.0]];
        let result = fitted.predict(&bad_x);
        assert!(result.is_err());
        match result.unwrap_err() {
            RustMlError::ShapeMismatch(_) => {} // expected
            other => panic!("expected ShapeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn test_large_feature_values() {
        // Very large feature values should not cause panics or NaN
        let x = array![
            [1e10_f64, -1e10],
            [2e10, -2e10],
            [3e10, -3e10],
            [4e10, -4e10],
        ];
        let y = array![0.0_f64, 0.0, 1.0, 1.0];

        let tree = DecisionTreeClassifier::default();
        let fitted = Fit::fit(&tree, &x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(p.is_finite(), "prediction should be finite, got {}", p);
        }
    }

    #[test]
    fn test_small_feature_values() {
        // Very small feature values should still produce valid splits
        let x = array![[1e-10], [2e-10], [3e-10], [4e-10],];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let tree = DecisionTreeClassifier::default();
        let fitted = Fit::fit(&tree, &x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        // Should separate the two classes
        assert_abs_diff_eq!(preds[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[3], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_near_identical_feature_values() {
        // Features that differ by tiny amounts (near machine epsilon)
        let x = array![[1.0 + 1e-14], [1.0 + 2e-14], [1.0 + 3e-14], [1.0 + 4e-14],];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let tree = DecisionTreeClassifier::default();
        let fitted = Fit::fit(&tree, &x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        // Should not panic; predictions should be valid labels
        for &p in preds.iter() {
            assert!(
                p == 0.0 || p == 1.0,
                "prediction should be 0 or 1, got {}",
                p
            );
        }
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
            fn tree_predictions_are_valid_labels(
                n_samples in 4..30usize,
                n_features in 1..5usize,
                seed in 0u64..1000,
            ) {
                let n_classes = 3;
                let (x, y) = make_classification_data(n_samples, n_features, n_classes, seed);

                // Collect unique training labels
                let train_labels: HashSet<u64> = y.iter()
                    .map(|&v| v.to_bits())
                    .collect();

                let tree = DecisionTreeClassifier::new()
                    .with_max_depth(Some(5));
                let fitted = Fit::fit(&tree, &x, &y).unwrap();
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
            fn tree_deterministic(seed in 0u64..1000) {
                let (x, y) = make_classification_data(20, 3, 3, seed);

                let tree = DecisionTreeClassifier::new()
                    .with_max_depth(Some(4));

                let fitted1 = Fit::fit(&tree, &x, &y).unwrap();
                let fitted2 = Fit::fit(&tree, &x, &y).unwrap();

                let preds1 = fitted1.predict(&x).unwrap();
                let preds2 = fitted2.predict(&x).unwrap();

                for (i, (&a, &b)) in preds1.iter().zip(preds2.iter()).enumerate() {
                    prop_assert!((a - b).abs() < 1e-15,
                        "non-deterministic prediction at index {}: {} vs {}", i, a, b);
                }
            }
        }
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;
    use crate::regressor::DecisionTreeRegressor;
    use crate::split::majority_vote;
    use ndarray::{array, ShapeBuilder};

    fn data(n: usize, d: usize) -> (Array2<f64>, Array1<f64>) {
        let mut s = 99u64;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let x = Array2::from_shape_fn((n, d), |_| next());
        let y = Array1::from_iter(x.rows().into_iter().map(|r| {
            if r.iter().take(3).sum::<f64>() > 1.5 {
                1.0
            } else {
                0.0
            }
        }));
        (x, y)
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

    /// #42: Fortran-order input used to panic in `row.as_slice().unwrap()`.
    #[test]
    fn fortran_order_input_fit_and_predict() {
        let (x, y) = data(60, 4);
        let mut x_f = Array2::<f64>::zeros((60, 4).f());
        x_f.assign(&x);
        assert!(!x_f.is_standard_layout());

        let clf = DecisionTreeClassifier::new().with_max_depth(Some(4));
        let a = Fit::fit(&clf, &x, &y).unwrap();
        let b = Fit::fit(&clf, &x_f, &y).unwrap();
        assert_eq!(a.predict(&x).unwrap(), b.predict(&x_f).unwrap());
        assert_eq!(a.predict_proba(&x).unwrap(), b.predict_proba(&x_f).unwrap());

        // `x.t().to_owned().t()` is the other common non-standard layout.
        let x_t = x.t().to_owned();
        let x_tt = x_t.t().to_owned();
        let reg = DecisionTreeRegressor::new().with_max_depth(Some(4));
        let ra = Fit::fit(&reg, &x, &y).unwrap();
        let rb = Fit::fit(&reg, &x_f, &y).unwrap();
        assert_eq!(ra.predict(&x).unwrap(), rb.predict(&x_tt).unwrap());
        assert_eq!(ra.predict(&x).unwrap(), rb.predict(&x_f).unwrap());
    }

    /// #42: ties must go to the smallest class label, deterministically.
    #[test]
    fn leaf_ties_go_to_smallest_label() {
        let x = array![[1.0], [1.0], [1.0], [1.0]];
        let y = array![5.0, 2.0, 5.0, 2.0];
        for _ in 0..20 {
            let f = Fit::fit(&DecisionTreeClassifier::new(), &x, &y).unwrap();
            assert_eq!(f.predict(&x).unwrap()[0], 2.0);
            let fw = Fit::fit(
                &DecisionTreeClassifier::new().with_sample_weight(Some(Array1::ones(4))),
                &x,
                &y,
            )
            .unwrap();
            assert_eq!(fw.predict(&x).unwrap()[0], 2.0);
        }
        for _ in 0..20 {
            assert_eq!(majority_vote(&[3.0, 1.0, 3.0, 1.0, 2.0]), 1.0);
            assert_eq!(majority_vote(&[3.0, 3.0, 1.0]), 3.0);
        }
    }

    /// #45: with `max_features` the subset is drawn per split, and the seed
    /// changes the tree.
    #[test]
    fn max_features_is_per_split_and_seeded() {
        let (x, y) = data(300, 16);
        let base = DecisionTreeClassifier::new().with_max_features(Some(MaxFeatures::Sqrt));

        let t1 = Fit::fit(&base.clone().with_seed(1), &x, &y).unwrap();
        let mut feats = Vec::new();
        split_features(t1.tree(), &mut feats);
        feats.sort_unstable();
        feats.dedup();
        assert!(
            feats.len() > 4,
            "a deep tree with sqrt(16)=4 features per split should use more than 4 features overall, used {feats:?}"
        );

        let structure = |t: &FittedDecisionTreeClassifier<f64>| {
            let mut v = Vec::new();
            split_features(t.tree(), &mut v);
            v
        };
        let t2 = Fit::fit(&base.clone().with_seed(2), &x, &y).unwrap();
        assert_ne!(
            structure(&t1),
            structure(&t2),
            "different seeds must give different trees"
        );
        let (x_new, _) = data(500, 16);
        let x_new = x_new.mapv(|v| 1.0 - v);
        assert_ne!(
            t1.predict_proba(&x_new).unwrap(),
            t2.predict_proba(&x_new).unwrap()
        );

        // Same seed -> identical tree.
        let t1b = Fit::fit(&base.with_seed(1), &x, &y).unwrap();
        assert_eq!(structure(&t1), structure(&t1b));
    }

    #[test]
    fn sample_weight_length_is_validated() {
        let x = array![[1.0], [2.0], [3.0]];
        let y = array![0.0, 1.0, 1.0];
        let clf = DecisionTreeClassifier::new().with_sample_weight(Some(array![1.0, 1.0]));
        assert!(Fit::<f64>::fit(&clf, &x, &y).is_err());
        let reg = DecisionTreeRegressor::new().with_sample_weight(Some(array![1.0]));
        assert!(Fit::<f64>::fit(&reg, &x, &y).is_err());
    }

    #[test]
    fn nan_feature_does_not_panic() {
        let x = array![[1.0], [f64::NAN], [3.0], [4.0]];
        let y = array![0.0, 0.0, 1.0, 1.0];
        let f = Fit::fit(&DecisionTreeClassifier::new(), &x, &y).unwrap();
        assert_eq!(f.predict(&x).unwrap().len(), 4);
    }
}
