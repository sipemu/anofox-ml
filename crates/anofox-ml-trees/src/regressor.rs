use anofox_ml_core::{Fit, FitWeighted, Float, Predict, Result, RustMlError};
use ndarray::{Array1, Array2};

use crate::classifier::{check_rows, check_xy};
use crate::engine::{grow, GrowParams};
use crate::node::TreeNode;
use crate::split::{MaxFeatures, SplitCriterion};

/// Decision tree regressor parameters (unfitted state).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DecisionTreeRegressor {
    pub max_depth: Option<usize>,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    /// Maximum number of features to consider at each split.
    pub max_features: Option<MaxFeatures>,
    /// Per-sample weights.
    #[serde(skip)]
    pub sample_weight: Option<Array1<f64>>,
    /// Seed for the per-split feature subsampling RNG (only used when
    /// `max_features` is set). Different seeds give different trees.
    #[serde(default)]
    pub seed: u64,
}

impl DecisionTreeRegressor {
    /// Create a new `DecisionTreeRegressor` with sensible defaults.
    pub fn new() -> Self {
        Self {
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            sample_weight: None,
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

    /// Set the seed of the per-split feature subsampling RNG.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl Default for DecisionTreeRegressor {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted decision tree regressor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedDecisionTreeRegressor<F: Float> {
    tree: TreeNode<F>,
    n_features: usize,
}

impl<F: Float> Fit<F> for DecisionTreeRegressor {
    type Fitted = FittedDecisionTreeRegressor<F>;

    /// Fits with the configured `sample_weight` (if any).
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Self::Fitted> {
        let sw = self
            .sample_weight
            .as_ref()
            .map(|sw| sw.mapv(|v| F::from_f64(v).unwrap()));
        self.fit_impl(x, y, sw.as_ref())
    }
}

impl<F: Float> FitWeighted<F> for DecisionTreeRegressor {
    type Fitted = FittedDecisionTreeRegressor<F>;

    /// `sample_weight` replaces the configured one.
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

impl DecisionTreeRegressor {
    fn fit_impl<F: Float>(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        sample_weight: Option<&Array1<F>>,
    ) -> Result<FittedDecisionTreeRegressor<F>> {
        check_xy(x, y)?;
        if let Some(sw) = sample_weight {
            anofox_ml_core::check_sample_weight(sw, y.len())?;
        }
        let rows: Vec<usize> = (0..x.nrows()).collect();
        self.fit_on_rows(x, y, &rows, sample_weight)
    }

    /// Fit on the sample `rows` of `(x, y)` without copying `X`. Rows may
    /// repeat (a bootstrap draw counts each copy as a sample, exactly as if
    /// the rows had been copied). `sample_weight` is indexed by row of `x`.
    pub fn fit_on_rows<F: Float>(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        rows: &[usize],
        sample_weight: Option<&Array1<F>>,
    ) -> Result<FittedDecisionTreeRegressor<F>> {
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
            let drawn = Array1::from_iter(rows.iter().map(|&r| sw[r]));
            anofox_ml_core::check_sample_weight(&drawn, drawn.len())?;
        }
        let n_features = x.ncols();
        let params = GrowParams {
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            criterion: SplitCriterion::Mse,
            max_features_k: self.max_features.map(|mf| mf.resolve(n_features)),
            n_features,
        };
        let tree = grow(x, y, rows, sample_weight, &params, self.seed);
        Ok(FittedDecisionTreeRegressor { tree, n_features })
    }
}

impl<F: Float> Predict<F> for FittedDecisionTreeRegressor<F> {
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

impl<F: Float> FittedDecisionTreeRegressor<F> {
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

    /// Number of leaves of the fitted tree.
    pub fn n_leaves(&self) -> usize {
        self.tree.n_leaves()
    }

    /// Index of the leaf each row of `x` ends up in (like sklearn's
    /// `apply`, but with leaves numbered `0..n_leaves()` in pre-order, left
    /// subtree first).
    pub fn apply(&self, x: &Array2<F>) -> Result<Vec<usize>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        Ok(self.tree.apply(x))
    }

    /// Overwrite the leaf values, indexed like [`Self::apply`]. Used by
    /// boosting to replace the fitted mean with a loss-specific leaf
    /// estimate (e.g. a Newton step).
    pub fn set_leaf_values(&mut self, values: &[F]) -> Result<()> {
        let mut leaves = self.tree.leaf_values_mut();
        if leaves.len() != values.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "tree has {} leaves but {} values were given",
                leaves.len(),
                values.len()
            )));
        }
        for (leaf, &v) in leaves.iter_mut().zip(values) {
            **leaf = v;
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
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
    fn fit_weighted_integer_weights_equal_duplicated_rows() {
        let x = array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![1.0, 1.5, 3.0, 2.0, 7.0, 6.5];
        let w = array![1.0, 3.0, 1.0, 2.0, 1.0, 1.0];
        let idx: Vec<usize> = vec![0, 1, 1, 1, 2, 3, 3, 4, 5];
        let xd = x.select(ndarray::Axis(0), &idx);
        let yd = y.select(ndarray::Axis(0), &idx);
        let tree = DecisionTreeRegressor::new().with_max_depth(Some(2));
        let fw = tree.fit_weighted(&x, &y, Some(&w)).unwrap();
        let fd = Fit::fit(&tree, &xd, &yd).unwrap();
        let q = array![[0.2], [1.1], [2.6], [3.4], [4.9]];
        assert_eq!(fw.predict(&q).unwrap(), fd.predict(&q).unwrap());
        assert!(tree.fit_weighted(&x, &y, Some(&array![1.0])).is_err());
    }

    #[test]
    fn test_simple_regression() {
        // y = 2*x, tree should learn perfect piecewise approximation
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let tree = DecisionTreeRegressor::default();
        let fitted = Fit::fit(&tree, &x, &y).unwrap();

        // Predict on training data — should be perfect with unlimited depth
        let preds = fitted.predict(&x).unwrap();
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(p, t, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_max_depth_regression() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![1.0, 2.0, 3.0, 4.0];

        let tree = DecisionTreeRegressor {
            max_depth: Some(1),
            ..Default::default()
        };
        let fitted = Fit::fit(&tree, &x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        // With depth 1: left leaf gets mean(1,2)=1.5, right gets mean(3,4)=3.5
        assert_abs_diff_eq!(preds[0], 1.5, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[1], 1.5, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[2], 3.5, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[3], 3.5, epsilon = 1e-10);
    }

    #[test]
    fn test_feature_importances() {
        let x = array![[1.0, 100.0], [2.0, 200.0], [3.0, 300.0], [4.0, 400.0]];
        let y = array![1.0, 2.0, 3.0, 4.0];

        let tree = DecisionTreeRegressor::default();
        let fitted = Fit::fit(&tree, &x, &y).unwrap();

        let importances = fitted.feature_importances();
        // All importances should be non-negative
        for &imp in importances.iter() {
            assert!(imp >= 0.0, "importance should be non-negative, got {}", imp);
        }
        // Sum should be 1.0
        let sum: f64 = importances.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_min_samples_split_regression() {
        // 4 samples with min_samples_split=5: root cannot split
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![1.0, 2.0, 3.0, 4.0];

        let tree = DecisionTreeRegressor::new().with_min_samples_split(5);
        let fitted = Fit::fit(&tree, &x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        // Single leaf predicts the mean: (1+2+3+4)/4 = 2.5
        for &p in preds.iter() {
            assert_abs_diff_eq!(p, 2.5, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_min_samples_leaf_regression() {
        // 4 samples, min_samples_leaf=3 means no valid split can produce
        // leaves with >= 3 samples each (4 total), so tree is a single leaf.
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![1.0, 2.0, 3.0, 4.0];

        let tree = DecisionTreeRegressor::new().with_min_samples_leaf(3);
        let fitted = Fit::fit(&tree, &x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        // Single leaf predicts the mean: 2.5
        for &p in preds.iter() {
            assert_abs_diff_eq!(p, 2.5, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_single_feature_regression() {
        // Simple linear data with one feature
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![10.0, 20.0, 30.0, 40.0, 50.0];

        let tree = DecisionTreeRegressor::default();
        let fitted = Fit::fit(&tree, &x, &y).unwrap();

        // With unlimited depth, should perfectly fit training data
        let preds = fitted.predict(&x).unwrap();
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(p, t, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_shape_mismatch_error() {
        let x = array![[1.0], [2.0], [3.0]];
        let y = array![1.0, 2.0]; // 3 rows vs 2 labels

        let tree = DecisionTreeRegressor::default();
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

        let tree = DecisionTreeRegressor::default();
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
        let y = array![1.0, 2.0, 3.0, 4.0];

        let tree = DecisionTreeRegressor::default();
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
    fn test_apply_and_set_leaf_values() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![1.0, 1.0, 5.0, 5.0, 9.0, 9.0];
        let mut fitted = DecisionTreeRegressor::new()
            .with_max_depth(Some(2))
            .fit(&x, &y)
            .unwrap();
        let n_leaves = fitted.n_leaves();
        assert_eq!(n_leaves, 3);
        let leaves = fitted.apply(&x).unwrap();
        // Same leaf <=> same prediction; numbering covers 0..n_leaves.
        let preds = fitted.predict(&x).unwrap();
        for i in 0..6 {
            for j in 0..6 {
                assert_eq!(leaves[i] == leaves[j], preds[i] == preds[j]);
            }
        }
        assert!(leaves.iter().all(|&l| l < n_leaves));
        // Overwrite leaf values with their index and check predict routes there.
        let vals: Vec<f64> = (0..n_leaves).map(|l| l as f64 * 10.0).collect();
        fitted.set_leaf_values(&vals).unwrap();
        let preds = fitted.predict(&x).unwrap();
        for i in 0..6 {
            assert_eq!(preds[i], leaves[i] as f64 * 10.0);
        }
        assert!(fitted.set_leaf_values(&[1.0]).is_err());
        assert!(fitted.apply(&array![[1.0, 2.0]]).is_err());
    }
}
