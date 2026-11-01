use anofox_ml_core::{Fit, FitWeighted, Float, Predict, PredictProba, Result, RustMlError};
use anofox_ml_trees::{
    DecisionTreeClassifier, FittedDecisionTreeClassifier, MaxFeatures, SplitCriterion,
};
use ndarray::{Array1, Array2, Axis};
use rayon::prelude::*;

use crate::bagging_common::{add_importances, out_of_bag, select_columns, BagSampling};
use crate::forest_common::{argmax_first, sorted_classes};

/// Bagging (Bootstrap Aggregating) classifier parameters (unfitted state).
///
/// Trains an ensemble of decision tree classifiers (sklearn's
/// `BaggingClassifier` with its default tree estimator), each on a
/// bootstrap sample of the rows and, with `max_features`, on a random subset
/// of the features drawn once per estimator (`bootstrap_features` draws it
/// with replacement). Unlike a random forest, the features are not re-drawn
/// at every split.
///
/// As in sklearn, `predict` is the class with the highest averaged
/// probability. With `oob_score`, each training sample is also scored by
/// the trees that did not see it
/// ([`FittedBaggingClassifier::oob_decision_function`],
/// [`FittedBaggingClassifier::oob_score`]).
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
    /// Features drawn for each tree (`None` = all).
    #[serde(default)]
    pub max_features: Option<MaxFeatures>,
    /// Draw each tree's features with replacement.
    #[serde(default)]
    pub bootstrap_features: bool,
    /// Compute the out-of-bag decision function and accuracy (needs
    /// `bootstrap`).
    #[serde(default)]
    pub oob_score: bool,
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
            max_features: None,
            bootstrap_features: false,
            oob_score: false,
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
    /// Features drawn per tree, with replacement if `bootstrap_features`.
    pub fn with_max_features(
        mut self,
        max_features: Option<MaxFeatures>,
        bootstrap_features: bool,
    ) -> Self {
        self.max_features = max_features;
        self.bootstrap_features = bootstrap_features;
        self
    }
    /// Compute the out-of-bag decision function and score.
    pub fn with_oob_score(mut self, oob_score: bool) -> Self {
        self.oob_score = oob_score;
        self
    }
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    fn sampling(&self) -> BagSampling {
        BagSampling {
            n_estimators: self.n_estimators,
            max_samples: self.max_samples,
            bootstrap: self.bootstrap,
            max_features: self.max_features,
            bootstrap_features: self.bootstrap_features,
            oob_score: self.oob_score,
            seed: self.seed,
        }
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
    /// Feature subset of each tree (empty = all features).
    #[serde(default)]
    features: Vec<Vec<usize>>,
    /// Sorted training labels (empty for models saved before they were
    /// stored: then the union of the trees' classes).
    #[serde(default)]
    classes: Vec<F>,
    n_features: usize,
    #[serde(default)]
    oob_decision_function: Option<Array2<F>>,
    #[serde(default)]
    oob_score: Option<f64>,
}

impl<F: Float> Fit<F> for BaggingClassifier {
    type Fitted = FittedBaggingClassifier<F>;

    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Self::Fitted> {
        self.fit_impl(x, y, None)
    }
}

impl<F: Float> FitWeighted<F> for BaggingClassifier {
    type Fitted = FittedBaggingClassifier<F>;

    /// Per-sample weights (sklearn's `sample_weight`), applied on top of
    /// the bootstrap counts.
    fn fit_weighted(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        sample_weight: Option<&Array1<F>>,
    ) -> Result<Self::Fitted> {
        self.fit_impl(x, y, sample_weight)
    }
}

impl BaggingClassifier {
    fn fit_impl<F: Float>(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        sample_weight: Option<&Array1<F>>,
    ) -> Result<FittedBaggingClassifier<F>> {
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
        if let Some(w) = sample_weight {
            anofox_ml_core::check_sample_weight(w, x.nrows())?;
        }
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let plans = self.sampling().plans(n_samples, n_features)?;

        let tree_params = DecisionTreeClassifier {
            max_depth: self.max_depth,
            min_samples_split: 2,
            min_samples_leaf: 1,
            criterion: SplitCriterion::Gini,
            max_features: None,
            sample_weight: None,
            class_weight: None,
            seed: 0,
            ..DecisionTreeClassifier::new()
        };

        let trees: Vec<FittedDecisionTreeClassifier<F>> = plans
            .par_iter()
            .map(|plan| {
                let xs = select_columns(x, &plan.features);
                crate::forest_common::fit_tree_on_rows(
                    &tree_params.clone().with_seed(plan.seed),
                    &xs,
                    y,
                    &plan.rows,
                    sample_weight,
                )
            })
            .collect::<Result<_>>()?;

        let mut fitted = FittedBaggingClassifier {
            trees,
            features: plans.iter().map(|p| p.features.clone()).collect(),
            classes: sorted_classes(y),
            n_features,
            oob_decision_function: None,
            oob_score: None,
        };

        if self.oob_score {
            let n_classes = fitted.classes.len();
            let mut sum = Array2::<F>::zeros((n_samples, n_classes));
            let mut count = vec![0usize; n_samples];
            for (t, plan) in plans.iter().enumerate() {
                let oob = out_of_bag(n_samples, &plan.rows);
                if oob.is_empty() {
                    continue;
                }
                let xs = x.select(Axis(0), &oob);
                let proba = fitted.tree_proba(t, &xs)?;
                for (k, &i) in oob.iter().enumerate() {
                    let mut row = sum.row_mut(i);
                    row += &proba.row(k);
                    count[i] += 1;
                }
            }
            let (mut correct, mut scored) = (0usize, 0usize);
            for (i, mut row) in sum.rows_mut().into_iter().enumerate() {
                if count[i] == 0 {
                    row.fill(F::nan());
                    continue;
                }
                row /= F::from_usize(count[i]).unwrap();
                scored += 1;
                if fitted.classes[argmax_first(row.iter().copied())] == y[i] {
                    correct += 1;
                }
            }
            fitted.oob_score = Some(if scored > 0 {
                correct as f64 / scored as f64
            } else {
                f64::NAN
            });
            fitted.oob_decision_function = Some(sum);
        }
        Ok(fitted)
    }
}

impl<F: Float> Predict<F> for FittedBaggingClassifier<F> {
    /// The class with the highest averaged probability (sklearn).
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        let proba = self.predict_proba(x)?;
        let classes = self.classes();
        Ok(proba
            .rows()
            .into_iter()
            .map(|row| classes[argmax_first(row.iter().copied())])
            .collect())
    }
}

impl<F: Float> FittedBaggingClassifier<F> {
    fn tree_features(&self, t: usize) -> &[usize] {
        self.features.get(t).map_or(&[], Vec::as_slice)
    }

    /// Probabilities of tree `t` for the (full-width) rows `x`, mapped to
    /// the ensemble's class columns.
    fn tree_proba(&self, t: usize, x: &Array2<F>) -> Result<Array2<F>> {
        let tree = &self.trees[t];
        let local = tree.predict_proba(&select_columns(x, self.tree_features(t)))?;
        let classes = self.classes();
        let mut out = Array2::<F>::zeros((x.nrows(), classes.len()));
        for (local_ci, tc) in tree.classes().into_iter().enumerate() {
            if let Some(ci) = classes.iter().position(|&c| c == tc) {
                out.column_mut(ci).assign(&local.column(local_ci));
            }
        }
        Ok(out)
    }

    /// Feature importances averaged across all trees and normalized to sum to 1.
    pub fn feature_importances(&self) -> Array1<F> {
        let mut importances = vec![F::zero(); self.n_features];
        for (t, tree) in self.trees.iter().enumerate() {
            add_importances(
                &mut importances,
                &tree.feature_importances(),
                self.tree_features(t),
            );
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

    /// Feature subset each tree was trained on (empty = all features).
    pub fn estimators_features(&self) -> &[Vec<usize>] {
        &self.features
    }

    /// Out-of-bag class probabilities of every training sample (NaN rows
    /// for samples that every tree saw); `None` unless fitted with
    /// `oob_score`.
    pub fn oob_decision_function(&self) -> Option<&Array2<F>> {
        self.oob_decision_function.as_ref()
    }

    /// Out-of-bag accuracy over the samples with an out-of-bag prediction;
    /// `None` unless fitted with `oob_score`.
    pub fn oob_score(&self) -> Option<f64> {
        self.oob_score
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
        let all: Vec<Array2<F>> = (0..self.trees.len())
            .into_par_iter()
            .map(|t| self.tree_proba(t, x))
            .collect::<Result<_>>()?;
        let mut avg = Array2::<F>::zeros((x.nrows(), self.classes().len()));
        for p in &all {
            avg += p;
        }
        Ok(avg / F::from_usize(self.trees.len()).unwrap())
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

    /// The sorted class labels.
    pub fn classes(&self) -> Vec<F> {
        if !self.classes.is_empty() {
            return self.classes.clone();
        }
        let mut classes: Vec<F> = Vec::new();
        for tree in &self.trees {
            for c in tree.classes() {
                if !classes.contains(&c) {
                    classes.push(c);
                }
            }
        }
        classes.sort_by(anofox_ml_trees::split::cmp_nan_last);
        classes
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
        let valid_labels: std::collections::HashSet<u64> = y.iter().map(|v| v.to_bits()).collect();
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

impl<F: Float> PredictProba<F> for FittedBaggingClassifier<F> {
    fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>> {
        Self::predict_proba(self, x)
    }
}
