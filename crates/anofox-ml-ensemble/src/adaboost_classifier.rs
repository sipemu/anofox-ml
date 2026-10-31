use anofox_ml_core::{Fit, FitWeighted, Float, Predict, PredictProba, Result, RustMlError};
use anofox_ml_trees::{DecisionTreeClassifier, FittedDecisionTreeClassifier, SplitCriterion};
use ndarray::{Array1, Array2};

use crate::forest_common::{argmax_first, class_index, sorted_classes};

/// AdaBoost classifier parameters (unfitted state).
///
/// Implements the SAMME (Stagewise Additive Modeling using a Multi-class
/// Exponential loss function) algorithm as in scikit-learn >= 1.4 (which
/// removed SAMME.R). Each boosting round fits a decision tree (typically a
/// stump with `max_depth = 1`) on the full training set with the current
/// sample weights, then up-weights misclassified samples by
/// `exp(alpha)` so they receive greater emphasis in subsequent rounds.
///
/// `predict_proba` is `softmax(decision_function / (K - 1))` like sklearn.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AdaBoostClassifier {
    /// Number of boosting rounds (weak learners).
    pub n_estimators: usize,
    /// Learning rate that shrinks each estimator's contribution.
    pub learning_rate: f64,
    /// Maximum depth of each decision tree. `Some(1)` yields stumps.
    pub max_depth: Option<usize>,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl AdaBoostClassifier {
    /// Create a new `AdaBoostClassifier` with default parameters.
    pub fn new() -> Self {
        Self {
            n_estimators: 50,
            learning_rate: 1.0,
            max_depth: Some(1),
            seed: 0,
        }
    }

    /// Set the number of boosting rounds.
    pub fn with_n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators;
        self
    }

    /// Set the learning rate (shrinkage).
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the maximum depth of each tree.
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set the random seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl Default for AdaBoostClassifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted AdaBoost classifier.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedAdaBoostClassifier<F: Float> {
    /// Weak learners.
    estimators: Vec<FittedDecisionTreeClassifier<F>>,
    /// Weight (alpha) for each estimator.
    estimator_weights: Vec<F>,
    /// Weighted training error of each estimator.
    #[serde(default)]
    estimator_errors: Vec<F>,
    /// Unique class labels sorted in ascending order.
    classes: Vec<F>,
    /// Number of features expected at prediction time.
    n_features: usize,
}

impl<F: Float> Fit<F> for AdaBoostClassifier {
    type Fitted = FittedAdaBoostClassifier<F>;

    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Self::Fitted> {
        self.fit_impl(x, y, None)
    }
}

impl<F: Float> FitWeighted<F> for AdaBoostClassifier {
    type Fitted = FittedAdaBoostClassifier<F>;

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

impl AdaBoostClassifier {
    fn fit_impl<F: Float>(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        sample_weight: Option<&Array1<F>>,
    ) -> Result<FittedAdaBoostClassifier<F>> {
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
        if self.learning_rate <= 0.0 {
            return Err(RustMlError::InvalidParameter(
                "learning_rate must be > 0".into(),
            ));
        }

        let classes = sorted_classes(y);
        let n_classes = classes.len();
        if n_classes < 2 {
            return Err(RustMlError::InvalidParameter(
                "y must contain at least 2 distinct classes".into(),
            ));
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();
        let lr = self.learning_rate;
        let k_f = n_classes as f64;

        let base_tree = DecisionTreeClassifier {
            max_depth: self.max_depth,
            min_samples_split: 2,
            min_samples_leaf: 1,
            criterion: SplitCriterion::Gini,
            max_features: None,
            sample_weight: None,
            class_weight: None,
            seed: self.seed,
        };

        // Sample weights, kept normalized to sum to 1 (sklearn starts from
        // the normalised `sample_weight`).
        let mut weights = match sample_weight {
            Some(w) => {
                anofox_ml_core::check_sample_weight(w, n_samples)?;
                let w = w.mapv(|v| v.to_f64().unwrap());
                let total = w.sum();
                w / total
            }
            None => Array1::<f64>::from_elem(n_samples, 1.0 / n_samples as f64),
        };

        let mut estimators = Vec::with_capacity(self.n_estimators);
        let mut estimator_weights = Vec::with_capacity(self.n_estimators);
        let mut estimator_errors = Vec::with_capacity(self.n_estimators);
        let mut incorrect = vec![false; n_samples];

        for iboost in 0..self.n_estimators {
            // SAMME (sklearn >= 1.4): the weak learner is fit on the full
            // data with the current sample weights.
            let tree = base_tree.clone().with_sample_weight(Some(weights.clone()));
            let fitted_tree: FittedDecisionTreeClassifier<F> = tree.fit(x, y)?;
            let preds = fitted_tree.predict(x)?;

            let mut err = 0.0;
            let mut w_sum = 0.0;
            for i in 0..n_samples {
                incorrect[i] = preds[i] != y[i];
                w_sum += weights[i];
                if incorrect[i] {
                    err += weights[i];
                }
            }
            let err = err / w_sum;

            if err <= 0.0 {
                // Perfect fit: keep it with unit weight and stop.
                estimators.push(fitted_tree);
                estimator_weights.push(F::one());
                estimator_errors.push(F::zero());
                break;
            }
            if err >= 1.0 - 1.0 / k_f {
                // Worse than random guessing: discard and stop.
                if estimators.is_empty() {
                    return Err(RustMlError::InvalidParameter(
                        "base estimator in AdaBoostClassifier is worse than random; \
                         the ensemble cannot be fit"
                            .into(),
                    ));
                }
                break;
            }

            let alpha = lr * (((1.0 - err) / err).ln() + (k_f - 1.0).ln());
            estimators.push(fitted_tree);
            estimator_weights.push(F::from_f64(alpha).unwrap());
            estimator_errors.push(F::from_f64(err).unwrap());

            if iboost + 1 == self.n_estimators {
                break;
            }
            let boost = alpha.exp();
            for i in 0..n_samples {
                if incorrect[i] && weights[i] > 0.0 {
                    weights[i] *= boost;
                }
            }
            let total = weights.sum();
            if !total.is_finite() || total <= 0.0 {
                break;
            }
            weights /= total;
        }

        Ok(FittedAdaBoostClassifier {
            estimators,
            estimator_weights,
            estimator_errors,
            classes,
            n_features,
        })
    }
}

impl<F: Float> Predict<F> for FittedAdaBoostClassifier<F> {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        let dec = self.decision_function(x)?;
        let preds = if self.classes.len() == 2 {
            dec.column(0)
                .iter()
                .map(|&d| {
                    if d > F::zero() {
                        self.classes[1]
                    } else {
                        self.classes[0]
                    }
                })
                .collect()
        } else {
            dec.rows()
                .into_iter()
                .map(|row| self.classes[argmax_first(row.iter().copied())])
                .collect()
        };
        Ok(preds)
    }
}

impl<F: Float> FittedAdaBoostClassifier<F> {
    /// Number of estimators in the ensemble.
    pub fn n_estimators(&self) -> usize {
        self.estimators.len()
    }

    /// The unique classes discovered during training.
    pub fn classes(&self) -> &[F] {
        &self.classes
    }

    /// The weight of each estimator.
    pub fn estimator_weights(&self) -> &[F] {
        &self.estimator_weights
    }

    /// The weighted training error of each estimator (empty for models
    /// deserialized from older versions).
    pub fn estimator_errors(&self) -> &[F] {
        &self.estimator_errors
    }

    /// SAMME decision function (sklearn's `decision_function`).
    ///
    /// Every estimator votes `w` for its predicted class and `-w / (K - 1)`
    /// for every other class; the sum is divided by the total estimator
    /// weight. For binary problems the result has a single column
    /// (`score[class_1] - score[class_0]`, positive => `classes[1]`),
    /// otherwise shape `(n_samples, n_classes)`.
    pub fn decision_function(&self, x: &Array2<F>) -> Result<Array2<F>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        let n_samples = x.nrows();
        let n_classes = self.classes.len();
        let km1 = F::from_usize(n_classes - 1).unwrap();
        let mut scores = Array2::<F>::zeros((n_samples, n_classes));
        let mut w_total = F::zero();

        for (tree, &w) in self.estimators.iter().zip(self.estimator_weights.iter()) {
            w_total += w;
            let preds = tree.predict(x)?;
            let other = -w / km1;
            for (i, &p) in preds.iter().enumerate() {
                let c = class_index(&self.classes, p);
                for k in 0..n_classes {
                    scores[[i, k]] += if Some(k) == c { w } else { other };
                }
            }
        }
        if w_total > F::zero() {
            scores.mapv_inplace(|v| v / w_total);
        }

        if n_classes == 2 {
            let mut out = Array2::<F>::zeros((n_samples, 1));
            for i in 0..n_samples {
                out[[i, 0]] = scores[[i, 1]] - scores[[i, 0]];
            }
            Ok(out)
        } else {
            Ok(scores)
        }
    }

    /// Predict class probabilities (sklearn's SAMME `predict_proba`):
    /// `softmax(decision / (K - 1))`; for binary problems
    /// `softmax([-d / 2, d / 2])`.
    ///
    /// Returns an `Array2` of shape `(n_samples, n_classes)`.
    pub fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>> {
        let dec = self.decision_function(x)?;
        let n_samples = dec.nrows();
        let n_classes = self.classes.len();
        let mut proba = Array2::<F>::zeros((n_samples, n_classes));
        let two = F::from_f64(2.0).unwrap();
        let km1 = F::from_usize(n_classes - 1).unwrap();
        for i in 0..n_samples {
            if n_classes == 2 {
                let d = dec[[i, 0]] / two;
                proba[[i, 0]] = -d;
                proba[[i, 1]] = d;
            } else {
                for k in 0..n_classes {
                    proba[[i, k]] = dec[[i, k]] / km1;
                }
            }
            let mut row = proba.row_mut(i);
            let max = row.iter().copied().fold(F::neg_infinity(), F::max);
            row.mapv_inplace(|v| (v - max).exp());
            let sum = row.sum();
            row.mapv_inplace(|v| v / sum);
        }
        Ok(proba)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    /// numpy seed-0 data; expected values from sklearn 1.8.0
    /// `AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
    /// n_estimators=8, learning_rate=0.7, random_state=0)`.
    fn sklearn_data() -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let x = array![
            [0.195, 0.861],
            [0.411, 0.18],
            [-0.305, 0.584],
            [-0.25, 1.567],
            [1.855, -0.466],
            [1.167, 0.116],
            [0.272, 1.702],
            [-1.716, -1.651],
            [-1.919, 1.33],
            [1.113, 1.48],
            [1.914, 1.197],
            [-0.154, 1.122],
            [-1.527, 0.56],
            [-1.427, 1.779],
            [0.087, -0.341],
            [-0.942, 1.097],
            [-0.175, 0.274],
            [-1.925, 0.471],
            [0.448, 0.468],
            [1.775, 0.727],
            [-0.562, -0.252],
            [0.791, -1.759],
            [0.667, 0.683],
            [-1.158, -1.484]
        ];
        let yb = array![
            0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0
        ];
        let ym = array![
            1.0, 1.0, 0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 0.0, 2.0, 2.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0,
            0.0, 1.0, 2.0, 1.0, 2.0, 2.0, 0.0
        ];
        (x, yb, ym)
    }

    fn sklearn_params() -> AdaBoostClassifier {
        AdaBoostClassifier::new()
            .with_n_estimators(8)
            .with_learning_rate(0.7)
            .with_max_depth(Some(1))
    }

    #[test]
    fn test_binary_samme_matches_sklearn() {
        let (x, yb, _) = sklearn_data();
        let fitted: FittedAdaBoostClassifier<f64> = sklearn_params().fit(&x, &yb).unwrap();
        let expected_w = [
            1.126606538704,
            1.035548457138,
            0.642496518324,
            0.750266726612,
            0.799534569873,
            0.491001544562,
            0.739634426191,
            0.60866724212,
        ];
        assert_eq!(fitted.n_estimators(), 8);
        for (w, e) in fitted.estimator_weights().iter().zip(expected_w) {
            assert_abs_diff_eq!(*w, e, epsilon = 1e-9);
        }
        let dec = fitted.decision_function(&x).unwrap();
        let expected_dec = [
            -0.055867026878,
            -0.540398044091,
            -1.515468982787,
            0.421797526509,
        ];
        for (i, e) in expected_dec.iter().enumerate() {
            assert_abs_diff_eq!(dec[[i, 0]], *e, epsilon = 1e-9);
        }
        let proba = fitted.predict_proba(&x).unwrap();
        let expected_p0 = [
            0.513963125187,
            0.631905008077,
            0.819870295974,
            0.396086697941,
        ];
        for (i, e) in expected_p0.iter().enumerate() {
            assert_abs_diff_eq!(proba[[i, 0]], *e, epsilon = 1e-9);
        }
        let expected_pred = [
            0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,
        ];
        assert_eq!(fitted.predict(&x).unwrap().to_vec(), expected_pred.to_vec());
    }

    #[test]
    fn test_multiclass_samme_matches_sklearn() {
        let (x, _, ym) = sklearn_data();
        let fitted: FittedAdaBoostClassifier<f64> = sklearn_params().fit(&x, &ym).unwrap();
        let expected_w = [
            0.970406052784,
            1.389402312867,
            1.47215778907,
            1.42161121083,
            1.321664154882,
            1.49914525528,
            1.228163874927,
            1.262679716103,
        ];
        for (w, e) in fitted.estimator_weights().iter().zip(expected_w) {
            assert_abs_diff_eq!(*w, e, epsilon = 1e-9);
        }
        let proba = fitted.predict_proba(&x).unwrap();
        let expected = [
            [0.312672951283, 0.434113756491, 0.253213292226],
            [0.312672951283, 0.434113756491, 0.253213292226],
            [0.381150754432, 0.363149329936, 0.255699915632],
            [0.312672951283, 0.434113756491, 0.253213292226],
        ];
        for (i, row) in expected.iter().enumerate() {
            for (k, e) in row.iter().enumerate() {
                assert_abs_diff_eq!(proba[[i, k]], *e, epsilon = 1e-9);
            }
        }
        let expected_pred = [
            1.0, 1.0, 0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 0.0, 2.0, 2.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0,
            0.0, 1.0, 2.0, 0.0, 2.0, 2.0, 0.0,
        ];
        assert_eq!(fitted.predict(&x).unwrap().to_vec(), expected_pred.to_vec());
    }

    #[test]
    fn test_basic_binary_classification() {
        let x = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [10.0, 1.0],
            [11.0, 1.0],
            [12.0, 1.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let ada = AdaBoostClassifier {
            n_estimators: 50,
            learning_rate: 1.0,
            max_depth: Some(1),
            seed: 42,
        };
        let fitted: FittedAdaBoostClassifier<f64> = ada.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(*p, *t, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_multiclass_classification() {
        let x = array![
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [5.0, 5.0],
            [5.5, 5.0],
            [6.0, 5.0],
            [10.0, 10.0],
            [10.5, 10.0],
            [11.0, 10.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];

        let ada = AdaBoostClassifier {
            n_estimators: 100,
            learning_rate: 1.0,
            max_depth: Some(2),
            seed: 42,
        };
        let fitted: FittedAdaBoostClassifier<f64> = ada.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(*p, *t, epsilon = 1e-10);
        }

        assert_eq!(fitted.classes().len(), 3);
    }

    #[test]
    fn test_reproducibility() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let ada = AdaBoostClassifier {
            n_estimators: 20,
            learning_rate: 1.0,
            max_depth: Some(1),
            seed: 123,
        };

        let fitted1: FittedAdaBoostClassifier<f64> = ada.fit(&x, &y).unwrap();
        let fitted2: FittedAdaBoostClassifier<f64> = ada.fit(&x, &y).unwrap();

        let preds1 = fitted1.predict(&x).unwrap();
        let preds2 = fitted2.predict(&x).unwrap();

        for (a, b) in preds1.iter().zip(preds2.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-15);
        }
    }

    #[test]
    fn test_predict_proba_sums_to_one() {
        let x = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [10.0, 1.0],
            [11.0, 1.0],
            [12.0, 1.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let ada = AdaBoostClassifier {
            n_estimators: 20,
            learning_rate: 1.0,
            max_depth: Some(1),
            seed: 42,
        };
        let fitted: FittedAdaBoostClassifier<f64> = ada.fit(&x, &y).unwrap();

        let proba = fitted.predict_proba(&x).unwrap();
        assert_eq!(proba.nrows(), x.nrows());
        assert_eq!(proba.ncols(), 2);

        for i in 0..proba.nrows() {
            let row_sum: f64 = (0..proba.ncols()).map(|k| proba[[i, k]]).sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_predict_proba_high_confidence() {
        let x = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [10.0, 1.0],
            [11.0, 1.0],
            [12.0, 1.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let ada = AdaBoostClassifier {
            n_estimators: 50,
            learning_rate: 1.0,
            max_depth: Some(1),
            seed: 42,
        };
        let fitted: FittedAdaBoostClassifier<f64> = ada.fit(&x, &y).unwrap();

        let proba = fitted.predict_proba(&x).unwrap();
        // Class 0 samples should have high probability for class 0.
        for i in 0..3 {
            assert!(
                proba[[i, 0]] > 0.5,
                "expected high prob for class 0 at sample {i}"
            );
        }
        // Class 1 samples should have high probability for class 1.
        for i in 3..6 {
            assert!(
                proba[[i, 1]] > 0.5,
                "expected high prob for class 1 at sample {i}"
            );
        }
    }

    #[test]
    fn test_shape_mismatch_error() {
        let x = array![[1.0], [2.0]];
        let y = array![0.0, 1.0, 2.0];

        let ada = AdaBoostClassifier::default();
        let result: std::result::Result<FittedAdaBoostClassifier<f64>, _> = ada.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_predict_wrong_features_error() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0];

        let ada = AdaBoostClassifier {
            n_estimators: 5,
            seed: 0,
            ..Default::default()
        };
        let fitted: FittedAdaBoostClassifier<f64> = ada.fit(&x, &y).unwrap();

        let x_bad = array![[1.0], [2.0]];
        let result = fitted.predict(&x_bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_single_class_error() {
        let x = array![[1.0], [2.0], [3.0]];
        let y = array![1.0, 1.0, 1.0];

        let ada = AdaBoostClassifier::default();
        let result: std::result::Result<FittedAdaBoostClassifier<f64>, _> = ada.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_parameters() {
        let x = array![[1.0], [2.0]];
        let y = array![0.0, 1.0];

        let ada = AdaBoostClassifier {
            n_estimators: 0,
            ..Default::default()
        };
        assert!(Fit::<f64>::fit(&ada, &x, &y).is_err());

        let ada = AdaBoostClassifier {
            learning_rate: -0.1,
            ..Default::default()
        };
        assert!(Fit::<f64>::fit(&ada, &x, &y).is_err());
    }

    #[test]
    fn test_n_estimators_accessor() {
        let x = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [10.0, 1.0],
            [11.0, 1.0],
            [12.0, 1.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let ada = AdaBoostClassifier {
            n_estimators: 10,
            max_depth: Some(1),
            seed: 42,
            ..Default::default()
        };
        let fitted: FittedAdaBoostClassifier<f64> = ada.fit(&x, &y).unwrap();
        // The ensemble may stop early, but should have at most n_estimators.
        assert!(fitted.n_estimators() <= 10);
        assert!(fitted.n_estimators() >= 1);
    }

    #[test]
    fn test_predictions_are_valid_labels() {
        let x = array![
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [5.0, 5.0],
            [5.5, 5.0],
            [6.0, 5.0],
            [10.0, 10.0],
            [10.5, 10.0],
            [11.0, 10.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];

        let ada = AdaBoostClassifier {
            n_estimators: 50,
            learning_rate: 1.0,
            max_depth: Some(2),
            seed: 42,
        };
        let fitted: FittedAdaBoostClassifier<f64> = ada.fit(&x, &y).unwrap();

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
    fn test_with_builder_pattern() {
        let x = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [10.0, 1.0],
            [11.0, 1.0],
            [12.0, 1.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let ada = AdaBoostClassifier::new()
            .with_n_estimators(30)
            .with_learning_rate(0.5)
            .with_max_depth(Some(2))
            .with_seed(42);

        let fitted: FittedAdaBoostClassifier<f64> = ada.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), y.len());
    }

    #[test]
    fn test_estimator_weights_positive() {
        let x = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [10.0, 1.0],
            [11.0, 1.0],
            [12.0, 1.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let ada = AdaBoostClassifier {
            n_estimators: 10,
            learning_rate: 1.0,
            max_depth: Some(1),
            seed: 42,
        };
        let fitted: FittedAdaBoostClassifier<f64> = ada.fit(&x, &y).unwrap();

        for &w in fitted.estimator_weights() {
            assert!(w >= 0.0, "estimator weight must be non-negative, got {w}");
        }
    }

    #[test]
    fn test_empty_input_error() {
        let x: Array2<f64> = Array2::zeros((0, 2));
        let y: Array1<f64> = Array1::zeros(0);

        let ada = AdaBoostClassifier::default();
        let result: std::result::Result<FittedAdaBoostClassifier<f64>, _> = ada.fit(&x, &y);
        assert!(result.is_err());
    }
}

impl<F: Float> PredictProba<F> for FittedAdaBoostClassifier<F> {
    fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>> {
        Self::predict_proba(self, x)
    }
}
