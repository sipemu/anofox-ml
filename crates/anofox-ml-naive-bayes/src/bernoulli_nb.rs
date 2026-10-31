use anofox_ml_core::{
    Fit, FitWeighted, Float, PartialFit, Predict, PredictLogProba, PredictProba, Result,
};
use ndarray::{Array1, Array2, Axis};

use crate::common::{
    argmax_labels, check_alpha, check_n_features, class_log_prior, log_proba_from_jll,
    proba_from_jll,
};
use crate::discrete::{accumulate, Counts};

fn default_true() -> bool {
    true
}

/// Bernoulli Naive Bayes classifier parameters (unfitted state).
///
/// Designed for binary/boolean features.  Input is binarized at a configurable
/// threshold before fitting and prediction: values strictly greater than the
/// threshold become 1, otherwise 0.  Uses Laplace (additive) smoothing
/// controlled by the `alpha` parameter. Mirrors sklearn's `BernoulliNB`.
///
/// Uses the type-state pattern: call [`Fit::fit`] to produce a
/// [`FittedBernoulliNB`] that can make predictions.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BernoulliNB {
    /// Additive (Laplace) smoothing parameter (>= 0).
    pub alpha: f64,
    /// Threshold for binarizing features (values > threshold become 1).
    pub binarize_threshold: f64,
    /// If `false` and `alpha < 1e-10`, alpha is clamped to `1e-10` (sklearn's
    /// `force_alpha`). Default `true`.
    #[serde(default = "default_true")]
    pub force_alpha: bool,
    /// Whether to learn class priors from the data (`false`: uniform prior).
    #[serde(default = "default_true")]
    pub fit_prior: bool,
    /// Fixed class prior probabilities (sorted class order).
    #[serde(default)]
    pub class_prior: Option<Vec<f64>>,
}

impl BernoulliNB {
    /// Create a new `BernoulliNB` with default parameters
    /// (`alpha = 1.0`, `binarize_threshold = 0.0`).
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            binarize_threshold: 0.0,
            force_alpha: true,
            fit_prior: true,
            class_prior: None,
        }
    }

    /// Set the additive smoothing parameter.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the binarization threshold.
    pub fn with_binarize_threshold(mut self, threshold: f64) -> Self {
        self.binarize_threshold = threshold;
        self
    }

    /// Set `force_alpha` (see field docs).
    pub fn with_force_alpha(mut self, force_alpha: bool) -> Self {
        self.force_alpha = force_alpha;
        self
    }

    /// Set whether class priors are learned.
    pub fn with_fit_prior(mut self, fit_prior: bool) -> Self {
        self.fit_prior = fit_prior;
        self
    }

    /// Fix the class priors.
    pub fn with_class_prior(mut self, class_prior: Option<Vec<f64>>) -> Self {
        self.class_prior = class_prior;
        self
    }

    fn finish<F: Float>(&self, counts: Counts<F>, threshold: F) -> Result<FittedBernoulliNB<F>> {
        let alpha = F::from_f64(check_alpha(self.alpha, self.force_alpha)?).unwrap();
        let two = F::one() + F::one();
        let log_class_prior = class_log_prior(
            &counts.class_count,
            self.fit_prior,
            self.class_prior.as_deref(),
        )?;
        let mut log_feature_prob = counts.feature_count.mapv(|c| c + alpha);
        for (mut row, &cc) in log_feature_prob
            .rows_mut()
            .into_iter()
            .zip(counts.class_count.iter())
        {
            let log_den = (cc + alpha * two).ln();
            row.mapv_inplace(|v| v.ln() - log_den);
        }
        // log(1 - p) computed as sklearn: np.log(1 - np.exp(feature_log_prob_)).
        let log_feature_neg_prob = log_feature_prob.mapv(|v| (F::one() - v.exp()).ln());
        Ok(FittedBernoulliNB {
            class_labels: counts.classes,
            log_class_prior: log_class_prior.to_vec(),
            log_feature_prob,
            log_feature_neg_prob,
            binarize_threshold: threshold,
            class_count: counts.class_count,
            feature_count: counts.feature_count,
        })
    }

    /// Incremental fit with optional per-sample weights (sklearn's
    /// `partial_fit(X, y, classes, sample_weight)`).
    pub fn partial_fit_weighted<F: Float>(
        &self,
        state: Option<FittedBernoulliNB<F>>,
        x: &Array2<F>,
        y: &Array1<F>,
        classes: Option<&[F]>,
        sample_weight: Option<&Array1<F>>,
    ) -> Result<FittedBernoulliNB<F>> {
        check_alpha(self.alpha, self.force_alpha)?;
        let threshold = F::from_f64(self.binarize_threshold).unwrap();
        let x_bin = binarize(x, threshold);
        let prev = state
            .map(|s| Counts::from_parts(s.class_labels, s.class_count, s.feature_count))
            .transpose()?;
        let counts = accumulate(prev, &x_bin, y, classes, sample_weight)?;
        self.finish(counts, threshold)
    }
}

impl Default for BernoulliNB {
    fn default() -> Self {
        Self::new()
    }
}

fn binarize<F: Float>(x: &Array2<F>, threshold: F) -> Array2<F> {
    x.mapv(|v| if v > threshold { F::one() } else { F::zero() })
}

/// Fitted Bernoulli Naive Bayes classifier.
///
/// Stores log class priors, log feature probabilities, the binarization
/// threshold, and the raw (weighted) counts needed for `partial_fit`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedBernoulliNB<F: Float> {
    /// Unique sorted class labels.
    class_labels: Vec<F>,
    /// Log prior probability of each class, shape `(n_classes,)`.
    log_class_prior: Vec<F>,
    /// Log probability of feature being 1 per class, shape `(n_classes, n_features)`.
    log_feature_prob: Array2<F>,
    /// Log probability of feature being 0 per class, shape `(n_classes, n_features)`.
    log_feature_neg_prob: Array2<F>,
    /// Binarization threshold used during fitting.
    binarize_threshold: F,
    /// Weighted number of samples per class.
    #[serde(default)]
    class_count: Array1<F>,
    /// Weighted number of samples per class with each feature active.
    #[serde(default)]
    feature_count: Array2<F>,
}

impl<F: Float> FittedBernoulliNB<F> {
    /// Returns the unique sorted class labels.
    pub fn classes(&self) -> &[F] {
        &self.class_labels
    }

    /// Returns the log prior probability of each class.
    pub fn log_class_prior(&self) -> &[F] {
        &self.log_class_prior
    }

    /// Returns the log feature probabilities, shape `(n_classes, n_features)`.
    pub fn log_feature_prob(&self) -> &Array2<F> {
        &self.log_feature_prob
    }

    /// Weighted number of samples per class (sklearn's `class_count_`).
    pub fn class_count(&self) -> &Array1<F> {
        &self.class_count
    }

    /// Weighted active-feature counts per class (sklearn's `feature_count_`).
    pub fn feature_count(&self) -> &Array2<F> {
        &self.feature_count
    }

    /// Joint log-likelihood, shape `(n_samples, n_classes)`:
    /// `X_bin · (log p − log(1−p))ᵀ + Σ_j log(1−p_j) + log_prior`.
    pub fn joint_log_likelihood(&self, x: &Array2<F>) -> Result<Array2<F>> {
        check_n_features(x, self.log_feature_prob.ncols())?;
        let x_bin = binarize(x, self.binarize_threshold);
        let diff = &self.log_feature_prob - &self.log_feature_neg_prob;
        let offset = self.log_feature_neg_prob.sum_axis(Axis(1))
            + Array1::from(self.log_class_prior.clone());
        Ok(x_bin.dot(&diff.t()) + &offset)
    }

    /// Predict class probabilities for each sample in `x`.
    ///
    /// Returns an array of shape `(n_samples, n_classes)` where each row sums
    /// to 1.
    pub fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>> {
        Ok(proba_from_jll(self.joint_log_likelihood(x)?))
    }
}

impl<F: Float> FitWeighted<F> for BernoulliNB {
    type Fitted = FittedBernoulliNB<F>;

    fn fit_weighted(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        sample_weight: Option<&Array1<F>>,
    ) -> Result<Self::Fitted> {
        self.partial_fit_weighted(None, x, y, None, sample_weight)
    }
}

impl<F: Float> Fit<F> for BernoulliNB {
    type Fitted = FittedBernoulliNB<F>;

    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Self::Fitted> {
        self.partial_fit_weighted(None, x, y, None, None)
    }
}

impl<F: Float> PartialFit<F> for BernoulliNB {
    type Fitted = FittedBernoulliNB<F>;

    fn partial_fit(
        &self,
        state: Option<Self::Fitted>,
        x: &Array2<F>,
        y: &Array1<F>,
        classes: Option<&[F]>,
    ) -> Result<Self::Fitted> {
        self.partial_fit_weighted(state, x, y, classes, None)
    }
}

impl<F: Float> Predict<F> for FittedBernoulliNB<F> {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        let jll = self.joint_log_likelihood(x)?;
        Ok(argmax_labels(&jll, &self.class_labels))
    }
}

impl<F: Float> PredictProba<F> for FittedBernoulliNB<F> {
    fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>> {
        Self::predict_proba(self, x)
    }
}

impl<F: Float> PredictLogProba<F> for FittedBernoulliNB<F> {
    fn predict_log_proba(&self, x: &Array2<F>) -> Result<Array2<F>> {
        Ok(log_proba_from_jll(self.joint_log_likelihood(x)?))
    }
}

impl<F: Float> anofox_ml_core::ClassifierScore<F> for FittedBernoulliNB<F> {}

#[cfg(test)]
mod tests {
    use super::*;
    use anofox_ml_core::RustMlError;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_basic_binary_classification() {
        // Class 0: features 0,1 are active; Class 1: features 2,3 are active.
        let x_train = array![
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0]
        ];
        let y_train = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let nb = BernoulliNB::new();
        let fitted: FittedBernoulliNB<f64> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        let x_test = array![[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]];
        let preds = fitted.predict(&x_test).unwrap();

        assert_abs_diff_eq!(preds[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_predict_proba_sums_to_one() {
        let x_train = array![[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]];
        let y_train = array![0.0, 0.0, 1.0, 1.0];

        let nb = BernoulliNB::new();
        let fitted: FittedBernoulliNB<f64> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        let x_test = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]];
        let proba = fitted.predict_proba(&x_test).unwrap();

        for i in 0..proba.nrows() {
            let row_sum: f64 = proba.row(i).iter().sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }

        for &p in proba.iter() {
            assert!(p >= 0.0 && p <= 1.0, "probability {} out of range", p);
        }
    }

    #[test]
    fn test_binarize_threshold() {
        // Use a non-default threshold of 0.5.
        // Values > 0.5 become 1, else 0.
        let x_train = array![[0.8, 0.2], [0.9, 0.1], [0.2, 0.8], [0.1, 0.9]];
        let y_train = array![0.0, 0.0, 1.0, 1.0];

        let nb = BernoulliNB::new().with_binarize_threshold(0.5);
        let fitted: FittedBernoulliNB<f64> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        // After binarization at 0.5: class 0 -> [1,0], class 1 -> [0,1]
        let x_test = array![[0.7, 0.3], [0.3, 0.7]];
        let preds = fitted.predict(&x_test).unwrap();

        assert_abs_diff_eq!(preds[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_alpha_smoothing_effect() {
        let x_train = array![[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]];
        let y_train = array![0.0, 0.0, 1.0, 1.0];

        let nb_small = BernoulliNB::new().with_alpha(1e-10);
        let fitted_small: FittedBernoulliNB<f64> = Fit::fit(&nb_small, &x_train, &y_train).unwrap();

        let nb_large = BernoulliNB::new().with_alpha(100.0);
        let fitted_large: FittedBernoulliNB<f64> = Fit::fit(&nb_large, &x_train, &y_train).unwrap();

        let x_test = array![[1.0, 0.0]];
        let proba_small = fitted_small.predict_proba(&x_test).unwrap();
        let proba_large = fitted_large.predict_proba(&x_test).unwrap();

        // With small alpha, class 0 probability should be more extreme.
        assert!(
            proba_small[[0, 0]] > proba_large[[0, 0]],
            "smaller alpha should give more extreme probabilities: small={}, large={}",
            proba_small[[0, 0]],
            proba_large[[0, 0]]
        );
    }

    #[test]
    fn test_multiclass() {
        // Three classes with distinctive binary patterns.
        let x_train = array![
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0]
        ];
        let y_train = array![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];

        let nb = BernoulliNB::new();
        let fitted: FittedBernoulliNB<f64> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        assert_eq!(fitted.classes().len(), 3);

        let x_test = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let preds = fitted.predict(&x_test).unwrap();

        assert_abs_diff_eq!(preds[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[1], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[2], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_shape_errors() {
        // Fit with mismatched X rows and y length.
        let x = array![[1.0, 0.0], [0.0, 1.0]];
        let y = array![0.0, 1.0, 0.0];

        let nb = BernoulliNB::new();
        let result: Result<FittedBernoulliNB<f64>> = Fit::fit(&nb, &x, &y);
        assert!(result.is_err());
        match result {
            Err(RustMlError::ShapeMismatch(_)) => {}
            other => panic!("expected ShapeMismatch error, got {:?}", other),
        }

        // Predict with wrong number of features.
        let x_train = array![[1.0, 0.0], [0.0, 1.0]];
        let y_train = array![0.0, 1.0];
        let fitted: FittedBernoulliNB<f64> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        let x_test = array![[1.0, 0.0, 1.0]];
        let result = fitted.predict(&x_test);
        assert!(result.is_err());
        match result {
            Err(RustMlError::ShapeMismatch(_)) => {}
            other => panic!("expected ShapeMismatch error, got {:?}", other),
        }
    }

    #[test]
    fn test_empty_input_error() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<f64>::zeros(0);

        let nb = BernoulliNB::new();
        let result: Result<FittedBernoulliNB<f64>> = Fit::fit(&nb, &x, &y);
        assert!(result.is_err());
        match result {
            Err(RustMlError::EmptyInput(_)) => {}
            other => panic!("expected EmptyInput error, got {:?}", other),
        }
    }

    #[test]
    fn test_f32_support() {
        let x_train: Array2<f32> = array![[1.0f32, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]];
        let y_train: Array1<f32> = array![0.0f32, 0.0, 1.0, 1.0];

        let nb = BernoulliNB::new();
        let fitted: FittedBernoulliNB<f32> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        let x_test: Array2<f32> = array![[1.0f32, 0.0], [0.0, 1.0]];
        let preds = fitted.predict(&x_test).unwrap();

        assert_abs_diff_eq!(preds[0], 0.0f32, epsilon = 1e-5);
        assert_abs_diff_eq!(preds[1], 1.0f32, epsilon = 1e-5);
    }

    #[test]
    fn test_predict_proba_shape_error() {
        let x_train = array![[1.0, 0.0], [0.0, 1.0]];
        let y_train = array![0.0, 1.0];

        let nb = BernoulliNB::new();
        let fitted: FittedBernoulliNB<f64> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        let x_test = array![[1.0]];
        let result = fitted.predict_proba(&x_test);
        assert!(result.is_err());
        match result {
            Err(RustMlError::ShapeMismatch(_)) => {}
            other => panic!("expected ShapeMismatch error, got {:?}", other),
        }
    }
}

/// Reference values generated with scikit-learn 1.8.0.
#[cfg(test)]
mod sklearn_parity {
    use super::*;
    use crate::common::assert_close2;
    use ndarray::array;

    fn data() -> (Array2<f64>, Array1<f64>, Array1<f64>, Array2<f64>) {
        let x = array![
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0]
        ];
        let y = array![0.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let w = array![1.0, 2.0, 0.5, 1.0, 1.0, 3.0];
        let t = array![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 0.0]
        ];
        (x, y, w, t)
    }

    #[test]
    fn sample_weight_and_alpha() {
        let (x, y, w, t) = data();
        let m: FittedBernoulliNB<f64> = BernoulliNB::new()
            .with_alpha(0.7)
            .fit_weighted(&x, &y, Some(&w))
            .unwrap();
        assert_close2(
            &m.predict_proba(&t).unwrap(),
            &[
                &[0.9861878646810236, 0.01381213531897653],
                &[0.026946248142374372, 0.9730537518576254],
                &[0.9571401323881754, 0.04285986761182453],
            ],
            1e-12,
            "proba",
        );
        assert_close2(
            &PredictLogProba::predict_log_proba(&m, &t).unwrap(),
            &[
                &[-0.013908410397587279, -4.282207702298754],
                &[-3.6139112068148203, -0.027315954891752092],
                &[-0.043805469421067667, -3.149819377792981],
            ],
            1e-12,
            "log_proba",
        );
    }

    #[test]
    fn partial_fit_mixed_weights() {
        let (x, y, w, t) = data();
        let nb = BernoulliNB::new();
        let m = nb
            .partial_fit(
                None,
                &x.slice(ndarray::s![..3, ..]).to_owned(),
                &y.slice(ndarray::s![..3]).to_owned(),
                Some(&[0.0, 1.0]),
            )
            .unwrap();
        let m = nb
            .partial_fit_weighted(
                Some(m),
                &x.slice(ndarray::s![3.., ..]).to_owned(),
                &y.slice(ndarray::s![3..]).to_owned(),
                None,
                Some(&w.slice(ndarray::s![3..]).to_owned()),
            )
            .unwrap();
        assert_close2(
            &m.predict_proba(&t).unwrap(),
            &[
                &[0.9688638028007635, 0.031136197199236736],
                &[0.03302000316309905, 0.966979996836901],
                &[0.8806094457985988, 0.1193905542014011],
            ],
            1e-12,
            "partial_fit proba",
        );
    }

    #[test]
    fn negative_alpha_rejected() {
        let (x, y, _, _) = data();
        let r: Result<FittedBernoulliNB<f64>> = BernoulliNB::new().with_alpha(-1.0).fit(&x, &y);
        assert!(r.is_err());
    }
}
