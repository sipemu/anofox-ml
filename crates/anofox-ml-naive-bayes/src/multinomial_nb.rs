use anofox_ml_core::{
    Fit, FitWeighted, Float, PartialFit, Predict, PredictLogProba, PredictProba, Result,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, AsArray, Ix1, Ix2};

use crate::common::{
    argmax_labels, check_alpha, check_n_features, class_log_prior, log_proba_from_jll,
    proba_from_jll,
};
use crate::discrete::{accumulate, check_non_negative, Counts};

fn default_true() -> bool {
    true
}

/// Multinomial Naive Bayes classifier parameters (unfitted state).
///
/// Suitable for count-based or TF-IDF feature representations where all
/// feature values must be non-negative.  Uses Laplace / Lidstone (additive)
/// smoothing controlled by the `alpha` parameter. Mirrors sklearn's
/// `MultinomialNB`.
///
/// Uses the type-state pattern: call [`Fit::fit`] to produce a
/// [`FittedMultinomialNB`] that can make predictions.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MultinomialNB {
    /// Additive (Laplace/Lidstone) smoothing parameter (>= 0).
    pub alpha: f64,
    /// If `false` and `alpha < 1e-10`, alpha is clamped to `1e-10` (sklearn's
    /// `force_alpha`). Default `true`: alpha is used as given.
    #[serde(default = "default_true")]
    pub force_alpha: bool,
    /// Whether to learn class priors from the data (`false`: uniform prior).
    #[serde(default = "default_true")]
    pub fit_prior: bool,
    /// Fixed class prior probabilities (sorted class order); overrides
    /// `fit_prior` when set.
    #[serde(default)]
    pub class_prior: Option<Vec<f64>>,
}

impl MultinomialNB {
    /// Create a new `MultinomialNB` with default parameters (`alpha = 1.0`).
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
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

    fn finish<F: Float>(&self, counts: Counts<F>) -> Result<FittedMultinomialNB<F>> {
        let alpha = F::from_f64(check_alpha(self.alpha, self.force_alpha)?).unwrap();
        let log_class_prior = class_log_prior(
            &counts.class_count,
            self.fit_prior,
            self.class_prior.as_deref(),
        )?;
        let smoothed = counts.feature_count.mapv(|c| c + alpha);
        let mut log_feature_prob = smoothed.clone();
        for (mut row, srow) in log_feature_prob.rows_mut().into_iter().zip(smoothed.rows()) {
            let log_total = srow.sum().ln();
            row.mapv_inplace(|v| v.ln() - log_total);
        }
        Ok(FittedMultinomialNB {
            class_labels: counts.classes,
            log_class_prior: log_class_prior.to_vec(),
            log_feature_prob,
            class_count: counts.class_count,
            feature_count: counts.feature_count,
        })
    }

    /// Incremental fit with optional per-sample weights (sklearn's
    /// `partial_fit(X, y, classes, sample_weight)`).
    pub fn partial_fit_weighted<'a, 'b, F: Float>(
        &self,
        state: Option<FittedMultinomialNB<F>>,
        x: impl AsArray<'a, F, Ix2>,
        y: impl AsArray<'b, F, Ix1>,
        classes: Option<&[F]>,
        sample_weight: Option<ArrayView1<F>>,
    ) -> Result<FittedMultinomialNB<F>> {
        let x: ArrayView2<F> = x.into();
        let y: ArrayView1<F> = y.into();
        check_alpha(self.alpha, self.force_alpha)?;
        check_non_negative(x, "MultinomialNB")?;
        let prev = state
            .map(|s| Counts::from_parts(s.class_labels, s.class_count, s.feature_count))
            .transpose()?;
        let counts = accumulate(prev, x, y, classes, sample_weight)?;
        self.finish(counts)
    }
}

impl Default for MultinomialNB {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Multinomial Naive Bayes classifier.
///
/// Stores log class priors and log feature probabilities, plus the raw
/// (weighted) counts needed to continue training with `partial_fit`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedMultinomialNB<F: Float> {
    /// Unique sorted class labels.
    class_labels: Vec<F>,
    /// Log prior probability of each class, shape `(n_classes,)`.
    log_class_prior: Vec<F>,
    /// Log probability of each feature per class, shape `(n_classes, n_features)`.
    log_feature_prob: Array2<F>,
    /// Weighted number of samples per class.
    #[serde(default)]
    class_count: Array1<F>,
    /// Weighted feature totals per class, shape `(n_classes, n_features)`.
    #[serde(default)]
    feature_count: Array2<F>,
}

impl<F: Float> FittedMultinomialNB<F> {
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

    /// Weighted feature totals per class (sklearn's `feature_count_`).
    pub fn feature_count(&self) -> &Array2<F> {
        &self.feature_count
    }

    /// Joint log-likelihood `X · log_prob_ᵀ + log_prior`, shape
    /// `(n_samples, n_classes)`.
    pub fn joint_log_likelihood<'a>(&self, x: impl AsArray<'a, F, Ix2>) -> Result<Array2<F>> {
        let x: ArrayView2<F> = x.into();
        check_n_features(x, self.log_feature_prob.ncols())?;
        let prior = Array1::from(self.log_class_prior.clone());
        Ok(x.dot(&self.log_feature_prob.t()) + &prior)
    }

    /// Predict class probabilities for each sample in `x`.
    ///
    /// Returns an array of shape `(n_samples, n_classes)` where each row sums
    /// to 1.
    pub fn predict_proba<'a>(&self, x: impl AsArray<'a, F, Ix2>) -> Result<Array2<F>> {
        let x: ArrayView2<F> = x.into();
        Ok(proba_from_jll(self.joint_log_likelihood(x)?))
    }
}

impl<F: Float> FitWeighted<F> for MultinomialNB {
    type Fitted = FittedMultinomialNB<F>;

    fn fit_weighted_view(
        &self,
        x: ArrayView2<F>,
        y: ArrayView1<F>,
        sample_weight: Option<ArrayView1<F>>,
    ) -> Result<Self::Fitted> {
        self.partial_fit_weighted(None, x, y, None, sample_weight)
    }
}

impl<F: Float> Fit<F> for MultinomialNB {
    type Fitted = FittedMultinomialNB<F>;

    fn fit_view(&self, x: ArrayView2<F>, y: ArrayView1<F>) -> Result<Self::Fitted> {
        self.partial_fit_weighted(None, x, y, None, None)
    }
}

impl<F: Float> PartialFit<F> for MultinomialNB {
    type Fitted = FittedMultinomialNB<F>;

    fn partial_fit_view(
        &self,
        state: Option<Self::Fitted>,
        x: ArrayView2<F>,
        y: ArrayView1<F>,
        classes: Option<&[F]>,
    ) -> Result<Self::Fitted> {
        self.partial_fit_weighted(state, x, y, classes, None)
    }
}

impl<F: Float> Predict<F> for FittedMultinomialNB<F> {
    fn predict_view(&self, x: ArrayView2<F>) -> Result<Array1<F>> {
        let jll = self.joint_log_likelihood(x)?;
        Ok(argmax_labels(&jll, &self.class_labels))
    }
}

impl<F: Float> PredictProba<F> for FittedMultinomialNB<F> {
    fn predict_proba_view(&self, x: ArrayView2<F>) -> Result<Array2<F>> {
        Self::predict_proba(self, x)
    }
}

impl<F: Float> PredictLogProba<F> for FittedMultinomialNB<F> {
    fn predict_log_proba_view(&self, x: ArrayView2<F>) -> Result<Array2<F>> {
        Ok(log_proba_from_jll(self.joint_log_likelihood(x)?))
    }
}

impl<F: Float> anofox_ml_core::ClassifierScore<F> for FittedMultinomialNB<F> {}

#[cfg(test)]
mod tests {
    use super::*;
    use anofox_ml_core::RustMlError;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_basic_classification() {
        // Simple count data: class 0 has high counts in feature 0,
        // class 1 has high counts in feature 1.
        let x_train = array![
            [5.0, 0.0],
            [4.0, 1.0],
            [6.0, 0.0],
            [0.0, 5.0],
            [1.0, 4.0],
            [0.0, 6.0]
        ];
        let y_train = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let nb = MultinomialNB::new();
        let fitted: FittedMultinomialNB<f64> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        let x_test = array![[5.0, 0.0], [0.0, 5.0]];
        let preds = fitted.predict(&x_test).unwrap();

        assert_abs_diff_eq!(preds[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_predict_proba_sums_to_one() {
        let x_train = array![[3.0, 1.0], [4.0, 0.0], [1.0, 3.0], [0.0, 4.0]];
        let y_train = array![0.0, 0.0, 1.0, 1.0];

        let nb = MultinomialNB::new();
        let fitted: FittedMultinomialNB<f64> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        let x_test = array![[2.0, 2.0], [5.0, 0.0], [0.0, 5.0]];
        let proba = fitted.predict_proba(x_test.view()).unwrap();

        // Each row should sum to 1.
        for i in 0..proba.nrows() {
            let row_sum: f64 = proba.row(i).iter().sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }

        // Probabilities should be in [0, 1].
        for &p in proba.iter() {
            assert!(p >= 0.0 && p <= 1.0, "probability {} out of range", p);
        }
    }

    #[test]
    fn test_alpha_smoothing_effect() {
        // With very small alpha, probabilities should be more extreme.
        let x_train = array![[10.0, 0.0], [0.0, 10.0]];
        let y_train = array![0.0, 1.0];

        let nb_small = MultinomialNB::new().with_alpha(1e-10);
        let fitted_small: FittedMultinomialNB<f64> =
            Fit::fit(&nb_small, &x_train, &y_train).unwrap();

        let nb_large = MultinomialNB::new().with_alpha(100.0);
        let fitted_large: FittedMultinomialNB<f64> =
            Fit::fit(&nb_large, &x_train, &y_train).unwrap();

        let x_test = array![[10.0, 0.0]];
        let proba_small = fitted_small.predict_proba(x_test.view()).unwrap();
        let proba_large = fitted_large.predict_proba(x_test.view()).unwrap();

        // With small alpha, class 0 probability should be very high.
        assert!(
            proba_small[[0, 0]] > proba_large[[0, 0]],
            "smaller alpha should give more extreme probabilities"
        );
    }

    #[test]
    fn test_negative_feature_validation() {
        let x_train = array![[1.0, -1.0], [2.0, 3.0]];
        let y_train = array![0.0, 1.0];

        let nb = MultinomialNB::new();
        let result: Result<FittedMultinomialNB<f64>> = Fit::fit(&nb, &x_train, &y_train);
        assert!(result.is_err());

        match result {
            Err(RustMlError::InvalidParameter(msg)) => {
                assert!(
                    msg.contains("non-negative"),
                    "error should mention non-negative: {}",
                    msg
                );
            }
            other => panic!("expected InvalidParameter error, got {:?}", other),
        }
    }

    #[test]
    fn test_multiclass() {
        // Three classes with distinctive feature patterns.
        let x_train = array![
            [5.0, 0.0, 0.0],
            [4.0, 1.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 4.0, 1.0],
            [0.0, 0.0, 5.0],
            [1.0, 0.0, 4.0]
        ];
        let y_train = array![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];

        let nb = MultinomialNB::new();
        let fitted: FittedMultinomialNB<f64> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        assert_eq!(fitted.classes().len(), 3);

        let x_test = array![[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]];
        let preds = fitted.predict(&x_test).unwrap();

        assert_abs_diff_eq!(preds[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[1], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[2], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_shape_errors() {
        // Fit with mismatched X rows and y length.
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0, 0.0];

        let nb = MultinomialNB::new();
        let result: Result<FittedMultinomialNB<f64>> = Fit::fit(&nb, &x, &y);
        assert!(result.is_err());
        match result {
            Err(RustMlError::ShapeMismatch(_)) => {}
            other => panic!("expected ShapeMismatch error, got {:?}", other),
        }

        // Predict with wrong number of features.
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let y_train = array![0.0, 1.0];
        let fitted: FittedMultinomialNB<f64> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        let x_test = array![[1.0, 2.0, 3.0]];
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

        let nb = MultinomialNB::new();
        let result: Result<FittedMultinomialNB<f64>> = Fit::fit(&nb, &x, &y);
        assert!(result.is_err());
        match result {
            Err(RustMlError::EmptyInput(_)) => {}
            other => panic!("expected EmptyInput error, got {:?}", other),
        }
    }

    #[test]
    fn test_f32_support() {
        let x_train: Array2<f32> = array![[5.0f32, 0.0], [4.0, 1.0], [0.0, 5.0], [1.0, 4.0]];
        let y_train: Array1<f32> = array![0.0f32, 0.0, 1.0, 1.0];

        let nb = MultinomialNB::new();
        let fitted: FittedMultinomialNB<f32> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        let x_test: Array2<f32> = array![[5.0f32, 0.0], [0.0, 5.0]];
        let preds = fitted.predict(&x_test).unwrap();

        assert_abs_diff_eq!(preds[0], 0.0f32, epsilon = 1e-5);
        assert_abs_diff_eq!(preds[1], 1.0f32, epsilon = 1e-5);
    }

    #[test]
    fn test_predict_proba_shape_error() {
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let y_train = array![0.0, 1.0];

        let nb = MultinomialNB::new();
        let fitted: FittedMultinomialNB<f64> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        let x_test = array![[1.0]];
        let result = fitted.predict_proba(x_test.view());
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
    use anofox_ml_core::RustMlError;
    use ndarray::array;

    fn data() -> (Array2<f64>, Array1<f64>, Array1<f64>, Array2<f64>) {
        let x = array![
            [3.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 4.0],
            [0.0, 4.0, 2.0, 0.0],
            [1.0, 3.0, 3.0, 1.0],
            [0.0, 0.0, 5.0, 1.0],
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 4.0, 2.0]
        ];
        let y = array![0.0, 0.0, 1.0, 1.0, 2.0, 0.0, 2.0];
        let w = array![1.0, 0.5, 2.0, 1.0, 1.5, 1.0, 0.25];
        let t = array![
            [1.0, 0.0, 0.0, 3.0],
            [0.0, 2.0, 2.0, 1.0],
            [2.0, 2.0, 2.0, 2.0]
        ];
        (x, y, w, t)
    }

    #[test]
    fn sample_weight_and_alpha() {
        let (x, y, w, t) = data();
        let m: FittedMultinomialNB<f64> = MultinomialNB::new()
            .with_alpha(0.5)
            .fit_weighted(&x, &y, Some(&w))
            .unwrap();
        assert_close2(
            &m.predict_proba(t.view()).unwrap(),
            &[
                &[
                    0.9709058805672579,
                    0.0024094171668488915,
                    0.02668470226589333,
                ],
                &[0.0116355354668712, 0.9129108750365533, 0.07545358949657562],
                &[0.7138033400637324, 0.24526059881193704, 0.0409360611243308],
            ],
            1e-12,
            "proba",
        );
        assert_close2(
            &PredictLogProba::predict_log_proba(&m, &t).unwrap(),
            &[
                &[-0.02952574580363887, -6.028370400209244, -3.623664826616789],
                &[
                    -4.453691461209409,
                    -0.09111702085575235,
                    -2.5842375203926125,
                ],
                &[
                    -0.3371477886658951,
                    -1.4054339651119037,
                    -3.1957439143123256,
                ],
            ],
            1e-12,
            "log_proba",
        );
    }

    #[test]
    fn fit_prior_false_and_class_prior() {
        let (x, y, _, t) = data();
        let m: FittedMultinomialNB<f64> = MultinomialNB::new()
            .with_fit_prior(false)
            .fit(&x, &y)
            .unwrap();
        assert_close2(
            &m.predict_proba(t.view()).unwrap(),
            &[
                &[0.9143402300632678, 0.0095177522151925, 0.07614201772153988],
                &[
                    0.014664993219826554,
                    0.7313826854450772,
                    0.25395232133509654,
                ],
                &[0.3618827427585317, 0.37659379115889957, 0.2615234660825695],
            ],
            1e-12,
            "uniform prior",
        );
        let m: FittedMultinomialNB<f64> = MultinomialNB::new()
            .with_class_prior(Some(vec![0.1, 0.3, 0.6]))
            .fit(&x, &y)
            .unwrap();
        assert_close2(
            &m.predict_proba(t.view()).unwrap(),
            &[
                &[0.6532188667790225, 0.020398890189469304, 0.3263822430315087],
                &[0.003928971795239665, 0.5878451969733006, 0.4082258312314593],
                &[0.1182312342917941, 0.36911250657552735, 0.5126562591326775],
            ],
            1e-12,
            "class prior",
        );
        let bad: Result<FittedMultinomialNB<f64>> = MultinomialNB::new()
            .with_class_prior(Some(vec![0.5, 0.5]))
            .fit(&x, &y);
        assert!(bad.is_err());
    }

    #[test]
    fn weighted_partial_fit() {
        let (x, y, w, t) = data();
        let nb = MultinomialNB::new();
        let m = nb
            .partial_fit_weighted(
                None,
                &x.slice(ndarray::s![..3, ..]).to_owned(),
                &y.slice(ndarray::s![..3]).to_owned(),
                Some(&[0.0, 1.0, 2.0]),
                Some(w.slice(ndarray::s![..3])),
            )
            .unwrap();
        let m = nb
            .partial_fit_weighted(
                Some(m),
                &x.slice(ndarray::s![3.., ..]).to_owned(),
                &y.slice(ndarray::s![3..]).to_owned(),
                None,
                Some(w.slice(ndarray::s![3..])),
            )
            .unwrap();
        let efc = array![
            [8.0, 1.5, 1.0, 4.0],
            [1.0, 11.0, 7.0, 1.0],
            [0.25, 0.25, 8.5, 2.0]
        ];
        assert!((m.feature_count() - &efc).iter().all(|d| d.abs() < 1e-12));
        assert_close2(
            &m.predict_proba(t.view()).unwrap(),
            &[
                &[0.9482134859856537, 0.00571342828931442, 0.04607308572503148],
                &[0.01788396035371436, 0.861211512331716, 0.12090452731456926],
                &[0.6319241859333893, 0.2753135140619913, 0.09276230000462023],
            ],
            1e-12,
            "partial_fit proba",
        );
        // Labels outside the declared classes are rejected.
        let r = nb.partial_fit(Some(m), &x, &y.mapv(|v| v + 1.0), None);
        assert!(r.is_err());
    }

    #[test]
    fn alpha_validation_and_force_alpha() {
        let (x, y, _, t) = data();
        let neg: Result<FittedMultinomialNB<f64>> =
            MultinomialNB::new().with_alpha(-0.1).fit(&x, &y);
        assert!(matches!(neg, Err(RustMlError::InvalidParameter(_))));
        // force_alpha = false clamps alpha = 0 to 1e-10 (sklearn warns).
        let m: FittedMultinomialNB<f64> = MultinomialNB::new()
            .with_alpha(0.0)
            .with_force_alpha(false)
            .fit(&x, &y)
            .unwrap();
        assert_close2(
            &m.predict_proba(t.view()).unwrap(),
            &[
                &[
                    0.9744318181788123,
                    0.0009131493509327974,
                    0.024655032470254523,
                ],
                &[0.00693159631803491, 0.8286844649134135, 0.1643839387685514],
                &[0.5452702615452873, 0.28507877666468817, 0.1696509617900249],
            ],
            1e-10,
            "alpha clamp",
        );
    }
}
