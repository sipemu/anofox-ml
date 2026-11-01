use anofox_ml_core::{
    Fit, FitWeighted, Float, PartialFit, Predict, PredictLogProba, PredictProba, Result,
    RustMlError,
};
use ndarray::{Array1, Array2, Axis};

use crate::common::{
    argmax_labels, check_n_features, check_partial_fit_classes, check_priors, check_xy,
    encode_labels, log_proba_from_jll, proba_from_jll, sorted_unique,
};

/// Gaussian Naive Bayes classifier parameters (unfitted state).
///
/// Mirrors sklearn's `GaussianNB`: per-class feature means and variances,
/// with `epsilon = var_smoothing * max_j Var(X_j)` added to every variance.
/// As in sklearn 1.8, `epsilon` is computed from the batch passed to each
/// `fit` / `partial_fit` call.
///
/// Uses the type-state pattern: call [`Fit::fit`] to produce a
/// [`FittedGaussianNB`] that can make predictions.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GaussianNB {
    /// Portion of the largest variance of all features that is added to
    /// variances for calculation stability.
    pub var_smoothing: f64,
    /// Prior probabilities of the classes (sorted class order). When `None`
    /// the priors are the (weighted) empirical class frequencies.
    #[serde(default)]
    pub priors: Option<Vec<f64>>,
}

impl GaussianNB {
    /// Create a new `GaussianNB` with default parameters.
    pub fn new() -> Self {
        Self {
            var_smoothing: 1e-9,
            priors: None,
        }
    }

    /// Set the variance smoothing factor (relative to the largest feature
    /// variance) added to variances for calculation stability.
    pub fn with_var_smoothing(mut self, var_smoothing: f64) -> Self {
        self.var_smoothing = var_smoothing;
        self
    }

    /// Fix the class priors instead of estimating them from the data.
    pub fn with_priors(mut self, priors: Option<Vec<f64>>) -> Self {
        self.priors = priors;
        self
    }

    fn validate(&self) -> Result<()> {
        if !self.var_smoothing.is_finite() || self.var_smoothing < 0.0 {
            return Err(RustMlError::InvalidParameter(
                "var_smoothing must be a finite value >= 0".into(),
            ));
        }
        Ok(())
    }

    /// Incremental fit with optional per-sample weights (sklearn's
    /// `partial_fit(X, y, classes, sample_weight)`).
    ///
    /// On the first call (`state = None`) the class set is taken from
    /// `classes` (or from `y` when omitted). `epsilon` is recomputed from each
    /// batch exactly like sklearn. Labels outside the class set are an error.
    pub fn partial_fit_weighted<F: Float>(
        &self,
        state: Option<FittedGaussianNB<F>>,
        x: &Array2<F>,
        y: &Array1<F>,
        classes: Option<&[F]>,
        sample_weight: Option<&Array1<F>>,
    ) -> Result<FittedGaussianNB<F>> {
        self.validate()?;
        check_xy(x, y, sample_weight)?;
        let n_features = x.ncols();
        // epsilon = var_smoothing * max_j Var(X_j) of *this batch* (unweighted).
        // Like sklearn 1.8, it is recomputed on every call and the new value
        // is removed from / re-added to the stored variances.
        let max_var = x
            .var_axis(Axis(0), F::zero())
            .iter()
            .copied()
            .fold(F::zero(), |a, b| if b > a { b } else { a });
        let eps = F::from_f64(self.var_smoothing).unwrap() * max_var;

        let mut model = match state {
            Some(mut s) => {
                check_n_features(x, s.theta.ncols())?;
                check_partial_fit_classes(&s.class_labels, classes)?;
                if s.class_count.len() != s.class_labels.len() {
                    return Err(RustMlError::InvalidParameter(
                        "model state has no class counts; cannot continue partial_fit".into(),
                    ));
                }
                s.epsilon = Some(eps);
                s.sigma.mapv_inplace(|v| v - eps);
                s
            }
            None => {
                let class_labels = match classes {
                    Some(c) => sorted_unique(c),
                    None => sorted_unique(&y.to_vec()),
                };
                let k = class_labels.len();
                FittedGaussianNB {
                    class_labels,
                    class_prior: vec![F::zero(); k],
                    theta: Array2::zeros((k, n_features)),
                    sigma: Array2::zeros((k, n_features)),
                    class_count: vec![F::zero(); k],
                    epsilon: Some(eps),
                    jll_const: Vec::new(),
                }
            }
        };
        let k = model.class_labels.len();
        if let Some(p) = &self.priors {
            check_priors(p, k)?;
        }

        let idx = encode_labels(y, &model.class_labels)?;
        for ci in 0..k {
            let rows: Vec<usize> = (0..idx.len()).filter(|&i| idx[i] == ci).collect();
            if rows.is_empty() {
                continue;
            }
            let xi = x.select(Axis(0), &rows);
            let swi = sample_weight.map(|w| Array1::from_iter(rows.iter().map(|&i| w[i])));
            let n_past = model.class_count[ci];
            let mut mu = model.theta.row(ci).to_owned();
            let mut var = model.sigma.row(ci).to_owned();
            let n_new = update_mean_variance(n_past, &mut mu, &mut var, &xi, swi.as_ref());
            model.theta.row_mut(ci).assign(&mu);
            model.sigma.row_mut(ci).assign(&var);
            model.class_count[ci] = n_past + n_new;
        }

        let eps = model.epsilon.unwrap_or_else(F::zero);
        model.sigma.mapv_inplace(|v| v + eps);
        model.class_prior = match &self.priors {
            Some(p) => p.iter().map(|&v| F::from_f64(v).unwrap()).collect(),
            None => {
                let total = model.class_count.iter().fold(F::zero(), |a, &b| a + b);
                model.class_count.iter().map(|&c| c / total).collect()
            }
        };
        model.refresh_constants();
        Ok(model)
    }
}

impl Default for GaussianNB {
    fn default() -> Self {
        Self::new()
    }
}

/// sklearn's `GaussianNB._update_mean_variance`: combine the running
/// `(n_past, mu, var)` with a new batch. Returns the (weighted) batch size.
fn update_mean_variance<F: Float>(
    n_past: F,
    mu: &mut Array1<F>,
    var: &mut Array1<F>,
    x: &Array2<F>,
    sample_weight: Option<&Array1<F>>,
) -> F {
    let (n_new, new_mu, new_var) = match sample_weight {
        Some(w) => {
            let n_new = w.sum();
            if n_new.abs() <= F::from_f64(1e-8).unwrap() {
                return F::zero();
            }
            let new_mu = w.dot(x) / n_new;
            let sq = (x - &new_mu).mapv(|v| v * v);
            let new_var = w.dot(&sq) / n_new;
            (n_new, new_mu, new_var)
        }
        None => {
            let n_new = F::from_usize(x.nrows()).unwrap();
            let new_mu = x.mean_axis(Axis(0)).unwrap();
            let new_var = x.var_axis(Axis(0), F::zero());
            (n_new, new_mu, new_var)
        }
    };
    if n_past == F::zero() {
        *mu = new_mu;
        *var = new_var;
        return n_new;
    }
    let n_total = n_past + n_new;
    let total_mu = (&new_mu * n_new + &*mu * n_past) / n_total;
    let d = &*mu - &new_mu;
    let total_ssd =
        &*var * n_past + &new_var * n_new + d.mapv(|v| v * v) * (n_new * n_past / n_total);
    *mu = total_mu;
    *var = total_ssd / n_total;
    n_new
}

/// Fitted Gaussian Naive Bayes classifier.
///
/// Stores learned class priors, per-class feature means (theta), and
/// per-class feature variances (sigma, sklearn's `var_`) needed for
/// prediction.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedGaussianNB<F: Float> {
    /// Unique sorted class labels.
    class_labels: Vec<F>,
    /// Prior probability of each class.
    class_prior: Vec<F>,
    /// Mean of each feature per class, shape `(n_classes, n_features)`.
    theta: Array2<F>,
    /// Variance of each feature per class + epsilon, shape `(n_classes, n_features)`.
    sigma: Array2<F>,
    /// Per-class cumulative (weighted) observation count.
    #[serde(default = "default_class_count")]
    class_count: Vec<F>,
    /// Absolute additive variance smoothing of the last fit / partial_fit batch.
    #[serde(default)]
    epsilon: Option<F>,
    /// Cached `ln(prior_c) - ½ Σ_j ln(2π σ²_cj)` per class (recomputed on
    /// the fly for models deserialised without it).
    #[serde(default)]
    jll_const: Vec<F>,
}

fn default_class_count<F: Float>() -> Vec<F> {
    Vec::new()
}

impl<F: Float> FittedGaussianNB<F> {
    /// Returns the prior probability of each class.
    pub fn class_prior(&self) -> &[F] {
        &self.class_prior
    }

    /// Returns the per-class feature means, shape `(n_classes, n_features)`.
    pub fn theta(&self) -> &Array2<F> {
        &self.theta
    }

    /// Returns the per-class feature variances (with smoothing), shape `(n_classes, n_features)`.
    pub fn sigma(&self) -> &Array2<F> {
        &self.sigma
    }

    /// Alias of [`sigma`](Self::sigma) using sklearn's attribute name `var_`.
    pub fn var(&self) -> &Array2<F> {
        &self.sigma
    }

    /// Returns the unique sorted class labels.
    pub fn class_labels(&self) -> &[F] {
        &self.class_labels
    }

    /// Per-class (weighted) number of training samples seen so far.
    pub fn class_count(&self) -> &[F] {
        &self.class_count
    }

    /// Absolute variance smoothing `var_smoothing * max_j Var(X_j)` of the last
    /// batch (sklearn's `epsilon_`). `None` for models serialised by older
    /// versions.
    pub fn epsilon(&self) -> Option<F> {
        self.epsilon
    }

    fn compute_constants(&self) -> Vec<F> {
        let two_pi = F::from_f64(2.0 * std::f64::consts::PI).unwrap();
        let half = F::from_f64(0.5).unwrap();
        self.sigma
            .rows()
            .into_iter()
            .zip(self.class_prior.iter())
            .map(|(var, &p)| {
                p.ln() - half * var.iter().fold(F::zero(), |a, &v| a + (two_pi * v).ln())
            })
            .collect()
    }

    fn refresh_constants(&mut self) {
        self.jll_const = self.compute_constants();
    }

    /// Unnormalised joint log-likelihood `ln P(c) + ln P(x | c)`, shape
    /// `(n_samples, n_classes)` (sklearn's `_joint_log_likelihood`).
    ///
    /// The per-class constants are precomputed at fit time; the quadratic
    /// term is evaluated per class with whole-array operations. (The GEMM
    /// expansion `x²/σ² − 2xμ/σ² + μ²/σ²` is deliberately avoided: it
    /// suffers catastrophic cancellation when `|x − μ| ≪ |x|`.)
    pub fn joint_log_likelihood(&self, x: &Array2<F>) -> Result<Array2<F>> {
        check_n_features(x, self.theta.ncols())?;
        let consts_owned;
        let consts = if self.jll_const.len() == self.class_labels.len() {
            &self.jll_const
        } else {
            consts_owned = self.compute_constants();
            &consts_owned
        };
        let half = F::from_f64(0.5).unwrap();
        let mut jll = Array2::<F>::zeros((x.nrows(), self.class_labels.len()));
        for (ci, &c) in consts.iter().enumerate() {
            let mu = self.theta.row(ci);
            let inv_var = self.sigma.row(ci).mapv(|v| F::one() / v);
            let mut z = x - &mu;
            z.mapv_inplace(|v| v * v);
            let quad = z.dot(&inv_var);
            jll.column_mut(ci).assign(&quad.mapv(|q| c - half * q));
        }
        Ok(jll)
    }
}

impl<F: Float> FitWeighted<F> for GaussianNB {
    type Fitted = FittedGaussianNB<F>;

    fn fit_weighted(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        sample_weight: Option<&Array1<F>>,
    ) -> Result<Self::Fitted> {
        check_xy(x, y, sample_weight)?;
        let classes = sorted_unique(&y.to_vec());
        self.partial_fit_weighted(None, x, y, Some(&classes), sample_weight)
    }
}

impl<F: Float> Fit<F> for GaussianNB {
    type Fitted = FittedGaussianNB<F>;

    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Self::Fitted> {
        self.fit_weighted(x, y, None)
    }
}

impl<F: Float> PartialFit<F> for GaussianNB {
    type Fitted = FittedGaussianNB<F>;

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

impl<F: Float> Predict<F> for FittedGaussianNB<F> {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        let jll = self.joint_log_likelihood(x)?;
        Ok(argmax_labels(&jll, &self.class_labels))
    }
}

impl<F: Float> PredictProba<F> for FittedGaussianNB<F> {
    fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>> {
        Ok(proba_from_jll(self.joint_log_likelihood(x)?))
    }
}

impl<F: Float> PredictLogProba<F> for FittedGaussianNB<F> {
    fn predict_log_proba(&self, x: &Array2<F>) -> Result<Array2<F>> {
        Ok(log_proba_from_jll(self.joint_log_likelihood(x)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_partial_fit_two_halves_matches_one_shot() {
        // GaussianNB partial_fit on two halves should produce identical
        // theta/sigma/prior to one-shot fit on the union.
        let x = array![
            [1.0_f64, 2.0],
            [1.5, 2.5],
            [0.8, 1.7],
            [1.2, 1.9],
            [5.0, 5.0],
            [5.5, 5.5],
            [4.8, 4.9],
            [5.1, 5.3]
        ];
        let y = array![0.0_f64, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        // epsilon depends on the batch (sklearn semantics), so disable the
        // smoothing for an exact one-shot vs. streamed comparison.
        let nb = GaussianNB::new().with_var_smoothing(0.0);
        let one_shot: FittedGaussianNB<f64> = Fit::fit(&nb, &x, &y).unwrap();

        let classes = [0.0_f64, 1.0];
        let h1 = x.slice(ndarray::s![0..4, ..]).to_owned();
        let y1 = y.slice(ndarray::s![0..4]).to_owned();
        let h2 = x.slice(ndarray::s![4..8, ..]).to_owned();
        let y2 = y.slice(ndarray::s![4..8]).to_owned();
        let s1 = PartialFit::partial_fit(&nb, None, &h1, &y1, Some(&classes)).unwrap();
        let s2 = PartialFit::partial_fit(&nb, Some(s1), &h2, &y2, Some(&classes)).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(one_shot.theta[[i, j]], s2.theta[[i, j]], epsilon = 1e-9);
                assert_abs_diff_eq!(one_shot.sigma[[i, j]], s2.sigma[[i, j]], epsilon = 1e-12);
            }
            assert_abs_diff_eq!(one_shot.class_prior[i], s2.class_prior[i], epsilon = 1e-9);
        }
    }

    #[test]
    fn test_two_class_well_separated() {
        // Two well-separated Gaussian clusters.
        let x_train = array![
            [1.0, 1.0],
            [1.1, 1.1],
            [0.9, 0.9],
            [1.0, 1.2],
            [10.0, 10.0],
            [10.1, 10.1],
            [9.9, 9.9],
            [10.0, 10.2]
        ];
        let y_train = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let nb = GaussianNB::default();
        let fitted: FittedGaussianNB<f64> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        // Verify class labels.
        assert_eq!(fitted.class_labels(), &[0.0, 1.0]);

        // Verify equal priors.
        assert_abs_diff_eq!(fitted.class_prior()[0], 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(fitted.class_prior()[1], 0.5, epsilon = 1e-10);

        // Predict points near each cluster center.
        let x_test = array![[1.0, 1.0], [10.0, 10.0], [0.5, 0.5], [9.5, 9.5]];
        let preds = fitted.predict(&x_test).unwrap();

        assert_abs_diff_eq!(preds[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[1], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[2], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[3], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_predictions_known_data() {
        // Simple 1-D data: class 0 centred at 0, class 1 centred at 5.
        let x_train = array![[0.0], [0.5], [-0.5], [5.0], [5.5], [4.5]];
        let y_train = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let nb = GaussianNB::default();
        let fitted: FittedGaussianNB<f64> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        // Means should be approximately 0.0 and 5.0.
        assert_abs_diff_eq!(fitted.theta()[[0, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(fitted.theta()[[1, 0]], 5.0, epsilon = 1e-10);

        // A point at 1.0 should be class 0; at 4.0 should be class 1.
        let x_test = array![[1.0], [4.0]];
        let preds = fitted.predict(&x_test).unwrap();
        assert_abs_diff_eq!(preds[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_empty_input_error() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<f64>::zeros(0);

        let nb = GaussianNB::default();
        let result: Result<FittedGaussianNB<f64>> = Fit::fit(&nb, &x, &y);
        assert!(result.is_err());

        match result {
            Err(RustMlError::EmptyInput(_)) => {}
            other => panic!("expected EmptyInput error, got {:?}", other),
        }
    }

    #[test]
    fn test_shape_mismatch_fit() {
        // X has 2 rows, y has 3 elements.
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0, 0.0];

        let nb = GaussianNB::default();
        let result: Result<FittedGaussianNB<f64>> = Fit::fit(&nb, &x, &y);
        assert!(result.is_err());

        match result {
            Err(RustMlError::ShapeMismatch(_)) => {}
            other => panic!("expected ShapeMismatch error, got {:?}", other),
        }
    }

    #[test]
    fn test_shape_mismatch_predict() {
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let y_train = array![0.0, 1.0];

        let nb = GaussianNB::default();
        let fitted: FittedGaussianNB<f64> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        // Predict with wrong number of features (3 instead of 2).
        let x_test = array![[1.0, 2.0, 3.0]];
        let result = fitted.predict(&x_test);
        assert!(result.is_err());

        match result {
            Err(RustMlError::ShapeMismatch(_)) => {}
            other => panic!("expected ShapeMismatch error, got {:?}", other),
        }
    }

    #[test]
    fn test_zero_variance_feature() {
        // One feature column is constant — var_smoothing should prevent NaN.
        let x_train = array![
            [1.0, 5.0],
            [2.0, 5.0],
            [3.0, 5.0],
            [10.0, 5.0],
            [11.0, 5.0],
            [12.0, 5.0]
        ];
        let y_train = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let nb = GaussianNB::default();
        let fitted: FittedGaussianNB<f64> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        // Variance for the second feature should be the smoothing value, not zero.
        assert!(fitted.sigma()[[0, 1]] > 0.0);
        assert!(fitted.sigma()[[1, 1]] > 0.0);

        // Predictions should still work without NaN.
        let x_test = array![[2.0, 5.0], [11.0, 5.0]];
        let preds = fitted.predict(&x_test).unwrap();
        assert_abs_diff_eq!(preds[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[1], 1.0, epsilon = 1e-10);
        // Verify no NaN in predictions.
        for &p in preds.iter() {
            assert!(!p.is_nan(), "prediction should not be NaN");
        }
    }

    #[test]
    fn test_highly_imbalanced_classes() {
        // 50 class-0 samples and 2 class-1 samples.
        let mut x_data = Vec::new();
        let mut y_data = Vec::new();
        for i in 0..50 {
            x_data.push(i as f64 * 0.1);
            x_data.push(i as f64 * 0.1);
            y_data.push(0.0);
        }
        // Class-1 samples far away.
        x_data.push(100.0);
        x_data.push(100.0);
        y_data.push(1.0);
        x_data.push(101.0);
        x_data.push(101.0);
        y_data.push(1.0);

        let x_train = Array2::from_shape_vec((52, 2), x_data).unwrap();
        let y_train = Array1::from_vec(y_data);

        let nb = GaussianNB::default();
        let fitted: FittedGaussianNB<f64> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        // Should not crash, and priors should reflect the imbalance.
        assert_abs_diff_eq!(fitted.class_prior()[0], 50.0 / 52.0, epsilon = 1e-10);
        assert_abs_diff_eq!(fitted.class_prior()[1], 2.0 / 52.0, epsilon = 1e-10);

        // A point near class-1 cluster should still be predicted as class 1.
        let x_test = array![[100.5, 100.5]];
        let preds = fitted.predict(&x_test).unwrap();
        assert_abs_diff_eq!(preds[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_many_classes() {
        // 10 different classes, each with 3 samples.
        let mut x_data = Vec::new();
        let mut y_data = Vec::new();
        for class in 0..10 {
            let center = class as f64 * 100.0;
            for offset in &[-0.1, 0.0, 0.1] {
                x_data.push(center + offset);
                x_data.push(center + offset);
                y_data.push(class as f64);
            }
        }
        let x_train = Array2::from_shape_vec((30, 2), x_data).unwrap();
        let y_train = Array1::from_vec(y_data);

        let nb = GaussianNB::default();
        let fitted: FittedGaussianNB<f64> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        assert_eq!(fitted.class_labels().len(), 10);

        // Predict at each class center; verify predictions are valid labels.
        let mut test_data = Vec::new();
        for class in 0..10 {
            test_data.push(class as f64 * 100.0);
            test_data.push(class as f64 * 100.0);
        }
        let x_test = Array2::from_shape_vec((10, 2), test_data).unwrap();
        let preds = fitted.predict(&x_test).unwrap();

        for &p in preds.iter() {
            assert!(
                fitted.class_labels().contains(&p),
                "prediction {} is not a valid class label",
                p
            );
        }
    }

    #[test]
    fn test_f32_support() {
        let x_train: Array2<f32> = array![[1.0f32, 1.0], [1.1, 1.1], [10.0, 10.0], [10.1, 10.1]];
        let y_train: Array1<f32> = array![0.0f32, 0.0, 1.0, 1.0];

        let nb = GaussianNB::default();
        let fitted: FittedGaussianNB<f32> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        let x_test: Array2<f32> = array![[1.0f32, 1.0], [10.0, 10.0]];
        let preds = fitted.predict(&x_test).unwrap();

        assert_abs_diff_eq!(preds[0], 0.0f32, epsilon = 1e-5);
        assert_abs_diff_eq!(preds[1], 1.0f32, epsilon = 1e-5);
    }

    #[test]
    fn test_shape_mismatch_error() {
        // x has 3 rows but y has 2 elements.
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![0.0, 1.0];

        let nb = GaussianNB::default();
        let result: Result<FittedGaussianNB<f64>> = Fit::fit(&nb, &x, &y);
        assert!(result.is_err());
        match result {
            Err(RustMlError::ShapeMismatch(msg)) => {
                assert!(msg.contains("3"), "error should mention row count");
                assert!(msg.contains("2"), "error should mention y length");
            }
            other => panic!("expected ShapeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn test_predict_wrong_features() {
        let x_train = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let y_train = array![0.0, 1.0];

        let nb = GaussianNB::default();
        let fitted: FittedGaussianNB<f64> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        // Predict with 2 features instead of 3.
        let x_test = array![[1.0, 2.0]];
        let result = fitted.predict(&x_test);
        assert!(result.is_err());
        match result {
            Err(RustMlError::ShapeMismatch(msg)) => {
                assert!(msg.contains("3"), "error should mention expected features");
                assert!(msg.contains("2"), "error should mention actual features");
            }
            other => panic!("expected ShapeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn test_prior_probabilities() {
        // 6 class-0 and 4 class-1 samples.
        let x_train = array![
            [1.0],
            [1.1],
            [1.2],
            [0.9],
            [0.8],
            [1.3],
            [10.0],
            [10.1],
            [10.2],
            [9.9]
        ];
        let y_train = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let nb = GaussianNB::default();
        let fitted: FittedGaussianNB<f64> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        // Priors should reflect 6/10 and 4/10.
        assert_abs_diff_eq!(fitted.class_prior()[0], 0.6, epsilon = 1e-10);
        assert_abs_diff_eq!(fitted.class_prior()[1], 0.4, epsilon = 1e-10);

        // Priors should sum to 1.
        let prior_sum: f64 = fitted.class_prior().iter().sum();
        assert_abs_diff_eq!(prior_sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_single_sample_per_class() {
        // Two classes, one sample each.
        let x_train = array![[0.0, 0.0], [10.0, 10.0]];
        let y_train = array![0.0, 1.0];

        let nb = GaussianNB::default();
        let fitted: FittedGaussianNB<f64> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        // Means should match the single sample.
        assert_abs_diff_eq!(fitted.theta()[[0, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(fitted.theta()[[0, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(fitted.theta()[[1, 0]], 10.0, epsilon = 1e-10);
        assert_abs_diff_eq!(fitted.theta()[[1, 1]], 10.0, epsilon = 1e-10);

        // Should still predict correctly (variance is just smoothing).
        let x_test = array![[0.1, 0.1], [9.9, 9.9]];
        let preds = fitted.predict(&x_test).unwrap();
        assert_abs_diff_eq!(preds[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_three_features_two_classes() {
        // Well-separated 3D data.
        let x_train = array![
            [1.0, 2.0, 3.0],
            [1.1, 2.1, 3.1],
            [0.9, 1.9, 2.9],
            [1.0, 2.0, 3.2],
            [20.0, 21.0, 22.0],
            [20.1, 21.1, 22.1],
            [19.9, 20.9, 21.9],
            [20.0, 21.0, 22.2]
        ];
        let y_train = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let nb = GaussianNB::default();
        let fitted: FittedGaussianNB<f64> = Fit::fit(&nb, &x_train, &y_train).unwrap();

        // Verify theta has the right shape.
        assert_eq!(fitted.theta().shape(), &[2, 3]);
        assert_eq!(fitted.sigma().shape(), &[2, 3]);

        // Predict near each cluster.
        let x_test = array![[1.0, 2.0, 3.0], [20.0, 21.0, 22.0], [10.0, 11.0, 12.0]];
        let preds = fitted.predict(&x_test).unwrap();
        assert_abs_diff_eq!(preds[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(preds[1], 1.0, epsilon = 1e-10);
        // The midpoint could go either way, but it should be a valid label.
        assert!(preds[2] == 0.0 || preds[2] == 1.0);
    }

    mod prop_tests {
        use super::*;
        use proptest::prelude::*;

        /// Generate well-separated two-class Gaussian data.
        ///
        /// Class 0 is centred at -5.0, class 1 at 5.0 on every feature,
        /// with small deterministic perturbations derived from a hash so
        /// that proptest does not rely on external RNGs.
        fn make_separated_data(
            n_per_class: usize,
            n_features: usize,
            seed: u64,
        ) -> (Array2<f64>, Array1<f64>) {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let n_samples = n_per_class * 2;
            let mut x_data = Vec::with_capacity(n_samples * n_features);
            let mut y_data = Vec::with_capacity(n_samples);

            for i in 0..n_samples {
                let class = if i < n_per_class { 0.0 } else { 1.0 };
                let center = if i < n_per_class { -5.0 } else { 5.0 };
                for j in 0..n_features {
                    let mut h = DefaultHasher::new();
                    seed.hash(&mut h);
                    (i as u64).hash(&mut h);
                    (j as u64).hash(&mut h);
                    let bits = h.finish();
                    // Small perturbation in [-0.5, 0.5]
                    let noise = (bits as f64 / u64::MAX as f64) - 0.5;
                    x_data.push(center + noise);
                }
                y_data.push(class);
            }

            let x = Array2::from_shape_vec((n_samples, n_features), x_data).unwrap();
            let y = Array1::from_vec(y_data);
            (x, y)
        }

        proptest! {
            #[test]
            fn well_separated_accuracy_above_80(
                n_per_class in 5..30usize,
                n_features in 1..5usize,
                seed in 0u64..1000,
            ) {
                let (x, y) = make_separated_data(n_per_class, n_features, seed);

                let nb = GaussianNB::default();
                let fitted: FittedGaussianNB<f64> = Fit::fit(&nb, &x, &y).unwrap();
                let preds = fitted.predict(&x).unwrap();

                let correct = preds.iter()
                    .zip(y.iter())
                    .filter(|(&p, &t)| (p - t).abs() < 1e-10)
                    .count();
                let accuracy = correct as f64 / y.len() as f64;

                prop_assert!(
                    accuracy >= 0.8,
                    "accuracy was {:.3} (expected >= 0.8), n_per_class={}, n_features={}, seed={}",
                    accuracy, n_per_class, n_features, seed
                );
            }
        }
    }
}

impl<F: Float> anofox_ml_core::ClassifierScore<F> for FittedGaussianNB<F> {}

/// Reference values generated with scikit-learn 1.8.0.
#[cfg(test)]
mod sklearn_parity {
    use super::*;
    use crate::common::assert_close2;
    use ndarray::array;

    fn three_class() -> (Array2<f64>, Array1<f64>, Array1<f64>, Array2<f64>) {
        let x = array![
            [1.0, 2.0],
            [1.5, 1.0],
            [2.0, 2.5],
            [5.0, 5.5],
            [6.0, 5.0],
            [5.5, 6.5],
            [9.0, 1.0],
            [8.5, 2.0],
            [9.5, 1.5],
            [3.0, 3.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0];
        let w = array![1.0, 2.0, 0.5, 1.0, 1.0, 3.0, 2.0, 1.0, 1.0, 0.5];
        let t = array![[3.0, 3.0], [7.0, 3.0], [2.0, 2.0], [6.0, 4.0]];
        (x, y, w, t)
    }

    #[test]
    fn var_smoothing_is_relative_to_data_variance() {
        // Features at the 1e-4 scale: a raw 1e-9 smoothing (the old
        // behaviour) would dominate the class variances (~7e-10).
        let x = array![
            [1.0e-4, 2.0e-4],
            [1.5e-4, 2.6e-4],
            [0.8e-4, 1.7e-4],
            [1.2e-4, 1.9e-4],
            [5.0e-4, 5.0e-4],
            [5.5e-4, 5.6e-4],
            [4.8e-4, 4.9e-4],
            [5.1e-4, 5.3e-4]
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let t = array![[2.0e-4, 2.5e-4], [3.0e-4, 3.5e-4], [4.5e-4, 4.0e-4]];
        let m: FittedGaussianNB<f64> = GaussianNB::new().fit(&x, &y).unwrap();
        let eps = m.epsilon().unwrap();
        assert!((eps - 4.016093750000001e-17).abs() < 1e-28);
        let var = m.var();
        let ev = [
            [6.68750040160937e-10, 1.1250000401609366e-09],
            [6.500000401609379e-10, 7.500000401609364e-10],
        ];
        for i in 0..2 {
            for j in 0..2 {
                assert!((var[[i, j]] - ev[i][j]).abs() < 1e-22);
            }
        }
        assert_close2(
            &m.predict_proba(&t).unwrap(),
            &[
                &[1.0, 5.754718867860772e-51],
                &[0.9999999706302245, 2.9369776011325412e-08],
                &[8.954324569731785e-40, 1.0],
            ],
            1e-10,
            "proba",
        );
        assert_close2(
            &m.predict_log_proba(&t).unwrap(),
            &[
                &[0.0, -115.68181955178332],
                &[-2.9369775944587673e-08, -17.34329971820945],
                &[-89.911267112026, 0.0],
            ],
            1e-10,
            "log_proba",
        );
    }

    #[test]
    fn priors_and_sample_weight() {
        let (x, y, w, t) = three_class();
        let m: FittedGaussianNB<f64> = GaussianNB::new()
            .with_priors(Some(vec![0.2, 0.5, 0.3]))
            .fit_weighted(&x, &y, Some(&w))
            .unwrap();
        let theta = m.theta();
        let et = [
            [1.4285714285714286, 1.5],
            [5.2727272727272725, 5.7272727272727275],
            [9.0, 1.375],
        ];
        let ev = [
            [0.10204082541653062, 0.3571428662328571],
            [0.6074380256189256, 1.1074380256189258],
            [0.12500000909, 0.17187500909],
        ];
        for i in 0..3 {
            for j in 0..2 {
                assert!((theta[[i, j]] - et[i][j]).abs() < 1e-12);
                assert!((m.var()[[i, j]] - ev[i][j]).abs() < 1e-12);
            }
        }
        assert_eq!(m.class_prior(), &[0.2, 0.5, 0.3]);
        assert_close2(
            &m.predict_proba(&t).unwrap(),
            &[
                &[
                    0.0008255728655607673,
                    0.9991744271344397,
                    9.036737776544631e-63,
                ],
                &[
                    2.167377694295476e-65,
                    0.9999999416417092,
                    5.835829039019624e-08,
                ],
                &[
                    0.9999988551052933,
                    1.1448947064783325e-06,
                    3.3302959112343765e-85,
                ],
                &[5.462819493580652e-48, 1.0, 9.117006180924135e-24],
            ],
            1e-10,
            "proba",
        );
        assert_close2(
            &m.predict_log_proba(&t).unwrap(),
            &[
                &[
                    -7.099433030103675,
                    -0.000825913838516712,
                    -142.86156261477487,
                ],
                &[
                    -148.89451304361646,
                    -5.835829242073487e-08,
                    -16.65666440792591,
                ],
                &[
                    -1.1448953620707414e-06,
                    -13.680197884594167,
                    -194.516671742217,
                ],
                &[-108.82611941647552, 0.0, -53.05190075129864],
            ],
            1e-10,
            "log_proba",
        );
    }

    #[test]
    fn weighted_partial_fit_matches_sklearn() {
        let (x, y, w, t) = three_class();
        let nb = GaussianNB::new();
        let s = ndarray::s![..5, ..];
        let s2 = ndarray::s![5.., ..];
        let m = nb
            .partial_fit_weighted(
                None,
                &x.slice(s).to_owned(),
                &y.slice(ndarray::s![..5]).to_owned(),
                Some(&[0.0, 1.0, 2.0]),
                Some(&w.slice(ndarray::s![..5]).to_owned()),
            )
            .unwrap();
        let m = nb
            .partial_fit_weighted(
                Some(m),
                &x.slice(s2).to_owned(),
                &y.slice(ndarray::s![5..]).to_owned(),
                None,
                Some(&w.slice(ndarray::s![5..]).to_owned()),
            )
            .unwrap();
        assert!((m.epsilon().unwrap() - 6.140000000000001e-09).abs() < 1e-20);
        let ev = [
            [0.10204082036653062, 0.35714286118285715],
            [0.6074380219052892, 1.1074380219052893],
            [0.12500000614, 0.17187500614],
        ];
        for i in 0..3 {
            for j in 0..2 {
                assert!((m.var()[[i, j]] - ev[i][j]).abs() < 1e-12);
            }
        }
        let ep = [0.2692307692307692, 0.4230769230769231, 0.3076923076923077];
        for (a, e) in m.class_prior().iter().zip(ep.iter()) {
            assert!((a - e).abs() < 1e-14);
        }
        assert_close2(
            &m.predict_proba(&t).unwrap(),
            &[
                &[
                    0.0013127701976582422,
                    0.9986872298023417,
                    1.0948242491881492e-62,
                ],
                &[
                    3.4480749097673627e-65,
                    0.9999999292627128,
                    7.073728773800095e-08,
                ],
                &[
                    0.9999992803516058,
                    7.196483943745149e-07,
                    2.5373577904784893e-85,
                ],
                &[8.690804373652381e-48, 1.0, 1.1050903644248977e-23],
            ],
            1e-10,
            "proba",
        );
    }

    #[test]
    fn unknown_label_in_partial_fit_is_an_error() {
        let (x, y, _, _) = three_class();
        let r = GaussianNB::new().partial_fit(None, &x, &y, Some(&[0.0, 1.0]));
        assert!(matches!(r, Err(RustMlError::InvalidParameter(_))));
        let m = GaussianNB::new()
            .partial_fit(None, &x, &y, Some(&[0.0, 1.0, 2.0]))
            .unwrap();
        let r = GaussianNB::new().partial_fit(Some(m), &x, &y, Some(&[0.0, 1.0, 3.0]));
        assert!(r.is_err(), "classes must not change between calls");
    }

    #[test]
    fn invalid_priors_rejected() {
        let (x, y, _, _) = three_class();
        for p in [vec![0.5, 0.5], vec![0.5, 0.6, 0.1], vec![-0.1, 0.6, 0.5]] {
            let r: Result<FittedGaussianNB<f64>> =
                GaussianNB::new().with_priors(Some(p)).fit(&x, &y);
            assert!(r.is_err());
        }
    }

    #[test]
    fn log_proba_f32_is_finite() {
        // The old ln(predict_proba) default produced -inf in f32.
        let x = array![[0.0_f32], [0.1], [10.0], [10.1]];
        let y = array![0.0_f32, 0.0, 1.0, 1.0];
        let m: FittedGaussianNB<f32> = GaussianNB::new().fit(&x, &y).unwrap();
        let lp = m.predict_log_proba(&array![[0.0_f32]]).unwrap();
        assert!(lp.iter().all(|v| v.is_finite()), "{lp:?}");
        assert!(lp[[0, 1]] < -100.0);
    }
}
