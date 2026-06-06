use ndarray::{Array1, Array2};

use crate::error::Result;
use crate::float::Float;

/// Supervised learning: fit on (X, y), produce a fitted model.
///
/// The type-state pattern ensures `predict` is only callable on `Self::Fitted`,
/// making it a compile error to predict with unfitted parameters.
pub trait Fit<F: Float> {
    type Fitted;
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Self::Fitted>;
}

/// Unsupervised learning: fit on X only (e.g., scalers, PCA).
pub trait FitUnsupervised<F: Float> {
    type Fitted;
    fn fit(&self, x: &Array2<F>) -> Result<Self::Fitted>;
}

/// Predict target values from input features.
pub trait Predict<F: Float> {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>>;
}

/// Transform input features (e.g., scaling, encoding).
pub trait Transform<F: Float> {
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>>;
}

/// Reverse a transformation back to the original space.
pub trait InverseTransform<F: Float> {
    fn inverse_transform(&self, x: &Array2<F>) -> Result<Array2<F>>;
}

/// Produce a probability distribution over classes (classifier) or a posterior
/// uncertainty (regressor). Output shape is `(n_samples, n_outputs)` where
/// `n_outputs` is the number of classes for classifiers or 1 for the
/// regressor variants (which return std-dev). The columns sum to 1 for
/// classifiers.
///
/// Mirrors sklearn's `predict_proba` (and partially `predict_std` for
/// probabilistic regressors like Bayesian Ridge / Gaussian Process).
pub trait PredictProba<F: Float> {
    fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>>;
}

/// Supervised fit with optional per-sample weights.
///
/// Mirrors sklearn's `fit(X, y, sample_weight=...)`. Estimators that support
/// importance-weighted training implement this in addition to (or instead of)
/// the unweighted [`Fit`] trait. When `sample_weight` is `None`, the result
/// should be identical to `Fit::fit(x, y)`.
pub trait FitWeighted<F: Float> {
    type Fitted;
    fn fit_weighted(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        sample_weight: Option<&Array1<F>>,
    ) -> Result<Self::Fitted>;
}

/// Incremental / online fit.
///
/// Mirrors sklearn's `partial_fit(X, y, classes=...)`. Calling `partial_fit`
/// with `state = None` initialises a fresh fitted model on the batch; later
/// calls with `state = Some(prev)` update the model in place from `prev`.
///
/// The `classes` argument is required on the first call for classifiers
/// where the label space is not derivable from a single mini-batch.
pub trait PartialFit<F: Float> {
    type Fitted;
    fn partial_fit(
        &self,
        state: Option<Self::Fitted>,
        x: &Array2<F>,
        y: &Array1<F>,
        classes: Option<&[F]>,
    ) -> Result<Self::Fitted>;
}

/// Unsupervised fit with optional per-sample weights.
///
/// Mirrors sklearn's `fit(X, sample_weight=...)` for unsupervised estimators
/// such as `KMeans`, `GaussianMixture`, and density-style scalers. When
/// `sample_weight` is `None`, the result must be identical to
/// `FitUnsupervised::fit(x)`.
pub trait FitUnsupervisedWeighted<F: Float> {
    type Fitted;
    fn fit_unsupervised_weighted(
        &self,
        x: &Array2<F>,
        sample_weight: Option<&Array1<F>>,
    ) -> Result<Self::Fitted>;
}

/// Log-probability output. Default implementation takes `log(predict_proba)`
/// with an epsilon clamp to avoid log(0).
///
/// Mirrors sklearn's `predict_log_proba`.
pub trait PredictLogProba<F: Float>: PredictProba<F> {
    fn predict_log_proba(&self, x: &Array2<F>) -> Result<Array2<F>> {
        let p = self.predict_proba(x)?;
        let eps = F::from_f64(1e-300).unwrap();
        Ok(p.mapv(|v| if v < eps { eps.ln() } else { v.ln() }))
    }
}

/// Real-valued decision scores per class, before the softmax/argmax. Mirrors
/// `sklearn.base.ClassifierMixin.decision_function`. Shape `(n_samples, n_classes)`
/// for multi-class (sklearn returns 1-D for binary; we always return 2-D for
/// consistency).
pub trait DecisionFunction<F: Float> {
    fn decision_function(&self, x: &Array2<F>) -> Result<Array2<F>>;
}

/// Default scoring for regressors: R² (coefficient of determination).
///
/// Mirrors `sklearn.base.RegressorMixin.score`. Higher is better; 1.0 is
/// perfect prediction, 0.0 means equivalent to predicting `y.mean()`.
pub trait RegressorScore<F: Float>: Predict<F> {
    fn score(&self, x: &Array2<F>, y: &Array1<F>) -> Result<F> {
        let pred = self.predict(x)?;
        let n = y.len();
        if n != pred.len() {
            return Err(crate::error::RustMlError::ShapeMismatch(format!(
                "y len {} != pred len {}",
                n,
                pred.len()
            )));
        }
        let y_mean = y.iter().fold(F::zero(), |acc, &v| acc + v) / F::from_usize(n).unwrap();
        let mut rss = F::zero();
        let mut tss = F::zero();
        for (a, b) in y.iter().zip(pred.iter()) {
            let r = *a - *b;
            rss = rss + r * r;
            let t = *a - y_mean;
            tss = tss + t * t;
        }
        let tss_safe = if tss == F::zero() {
            F::from_f64(1e-12).unwrap()
        } else {
            tss
        };
        Ok(F::one() - rss / tss_safe)
    }
}

/// Default scoring for classifiers: accuracy.
///
/// Mirrors `sklearn.base.ClassifierMixin.score`.
pub trait ClassifierScore<F: Float>: Predict<F> {
    fn score(&self, x: &Array2<F>, y: &Array1<F>) -> Result<F> {
        let pred = self.predict(x)?;
        let n = y.len();
        if n != pred.len() {
            return Err(crate::error::RustMlError::ShapeMismatch(format!(
                "y len {} != pred len {}",
                n,
                pred.len()
            )));
        }
        let eps = F::from_f64(1e-9).unwrap();
        let correct = y
            .iter()
            .zip(pred.iter())
            .filter(|(a, b)| (**a - **b).abs() < eps)
            .count();
        Ok(F::from_usize(correct).unwrap() / F::from_usize(n).unwrap())
    }
}
