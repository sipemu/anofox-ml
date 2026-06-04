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
