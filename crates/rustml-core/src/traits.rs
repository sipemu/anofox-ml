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
