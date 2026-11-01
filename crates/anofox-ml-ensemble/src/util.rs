//! Helpers shared by the meta-estimators.

use anofox_ml_core::{Fit, Float, Predict, Result};
use ndarray::{Array1, Array2};

/// A boxed estimator that can be fitted into a boxed predictor.
pub(crate) trait FitPredBox<F: Float>: Send + Sync {
    fn fit_box(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Box<dyn PredBox<F>>>;
}

/// A boxed fitted predictor.
pub(crate) trait PredBox<F: Float>: Send + Sync {
    fn predict_box(&self, x: &Array2<F>) -> Result<Array1<F>>;
}

impl<F, T> FitPredBox<F> for T
where
    F: Float,
    T: Fit<F> + Send + Sync,
    T::Fitted: Predict<F> + Send + Sync + 'static,
{
    fn fit_box(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Box<dyn PredBox<F>>> {
        let fitted = Fit::fit(self, x, y)?;
        Ok(Box::new(fitted))
    }
}

impl<F, T> PredBox<F> for T
where
    F: Float,
    T: Predict<F> + Send + Sync,
{
    fn predict_box(&self, x: &Array2<F>) -> Result<Array1<F>> {
        self.predict(x)
    }
}
