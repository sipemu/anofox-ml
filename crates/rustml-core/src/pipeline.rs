use ndarray::{Array1, Array2};

use crate::error::{Result, RustMlError};
use crate::float::Float;
use crate::traits::{Fit, Predict, Transform};

/// A transformer that can be fit on data and then transform it.
pub trait FitTransform<F: Float>: Send + Sync {
    fn fit_transform(&self, x: &Array2<F>) -> Result<(Box<dyn TransformStep<F>>, Array2<F>)>;
}

/// A fitted transformer step that can transform new data.
pub trait TransformStep<F: Float>: Send + Sync {
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>>;
}

/// A supervised estimator that can be fit and then predict.
pub trait FitPredict<F: Float>: Send + Sync {
    fn fit_predict_step(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn PredictStep<F>>>;
}

/// A fitted estimator step that can predict.
pub trait PredictStep<F: Float>: Send + Sync {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>>;
}

// --- Blanket implementations ---

/// Blanket impl: any type implementing FitUnsupervised + Transform can be a pipeline transformer.
impl<F, T> FitTransform<F> for T
where
    F: Float,
    T: crate::traits::FitUnsupervised<F> + Send + Sync,
    T::Fitted: Transform<F> + Send + Sync + 'static,
{
    fn fit_transform(&self, x: &Array2<F>) -> Result<(Box<dyn TransformStep<F>>, Array2<F>)> {
        let fitted = crate::traits::FitUnsupervised::fit(self, x)?;
        let transformed = fitted.transform(x)?;
        Ok((Box::new(FittedTransformWrapper(fitted)), transformed))
    }
}

struct FittedTransformWrapper<T>(T);

// Safety: T is already Send + Sync via trait bounds
unsafe impl<T: Send> Send for FittedTransformWrapper<T> {}
unsafe impl<T: Sync> Sync for FittedTransformWrapper<T> {}

impl<F: Float, T: Transform<F> + Send + Sync> TransformStep<F> for FittedTransformWrapper<T> {
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>> {
        self.0.transform(x)
    }
}

/// Blanket impl: any type implementing Fit + Predict can be a pipeline estimator.
impl<F, T> FitPredict<F> for T
where
    F: Float,
    T: Fit<F> + Send + Sync,
    T::Fitted: Predict<F> + Send + Sync + 'static,
{
    fn fit_predict_step(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn PredictStep<F>>> {
        let fitted = Fit::fit(self, x, y)?;
        Ok(Box::new(FittedPredictWrapper(fitted)))
    }
}

struct FittedPredictWrapper<T>(T);

unsafe impl<T: Send> Send for FittedPredictWrapper<T> {}
unsafe impl<T: Sync> Sync for FittedPredictWrapper<T> {}

impl<F: Float, T: Predict<F> + Send + Sync> PredictStep<F> for FittedPredictWrapper<T> {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        self.0.predict(x)
    }
}

/// An unfitted pipeline that chains transformers and a final estimator.
///
/// Follows sklearn's Pipeline pattern: chain transformers in sequence,
/// then optionally fit/predict with a final estimator.
///
/// # Example
/// ```ignore
/// use rustml::prelude::*;
///
/// let pipeline = Pipeline::new()
///     .push_transformer("scaler", StandardScaler::new())
///     .set_estimator("knn", KnnClassifier::new(5));
///
/// let fitted = pipeline.fit(&x_train, &y_train)?;
/// let preds = fitted.predict(&x_test)?;
/// ```
pub struct Pipeline<F: Float> {
    transformers: Vec<(String, Box<dyn FitTransform<F>>)>,
    estimator: Option<(String, Box<dyn FitPredict<F>>)>,
}

impl<F: Float> Pipeline<F> {
    /// Create an empty pipeline.
    pub fn new() -> Self {
        Pipeline {
            transformers: Vec::new(),
            estimator: None,
        }
    }

    /// Add a transformer step to the pipeline.
    pub fn push_transformer(
        mut self,
        name: impl Into<String>,
        transformer: impl FitTransform<F> + 'static,
    ) -> Self {
        self.transformers
            .push((name.into(), Box::new(transformer)));
        self
    }

    /// Set the final estimator of the pipeline.
    pub fn set_estimator(
        mut self,
        name: impl Into<String>,
        estimator: impl FitPredict<F> + 'static,
    ) -> Self {
        self.estimator = Some((name.into(), Box::new(estimator)));
        self
    }

    /// Fit the pipeline: fit each transformer sequentially, transforming
    /// X at each step, then fit the final estimator on the transformed X.
    pub fn fit(self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedPipeline<F>> {
        let mut current_x_owned: Option<Array2<F>> = None;
        let mut fitted_transformers = Vec::with_capacity(self.transformers.len());

        for (name, transformer) in self.transformers {
            let x_ref = current_x_owned.as_ref().unwrap_or(x);
            let (fitted, transformed) = transformer.fit_transform(x_ref)?;
            fitted_transformers.push((name, fitted));
            current_x_owned = Some(transformed);
        }

        let x_final = current_x_owned.as_ref().unwrap_or(x);
        let fitted_estimator = match self.estimator {
            Some((name, estimator)) => {
                let fitted = estimator.fit_predict_step(x_final, y)?;
                Some((name, fitted))
            }
            None => None,
        };

        Ok(FittedPipeline {
            transformers: fitted_transformers,
            estimator: fitted_estimator,
        })
    }
}

impl<F: Float> Default for Pipeline<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// A fitted pipeline that can transform and predict on new data.
pub struct FittedPipeline<F: Float> {
    transformers: Vec<(String, Box<dyn TransformStep<F>>)>,
    estimator: Option<(String, Box<dyn PredictStep<F>>)>,
}

impl<F: Float> FittedPipeline<F> {
    /// Transform new data through all transformer steps.
    pub fn transform(&self, x: &Array2<F>) -> Result<Array2<F>> {
        let mut current_x_owned: Option<Array2<F>> = None;
        for (_, transformer) in &self.transformers {
            let x_ref = current_x_owned.as_ref().unwrap_or(x);
            current_x_owned = Some(transformer.transform(x_ref)?);
        }
        Ok(current_x_owned.unwrap_or_else(|| x.clone()))
    }

    /// Transform new data through all steps, then predict with the final estimator.
    pub fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        let transformed = self.transform(x)?;
        match &self.estimator {
            Some((_, estimator)) => estimator.predict(&transformed),
            None => Err(RustMlError::NotFitted(
                "Pipeline has no estimator set".into(),
            )),
        }
    }

    /// Get the names of all steps in the pipeline.
    pub fn step_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self
            .transformers
            .iter()
            .map(|(name, _)| name.as_str())
            .collect();
        if let Some((name, _)) = &self.estimator {
            names.push(name.as_str());
        }
        names
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // A simple test transformer that doubles all values
    struct Doubler;
    struct FittedDoubler;

    impl FitTransform<f64> for Doubler {
        fn fit_transform(
            &self,
            x: &Array2<f64>,
        ) -> Result<(Box<dyn TransformStep<f64>>, Array2<f64>)> {
            let transformed = x.mapv(|v| v * 2.0);
            Ok((Box::new(FittedDoubler), transformed))
        }
    }

    impl TransformStep<f64> for FittedDoubler {
        fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
            Ok(x.mapv(|v| v * 2.0))
        }
    }

    // A simple test estimator that predicts the mean of features
    struct MeanPredictor;
    struct FittedMeanPredictor;

    impl FitPredict<f64> for MeanPredictor {
        fn fit_predict_step(
            &self,
            _x: &Array2<f64>,
            _y: &Array1<f64>,
        ) -> Result<Box<dyn PredictStep<f64>>> {
            Ok(Box::new(FittedMeanPredictor))
        }
    }

    impl PredictStep<f64> for FittedMeanPredictor {
        fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
            Ok(x.mean_axis(ndarray::Axis(1)).unwrap())
        }
    }

    #[test]
    fn test_pipeline_transform_only() {
        let pipeline = Pipeline::<f64>::new().push_transformer("doubler", Doubler);

        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0];

        let fitted = pipeline.fit(&x, &y).unwrap();
        let result = fitted.transform(&array![[1.0, 1.0]]).unwrap();
        assert_eq!(result, array![[2.0, 2.0]]);
    }

    #[test]
    fn test_pipeline_with_estimator() {
        let pipeline = Pipeline::<f64>::new()
            .push_transformer("doubler", Doubler)
            .set_estimator("mean", MeanPredictor);

        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0];

        let fitted = pipeline.fit(&x, &y).unwrap();

        // Input [1, 2] -> doubled to [2, 4] -> mean = 3.0
        let preds = fitted.predict(&array![[1.0, 2.0]]).unwrap();
        assert!((preds[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_pipeline_step_names() {
        let pipeline = Pipeline::<f64>::new()
            .push_transformer("step1", Doubler)
            .push_transformer("step2", Doubler)
            .set_estimator("classifier", MeanPredictor);

        let x = array![[1.0], [2.0]];
        let y = array![0.0, 1.0];

        let fitted = pipeline.fit(&x, &y).unwrap();
        assert_eq!(fitted.step_names(), vec!["step1", "step2", "classifier"]);
    }

    #[test]
    fn test_pipeline_no_estimator_predict_errors() {
        let pipeline = Pipeline::<f64>::new().push_transformer("doubler", Doubler);

        let x = array![[1.0], [2.0]];
        let y = array![0.0, 1.0];

        let fitted = pipeline.fit(&x, &y).unwrap();
        assert!(fitted.predict(&x).is_err());
    }
}
