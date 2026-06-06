use ndarray::{Array2, Axis};

use crate::error::{Result, RustMlError};
use crate::float::Float;
use crate::pipeline::{FitTransform, TransformStep};
use crate::traits::{FitUnsupervised, Transform};

/// Concatenates results of multiple transformers applied in parallel.
///
/// Each branch receives the same input and their outputs are concatenated
/// column-wise (along axis 1), similar to sklearn's `FeatureUnion`.
///
/// # Example
/// ```ignore
/// use rustml::prelude::*;
///
/// let union = FeatureUnion::new()
///     .push("scaled", StandardScaler::new())
///     .push("pca", Pca::new(2));
///
/// let pipeline = Pipeline::new()
///     .push_transformer("features", union)
///     .set_estimator("knn", KnnClassifier::new(5));
/// ```
pub struct FeatureUnion<F: Float> {
    branches: Vec<(String, Box<dyn FitTransform<F>>)>,
}

impl<F: Float> FeatureUnion<F> {
    /// Create an empty `FeatureUnion`.
    pub fn new() -> Self {
        FeatureUnion {
            branches: Vec::new(),
        }
    }

    /// Add a named transformer branch.
    pub fn push(
        mut self,
        name: impl Into<String>,
        transformer: impl FitTransform<F> + 'static,
    ) -> Self {
        self.branches.push((name.into(), Box::new(transformer)));
        self
    }
}

impl<F: Float> Default for FeatureUnion<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// A fitted `FeatureUnion` whose branches have all been fit.
pub struct FittedFeatureUnion<F: Float> {
    branches: Vec<(String, Box<dyn TransformStep<F>>)>,
}

impl<F: Float> FittedFeatureUnion<F> {
    /// Get the names of all branches.
    pub fn branch_names(&self) -> Vec<&str> {
        self.branches.iter().map(|(n, _)| n.as_str()).collect()
    }

    /// Downcast a named branch to its concrete fitted type.
    pub fn get_transformer<T: 'static>(&self, name: &str) -> Result<&T> {
        let step = self
            .branches
            .iter()
            .find(|(n, _)| n == name)
            .ok_or_else(|| RustMlError::NotFitted(format!("No branch named '{name}'")))?;
        step.1.as_any().downcast_ref::<T>().ok_or_else(|| {
            RustMlError::NotFitted(format!(
                "Branch '{name}' could not be downcast to the requested type"
            ))
        })
    }
}

fn concat_columns<F: Float>(arrays: &[Array2<F>]) -> Result<Array2<F>> {
    let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();
    ndarray::concatenate(Axis(1), &views).map_err(|e| {
        RustMlError::ShapeMismatch(format!("Failed to concatenate branch outputs: {e}"))
    })
}

/// Implement `FitUnsupervised` so the blanket `FitTransform` impl kicks in.
impl<F: Float> FitUnsupervised<F> for FeatureUnion<F> {
    type Fitted = FittedFeatureUnion<F>;

    fn fit(&self, x: &Array2<F>) -> Result<Self::Fitted> {
        if self.branches.is_empty() {
            return Err(RustMlError::InvalidParameter(
                "FeatureUnion has no branches".into(),
            ));
        }

        let mut fitted_branches = Vec::with_capacity(self.branches.len());
        for (name, transformer) in &self.branches {
            let (fitted, _) = transformer.fit_transform(x)?;
            fitted_branches.push((name.clone(), fitted));
        }

        Ok(FittedFeatureUnion {
            branches: fitted_branches,
        })
    }
}

/// Implement `Transform` so the blanket `TransformStep` wrapper works.
impl<F: Float> Transform<F> for FittedFeatureUnion<F> {
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>> {
        let mut outputs = Vec::with_capacity(self.branches.len());
        for (_, branch) in &self.branches {
            outputs.push(branch.transform(x)?);
        }
        concat_columns(&outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{FitPredict, Pipeline, PredictStep};
    use ndarray::{array, Array1};
    use std::any::Any;

    // Doubles all values
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
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    // Triples all values
    struct Tripler;
    struct FittedTripler;

    impl FitTransform<f64> for Tripler {
        fn fit_transform(
            &self,
            x: &Array2<f64>,
        ) -> Result<(Box<dyn TransformStep<f64>>, Array2<f64>)> {
            let transformed = x.mapv(|v| v * 3.0);
            Ok((Box::new(FittedTripler), transformed))
        }
    }

    impl TransformStep<f64> for FittedTripler {
        fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
            Ok(x.mapv(|v| v * 3.0))
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    // Simple estimator for pipeline integration test
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
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[test]
    fn test_feature_union_column_concat() {
        let union = FeatureUnion::new()
            .push("double", Doubler)
            .push("triple", Tripler);

        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let (fitted, transformed) = union.fit_transform(&x).unwrap();

        // 2 original cols x 2 branches = 4 cols
        assert_eq!(transformed.ncols(), 4);
        assert_eq!(transformed.nrows(), 2);
        // First two cols: doubled
        assert_eq!(transformed[[0, 0]], 2.0);
        assert_eq!(transformed[[0, 1]], 4.0);
        // Last two cols: tripled
        assert_eq!(transformed[[0, 2]], 3.0);
        assert_eq!(transformed[[0, 3]], 6.0);

        // Transform on new data
        let new_x = array![[10.0, 20.0]];
        let result = fitted.transform(&new_x).unwrap();
        assert_eq!(result.ncols(), 4);
        assert_eq!(result[[0, 0]], 20.0);
        assert_eq!(result[[0, 2]], 30.0);
    }

    #[test]
    fn test_feature_union_get_transformer() {
        let union = FeatureUnion::new()
            .push("double", Doubler)
            .push("triple", Tripler);

        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let (fitted_step, _) = union.fit_transform(&x).unwrap();

        let fitted_union = fitted_step
            .as_any()
            .downcast_ref::<FittedFeatureUnion<f64>>()
            .unwrap();

        let _doubler: &FittedDoubler = fitted_union.get_transformer("double").unwrap();
        let _tripler: &FittedTripler = fitted_union.get_transformer("triple").unwrap();
        assert!(fitted_union
            .get_transformer::<FittedDoubler>("missing")
            .is_err());
    }

    #[test]
    fn test_feature_union_branch_names() {
        let union = FeatureUnion::new().push("a", Doubler).push("b", Tripler);

        let x = array![[1.0], [2.0]];
        let (fitted_step, _) = union.fit_transform(&x).unwrap();

        let fitted_union = fitted_step
            .as_any()
            .downcast_ref::<FittedFeatureUnion<f64>>()
            .unwrap();
        assert_eq!(fitted_union.branch_names(), vec!["a", "b"]);
    }

    #[test]
    fn test_feature_union_in_pipeline() {
        let union = FeatureUnion::new()
            .push("double", Doubler)
            .push("triple", Tripler);

        let pipeline = Pipeline::<f64>::new()
            .push_transformer("union", union)
            .set_estimator("mean", MeanPredictor);

        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0];

        let fitted = pipeline.fit(&x, &y).unwrap();

        // Input [1, 2] -> union -> [2, 4, 3, 6] -> mean = 3.75
        let preds = fitted.predict(&array![[1.0, 2.0]]).unwrap();
        assert!((preds[0] - 3.75).abs() < 1e-10);
    }

    #[test]
    fn test_feature_union_empty_errors() {
        let union = FeatureUnion::<f64>::new();
        let x = array![[1.0], [2.0]];
        assert!(union.fit_transform(&x).is_err());
    }
}
