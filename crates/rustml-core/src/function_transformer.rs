//! FunctionTransformer — wraps a closure as a pipeline-compatible transformer.
//!
//! Useful for inserting arbitrary transformations into a Pipeline without
//! defining a full struct + impl.

use std::sync::Arc;

use ndarray::Array2;

use crate::error::Result;
use crate::float::Float;
use crate::traits::{FitUnsupervised, Transform};

/// A transformer that applies an arbitrary function to the data.
///
/// Implements `FitTransform` so it can be used directly in a `Pipeline`.
/// The fit step is a no-op — only the transform function is called.
///
/// # Example
///
/// ```ignore
/// use rustml_core::{Pipeline, FunctionTransformer};
///
/// let log_transform = FunctionTransformer::new(|x: &Array2<f64>| {
///     Ok(x.mapv(|v| v.ln()))
/// });
///
/// let pipeline = Pipeline::new()
///     .push_transformer("log", log_transform);
/// ```
pub struct FunctionTransformer<F: Float> {
    func: Arc<dyn Fn(&Array2<F>) -> Result<Array2<F>> + Send + Sync>,
}

impl<F: Float> FunctionTransformer<F> {
    /// Create a new FunctionTransformer from a closure.
    pub fn new(func: impl Fn(&Array2<F>) -> Result<Array2<F>> + Send + Sync + 'static) -> Self {
        Self {
            func: Arc::new(func),
        }
    }
}

impl<F: Float> std::fmt::Debug for FunctionTransformer<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunctionTransformer").finish()
    }
}

impl<F: Float> Clone for FunctionTransformer<F> {
    fn clone(&self) -> Self {
        Self {
            func: Arc::clone(&self.func),
        }
    }
}

/// Fitted function transformer — just holds the closure.
pub struct FittedFunctionTransformer<F: Float> {
    func: Arc<dyn Fn(&Array2<F>) -> Result<Array2<F>> + Send + Sync>,
}

unsafe impl<F: Float> Send for FittedFunctionTransformer<F> {}
unsafe impl<F: Float> Sync for FittedFunctionTransformer<F> {}

impl<F: Float> Transform<F> for FittedFunctionTransformer<F> {
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>> {
        (self.func)(x)
    }
}

impl<F: Float + 'static> FitUnsupervised<F> for FunctionTransformer<F> {
    type Fitted = FittedFunctionTransformer<F>;

    fn fit(&self, _x: &Array2<F>) -> Result<Self::Fitted> {
        Ok(FittedFunctionTransformer {
            func: Arc::clone(&self.func),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{FitUnsupervised, Transform};
    use ndarray::array;

    #[test]
    fn test_function_transformer_identity() {
        let ft = FunctionTransformer::<f64>::new(|x| Ok(x.to_owned()));
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let fitted = FitUnsupervised::fit(&ft, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();
        assert_eq!(transformed, x);
    }

    #[test]
    fn test_function_transformer_log() {
        let ft = FunctionTransformer::<f64>::new(|x| Ok(x.mapv(|v| v.ln())));
        let x = array![[1.0, std::f64::consts::E], [std::f64::consts::E, 1.0]];
        let fitted = FitUnsupervised::fit(&ft, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();
        assert!((transformed[[0, 0]] - 0.0).abs() < 1e-10);
        assert!((transformed[[0, 1]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_function_transformer_scale() {
        let ft = FunctionTransformer::<f64>::new(|x| Ok(x.mapv(|v| v * 2.0)));
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let fitted = FitUnsupervised::fit(&ft, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();
        assert_eq!(transformed, array![[2.0, 4.0], [6.0, 8.0]]);
    }

    #[test]
    fn test_function_transformer_clone() {
        let ft = FunctionTransformer::<f64>::new(|x| Ok(x.mapv(|v| v + 1.0)));
        let ft2 = ft.clone();
        let x = array![[1.0]];
        let f1 = FitUnsupervised::fit(&ft, &x).unwrap();
        let f2 = FitUnsupervised::fit(&ft2, &x).unwrap();
        assert_eq!(f1.transform(&x).unwrap(), f2.transform(&x).unwrap());
    }
}
