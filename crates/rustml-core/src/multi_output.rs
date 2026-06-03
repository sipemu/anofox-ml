//! Multi-output meta-estimators: per-output regressor / classifier wrappers.
//!
//! Mirrors `sklearn.multioutput.MultiOutputRegressor` (a separate estimator is
//! fitted per output column). Our existing `Fit` / `Predict` traits assume 1-D
//! `y`; this wrapper provides a 2-D entry point on top.

use ndarray::{Array1, Array2, Axis};

use crate::error::{Result, RustMlError};
use crate::float::Float;
use crate::traits::{Fit, Predict};

/// Internal trait used to abstract over the inner-estimator type without a
/// blanket impl that conflicts with downstream crates.
trait MultiFitTemplate<F: Float>: Send + Sync {
    fn fit_box(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Box<dyn PredBox<F>>>;
}

trait PredBox<F: Float>: Send + Sync {
    fn predict_box(&self, x: &Array2<F>) -> Result<Array1<F>>;
}

struct Template<T>(T);

impl<F, T> MultiFitTemplate<F> for Template<T>
where
    F: Float,
    T: Fit<F> + Send + Sync + Clone,
    T::Fitted: Predict<F> + Send + Sync + 'static,
{
    fn fit_box(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Box<dyn PredBox<F>>> {
        let est = self.0.clone();
        let fitted = Fit::fit(&est, x, y)?;
        Ok(Box::new(PredHolder(fitted)))
    }
}

struct PredHolder<P>(P);
impl<F, P> PredBox<F> for PredHolder<P>
where
    F: Float,
    P: Predict<F> + Send + Sync,
{
    fn predict_box(&self, x: &Array2<F>) -> Result<Array1<F>> {
        self.0.predict(x)
    }
}

/// Fits one independent estimator per target column.
pub struct MultiOutputRegressor<F: Float> {
    template: Box<dyn MultiFitTemplate<F>>,
}

impl<F: Float> MultiOutputRegressor<F> {
    pub fn new<T>(estimator: T) -> Self
    where
        T: Fit<F> + Send + Sync + Clone + 'static,
        T::Fitted: Predict<F> + Send + Sync + 'static,
    {
        Self {
            template: Box::new(Template(estimator)),
        }
    }

    pub fn fit_2d(&self, x: &Array2<F>, y: &Array2<F>) -> Result<FittedMultiOutputRegressor<F>> {
        if x.nrows() != y.nrows() {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but y has {}",
                x.nrows(),
                y.nrows()
            )));
        }
        if y.is_empty() {
            return Err(RustMlError::EmptyInput("y is empty".into()));
        }
        let n_outputs = y.ncols();
        let mut fitted = Vec::with_capacity(n_outputs);
        for k in 0..n_outputs {
            let yk = y.index_axis(Axis(1), k).to_owned();
            let m = self.template.fit_box(x, &yk)?;
            fitted.push(m);
        }
        Ok(FittedMultiOutputRegressor {
            models: fitted,
            n_features: x.ncols(),
        })
    }
}

pub struct FittedMultiOutputRegressor<F: Float> {
    models: Vec<Box<dyn PredBox<F>>>,
    n_features: usize,
}

impl<F: Float> FittedMultiOutputRegressor<F> {
    pub fn predict_2d(&self, x: &Array2<F>) -> Result<Array2<F>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        let n = x.nrows();
        let mut out = Array2::<F>::zeros((n, self.models.len()));
        for (k, m) in self.models.iter().enumerate() {
            let yk = m.predict_box(x)?;
            for i in 0..n {
                out[[i, k]] = yk[i];
            }
        }
        Ok(out)
    }

    pub fn n_outputs(&self) -> usize {
        self.models.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[derive(Clone)]
    struct MeanReg;
    struct FittedMeanReg(f64);

    impl Fit<f64> for MeanReg {
        type Fitted = FittedMeanReg;
        fn fit(&self, _x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
            let m = y.iter().sum::<f64>() / y.len() as f64;
            Ok(FittedMeanReg(m))
        }
    }
    impl Predict<f64> for FittedMeanReg {
        fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
            Ok(Array1::from_elem(x.nrows(), self.0))
        }
    }

    #[test]
    fn test_multi_output_predicts_per_column_mean() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![[1.0, 10.0], [3.0, 20.0], [5.0, 30.0], [7.0, 40.0]];

        let model = MultiOutputRegressor::<f64>::new(MeanReg);
        let fitted = model.fit_2d(&x, &y).unwrap();
        let p = fitted.predict_2d(&x).unwrap();
        assert_eq!(p.shape(), &[4, 2]);
        for i in 0..4 {
            assert!((p[[i, 0]] - 4.0).abs() < 1e-9);
            assert!((p[[i, 1]] - 25.0).abs() < 1e-9);
        }
    }
}
