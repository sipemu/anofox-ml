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

    pub fn n_features(&self) -> usize { self.n_features }

    pub fn n_outputs(&self) -> usize {
        self.models.len()
    }
}

// ---------------------------------------------------------------------------
// MultiOutputClassifier — same one-per-output pattern as MultiOutputRegressor.
// Just a re-export under a clearer name since the math is identical.
// ---------------------------------------------------------------------------

/// Multi-output classifier. Fits one independent classifier per output column;
/// each column of `y` is the class label for that output dimension. Identical
/// implementation to `MultiOutputRegressor` — the distinction is purely about
/// downstream meaning (sklearn ships them as separate classes for the same
/// reason).
pub type MultiOutputClassifier<F> = MultiOutputRegressor<F>;
pub type FittedMultiOutputClassifier<F> = FittedMultiOutputRegressor<F>;

// ---------------------------------------------------------------------------
// RegressorChain — chain feeds previous predictions as features.
// ---------------------------------------------------------------------------

/// Chain of regressors where each step's prediction becomes a feature for the
/// next. Mirrors `sklearn.multioutput.RegressorChain`. With `order` = `[2, 0, 1]`,
/// the regressor for output 2 sees only the original X, the regressor for
/// output 0 sees X + prediction-of-2, and so on.
pub struct RegressorChain<F: Float> {
    template: Box<dyn MultiFitTemplate<F>>,
    order: Option<Vec<usize>>,
}

impl<F: Float> RegressorChain<F> {
    pub fn new<T>(estimator: T) -> Self
    where
        T: Fit<F> + Send + Sync + Clone + 'static,
        T::Fitted: Predict<F> + Send + Sync + 'static,
    {
        Self {
            template: Box::new(Template(estimator)),
            order: None,
        }
    }

    /// Set the chain order. Default is `0..n_outputs`.
    pub fn with_order(mut self, order: Vec<usize>) -> Self {
        self.order = Some(order);
        self
    }

    pub fn fit_2d(&self, x: &Array2<F>, y: &Array2<F>) -> Result<FittedRegressorChain<F>> {
        if x.nrows() != y.nrows() {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but y has {}", x.nrows(), y.nrows()
            )));
        }
        if y.is_empty() {
            return Err(RustMlError::EmptyInput("y is empty".into()));
        }
        let n = x.nrows();
        let d = x.ncols();
        let n_out = y.ncols();
        let order = self.order.clone().unwrap_or_else(|| (0..n_out).collect());
        if order.len() != n_out {
            return Err(RustMlError::InvalidParameter(format!(
                "order length {} != n_outputs {}", order.len(), n_out
            )));
        }
        let mut models: Vec<Box<dyn PredBox<F>>> = Vec::with_capacity(n_out);
        // Build per-step features: original X plus all already-predicted columns.
        let mut x_ext = Array2::<F>::zeros((n, d + n_out));
        // Copy original X.
        for i in 0..n {
            for j in 0..d {
                x_ext[[i, j]] = x[[i, j]];
            }
        }
        for (step, &out_idx) in order.iter().enumerate() {
            // Build the feature view containing original + first `step` predicted columns.
            let cur_cols = d + step;
            let xs = sub_x(&x_ext, n, cur_cols);
            let yk = y.index_axis(Axis(1), out_idx).to_owned();
            let m = self.template.fit_box(&xs, &yk)?;
            // For subsequent steps we feed the *predicted* values of yk
            // (sklearn does this — at fit time it's the true value to avoid
            // exposure bias; we follow sklearn and use the true y at fit).
            for i in 0..n {
                x_ext[[i, d + step]] = y[[i, out_idx]];
            }
            models.push(m);
        }
        Ok(FittedRegressorChain {
            models,
            order,
            n_features: d,
            n_outputs: n_out,
        })
    }
}

pub struct FittedRegressorChain<F: Float> {
    models: Vec<Box<dyn PredBox<F>>>,
    order: Vec<usize>,
    n_features: usize,
    n_outputs: usize,
}

impl<F: Float> FittedRegressorChain<F> {
    pub fn predict_2d(&self, x: &Array2<F>) -> Result<Array2<F>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}", self.n_features, x.ncols()
            )));
        }
        let n = x.nrows();
        let d = self.n_features;
        let mut x_ext = Array2::<F>::zeros((n, d + self.n_outputs));
        for i in 0..n {
            for j in 0..d {
                x_ext[[i, j]] = x[[i, j]];
            }
        }
        let mut out = Array2::<F>::zeros((n, self.n_outputs));
        for (step, &out_idx) in self.order.iter().enumerate() {
            let xs = sub_x(&x_ext, n, d + step);
            let pred = self.models[step].predict_box(&xs)?;
            for i in 0..n {
                out[[i, out_idx]] = pred[i];
                x_ext[[i, d + step]] = pred[i];
            }
        }
        Ok(out)
    }
}

fn sub_x<F: Float>(x_ext: &Array2<F>, n: usize, cols: usize) -> Array2<F> {
    let mut out = Array2::<F>::zeros((n, cols));
    for i in 0..n {
        for j in 0..cols {
            out[[i, j]] = x_ext[[i, j]];
        }
    }
    out
}

// `ClassifierChain` is the same as `RegressorChain` for our purposes
// (per-step prediction is a class label, fed as a feature to the next step).
pub type ClassifierChain<F> = RegressorChain<F>;
pub type FittedClassifierChain<F> = FittedRegressorChain<F>;

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

    #[test]
    fn test_regressor_chain_predicts_2d() {
        // With MeanReg as the base, each step's prediction is the (constant)
        // mean of its target column — chain ordering doesn't affect output for
        // this trivial estimator, but predict_2d should produce the expected
        // 2-D shape and values.
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![[1.0, 10.0], [3.0, 20.0], [5.0, 30.0], [7.0, 40.0]];

        let chain = RegressorChain::<f64>::new(MeanReg);
        let fitted = chain.fit_2d(&x, &y).unwrap();
        let p = fitted.predict_2d(&x).unwrap();
        assert_eq!(p.shape(), &[4, 2]);
        for i in 0..4 {
            assert!(p[[i, 0]].is_finite());
            assert!(p[[i, 1]].is_finite());
        }
    }
}
