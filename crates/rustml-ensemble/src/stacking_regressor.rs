//! Stacking regressor: two-level ensemble where base models' predictions
//! are combined by a meta-estimator.

use ndarray::{Array1, Array2};
use rustml_core::{Fit, Float, Predict, Result, RustMlError};

/// Internal trait for type-erased fit/predict.
trait FitPredBox<F: Float>: Send + Sync {
    fn fit_box(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Box<dyn PredBox<F>>>;
}

trait PredBox<F: Float>: Send + Sync {
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

/// Stacking regressor.
///
/// Base estimators produce predictions, which become features for a
/// meta-estimator that learns to combine them. During fitting, base model
/// predictions are generated via cross-validation to avoid overfitting.
pub struct StackingRegressor<F: Float> {
    base_estimators: Vec<(String, Box<dyn FitPredBox<F>>)>,
    meta_estimator: Box<dyn FitPredBox<F>>,
    cv_folds: usize,
}

impl<F: Float> StackingRegressor<F> {
    /// Create a new StackingRegressor with the given meta-estimator.
    pub fn new<M>(meta_estimator: M) -> Self
    where
        M: Fit<F> + Send + Sync + 'static,
        M::Fitted: Predict<F> + Send + Sync + 'static,
    {
        Self {
            base_estimators: Vec::new(),
            meta_estimator: Box::new(meta_estimator),
            cv_folds: 5,
        }
    }

    /// Add a base estimator.
    pub fn push<T>(mut self, name: impl Into<String>, estimator: T) -> Self
    where
        T: Fit<F> + Send + Sync + 'static,
        T::Fitted: Predict<F> + Send + Sync + 'static,
    {
        self.base_estimators.push((name.into(), Box::new(estimator)));
        self
    }

    /// Set the number of CV folds for generating meta-features. Default: 5.
    pub fn with_cv_folds(mut self, cv_folds: usize) -> Self {
        self.cv_folds = cv_folds;
        self
    }
}

/// Fitted stacking regressor.
pub struct FittedStackingRegressor<F: Float> {
    fitted_base: Vec<(String, Box<dyn PredBox<F>>)>,
    fitted_meta: Box<dyn PredBox<F>>,
    n_features: usize,
}

impl<F: Float> FittedStackingRegressor<F> {
    pub fn estimator_names(&self) -> Vec<&str> {
        self.fitted_base.iter().map(|(n, _)| n.as_str()).collect()
    }
}

impl<F: Float + 'static> Fit<F> for StackingRegressor<F> {
    type Fitted = FittedStackingRegressor<F>;

    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Self::Fitted> {
        if self.base_estimators.is_empty() {
            return Err(RustMlError::InvalidParameter(
                "StackingRegressor needs at least one base estimator".into(),
            ));
        }
        if x.nrows() != y.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but y has {} elements",
                x.nrows(),
                y.len()
            )));
        }
        let n = x.nrows();
        if n < 2 {
            return Err(RustMlError::EmptyInput("need at least 2 samples".into()));
        }

        let n_base = self.base_estimators.len();
        let k = self.cv_folds.min(n);

        // Generate out-of-fold predictions for meta-features
        let folds = simple_k_fold(n, k);
        let mut meta_features = Array2::zeros((n, n_base));

        for (bi, (_, est)) in self.base_estimators.iter().enumerate() {
            for (train_idx, test_idx) in &folds {
                let x_train = select_rows(x, train_idx);
                let y_train = select_elements(y, train_idx);
                let x_test = select_rows(x, test_idx);

                let fitted = est.fit_box(&x_train, &y_train)?;
                let preds = fitted.predict_box(&x_test)?;

                for (li, &gi) in test_idx.iter().enumerate() {
                    meta_features[[gi, bi]] = preds[li];
                }
            }
        }

        // Fit meta-estimator on stacked predictions
        let fitted_meta = self.meta_estimator.fit_box(&meta_features, y)?;

        // Refit base estimators on full data
        let mut fitted_base = Vec::with_capacity(n_base);
        for (name, est) in &self.base_estimators {
            let fitted = est.fit_box(x, y)?;
            fitted_base.push((name.clone(), fitted));
        }

        Ok(FittedStackingRegressor {
            fitted_base,
            fitted_meta,
            n_features: x.ncols(),
        })
    }
}

impl<F: Float> Predict<F> for FittedStackingRegressor<F> {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }

        let n = x.nrows();
        let n_base = self.fitted_base.len();
        let mut meta_features = Array2::zeros((n, n_base));

        for (bi, (_, model)) in self.fitted_base.iter().enumerate() {
            let preds = model.predict_box(x)?;
            for i in 0..n {
                meta_features[[i, bi]] = preds[i];
            }
        }

        self.fitted_meta.predict_box(&meta_features)
    }
}

/// Simple non-stratified k-fold for internal use.
fn simple_k_fold(n: usize, k: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
    let fold_size = n / k;
    let remainder = n % k;
    let mut folds = Vec::with_capacity(k);
    let mut start = 0;

    for f in 0..k {
        let end = start + fold_size + if f < remainder { 1 } else { 0 };
        let test: Vec<usize> = (start..end).collect();
        let train: Vec<usize> = (0..start).chain(end..n).collect();
        folds.push((train, test));
        start = end;
    }
    folds
}

fn select_rows<F: Float>(x: &Array2<F>, indices: &[usize]) -> Array2<F> {
    let ncols = x.ncols();
    let mut data = Vec::with_capacity(indices.len() * ncols);
    for &i in indices {
        for j in 0..ncols {
            data.push(x[[i, j]]);
        }
    }
    Array2::from_shape_vec((indices.len(), ncols), data).unwrap()
}

fn select_elements<F: Float>(y: &Array1<F>, indices: &[usize]) -> Array1<F> {
    Array1::from_vec(indices.iter().map(|&i| y[i]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;
    use rustml_trees::DecisionTreeRegressor;

    #[test]
    fn test_stacking_regressor_basic() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];

        let sr = StackingRegressor::new(DecisionTreeRegressor::default())
            .push("t1", DecisionTreeRegressor { max_depth: Some(2), ..Default::default() })
            .push("t2", DecisionTreeRegressor { max_depth: Some(3), ..Default::default() })
            .with_cv_folds(2);

        let fitted: FittedStackingRegressor<f64> = sr.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 8);

        for &p in preds.iter() {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn test_stacking_regressor_names() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![1.0, 2.0, 3.0, 4.0];

        let sr = StackingRegressor::new(DecisionTreeRegressor::default())
            .push("a", DecisionTreeRegressor::default())
            .push("b", DecisionTreeRegressor::default())
            .with_cv_folds(2);

        let fitted: FittedStackingRegressor<f64> = sr.fit(&x, &y).unwrap();
        assert_eq!(fitted.estimator_names(), vec!["a", "b"]);
    }

    #[test]
    fn test_stacking_regressor_empty_base_error() {
        let x = array![[1.0], [2.0]];
        let y = array![1.0, 2.0];

        let sr = StackingRegressor::<f64>::new(DecisionTreeRegressor::default());
        assert!(sr.fit(&x, &y).is_err());
    }

    #[test]
    fn test_stacking_regressor_predict_shape_mismatch() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let y = array![1.0, 2.0, 3.0, 4.0];

        let sr = StackingRegressor::new(DecisionTreeRegressor::default())
            .push("t1", DecisionTreeRegressor::default())
            .with_cv_folds(2);

        let fitted: FittedStackingRegressor<f64> = sr.fit(&x, &y).unwrap();
        let x_bad = array![[1.0]];
        assert!(fitted.predict(&x_bad).is_err());
    }
}
