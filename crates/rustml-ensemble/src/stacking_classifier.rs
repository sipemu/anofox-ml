//! Stacking classifier: two-level ensemble where base classifiers' predictions
//! are used as features for a meta-classifier.
//!
//! Mirrors `sklearn.ensemble.StackingClassifier` with `stack_method='predict'`:
//! base classifier *predictions* (not class probabilities) become inputs to the
//! meta-estimator. Out-of-fold predictions are generated via k-fold CV during
//! fitting to avoid leakage.

use ndarray::{Array1, Array2};
use rustml_core::{Fit, Float, Predict, Result, RustMlError};

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

/// Stacking classifier.
pub struct StackingClassifier<F: Float> {
    base_estimators: Vec<(String, Box<dyn FitPredBox<F>>)>,
    meta_estimator: Box<dyn FitPredBox<F>>,
    cv_folds: usize,
}

impl<F: Float> StackingClassifier<F> {
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

    pub fn push<T>(mut self, name: impl Into<String>, estimator: T) -> Self
    where
        T: Fit<F> + Send + Sync + 'static,
        T::Fitted: Predict<F> + Send + Sync + 'static,
    {
        self.base_estimators.push((name.into(), Box::new(estimator)));
        self
    }

    pub fn with_cv_folds(mut self, k: usize) -> Self {
        self.cv_folds = k;
        self
    }
}

pub struct FittedStackingClassifier<F: Float> {
    fitted_base: Vec<(String, Box<dyn PredBox<F>>)>,
    fitted_meta: Box<dyn PredBox<F>>,
    n_features: usize,
}

impl<F: Float> FittedStackingClassifier<F> {
    pub fn estimator_names(&self) -> Vec<&str> {
        self.fitted_base.iter().map(|(n, _)| n.as_str()).collect()
    }
}

impl<F: Float + 'static> Fit<F> for StackingClassifier<F> {
    type Fitted = FittedStackingClassifier<F>;

    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Self::Fitted> {
        if self.base_estimators.is_empty() {
            return Err(RustMlError::InvalidParameter(
                "StackingClassifier needs at least one base estimator".into(),
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

        let fitted_meta = self.meta_estimator.fit_box(&meta_features, y)?;

        let mut fitted_base = Vec::with_capacity(n_base);
        for (name, est) in &self.base_estimators {
            let fitted = est.fit_box(x, y)?;
            fitted_base.push((name.clone(), fitted));
        }

        Ok(FittedStackingClassifier {
            fitted_base,
            fitted_meta,
            n_features: x.ncols(),
        })
    }
}

impl<F: Float> Predict<F> for FittedStackingClassifier<F> {
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

        for (bi, (_, m)) in self.fitted_base.iter().enumerate() {
            let preds = m.predict_box(x)?;
            for i in 0..n {
                meta_features[[i, bi]] = preds[i];
            }
        }
        self.fitted_meta.predict_box(&meta_features)
    }
}

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
    use ndarray::array;
    use rustml_trees::DecisionTreeClassifier;

    #[test]
    fn test_stacking_classifier_basic() {
        // Two well-separated clusters, interleaved so simple k-fold sees both
        // classes in each fold.
        let x = array![
            [0.0, 0.0], [5.0, 5.0],
            [0.1, 0.1], [5.1, 5.0],
            [0.2, -0.1], [4.9, 5.1],
            [-0.1, 0.2], [5.2, 4.8],
        ];
        let y = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];

        let sc = StackingClassifier::new(DecisionTreeClassifier::default())
            .push("t1", DecisionTreeClassifier { max_depth: Some(2), ..Default::default() })
            .push("t2", DecisionTreeClassifier { max_depth: Some(3), ..Default::default() })
            .with_cv_folds(2);

        let fitted: FittedStackingClassifier<f64> = sc.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_eq!(*p, *t, "p={p}, t={t}");
        }
    }

    #[test]
    fn test_stacking_classifier_empty_base_error() {
        let x = array![[1.0], [2.0]];
        let y = array![0.0, 1.0];

        let sc = StackingClassifier::<f64>::new(DecisionTreeClassifier::default());
        assert!(sc.fit(&x, &y).is_err());
    }
}
