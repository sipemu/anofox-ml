//! Stacking classifier: two-level ensemble where base classifiers' predictions
//! are used as features for a meta-classifier.
//!
//! Mirrors `sklearn.ensemble.StackingClassifier` with `stack_method='predict'`:
//! base classifier *predictions* (not class probabilities) become inputs to the
//! meta-estimator. Out-of-fold predictions are generated via k-fold CV during
//! fitting to avoid leakage.

use ndarray::{Array1, Array2};
use rustml_core::{Fit, Float, Predict, PredictProba, Result, RustMlError};

/// Choice of base-estimator output used as meta-features.
///
/// - `Predict`: hard class labels (sklearn `stack_method='predict'`).
/// - `PredictProba`: class probabilities (sklearn `stack_method='predict_proba'`).
///
/// sklearn's default is `'auto'`, which prefers `predict_proba` then
/// `decision_function` then `predict`. We require an explicit choice and
/// only support the two forms above.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StackMethod {
    Predict,
    PredictProba,
}

trait FitPredBox<F: Float>: Send + Sync {
    fn fit_box(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Box<dyn PredBox<F>>>;
}

trait FitProbaBox<F: Float>: Send + Sync {
    fn fit_proba_box(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Box<dyn ProbaBox<F>>>;
}

trait PredBox<F: Float>: Send + Sync {
    fn predict_box(&self, x: &Array2<F>) -> Result<Array1<F>>;
}

trait ProbaBox<F: Float>: Send + Sync {
    fn predict_proba_box(&self, x: &Array2<F>) -> Result<Array2<F>>;
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

/// Wrapper for estimators whose Fitted type implements both Predict and PredictProba.
struct ProbaWrap<T>(T);

impl<F, T> FitProbaBox<F> for ProbaWrap<T>
where
    F: Float,
    T: Fit<F> + Send + Sync,
    T::Fitted: Predict<F> + PredictProba<F> + Send + Sync + 'static,
{
    fn fit_proba_box(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Box<dyn ProbaBox<F>>> {
        let fitted = Fit::fit(&self.0, x, y)?;
        Ok(Box::new(fitted))
    }
}

impl<F, T> ProbaBox<F> for T
where
    F: Float,
    T: PredictProba<F> + Send + Sync,
{
    fn predict_proba_box(&self, x: &Array2<F>) -> Result<Array2<F>> {
        self.predict_proba(x)
    }
}

/// Stacking classifier.
///
/// Base estimators contribute either hard predictions (one meta-feature each)
/// or class probabilities (n_classes - 1 meta-features each, dropping the last
/// column to avoid colinearity — sklearn's convention).
pub struct StackingClassifier<F: Float> {
    base_estimators: Vec<(String, BaseEstimator<F>)>,
    meta_estimator: Box<dyn FitPredBox<F>>,
    cv_folds: usize,
}

enum BaseEstimator<F: Float> {
    Predict(Box<dyn FitPredBox<F>>),
    PredictProba(Box<dyn FitProbaBox<F>>),
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

    /// Add a base estimator using hard predictions (`stack_method='predict'`).
    pub fn push<T>(mut self, name: impl Into<String>, estimator: T) -> Self
    where
        T: Fit<F> + Send + Sync + 'static,
        T::Fitted: Predict<F> + Send + Sync + 'static,
    {
        self.base_estimators
            .push((name.into(), BaseEstimator::Predict(Box::new(estimator))));
        self
    }

    /// Add a base estimator using `predict_proba` outputs (sklearn's
    /// `stack_method='predict_proba'`).
    pub fn push_proba<T>(mut self, name: impl Into<String>, estimator: T) -> Self
    where
        T: Fit<F> + Send + Sync + 'static,
        T::Fitted: Predict<F> + PredictProba<F> + Send + Sync + 'static,
    {
        self.base_estimators.push((
            name.into(),
            BaseEstimator::PredictProba(Box::new(ProbaWrap(estimator))),
        ));
        self
    }

    pub fn with_cv_folds(mut self, k: usize) -> Self {
        self.cv_folds = k;
        self
    }
}

pub struct FittedStackingClassifier<F: Float> {
    fitted_base: Vec<(String, FittedBase<F>)>,
    fitted_meta: Box<dyn PredBox<F>>,
    n_features: usize,
}

enum FittedBase<F: Float> {
    Predict(Box<dyn PredBox<F>>),
    PredictProba(Box<dyn ProbaBox<F>>),
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

        let k = self.cv_folds.min(n);
        let folds = simple_k_fold(n, k);

        // Two passes through base estimators. Pass 1: out-of-fold predictions
        // build the meta-feature matrix. We need to know each base's output
        // width — for Predict it's 1, for PredictProba it's n_classes. We
        // discover n_classes from the first proba estimator's prediction;
        // otherwise default to 1.

        // Generate meta-features per estimator first, accumulate into a Vec<Vec<f64>>
        // (one column per meta-feature, length n).
        let mut meta_cols: Vec<Array1<F>> = Vec::new();
        for (_name, est) in self.base_estimators.iter() {
            match est {
                BaseEstimator::Predict(b) => {
                    let mut col = Array1::<F>::zeros(n);
                    for (train_idx, test_idx) in &folds {
                        let x_train = select_rows(x, train_idx);
                        let y_train = select_elements(y, train_idx);
                        let x_test = select_rows(x, test_idx);
                        let fitted = b.fit_box(&x_train, &y_train)?;
                        let preds = fitted.predict_box(&x_test)?;
                        for (li, &gi) in test_idx.iter().enumerate() {
                            col[gi] = preds[li];
                        }
                    }
                    meta_cols.push(col);
                }
                BaseEstimator::PredictProba(b) => {
                    // Need to know n_classes; defer column creation until we
                    // see the first proba output.
                    let mut buf: Option<Array2<F>> = None;
                    for (train_idx, test_idx) in &folds {
                        let x_train = select_rows(x, train_idx);
                        let y_train = select_elements(y, train_idx);
                        let x_test = select_rows(x, test_idx);
                        let fitted = b.fit_proba_box(&x_train, &y_train)?;
                        let probs = fitted.predict_proba_box(&x_test)?;
                        let nc = probs.ncols();
                        let bufm = buf.get_or_insert_with(|| Array2::<F>::zeros((n, nc)));
                        for (li, &gi) in test_idx.iter().enumerate() {
                            for c in 0..nc {
                                bufm[[gi, c]] = probs[[li, c]];
                            }
                        }
                    }
                    if let Some(bufm) = buf {
                        for c in 0..bufm.ncols() {
                            meta_cols.push(bufm.column(c).to_owned());
                        }
                    }
                }
            }
        }

        let n_meta = meta_cols.len();
        let mut meta_features = Array2::<F>::zeros((n, n_meta));
        for (c, col) in meta_cols.iter().enumerate() {
            for i in 0..n {
                meta_features[[i, c]] = col[i];
            }
        }

        let fitted_meta = self.meta_estimator.fit_box(&meta_features, y)?;

        let mut fitted_base = Vec::with_capacity(self.base_estimators.len());
        for (name, est) in &self.base_estimators {
            let f = match est {
                BaseEstimator::Predict(b) => FittedBase::Predict(b.fit_box(x, y)?),
                BaseEstimator::PredictProba(b) => FittedBase::PredictProba(b.fit_proba_box(x, y)?),
            };
            fitted_base.push((name.clone(), f));
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
        let mut meta_cols: Vec<Array1<F>> = Vec::new();
        for (_name, m) in &self.fitted_base {
            match m {
                FittedBase::Predict(p) => {
                    meta_cols.push(p.predict_box(x)?);
                }
                FittedBase::PredictProba(p) => {
                    let probs = p.predict_proba_box(x)?;
                    for c in 0..probs.ncols() {
                        meta_cols.push(probs.column(c).to_owned());
                    }
                }
            }
        }
        let mut meta_features = Array2::<F>::zeros((n, meta_cols.len()));
        for (c, col) in meta_cols.iter().enumerate() {
            for i in 0..n {
                meta_features[[i, c]] = col[i];
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
            [0.0, 0.0],
            [5.0, 5.0],
            [0.1, 0.1],
            [5.1, 5.0],
            [0.2, -0.1],
            [4.9, 5.1],
            [-0.1, 0.2],
            [5.2, 4.8],
        ];
        let y = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];

        let sc = StackingClassifier::new(DecisionTreeClassifier::default())
            .push(
                "t1",
                DecisionTreeClassifier {
                    max_depth: Some(2),
                    ..Default::default()
                },
            )
            .push(
                "t2",
                DecisionTreeClassifier {
                    max_depth: Some(3),
                    ..Default::default()
                },
            )
            .with_cv_folds(2);

        let fitted: FittedStackingClassifier<f64> = sc.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_eq!(*p, *t, "p={p}, t={t}");
        }
    }

    #[test]
    fn test_stacking_classifier_proba_path() {
        // Stack two DT classifiers via predict_proba into a DT meta.
        let x = array![
            [0.0, 0.0],
            [5.0, 5.0],
            [0.1, 0.1],
            [5.1, 5.0],
            [0.2, -0.1],
            [4.9, 5.1],
            [-0.1, 0.2],
            [5.2, 4.8],
        ];
        let y = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];

        let sc = StackingClassifier::new(DecisionTreeClassifier::default())
            .push_proba(
                "t1",
                DecisionTreeClassifier {
                    max_depth: Some(2),
                    ..Default::default()
                },
            )
            .push_proba(
                "t2",
                DecisionTreeClassifier {
                    max_depth: Some(3),
                    ..Default::default()
                },
            )
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
