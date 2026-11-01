//! Stacking classifier: two-level ensemble where base classifiers' predictions
//! are used as features for a meta-classifier.
//!
//! Mirrors `sklearn.ensemble.StackingClassifier` with an explicit
//! `stack_method` per base estimator (`'predict'` via [`StackingClassifier::push`],
//! `'predict_proba'` via [`StackingClassifier::push_proba`]). Out-of-fold
//! predictions are generated with stratified, shuffled k-fold CV during
//! fitting to avoid leakage.

use crate::util::{FitPredBox, PredBox};
use anofox_ml_core::utils::stratified_k_fold;
use anofox_ml_core::{Fit, Float, Predict, PredictProba, Result, RustMlError};
use ndarray::{Array1, Array2};

use crate::forest_common::{class_index, select_rows, sorted_classes};

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

trait FitProbaBox<F: Float>: Send + Sync {
    fn fit_proba_box(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Box<dyn ProbaBox<F>>>;
}

trait ProbaBox<F: Float>: Send + Sync {
    fn predict_proba_box(&self, x: &Array2<F>) -> Result<Array2<F>>;
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
/// or class probabilities: `n_classes` meta-features each, except for binary
/// problems where only `P(classes[1])` is kept (the two columns are perfectly
/// collinear — sklearn drops the first one).
///
/// The class list is taken from the full `y`, so the meta-feature width is
/// fixed even when a CV training fold misses a class (that class then gets
/// probability 0 for the fold, as in sklearn's `cross_val_predict`).
/// Base estimators' `predict_proba` columns must follow ascending class
/// order (the convention of every classifier in this workspace).
pub struct StackingClassifier<F: Float> {
    base_estimators: Vec<(String, BaseEstimator<F>)>,
    meta_estimator: Box<dyn FitPredBox<F>>,
    cv_folds: usize,
    seed: u64,
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
            seed: 0,
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

    /// Number of stratified CV folds used to build the meta-features
    /// (default 5).
    pub fn with_cv_folds(mut self, k: usize) -> Self {
        self.cv_folds = k;
        self
    }

    /// Seed for shuffling the stratified CV folds (default 0).
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

pub struct FittedStackingClassifier<F: Float> {
    fitted_base: Vec<(String, FittedBase<F>)>,
    fitted_meta: Box<dyn PredBox<F>>,
    n_features: usize,
    classes: Vec<F>,
}

enum FittedBase<F: Float> {
    Predict(Box<dyn PredBox<F>>),
    PredictProba(Box<dyn ProbaBox<F>>),
}

impl<F: Float> FittedStackingClassifier<F> {
    pub fn estimator_names(&self) -> Vec<&str> {
        self.fitted_base.iter().map(|(n, _)| n.as_str()).collect()
    }

    /// Class labels (sorted) seen in the full training `y`.
    pub fn classes(&self) -> &[F] {
        &self.classes
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

        let classes = sorted_classes(y);
        if classes.len() < 2 {
            return Err(RustMlError::InvalidParameter(
                "StackingClassifier needs at least 2 classes".into(),
            ));
        }
        let k = self.cv_folds.min(n);
        let folds = stratified_k_fold(x, y, k, self.seed)?;

        // Out-of-fold meta-features, one column per meta-feature, length n.
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
                    // Width pinned to the classes of the full `y`.
                    let mut buf = Array2::<F>::zeros((n, classes.len()));
                    for (train_idx, test_idx) in &folds {
                        let x_train = select_rows(x, train_idx);
                        let y_train = select_elements(y, train_idx);
                        let x_test = select_rows(x, test_idx);
                        let fitted = b.fit_proba_box(&x_train, &y_train)?;
                        let probs = fitted.predict_proba_box(&x_test)?;
                        let col_map = proba_column_map(&classes, &y_train, probs.ncols())?;
                        for (li, &gi) in test_idx.iter().enumerate() {
                            for (c, &gc) in col_map.iter().enumerate() {
                                buf[[gi, gc]] = probs[[li, c]];
                            }
                        }
                    }
                    push_proba_columns(&mut meta_cols, &buf);
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
            classes,
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
                    if probs.ncols() != self.classes.len() {
                        return Err(RustMlError::ShapeMismatch(format!(
                            "base estimator predict_proba returned {} columns, expected {}",
                            probs.ncols(),
                            self.classes.len()
                        )));
                    }
                    push_proba_columns(&mut meta_cols, &probs);
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

/// Map the `predict_proba` columns of an estimator fitted on `y_train` to
/// positions in the global class list. A fold whose training part misses a
/// class yields fewer columns; those are scattered to their class positions
/// (missing classes keep probability 0).
fn proba_column_map<F: Float>(
    classes: &[F],
    y_train: &Array1<F>,
    n_cols: usize,
) -> Result<Vec<usize>> {
    if n_cols == classes.len() {
        return Ok((0..n_cols).collect());
    }
    let fold_classes = sorted_classes(y_train);
    if n_cols != fold_classes.len() {
        return Err(RustMlError::ShapeMismatch(format!(
            "base estimator predict_proba returned {} columns, but its training fold has {} classes \
             ({} overall)",
            n_cols,
            fold_classes.len(),
            classes.len()
        )));
    }
    Ok(fold_classes
        .iter()
        .map(|&c| class_index(classes, c).expect("fold class is a training class"))
        .collect())
}

/// Append probability meta-features: all columns for multiclass, only the
/// positive-class column for binary problems (sklearn drops the first,
/// perfectly collinear column).
fn push_proba_columns<F: Float>(meta_cols: &mut Vec<Array1<F>>, probs: &Array2<F>) {
    let first = usize::from(probs.ncols() == 2);
    for c in first..probs.ncols() {
        meta_cols.push(probs.column(c).to_owned());
    }
}

fn select_elements<F: Float>(y: &Array1<F>, indices: &[usize]) -> Array1<F> {
    Array1::from_vec(indices.iter().map(|&i| y[i]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use anofox_ml_trees::DecisionTreeClassifier;
    use ndarray::array;

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

    /// Class-sorted labels: contiguous unshuffled folds would leave whole
    /// classes out of a training fold, so the proba width used to vary per
    /// fold (overflowing / shifting the meta-feature buffer).
    fn sorted_three_class() -> (Array2<f64>, Array1<f64>) {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for c in 0..3 {
            for i in 0..6 {
                xs.push([c as f64 * 5.0 + i as f64 * 0.1, (i % 3) as f64]);
                ys.push(c as f64);
            }
        }
        let x = Array2::from_shape_vec((18, 2), xs.into_iter().flatten().collect()).unwrap();
        (x, Array1::from_vec(ys))
    }

    #[test]
    fn test_stacking_proba_class_sorted_data_stable_width() {
        let (x, y) = sorted_three_class();
        let sc = StackingClassifier::new(DecisionTreeClassifier::default())
            .push_proba("t1", DecisionTreeClassifier::default())
            .with_cv_folds(3);
        let fitted: FittedStackingClassifier<f64> = sc.fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[0.0, 1.0, 2.0]);
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds, y);
    }

    #[test]
    fn test_stacking_stratified_folds_contain_every_class() {
        let (x, y) = sorted_three_class();
        let folds = stratified_k_fold(&x, &y, 3, 0).unwrap();
        for (train, test) in &folds {
            for c in 0..3 {
                assert!(train.iter().any(|&i| y[i] == c as f64));
                assert!(test.iter().any(|&i| y[i] == c as f64));
            }
        }
    }

    #[test]
    fn test_proba_column_map_scatters_missing_class() {
        let classes = [0.0, 1.0, 2.0];
        let y_train = array![0.0, 2.0, 2.0];
        assert_eq!(proba_column_map(&classes, &y_train, 2).unwrap(), vec![0, 2]);
        assert_eq!(
            proba_column_map(&classes, &y_train, 3).unwrap(),
            vec![0, 1, 2]
        );
        assert!(proba_column_map(&classes, &y_train, 1).is_err());
    }

    #[test]
    fn test_binary_proba_drops_first_column() {
        let probs = array![[0.2, 0.8], [0.6, 0.4]];
        let mut cols = Vec::new();
        push_proba_columns(&mut cols, &probs);
        assert_eq!(cols, vec![array![0.8, 0.4]]);
        let probs3 = array![[0.2, 0.3, 0.5]];
        let mut cols = Vec::new();
        push_proba_columns(&mut cols, &probs3);
        assert_eq!(cols.len(), 3);
    }

    #[test]
    fn test_stacking_classifier_empty_base_error() {
        let x = array![[1.0], [2.0]];
        let y = array![0.0, 1.0];

        let sc = StackingClassifier::<f64>::new(DecisionTreeClassifier::default());
        assert!(sc.fit(&x, &y).is_err());
    }
}
