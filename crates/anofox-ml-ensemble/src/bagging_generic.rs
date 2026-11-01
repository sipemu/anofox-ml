//! Bagging around any base estimator (sklearn's `BaggingRegressor(estimator=...)`
//! / `BaggingClassifier(estimator=...)`).
//!
//! [`BaggingRegressor::with_estimator`] and
//! [`BaggingClassifier::with_estimator`] keep the sampling settings
//! (`n_estimators`, `max_samples`, `bootstrap`, `max_features`,
//! `bootstrap_features`, `oob_score`, `seed`) and replace the decision tree
//! by any estimator implementing [`Fit`] whose fitted model implements
//! [`Predict`]. As sklearn does for estimators without `sample_weight`, each
//! estimator is fitted on its drawn rows (repeated rows for a bootstrap) and
//! columns. With [`FitWeighted`] base estimators, `fit_weighted` passes the
//! drawn rows' weights on.
//!
//! The regressor averages the predictions. The classifier takes a majority
//! vote; `predict_proba` is the fraction of votes per class (sklearn's
//! behaviour for base estimators without `predict_proba`). The base
//! estimators are fitted as given, so a randomised base estimator should
//! carry its own seed.

use anofox_ml_core::{
    encode_labels, label_index, Fit, FitWeighted, Float, Predict, Result, RustMlError,
};
use ndarray::{Array1, Array2, Axis};
use rayon::prelude::*;

use crate::bagging_common::{out_of_bag, select_columns, BagPlan};
use crate::{BaggingClassifier, BaggingRegressor};

/// Bagging regressor around the base estimator `E`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GenericBaggingRegressor<E> {
    /// Sampling settings (the tree-specific `max_depth` is ignored).
    pub params: BaggingRegressor,
    pub estimator: E,
}

/// Bagging classifier around the base estimator `E`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GenericBaggingClassifier<E> {
    /// Sampling settings (the tree-specific `max_depth` is ignored).
    pub params: BaggingClassifier,
    pub estimator: E,
}

impl BaggingRegressor {
    /// Bag `estimator` instead of a decision tree, with these sampling
    /// settings.
    pub fn with_estimator<E>(self, estimator: E) -> GenericBaggingRegressor<E> {
        GenericBaggingRegressor {
            params: self,
            estimator,
        }
    }
}

impl BaggingClassifier {
    /// Bag `estimator` instead of a decision tree, with these sampling
    /// settings.
    pub fn with_estimator<E>(self, estimator: E) -> GenericBaggingClassifier<E> {
        GenericBaggingClassifier {
            params: self,
            estimator,
        }
    }
}

/// Fitted generic bagging ensemble (regressor or classifier).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedGenericBagging<M, F> {
    estimators: Vec<M>,
    /// Feature subset of each estimator (empty = all features).
    features: Vec<Vec<usize>>,
    n_features: usize,
    /// Sorted class labels (classifier only).
    classes: Vec<F>,
    /// Out-of-bag prediction (regressor) or vote fractions (classifier).
    oob: Option<Array2<F>>,
    oob_score: Option<f64>,
}

fn check_xy<F: Float>(x: &Array2<F>, y: &Array1<F>, w: Option<&Array1<F>>) -> Result<()> {
    if x.nrows() != y.len() {
        return Err(RustMlError::ShapeMismatch(format!(
            "X has {} rows but y has {} elements",
            x.nrows(),
            y.len()
        )));
    }
    if x.is_empty() {
        return Err(RustMlError::EmptyInput("training data is empty".into()));
    }
    if let Some(w) = w {
        anofox_ml_core::check_sample_weight(w, x.nrows())?;
    }
    Ok(())
}

/// Fit one base estimator per plan, in parallel. `fit_one` gets the drawn
/// rows and columns of `x`, `y` and (if given) the weights.
fn fit_all<F, M, G>(
    plans: &[BagPlan],
    x: &Array2<F>,
    y: &Array1<F>,
    w: Option<&Array1<F>>,
    fit_one: G,
) -> Result<Vec<M>>
where
    F: Float,
    M: Send,
    G: Fn(&Array2<F>, &Array1<F>, Option<&Array1<F>>) -> Result<M> + Sync,
{
    plans
        .par_iter()
        .map(|plan| {
            let xs = select_columns(x, &plan.features).select(Axis(0), &plan.rows);
            let ys = y.select(Axis(0), &plan.rows);
            let ws = w.map(|w| w.select(Axis(0), &plan.rows));
            fit_one(&xs, &ys, ws.as_ref())
        })
        .collect()
}

impl<M, F: Float> FittedGenericBagging<M, F> {
    fn check_features(&self, x: &Array2<F>) -> Result<()> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        Ok(())
    }

    /// The fitted base estimators.
    pub fn estimators(&self) -> &[M] {
        &self.estimators
    }

    /// Feature subset each estimator was trained on (empty = all features).
    pub fn estimators_features(&self) -> &[Vec<usize>] {
        &self.features
    }

    /// Out-of-bag score (R² for the regressor, accuracy for the classifier)
    /// over the samples with at least one out-of-bag estimate; `None` unless
    /// fitted with `oob_score`.
    pub fn oob_score(&self) -> Option<f64> {
        self.oob_score
    }
}

impl<M: Predict<F> + Sync, F: Float> FittedGenericBagging<M, F> {
    /// Predictions of every estimator on its own feature subset.
    fn all_predictions(&self, x: &Array2<F>) -> Result<Vec<Array1<F>>> {
        self.check_features(x)?;
        self.estimators
            .par_iter()
            .zip(&self.features)
            .map(|(m, f)| m.predict(&select_columns(x, f)))
            .collect()
    }

    /// Out-of-bag predictions `(sample, prediction)` of every estimator.
    fn oob_predictions(
        estimators: &[M],
        plans: &[BagPlan],
        x: &Array2<F>,
    ) -> Result<Vec<(usize, F)>> {
        let mut out = Vec::new();
        for (plan, m) in plans.iter().zip(estimators) {
            let oob = out_of_bag(x.nrows(), &plan.rows);
            if oob.is_empty() {
                continue;
            }
            let xs = select_columns(x, &plan.features).select(Axis(0), &oob);
            out.extend(oob.iter().copied().zip(m.predict(&xs)?));
        }
        Ok(out)
    }
}

// ----------------------------------------------------------------- regressor

impl<E> GenericBaggingRegressor<E> {
    fn finish<F: Float, M: Predict<F> + Sync>(
        &self,
        plans: &[BagPlan],
        estimators: Vec<M>,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<FittedGenericBagging<M, F>> {
        let n = x.nrows();
        let (oob, oob_score) = if self.params.oob_score {
            let mut sum = vec![F::zero(); n];
            let mut count = vec![0usize; n];
            for (i, p) in FittedGenericBagging::oob_predictions(&estimators, plans, x)? {
                sum[i] += p;
                count[i] += 1;
            }
            let pred = Array2::from_shape_fn((n, 1), |(i, _)| {
                if count[i] > 0 {
                    sum[i] / F::from_usize(count[i]).unwrap()
                } else {
                    F::nan()
                }
            });
            let score = r2_where_defined(y, &pred.column(0).to_owned());
            (Some(pred), Some(score))
        } else {
            (None, None)
        };
        Ok(FittedGenericBagging {
            estimators,
            features: plans.iter().map(|p| p.features.clone()).collect(),
            n_features: x.ncols(),
            classes: Vec::new(),
            oob,
            oob_score,
        })
    }
}

impl<F, E> Fit<F> for GenericBaggingRegressor<E>
where
    F: Float,
    E: Fit<F> + Sync,
    E::Fitted: Predict<F> + Send + Sync,
{
    type Fitted = FittedGenericBagging<E::Fitted, F>;

    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Self::Fitted> {
        check_xy(x, y, None)?;
        let plans = self.params.sampling().plans(x.nrows(), x.ncols())?;
        let est = fit_all(&plans, x, y, None, |xs, ys, _| self.estimator.fit(xs, ys))?;
        self.finish(&plans, est, x, y)
    }
}

impl<F, E> FitWeighted<F> for GenericBaggingRegressor<E>
where
    F: Float,
    E: FitWeighted<F> + Sync,
    E::Fitted: Predict<F> + Send + Sync,
{
    type Fitted = FittedGenericBagging<E::Fitted, F>;

    fn fit_weighted(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        sample_weight: Option<&Array1<F>>,
    ) -> Result<Self::Fitted> {
        check_xy(x, y, sample_weight)?;
        let plans = self.params.sampling().plans(x.nrows(), x.ncols())?;
        let est = fit_all(&plans, x, y, sample_weight, |xs, ys, ws| {
            self.estimator.fit_weighted(xs, ys, ws)
        })?;
        self.finish(&plans, est, x, y)
    }
}

impl<M: Predict<F> + Sync, F: Float> FittedGenericBagging<M, F> {
    /// Out-of-bag prediction of every training sample, shape `(n, 1)` for a
    /// regressor (NaN where no estimator left the sample out), or the
    /// out-of-bag vote fractions `(n, n_classes)` for a classifier; `None`
    /// unless fitted with `oob_score`.
    pub fn oob_decision(&self) -> Option<&Array2<F>> {
        self.oob.as_ref()
    }
}

/// R² over the samples whose prediction is not NaN.
fn r2_where_defined<F: Float>(y: &Array1<F>, pred: &Array1<F>) -> f64 {
    let pairs: Vec<(f64, f64)> = y
        .iter()
        .zip(pred)
        .filter(|(_, p)| !p.is_nan())
        .map(|(t, p)| (t.to_f64().unwrap(), p.to_f64().unwrap()))
        .collect();
    if pairs.is_empty() {
        return f64::NAN;
    }
    let mean = pairs.iter().map(|p| p.0).sum::<f64>() / pairs.len() as f64;
    let ss_res: f64 = pairs.iter().map(|(t, p)| (t - p).powi(2)).sum();
    let ss_tot: f64 = pairs.iter().map(|(t, _)| (t - mean).powi(2)).sum();
    if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    }
}

// ---------------------------------------------------------------- classifier

impl<E> GenericBaggingClassifier<E> {
    fn finish<F: Float, M: Predict<F> + Sync>(
        &self,
        plans: &[BagPlan],
        estimators: Vec<M>,
        x: &Array2<F>,
        y: &Array1<F>,
        classes: Vec<F>,
    ) -> Result<FittedGenericBagging<M, F>> {
        let n = x.nrows();
        let k = classes.len();
        let (oob, oob_score) = if self.params.oob_score {
            let mut votes = Array2::<F>::zeros((n, k));
            for (i, p) in FittedGenericBagging::oob_predictions(&estimators, plans, x)? {
                if let Some(c) = label_index(&classes, p) {
                    votes[[i, c]] += F::one();
                }
            }
            let (mut correct, mut seen) = (0usize, 0usize);
            for (i, mut row) in votes.rows_mut().into_iter().enumerate() {
                let total = row.sum();
                if total > F::zero() {
                    seen += 1;
                    if classes[argmax(row.view())] == y[i] {
                        correct += 1;
                    }
                    row /= total;
                } else {
                    row.fill(F::nan());
                }
            }
            let score = if seen > 0 {
                correct as f64 / seen as f64
            } else {
                f64::NAN
            };
            (Some(votes), Some(score))
        } else {
            (None, None)
        };
        Ok(FittedGenericBagging {
            estimators,
            features: plans.iter().map(|p| p.features.clone()).collect(),
            n_features: x.ncols(),
            classes,
            oob,
            oob_score,
        })
    }
}

/// First index of the largest value (ties go to the smaller class, as
/// numpy's `argmax`).
fn argmax<F: Float>(row: ndarray::ArrayView1<F>) -> usize {
    let mut best = 0;
    for (j, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = j;
        }
    }
    best
}

impl<F, E> Fit<F> for GenericBaggingClassifier<E>
where
    F: Float,
    E: Fit<F> + Sync,
    E::Fitted: Predict<F> + Send + Sync,
{
    type Fitted = FittedGenericBagging<E::Fitted, F>;

    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Self::Fitted> {
        check_xy(x, y, None)?;
        let (classes, _) = encode_labels(y)?;
        let plans = self.params.sampling().plans(x.nrows(), x.ncols())?;
        let est = fit_all(&plans, x, y, None, |xs, ys, _| self.estimator.fit(xs, ys))?;
        self.finish(&plans, est, x, y, classes)
    }
}

impl<F, E> FitWeighted<F> for GenericBaggingClassifier<E>
where
    F: Float,
    E: FitWeighted<F> + Sync,
    E::Fitted: Predict<F> + Send + Sync,
{
    type Fitted = FittedGenericBagging<E::Fitted, F>;

    fn fit_weighted(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        sample_weight: Option<&Array1<F>>,
    ) -> Result<Self::Fitted> {
        check_xy(x, y, sample_weight)?;
        let (classes, _) = encode_labels(y)?;
        let plans = self.params.sampling().plans(x.nrows(), x.ncols())?;
        let est = fit_all(&plans, x, y, sample_weight, |xs, ys, ws| {
            self.estimator.fit_weighted(xs, ys, ws)
        })?;
        self.finish(&plans, est, x, y, classes)
    }
}

impl<M: Predict<F> + Sync, F: Float> FittedGenericBagging<M, F> {
    /// Classifier: fraction of the estimators voting for each class, shape
    /// `(n, n_classes)` in the order of [`classes`](Self::classes).
    pub fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>> {
        if self.classes.is_empty() {
            return Err(RustMlError::InvalidParameter(
                "predict_proba is only available for a bagging classifier".into(),
            ));
        }
        let preds = self.all_predictions(x)?;
        let mut votes = Array2::<F>::zeros((x.nrows(), self.classes.len()));
        for p in &preds {
            for (i, &v) in p.iter().enumerate() {
                if let Some(c) = label_index(&self.classes, v) {
                    votes[[i, c]] += F::one();
                }
            }
        }
        votes /= F::from_usize(preds.len()).unwrap();
        Ok(votes)
    }

    /// Classifier: sorted class labels (empty for a regressor).
    pub fn classes(&self) -> &[F] {
        &self.classes
    }
}

impl<M: Predict<F> + Sync, F: Float> Predict<F> for FittedGenericBagging<M, F> {
    /// Regressor: mean prediction. Classifier: the class with the most votes
    /// (ties to the smaller label).
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        if self.classes.is_empty() {
            let preds = self.all_predictions(x)?;
            let mut sum = Array1::<F>::zeros(x.nrows());
            for p in &preds {
                sum += p;
            }
            Ok(sum / F::from_usize(preds.len()).unwrap())
        } else {
            let votes = self.predict_proba(x)?;
            Ok(votes
                .rows()
                .into_iter()
                .map(|r| self.classes[argmax(r)])
                .collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anofox_ml_core::FitWeighted;
    use anofox_ml_neighbors::KnnClassifier;
    use anofox_ml_regression::OlsRegressor;
    use approx::assert_abs_diff_eq;

    fn data() -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let x = Array2::from_shape_fn((60, 3), |(i, j)| {
            let t = i as f64;
            ((t * (0.7 + j as f64 * 0.31)).sin() * 2.0) + j as f64
        });
        let y = x.column(0).mapv(|v| 2.0 * v) - &x.column(1)
            + 0.5
            + x.column(2).mapv(|v| (v * 3.0).sin() * 0.1);
        let yc = y.mapv(|v| if v > 0.5 { 1.0 } else { 0.0 });
        (x, y, yc)
    }

    #[test]
    fn test_no_resampling_equals_base_estimator() {
        // bootstrap = false, all rows and features: every estimator is the
        // base estimator fitted on all the data.
        let (x, y, yc) = data();
        let bag = BaggingRegressor::new(4)
            .with_bootstrap(false)
            .with_estimator(OlsRegressor::new());
        let f = bag.fit(&x, &y).unwrap();
        let ols = OlsRegressor::new().fit(&x, &y).unwrap();
        for (a, b) in f
            .predict(&x)
            .unwrap()
            .iter()
            .zip(ols.predict(&x).unwrap().iter())
        {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-9);
        }
        let bag = BaggingClassifier::new(3)
            .with_bootstrap(false)
            .with_estimator(KnnClassifier::new(3));
        let f = bag.fit(&x, &yc).unwrap();
        let knn = KnnClassifier::new(3).fit(&x, &yc).unwrap();
        assert_eq!(f.predict(&x).unwrap(), knn.predict(&x).unwrap());
        assert_eq!(f.classes(), &[0.0, 1.0]);
    }

    #[test]
    fn test_bootstrap_ensemble_and_oob() {
        let (x, y, yc) = data();
        let bag = BaggingRegressor::new(25)
            .with_oob_score(true)
            .with_max_features(Some(anofox_ml_trees::MaxFeatures::Fixed(2)), false)
            .with_seed(3)
            .with_estimator(OlsRegressor::new());
        let f = bag.fit(&x, &y).unwrap();
        assert_eq!(f.estimators().len(), 25);
        assert!(f.estimators_features().iter().all(|s| s.len() == 2));
        assert!(f.oob_score().unwrap() > 0.5, "{:?}", f.oob_score());
        // Reproducible with the same seed.
        assert_eq!(
            f.predict(&x).unwrap(),
            bag.fit(&x, &y).unwrap().predict(&x).unwrap()
        );

        let bag = BaggingClassifier::new(15)
            .with_oob_score(true)
            .with_seed(1)
            .with_estimator(KnnClassifier::new(5));
        let f = bag.fit(&x, &yc).unwrap();
        let p = f.predict_proba(&x).unwrap();
        for r in p.rows() {
            assert_abs_diff_eq!(r.sum(), 1.0, epsilon = 1e-12);
        }
        let acc = f
            .predict(&x)
            .unwrap()
            .iter()
            .zip(&yc)
            .filter(|(a, b)| a == b)
            .count() as f64
            / 60.0;
        assert!(acc > 0.8 && f.oob_score().unwrap() > 0.7);
        assert_eq!(f.oob_decision().unwrap().dim(), (60, 2));
    }

    #[test]
    fn test_weighted_base_estimator() {
        // Zero weight on half the rows equals fitting on the other half
        // (no resampling).
        let (x, y, _) = data();
        let w = Array1::from_shape_fn(60, |i| if i < 30 { 1.0 } else { 0.0 });
        let bag = BaggingRegressor::new(2)
            .with_bootstrap(false)
            .with_estimator(OlsRegressor::new());
        let f = bag.fit_weighted(&x, &y, Some(&w)).unwrap();
        let half = ndarray::s![..30, ..];
        let ols = OlsRegressor::new()
            .fit(
                &x.slice(half).to_owned(),
                &y.slice(ndarray::s![..30]).to_owned(),
            )
            .unwrap();
        for (a, b) in f
            .predict(&x)
            .unwrap()
            .iter()
            .zip(ols.predict(&x).unwrap().iter())
        {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-8);
        }
    }
}
