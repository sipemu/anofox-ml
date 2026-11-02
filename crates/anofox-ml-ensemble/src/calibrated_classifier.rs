//! CalibratedClassifierCV — probability calibration for classifiers.
//!
//! Wraps a classifier to produce well-calibrated probabilities using either
//! Platt scaling (sigmoid) or isotonic regression, fitted on out-of-fold
//! **scores** of the base model (its positive-class probability or its
//! decision function), mirroring `sklearn.calibration.CalibratedClassifierCV`
//! with `ensemble=False`.
//!
//! Only binary problems are supported; fitting on more than two classes
//! returns [`RustMlError::InvalidParameter`].

use anofox_ml_core::{DecisionFunction, Fit, Float, Predict, PredictProba, Result, RustMlError};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, AsArray, Ix2};

/// Calibration method.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub enum CalibrationMethod {
    /// Platt scaling: fits a sigmoid `1 / (1 + exp(A*f + B))` to map scores to
    /// probabilities.
    #[default]
    Sigmoid,
    /// Isotonic regression: non-parametric monotonic mapping.
    Isotonic,
}

/// Type-erased "fit, then produce a real-valued positive-class score".
trait ScoreFit<F: Float>: Send + Sync {
    fn fit_box(&self, x: ArrayView2<F>, y: ArrayView1<F>) -> Result<Box<dyn ScoreModel<F>>>;
}

/// A fitted model that maps samples to positive-class scores.
trait ScoreModel<F: Float>: Send + Sync {
    fn scores(&self, x: ArrayView2<F>) -> Result<Vec<f64>>;
}

/// Base estimator whose fitted model provides `predict_proba`; the score is
/// the probability of the larger (positive) class, i.e. column 1.
struct ProbaScorer<T>(T);
/// Base estimator whose fitted model provides `decision_function`.
struct DecisionScorer<T>(T);

struct ProbaModel<M>(M);
struct DecisionModel<M>(M);

impl<F, T> ScoreFit<F> for ProbaScorer<T>
where
    F: Float,
    T: Fit<F> + Send + Sync,
    T::Fitted: PredictProba<F> + Send + Sync + 'static,
{
    fn fit_box(&self, x: ArrayView2<F>, y: ArrayView1<F>) -> Result<Box<dyn ScoreModel<F>>> {
        Ok(Box::new(ProbaModel(Fit::fit(&self.0, x, y)?)))
    }
}

impl<F, M> ScoreModel<F> for ProbaModel<M>
where
    F: Float,
    M: PredictProba<F> + Send + Sync,
{
    fn scores(&self, x: ArrayView2<F>) -> Result<Vec<f64>> {
        let proba = self.0.predict_proba(x)?;
        if proba.ncols() != 2 {
            return Err(RustMlError::InvalidParameter(format!(
                "base estimator returned {} probability columns; binary calibration needs 2",
                proba.ncols()
            )));
        }
        Ok(proba
            .column(1)
            .iter()
            .map(|v| v.to_f64().unwrap())
            .collect())
    }
}

impl<F, T> ScoreFit<F> for DecisionScorer<T>
where
    F: Float,
    T: Fit<F> + Send + Sync,
    T::Fitted: DecisionFunction<F> + Send + Sync + 'static,
{
    fn fit_box(&self, x: ArrayView2<F>, y: ArrayView1<F>) -> Result<Box<dyn ScoreModel<F>>> {
        Ok(Box::new(DecisionModel(Fit::fit(&self.0, x, y)?)))
    }
}

impl<F, M> ScoreModel<F> for DecisionModel<M>
where
    F: Float,
    M: DecisionFunction<F> + Send + Sync,
{
    fn scores(&self, x: ArrayView2<F>) -> Result<Vec<f64>> {
        let df = self.0.decision_function(x)?;
        match df.ncols() {
            // Single binary score.
            1 => Ok(df.column(0).iter().map(|v| v.to_f64().unwrap()).collect()),
            // Per-class scores: margin of the positive over the negative class.
            2 => Ok(df
                .rows()
                .into_iter()
                .map(|r| (r[1] - r[0]).to_f64().unwrap())
                .collect()),
            k => Err(RustMlError::InvalidParameter(format!(
                "decision_function returned {k} columns; binary calibration needs 1 or 2"
            ))),
        }
    }
}

/// Calibrated classifier with cross-validation (binary only).
///
/// Out-of-fold scores of the base model (positive-class probability, or the
/// decision function) are used to fit the calibrator; the base model is then
/// refit on all data.
pub struct CalibratedClassifierCV<F: Float> {
    base_estimator: Box<dyn ScoreFit<F>>,
    method: CalibrationMethod,
    cv_folds: usize,
}

impl<F: Float> CalibratedClassifierCV<F> {
    /// Wrap a base estimator whose fitted model implements [`PredictProba`].
    /// The positive-class probability (column 1, i.e. the larger label) is
    /// calibrated.
    pub fn new<T>(base_estimator: T) -> Self
    where
        T: Fit<F> + Send + Sync + 'static,
        T::Fitted: PredictProba<F> + Send + Sync + 'static,
    {
        Self {
            base_estimator: Box::new(ProbaScorer(base_estimator)),
            method: CalibrationMethod::Sigmoid,
            cv_folds: 5,
        }
    }

    /// Wrap a base estimator whose fitted model implements
    /// [`DecisionFunction`] (e.g. margin classifiers). A one-column output is
    /// used as is; a two-column output uses `col1 - col0`.
    pub fn with_decision_function<T>(base_estimator: T) -> Self
    where
        T: Fit<F> + Send + Sync + 'static,
        T::Fitted: DecisionFunction<F> + Send + Sync + 'static,
    {
        Self {
            base_estimator: Box::new(DecisionScorer(base_estimator)),
            method: CalibrationMethod::Sigmoid,
            cv_folds: 5,
        }
    }

    pub fn with_method(mut self, method: CalibrationMethod) -> Self {
        self.method = method;
        self
    }

    /// Number of stratified CV folds (clamped to the size of the smallest
    /// class; at least 2 samples per class are required).
    pub fn with_cv_folds(mut self, cv_folds: usize) -> Self {
        self.cv_folds = cv_folds;
        self
    }
}

/// Fitted calibrated classifier.
pub struct FittedCalibratedClassifier<F: Float> {
    /// Base model fitted on full data.
    base_model: Box<dyn ScoreModel<F>>,
    /// Platt parameters: `P(y=1|f) = 1 / (1 + exp(a*f + b))`.
    cal_a: f64,
    cal_b: f64,
    /// For isotonic: sorted (score, prob) knots.
    isotonic_x: Vec<f64>,
    isotonic_y: Vec<f64>,
    method: CalibrationMethod,
    n_features: usize,
    /// Sorted class labels `[negative, positive]`.
    classes: [F; 2],
}

impl<F: Float> FittedCalibratedClassifier<F> {
    /// Calibrated probability of the positive class (`classes()[1]`).
    pub fn predict_proba_positive<'a>(&self, x: impl AsArray<'a, F, Ix2>) -> Result<Array1<F>> {
        let x: ArrayView2<F> = x.into();
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }

        let scores = self.base_model.scores(x)?;
        Ok(Array1::from_iter(scores.into_iter().map(|score| {
            let p = match self.method {
                CalibrationMethod::Sigmoid => platt_predict(score, self.cal_a, self.cal_b),
                CalibrationMethod::Isotonic => {
                    isotonic_predict(score, &self.isotonic_x, &self.isotonic_y)
                }
            };
            F::from_f64(p.clamp(0.0, 1.0)).unwrap()
        })))
    }

    /// Calibrated class probabilities, shape `(n_samples, 2)`; columns follow
    /// [`classes`](Self::classes).
    pub fn predict_proba<'a>(&self, x: impl AsArray<'a, F, Ix2>) -> Result<Array2<F>> {
        let x: ArrayView2<F> = x.into();
        let p1 = self.predict_proba_positive(x)?;
        let mut out = Array2::<F>::zeros((p1.len(), 2));
        for (i, &p) in p1.iter().enumerate() {
            out[[i, 0]] = F::one() - p;
            out[[i, 1]] = p;
        }
        Ok(out)
    }

    /// Sorted class labels `[negative, positive]`.
    pub fn classes(&self) -> &[F] {
        &self.classes
    }

    /// Fitted Platt parameters `(A, B)` (only meaningful for
    /// [`CalibrationMethod::Sigmoid`]).
    pub fn platt_params(&self) -> (f64, f64) {
        (self.cal_a, self.cal_b)
    }
}

impl<F: Float> PredictProba<F> for FittedCalibratedClassifier<F> {
    fn predict_proba_view(&self, x: ArrayView2<F>) -> Result<Array2<F>> {
        Self::predict_proba(self, x)
    }
}

impl<F: Float + 'static> Fit<F> for CalibratedClassifierCV<F> {
    type Fitted = FittedCalibratedClassifier<F>;

    fn fit_view(&self, x: ArrayView2<F>, y: ArrayView1<F>) -> Result<Self::Fitted> {
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

        let all: Vec<usize> = (0..n).collect();
        let class_counts = anofox_ml_trees::split::count_classes(y, &all);
        match class_counts.len() {
            2 => {}
            k if k < 2 => {
                return Err(RustMlError::InvalidParameter(
                    "CalibratedClassifierCV needs 2 classes, got 1".into(),
                ))
            }
            k => {
                return Err(RustMlError::InvalidParameter(format!(
                "CalibratedClassifierCV currently supports binary problems only, got {k} classes"
            )))
            }
        }
        let classes = [class_counts[0].0, class_counts[1].0];
        let min_class = class_counts.iter().map(|&(_, c)| c).min().unwrap();
        let k = self.cv_folds.min(min_class);
        if k < 2 {
            return Err(RustMlError::InvalidParameter(format!(
                "need cv_folds >= 2 and at least 2 samples per class (smallest class has {min_class})"
            )));
        }

        // Out-of-fold scores via stratified folds (every training fold
        // contains both classes).
        let folds = stratified_k_fold(y, &classes, k);
        let mut oof_scores = vec![0.0f64; n];
        let labels: Vec<f64> = y
            .iter()
            .map(|&v| if v == classes[1] { 1.0 } else { 0.0 })
            .collect();

        for (train_idx, test_idx) in &folds {
            let x_train = crate::forest_common::select_rows(x, train_idx);
            let y_train = select_elements(y, train_idx);
            let x_test = crate::forest_common::select_rows(x, test_idx);

            let fitted = self
                .base_estimator
                .fit_box(x_train.view(), y_train.view())?;
            let scores = fitted.scores(x_test.view())?;

            for (li, &gi) in test_idx.iter().enumerate() {
                oof_scores[gi] = scores[li];
            }
        }

        // Fit calibration mapping on OOF scores.
        let (cal_a, cal_b, isotonic_x, isotonic_y) = match self.method {
            CalibrationMethod::Sigmoid => {
                let (a, b) = fit_platt_sigmoid(&oof_scores, &labels);
                (a, b, Vec::new(), Vec::new())
            }
            CalibrationMethod::Isotonic => {
                let (ix, iy) = fit_isotonic(&oof_scores, &labels);
                (0.0, 0.0, ix, iy)
            }
        };

        // Refit base model on full data.
        let base_model = self.base_estimator.fit_box(x, y)?;

        Ok(FittedCalibratedClassifier {
            base_model,
            cal_a,
            cal_b,
            isotonic_x,
            isotonic_y,
            method: self.method,
            n_features: x.ncols(),
            classes,
        })
    }
}

impl<F: Float> Predict<F> for FittedCalibratedClassifier<F> {
    /// Predict the original class labels (positive class when the calibrated
    /// probability is `>= 0.5`).
    fn predict_view(&self, x: ArrayView2<F>) -> Result<Array1<F>> {
        let proba = self.predict_proba_positive(x)?;
        let threshold = F::from_f64(0.5).unwrap();
        let [neg, pos] = self.classes;
        Ok(proba.mapv(|p| if p >= threshold { pos } else { neg }))
    }
}

/// Numerically stable `1 / (1 + exp(a*f + b))`.
#[inline]
fn platt_predict(f: f64, a: f64, b: f64) -> f64 {
    let fapb = a * f + b;
    if fapb >= 0.0 {
        let e = (-fapb).exp();
        e / (1.0 + e)
    } else {
        1.0 / (1.0 + fapb.exp())
    }
}

/// Fit Platt scaling `P(y=1|f) = 1 / (1 + exp(A*f + B))` by Newton's method
/// with backtracking line search on Platt's prior-corrected targets
/// (Platt 1999; Lin, Lin & Weng 2007 — libsvm's `sigmoid_train`).
fn fit_platt_sigmoid(scores: &[f64], labels: &[f64]) -> (f64, f64) {
    let n = scores.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let prior1 = labels.iter().filter(|&&l| l > 0.5).count() as f64;
    let prior0 = n as f64 - prior1;

    let max_iter = 100;
    let min_step = 1e-10;
    let sigma = 1e-12; // keeps the Hessian positive definite
    let eps = 1e-5;

    let hi_target = (prior1 + 1.0) / (prior1 + 2.0);
    let lo_target = 1.0 / (prior0 + 2.0);
    let t: Vec<f64> = labels
        .iter()
        .map(|&l| if l > 0.5 { hi_target } else { lo_target })
        .collect();

    let mut a = 0.0f64;
    let mut b = ((prior0 + 1.0) / (prior1 + 1.0)).ln();

    let objective = |a: f64, b: f64| -> f64 {
        let mut fval = 0.0;
        for i in 0..n {
            let fapb = scores[i] * a + b;
            if fapb >= 0.0 {
                fval += t[i] * fapb + (1.0 + (-fapb).exp()).ln();
            } else {
                fval += (t[i] - 1.0) * fapb + (1.0 + fapb.exp()).ln();
            }
        }
        fval
    };
    let mut fval = objective(a, b);

    for _ in 0..max_iter {
        // Gradient and Hessian.
        let (mut h11, mut h22, mut h21) = (sigma, sigma, 0.0);
        let (mut g1, mut g2) = (0.0, 0.0);
        for i in 0..n {
            let fapb = scores[i] * a + b;
            let (p, q) = if fapb >= 0.0 {
                let e = (-fapb).exp();
                (e / (1.0 + e), 1.0 / (1.0 + e))
            } else {
                let e = fapb.exp();
                (1.0 / (1.0 + e), e / (1.0 + e))
            };
            let d2 = p * q;
            h11 += scores[i] * scores[i] * d2;
            h22 += d2;
            h21 += scores[i] * d2;
            let d1 = t[i] - p;
            g1 += scores[i] * d1;
            g2 += d1;
        }

        if g1.abs() < eps && g2.abs() < eps {
            break;
        }

        // Newton direction.
        let det = h11 * h22 - h21 * h21;
        let da = -(h22 * g1 - h21 * g2) / det;
        let db = -(-h21 * g1 + h11 * g2) / det;
        let gd = g1 * da + g2 * db;

        // Backtracking line search (Armijo).
        let mut step = 1.0;
        let mut accepted = false;
        while step >= min_step {
            let new_a = a + step * da;
            let new_b = b + step * db;
            let new_f = objective(new_a, new_b);
            if new_f < fval + 1e-4 * step * gd {
                a = new_a;
                b = new_b;
                fval = new_f;
                accepted = true;
                break;
            }
            step /= 2.0;
        }
        if !accepted {
            break;
        }
    }

    (a, b)
}

/// Fit isotonic regression (pool adjacent violators) of `labels` on
/// `scores`. Tied scores are pooled first so the fit is order-independent.
fn fit_isotonic(scores: &[f64], labels: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = scores.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut pairs: Vec<(f64, f64)> = scores
        .iter()
        .zip(labels.iter())
        .map(|(&s, &l)| (s, l))
        .collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

    // (x, mean y, weight) blocks; equal x values start pooled.
    let mut x_out: Vec<f64> = Vec::with_capacity(n);
    let mut y_out: Vec<f64> = Vec::with_capacity(n);
    let mut weights: Vec<f64> = Vec::with_capacity(n);

    let mut i = 0;
    while i < pairs.len() {
        let xi = pairs[i].0;
        let mut j = i;
        let mut ysum = 0.0;
        while j < pairs.len() && pairs[j].0 == xi {
            ysum += pairs[j].1;
            j += 1;
        }
        let w = (j - i) as f64;
        x_out.push(xi);
        y_out.push(ysum / w);
        weights.push(w);
        i = j;

        while y_out.len() >= 2 {
            let len = y_out.len();
            if y_out[len - 2] > y_out[len - 1] {
                let w1 = weights[len - 2];
                let w2 = weights[len - 1];
                let merged = (y_out[len - 2] * w1 + y_out[len - 1] * w2) / (w1 + w2);
                let merged_x = (x_out[len - 2] * w1 + x_out[len - 1] * w2) / (w1 + w2);
                y_out.pop();
                x_out.pop();
                weights.pop();
                *y_out.last_mut().unwrap() = merged;
                *x_out.last_mut().unwrap() = merged_x;
                *weights.last_mut().unwrap() = w1 + w2;
            } else {
                break;
            }
        }
    }

    (x_out, y_out)
}

/// Predict using isotonic regression (linear interpolation).
fn isotonic_predict(score: f64, x: &[f64], y: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.5;
    }
    if score <= x[0] {
        return y[0];
    }
    if score >= x[x.len() - 1] {
        return y[y.len() - 1];
    }

    // Binary search for the interval
    let pos = x.partition_point(|&v| v < score);
    if pos == 0 {
        return y[0];
    }
    if pos >= x.len() {
        return y[y.len() - 1];
    }

    // Linear interpolation
    let x0 = x[pos - 1];
    let x1 = x[pos];
    let y0 = y[pos - 1];
    let y1 = y[pos];

    if (x1 - x0).abs() < 1e-15 {
        return (y0 + y1) / 2.0;
    }

    y0 + (y1 - y0) * (score - x0) / (x1 - x0)
}

/// Stratified K-fold: distributes each class's samples across the folds in
/// round-robin fashion, preserving the class proportions in every fold.
/// Deterministic (classes are visited in sorted order).
fn stratified_k_fold<F: Float>(
    y: ArrayView1<F>,
    classes: &[F],
    k: usize,
) -> Vec<(Vec<usize>, Vec<usize>)> {
    let n = y.len();
    let mut fold_of = vec![0usize; n];
    for &c in classes {
        for (j, idx) in (0..n).filter(|&i| y[i] == c).enumerate() {
            fold_of[idx] = j % k;
        }
    }

    (0..k)
        .map(|f| {
            let (test, train): (Vec<usize>, Vec<usize>) = (0..n).partition(|&i| fold_of[i] == f);
            (train, test)
        })
        .collect()
}

fn select_elements<F: Float>(y: ArrayView1<F>, indices: &[usize]) -> Array1<F> {
    Array1::from_vec(indices.iter().map(|&i| y[i]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RandomForestClassifier;
    use anofox_ml_trees::DecisionTreeClassifier;
    use ndarray::array;

    /// Deterministic pseudo-random uniform in [0, 1).
    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Overlapping 2-class data with labels {3, 7}.
    fn noisy_binary(n: usize) -> (Array2<f64>, Array1<f64>) {
        let mut s = 12345u64;
        let mut xs = Vec::with_capacity(n * 2);
        let mut ys = Vec::with_capacity(n);
        for i in 0..n {
            let cls = i % 2;
            let center = if cls == 1 { 1.0 } else { -1.0 };
            xs.push(center + 2.0 * (lcg(&mut s) - 0.5) * 2.0);
            xs.push(lcg(&mut s));
            ys.push(if cls == 1 { 7.0 } else { 3.0 });
        }
        (
            Array2::from_shape_vec((n, 2), xs).unwrap(),
            Array1::from_vec(ys),
        )
    }

    #[test]
    fn test_calibrated_classifier_sigmoid() {
        let x = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [10.0, 1.0],
            [11.0, 1.0],
            [12.0, 1.0],
            [13.0, 1.0]
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let cal = CalibratedClassifierCV::new(DecisionTreeClassifier {
            max_depth: Some(3),
            ..Default::default()
        })
        .with_method(CalibrationMethod::Sigmoid)
        .with_cv_folds(2);

        let fitted: FittedCalibratedClassifier<f64> = cal.fit(&x, &y).unwrap();

        let proba = fitted.predict_proba(&x).unwrap();
        assert_eq!(proba.dim(), (8, 2));
        for row in proba.rows() {
            assert!(row.iter().all(|&p| (0.0..=1.0).contains(&p)));
            assert!((row.sum() - 1.0).abs() < 1e-12);
        }

        let preds = fitted.predict(&x).unwrap();
        for (&p, &t) in preds.iter().zip(y.iter()) {
            assert_eq!(p, t);
        }
    }

    #[test]
    fn test_calibrated_classifier_isotonic() {
        let x = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [10.0, 1.0],
            [11.0, 1.0],
            [12.0, 1.0],
            [13.0, 1.0]
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let cal = CalibratedClassifierCV::new(DecisionTreeClassifier::default())
            .with_method(CalibrationMethod::Isotonic)
            .with_cv_folds(2);

        let fitted: FittedCalibratedClassifier<f64> = cal.fit(&x, &y).unwrap();
        let proba = fitted.predict_proba(&x).unwrap();
        for &p in proba.iter() {
            assert!((0.0..=1.0).contains(&p));
        }
    }

    #[test]
    fn test_calibrated_classifier_predict_classes() {
        let x = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [10.0, 1.0],
            [11.0, 1.0],
            [12.0, 1.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let cal = CalibratedClassifierCV::new(DecisionTreeClassifier::default()).with_cv_folds(2);

        let fitted: FittedCalibratedClassifier<f64> = cal.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_calibrated_classifier_shape_mismatch() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let cal = CalibratedClassifierCV::new(DecisionTreeClassifier::default()).with_cv_folds(2);
        let fitted: FittedCalibratedClassifier<f64> = cal.fit(&x, &y).unwrap();

        let x_bad = array![[1.0]];
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_calibrated_classifier_empty_error() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<f64>::zeros(0);

        let cal = CalibratedClassifierCV::new(DecisionTreeClassifier::default());
        assert!(Fit::<f64>::fit(&cal, &x, &y).is_err());
    }

    /// Regression test for #43: calibration used hard 0/1 labels, so the
    /// "probabilities" could take only two values.
    #[test]
    fn test_calibrated_proba_is_continuous_and_maps_labels() {
        let (x, y) = noisy_binary(200);
        let cal = CalibratedClassifierCV::new(
            RandomForestClassifier::new(25)
                .with_max_depth(Some(4))
                .with_seed(1),
        )
        .with_cv_folds(3);
        let fitted: FittedCalibratedClassifier<f64> = cal.fit(&x, &y).unwrap();

        let p1 = fitted.predict_proba_positive(&x).unwrap();
        let mut distinct: Vec<u64> = p1.iter().map(|v| v.to_bits()).collect();
        distinct.sort_unstable();
        distinct.dedup();
        assert!(
            distinct.len() > 2,
            "calibrated probabilities take only {} distinct values",
            distinct.len()
        );
        assert!(p1.iter().all(|&p| p > 0.0 && p < 1.0));

        // Predictions are the original labels {3, 7}, not {0, 1}.
        assert_eq!(fitted.classes(), &[3.0, 7.0]);
        let preds = fitted.predict(&x).unwrap();
        assert!(preds.iter().all(|&p| p == 3.0 || p == 7.0));
        let acc =
            preds.iter().zip(y.iter()).filter(|(p, t)| p == t).count() as f64 / y.len() as f64;
        assert!(acc > 0.7, "accuracy {acc}");
    }

    #[test]
    fn test_multiclass_is_rejected() {
        let x = array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
        let cal = CalibratedClassifierCV::new(DecisionTreeClassifier::default()).with_cv_folds(2);
        match Fit::<f64>::fit(&cal, &x, &y) {
            Err(RustMlError::InvalidParameter(msg)) => assert!(msg.contains("binary")),
            Err(e) => panic!("unexpected error {e:?}"),
            Ok(_) => panic!("multiclass should be rejected"),
        }
    }

    /// Minimal margin classifier: decision = x0 - midpoint of class means.
    #[derive(Clone)]
    struct Stump;
    struct FittedStump(f64);
    impl Fit<f64> for Stump {
        type Fitted = FittedStump;
        fn fit_view(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<FittedStump> {
            let (mut s0, mut n0, mut s1, mut n1) = (0.0, 0.0, 0.0, 0.0);
            for (row, &t) in x.rows().into_iter().zip(y.iter()) {
                if t > 0.5 {
                    s1 += row[0];
                    n1 += 1.0;
                } else {
                    s0 += row[0];
                    n0 += 1.0;
                }
            }
            Ok(FittedStump(0.5 * (s0 / n0 + s1 / n1)))
        }
    }
    impl DecisionFunction<f64> for FittedStump {
        fn decision_function_view(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
            Ok(x.column(0)
                .mapv(|v| v - self.0)
                .insert_axis(ndarray::Axis(1)))
        }
    }

    /// Platt on a decision function recovers the generating sigmoid
    /// `P(y=1|f) = 1 / (1 + exp(-2 f))`, i.e. A ≈ -2, B ≈ 0.
    #[test]
    fn test_platt_recovers_sigmoid_from_decision_function() {
        let n = 4000;
        let mut s = 7u64;
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        for _ in 0..n {
            let f = 6.0 * (lcg(&mut s) - 0.5);
            let p = 1.0 / (1.0 + (-2.0 * f).exp());
            xs.push(f);
            ys.push(if lcg(&mut s) < p { 1.0 } else { 0.0 });
        }
        let x = Array2::from_shape_vec((n, 1), xs).unwrap();
        let y = Array1::from_vec(ys);

        let cal = CalibratedClassifierCV::with_decision_function(Stump).with_cv_folds(3);
        let fitted = Fit::fit(&cal, &x, &y).unwrap();
        let (a, b) = fitted.platt_params();
        assert!((a + 2.0).abs() < 0.3, "A = {a}");
        assert!(b.abs() < 0.3, "B = {b}");
    }

    #[test]
    fn test_fit_platt_converges_on_separable_scores() {
        // Perfectly separable: Newton must still terminate with finite
        // parameters thanks to the prior-corrected targets.
        let scores = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0];
        let labels = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let (a, b) = fit_platt_sigmoid(&scores, &labels);
        assert!(a.is_finite() && b.is_finite());
        assert!(a < 0.0);
        assert!(platt_predict(3.0, a, b) > 0.75);
        assert!(platt_predict(-3.0, a, b) < 0.25);
    }
}
