//! CalibratedClassifierCV — probability calibration for classifiers.
//!
//! Wraps any classifier to produce well-calibrated probabilities using either
//! Platt scaling (sigmoid) or isotonic regression, fitted via cross-validation.

use anofox_ml_core::{Fit, Float, Predict, Result, RustMlError};
use ndarray::{Array1, Array2};

/// Calibration method.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum CalibrationMethod {
    /// Platt scaling: fits a sigmoid A*f(x)+B to map scores to probabilities.
    Sigmoid,
    /// Isotonic regression: non-parametric monotonic mapping.
    Isotonic,
}

impl Default for CalibrationMethod {
    fn default() -> Self {
        CalibrationMethod::Sigmoid
    }
}

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

/// Calibrated classifier with cross-validation.
///
/// Wraps a base classifier and calibrates its predictions to produce
/// well-calibrated probabilities. Uses cross-validation to generate
/// out-of-fold predictions for calibration fitting.
pub struct CalibratedClassifierCV<F: Float> {
    base_estimator: Box<dyn FitPredBox<F>>,
    method: CalibrationMethod,
    cv_folds: usize,
}

impl<F: Float> CalibratedClassifierCV<F> {
    /// Create a new CalibratedClassifierCV wrapping the given base estimator.
    pub fn new<T>(base_estimator: T) -> Self
    where
        T: Fit<F> + Send + Sync + 'static,
        T::Fitted: Predict<F> + Send + Sync + 'static,
    {
        Self {
            base_estimator: Box::new(base_estimator),
            method: CalibrationMethod::Sigmoid,
            cv_folds: 5,
        }
    }

    pub fn with_method(mut self, method: CalibrationMethod) -> Self {
        self.method = method;
        self
    }

    pub fn with_cv_folds(mut self, cv_folds: usize) -> Self {
        self.cv_folds = cv_folds;
        self
    }
}

/// Fitted calibrated classifier.
pub struct FittedCalibratedClassifier<F: Float> {
    /// Base model fitted on full data.
    base_model: Box<dyn PredBox<F>>,
    /// Calibration parameters (Platt sigmoid: a, b).
    cal_a: f64,
    cal_b: f64,
    /// For isotonic: sorted (score, prob) pairs.
    isotonic_x: Vec<f64>,
    isotonic_y: Vec<f64>,
    method: CalibrationMethod,
    n_features: usize,
}

impl<F: Float> FittedCalibratedClassifier<F> {
    /// Predict calibrated probabilities for class 1.
    pub fn predict_proba(&self, x: &Array2<F>) -> Result<Array1<F>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }

        let raw_preds = self.base_model.predict_box(x)?;
        let n = raw_preds.len();
        let mut proba = Array1::zeros(n);

        for i in 0..n {
            let score = raw_preds[i].to_f64().unwrap();
            let p = match self.method {
                CalibrationMethod::Sigmoid => {
                    1.0 / (1.0 + (-(self.cal_a * score + self.cal_b)).exp())
                }
                CalibrationMethod::Isotonic => {
                    isotonic_predict(score, &self.isotonic_x, &self.isotonic_y)
                }
            };
            proba[i] = F::from_f64(p.clamp(0.0, 1.0)).unwrap();
        }

        Ok(proba)
    }
}

impl<F: Float + 'static> Fit<F> for CalibratedClassifierCV<F> {
    type Fitted = FittedCalibratedClassifier<F>;

    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Self::Fitted> {
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

        // Generate out-of-fold predictions for calibration using stratified
        // splits so each fold keeps the class distribution — otherwise on
        // class-sorted data we end up training each fold on a single class.
        let folds = stratified_k_fold(y, k);
        let mut oof_scores = vec![0.0f64; n];
        let mut oof_labels = vec![0.0f64; n];

        for (train_idx, test_idx) in &folds {
            let x_train = select_rows(x, train_idx);
            let y_train = select_elements(y, train_idx);
            let x_test = select_rows(x, test_idx);

            let fitted = self.base_estimator.fit_box(&x_train, &y_train)?;
            let preds = fitted.predict_box(&x_test)?;

            for (li, &gi) in test_idx.iter().enumerate() {
                oof_scores[gi] = preds[li].to_f64().unwrap();
                oof_labels[gi] = y[gi].to_f64().unwrap();
            }
        }

        // Fit calibration mapping on OOF predictions
        let (cal_a, cal_b, isotonic_x, isotonic_y) = match self.method {
            CalibrationMethod::Sigmoid => {
                let (a, b) = fit_platt_sigmoid(&oof_scores, &oof_labels);
                (a, b, Vec::new(), Vec::new())
            }
            CalibrationMethod::Isotonic => {
                let (ix, iy) = fit_isotonic(&oof_scores, &oof_labels);
                (0.0, 0.0, ix, iy)
            }
        };

        // Refit base model on full data
        let base_model = self.base_estimator.fit_box(x, y)?;

        Ok(FittedCalibratedClassifier {
            base_model,
            cal_a,
            cal_b,
            isotonic_x,
            isotonic_y,
            method: self.method,
            n_features: x.ncols(),
        })
    }
}

impl<F: Float> Predict<F> for FittedCalibratedClassifier<F> {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        let proba = self.predict_proba(x)?;
        let threshold = F::from_f64(0.5).unwrap();
        Ok(proba.mapv(|p| if p >= threshold { F::one() } else { F::zero() }))
    }
}

/// Fit Platt scaling: find A, B such that P(y=1|f) = 1/(1+exp(A*f+B)).
/// Uses gradient descent on cross-entropy loss.
fn fit_platt_sigmoid(scores: &[f64], labels: &[f64]) -> (f64, f64) {
    let n = scores.len();
    if n == 0 {
        return (1.0, 0.0);
    }

    let mut a = 0.0f64;
    let mut b = 0.0f64;
    let lr = 0.01;

    for _ in 0..1000 {
        let mut grad_a = 0.0;
        let mut grad_b = 0.0;

        for i in 0..n {
            let p = 1.0 / (1.0 + (-(a * scores[i] + b)).exp());
            let err = p - labels[i];
            grad_a += err * scores[i];
            grad_b += err;
        }

        grad_a /= n as f64;
        grad_b /= n as f64;

        a -= lr * grad_a;
        b -= lr * grad_b;
    }

    (a, b)
}

/// Fit isotonic regression (pool adjacent violators algorithm).
fn fit_isotonic(scores: &[f64], labels: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = scores.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    // Sort by score
    let mut pairs: Vec<(f64, f64)> = scores
        .iter()
        .zip(labels.iter())
        .map(|(&s, &l)| (s, l))
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Pool adjacent violators
    let mut x_out: Vec<f64> = Vec::with_capacity(n);
    let mut y_out: Vec<f64> = Vec::with_capacity(n);
    let mut weights: Vec<f64> = Vec::with_capacity(n);

    for &(xi, yi) in &pairs {
        x_out.push(xi);
        y_out.push(yi);
        weights.push(1.0);

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

/// Stratified K-fold for classification calibration: groups samples by class
/// label and distributes each class's samples across the folds in round-robin
/// fashion, preserving the class proportions in every fold.
fn stratified_k_fold<F: Float>(y: &Array1<F>, k: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
    use std::collections::HashMap;
    let n = y.len();

    // Group sample indices by class (keyed by f64::to_bits to support arbitrary labels).
    let mut by_class: HashMap<u64, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let key = y[i].to_f64().unwrap().to_bits();
        by_class.entry(key).or_default().push(i);
    }

    // Assign each sample a fold number, round-robin within each class.
    let mut fold_of = vec![0usize; n];
    for (_, class_indices) in by_class.iter() {
        for (j, &idx) in class_indices.iter().enumerate() {
            fold_of[idx] = j % k;
        }
    }

    // Build (train, test) index pairs per fold.
    let mut folds: Vec<(Vec<usize>, Vec<usize>)> =
        (0..k).map(|_| (Vec::new(), Vec::new())).collect();
    for i in 0..n {
        for (f, (train, test)) in folds.iter_mut().enumerate() {
            if fold_of[i] == f {
                test.push(i);
            } else {
                train.push(i);
            }
        }
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
    use anofox_ml_trees::DecisionTreeClassifier;
    use ndarray::array;

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
        for &p in proba.iter() {
            assert!(
                p >= 0.0 && p <= 1.0,
                "probability must be in [0,1], got {}",
                p
            );
        }

        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(p == 0.0 || p == 1.0);
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
            assert!(p >= 0.0 && p <= 1.0);
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
}
