//! Ridge regression as a classifier (sklearn's `RidgeClassifier`).
//!
//! Labels are encoded as ±1 (one column for two classes, one per class
//! otherwise) and each column is fitted by ridge regression with the same
//! `alpha`; the prediction is the class with the largest decision value
//! (two classes: the positive class when it is > 0).

use anofox_ml_core::{
    compute_sample_weight, ClassWeight, DecisionFunction, Fit, FitWeighted, Predict, Result,
    RustMlError,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

use crate::RidgeRegressor;

/// Ridge classifier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RidgeClassifier {
    /// L2 penalty (sklearn's `alpha`).
    pub alpha: f64,
    pub fit_intercept: bool,
    /// Per-class weights, applied as sample weights.
    pub class_weight: Option<ClassWeight>,
}

impl Default for RidgeClassifier {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            fit_intercept: true,
            class_weight: None,
        }
    }
}

impl RidgeClassifier {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    pub fn with_class_weight(mut self, class_weight: ClassWeight) -> Self {
        self.class_weight = Some(class_weight);
        self
    }
}

/// Fitted [`RidgeClassifier`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedRidgeClassifier {
    classes: Vec<f64>,
    /// One row per decision column (1 for two classes, else one per class).
    coef: Array2<f64>,
    intercept: Array1<f64>,
}

impl FittedRidgeClassifier {
    pub fn classes(&self) -> &[f64] {
        &self.classes
    }

    /// `n_columns × n_features` (sklearn's `coef_`).
    pub fn coefficients(&self) -> &Array2<f64> {
        &self.coef
    }

    pub fn intercept(&self) -> &Array1<f64> {
        &self.intercept
    }

    fn scores(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.coef.ncols() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.coef.ncols(),
                x.ncols()
            )));
        }
        Ok(x.dot(&self.coef.t()) + &self.intercept)
    }
}

impl FitWeighted<f64> for RidgeClassifier {
    type Fitted = FittedRidgeClassifier;

    fn fit_weighted_view(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        sample_weight: Option<ArrayView1<f64>>,
    ) -> Result<FittedRidgeClassifier> {
        if y.len() != x.nrows() {
            return Err(RustMlError::ShapeMismatch(format!(
                "x has {} rows but y has {}",
                x.nrows(),
                y.len()
            )));
        }
        let mut classes = y.to_vec();
        classes.sort_by(f64::total_cmp);
        classes.dedup();
        if classes.len() < 2 {
            return Err(RustMlError::InvalidParameter(format!(
                "RidgeClassifier needs at least 2 classes, got {}",
                classes.len()
            )));
        }
        let mut weights = sample_weight.map(|w| w.to_owned());
        if let Some(cw) = &self.class_weight {
            let cw = compute_sample_weight(cw, y)?;
            weights = Some(match weights {
                Some(w) => w * cw,
                None => cw,
            });
        }
        let ridge = RidgeRegressor::new()
            .with_lambda(self.alpha)
            .with_intercept(self.fit_intercept);
        // ±1 targets; the binary case fits only the positive class.
        let columns: Vec<usize> = if classes.len() == 2 {
            vec![1]
        } else {
            (0..classes.len()).collect()
        };
        let mut coef = Array2::zeros((columns.len(), x.ncols()));
        let mut intercept = Array1::zeros(columns.len());
        for (row, &c) in columns.iter().enumerate() {
            let target: Array1<f64> = y.mapv(|v| if v == classes[c] { 1.0 } else { -1.0 });
            let fit =
                ridge.fit_weighted_view(x, target.view(), weights.as_ref().map(|w| w.view()))?;
            coef.row_mut(row).assign(&fit.coefficients());
            intercept[row] = fit.intercept().unwrap_or(0.0);
        }
        Ok(FittedRidgeClassifier {
            classes,
            coef,
            intercept,
        })
    }
}

impl Fit<f64> for RidgeClassifier {
    type Fitted = FittedRidgeClassifier;

    fn fit_view(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<FittedRidgeClassifier> {
        self.fit_weighted_view(x, y, None)
    }
}

impl DecisionFunction<f64> for FittedRidgeClassifier {
    /// `n × 1` for two classes (positive class), else `n × classes`.
    fn decision_function_view(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        self.scores(x)
    }
}

impl Predict<f64> for FittedRidgeClassifier {
    fn predict_view(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        let s = self.scores(x)?;
        if self.classes.len() == 2 {
            return Ok(s.column(0).mapv(|v| {
                if v > 0.0 {
                    self.classes[1]
                } else {
                    self.classes[0]
                }
            }));
        }
        Ok(s.axis_iter(Axis(0))
            .map(|r| {
                let best = r
                    .iter()
                    .enumerate()
                    .fold(0, |b, (j, &v)| if v > r[b] { j } else { b });
                self.classes[best]
            })
            .collect())
    }
}

impl anofox_ml_core::ClassifierScore<f64> for FittedRidgeClassifier {}

/// Ridge classifier with `alpha` chosen by exact leave-one-out over
/// `alphas` (sklearn's `RidgeClassifierCV` with `cv=None`): the mean
/// squared LOO residual of the ±1 targets (all columns), with the sample
/// and class weights as in sklearn's `_RidgeGCV`; the first alpha with the
/// lowest error is refitted as a [`RidgeClassifier`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RidgeClassifierCV {
    pub alphas: Vec<f64>,
    pub fit_intercept: bool,
    pub class_weight: Option<ClassWeight>,
}

impl Default for RidgeClassifierCV {
    fn default() -> Self {
        Self {
            alphas: vec![0.1, 1.0, 10.0],
            fit_intercept: true,
            class_weight: None,
        }
    }
}

/// Fitted [`RidgeClassifierCV`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedRidgeClassifierCV {
    /// The selected alpha (sklearn's `alpha_`).
    pub alpha: f64,
    /// Mean squared LOO residual per alpha (sklearn's `-best_score_` for
    /// the selected one).
    pub loo_mse: Array1<f64>,
    pub fitted: FittedRidgeClassifier,
}

impl RidgeClassifierCV {
    /// Mean squared LOO residual per alpha, for the ±1 targets
    /// `t` (`n × k`) with weights `sw`.
    fn loo_mse(
        &self,
        x: ArrayView2<f64>,
        t: &Array2<f64>,
        sw: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        let (n, p) = x.dim();
        let total = sw.sum();
        let (x_mean, t_mean) = if self.fit_intercept {
            (x.t().dot(sw) / total, t.t().dot(sw) / total)
        } else {
            (Array1::zeros(p), Array1::zeros(t.ncols()))
        };
        let sq = sw.mapv(f64::sqrt);
        let sq_col = sq.view().insert_axis(Axis(1));
        let xs = (&x - &x_mean) * sq_col;
        let ts = (t - &t_mean) * sq_col;
        let m = faer::Mat::from_fn(n, p, |i, j| xs[[i, j]]);
        let svd = m
            .thin_svd()
            .map_err(|e| RustMlError::InvalidParameter(format!("SVD failed: {e:?}")))?;
        let (u, s) = (svd.U(), svd.S().column_vector());
        let r = s.nrows();
        let u = Array2::from_shape_fn((n, r), |(i, k)| u[(i, k)]);
        let s: Vec<f64> = (0..r).map(|k| s[k]).collect();
        let cutoff = s.iter().fold(0.0_f64, |a, &b| a.max(b)) * 1e-12 * n.max(p) as f64;
        let ut = u.t().dot(&ts);
        let u2 = u.mapv(|v| v * v);
        let base = if self.fit_intercept {
            sw / total
        } else {
            Array1::zeros(n)
        };
        let mut out = Array1::zeros(self.alphas.len());
        for (l, &alpha) in self.alphas.iter().enumerate() {
            let h = Array1::from_iter(s.iter().map(|&v| {
                if v <= cutoff {
                    0.0
                } else {
                    v * v / (v * v + alpha)
                }
            }));
            let fitted = u.dot(&(&ut * &h.view().insert_axis(Axis(1))));
            let hii = u2.dot(&h) + &base;
            let e = (&ts - &fitted) / &(1.0 - &hii).insert_axis(Axis(1));
            out[l] = e.mapv(|v| v * v).mean().unwrap_or(f64::NAN);
        }
        Ok(out)
    }
}

impl FitWeighted<f64> for RidgeClassifierCV {
    type Fitted = FittedRidgeClassifierCV;

    fn fit_weighted_view(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        sample_weight: Option<ArrayView1<f64>>,
    ) -> Result<FittedRidgeClassifierCV> {
        if self.alphas.is_empty() || self.alphas.iter().any(|a| !(a.is_finite() && *a > 0.0)) {
            return Err(RustMlError::InvalidParameter(
                "alphas must be non-empty and positive".into(),
            ));
        }
        if y.len() != x.nrows() {
            return Err(RustMlError::ShapeMismatch(format!(
                "x has {} rows but y has {}",
                x.nrows(),
                y.len()
            )));
        }
        let mut classes = y.to_vec();
        classes.sort_by(f64::total_cmp);
        classes.dedup();
        if classes.len() < 2 {
            return Err(RustMlError::InvalidParameter(format!(
                "RidgeClassifierCV needs at least 2 classes, got {}",
                classes.len()
            )));
        }
        let mut sw = sample_weight.map_or_else(|| Array1::ones(y.len()), |w| w.to_owned());
        if let Some(cw) = &self.class_weight {
            sw = sw * compute_sample_weight(cw, y)?;
        }
        let columns: Vec<usize> = if classes.len() == 2 {
            vec![1]
        } else {
            (0..classes.len()).collect()
        };
        let t = Array2::from_shape_fn((y.len(), columns.len()), |(i, c)| {
            if y[i] == classes[columns[c]] {
                1.0
            } else {
                -1.0
            }
        });
        let loo_mse = self.loo_mse(x, &t, &sw)?;
        let best = (0..loo_mse.len()).fold(0, |b, l| if loo_mse[l] < loo_mse[b] { l } else { b });
        let alpha = self.alphas[best];
        let classifier = RidgeClassifier {
            alpha,
            fit_intercept: self.fit_intercept,
            class_weight: self.class_weight.clone(),
        };
        Ok(FittedRidgeClassifierCV {
            alpha,
            loo_mse,
            fitted: classifier.fit_weighted_view(x, y, sample_weight)?,
        })
    }
}

impl Fit<f64> for RidgeClassifierCV {
    type Fitted = FittedRidgeClassifierCV;

    fn fit_view(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<FittedRidgeClassifierCV> {
        self.fit_weighted_view(x, y, None)
    }
}

impl DecisionFunction<f64> for FittedRidgeClassifierCV {
    fn decision_function_view(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        self.fitted.decision_function_view(x)
    }
}

impl Predict<f64> for FittedRidgeClassifierCV {
    fn predict_view(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        self.fitted.predict_view(x)
    }
}

impl anofox_ml_core::ClassifierScore<f64> for FittedRidgeClassifierCV {}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn separates_classes() {
        let x = array![
            [0.0, 0.0],
            [0.0, 1.0],
            [5.0, 5.0],
            [5.0, 6.0],
            [10.0, 0.0],
            [10.0, 1.0]
        ];
        let y = array![1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
        let m = RidgeClassifier::new().with_alpha(0.1).fit(&x, &y).unwrap();
        assert_eq!(m.coefficients().dim(), (3, 2));
        let yb = array![0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let b = RidgeClassifier::new().fit(&x, &yb).unwrap();
        assert_eq!(b.coefficients().dim(), (1, 2));
        assert_eq!(b.predict(&x).unwrap(), yb);
        assert!(RidgeClassifier::new().fit(&x, &Array1::ones(6)).is_err());
    }

    #[test]
    fn ridge_classifier_cv_loo_equals_refitting_without_each_row() {
        let x = Array2::from_shape_fn((24, 3), |(i, j)| (i as f64 * (j as f64 + 0.9)).sin());
        let y = Array1::from_shape_fn(24, |i| {
            ((x[[i, 0]] + x[[i, 1]] > 0.0) as u8 + (i % 3 == 0) as u8) as f64
        });
        let w = Array1::from_shape_fn(24, |i| 0.5 + (i % 4) as f64);
        let cv = RidgeClassifierCV {
            alphas: vec![0.05, 1.0, 20.0],
            ..Default::default()
        };
        let fit = cv
            .fit_weighted_view(x.view(), y.view(), Some(w.view()))
            .unwrap();
        // Brute force: refit the weighted ridge without row i, score the
        // sqrt(w)-scaled residual on the ±1 targets (sklearn's GCV error).
        for (l, &alpha) in cv.alphas.iter().enumerate() {
            let mut sse = 0.0;
            for i in 0..24 {
                let keep: Vec<usize> = (0..24).filter(|&r| r != i).collect();
                let xt = x.select(Axis(0), &keep);
                let wt = w.select(Axis(0), &keep);
                for c in 0..3 {
                    let t = y.mapv(|v| if v == c as f64 { 1.0 } else { -1.0 });
                    let tt = t.select(Axis(0), &keep);
                    let r = RidgeRegressor::new()
                        .with_lambda(alpha)
                        .fit_weighted_view(xt.view(), tt.view(), Some(wt.view()))
                        .unwrap();
                    let pred = x.row(i).dot(&r.coefficients()) + r.intercept().unwrap();
                    sse += w[i] * (t[i] - pred).powi(2);
                }
            }
            let brute = sse / (24.0 * 3.0);
            assert!(
                (fit.loo_mse[l] - brute).abs() < 1e-9,
                "{l}: {} vs {brute}",
                fit.loo_mse[l]
            );
        }
        let best = (0..3).fold(0, |b, l| {
            if fit.loo_mse[l] < fit.loo_mse[b] {
                l
            } else {
                b
            }
        });
        assert_eq!(fit.alpha, cv.alphas[best]);
    }
}
