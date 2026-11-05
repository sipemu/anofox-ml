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
}
