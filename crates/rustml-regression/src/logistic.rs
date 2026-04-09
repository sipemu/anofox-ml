//! Logistic regression classifier wrapper.
//!
//! Wraps `anofox_regression::LogisticRegression` to provide an sklearn-like
//! binary classifier with the rustml [`Fit`] / [`Predict`] type-state pattern.

use crate::convert::{col_to_ndarray, ndarray_to_col, ndarray_to_mat};
use anofox_regression::solvers::{
    FittedLogistic, LogisticRegression as InnerLogistic, Penalty,
};
use ndarray::{Array1, Array2};
use rustml_core::{Fit, Predict, Result, RustMlError};

/// Logistic regression binary classifier.
///
/// Wraps a binomial GLM (logit link) with a classifier-oriented API:
/// `predict` returns class labels (0/1), `predict_proba` returns probabilities.
///
/// Supports optional L2 regularization via `with_c` (sklearn convention) or
/// `with_lambda` (direct regularization strength).
#[derive(Debug, Clone)]
pub struct LogisticRegressor {
    c: Option<f64>,
    lambda: Option<f64>,
    with_intercept: bool,
    threshold: f64,
    max_iter: usize,
    tol: f64,
}

impl LogisticRegressor {
    pub fn new() -> Self {
        Self {
            c: None,
            lambda: None,
            with_intercept: true,
            threshold: 0.5,
            max_iter: 100,
            tol: 1e-8,
        }
    }

    /// Set inverse regularization strength (sklearn convention). Larger C = less regularization.
    pub fn with_c(mut self, c: f64) -> Self {
        self.c = Some(c);
        self.lambda = None;
        self
    }

    /// Set L2 regularization strength directly. Larger lambda = more regularization.
    pub fn with_lambda(mut self, lambda: f64) -> Self {
        self.lambda = Some(lambda);
        self.c = None;
        self
    }

    /// Set whether to include an intercept (bias) term. Default: true.
    pub fn with_intercept(mut self, include: bool) -> Self {
        self.with_intercept = include;
        self
    }

    /// Set the classification threshold. Default: 0.5.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set the maximum number of IRLS iterations. Default: 100.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance. Default: 1e-8.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }
}

impl Default for LogisticRegressor {
    fn default() -> Self {
        Self::new()
    }
}

/// A fitted logistic regression classifier.
#[derive(Debug)]
pub struct FittedLogisticRegressor {
    inner: FittedLogistic,
    n_features: usize,
}

impl FittedLogisticRegressor {
    /// Return the regression coefficients (excluding intercept).
    pub fn coefficients(&self) -> Array1<f64> {
        col_to_ndarray(self.inner.coefficients())
    }

    /// Return the intercept term, if the model was fit with one.
    pub fn intercept(&self) -> Option<f64> {
        self.inner.intercept()
    }

    /// Return predicted probabilities P(Y=1|X) for the given data.
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        let x_mat = ndarray_to_mat(x);
        let probs = self.inner.predict_proba(&x_mat);
        Ok(col_to_ndarray(&probs))
    }

    /// Return the decision function (log-odds) for the given data.
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        let x_mat = ndarray_to_mat(x);
        let decision = self.inner.decision_function(&x_mat);
        Ok(col_to_ndarray(&decision))
    }

    /// Compute classification accuracy on the given data.
    pub fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<f64> {
        let preds = self.predict(x)?;
        let n = y.len();
        let correct = preds
            .iter()
            .zip(y.iter())
            .filter(|(&p, &a)| p == a)
            .count();
        Ok(correct as f64 / n as f64)
    }

    /// Return the number of IRLS iterations used during fitting.
    pub fn n_iter(&self) -> usize {
        self.inner.n_iter()
    }
}

impl Fit<f64> for LogisticRegressor {
    type Fitted = FittedLogisticRegressor;

    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
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

        let x_mat = ndarray_to_mat(x);
        let y_col = ndarray_to_col(y);

        let mut builder = InnerLogistic::builder()
            .with_intercept(self.with_intercept)
            .threshold(self.threshold)
            .max_iterations(self.max_iter)
            .tolerance(self.tol)
            .compute_inference(false);

        if let Some(c) = self.c {
            builder = builder.c(c);
        } else if let Some(lambda) = self.lambda {
            builder = builder.penalty(Penalty::L2(lambda));
        }

        let inner_model = builder.build();
        let fitted = inner_model
            .fit(&x_mat, &y_col)
            .map_err(|e| RustMlError::InvalidParameter(e.to_string()))?;

        Ok(FittedLogisticRegressor {
            inner: fitted,
            n_features: x.ncols(),
        })
    }
}

impl Predict<f64> for FittedLogisticRegressor {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        let x_mat = ndarray_to_mat(x);
        let labels = self.inner.predict(&x_mat);
        Ok(col_to_ndarray(&labels))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_logistic_basic() {
        let x = Array2::from_shape_vec(
            (8, 1),
            vec![-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0],
        )
        .unwrap();
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let fitted = LogisticRegressor::new().fit(&x, &y).unwrap();

        // Coefficient should be positive (higher x -> more likely class 1)
        assert!(fitted.coefficients()[0] > 0.0);
        assert!(fitted.intercept().is_some());
    }

    #[test]
    fn test_predict_labels() {
        let x = Array2::from_shape_vec(
            (8, 1),
            vec![-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0],
        )
        .unwrap();
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let fitted = LogisticRegressor::new().fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        for &p in preds.iter() {
            assert!(p == 0.0 || p == 1.0, "labels must be 0 or 1, got {}", p);
        }
    }

    #[test]
    fn test_predict_proba_range() {
        let x = Array2::from_shape_vec(
            (8, 1),
            vec![-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0],
        )
        .unwrap();
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let fitted = LogisticRegressor::new().fit(&x, &y).unwrap();
        let probs = fitted.predict_proba(&x).unwrap();

        for &p in probs.iter() {
            assert!(
                (0.0..=1.0).contains(&p),
                "probability must be in [0,1], got {}",
                p
            );
        }
    }

    #[test]
    fn test_score() {
        let x = Array2::from_shape_vec(
            (8, 1),
            vec![-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0],
        )
        .unwrap();
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let fitted = LogisticRegressor::new().fit(&x, &y).unwrap();
        let acc = fitted.score(&x, &y).unwrap();

        assert!(acc > 0.7, "accuracy should be > 0.7 on separable data, got {}", acc);
    }

    #[test]
    fn test_l2_regularization() {
        let x = Array2::from_shape_vec(
            (8, 1),
            vec![-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0],
        )
        .unwrap();
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let no_reg = LogisticRegressor::new().fit(&x, &y).unwrap();
        let l2 = LogisticRegressor::new().with_lambda(10.0).fit(&x, &y).unwrap();

        assert!(
            l2.coefficients()[0].abs() < no_reg.coefficients()[0].abs(),
            "L2 should shrink coefficients"
        );
    }

    #[test]
    fn test_c_convention() {
        let x = Array2::from_shape_vec(
            (8, 1),
            vec![-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0],
        )
        .unwrap();
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        // Small C = strong regularization
        let fitted = LogisticRegressor::new().with_c(0.01).fit(&x, &y).unwrap();
        assert!(fitted.coefficients()[0].abs() < 5.0);
    }

    #[test]
    fn test_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![0.0, 1.0];

        assert!(LogisticRegressor::new().fit(&x, &y).is_err());
    }

    #[test]
    fn test_empty_input() {
        let x = Array2::<f64>::zeros((0, 1));
        let y = Array1::<f64>::zeros(0);

        assert!(LogisticRegressor::new().fit(&x, &y).is_err());
    }
}
