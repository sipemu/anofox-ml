//! Ridge regression with built-in cross-validation for alpha selection.
//!
//! Automatically selects the best regularization parameter (`alpha`) from a
//! grid of candidates using k-fold cross-validation.

use crate::ridge::{FittedRidgeRegressor, RidgeRegressor};
use ndarray::{Array1, Array2};
use rustml_core::{cross_val_score, Fit, Predict, Result, RustMlError};
use rustml_metrics::r2_score;

/// Ridge regression with cross-validated alpha selection.
///
/// Searches over a grid of `alphas` using k-fold cross-validation and selects
/// the alpha that maximizes the mean R2 score.
#[derive(Debug, Clone)]
pub struct RidgeCrossValidated {
    alphas: Vec<f64>,
    cv_folds: usize,
    with_intercept: bool,
}

impl RidgeCrossValidated {
    pub fn new() -> Self {
        Self {
            alphas: vec![0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            cv_folds: 5,
            with_intercept: true,
        }
    }

    pub fn with_alphas(mut self, alphas: Vec<f64>) -> Self {
        self.alphas = alphas;
        self
    }

    pub fn with_cv_folds(mut self, folds: usize) -> Self {
        self.cv_folds = folds;
        self
    }

    pub fn with_intercept(mut self, include: bool) -> Self {
        self.with_intercept = include;
        self
    }
}

impl Default for RidgeCrossValidated {
    fn default() -> Self {
        Self::new()
    }
}

/// A fitted Ridge regression model selected via cross-validation.
#[derive(Debug, Clone)]
pub struct FittedRidgeCrossValidated {
    inner: FittedRidgeRegressor,
    best_alpha: f64,
}

impl FittedRidgeCrossValidated {
    /// Returns the alpha value selected by cross-validation.
    pub fn best_alpha(&self) -> f64 {
        self.best_alpha
    }

    pub fn coefficients(&self) -> Array1<f64> {
        self.inner.coefficients()
    }

    pub fn intercept(&self) -> Option<f64> {
        self.inner.intercept()
    }

    pub fn r_squared(&self) -> f64 {
        self.inner.r_squared()
    }
}

impl Fit<f64> for RidgeCrossValidated {
    type Fitted = FittedRidgeCrossValidated;

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
        if self.alphas.is_empty() {
            return Err(RustMlError::InvalidParameter(
                "alphas must not be empty".into(),
            ));
        }
        if self.cv_folds < 2 {
            return Err(RustMlError::InvalidParameter(
                "cv_folds must be at least 2".into(),
            ));
        }
        for &alpha in &self.alphas {
            if alpha < 0.0 {
                return Err(RustMlError::InvalidParameter(
                    "all alphas must be non-negative".into(),
                ));
            }
        }

        let with_intercept = self.with_intercept;

        let mut best_alpha = self.alphas[0];
        let mut best_mean_score = f64::NEG_INFINITY;

        for &alpha in &self.alphas {
            let scores = cross_val_score(
                x,
                y,
                self.cv_folds,
                |x_train, y_train, x_test| {
                    let model = RidgeRegressor::new()
                        .with_lambda(alpha)
                        .with_intercept(with_intercept);
                    let fitted = model.fit(x_train, y_train)?;
                    fitted.predict(x_test)
                },
                |y_true, y_pred| r2_score(y_true, y_pred),
            )?;

            let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;
            if mean_score > best_mean_score {
                best_mean_score = mean_score;
                best_alpha = alpha;
            }
        }

        // Fit the final model with the best alpha on all data
        let final_model = RidgeRegressor::new()
            .with_lambda(best_alpha)
            .with_intercept(with_intercept);
        let inner = final_model.fit(x, y)?;

        Ok(FittedRidgeCrossValidated {
            inner,
            best_alpha,
        })
    }
}

impl Predict<f64> for FittedRidgeCrossValidated {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        self.inner.predict(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    fn make_linear_data(n: usize) -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec(
            (n, 1),
            (0..n).map(|i| i as f64).collect(),
        )
        .unwrap();
        let y = Array1::from_vec((0..n).map(|i| 2.0 + 3.0 * i as f64).collect());
        (x, y)
    }

    #[test]
    fn test_ridge_cv_basic() {
        let (x, y) = make_linear_data(50);

        let fitted = RidgeCrossValidated::new()
            .fit(&x, &y)
            .unwrap();

        assert!(fitted.r_squared() > 0.99);
        assert!(fitted.best_alpha() > 0.0);
    }

    #[test]
    fn test_ridge_cv_selects_best_alpha() {
        let (x, y) = make_linear_data(50);

        let fitted = RidgeCrossValidated::new()
            .with_alphas(vec![0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
            .with_cv_folds(5)
            .fit(&x, &y)
            .unwrap();

        // For a clean linear relationship, small alpha should be preferred
        assert!(
            fitted.best_alpha() <= 1.0,
            "expected small alpha for linear data, got {}",
            fitted.best_alpha()
        );
    }

    #[test]
    fn test_ridge_cv_coefficients() {
        let (x, y) = make_linear_data(50);

        let fitted = RidgeCrossValidated::new()
            .with_alphas(vec![0.001, 0.01])
            .fit(&x, &y)
            .unwrap();

        assert_abs_diff_eq!(fitted.coefficients()[0], 3.0, epsilon = 0.1);
    }

    #[test]
    fn test_ridge_cv_predict() {
        let (x, y) = make_linear_data(50);

        let fitted = RidgeCrossValidated::new()
            .with_alphas(vec![0.001, 0.01])
            .fit(&x, &y)
            .unwrap();

        let x_test = Array2::from_shape_vec((3, 1), vec![50.0, 51.0, 52.0]).unwrap();
        let preds = fitted.predict(&x_test).unwrap();
        assert_eq!(preds.len(), 3);
        assert_abs_diff_eq!(preds[0], 2.0 + 3.0 * 50.0, epsilon = 1.0);
    }

    #[test]
    fn test_ridge_cv_shape_mismatch() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];

        let result = RidgeCrossValidated::new().fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_ridge_cv_empty_input() {
        let x = Array2::<f64>::zeros((0, 1));
        let y = Array1::<f64>::zeros(0);

        let result = RidgeCrossValidated::new().fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_ridge_cv_empty_alphas() {
        let (x, y) = make_linear_data(20);

        let result = RidgeCrossValidated::new()
            .with_alphas(vec![])
            .fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_ridge_cv_negative_alpha() {
        let (x, y) = make_linear_data(20);

        let result = RidgeCrossValidated::new()
            .with_alphas(vec![0.1, -1.0])
            .fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_ridge_cv_invalid_folds() {
        let (x, y) = make_linear_data(20);

        let result = RidgeCrossValidated::new()
            .with_cv_folds(1)
            .fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_ridge_cv_no_intercept() {
        let x = Array2::from_shape_vec(
            (50, 1),
            (0..50).map(|i| i as f64).collect(),
        )
        .unwrap();
        let y = Array1::from_vec((0..50).map(|i| 3.0 * i as f64).collect());

        let fitted = RidgeCrossValidated::new()
            .with_alphas(vec![0.001, 0.01])
            .with_intercept(false)
            .fit(&x, &y)
            .unwrap();

        assert!(fitted.intercept().is_none());
    }

    #[test]
    fn test_ridge_cv_default() {
        let rcv = RidgeCrossValidated::default();
        let (x, y) = make_linear_data(50);

        let result = rcv.fit(&x, &y);
        assert!(result.is_ok());
    }
}

impl rustml_core::RegressorScore<f64> for FittedRidgeCrossValidated {}
