//! Elastic Net regression with built-in cross-validation for hyperparameter selection.
//!
//! Performs a grid search over `alphas x l1_ratios` using k-fold cross-validation
//! to find the best combination of regularization parameters.

use crate::elastic_net::{ElasticNetRegressor, FittedElasticNetRegressor};
use anofox_ml_core::{cross_val_score, Fit, Predict, Result, RustMlError};
use anofox_ml_metrics::r2_score;
use ndarray::{Array1, Array2};

/// Elastic Net regression with cross-validated hyperparameter selection.
///
/// Performs a grid search over `alphas` (lambda values) and `l1_ratios` (alpha
/// mixing parameters) using k-fold cross-validation, selecting the combination
/// that maximizes the mean R2 score.
#[derive(Debug, Clone)]
pub struct ElasticNetCrossValidated {
    alphas: Vec<f64>,
    l1_ratios: Vec<f64>,
    cv_folds: usize,
    with_intercept: bool,
}

impl ElasticNetCrossValidated {
    pub fn new() -> Self {
        Self {
            alphas: vec![0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            l1_ratios: vec![0.1, 0.3, 0.5, 0.7, 0.9],
            cv_folds: 5,
            with_intercept: true,
        }
    }

    /// Set the grid of lambda values to search over.
    pub fn with_alphas(mut self, alphas: Vec<f64>) -> Self {
        self.alphas = alphas;
        self
    }

    /// Set the grid of L1 ratio values to search over.
    /// Each value must be in [0.0, 1.0]. 1.0 = pure Lasso, 0.0 = pure Ridge.
    pub fn with_l1_ratios(mut self, l1_ratios: Vec<f64>) -> Self {
        self.l1_ratios = l1_ratios;
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

impl Default for ElasticNetCrossValidated {
    fn default() -> Self {
        Self::new()
    }
}

/// A fitted Elastic Net regression model selected via cross-validation.
#[derive(Debug, Clone)]
pub struct FittedElasticNetCrossValidated {
    inner: FittedElasticNetRegressor,
    best_alpha: f64,
    best_l1_ratio: f64,
}

impl FittedElasticNetCrossValidated {
    /// Returns the lambda value selected by cross-validation.
    pub fn best_alpha(&self) -> f64 {
        self.best_alpha
    }

    /// Returns the L1 ratio selected by cross-validation.
    pub fn best_l1_ratio(&self) -> f64 {
        self.best_l1_ratio
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

impl Fit<f64> for ElasticNetCrossValidated {
    type Fitted = FittedElasticNetCrossValidated;

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
        if self.l1_ratios.is_empty() {
            return Err(RustMlError::InvalidParameter(
                "l1_ratios must not be empty".into(),
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
        for &ratio in &self.l1_ratios {
            if !(0.0..=1.0).contains(&ratio) {
                return Err(RustMlError::InvalidParameter(
                    "all l1_ratios must be between 0.0 and 1.0".into(),
                ));
            }
        }

        let with_intercept = self.with_intercept;

        let mut best_alpha = self.alphas[0];
        let mut best_l1_ratio = self.l1_ratios[0];
        let mut best_mean_score = f64::NEG_INFINITY;

        for &alpha in &self.alphas {
            for &l1_ratio in &self.l1_ratios {
                let scores = cross_val_score(
                    x,
                    y,
                    self.cv_folds,
                    |x_train, y_train, x_test| {
                        let model = ElasticNetRegressor::new()
                            .with_lambda(alpha)
                            .with_alpha(l1_ratio)
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
                    best_l1_ratio = l1_ratio;
                }
            }
        }

        // Fit the final model with the best parameters on all data
        let final_model = ElasticNetRegressor::new()
            .with_lambda(best_alpha)
            .with_alpha(best_l1_ratio)
            .with_intercept(with_intercept);
        let inner = final_model.fit(x, y)?;

        Ok(FittedElasticNetCrossValidated {
            inner,
            best_alpha,
            best_l1_ratio,
        })
    }
}

impl Predict<f64> for FittedElasticNetCrossValidated {
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
        let x = Array2::from_shape_vec((n, 1), (0..n).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_vec((0..n).map(|i| 2.0 + 3.0 * i as f64).collect());
        (x, y)
    }

    #[test]
    fn test_elastic_net_cv_basic() {
        let (x, y) = make_linear_data(50);

        let fitted = ElasticNetCrossValidated::new().fit(&x, &y).unwrap();

        assert!(fitted.r_squared() > 0.99);
        assert!(fitted.best_alpha() > 0.0);
        assert!(fitted.best_l1_ratio() >= 0.0 && fitted.best_l1_ratio() <= 1.0);
    }

    #[test]
    fn test_elastic_net_cv_selects_small_alpha() {
        let (x, y) = make_linear_data(50);

        let fitted = ElasticNetCrossValidated::new()
            .with_alphas(vec![0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
            .with_l1_ratios(vec![0.3, 0.5, 0.7])
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
    fn test_elastic_net_cv_coefficients() {
        let (x, y) = make_linear_data(50);

        let fitted = ElasticNetCrossValidated::new()
            .with_alphas(vec![0.001, 0.01])
            .with_l1_ratios(vec![0.5])
            .fit(&x, &y)
            .unwrap();

        assert_abs_diff_eq!(fitted.coefficients()[0], 3.0, epsilon = 0.1);
    }

    #[test]
    fn test_elastic_net_cv_predict() {
        let (x, y) = make_linear_data(50);

        let fitted = ElasticNetCrossValidated::new()
            .with_alphas(vec![0.001, 0.01])
            .with_l1_ratios(vec![0.5])
            .fit(&x, &y)
            .unwrap();

        let x_test = Array2::from_shape_vec((3, 1), vec![50.0, 51.0, 52.0]).unwrap();
        let preds = fitted.predict(&x_test).unwrap();
        assert_eq!(preds.len(), 3);
        assert_abs_diff_eq!(preds[0], 2.0 + 3.0 * 50.0, epsilon = 1.0);
    }

    #[test]
    fn test_elastic_net_cv_grid_search() {
        let (x, y) = make_linear_data(50);

        let fitted = ElasticNetCrossValidated::new()
            .with_alphas(vec![0.01, 1.0])
            .with_l1_ratios(vec![0.1, 0.5, 0.9])
            .with_cv_folds(3)
            .fit(&x, &y)
            .unwrap();

        // Should pick from the provided grid
        assert!(
            [0.01, 1.0].contains(&fitted.best_alpha()),
            "best_alpha {} not in grid",
            fitted.best_alpha()
        );
        assert!(
            [0.1, 0.5, 0.9].contains(&fitted.best_l1_ratio()),
            "best_l1_ratio {} not in grid",
            fitted.best_l1_ratio()
        );
    }

    #[test]
    fn test_elastic_net_cv_shape_mismatch() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];

        let result = ElasticNetCrossValidated::new().fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_elastic_net_cv_empty_input() {
        let x = Array2::<f64>::zeros((0, 1));
        let y = Array1::<f64>::zeros(0);

        let result = ElasticNetCrossValidated::new().fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_elastic_net_cv_empty_alphas() {
        let (x, y) = make_linear_data(20);

        let result = ElasticNetCrossValidated::new()
            .with_alphas(vec![])
            .fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_elastic_net_cv_empty_l1_ratios() {
        let (x, y) = make_linear_data(20);

        let result = ElasticNetCrossValidated::new()
            .with_l1_ratios(vec![])
            .fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_elastic_net_cv_negative_alpha() {
        let (x, y) = make_linear_data(20);

        let result = ElasticNetCrossValidated::new()
            .with_alphas(vec![0.1, -1.0])
            .fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_elastic_net_cv_invalid_l1_ratio() {
        let (x, y) = make_linear_data(20);

        let result = ElasticNetCrossValidated::new()
            .with_l1_ratios(vec![0.5, 1.5])
            .fit(&x, &y);
        assert!(result.is_err());

        let result = ElasticNetCrossValidated::new()
            .with_l1_ratios(vec![-0.1, 0.5])
            .fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_elastic_net_cv_invalid_folds() {
        let (x, y) = make_linear_data(20);

        let result = ElasticNetCrossValidated::new().with_cv_folds(1).fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_elastic_net_cv_no_intercept() {
        let x = Array2::from_shape_vec((50, 1), (0..50).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_vec((0..50).map(|i| 3.0 * i as f64).collect());

        let fitted = ElasticNetCrossValidated::new()
            .with_alphas(vec![0.001, 0.01])
            .with_l1_ratios(vec![0.5])
            .with_intercept(false)
            .fit(&x, &y)
            .unwrap();

        assert!(fitted.intercept().is_none());
    }

    #[test]
    fn test_elastic_net_cv_default() {
        let encv = ElasticNetCrossValidated::default();
        let (x, y) = make_linear_data(50);

        let result = encv.fit(&x, &y);
        assert!(result.is_ok());
    }
}

impl anofox_ml_core::RegressorScore<f64> for FittedElasticNetCrossValidated {}
