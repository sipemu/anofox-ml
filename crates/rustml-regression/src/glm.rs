//! GLM (Generalized Linear Model) wrappers for Poisson and Binomial regression.

use crate::convert::{col_to_ndarray, ndarray_to_col, ndarray_to_mat};
use anofox_regression::{FittedRegressor as _, Regressor as _};
use anofox_regression::{
    BinomialRegressor as InnerBinomial, FittedBinomial, FittedPoisson,
    PoissonRegressor as InnerPoisson,
};
use ndarray::{Array1, Array2};
use rustml_core::{Fit, Predict, Result, RustMlError};

// ---------------------------------------------------------------------------
// Poisson
// ---------------------------------------------------------------------------

/// Poisson GLM regression estimator (log link).
///
/// Models count data where `E[Y] = exp(Xβ)`.
#[derive(Debug, Clone)]
pub struct PoissonRegressor {
    with_intercept: bool,
}

impl PoissonRegressor {
    pub fn new() -> Self {
        Self {
            with_intercept: true,
        }
    }

    pub fn with_intercept(mut self, include: bool) -> Self {
        self.with_intercept = include;
        self
    }
}

impl Default for PoissonRegressor {
    fn default() -> Self {
        Self::new()
    }
}

/// A fitted Poisson regression model.
#[derive(Debug, Clone)]
pub struct FittedPoissonRegressor {
    inner: FittedPoisson,
    n_features: usize,
}

impl FittedPoissonRegressor {
    pub fn coefficients(&self) -> Array1<f64> {
        col_to_ndarray(self.inner.coefficients())
    }

    pub fn intercept(&self) -> Option<f64> {
        self.inner.intercept()
    }

    pub fn r_squared(&self) -> f64 {
        self.inner.r_squared()
    }
}

impl Fit<f64> for PoissonRegressor {
    type Fitted = FittedPoissonRegressor;

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

        let inner_model = InnerPoisson::log()
            .with_intercept(self.with_intercept)
            .build();

        let fitted = inner_model
            .fit(&x_mat, &y_col)
            .map_err(|e| RustMlError::InvalidParameter(e.to_string()))?;

        Ok(FittedPoissonRegressor {
            inner: fitted,
            n_features: x.ncols(),
        })
    }
}

impl Predict<f64> for FittedPoissonRegressor {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }

        let x_mat = ndarray_to_mat(x);
        let preds = self.inner.predict(&x_mat);
        Ok(col_to_ndarray(&preds))
    }
}

// ---------------------------------------------------------------------------
// Binomial (logistic)
// ---------------------------------------------------------------------------

/// Binomial GLM regression estimator (logistic regression).
///
/// Models binary outcomes where `E[Y] = logit⁻¹(Xβ)`.
#[derive(Debug, Clone)]
pub struct BinomialRegressor {
    with_intercept: bool,
}

impl BinomialRegressor {
    pub fn new() -> Self {
        Self {
            with_intercept: true,
        }
    }

    pub fn with_intercept(mut self, include: bool) -> Self {
        self.with_intercept = include;
        self
    }
}

impl Default for BinomialRegressor {
    fn default() -> Self {
        Self::new()
    }
}

/// A fitted Binomial (logistic) regression model.
#[derive(Debug, Clone)]
pub struct FittedBinomialRegressor {
    inner: FittedBinomial,
    n_features: usize,
}

impl FittedBinomialRegressor {
    pub fn coefficients(&self) -> Array1<f64> {
        col_to_ndarray(self.inner.coefficients())
    }

    pub fn intercept(&self) -> Option<f64> {
        self.inner.intercept()
    }

    pub fn r_squared(&self) -> f64 {
        self.inner.r_squared()
    }
}

impl Fit<f64> for BinomialRegressor {
    type Fitted = FittedBinomialRegressor;

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

        let inner_model = InnerBinomial::logistic()
            .with_intercept(self.with_intercept)
            .build();

        let fitted = inner_model
            .fit(&x_mat, &y_col)
            .map_err(|e| RustMlError::InvalidParameter(e.to_string()))?;

        Ok(FittedBinomialRegressor {
            inner: fitted,
            n_features: x.ncols(),
        })
    }
}

impl Predict<f64> for FittedBinomialRegressor {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }

        let x_mat = ndarray_to_mat(x);
        let preds = self.inner.predict(&x_mat);
        Ok(col_to_ndarray(&preds))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_poisson_basic() {
        // Simple count data
        let x = Array2::from_shape_vec(
            (8, 1),
            vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        )
        .unwrap();
        // Roughly y ~ exp(0.5 * x), i.e. Poisson counts increasing with x
        let y = array![1.0, 1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 8.0];

        let fitted = PoissonRegressor::new().fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        // Predictions should all be positive (exp link)
        for &p in preds.iter() {
            assert!(p > 0.0, "Poisson predictions must be positive, got {}", p);
        }
    }

    #[test]
    fn test_binomial_basic() {
        // Simple binary classification data
        let x = Array2::from_shape_vec(
            (8, 1),
            vec![-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0],
        )
        .unwrap();
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let fitted = BinomialRegressor::new().fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        // Predictions should be probabilities in [0, 1]
        for &p in preds.iter() {
            assert!(
                (0.0..=1.0).contains(&p),
                "Binomial predictions must be in [0,1], got {}",
                p
            );
        }
    }
}
