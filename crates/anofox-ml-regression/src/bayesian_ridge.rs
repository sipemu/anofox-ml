//! Bayesian Ridge Regression and Automatic Relevance Determination (ARD).
//!
//! Mirrors `sklearn.linear_model.BayesianRidge` and `ARDRegression`. The model
//! is `y ~ N(Xβ + b, 1/α)` with prior `β_j ~ N(0, 1/λ_j)`. BayesianRidge ties
//! all `λ_j = λ`; ARD has a per-feature `λ_j` and prunes features whose
//! precision exceeds `threshold_lambda`. Hyperparameters are estimated by
//! evidence maximisation with MacKay's fixed-point updates (as sklearn),
//! delegated to `anofox_regression::solvers::{BayesianRidge, ArdRegression}`.
//!
//! `predict_std` delegates to upstream, which centres the query rows at the
//! training mean before evaluating `sqrt(xᵀ Σ x + 1/α)`; sklearn's
//! `return_std` uses the raw rows, so the two differ for non-centred data
//! (tracked in #72).

use crate::convert::{
    check_n_features, check_xy, col_to_ndarray, linear_predict, mat_to_ndarray, ndarray_to_col,
    ndarray_to_mat, upstream_err,
};
use anofox_ml_core::{Fit, Predict, Result, RustMlError};
use anofox_regression::solvers::{
    ArdRegression as InnerArd, BayesianRidge as InnerBayesianRidge, FittedArd as InnerFittedArd,
    FittedBayesianRidge as InnerFittedBayesianRidge, FittedRegressor as _, Regressor as _,
};

fn no_posterior() -> RustMlError {
    RustMlError::NotFitted(
        "posterior predictive std needs the in-memory fitted model (not available after \
         deserialization)"
            .into(),
    )
}
use ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// BayesianRidge
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BayesianRidge {
    pub max_iter: usize,
    pub tol: f64,
    pub alpha_1: f64,
    pub alpha_2: f64,
    pub lambda_1: f64,
    pub lambda_2: f64,
    pub fit_intercept: bool,
}

impl BayesianRidge {
    pub fn new() -> Self {
        Self {
            max_iter: 300,
            tol: 1e-3,
            alpha_1: 1e-6,
            alpha_2: 1e-6,
            lambda_1: 1e-6,
            lambda_2: 1e-6,
            fit_intercept: true,
        }
    }
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
}

impl Default for BayesianRidge {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedBayesianRidge {
    pub coef: Array1<f64>,
    pub intercept: f64,
    /// Noise precision (sklearn `alpha_`).
    pub alpha: f64,
    /// Weight precision (sklearn `lambda_`).
    pub lambda: f64,
    pub n_iter: usize,
    /// Posterior covariance of the coefficients (sklearn `sigma_`).
    #[serde(default)]
    pub sigma: Array2<f64>,
    /// Log marginal likelihood at the optimum (sklearn `scores_[-1]`).
    #[serde(default)]
    pub log_marginal_likelihood: f64,
    n_features: usize,
    #[serde(skip)]
    inner: Option<InnerFittedBayesianRidge>,
}

impl FittedBayesianRidge {
    /// Posterior predictive standard deviation per query point (sklearn
    /// `predict(X, return_std=True)[1]`).
    pub fn predict_std(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        check_n_features(x, self.n_features)?;
        let inner = self.inner.as_ref().ok_or_else(no_posterior)?;
        Ok(col_to_ndarray(&inner.predict_std(&ndarray_to_mat(x))))
    }

    /// Predictive mean and standard deviation (sklearn `return_std=True`).
    pub fn predict_with_std(&self, x: &Array2<f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        Ok((self.predict(x)?, self.predict_std(x)?))
    }
}

impl Fit<f64> for BayesianRidge {
    type Fitted = FittedBayesianRidge;

    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        check_xy(x, y)?;
        let fitted = InnerBayesianRidge::builder()
            .fit_intercept(self.fit_intercept)
            .max_iter(self.max_iter)
            .tolerance(self.tol)
            .alpha_1(self.alpha_1)
            .alpha_2(self.alpha_2)
            .lambda_1(self.lambda_1)
            .lambda_2(self.lambda_2)
            .build()
            .fit(&ndarray_to_mat(x), &ndarray_to_col(y))
            .map_err(upstream_err)?;
        let res = fitted.result();
        Ok(FittedBayesianRidge {
            coef: col_to_ndarray(&res.coefficients),
            intercept: res.intercept.unwrap_or(0.0),
            alpha: fitted.alpha(),
            lambda: fitted.lambda(),
            n_iter: fitted.n_iter(),
            sigma: mat_to_ndarray(fitted.sigma()),
            log_marginal_likelihood: fitted.log_marginal_likelihood(),
            n_features: x.ncols(),
            inner: Some(fitted),
        })
    }
}

impl Predict<f64> for FittedBayesianRidge {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        linear_predict(x, &self.coef, self.intercept)
    }
}

// ---------------------------------------------------------------------------
// ARDRegression (per-feature lambda; sparsity-inducing)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ARDRegression {
    pub max_iter: usize,
    pub tol: f64,
    pub alpha_1: f64,
    pub alpha_2: f64,
    pub lambda_1: f64,
    pub lambda_2: f64,
    /// Drop features whose λ_j exceeds this (so prior precision is huge).
    pub threshold_lambda: f64,
    pub fit_intercept: bool,
}

impl ARDRegression {
    pub fn new() -> Self {
        Self {
            max_iter: 300,
            tol: 1e-3,
            alpha_1: 1e-6,
            alpha_2: 1e-6,
            lambda_1: 1e-6,
            lambda_2: 1e-6,
            threshold_lambda: 1e4,
            fit_intercept: true,
        }
    }
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }
    pub fn with_threshold_lambda(mut self, threshold_lambda: f64) -> Self {
        self.threshold_lambda = threshold_lambda;
        self
    }
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
}

impl Default for ARDRegression {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedARDRegression {
    pub coef: Array1<f64>,
    pub intercept: f64,
    /// Per-feature weight precisions (sklearn `lambda_`).
    pub lambdas: Array1<f64>,
    /// Noise precision (sklearn `alpha_`).
    pub alpha: f64,
    pub n_iter: usize,
    /// Posterior covariance over the kept features (sklearn `sigma_`).
    #[serde(default)]
    pub sigma: Array2<f64>,
    /// Features with `λ_j < threshold_lambda`, the rows/cols of `sigma`.
    #[serde(default)]
    pub kept_features: Vec<usize>,
    n_features: usize,
    #[serde(skip)]
    inner: Option<InnerFittedArd>,
}

impl FittedARDRegression {
    /// Posterior predictive standard deviation per query point (sklearn
    /// `predict(X, return_std=True)[1]`).
    pub fn predict_std(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        check_n_features(x, self.n_features)?;
        let inner = self.inner.as_ref().ok_or_else(no_posterior)?;
        Ok(col_to_ndarray(&inner.predict_std(&ndarray_to_mat(x))))
    }

    /// Predictive mean and standard deviation (sklearn `return_std=True`).
    pub fn predict_with_std(&self, x: &Array2<f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        Ok((self.predict(x)?, self.predict_std(x)?))
    }
}

impl Fit<f64> for ARDRegression {
    type Fitted = FittedARDRegression;

    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        check_xy(x, y)?;
        let fitted = InnerArd::builder()
            .fit_intercept(self.fit_intercept)
            .max_iter(self.max_iter)
            .tolerance(self.tol)
            .alpha_1(self.alpha_1)
            .alpha_2(self.alpha_2)
            .lambda_1(self.lambda_1)
            .lambda_2(self.lambda_2)
            .threshold_lambda(self.threshold_lambda)
            .build()
            .fit(&ndarray_to_mat(x), &ndarray_to_col(y))
            .map_err(upstream_err)?;
        let res = fitted.result();
        Ok(FittedARDRegression {
            coef: col_to_ndarray(&res.coefficients),
            intercept: res.intercept.unwrap_or(0.0),
            lambdas: Array1::from(fitted.lambdas().to_vec()),
            alpha: fitted.alpha(),
            n_iter: fitted.n_iter(),
            sigma: mat_to_ndarray(fitted.sigma()),
            kept_features: fitted.kept_features().to_vec(),
            n_features: x.ncols(),
            inner: Some(fitted),
        })
    }
}

impl Predict<f64> for FittedARDRegression {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        linear_predict(x, &self.coef, self.intercept)
    }
}

impl anofox_ml_core::RegressorScore<f64> for FittedBayesianRidge {}
impl anofox_ml_core::RegressorScore<f64> for FittedARDRegression {}

#[cfg(test)]
mod tests {
    use super::*;

    /// `data.make(n, p, m=7, coefs)` from the sklearn reference script.
    fn make(n: usize, p: usize, coefs: &[f64]) -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_fn((n, p), |(i, j)| {
            ((i * (j + 3) * 7 + j * 5 + (i * i) % 11) % 17) as f64 / 4.0 - 2.0
        });
        let y = Array1::from_shape_fn(n, |i| {
            (0..p).map(|j| x[[i, j]] * coefs[j]).sum::<f64>()
                + 1.0
                + 0.3 * (((i * 13) % 7) as f64 - 3.0)
        });
        (x, y)
    }

    fn close(a: &Array1<f64>, b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (i, (u, v)) in a.iter().zip(b).enumerate() {
            assert!((u - v).abs() <= tol * v.abs().max(1.0), "[{i}] {u} vs {v}");
        }
    }

    #[test]
    fn test_bayesian_ridge_recovers_simple_line() {
        // y = 2 + 3x
        let x = Array2::from_shape_vec((10, 1), (0..10).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_vec((0..10).map(|i| 2.0 + 3.0 * i as f64).collect());

        let fitted = BayesianRidge::new().fit(&x, &y).unwrap();
        assert!((fitted.coef[0] - 3.0).abs() < 0.1);
        assert!((fitted.intercept - 2.0).abs() < 0.2);
    }

    #[test]
    fn test_bayesian_ridge_matches_sklearn() {
        // sklearn BayesianRidge().fit(X + 3, y); non-centred features make
        // the return_std convention (raw rows, no centring) observable.
        let (x, y) = make(25, 3, &[1.5, -2.0, 0.0]);
        let x = x + 3.0;
        let f = BayesianRidge::new().fit(&x, &y).unwrap();
        close(
            &f.coef,
            &[1.4993452959666345, -2.1266612946889794, 0.11299029500510167],
            1e-8,
        );
        assert!((f.intercept - 2.5601473621468904).abs() < 1e-8);
        assert!((f.alpha - 2.560172761565585).abs() < 1e-7);
        assert!((f.lambda - 0.44040294405980474).abs() < 1e-8);
        assert_eq!(f.n_iter, 4);
        assert!((f.log_marginal_likelihood - -31.98716685015977).abs() < 1e-6);
        let q = x.slice(ndarray::s![..4, ..]).to_owned();
        let mean = f.predict(&q).unwrap();
        close(
            &mean,
            &[
                -0.33002922241882304,
                4.258974163403245,
                -0.44749580506147035,
                -3.225244384586525,
            ],
            1e-8,
        );
    }

    #[test]
    #[ignore = "upstream anofox-regression 0.7 predict_std centres X; sklearn uses raw rows (#72)"]
    fn test_bayesian_ridge_return_std_matches_sklearn() {
        let (x, y) = make(25, 3, &[1.5, -2.0, 0.0]);
        let x = x + 3.0;
        let f = BayesianRidge::new().fit(&x, &y).unwrap();
        let q = x.slice(ndarray::s![..4, ..]).to_owned();
        let std = f.predict_std(&q).unwrap();
        close(
            &std,
            &[
                0.7272844276581165,
                0.7679079183609776,
                0.9447285505810161,
                0.784118088769497,
            ],
            1e-8,
        );
    }

    #[test]
    fn test_ard_matches_sklearn() {
        let (x, y) = make(30, 5, &[1.5, 0.0, 0.0, -2.0, 0.0]);
        let f = ARDRegression::new().fit(&x, &y).unwrap();
        close(
            &f.coef,
            &[
                1.431949016054046,
                -0.06108482495597584,
                0.0067375797842431,
                -2.083046476069988,
                -0.0002046092433045455,
            ],
            1e-6,
        );
        assert!((f.intercept - 0.9822139161512056).abs() < 1e-6);
        assert!((f.alpha - 2.5847087985528785).abs() < 1e-5);
        assert_eq!(f.kept_features, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    #[ignore = "upstream anofox-regression 0.7 predict_std centres X; sklearn uses raw rows (#72)"]
    fn test_ard_return_std_matches_sklearn() {
        let (x, y) = make(30, 5, &[1.5, 0.0, 0.0, -2.0, 0.0]);
        let f = ARDRegression::new().fit(&x, &y).unwrap();
        let q = x.slice(ndarray::s![..4, ..]).to_owned();
        close(
            &f.predict_std(&q).unwrap(),
            &[
                0.6660447122857337,
                0.6391903592890759,
                0.6572354803017733,
                0.6460091907360573,
            ],
            1e-6,
        );
    }

    #[test]
    fn test_bayesian_rejects_empty() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<f64>::zeros(0);
        assert!(BayesianRidge::new().fit(&x, &y).is_err());
        assert!(ARDRegression::new().fit(&x, &y).is_err());
    }

    #[test]
    fn test_ard_drops_irrelevant_features() {
        // y = 5*x0 + 0*x1 + 0*x2 — ARD should shrink x1, x2.
        let n = 80;
        let mut x = Array2::<f64>::zeros((n, 3));
        for i in 0..n {
            x[[i, 0]] = i as f64 - 40.0;
            x[[i, 1]] = ((i * 7 % 13) as f64) - 6.0;
            x[[i, 2]] = ((i * 5 % 11) as f64) - 5.0;
        }
        let y = x.column(0).mapv(|v| 5.0 * v);
        let fitted = ARDRegression::new().fit(&x, &y).unwrap();
        assert!((fitted.coef[0] - 5.0).abs() < 0.5);
        assert!(fitted.coef[1].abs() < 0.5);
        assert!(fitted.coef[2].abs() < 0.5);
    }
}
