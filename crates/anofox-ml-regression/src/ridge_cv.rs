//! Ridge regression with built-in cross-validation (sklearn `RidgeCV`).
//!
//! Objective (sklearn scale): `||y − Xw − b||² + alpha·||w||²`.
//!
//! - Default (`cv=None` in sklearn): efficient leave-one-out. One thin SVD
//!   of the centred design `Xc = U S Vᵀ` gives, for every alpha, the hat
//!   diagonal `h_i = 1/n + Σ_k U_ik² s_k²/(s_k² + alpha)` (the `1/n` is the
//!   unpenalised intercept) and the exact LOO residuals
//!   `(y_i − ŷ_i) / (1 − h_i)`; the criterion is their mean square (sklearn's
//!   `scoring=None`).
//! - `with_cv_folds(k)` / `with_fold_ids(..)`: k-fold CV (contiguous, or seeded shuffled
//!   folds or given fold ids), one SVD per training fold for all alphas,
//!   scored by the pooled held-out MSE.
//!
//! The selected model is the closed-form ridge solution at `alpha_` on the
//! full data. Ties keep the first alpha in the given order (as sklearn).

use crate::convert::{check_xy, linear_predict};
use crate::elastic_net_cv::{r_squared, train_rows, FoldSpec};
use anofox_ml_core::{Fit, Predict, Result, RustMlError};
use faer::Mat;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

/// Ridge regression with cross-validated alpha selection.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RidgeCrossValidated {
    alphas: Vec<f64>,
    cv_folds: Option<usize>,
    fold_ids: Option<Vec<usize>>,
    seed: u64,
    #[serde(default)]
    shuffle: bool,
    with_intercept: bool,
}

impl RidgeCrossValidated {
    pub fn new() -> Self {
        Self {
            alphas: vec![0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            cv_folds: None,
            fold_ids: None,
            seed: 0,
            shuffle: false,
            with_intercept: true,
        }
    }

    pub fn with_alphas(mut self, alphas: Vec<f64>) -> Self {
        self.alphas = alphas;
        self
    }

    /// Use k-fold CV (contiguous folds, as sklearn's `cv=int`) instead of
    /// the default efficient leave-one-out.
    pub fn with_cv_folds(mut self, folds: usize) -> Self {
        self.cv_folds = Some(folds);
        self
    }

    /// Use k-fold CV with an explicit fold label per row.
    pub fn with_fold_ids(mut self, fold_ids: Vec<usize>) -> Self {
        self.fold_ids = Some(fold_ids);
        self
    }

    /// Shuffle the rows with this seed before cutting the k folds.
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.seed = seed;
        self.shuffle = true;
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
    coef: Array1<f64>,
    intercept: Option<f64>,
    best_alpha: f64,
    alphas: Array1<f64>,
    cv_mse: Array1<f64>,
    r_squared: f64,
}

impl FittedRidgeCrossValidated {
    /// Selected alpha (sklearn `alpha_`).
    pub fn best_alpha(&self) -> f64 {
        self.best_alpha
    }

    /// The alpha grid in the given order.
    pub fn alphas(&self) -> &Array1<f64> {
        &self.alphas
    }

    /// CV mean squared error per alpha: mean squared LOO residual (default,
    /// sklearn `cv_results_.mean(0)`) or pooled k-fold MSE.
    pub fn cv_mse(&self) -> &Array1<f64> {
        &self.cv_mse
    }

    pub fn coefficients(&self) -> Array1<f64> {
        self.coef.clone()
    }

    pub fn intercept(&self) -> Option<f64> {
        self.intercept
    }

    /// Training R² of the selected model.
    pub fn r_squared(&self) -> f64 {
        self.r_squared
    }
}

/// Thin SVD of the centred design and the projected response.
struct RidgeSvd {
    u: Array2<f64>,
    s: Vec<f64>,
    v: Array2<f64>,
    uty: Array1<f64>,
    x_mean: Array1<f64>,
    y_mean: f64,
    yc: Array1<f64>,
    cutoff: f64,
}

impl RidgeSvd {
    fn new(x: ArrayView2<f64>, y: ArrayView1<f64>, fit_intercept: bool) -> Result<Self> {
        let (n, p) = x.dim();
        let (x_mean, y_mean) = if fit_intercept {
            (x.mean_axis(Axis(0)).unwrap(), y.mean().unwrap())
        } else {
            (Array1::zeros(p), 0.0)
        };
        let xc = &x - &x_mean;
        let yc = &y - y_mean;
        let m = Mat::from_fn(n, p, |i, j| xc[[i, j]]);
        let svd = m
            .thin_svd()
            .map_err(|e| RustMlError::InvalidParameter(format!("SVD failed: {e:?}")))?;
        let (u, s, v) = (svd.U(), svd.S().column_vector(), svd.V());
        let r = s.nrows();
        let u = Array2::from_shape_fn((n, r), |(i, k)| u[(i, k)]);
        let v = Array2::from_shape_fn((p, r), |(j, k)| v[(j, k)]);
        let s: Vec<f64> = (0..r).map(|k| s[k]).collect();
        let smax = s.iter().fold(0.0_f64, |a, &b| a.max(b));
        let uty = u.t().dot(&yc);
        Ok(Self {
            u,
            v,
            uty,
            x_mean,
            y_mean,
            yc,
            cutoff: smax * 1e-12 * n.max(p) as f64,
            s,
        })
    }

    /// Shrinkage factors `s²/(s² + alpha)`; directions with a numerically
    /// zero singular value contribute nothing.
    fn shrink(&self, alpha: f64) -> Array1<f64> {
        Array1::from_iter(self.s.iter().map(|&s| {
            if s <= self.cutoff {
                0.0
            } else {
                s * s / (s * s + alpha)
            }
        }))
    }

    fn coef(&self, alpha: f64) -> (Array1<f64>, f64) {
        let d = Array1::from_iter(self.s.iter().zip(self.uty.iter()).map(|(&s, &uy)| {
            if s <= self.cutoff {
                0.0
            } else {
                s / (s * s + alpha) * uy
            }
        }));
        let coef = self.v.dot(&d);
        let b = self.y_mean - self.x_mean.dot(&coef);
        (coef, b)
    }

    /// Mean squared exact leave-one-out residual.
    fn loo_mse(&self, alpha: f64, fit_intercept: bool) -> f64 {
        let n = self.u.nrows();
        let h = self.shrink(alpha);
        let fitted = self.u.dot(&(&h * &self.uty));
        let base = if fit_intercept { 1.0 / n as f64 } else { 0.0 };
        let mut sse = 0.0;
        for i in 0..n {
            let hii = base
                + self
                    .u
                    .row(i)
                    .iter()
                    .zip(h.iter())
                    .map(|(u, h)| u * u * h)
                    .sum::<f64>();
            let e = (self.yc[i] - fitted[i]) / (1.0 - hii);
            sse += e * e;
        }
        sse / n as f64
    }
}

impl Fit<f64> for RidgeCrossValidated {
    type Fitted = FittedRidgeCrossValidated;

    fn fit_view(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<Self::Fitted> {
        check_xy(x, y)?;
        if self.alphas.is_empty() {
            return Err(RustMlError::InvalidParameter(
                "alphas must not be empty".into(),
            ));
        }
        if self.alphas.iter().any(|a| !(a.is_finite() && *a >= 0.0)) {
            return Err(RustMlError::InvalidParameter(
                "all alphas must be non-negative".into(),
            ));
        }
        let n = x.nrows();
        let full = RidgeSvd::new(x, y, self.with_intercept)?;

        let cv_mse: Array1<f64> = if self.cv_folds.is_some() || self.fold_ids.is_some() {
            let spec = FoldSpec {
                n_folds: self.cv_folds.unwrap_or(5),
                fold_ids: self.fold_ids.clone(),
                seed: self.seed,
                shuffle: self.shuffle,
            };
            let mut sse = Array1::<f64>::zeros(self.alphas.len());
            for test in spec.test_sets(n)? {
                let train = train_rows(n, &test);
                let xt = x.select(Axis(0), &train);
                let svd = RidgeSvd::new(
                    xt.view(),
                    y.select(Axis(0), &train).view(),
                    self.with_intercept,
                )?;
                let xv = x.select(Axis(0), &test);
                for (l, &a) in self.alphas.iter().enumerate() {
                    let (c, b) = svd.coef(a);
                    let pred = xv.dot(&c) + b;
                    sse[l] += test
                        .iter()
                        .zip(pred.iter())
                        .map(|(&r, p)| (y[r] - p).powi(2))
                        .sum::<f64>();
                }
            }
            sse / n as f64
        } else {
            self.alphas
                .iter()
                .map(|&a| full.loo_mse(a, self.with_intercept))
                .collect()
        };

        // First minimum in the given order; NaN/inf (degenerate LOO) never wins.
        let mut best = 0;
        for l in 1..cv_mse.len() {
            if cv_mse[l] < cv_mse[best] || !cv_mse[best].is_finite() {
                best = l;
            }
        }
        let best_alpha = self.alphas[best];
        let (coef, b) = full.coef(best_alpha);
        let b = if self.with_intercept { b } else { 0.0 };
        Ok(FittedRidgeCrossValidated {
            r_squared: r_squared(x, y, &coef, b),
            intercept: self.with_intercept.then_some(b),
            coef,
            best_alpha,
            alphas: Array1::from(self.alphas.clone()),
            cv_mse,
        })
    }
}

impl Predict<f64> for FittedRidgeCrossValidated {
    fn predict_view(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        linear_predict(x, &self.coef, self.intercept.unwrap_or(0.0))
    }
}

impl anofox_ml_core::RegressorScore<f64> for FittedRidgeCrossValidated {}

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
    fn test_ridge_cv_basic() {
        let (x, y) = make_linear_data(50);

        let fitted = RidgeCrossValidated::new().fit(&x, &y).unwrap();

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

        let result = RidgeCrossValidated::new().with_alphas(vec![]).fit(&x, &y);
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

        let result = RidgeCrossValidated::new().with_cv_folds(1).fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_ridge_cv_no_intercept() {
        let x = Array2::from_shape_vec((50, 1), (0..50).map(|i| i as f64).collect()).unwrap();
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

    const ALPHAS: [f64; 5] = [0.01, 0.1, 1.0, 10.0, 100.0];

    fn close(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (i, (u, v)) in a.iter().zip(b).enumerate() {
            assert!((u - v).abs() <= tol * v.abs().max(1.0), "[{i}] {u} vs {v}");
        }
    }

    #[test]
    fn test_ridge_cv_gcv_matches_sklearn() {
        // sklearn RidgeCV(alphas, store_cv_results=True) — efficient LOO.
        let (x, y, _) = crate::elastic_net_cv::tests::cv_data();
        let f = RidgeCrossValidated::new()
            .with_alphas(ALPHAS.to_vec())
            .fit(&x, &y)
            .unwrap();
        assert_eq!(f.best_alpha(), 1.0);
        close(
            f.cv_mse().as_slice().unwrap(),
            &[
                5.120197846008991,
                5.118904488928734,
                5.110245059792773,
                5.307168635521339,
                9.144178501214792,
            ],
            1e-12,
        );
        close(
            f.coefficients().as_slice().unwrap(),
            &[
                1.1736113494582519,
                -2.3218730973444663,
                0.0760779433772365,
                0.3986148863416048,
                0.021443627610655713,
            ],
            1e-12,
        );
        assert!((f.intercept().unwrap() - 0.962700162304877).abs() < 1e-12);

        let f = RidgeCrossValidated::new()
            .with_alphas(ALPHAS.to_vec())
            .with_intercept(false)
            .fit(&x, &y)
            .unwrap();
        close(
            f.cv_mse().as_slice().unwrap(),
            &[
                5.766465381753935,
                5.764581546934078,
                5.750219878014937,
                5.908241029821448,
                9.836512999089642,
            ],
            1e-12,
        );
    }

    #[test]
    fn test_ridge_cv_kfold_matches_sklearn() {
        // sklearn RidgeCV(alphas, cv=<fold ids>, scoring="neg_mean_squared_error").
        let (x, y, ids) = crate::elastic_net_cv::tests::cv_data();
        let f = RidgeCrossValidated::new()
            .with_alphas(ALPHAS.to_vec())
            .with_fold_ids(ids)
            .fit(&x, &y)
            .unwrap();
        assert_eq!(f.best_alpha(), 10.0);
        assert!((f.cv_mse()[3] - 5.087986451028778).abs() < 1e-10);
        close(
            f.coefficients().as_slice().unwrap(),
            &[
                0.9905700364045105,
                -2.0121995963938755,
                0.025268938525508083,
                0.27739261542622196,
                0.05139623213807374,
            ],
            1e-12,
        );
        assert!((f.intercept().unwrap() - 1.0119198098088795).abs() < 1e-12);
    }

    #[test]
    fn test_ridge_cv_seeded_kfold() {
        let (x, y, _) = crate::elastic_net_cv::tests::cv_data();
        let a = RidgeCrossValidated::new()
            .with_cv_folds(4)
            .with_random_state(3)
            .fit(&x, &y)
            .unwrap();
        let b = RidgeCrossValidated::new()
            .with_cv_folds(4)
            .with_random_state(3)
            .fit(&x, &y)
            .unwrap();
        assert_eq!(a.cv_mse(), b.cv_mse());
    }
}
