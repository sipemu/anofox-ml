//! Lasso regression with built-in cross-validation (sklearn `LassoCV`).
//!
//! `alpha` has sklearn's scale, `1/(2n)·||y − Xw − b||² + alpha·||w||₁`.
//! The whole alpha grid is solved by coordinate descent with warm starts
//! (upstream `fit_path`) per fold and on the full data; the default grid is
//! sklearn's (`n_alphas = 100`, `eps = 1e-3`, `alpha_max = max|Xcᵀyc|/n`).
//! Selection uses the pooled held-out MSE over contiguous (sklearn `KFold`), seeded shuffled or given
//! folds; the coefficients are the full-data path solution at `alpha_`.
//! See [`crate::elastic_net_cv`] for the shared machinery.

use crate::convert::check_xy;
use crate::elastic_net_cv::{
    default_alphas, path_cv, r_squared, sorted_alphas, CdOptions, FoldSpec, PathCv,
};
use anofox_ml_core::{Fit, Predict, Result, RustMlError};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Lasso regression with cross-validated alpha selection.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LassoCrossValidated {
    alphas: Option<Vec<f64>>,
    n_alphas: usize,
    eps: f64,
    folds: FoldSpec,
    with_intercept: bool,
    tol: f64,
    max_iter: usize,
}

impl LassoCrossValidated {
    pub fn new() -> Self {
        Self {
            alphas: None,
            n_alphas: 100,
            eps: 1e-3,
            folds: FoldSpec::new(5),
            with_intercept: true,
            tol: 1e-12,
            max_iter: 100_000,
        }
    }

    /// Explicit alpha grid (sklearn scale; sorted decreasingly). Default: the
    /// data-driven grid.
    pub fn with_alphas(mut self, alphas: Vec<f64>) -> Self {
        self.alphas = Some(alphas);
        self
    }

    /// Size of the automatic grid (sklearn `n_alphas`, default 100).
    pub fn with_n_alphas(mut self, n_alphas: usize) -> Self {
        self.n_alphas = n_alphas;
        self
    }

    /// `alpha_min / alpha_max` of the automatic grid (default `1e-3`).
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Number of k-fold splits (default 5).
    pub fn with_cv_folds(mut self, folds: usize) -> Self {
        self.folds.n_folds = folds;
        self
    }

    /// Explicit fold label per row (overrides `cv_folds` / the seed).
    pub fn with_fold_ids(mut self, fold_ids: Vec<usize>) -> Self {
        self.folds.fold_ids = Some(fold_ids);
        self
    }

    /// Shuffle the rows with this seed before cutting the folds (default:
    /// contiguous folds, as sklearn's `cv=int`).
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.folds.seed = seed;
        self.folds.shuffle = true;
        self
    }

    pub fn with_intercept(mut self, include: bool) -> Self {
        self.with_intercept = include;
        self
    }

    /// Coordinate-descent tolerance (default `1e-12`).
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Maximum coordinate-descent passes per alpha (default 100 000).
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }
}

impl Default for LassoCrossValidated {
    fn default() -> Self {
        Self::new()
    }
}

/// A fitted Lasso regression model selected via cross-validation.
#[derive(Debug, Clone)]
pub struct FittedLassoCrossValidated {
    cv: PathCv,
    intercept: Option<f64>,
    r_squared: f64,
}

impl FittedLassoCrossValidated {
    /// Selected alpha (sklearn `alpha_`).
    pub fn best_alpha(&self) -> f64 {
        self.cv.alphas[self.cv.best]
    }

    /// Largest alpha whose CV error is within one standard error of the
    /// minimum (glmnet `lambda.1se`).
    pub fn alpha_1se(&self) -> f64 {
        self.cv.alphas[self.cv.best_1se]
    }

    /// The alpha grid, decreasing (sklearn `alphas_`).
    pub fn alphas(&self) -> &Array1<f64> {
        &self.cv.alphas
    }

    /// Held-out MSE per alpha and fold, `n_alphas × n_folds` (sklearn
    /// `mse_path_`).
    pub fn mse_path(&self) -> &Array2<f64> {
        &self.cv.mse_path
    }

    /// Pooled held-out MSE per alpha (the selection criterion).
    pub fn cv_mse(&self) -> &Array1<f64> {
        &self.cv.cv_mean
    }

    /// Standard error of `cv_mse`.
    pub fn cv_mse_se(&self) -> &Array1<f64> {
        &self.cv.cv_se
    }

    pub fn coefficients(&self) -> Array1<f64> {
        self.cv.coef.clone()
    }

    pub fn intercept(&self) -> Option<f64> {
        self.intercept
    }

    /// Training R² of the selected model.
    pub fn r_squared(&self) -> f64 {
        self.r_squared
    }
}

impl Fit<f64> for LassoCrossValidated {
    type Fitted = FittedLassoCrossValidated;

    fn fit_view(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<Self::Fitted> {
        check_xy(x, y)?;
        if self.n_alphas == 0 || !(self.eps > 0.0 && self.eps < 1.0) {
            return Err(RustMlError::InvalidParameter(
                "n_alphas must be >= 1 and eps in (0, 1)".into(),
            ));
        }
        let alphas = match &self.alphas {
            Some(a) => sorted_alphas(a)?,
            None => default_alphas(x, y, 1.0, self.n_alphas, self.eps, self.with_intercept)?,
        };
        let test_sets = self.folds.test_sets(x.nrows())?;
        let cd = CdOptions {
            fit_intercept: self.with_intercept,
            tol: self.tol,
            max_iter: self.max_iter,
        };
        let cv = path_cv(x, y, 1.0, &alphas, &test_sets, cd)?;
        let r_squared = r_squared(x, y, &cv.coef, cv.intercept);
        Ok(FittedLassoCrossValidated {
            intercept: self.with_intercept.then_some(cv.intercept),
            r_squared,
            cv,
        })
    }
}

impl Predict<f64> for FittedLassoCrossValidated {
    fn predict_view(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        crate::convert::linear_predict(x, &self.cv.coef, self.intercept.unwrap_or(0.0))
    }
}

impl anofox_ml_core::RegressorScore<f64> for FittedLassoCrossValidated {}

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
    fn test_lasso_cv_basic() {
        let (x, y) = make_linear_data(50);

        let fitted = LassoCrossValidated::new().fit(&x, &y).unwrap();

        assert!(fitted.r_squared() > 0.99);
        assert!(fitted.best_alpha() > 0.0);
    }

    #[test]
    fn test_lasso_cv_selects_best_alpha() {
        let (x, y) = make_linear_data(50);

        let fitted = LassoCrossValidated::new()
            .with_alphas(vec![0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
            .with_cv_folds(5)
            .fit(&x, &y)
            .unwrap();

        // For a clean linear relationship, small lambda should be preferred
        assert!(
            fitted.best_alpha() <= 1.0,
            "expected small lambda for linear data, got {}",
            fitted.best_alpha()
        );
    }

    #[test]
    fn test_lasso_cv_coefficients() {
        let (x, y) = make_linear_data(50);

        let fitted = LassoCrossValidated::new()
            .with_alphas(vec![0.001, 0.01])
            .fit(&x, &y)
            .unwrap();

        assert_abs_diff_eq!(fitted.coefficients()[0], 3.0, epsilon = 0.1);
    }

    #[test]
    fn test_lasso_cv_predict() {
        let (x, y) = make_linear_data(50);

        let fitted = LassoCrossValidated::new()
            .with_alphas(vec![0.001, 0.01])
            .fit(&x, &y)
            .unwrap();

        let x_test = Array2::from_shape_vec((3, 1), vec![50.0, 51.0, 52.0]).unwrap();
        let preds = fitted.predict(&x_test).unwrap();
        assert_eq!(preds.len(), 3);
        assert_abs_diff_eq!(preds[0], 2.0 + 3.0 * 50.0, epsilon = 1.0);
    }

    #[test]
    fn test_lasso_cv_shape_mismatch() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];

        let result = LassoCrossValidated::new().fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_lasso_cv_empty_input() {
        let x = Array2::<f64>::zeros((0, 1));
        let y = Array1::<f64>::zeros(0);

        let result = LassoCrossValidated::new().fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_lasso_cv_empty_alphas() {
        let (x, y) = make_linear_data(20);

        let result = LassoCrossValidated::new().with_alphas(vec![]).fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_lasso_cv_negative_alpha() {
        let (x, y) = make_linear_data(20);

        let result = LassoCrossValidated::new()
            .with_alphas(vec![0.1, -1.0])
            .fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_lasso_cv_invalid_folds() {
        let (x, y) = make_linear_data(20);

        let result = LassoCrossValidated::new().with_cv_folds(1).fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_lasso_cv_no_intercept() {
        let x = Array2::from_shape_vec((50, 1), (0..50).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_vec((0..50).map(|i| 3.0 * i as f64).collect());

        let fitted = LassoCrossValidated::new()
            .with_alphas(vec![0.001, 0.01])
            .with_intercept(false)
            .fit(&x, &y)
            .unwrap();

        assert!(fitted.intercept().is_none());
    }

    #[test]
    fn test_lasso_cv_default() {
        let lcv = LassoCrossValidated::default();
        let (x, y) = make_linear_data(50);

        let result = lcv.fit(&x, &y);
        assert!(result.is_ok());
    }

    #[test]
    fn test_lasso_cv_matches_sklearn() {
        // sklearn LassoCV(cv=<fold ids (3i) mod 5>, tol=1e-12).
        let (x, y, ids) = crate::elastic_net_cv::tests::cv_data();
        let f = LassoCrossValidated::new()
            .with_fold_ids(ids)
            .fit(&x, &y)
            .unwrap();
        assert_eq!(f.alphas().len(), 100);
        assert!((f.alphas()[0] - 3.5731562500000003).abs() < 1e-12);
        assert!((f.alphas()[99] - 0.0035731562500000004).abs() < 1e-14);
        assert!((f.best_alpha() - 0.357315625).abs() < 1e-12);
        let c = f.coefficients();
        let sk = [
            0.8705730414498023,
            -2.087120088581602,
            0.0,
            0.03735636647266589,
            0.0,
        ];
        for j in 0..5 {
            assert!(
                (c[j] - sk[j]).abs() < 1e-7,
                "coef[{j}] {} vs {}",
                c[j],
                sk[j]
            );
        }
        assert!((f.intercept().unwrap() - 0.968021356394059).abs() < 1e-7);
        let rows = [
            (
                0,
                [
                    11.263632116779352,
                    8.669303855341656,
                    13.102639312015741,
                    15.923140258789065,
                    18.831821899414063,
                ],
            ),
            (
                33,
                [
                    5.807648100723549,
                    2.7451419710548994,
                    5.121345214511292,
                    5.02545978542319,
                    5.3338182380765815,
                ],
            ),
        ];
        for (l, sk) in rows {
            for (f_, s) in sk.iter().enumerate() {
                let v = f.mse_path()[[l, f_]];
                assert!((v - s).abs() < 1e-7 * s, "mse[{l},{f_}] {v} vs {s}");
            }
        }
        assert!(f.alpha_1se() >= f.best_alpha());
    }

    #[test]
    fn test_lasso_cv_explicit_alphas_sorted_like_sklearn() {
        let (x, y, ids) = crate::elastic_net_cv::tests::cv_data();
        let f = LassoCrossValidated::new()
            .with_alphas(vec![0.5, 0.05, 0.2, 0.01])
            .with_fold_ids(ids)
            .fit(&x, &y)
            .unwrap();
        assert_eq!(f.alphas().to_vec(), vec![0.5, 0.2, 0.05, 0.01]);
        assert_eq!(f.best_alpha(), 0.5);
        let sk = [
            4.855303157012374,
            4.898888693106557,
            5.169148078611116,
            5.311315764893715,
        ];
        for (l, v) in f.cv_mse().iter().enumerate() {
            assert!((v - sk[l]).abs() < 1e-6 * sk[l], "{v} vs {}", sk[l]);
        }
    }

    #[test]
    fn test_lasso_cv_coef_is_path_solution() {
        // The final model is the full-data CD path at alpha_, not a separate
        // single fit.
        let (x, y, _) = crate::elastic_net_cv::tests::cv_data();
        let f = LassoCrossValidated::new().fit(&x, &y).unwrap();
        let opts = anofox_regression::solvers::path::PathOptions::elastic_net(1.0);
        let mut opts = opts;
        opts.tol = 1e-12;
        let path = anofox_regression::solvers::path::fit_path(
            &crate::convert::ndarray_to_mat(x.view()),
            &crate::convert::ndarray_to_col(y.view()),
            &f.alphas().to_vec(),
            &opts,
        )
        .unwrap();
        let l = f
            .alphas()
            .iter()
            .position(|&a| a == f.best_alpha())
            .unwrap();
        for j in 0..x.ncols() {
            assert!((f.coefficients()[j] - path.coefficients[(j, l)]).abs() < 1e-12);
        }
    }
}
