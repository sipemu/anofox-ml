//! Elastic Net regression with built-in cross-validation (sklearn
//! `ElasticNetCV`), plus the path/CV machinery shared with `LassoCV`.
//!
//! # Objective and scale
//!
//! `alpha` and `l1_ratio` have sklearn's meaning:
//!
//! ```text
//! 1/(2n)·||y − Xw − b||² + alpha·l1_ratio·||w||₁ + alpha·(1 − l1_ratio)/2·||w||²
//! ```
//!
//! # Algorithm
//!
//! For every `l1_ratio` the whole alpha grid is solved by coordinate descent
//! with warm starts (`anofox_regression::solvers::path::fit_path`), once per
//! training fold and once on the full data. The default grid is sklearn's:
//! `n_alphas = 100` values log-spaced from `alpha_max = max|Xcᵀyc| /
//! (n·l1_ratio)` down to `eps·alpha_max` (`eps = 1e-3`), from upstream
//! `lambda_grid`. Folds are contiguous (sklearn `KFold`), seeded shuffled (`kfold_ids`) or
//! user-given fold ids. The CV curve is the pooled MSE (all held-out squared
//! errors / n); `alpha_1se` is the largest alpha within one standard error
//! (glmnet's `cvsd`) of the minimum. The returned coefficients are the
//! full-data path solution at the selected alpha.
//!
//! Upstream's path solves glmnet's objective, which rescales `y` to unit
//! variance internally; that leaves the lasso part unchanged but divides the
//! ridge part by `sd(y)`. Each path (every fold has its own `sd(y)`) is
//! therefore called with `λ = alpha·(ρ + (1 − ρ)·sd(y))` and glmnet mixing
//! `ρ / (ρ + (1 − ρ)·sd(y))`, which is exactly sklearn's objective.
//! (`cv_path` uses one mixing for all folds, so it is not used here.)

use crate::convert::check_xy;
use anofox_ml_core::{Fit, Predict, Result, RustMlError};
use anofox_regression::solvers::path::{fit_path, kfold_ids, lambda_grid, PathOptions, RegPath};
use faer::{Col, Mat};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2};

// ---------------------------------------------------------------------------
// Shared CV machinery
// ---------------------------------------------------------------------------

/// How to split the rows into folds.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct FoldSpec {
    pub n_folds: usize,
    pub fold_ids: Option<Vec<usize>>,
    pub seed: u64,
    /// Shuffle the rows (seeded) before cutting them into folds. Default
    /// false: contiguous folds, as sklearn's `cv=int` (`KFold` without
    /// shuffling).
    #[serde(default)]
    pub shuffle: bool,
}

/// sklearn `KFold(k)` without shuffling: contiguous blocks, the first
/// `n % k` of size `n / k + 1`.
pub(crate) fn contiguous_fold_ids(n: usize, k: usize) -> Vec<usize> {
    let (base, extra) = (n / k, n % k);
    let mut ids = Vec::with_capacity(n);
    for f in 0..k {
        let size = base + usize::from(f < extra);
        ids.extend(std::iter::repeat_n(f, size));
    }
    ids
}

impl FoldSpec {
    pub(crate) fn new(n_folds: usize) -> Self {
        Self {
            n_folds,
            fold_ids: None,
            seed: 0,
            shuffle: false,
        }
    }

    /// Held-out row sets, one per fold.
    pub(crate) fn test_sets(&self, n: usize) -> Result<Vec<Vec<usize>>> {
        let ids = match &self.fold_ids {
            Some(ids) => {
                if ids.len() != n {
                    return Err(RustMlError::ShapeMismatch(format!(
                        "fold_ids has {} entries but X has {n} rows",
                        ids.len()
                    )));
                }
                ids.clone()
            }
            None => {
                if self.n_folds < 2 {
                    return Err(RustMlError::InvalidParameter(
                        "cv_folds must be at least 2".into(),
                    ));
                }
                if self.n_folds > n {
                    return Err(RustMlError::InvalidParameter(format!(
                        "cv_folds={} > n_samples={n}",
                        self.n_folds
                    )));
                }
                if self.shuffle {
                    kfold_ids(n, self.n_folds, self.seed)
                } else {
                    contiguous_fold_ids(n, self.n_folds)
                }
            }
        };
        let mut labels: Vec<usize> = ids.clone();
        labels.sort_unstable();
        labels.dedup();
        if labels.len() < 2 {
            return Err(RustMlError::InvalidParameter(
                "cross-validation needs at least 2 folds".into(),
            ));
        }
        Ok(labels
            .iter()
            .map(|&l| (0..n).filter(|&i| ids[i] == l).collect())
            .collect())
    }
}

fn rows_mat(x: ArrayView2<f64>, rows: &[usize]) -> Mat<f64> {
    Mat::from_fn(rows.len(), x.ncols(), |i, j| x[[rows[i], j]])
}

fn rows_col(y: ArrayView1<f64>, rows: &[usize]) -> Col<f64> {
    Col::from_fn(rows.len(), |i| y[rows[i]])
}

/// Complement of `test` in `0..n`.
pub(crate) fn train_rows(n: usize, test: &[usize]) -> Vec<usize> {
    let mut mask = vec![true; n];
    for &i in test {
        mask[i] = false;
    }
    (0..n).filter(|&i| mask[i]).collect()
}

/// Options of the coordinate-descent path.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CdOptions {
    pub fit_intercept: bool,
    pub tol: f64,
    pub max_iter: usize,
}

/// Elastic-net path in sklearn's `(alpha, l1_ratio)` parametrisation (see
/// the module docs for the mapping onto upstream's glmnet objective).
fn sk_path(x: &Mat<f64>, y: &Col<f64>, alphas: &[f64], rho: f64, cd: CdOptions) -> Result<RegPath> {
    let n = y.nrows() as f64;
    let y_mean = if cd.fit_intercept {
        y.iter().sum::<f64>() / n
    } else {
        0.0
    };
    let ss = y.iter().map(|v| (v - y_mean).powi(2)).sum::<f64>() / n;
    let sd = if ss > 0.0 { ss.sqrt() } else { 1.0 };
    let scale = rho + (1.0 - rho) * sd;
    let mut opts = PathOptions::elastic_net(rho / scale).with_intercept(cd.fit_intercept);
    opts.tol = cd.tol;
    opts.max_iter = cd.max_iter;
    let lambdas: Vec<f64> = alphas.iter().map(|a| a * scale).collect();
    fit_path(x, y, &lambdas, &opts).map_err(|e| RustMlError::InvalidParameter(e.to_string()))
}

/// sklearn's default alpha grid for mixing `rho`.
pub(crate) fn default_alphas(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    rho: f64,
    n_alphas: usize,
    eps: f64,
    fit_intercept: bool,
) -> Result<Vec<f64>> {
    if rho < 1e-3 {
        return Err(RustMlError::InvalidParameter(format!(
            "automatic alpha grid needs l1_ratio >= 1e-3 (got {rho}); pass explicit alphas"
        )));
    }
    let opts = PathOptions::elastic_net(rho).with_intercept(fit_intercept);
    lambda_grid(
        &crate::convert::ndarray_to_mat(x),
        &crate::convert::ndarray_to_col(y),
        &opts,
        n_alphas,
        Some(eps),
    )
    .map_err(|e| RustMlError::InvalidParameter(e.to_string()))
}

/// Validated, de-duplicated alphas in decreasing order (as sklearn sorts
/// them).
pub(crate) fn sorted_alphas(alphas: &[f64]) -> Result<Vec<f64>> {
    if alphas.is_empty() {
        return Err(RustMlError::InvalidParameter(
            "alphas must not be empty".into(),
        ));
    }
    if alphas.iter().any(|a| !(a.is_finite() && *a >= 0.0)) {
        return Err(RustMlError::InvalidParameter(
            "all alphas must be finite and non-negative".into(),
        ));
    }
    let mut a = alphas.to_vec();
    a.sort_by(|u, v| v.total_cmp(u));
    a.dedup();
    Ok(a)
}

/// CV result over one alpha grid.
#[derive(Debug, Clone)]
pub(crate) struct PathCv {
    pub alphas: Array1<f64>,
    /// Held-out MSE per alpha and fold (`n_alphas × n_folds`).
    pub mse_path: Array2<f64>,
    /// Pooled held-out MSE per alpha.
    pub cv_mean: Array1<f64>,
    /// glmnet standard error of `cv_mean`.
    pub cv_se: Array1<f64>,
    pub best: usize,
    pub best_1se: usize,
    pub coef: Array1<f64>,
    pub intercept: f64,
}

pub(crate) fn path_cv(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    rho: f64,
    alphas: &[f64],
    test_sets: &[Vec<usize>],
    cd: CdOptions,
) -> Result<PathCv> {
    let n = x.nrows();
    let na = alphas.len();
    let k = test_sets.len();
    let mut mse_path = Array2::<f64>::zeros((na, k));
    let mut sse = vec![0.0; na];
    let mut sizes = vec![0usize; k];
    for (f, test) in test_sets.iter().enumerate() {
        let train = train_rows(n, test);
        let path = sk_path(&rows_mat(x, &train), &rows_col(y, &train), alphas, rho, cd)?;
        let pred = path.predict(&rows_mat(x, test));
        sizes[f] = test.len();
        for l in 0..na {
            let s: f64 = test
                .iter()
                .enumerate()
                .map(|(i, &r)| (y[r] - pred[(i, l)]).powi(2))
                .sum();
            sse[l] += s;
            mse_path[[l, f]] = s / test.len() as f64;
        }
    }
    let total: usize = sizes.iter().sum();
    let cv_mean = Array1::from_shape_fn(na, |l| sse[l] / total as f64);
    let cv_se = Array1::from_shape_fn(na, |l| {
        let v: f64 = (0..k)
            .map(|f| sizes[f] as f64 * (mse_path[[l, f]] - cv_mean[l]).powi(2))
            .sum::<f64>()
            / total as f64;
        (v / (k as f64 - 1.0)).sqrt()
    });
    // First minimum in decreasing-alpha order (largest alpha on ties).
    let best = (0..na).fold(0, |b, l| if cv_mean[l] < cv_mean[b] { l } else { b });
    let best_1se = (0..na)
        .find(|&l| cv_mean[l] <= cv_mean[best] + cv_se[best])
        .unwrap_or(best);

    let full = sk_path(
        &crate::convert::ndarray_to_mat(x),
        &crate::convert::ndarray_to_col(y),
        alphas,
        rho,
        cd,
    )?;
    Ok(PathCv {
        alphas: Array1::from(alphas.to_vec()),
        mse_path,
        cv_mean,
        cv_se,
        best,
        best_1se,
        coef: Array1::from_shape_fn(x.ncols(), |j| full.coefficients[(j, best)]),
        intercept: full.intercepts[best],
    })
}

pub(crate) fn r_squared(x: ArrayView2<f64>, y: ArrayView1<f64>, coef: &Array1<f64>, b: f64) -> f64 {
    let resid = &y - &(x.dot(coef) + b);
    let mean = y.mean().unwrap_or(0.0);
    let tss: f64 = y.iter().map(|v| (v - mean).powi(2)).sum();
    let rss = resid.dot(&resid);
    if tss > 0.0 {
        1.0 - rss / tss
    } else if rss == 0.0 {
        1.0
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// ElasticNetCV
// ---------------------------------------------------------------------------

/// Elastic Net regression with cross-validated `alpha` and `l1_ratio`
/// (sklearn `ElasticNetCV`; see the module docs).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ElasticNetCrossValidated {
    alphas: Option<Vec<f64>>,
    l1_ratios: Vec<f64>,
    n_alphas: usize,
    eps: f64,
    folds: FoldSpec,
    with_intercept: bool,
    tol: f64,
    max_iter: usize,
}

impl ElasticNetCrossValidated {
    pub fn new() -> Self {
        Self {
            alphas: None,
            l1_ratios: vec![0.1, 0.3, 0.5, 0.7, 0.9],
            n_alphas: 100,
            eps: 1e-3,
            folds: FoldSpec::new(5),
            with_intercept: true,
            tol: 1e-12,
            max_iter: 100_000,
        }
    }

    /// Explicit alpha grid (sklearn scale; sorted decreasingly). Default:
    /// data-driven grid per `l1_ratio`.
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

    /// Set the grid of L1 ratio values to search over.
    /// Each value must be in [0.0, 1.0]. 1.0 = pure Lasso, 0.0 = pure Ridge.
    pub fn with_l1_ratios(mut self, l1_ratios: Vec<f64>) -> Self {
        self.l1_ratios = l1_ratios;
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

    /// Coordinate-descent tolerance on the largest scaled coefficient
    /// change (default `1e-12`).
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

impl Default for ElasticNetCrossValidated {
    fn default() -> Self {
        Self::new()
    }
}

/// A fitted Elastic Net regression model selected via cross-validation.
#[derive(Debug, Clone)]
pub struct FittedElasticNetCrossValidated {
    coef: Array1<f64>,
    intercept: Option<f64>,
    r_squared: f64,
    l1_ratios: Vec<f64>,
    best_l1: usize,
    alphas: Array2<f64>,
    mse_path: Array3<f64>,
    cv: PathCv,
}

impl FittedElasticNetCrossValidated {
    /// Selected alpha (sklearn `alpha_`).
    pub fn best_alpha(&self) -> f64 {
        self.cv.alphas[self.cv.best]
    }

    /// Selected L1 ratio (sklearn `l1_ratio_`).
    pub fn best_l1_ratio(&self) -> f64 {
        self.l1_ratios[self.best_l1]
    }

    /// Largest alpha (at the selected `l1_ratio`) whose CV error is within
    /// one standard error of the minimum.
    pub fn alpha_1se(&self) -> f64 {
        self.cv.alphas[self.cv.best_1se]
    }

    /// Alpha grid per `l1_ratio`, `n_l1_ratio × n_alphas` (sklearn `alphas_`).
    pub fn alphas(&self) -> &Array2<f64> {
        &self.alphas
    }

    /// Held-out MSE, `n_l1_ratio × n_alphas × n_folds` (sklearn `mse_path_`).
    pub fn mse_path(&self) -> &Array3<f64> {
        &self.mse_path
    }

    /// Pooled CV MSE over the alpha grid of the selected `l1_ratio`.
    pub fn cv_mse(&self) -> &Array1<f64> {
        &self.cv.cv_mean
    }

    /// Standard error of `cv_mse`.
    pub fn cv_mse_se(&self) -> &Array1<f64> {
        &self.cv.cv_se
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

impl Fit<f64> for ElasticNetCrossValidated {
    type Fitted = FittedElasticNetCrossValidated;

    fn fit_view(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<Self::Fitted> {
        check_xy(x, y)?;
        if self.l1_ratios.is_empty() {
            return Err(RustMlError::InvalidParameter(
                "l1_ratios must not be empty".into(),
            ));
        }
        if self.l1_ratios.iter().any(|r| !(0.0..=1.0).contains(r)) {
            return Err(RustMlError::InvalidParameter(
                "all l1_ratios must be between 0.0 and 1.0".into(),
            ));
        }
        if self.n_alphas == 0 || !(self.eps > 0.0 && self.eps < 1.0) {
            return Err(RustMlError::InvalidParameter(
                "n_alphas must be >= 1 and eps in (0, 1)".into(),
            ));
        }
        let explicit = self.alphas.as_deref().map(sorted_alphas).transpose()?;
        let test_sets = self.folds.test_sets(x.nrows())?;
        let cd = CdOptions {
            fit_intercept: self.with_intercept,
            tol: self.tol,
            max_iter: self.max_iter,
        };

        let mut runs = Vec::with_capacity(self.l1_ratios.len());
        for &rho in &self.l1_ratios {
            let alphas = match &explicit {
                Some(a) => a.clone(),
                None => default_alphas(x, y, rho, self.n_alphas, self.eps, self.with_intercept)?,
            };
            runs.push(path_cv(x, y, rho, &alphas, &test_sets, cd)?);
        }
        // First minimum over (l1_ratio, alpha), as sklearn's argmin.
        let best_l1 = (0..runs.len()).fold(0, |b, r| {
            if runs[r].cv_mean[runs[r].best] < runs[b].cv_mean[runs[b].best] {
                r
            } else {
                b
            }
        });
        let na = runs[0].alphas.len();
        let k = test_sets.len();
        let alphas = Array2::from_shape_fn((runs.len(), na), |(r, l)| runs[r].alphas[l]);
        let mse_path =
            Array3::from_shape_fn((runs.len(), na, k), |(r, l, f)| runs[r].mse_path[[l, f]]);
        let cv = runs.swap_remove(best_l1);
        let r_squared = r_squared(x, y, &cv.coef, cv.intercept);
        Ok(FittedElasticNetCrossValidated {
            coef: cv.coef.clone(),
            intercept: self.with_intercept.then_some(cv.intercept),
            r_squared,
            l1_ratios: self.l1_ratios.clone(),
            best_l1,
            alphas,
            mse_path,
            cv,
        })
    }
}

impl Predict<f64> for FittedElasticNetCrossValidated {
    fn predict_view(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        crate::convert::linear_predict(x, &self.coef, self.intercept.unwrap_or(0.0))
    }
}

impl anofox_ml_core::RegressorScore<f64> for FittedElasticNetCrossValidated {}

#[cfg(test)]
pub(crate) mod tests {
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

    /// `data.make(40, 5, coefs=[1.5, -2, 0, 0.7, 0], noise=1)` and the fold
    /// ids `(3i) mod 5` of the sklearn reference script (equal fold sizes,
    /// so sklearn's mean-over-folds equals the pooled MSE).
    pub(crate) fn cv_data() -> (Array2<f64>, Array1<f64>, Vec<usize>) {
        let coefs = [1.5, -2.0, 0.0, 0.7, 0.0];
        let x = Array2::from_shape_fn((40, 5), |(i, j)| {
            ((i * (j + 3) * 7 + j * 5 + (i * i) % 11) % 17) as f64 / 4.0 - 2.0
        });
        let y = Array1::from_shape_fn(40, |i| {
            (0..5).map(|j| x[[i, j]] * coefs[j]).sum::<f64>() + 1.0 + (((i * 13) % 7) as f64 - 3.0)
        });
        let ids = (0..40).map(|i| (i * 3) % 5).collect();
        (x, y, ids)
    }

    fn close(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (i, (u, v)) in a.iter().zip(b).enumerate() {
            assert!((u - v).abs() <= tol * v.abs().max(1.0), "[{i}] {u} vs {v}");
        }
    }

    #[test]
    fn test_elastic_net_cv_matches_sklearn() {
        // sklearn ElasticNetCV(l1_ratio=[0.2, 0.7], cv=<fold ids>, tol=1e-12).
        let (x, y, ids) = cv_data();
        let f = ElasticNetCrossValidated::new()
            .with_l1_ratios(vec![0.2, 0.7])
            .with_fold_ids(ids)
            .fit(&x, &y)
            .unwrap();
        assert_eq!(f.best_l1_ratio(), 0.7);
        assert!((f.best_alpha() - 0.33584260196683474).abs() < 1e-10);
        assert_eq!(f.alphas().dim(), (2, 100));
        close(
            &[f.alphas()[[0, 0]], f.alphas()[[1, 0]]],
            &[17.86578125, 5.104508928571429],
            1e-12,
        );
        close(
            f.coefficients().as_slice().unwrap(),
            &[
                0.9081441484284528,
                -2.0411952514992846,
                0.0,
                0.123115731018162,
                0.0,
            ],
            1e-7,
        );
        assert!((f.intercept().unwrap() - 0.984725178386316).abs() < 1e-7);
        let mp = f.mse_path();
        close(
            &mp.slice(ndarray::s![0, 10, ..]).to_vec(),
            &[
                11.135180751863206,
                7.399933405146925,
                12.430936030348608,
                14.496792707637363,
                17.212403371321336,
            ],
            1e-8,
        );
        close(
            &mp.slice(ndarray::s![1, 60, ..]).to_vec(),
            &[
                5.985466194745058,
                5.725455483665522,
                4.452583274102809,
                4.619601553548464,
                4.712862954083031,
            ],
            1e-6,
        );
        assert!(f.alpha_1se() >= f.best_alpha());
    }

    #[test]
    fn test_elastic_net_path_matches_sklearn_single_fit() {
        // Mixed penalty (ridge part scale-sensitive): the path must solve
        // sklearn's objective, ElasticNet(alpha=0.3, l1_ratio=0.2).
        let (x, y, ids) = cv_data();
        let f = ElasticNetCrossValidated::new()
            .with_alphas(vec![0.3])
            .with_l1_ratios(vec![0.2])
            .with_fold_ids(ids)
            .fit(&x, &y)
            .unwrap();
        close(
            f.coefficients().as_slice().unwrap(),
            &[
                0.9536868072722745,
                -1.9868794396215301,
                0.0,
                0.22737328979390473,
                0.014974248296560955,
            ],
            1e-8,
        );
        assert!((f.intercept().unwrap() - 1.0067351832889115).abs() < 1e-8);
    }

    #[test]
    fn test_elastic_net_cv_fold_ids_validation() {
        let (x, y, _) = cv_data();
        assert!(ElasticNetCrossValidated::new()
            .with_fold_ids(vec![0; 40])
            .fit(&x, &y)
            .is_err());
        assert!(ElasticNetCrossValidated::new()
            .with_fold_ids(vec![0, 1])
            .fit(&x, &y)
            .is_err());
        // Pure ridge has no automatic grid (as sklearn).
        assert!(ElasticNetCrossValidated::new()
            .with_l1_ratios(vec![0.0])
            .fit(&x, &y)
            .is_err());
    }

    #[test]
    fn test_elastic_net_cv_seeded_folds_are_reproducible() {
        let (x, y, _) = cv_data();
        let a = ElasticNetCrossValidated::new()
            .with_random_state(7)
            .fit(&x, &y)
            .unwrap();
        let b = ElasticNetCrossValidated::new()
            .with_random_state(7)
            .fit(&x, &y)
            .unwrap();
        assert_eq!(a.best_alpha(), b.best_alpha());
        assert_eq!(a.mse_path(), b.mse_path());
    }
}
