//! Least Angle Regression (LARS), LassoLars and LassoLarsIC.
//!
//! Mirrors `sklearn.linear_model.{Lars, LassoLars, LassoLarsIC}`.
//!
//! - LARS and the Lasso path are delegated to
//!   `anofox_regression::solvers::LarsRegressor` (`LarsMethod::Lar` /
//!   `LarsMethod::Lasso`), converting at the ndarray boundary. The Lasso
//!   path drops a feature whose coefficient crosses zero, as sklearn's
//!   `lars_path(method="lasso")` does (anofox-regression 0.10).
//! - sklearn's `alpha` / `alphas_` are the maximum absolute correlation
//!   divided by `n_samples`; upstream reports the raw correlation, so its path
//!   alphas are divided by `n`.
//! - `LassoLarsIC` evaluates the criterion at every knot of **one** Lasso
//!   path (as sklearn does) instead of refitting per step.

use crate::convert::{
    check_xy, coef_from_col, linear_predict, ndarray_to_col, ndarray_to_mat, upstream_err,
};
use anofox_ml_core::{Fit, Predict, Result, RustMlError};
use anofox_regression::solvers::{FittedRegressor as _, LarsMethod, LarsRegressor, Regressor as _};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

/// LARS / LassoLars estimator.
///
/// `Lars::new(k)` is sklearn's `Lars(n_nonzero_coefs=k)`;
/// `Lars::lasso_lars(alpha)` is `LassoLars(alpha=alpha)`.
/// `Lars::lasso(k)` runs the Lasso path until `k` features are active.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Lars {
    /// Stop once this many features are active (`usize::MAX` = no limit).
    pub n_nonzero_coefs: usize,
    pub fit_intercept: bool,
    /// Use the Lasso modification (features leave the active set when their
    /// coefficient crosses zero).
    pub lasso: bool,
    /// LassoLars regularisation strength (sklearn scale: correlation / n).
    /// The path stops at this alpha and is interpolated to it. Ignored by
    /// plain LARS. `0.0` runs the full path.
    pub alpha: f64,
    /// Maximum number of path steps (sklearn `max_iter`, default 500).
    pub max_iter: usize,
}

impl Lars {
    /// Plain LARS with at most `n_nonzero_coefs` active features.
    pub fn new(n_nonzero_coefs: usize) -> Self {
        Self {
            n_nonzero_coefs,
            fit_intercept: true,
            lasso: false,
            alpha: 0.0,
            max_iter: 500,
        }
    }

    /// Lasso path, stopped once `n_nonzero_coefs` features are active.
    pub fn lasso(n_nonzero_coefs: usize) -> Self {
        Self {
            lasso: true,
            ..Self::new(n_nonzero_coefs)
        }
    }

    /// sklearn's `LassoLars(alpha)`: Lasso path solved exactly at `alpha`.
    pub fn lasso_lars(alpha: f64) -> Self {
        Self {
            alpha,
            ..Self::lasso(usize::MAX)
        }
    }

    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn with_n_nonzero_coefs(mut self, k: usize) -> Self {
        self.n_nonzero_coefs = k;
        self
    }

    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    fn path(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<LarsPath> {
        check_xy(x, y)?;
        if self.n_nonzero_coefs == 0 {
            return Err(RustMlError::InvalidParameter("need at least 1 coef".into()));
        }
        if !(self.alpha >= 0.0 && self.alpha.is_finite()) {
            return Err(RustMlError::InvalidParameter(
                "alpha must be finite and >= 0".into(),
            ));
        }
        self.path_upstream(x, y)
    }

    fn path_upstream(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<LarsPath> {
        let n = x.nrows() as f64;
        let method = if self.lasso {
            LarsMethod::Lasso
        } else {
            LarsMethod::Lar
        };
        let mut b = LarsRegressor::builder()
            .method(method)
            .with_intercept(self.fit_intercept);
        if self.lasso && self.alpha > 0.0 {
            // sklearn's alpha is the max correlation / n; upstream's is raw.
            b = b.alpha(self.alpha * n);
        }
        if self.n_nonzero_coefs != usize::MAX {
            b = b.n_nonzero_coefs(self.n_nonzero_coefs.min(self.max_iter));
        }
        let inner = b
            .build()
            .fit(&ndarray_to_mat(x), &ndarray_to_col(y))
            .map_err(upstream_err)?;
        let path = inner.path();
        let coef_path = Array2::from_shape_fn(
            (path.coefficients.nrows(), path.coefficients.ncols()),
            |(j, l)| {
                let v = path.coefficients[(j, l)];
                if v.is_finite() {
                    v
                } else {
                    0.0
                }
            },
        );
        let res = inner.result();
        let coef = coef_from_col(&res.coefficients);
        // sklearn's `active_`: the active set at the end, in entry order.
        let active = inner.active().to_vec();
        let mut alphas = Array1::from_iter(path.lambdas.iter().map(|a| a / n));
        // sklearn's knot alpha is the largest correlation of an inactive
        // feature: exactly 0 once every feature is active (upstream reports
        // the rounding residue there).
        if active.len() == x.ncols() {
            if let Some(last) = alphas.last_mut() {
                *last = 0.0;
            }
        }
        Ok(LarsPath {
            alphas,
            coef_path,
            intercepts: path.intercepts,
            intercept: res.intercept.unwrap_or(0.0),
            coef,
            active,
        })
    }
}

/// A LARS path on sklearn's scale.
struct LarsPath {
    /// Knots (max correlation / n), decreasing.
    alphas: Array1<f64>,
    /// `n_features × n_knots`.
    coef_path: Array2<f64>,
    intercepts: Vec<f64>,
    coef: Array1<f64>,
    intercept: f64,
    /// Active set at the end, in entry order.
    active: Vec<usize>,
}

/// A fitted LARS / LassoLars model.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedLars {
    pub coef: Array1<f64>,
    pub intercept: f64,
    /// Active features at the end of the path, in the order they entered
    /// (sklearn `active_`).
    pub active_set: Vec<usize>,
    /// Path knots on sklearn's scale (max correlation / n), decreasing
    /// (sklearn `alphas_`).
    #[serde(default)]
    pub alphas: Array1<f64>,
    /// Coefficients at every knot, `n_features × n_knots` (sklearn
    /// `coef_path_`).
    #[serde(default)]
    pub coef_path: Array2<f64>,
    /// Number of path steps taken (sklearn `n_iter_`).
    #[serde(default)]
    pub n_iter: usize,
    n_features: usize,
}

impl Fit<f64> for Lars {
    type Fitted = FittedLars;

    fn fit_view(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<Self::Fitted> {
        let path = self.path(x, y)?;
        Ok(FittedLars {
            n_iter: path.alphas.len().saturating_sub(1),
            coef: path.coef,
            intercept: path.intercept,
            active_set: path.active,
            alphas: path.alphas,
            coef_path: path.coef_path,
            n_features: x.ncols(),
        })
    }
}

impl Predict<f64> for FittedLars {
    fn predict_view(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        linear_predict(x, &self.coef, self.intercept)
    }
}

// ---------------------------------------------------------------------------
// LassoLarsIC — information criterion along one Lasso path.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum IcCriterion {
    Aic,
    Bic,
}

/// sklearn's `LassoLarsIC`.
///
/// The criterion (sklearn ≥ 1.1) is evaluated at every knot of a single
/// Lasso path:
/// `n·ln(2πσ²) + RSS/σ² + K·df`, `K = 2` (AIC) or `ln n` (BIC), `df` the
/// number of non-zero coefficients. `σ²` is `noise_variance` if given,
/// otherwise the OLS residual variance `RSS_ols / (n − p − fit_intercept)`
/// (requires `n > p + fit_intercept`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LassoLarsIC {
    pub criterion: IcCriterion,
    /// Cap on the number of active features along the path.
    pub max_features: Option<usize>,
    pub fit_intercept: bool,
    /// Known noise variance `σ²`; estimated from OLS when `None`.
    pub noise_variance: Option<f64>,
}

impl LassoLarsIC {
    pub fn new(criterion: IcCriterion) -> Self {
        Self {
            criterion,
            max_features: None,
            fit_intercept: true,
            noise_variance: None,
        }
    }

    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    pub fn with_noise_variance(mut self, noise_variance: f64) -> Self {
        self.noise_variance = Some(noise_variance);
        self
    }

    pub fn with_max_features(mut self, max_features: usize) -> Self {
        self.max_features = Some(max_features);
        self
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedLassoLarsIC {
    /// The path solution at the selected knot (`fitted.alphas` /
    /// `fitted.coef_path` hold the whole path).
    pub fitted: FittedLars,
    /// Criterion at the selected knot.
    pub criterion_value: f64,
    pub n_nonzero_coefs: usize,
    /// Selected alpha (sklearn `alpha_`).
    #[serde(default)]
    pub alpha: f64,
    /// Criterion at every knot of the path (sklearn `criterion_`).
    #[serde(default)]
    pub criterion_path: Array1<f64>,
    /// Noise variance used in the criterion (sklearn `noise_variance_`).
    #[serde(default)]
    pub noise_variance: f64,
}

/// Residual variance of the least-squares fit (sklearn
/// `LassoLarsIC._estimate_noise_variance`).
fn ols_noise_variance(x: ArrayView2<f64>, y: ArrayView1<f64>, fit_intercept: bool) -> Result<f64> {
    let (n, p) = x.dim();
    let dof = n as isize - p as isize - isize::from(fit_intercept);
    if dof <= 0 {
        return Err(RustMlError::InvalidParameter(format!(
            "noise variance estimation needs n_samples > n_features + fit_intercept \
             ({n} <= {}); pass `with_noise_variance`",
            p + usize::from(fit_intercept)
        )));
    }
    let ols = anofox_regression::solvers::OlsRegressor::builder()
        .with_intercept(fit_intercept)
        .build()
        .fit(&ndarray_to_mat(x), &ndarray_to_col(y))
        .map_err(upstream_err)?;
    let rss: f64 = ols.result().residuals.iter().map(|r| r * r).sum();
    Ok(rss / dof as f64)
}

impl Fit<f64> for LassoLarsIC {
    type Fitted = FittedLassoLarsIC;

    fn fit_view(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<Self::Fitted> {
        let n = x.nrows();
        let path = Lars::lasso(self.max_features.unwrap_or(usize::MAX))
            .with_fit_intercept(self.fit_intercept)
            .path(x, y)?;
        let sigma2 = match self.noise_variance {
            Some(v) if v > 0.0 && v.is_finite() => v,
            Some(v) => {
                return Err(RustMlError::InvalidParameter(format!(
                    "noise_variance must be finite and > 0, got {v}"
                )))
            }
            None => ols_noise_variance(x, y, self.fit_intercept)?,
        };
        let k = match self.criterion {
            IcCriterion::Aic => 2.0,
            IcCriterion::Bic => (n as f64).ln(),
        };
        let steps = path.coef_path.ncols();
        let mut crit = Array1::<f64>::zeros(steps);
        let mut dfs = vec![0usize; steps];
        for l in 0..steps {
            let c = path.coef_path.column(l);
            let resid = &y - &x.dot(&c) - path.intercepts[l];
            let rss = resid.dot(&resid);
            dfs[l] = c.iter().filter(|v| v.abs() > f64::EPSILON).count();
            crit[l] = n as f64 * (2.0 * std::f64::consts::PI * sigma2).ln()
                + rss / sigma2
                + k * dfs[l] as f64;
        }
        // First minimum, as `np.argmin`.
        let best = (0..steps).fold(0, |b, l| if crit[l] < crit[b] { l } else { b });
        let coef = path.coef_path.column(best).to_owned();
        // Entry order of the features active at the selected knot.
        let mut active: Vec<(usize, usize)> = (0..coef.len())
            .filter(|&j| coef[j] != 0.0)
            .map(|j| {
                let mut l = best;
                while l > 0 && path.coef_path[[j, l - 1]] != 0.0 {
                    l -= 1;
                }
                (l, j)
            })
            .collect();
        active.sort_unstable();
        let fitted = FittedLars {
            intercept: path.intercepts[best],
            active_set: active.into_iter().map(|(_, j)| j).collect(),
            n_iter: steps.saturating_sub(1),
            coef,
            alphas: path.alphas.clone(),
            coef_path: path.coef_path,
            n_features: x.ncols(),
        };
        Ok(FittedLassoLarsIC {
            fitted,
            criterion_value: crit[best],
            n_nonzero_coefs: dfs[best],
            alpha: path.alphas[best],
            criterion_path: crit,
            noise_variance: sigma2,
        })
    }
}

impl Predict<f64> for FittedLassoLarsIC {
    fn predict_view(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        self.fitted.predict(x)
    }
}

// ---------------------------------------------------------------------------
// LarsCV / LassoLarsCV — alpha chosen by cross-validated residues.
// ---------------------------------------------------------------------------

/// sklearn's `LarsCV` (`lasso = false`) / `LassoLarsCV` (`lasso = true`).
///
/// Each of `cv` contiguous folds (sklearn's `KFold`) runs a full path on
/// the training rows centred by their means (the held-out rows by the same
/// means; sklearn's `_lars_path_residues`); its held-out residues are linearly
/// interpolated (`scipy.interpolate.interp1d`) onto the union of all
/// folds' knots (thinned to at most `max_n_alphas`). The alpha with the
/// lowest mean squared residue, averaged over folds, is refitted on all
/// rows: the path stopped at that alpha, as `lars_path(alpha_min=)`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct LarsCV {
    pub lasso: bool,
    pub fit_intercept: bool,
    pub max_iter: usize,
    pub cv: usize,
    pub max_n_alphas: usize,
}

impl Default for LarsCV {
    fn default() -> Self {
        Self {
            lasso: false,
            fit_intercept: true,
            max_iter: 500,
            cv: 5,
            max_n_alphas: 1000,
        }
    }
}

impl LarsCV {
    /// sklearn's `LassoLarsCV()`.
    pub fn lasso() -> Self {
        Self {
            lasso: true,
            ..Self::default()
        }
    }

    fn lars(&self, fit_intercept: bool) -> Lars {
        let base = if self.lasso {
            Lars::lasso(usize::MAX)
        } else {
            Lars::new(usize::MAX)
        };
        base.with_fit_intercept(fit_intercept)
            .with_max_iter(self.max_iter)
    }
}

/// `scipy.interpolate.interp1d(x, y, axis=0)(x_new)` (linear, `x`
/// increasing, `x_new` inside `[x₀, x_last]`): row `i` of the result
/// interpolates the rows of `y`.
fn interp1d_rows(x: &[f64], y: &Array2<f64>, x_new: &[f64]) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((x_new.len(), y.ncols()));
    for (i, &v) in x_new.iter().enumerate() {
        let hi = x.partition_point(|&a| a < v).clamp(1, x.len() - 1);
        let lo = hi - 1;
        let dx = x[hi] - x[lo];
        for c in 0..y.ncols() {
            let slope = (y[[hi, c]] - y[[lo, c]]) / dx;
            out[[i, c]] = slope * (v - x[lo]) + y[[lo, c]];
        }
    }
    out
}

impl Fit<f64> for LarsCV {
    type Fitted = FittedLarsCV;

    fn fit_view(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<Self::Fitted> {
        check_xy(x, y)?;
        let n = x.nrows();
        if self.cv < 2 || self.cv > n {
            return Err(RustMlError::InvalidParameter(format!(
                "cv must be in 2..={n}, got {}",
                self.cv
            )));
        }
        if self.max_n_alphas == 0 {
            return Err(RustMlError::InvalidParameter(
                "max_n_alphas must be at least 1".into(),
            ));
        }
        let ids = crate::elastic_net_cv::contiguous_fold_ids(n, self.cv);
        // Per fold: knots (decreasing) and held-out residues per knot.
        let mut paths: Vec<(Vec<f64>, Array2<f64>)> = Vec::with_capacity(self.cv);
        for f in 0..self.cv {
            let test: Vec<usize> = (0..n).filter(|&i| ids[i] == f).collect();
            let train = crate::elastic_net_cv::train_rows(n, &test);
            let (mut xt, mut yt) = (x.select(Axis(0), &train), y.select(Axis(0), &train));
            let (mut xs, mut ys) = (x.select(Axis(0), &test), y.select(Axis(0), &test));
            if self.fit_intercept {
                let xm = xt.mean_axis(Axis(0)).unwrap();
                xt -= &xm;
                xs -= &xm;
                let ym = yt.mean().unwrap_or(0.0);
                yt -= ym;
                ys -= ym;
            }
            let path = self.lars(false).path(xt.view(), yt.view())?;
            // knots × test rows
            let resid = xs.dot(&path.coef_path).t().to_owned() - &ys.insert_axis(Axis(0));
            paths.push((path.alphas.to_vec(), resid));
        }
        let mut all: Vec<f64> = paths.iter().flat_map(|(a, _)| a.iter().copied()).collect();
        all.sort_by(f64::total_cmp);
        all.dedup();
        let stride = (all.len() / self.max_n_alphas).max(1);
        let mut all: Vec<f64> = all.into_iter().step_by(stride).collect();
        let top = *all.last().unwrap();
        let mut mse = Array2::<f64>::zeros((all.len(), self.cv));
        for (f, (alphas, resid)) in paths.iter().enumerate() {
            let mut a: Vec<f64> = alphas.iter().rev().copied().collect();
            let mut r: Vec<Array1<f64>> = resid
                .rows()
                .into_iter()
                .rev()
                .map(|v| v.to_owned())
                .collect();
            if a[0] != 0.0 {
                a.insert(0, 0.0);
                r.insert(0, r[0].clone());
            }
            if *a.last().unwrap() != top {
                a.push(top);
                r.push(r.last().unwrap().clone());
            }
            // interp1d sorts its points: the appended top alpha lies inside
            // the fold's range when thinning dropped the largest knots.
            let mut order: Vec<usize> = (0..a.len()).collect();
            order.sort_by(|&i, &j| a[i].total_cmp(&a[j]));
            let a: Vec<f64> = order.iter().map(|&i| a[i]).collect();
            let views: Vec<_> = order.iter().map(|&i| r[i].view()).collect();
            let r = ndarray::stack(Axis(0), &views)
                .map_err(|e| RustMlError::InvalidParameter(e.to_string()))?;
            let at = interp1d_rows(&a, &r, &all);
            for i in 0..all.len() {
                mse[[i, f]] = at.row(i).mapv(|v| v * v).mean().unwrap_or(0.0);
            }
        }
        let keep: Vec<usize> = (0..all.len())
            .filter(|&i| mse.row(i).iter().all(|v| v.is_finite()))
            .collect();
        all = keep.iter().map(|&i| all[i]).collect();
        let mse = mse.select(Axis(0), &keep);
        if all.is_empty() {
            return Err(RustMlError::InvalidParameter(
                "no alpha has a finite cross-validated error".into(),
            ));
        }
        let mean: Vec<f64> = mse
            .rows()
            .into_iter()
            .map(|r| r.mean().unwrap_or(0.0))
            .collect();
        let best = (0..mean.len()).fold(0, |b, i| if mean[i] < mean[b] { i } else { b });
        let alpha = all[best];

        // Refit: the full-data path stopped at `alpha` (sklearn's
        // `lars_path(alpha_min=alpha)`, float32-eps equality tolerance).
        let path = self.lars(self.fit_intercept).path(x, y)?;
        let tol = f32::EPSILON as f64;
        let knots = path.alphas.len();
        let stop = (0..knots).find(|&l| path.alphas[l] <= alpha + tol);
        let (last, coef) = match stop {
            None => (knots - 1, path.coef_path.column(knots - 1).to_owned()),
            Some(l) => {
                let c = path.coef_path.column(l).to_owned();
                if (path.alphas[l] - alpha).abs() > tol && l > 0 {
                    let prev = path.coef_path.column(l - 1);
                    let ss = (path.alphas[l - 1] - alpha) / (path.alphas[l - 1] - path.alphas[l]);
                    (l, &prev + &((&c - &prev) * ss))
                } else {
                    (l, c)
                }
            }
        };
        let mut alphas = path.alphas.slice(ndarray::s![..=last]).to_owned();
        let mut coef_path = path.coef_path.slice(ndarray::s![.., ..=last]).to_owned();
        if stop.is_some() {
            alphas[last] = alpha;
            coef_path.column_mut(last).assign(&coef);
        }
        let intercept = if self.fit_intercept {
            y.mean().unwrap_or(0.0) - x.mean_axis(Axis(0)).unwrap().dot(&coef)
        } else {
            0.0
        };
        let active_set = path
            .active
            .iter()
            .copied()
            .filter(|&j| coef[j] != 0.0)
            .collect();
        Ok(FittedLarsCV {
            alpha,
            cv_alphas: Array1::from(all),
            mse_path: mse,
            fitted: FittedLars {
                coef,
                intercept,
                active_set,
                n_iter: last,
                alphas,
                coef_path,
                n_features: x.ncols(),
            },
        })
    }
}

/// A fitted [`LarsCV`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedLarsCV {
    /// The selected alpha (sklearn's `alpha_`).
    pub alpha: f64,
    /// The alphas scored, increasing (sklearn's `cv_alphas_`).
    pub cv_alphas: Array1<f64>,
    /// Mean squared held-out residue per alpha and fold (`mse_path_`).
    pub mse_path: Array2<f64>,
    /// The refit on all rows (path up to `alpha`).
    pub fitted: FittedLars,
}

impl Predict<f64> for FittedLarsCV {
    fn predict_view(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        self.fitted.predict(x)
    }
}

impl anofox_ml_core::RegressorScore<f64> for FittedLarsCV {}
impl anofox_ml_core::RegressorScore<f64> for FittedLars {}
impl anofox_ml_core::RegressorScore<f64> for FittedLassoLarsIC {}

#[cfg(test)]
mod tests {
    use super::*;

    /// 16×4 design (sklearn `RandomState(19)`, rounded) whose Lasso path
    /// drops feature 2 at the 4th knot.
    fn drop_data() -> (Array2<f64>, Array1<f64>) {
        let x = vec![
            0.22, -0.34, -0.15, -0.4, -0.6, -0.24, -0.27, 0.58, 0.25, -1.46, -0.84, 0.27, -0.0,
            -1.02, -0.49, -1.19, 1.56, -0.93, 0.39, -0.15, 0.52, 1.34, 1.17, 1.26, 0.91, -0.84,
            0.18, -1.02, 0.18, -0.42, 0.18, 1.92, -1.03, -0.48, -0.85, 0.06, 0.55, 0.85, 0.52,
            0.37, 0.94, -0.35, 0.36, -0.05, 0.42, -0.7, -0.12, 0.47, 0.04, 0.72, 0.31, -0.05,
            -0.27, 0.31, -0.25, 1.15, -0.22, -1.22, -0.77, 1.55, -0.21, -0.24, -0.24, 0.56,
        ];
        let y = vec![
            0.41, -0.19, -0.73, -1.47, 0.58, 1.9, -0.61, -0.29, -1.87, 2.17, 0.7, -0.11, 0.4, 0.32,
            -1.39, -1.02,
        ];
        (
            Array2::from_shape_vec((16, 4), x).unwrap(),
            Array1::from_vec(y),
        )
    }

    fn close(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (i, (u, v)) in a.iter().zip(b).enumerate() {
            assert!((u - v).abs() < tol, "[{i}] {u} vs {v}");
        }
    }

    #[test]
    fn test_lars_recovers_two_features() {
        let n = 80;
        let mut data = Vec::new();
        for i in 0..n {
            let x0 = (i as f64) - 40.0;
            let x1 = ((i * 7 % 13) as f64) - 6.0;
            let x2 = ((i * 5 % 11) as f64) - 5.0;
            let x3 = ((i * 3 % 7) as f64) - 3.0;
            data.extend([x0, x1, x2, x3]);
        }
        let x = Array2::from_shape_vec((n, 4), data).unwrap();
        let y = x.column(0).mapv(|v| 3.0 * v) + x.column(2).mapv(|v| -2.0 * v);

        let fitted = Lars::new(2).fit(&x, &y).unwrap();
        let mut act = fitted.active_set.clone();
        act.sort();
        assert_eq!(act, vec![0, 2]);
    }

    #[test]
    fn test_lars_two_steps_matches_sklearn() {
        // sklearn: Lars(n_nonzero_coefs=2).fit(X, y)
        let (x, y) = drop_data();
        let f = Lars::new(2).fit(&x, &y).unwrap();
        close(
            f.coef.as_slice().unwrap(),
            &[0.0, 0.386944471149059, 0.24625562875831633, 0.0],
            1e-10,
        );
        assert!((f.intercept - 0.059793977636750706).abs() < 1e-10);
        assert_eq!(f.active_set, vec![1, 2]);
    }

    #[test]
    fn test_lasso_lars_drops_variable_like_sklearn() {
        // The Lasso path must drop feature 2 when its coefficient crosses
        // zero (sklearn lars_path(method="lasso")).
        let (x, y) = drop_data();
        let f = Lars::lasso_lars(0.0).fit(&x, &y).unwrap();
        let sk_alphas = [
            0.6023499999999999,
            0.3846051014575643,
            0.31453164524643185,
            0.248845466100102,
            0.08890635881956642,
            0.011642748360321697,
            0.0,
        ];
        close(f.alphas.as_slice().unwrap(), &sk_alphas, 1e-10);
        // Knot 4: feature 2 has left the active set.
        close(
            f.coef_path.column(4).to_vec().as_slice(),
            &[0.7642621439249324, 0.9312305546790625, 0.0, 0.0],
            1e-10,
        );
        close(
            f.coef.as_slice().unwrap(),
            &[
                1.7440316376641962,
                1.6367333404019406,
                -1.1654478794527279,
                0.2074207656525779,
            ],
            1e-9,
        );
    }

    #[test]
    fn test_lasso_lars_alpha_matches_sklearn() {
        let (x, y) = drop_data();
        for (alpha, coef, intercept) in [
            (
                0.01,
                [
                    1.1261018256099606,
                    1.1203168465327138,
                    -0.16444034808276728,
                    0.147434208454286,
                ],
                -0.01099930098672497,
            ),
            (
                0.003,
                [
                    1.5586526940479257,
                    1.4818083922411727,
                    -0.8651456200417398,
                    0.18942479849309032,
                ],
                -0.037802532434377215,
            ),
        ] {
            let f = Lars::lasso_lars(alpha).fit(&x, &y).unwrap();
            close(f.coef.as_slice().unwrap(), &coef, 1e-9);
            assert!((f.intercept - intercept).abs() < 1e-9);
            // sklearn active_: feature 2 re-entered after being dropped.
            assert_eq!(f.active_set, vec![1, 0, 3, 2]);
        }
    }

    #[test]
    fn test_lasso_lars_ic_matches_sklearn() {
        let (x, y) = drop_data();
        let aic = LassoLarsIC::new(IcCriterion::Aic).fit(&x, &y).unwrap();
        close(
            aic.criterion_path.as_slice().unwrap(),
            &[
                103.1803601494137,
                73.31919901584709,
                60.8123786235322,
                45.53095770563287,
                24.64281026569234,
                22.414004905157224,
                21.92165457437357,
            ],
            1e-8,
        );
        assert!((aic.noise_variance - 0.19103976946584808).abs() < 1e-12);
        assert!(aic.alpha.abs() < 1e-12);

        let bic = LassoLarsIC::new(IcCriterion::Bic).fit(&x, &y).unwrap();
        assert!((bic.alpha - 0.011642748360321697).abs() < 1e-10);
        close(
            bic.fitted.coef.as_slice().unwrap(),
            &[
                1.0245915070553617,
                1.0354826116780387,
                0.0,
                0.13757992660640225,
            ],
            1e-10,
        );
        assert!((bic.fitted.intercept - -0.004709163199303007).abs() < 1e-10);
        assert_eq!(bic.n_nonzero_coefs, 3);
        let mut act = bic.fitted.active_set.clone();
        act.sort();
        assert_eq!(act, vec![0, 1, 3]);
    }

    #[test]
    fn test_lasso_lars_ic_needs_noise_variance_when_p_large() {
        let x = Array2::from_shape_fn((4, 4), |(i, j)| ((i * 3 + j * 5) % 7) as f64);
        let y = Array1::from_vec(vec![1.0, 2.0, 0.5, 3.0]);
        assert!(LassoLarsIC::new(IcCriterion::Aic).fit(&x, &y).is_err());
        assert!(LassoLarsIC::new(IcCriterion::Aic)
            .with_noise_variance(1.0)
            .fit(&x, &y)
            .is_ok());
    }

    #[test]
    fn test_lasso_lars_ic_bic_picks_sparse() {
        // 3 informative features out of 6 — BIC should pick a sparse model.
        let n = 100;
        let mut data = Vec::new();
        for i in 0..n {
            for j in 0..6 {
                data.push(((i * (j + 1) * 7) % 19) as f64 - 9.0);
            }
        }
        let x = Array2::from_shape_vec((n, 6), data).unwrap();
        let y = x.column(0).mapv(|v| 3.0 * v)
            + x.column(2).mapv(|v| -2.0 * v)
            + x.column(4).mapv(|v| 1.5 * v)
            + Array1::from_shape_fn(n, |i| ((i * 13) % 7) as f64 * 0.1);
        let fitted = LassoLarsIC::new(IcCriterion::Bic).fit(&x, &y).unwrap();
        for j in [0_usize, 2, 4] {
            assert!(
                fitted.fitted.coef[j].abs() > 1.0,
                "{:?}",
                fitted.fitted.coef
            );
        }
    }
}
