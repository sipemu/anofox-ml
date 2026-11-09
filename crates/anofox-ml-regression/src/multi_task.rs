//! Multi-task Lasso / elastic net (sklearn's `MultiTaskLasso` /
//! `MultiTaskElasticNet`): one coefficient row per target, the features
//! selected jointly by the ℓ2,1 penalty,
//!
//! `1/(2n) ‖Y − XWᵀ‖²_F + α ρ Σ_j ‖W_{:,j}‖₂ + α (1 − ρ)/2 ‖W‖²_F`,
//!
//! solved by block coordinate descent (sklearn's
//! `enet_coordinate_descent_multi_task`) until the updates vanish, i.e.
//! to the unique minimum sklearn's duality-gap stop approaches.

use anofox_ml_core::{Result, RustMlError};
use ndarray::{s, Array1, Array2, Array3, ArrayView2, AsArray, Axis, Ix2};

/// sklearn's `MultiTaskElasticNet` (`MultiTaskLasso`: `l1_ratio = 1`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MultiTaskElasticNet {
    pub alpha: f64,
    pub l1_ratio: f64,
    pub fit_intercept: bool,
    pub max_iter: usize,
}

impl MultiTaskElasticNet {
    pub fn new(alpha: f64, l1_ratio: f64) -> Self {
        Self {
            alpha,
            l1_ratio,
            fit_intercept: true,
            max_iter: 100_000,
        }
    }

    /// sklearn's `MultiTaskLasso(alpha)`.
    pub fn lasso(alpha: f64) -> Self {
        Self::new(alpha, 1.0)
    }

    pub fn fit(&self, x: &Array2<f64>, y: &Array2<f64>) -> Result<FittedMultiTask> {
        check_xy(x, y)?;
        check_penalty(self.alpha, self.l1_ratio)?;
        let c = Centered::new(x.view(), y.view(), self.fit_intercept);
        let mut w = Array2::<f64>::zeros((y.ncols(), x.ncols()));
        c.solve(self.alpha, self.l1_ratio, self.max_iter, &mut w);
        Ok(c.fitted(w))
    }
}

fn check_xy(x: &Array2<f64>, y: &Array2<f64>) -> Result<()> {
    let (n, p) = x.dim();
    if y.nrows() != n {
        return Err(RustMlError::ShapeMismatch(format!(
            "X has {n} rows but Y has {}",
            y.nrows()
        )));
    }
    if n == 0 || p == 0 || y.ncols() == 0 {
        return Err(RustMlError::EmptyInput("X and Y must be non-empty".into()));
    }
    Ok(())
}

fn check_penalty(alpha: f64, l1_ratio: f64) -> Result<()> {
    if alpha.is_nan() || alpha < 0.0 || !(0.0..=1.0).contains(&l1_ratio) {
        return Err(RustMlError::InvalidParameter(format!(
            "alpha must be >= 0 and l1_ratio in [0, 1], got {alpha} and {l1_ratio}"
        )));
    }
    Ok(())
}

/// Centred data of one fit (or one training fold).
struct Centered {
    xc: Array2<f64>,
    yc: Array2<f64>,
    x_mean: Array1<f64>,
    y_mean: Array1<f64>,
    norms: Vec<f64>,
}

impl Centered {
    fn new(x: ArrayView2<f64>, y: ArrayView2<f64>, fit_intercept: bool) -> Self {
        let (p, k) = (x.ncols(), y.ncols());
        let (x_mean, y_mean) = if fit_intercept {
            (x.mean_axis(Axis(0)).unwrap(), y.mean_axis(Axis(0)).unwrap())
        } else {
            (Array1::zeros(p), Array1::zeros(k))
        };
        let xc = &x - &x_mean;
        let yc = &y - &y_mean;
        let norms = (0..p).map(|j| xc.column(j).dot(&xc.column(j))).collect();
        Self {
            xc,
            yc,
            x_mean,
            y_mean,
            norms,
        }
    }

    /// Block coordinate descent from `w` (a warm start) until the updates
    /// vanish.
    fn solve(&self, alpha: f64, l1_ratio: f64, max_iter: usize, w: &mut Array2<f64>) {
        let (n, k) = self.yc.dim();
        let l1 = alpha * l1_ratio * n as f64;
        let l2 = alpha * (1.0 - l1_ratio) * n as f64;
        let mut r = &self.yc - &self.xc.dot(&w.t());
        for _ in 0..max_iter {
            let mut max_change = 0.0f64;
            let mut max_w = 0.0f64;
            for (j, &norm_j) in self.norms.iter().enumerate() {
                if norm_j == 0.0 {
                    continue;
                }
                let xj = self.xc.column(j);
                let old = w.column(j).to_owned();
                // tmp = X_jᵀ R + ‖X_j‖² w_j
                let tmp: Array1<f64> = r.t().dot(&xj) + &old * norm_j;
                let norm = tmp.dot(&tmp).sqrt();
                let shrink = if norm > 0.0 {
                    (1.0 - l1 / norm).max(0.0)
                } else {
                    0.0
                };
                let new = &tmp * (shrink / (norm_j + l2));
                let delta = &new - &old;
                if delta.iter().any(|&d| d != 0.0) {
                    for i in 0..n {
                        for t in 0..k {
                            r[[i, t]] -= xj[i] * delta[t];
                        }
                    }
                }
                max_change = delta.iter().fold(max_change, |m, d| m.max(d.abs()));
                max_w = new.iter().fold(max_w, |m, v| m.max(v.abs()));
                w.column_mut(j).assign(&new);
            }
            if max_change <= 1e-15 * max_w.max(1e-300) || max_change == 0.0 {
                break;
            }
        }
    }

    fn fitted(&self, w: Array2<f64>) -> FittedMultiTask {
        let intercept = &self.y_mean - &w.dot(&self.x_mean);
        FittedMultiTask { coef: w, intercept }
    }

    /// sklearn's `_alpha_grid`: `n_alphas` values from `alpha_max =
    /// max_j ‖(XcᵀYc)_j‖₂ / (n·l1_ratio)` down to `eps·alpha_max`,
    /// geometrically spaced.
    fn alpha_grid(&self, l1_ratio: f64, n_alphas: usize, eps: f64) -> Vec<f64> {
        let n = self.xc.nrows() as f64;
        let xy = self.xc.t().dot(&self.yc);
        let max_sq = xy
            .rows()
            .into_iter()
            .map(|r| r.dot(&r))
            .fold(0.0f64, f64::max);
        let alpha_max = max_sq.sqrt() / (n * l1_ratio);
        if alpha_max <= 1e-15 {
            return vec![1e-15; n_alphas];
        }
        geomspace(alpha_max, alpha_max * eps, n_alphas)
    }
}

/// numpy's `geomspace(start, stop, num)` for positive bounds.
fn geomspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 1 {
        return vec![start];
    }
    let (a, b) = (start.log10(), stop.log10());
    let step = (b - a) / (num - 1) as f64;
    let mut out: Vec<f64> = (0..num).map(|i| 10f64.powf(a + i as f64 * step)).collect();
    out[0] = start;
    out[num - 1] = stop;
    out
}

/// sklearn's `MultiTaskElasticNetCV` / `MultiTaskLassoCV` (`l1_ratios =
/// [1]`): every `(l1_ratio, alpha)` is scored by the mean over `cv`
/// contiguous folds (sklearn's `KFold`) of the held-out MSE averaged over
/// the targets, the alphas of a path solved with warm starts. The best pair
/// is refitted on all rows.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct MultiTaskElasticNetCV {
    pub l1_ratios: Vec<f64>,
    /// Explicit alphas (sorted decreasingly); `None`: sklearn's grid of
    /// `n_alphas` per `l1_ratio`.
    pub alphas: Option<Vec<f64>>,
    pub n_alphas: usize,
    pub eps: f64,
    pub cv: usize,
    pub fit_intercept: bool,
    pub max_iter: usize,
}

impl Default for MultiTaskElasticNetCV {
    fn default() -> Self {
        Self {
            l1_ratios: vec![0.5],
            alphas: None,
            n_alphas: 100,
            eps: 1e-3,
            cv: 5,
            fit_intercept: true,
            max_iter: 100_000,
        }
    }
}

impl MultiTaskElasticNetCV {
    /// sklearn's `MultiTaskLassoCV()`.
    pub fn lasso() -> Self {
        Self {
            l1_ratios: vec![1.0],
            ..Self::default()
        }
    }

    pub fn fit(&self, x: &Array2<f64>, y: &Array2<f64>) -> Result<FittedMultiTaskCV> {
        check_xy(x, y)?;
        let n = x.nrows();
        if self.l1_ratios.is_empty() {
            return Err(RustMlError::InvalidParameter(
                "l1_ratios must not be empty".into(),
            ));
        }
        if self.cv < 2 || self.cv > n {
            return Err(RustMlError::InvalidParameter(format!(
                "cv must be in 2..={n}, got {}",
                self.cv
            )));
        }
        let full = Centered::new(x.view(), y.view(), self.fit_intercept);
        let mut grids = Vec::with_capacity(self.l1_ratios.len());
        for &rho in &self.l1_ratios {
            check_penalty(0.0, rho)?;
            let grid = match &self.alphas {
                Some(a) => crate::elastic_net_cv::sorted_alphas(a)?,
                None => {
                    if rho == 0.0 {
                        return Err(RustMlError::InvalidParameter(
                            "the automatic alpha grid needs l1_ratio > 0; pass alphas".into(),
                        ));
                    }
                    if self.n_alphas == 0 {
                        return Err(RustMlError::InvalidParameter(
                            "n_alphas must be at least 1".into(),
                        ));
                    }
                    full.alpha_grid(rho, self.n_alphas, self.eps)
                }
            };
            grids.push(grid);
        }
        let na = grids[0].len();
        let ids = crate::elastic_net_cv::contiguous_fold_ids(n, self.cv);
        let folds: Vec<(Vec<usize>, Vec<usize>)> = (0..self.cv)
            .map(|f| {
                let test: Vec<usize> = (0..n).filter(|&i| ids[i] == f).collect();
                (crate::elastic_net_cv::train_rows(n, &test), test)
            })
            .collect();
        let k = y.ncols();
        let mut mse_path = Array3::<f64>::zeros((self.l1_ratios.len(), na, self.cv));
        for (f, (train, test)) in folds.iter().enumerate() {
            let c = Centered::new(
                x.select(Axis(0), train).view(),
                y.select(Axis(0), train).view(),
                self.fit_intercept,
            );
            let (xt, yt) = (x.select(Axis(0), test), y.select(Axis(0), test));
            for (l, (&rho, grid)) in self.l1_ratios.iter().zip(&grids).enumerate() {
                let mut w = Array2::<f64>::zeros((k, x.ncols()));
                for (a, &alpha) in grid.iter().enumerate() {
                    c.solve(alpha, rho, self.max_iter, &mut w);
                    let b = &c.y_mean - &w.dot(&c.x_mean);
                    let resid = &yt - &(xt.dot(&w.t()) + &b);
                    mse_path[[l, a, f]] = resid.mapv(|v| v * v).mean().unwrap_or(0.0);
                }
            }
        }
        let mut best = (f64::INFINITY, 0, 0);
        for l in 0..self.l1_ratios.len() {
            let mean: Vec<f64> = (0..na)
                .map(|a| mse_path.slice(s![l, a, ..]).mean().unwrap_or(0.0))
                .collect();
            let a = (0..na).fold(0, |b, i| if mean[i] < mean[b] { i } else { b });
            if mean[a] < best.0 {
                best = (mean[a], l, a);
            }
        }
        let (alpha, l1_ratio) = (grids[best.1][best.2], self.l1_ratios[best.1]);
        let mut w = Array2::<f64>::zeros((k, x.ncols()));
        full.solve(alpha, l1_ratio, self.max_iter, &mut w);
        Ok(FittedMultiTaskCV {
            alpha,
            l1_ratio,
            alphas: Array2::from_shape_fn((grids.len(), na), |(l, a)| grids[l][a]),
            mse_path,
            fitted: full.fitted(w),
        })
    }
}

/// A fitted [`MultiTaskElasticNetCV`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedMultiTaskCV {
    /// The selected alpha and l1_ratio (sklearn's `alpha_`, `l1_ratio_`).
    pub alpha: f64,
    pub l1_ratio: f64,
    /// The alpha grid per l1_ratio (`n_l1_ratios × n_alphas`).
    pub alphas: Array2<f64>,
    /// Held-out MSE per l1_ratio, alpha and fold.
    pub mse_path: Array3<f64>,
    /// The refit on all rows at the selected pair.
    pub fitted: FittedMultiTask,
}

impl FittedMultiTaskCV {
    pub fn predict<'a>(&self, x: impl AsArray<'a, f64, Ix2>) -> Result<Array2<f64>> {
        self.fitted.predict(x)
    }
}

/// A fitted multi-task linear model.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedMultiTask {
    /// `(n_targets, n_features)` (sklearn `coef_`).
    pub coef: Array2<f64>,
    /// One per target (sklearn `intercept_`).
    pub intercept: Array1<f64>,
}

impl FittedMultiTask {
    /// `X Wᵀ + b`, one column per target.
    pub fn predict<'a>(&self, x: impl AsArray<'a, f64, Ix2>) -> Result<Array2<f64>> {
        let x: ArrayView2<f64> = x.into();
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
