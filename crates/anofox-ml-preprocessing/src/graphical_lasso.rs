//! Sparse inverse covariance (sklearn's `GraphicalLasso` and
//! `GraphicalLassoCV`, `mode="cd"`).
//!
//! sklearn's `_graphical_lasso`: start from the empirical covariance with
//! its off-diagonal shrunk by 0.95, then sweep the features, solving each
//! one's Lasso on the others' covariance block by coordinate descent
//! (`enet_coordinate_descent_gram`, ported with its gap-safe screening and
//! stopping rules), until the duality gap is below `tol`.
//! `GraphicalLassoCV` refines a log-spaced alpha grid by the mean
//! held-out log-likelihood over K folds, as sklearn does.

use anofox_ml_core::{FitUnsupervised, Result, RustMlError, Transform};
use ndarray::{Array1, Array2, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

use crate::covariance::{empirical, location_of, logdet, pinvh, FittedCovariance};

// ------------------------------------------------------------------
// enet_coordinate_descent_gram (beta = 0, cyclic, with screening)
// ------------------------------------------------------------------

/// `gap_enet_gram`: the duality gap and `‖XᵀA‖∞`; fills `xta`.
fn gap_gram(
    w: &[f64],
    alpha: f64,
    qw: &[f64],
    q: &[f64],
    y_norm2: f64,
    xta: &mut [f64],
) -> (f64, f64) {
    let q_dot_w: f64 = w.iter().zip(q).map(|(a, b)| a * b).sum();
    for j in 0..w.len() {
        xta[j] = q[j] - qw[j];
    }
    let dual_norm = xta.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let wqw: f64 = w.iter().zip(qw).map(|(a, b)| a * b).sum();
    let r_norm2 = y_norm2 + wqw - 2.0 * q_dot_w;
    let (mut gap, c) = if dual_norm > alpha {
        let c = alpha / dual_norm;
        (0.5 * (r_norm2 + r_norm2 * c * c), c)
    } else {
        (r_norm2, 1.0)
    };
    let l1: f64 = w.iter().map(|v| v.abs()).sum();
    gap += alpha * l1 - c * (y_norm2 - q_dot_w);
    (gap, dual_norm)
}

/// `screen_features_enet_gram`: the features the gap-safe sphere keeps, in
/// order; the others are set to 0 (and removed from `Qw`).
#[allow(clippy::too_many_arguments)]
fn screen(
    q_mat: &Array2<f64>,
    xta: &[f64],
    w: &mut [f64],
    qw: &mut [f64],
    alpha: f64,
    gap: f64,
    dual_norm: f64,
    active: &mut Vec<usize>,
) {
    active.clear();
    let radius = (2.0 * gap.abs()).sqrt() / alpha;
    for j in 0..w.len() {
        let qjj = q_mat[[j, j]];
        if qjj == 0.0 {
            w[j] = 0.0;
            continue;
        }
        let theta = xta[j] / alpha.max(dual_norm);
        let d = (1.0 - theta.abs()) / qjj.sqrt();
        if d <= radius {
            active.push(j);
        } else {
            let wj = w[j];
            for (k, v) in qw.iter_mut().enumerate() {
                *v -= wj * q_mat[[j, k]];
            }
            w[j] = 0.0;
        }
    }
}

/// sklearn's `enet_coordinate_descent_gram(w, alpha, beta=0, Q, q, y=q,
/// max_iter, tol, random=False, positive=False, do_screening=True)`.
fn enet_cd_gram(
    mut w: Vec<f64>,
    alpha: f64,
    q_mat: &Array2<f64>,
    q: &[f64],
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    let p = w.len();
    let mut qw: Vec<f64> = (0..p)
        .map(|i| (0..p).map(|j| q_mat[[i, j]] * w[j]).sum())
        .collect();
    let mut xta = vec![0.0; p];
    let y_norm2: f64 = q.iter().map(|v| v * v).sum();
    let d_w_tol = tol;
    let tol = tol * y_norm2;
    let (gap, dual_norm) = gap_gram(&w, alpha, &qw, q, y_norm2, &mut xta);
    if (0.0..=tol).contains(&gap) {
        return w;
    }
    let mut active = Vec::with_capacity(p);
    screen(
        q_mat,
        &xta,
        &mut w,
        &mut qw,
        alpha,
        gap,
        dual_norm,
        &mut active,
    );
    for n_iter in 0..max_iter {
        let (mut w_max, mut d_w_max) = (0.0_f64, 0.0_f64);
        for &j in &active {
            let qjj = q_mat[[j, j]];
            if qjj == 0.0 {
                continue;
            }
            let w_j = w[j];
            let tmp = q[j] - qw[j] + w_j * qjj;
            let sign = if tmp > 0.0 {
                1.0
            } else if tmp < 0.0 {
                -1.0
            } else {
                0.0
            };
            w[j] = sign * (tmp.abs() - alpha).max(0.0) / qjj;
            if w[j] != w_j {
                let d = w[j] - w_j;
                for (k, v) in qw.iter_mut().enumerate() {
                    *v += d * q_mat[[j, k]];
                }
            }
            d_w_max = d_w_max.max((w[j] - w_j).abs());
            w_max = w_max.max(w[j].abs());
        }
        if w_max == 0.0 || d_w_max / w_max <= d_w_tol || n_iter == max_iter - 1 {
            let (gap, dual_norm) = gap_gram(&w, alpha, &qw, q, y_norm2, &mut xta);
            if gap <= tol {
                break;
            }
            screen(
                q_mat,
                &xta,
                &mut w,
                &mut qw,
                alpha,
                gap,
                dual_norm,
                &mut active,
            );
        }
    }
    w
}

// ------------------------------------------------------------------
// _graphical_lasso
// ------------------------------------------------------------------

/// sklearn's `log_likelihood(emp_cov, precision)`.
fn log_likelihood(emp_cov: &Array2<f64>, precision: &Array2<f64>) -> Result<f64> {
    let p = precision.nrows() as f64;
    let ll =
        -(emp_cov * precision).sum() + logdet(precision)? - p * (2.0 * std::f64::consts::PI).ln();
    Ok(ll / 2.0)
}

fn off_diagonal_l1(precision: &Array2<f64>) -> f64 {
    precision.mapv(f64::abs).sum() - precision.diag().mapv(f64::abs).sum()
}

/// The result of [`graphical_lasso`].
struct GlassoFit {
    covariance: Array2<f64>,
    precision: Array2<f64>,
    /// `(cost, dual gap)` per iteration (sklearn's `costs_`).
    costs: Vec<(f64, f64)>,
    n_iter: usize,
}

fn ill_conditioned() -> RustMlError {
    RustMlError::InvalidParameter("The system is too ill-conditioned for this solver".into())
}

/// sklearn's `_graphical_lasso(emp_cov, alpha, cov_init, mode="cd", tol,
/// enet_tol, max_iter, eps)`.
fn graphical_lasso(
    emp_cov: &Array2<f64>,
    alpha: f64,
    cov_init: Option<&Array2<f64>>,
    tol: f64,
    enet_tol: f64,
    max_iter: usize,
    eps: f64,
) -> Result<GlassoFit> {
    let p = emp_cov.nrows();
    let ln2pi = (2.0 * std::f64::consts::PI).ln();
    if alpha == 0.0 {
        let precision = pinvh(emp_cov)?;
        let cost = -2.0 * log_likelihood(emp_cov, &precision)? + p as f64 * ln2pi;
        let d_gap = (emp_cov * &precision).sum() - p as f64;
        return Ok(GlassoFit {
            covariance: emp_cov.clone(),
            precision,
            costs: vec![(cost, d_gap)],
            n_iter: 0,
        });
    }
    let mut cov = cov_init.unwrap_or(emp_cov) * 0.95;
    for i in 0..p {
        cov[[i, i]] = emp_cov[[i, i]];
    }
    let mut prec = pinvh(&cov)?;
    let mut costs = Vec::new();
    let mut n_iter = 0;
    for it in 0..max_iter {
        n_iter = it + 1;
        for idx in 0..p {
            let others: Vec<usize> = (0..p).filter(|&j| j != idx).collect();
            let sub = cov.select(Axis(0), &others).select(Axis(1), &others);
            let row: Vec<f64> = others.iter().map(|&j| emp_cov[[idx, j]]).collect();
            let denom = prec[[idx, idx]] + 1000.0 * eps;
            let init: Vec<f64> = others.iter().map(|&j| -(prec[[j, idx]] / denom)).collect();
            let coefs = enet_cd_gram(init, alpha, &sub, &row, max_iter.max(20), enet_tol);
            let c_dot: f64 = others
                .iter()
                .zip(&coefs)
                .map(|(&j, c)| cov[[j, idx]] * c)
                .sum();
            prec[[idx, idx]] = 1.0 / (cov[[idx, idx]] - c_dot);
            for (&j, &c) in others.iter().zip(&coefs) {
                prec[[j, idx]] = -prec[[idx, idx]] * c;
                prec[[idx, j]] = -prec[[idx, idx]] * c;
            }
            let new_col = sub.dot(&Array1::from(coefs));
            for (&j, &v) in others.iter().zip(new_col.iter()) {
                cov[[idx, j]] = v;
                cov[[j, idx]] = v;
            }
        }
        if !prec.sum().is_finite() {
            return Err(ill_conditioned());
        }
        let d_gap = (emp_cov * &prec).sum() - p as f64 + alpha * off_diagonal_l1(&prec);
        let cost = -2.0 * log_likelihood(emp_cov, &prec)?
            + p as f64 * ln2pi
            + alpha * off_diagonal_l1(&prec);
        costs.push((cost, d_gap));
        if d_gap.abs() < tol {
            break;
        }
        if !cost.is_finite() && it > 0 {
            return Err(ill_conditioned());
        }
    }
    Ok(GlassoFit {
        covariance: cov,
        precision: prec,
        costs,
        n_iter,
    })
}

// ------------------------------------------------------------------
// GraphicalLasso
// ------------------------------------------------------------------

/// sklearn's `GraphicalLasso(mode="cd")`: the L1-penalised (`alpha`)
/// precision. With `precomputed`, `x` is the empirical covariance itself
/// (sklearn's `covariance="precomputed"`, location 0).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GraphicalLasso {
    pub alpha: f64,
    pub tol: f64,
    pub enet_tol: f64,
    pub max_iter: usize,
    pub assume_centered: bool,
    pub precomputed: bool,
    pub eps: f64,
}

impl Default for GraphicalLasso {
    fn default() -> Self {
        Self {
            alpha: 0.01,
            tol: 1e-4,
            enet_tol: 1e-4,
            max_iter: 100,
            assume_centered: false,
            precomputed: false,
            eps: f64::EPSILON,
        }
    }
}

/// The cross-validation grid of [`GraphicalLassoCV`]: the alphas searched
/// (decreasing, 0 last) and their held-out log-likelihood per fold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphicalLassoCvResults {
    pub alphas: Vec<f64>,
    /// `n_alphas × n_folds`.
    pub scores: Array2<f64>,
}

/// A fitted [`GraphicalLasso`] / [`GraphicalLassoCV`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedGraphicalLasso {
    /// Location, covariance and the penalised precision.
    pub cov: FittedCovariance,
    /// `(cost, dual gap)` per iteration (sklearn's `costs_`).
    pub costs: Vec<(f64, f64)>,
    pub n_iter: usize,
    /// The penalty used (sklearn's `alpha_` for the CV).
    pub alpha: f64,
    pub cv_results: Option<GraphicalLassoCvResults>,
}

fn check_x(x: ArrayView2<f64>) -> Result<()> {
    if x.nrows() < 2 || x.ncols() < 2 {
        return Err(RustMlError::InvalidParameter(
            "GraphicalLasso needs at least 2 samples and 2 features".into(),
        ));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(RustMlError::InvalidParameter("x must be finite".into()));
    }
    Ok(())
}

fn check_alpha(alpha: f64) -> Result<()> {
    if !(alpha >= 0.0 && alpha.is_finite()) {
        return Err(RustMlError::InvalidParameter(format!(
            "alpha must be finite and >= 0, got {alpha}"
        )));
    }
    Ok(())
}

impl FitUnsupervised<f64> for GraphicalLasso {
    type Fitted = FittedGraphicalLasso;

    fn fit_view(&self, x: ArrayView2<f64>) -> Result<FittedGraphicalLasso> {
        check_x(x)?;
        check_alpha(self.alpha)?;
        let (emp_cov, location) = if self.precomputed {
            if x.nrows() != x.ncols() {
                return Err(RustMlError::ShapeMismatch(
                    "a precomputed covariance must be square".into(),
                ));
            }
            (x.to_owned(), Array1::zeros(x.ncols()))
        } else {
            (
                empirical(x, self.assume_centered),
                location_of(x, self.assume_centered),
            )
        };
        let fit = graphical_lasso(
            &emp_cov,
            self.alpha,
            None,
            self.tol,
            self.enet_tol,
            self.max_iter,
            self.eps,
        )?;
        Ok(FittedGraphicalLasso {
            cov: FittedCovariance::with_precision(location, fit.covariance, fit.precision),
            costs: fit.costs,
            n_iter: fit.n_iter,
            alpha: self.alpha,
            cv_results: None,
        })
    }
}

impl Transform<f64> for FittedGraphicalLasso {
    /// The squared Mahalanobis distances (one column), as the other
    /// covariance models.
    fn transform_view(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        self.cov.transform(x)
    }
}

// ------------------------------------------------------------------
// GraphicalLassoCV
// ------------------------------------------------------------------

/// The penalties of [`GraphicalLassoCV`]: a grid of `n` log-spaced alphas
/// refined `n_refinements` times (sklearn's int `alphas`), or a fixed list.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlassoAlphas {
    Grid(usize),
    List(Vec<f64>),
}

/// sklearn's `GraphicalLassoCV(mode="cd")`: alpha by the mean held-out
/// Gaussian log-likelihood over `cv` contiguous folds (or the given
/// `splits`), the grid refined around the best alpha `n_refinements`
/// times; each fold's path is warm-started along the alphas with
/// `max_iter / 10` iterations. The model is then refitted at the best
/// alpha.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GraphicalLassoCV {
    pub alphas: GlassoAlphas,
    pub n_refinements: usize,
    pub cv: usize,
    /// `(train, test)` index lists; overrides `cv`.
    pub splits: Option<Vec<(Vec<usize>, Vec<usize>)>>,
    pub tol: f64,
    pub enet_tol: f64,
    pub max_iter: usize,
    pub assume_centered: bool,
    pub eps: f64,
}

impl Default for GraphicalLassoCV {
    fn default() -> Self {
        Self {
            alphas: GlassoAlphas::Grid(4),
            n_refinements: 4,
            cv: 5,
            splits: None,
            tol: 1e-4,
            enet_tol: 1e-4,
            max_iter: 100,
            assume_centered: false,
            eps: f64::EPSILON,
        }
    }
}

/// numpy's `logspace(log10(a), log10(b), n)` (`linspace` with the last
/// point set exactly).
fn logspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    let (start, stop) = (a.log10(), b.log10());
    if n == 1 {
        return vec![10f64.powf(start)];
    }
    let step = (stop - start) / (n - 1) as f64;
    (0..n)
        .map(|i| {
            let y = if i == n - 1 {
                stop
            } else {
                i as f64 * step + start
            };
            10f64.powf(y)
        })
        .collect()
}

impl GraphicalLassoCV {
    /// sklearn's `graphical_lasso_path(X_train, alphas, X_test)`: the
    /// held-out log-likelihood per alpha (`-inf` when the fit fails).
    fn path_scores(
        &self,
        train: ArrayView2<f64>,
        test: ArrayView2<f64>,
        alphas: &[f64],
    ) -> Result<Vec<f64>> {
        let emp_cov = empirical(train, false);
        let test_cov = empirical(test, false);
        let mut cov = emp_cov.clone();
        let mut scores = Vec::with_capacity(alphas.len());
        for &alpha in alphas {
            let score = match graphical_lasso(
                &emp_cov,
                alpha,
                Some(&cov),
                self.tol,
                self.enet_tol,
                (0.1 * self.max_iter as f64) as usize,
                self.eps,
            ) {
                Ok(fit) => {
                    let s = log_likelihood(&test_cov, &fit.precision)?;
                    cov = fit.covariance;
                    s
                }
                Err(_) => f64::NEG_INFINITY,
            };
            scores.push(if score.is_finite() {
                score
            } else {
                f64::NEG_INFINITY
            });
        }
        Ok(scores)
    }
}

impl FitUnsupervised<f64> for GraphicalLassoCV {
    type Fitted = FittedGraphicalLasso;

    fn fit_view(&self, x: ArrayView2<f64>) -> Result<FittedGraphicalLasso> {
        check_x(x)?;
        let n = x.nrows();
        let location = location_of(x, self.assume_centered);
        let emp_cov = empirical(x, self.assume_centered);
        let splits = match &self.splits {
            Some(s) => s.clone(),
            None => anofox_ml_core::utils::k_fold(n, self.cv)?,
        };
        let (mut alphas, n_refinements, grid) = match &self.alphas {
            GlassoAlphas::List(a) => {
                for &v in a {
                    check_alpha(v)?;
                }
                (a.clone(), 1, None)
            }
            GlassoAlphas::Grid(k) => {
                let mut a1 = 0.0_f64;
                for i in 0..emp_cov.nrows() {
                    for j in 0..emp_cov.ncols() {
                        if i != j {
                            a1 = a1.max(emp_cov[[i, j]].abs());
                        }
                    }
                }
                let mut a = logspace(1e-2 * a1, a1, *k);
                a.reverse();
                (a, self.n_refinements, Some(*k))
            }
        };
        // (alpha, held-out score per fold), kept sorted by decreasing alpha.
        let mut path: Vec<(f64, Vec<f64>)> = Vec::new();
        let mut best_index = 0;
        for _ in 0..n_refinements {
            let mut fold_scores: Vec<Vec<f64>> = Vec::with_capacity(splits.len());
            for (train, test) in &splits {
                fold_scores.push(self.path_scores(
                    x.select(Axis(0), train).view(),
                    x.select(Axis(0), test).view(),
                    &alphas,
                )?);
            }
            for (a, &alpha) in alphas.iter().enumerate() {
                path.push((alpha, fold_scores.iter().map(|f| f[a]).collect()));
            }
            // A stable sort, decreasing alpha (sklearn's sorted(reverse=True)).
            path.sort_by(|a, b| b.0.total_cmp(&a.0));
            let mut best_score = f64::NEG_INFINITY;
            let mut last_finite = 0;
            for (index, (_, scores)) in path.iter().enumerate() {
                let mut s = scores.iter().sum::<f64>() / scores.len() as f64;
                if s >= 0.1 / f64::EPSILON {
                    s = f64::NAN;
                }
                if s.is_finite() {
                    last_finite = index;
                }
                if s >= best_score {
                    best_score = s;
                    best_index = index;
                }
            }
            let len = path.len();
            let (a1, a0) = if best_index == 0 {
                (path[0].0, path[1.min(len - 1)].0)
            } else if best_index == last_finite && best_index != len - 1 {
                (path[best_index].0, path[best_index + 1].0)
            } else if best_index == len - 1 {
                (path[best_index].0, 0.01 * path[best_index].0)
            } else {
                (path[best_index - 1].0, path[best_index + 1].0)
            };
            if let Some(k) = grid {
                let refined = logspace(a1, a0, k + 2);
                alphas = refined[1..k + 1].to_vec();
            }
        }
        // The alpha = 0 row: EmpiricalCovariance's held-out score.
        let mut empirical_scores = Vec::with_capacity(splits.len());
        for (train, test) in &splits {
            let model = crate::EmpiricalCovariance {
                assume_centered: false,
            }
            .fit(&x.select(Axis(0), train))?;
            empirical_scores.push(model.score(x.select(Axis(0), test).view())?);
        }
        let best_alpha = path[best_index].0;
        let mut cv_alphas: Vec<f64> = path.iter().map(|p| p.0).collect();
        cv_alphas.push(0.0);
        let n_folds = splits.len();
        let mut scores = Array2::zeros((cv_alphas.len(), n_folds));
        for (r, (_, s)) in path.iter().enumerate() {
            for f in 0..n_folds {
                scores[[r, f]] = s[f];
            }
        }
        for f in 0..n_folds {
            scores[[path.len(), f]] = empirical_scores[f];
        }
        let fit = graphical_lasso(
            &emp_cov,
            best_alpha,
            None,
            self.tol,
            self.enet_tol,
            self.max_iter,
            self.eps,
        )?;
        Ok(FittedGraphicalLasso {
            cov: FittedCovariance::with_precision(location, fit.covariance, fit.precision),
            costs: fit.costs,
            n_iter: fit.n_iter,
            alpha: best_alpha,
            cv_results: Some(GraphicalLassoCvResults {
                alphas: cv_alphas,
                scores,
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn large_alpha_gives_a_diagonal_precision() {
        let x = array![
            [1.0, 2.0, 0.5],
            [2.0, 1.0, 0.0],
            [0.0, 1.5, 2.0],
            [1.0, 0.0, 1.0],
            [3.0, 2.0, 1.0]
        ];
        let fit = GraphicalLasso {
            alpha: 10.0,
            ..Default::default()
        }
        .fit(&x)
        .unwrap();
        let p = fit.cov.precision();
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    assert!(p[[i, j]].abs() < 1e-12, "{i},{j}: {}", p[[i, j]]);
                }
            }
        }
        // alpha = 0 is the inverse of the empirical covariance.
        let fit0 = GraphicalLasso {
            alpha: 0.0,
            ..Default::default()
        }
        .fit(&x)
        .unwrap();
        let back = fit0.cov.precision().dot(fit0.cov.covariance());
        for i in 0..3 {
            assert!((back[[i, i]] - 1.0).abs() < 1e-9);
        }
    }
}
