//! Smooth numerical optimisation shared by estimator crates.
//!
//! [`minimize_lbfgsb`] is a small, dependency-free limited-memory BFGS
//! minimiser with optional box constraints `lo_i <= x_i <= hi_i`. It plays
//! the role `scipy.optimize.minimize(method="L-BFGS-B")` plays in
//! scikit-learn (Gaussian-process hyperparameter search, `MLP(solver="lbfgs")`).
//!
//! # Algorithm
//!
//! A projected active-set variant of L-BFGS-B:
//!
//! 1. Variables sitting on a bound whose gradient pushes them further out
//!    are frozen for the iteration (the *active set*).
//! 2. The search direction is the L-BFGS two-loop recursion restricted to
//!    the free variables (curvature pairs measured on the free subspace;
//!    pairs with non-positive restricted curvature are skipped, so the
//!    restricted inverse-Hessian stays positive definite).
//! 3. Free variables on a bound whose direction component points outward
//!    are moved to the active set and the direction is recomputed.
//! 4. A strong-Wolfe line search (Nocedal & Wright, Alg. 3.5/3.6, cubic
//!    interpolation) is run along `x + a·d` for `a ∈ (0, a_max]`, where
//!    `a_max` is the largest feasible step. Hitting `a_max` with sufficient
//!    decrease is accepted; the blocking variable then lands on its bound.
//!
//! Stopping rules mirror SciPy: projected-gradient sup-norm `<= pgtol`, or
//! relative reduction `(f_k - f_{k+1}) / max(|f_k|, |f_{k+1}|, 1) <= ftol`,
//! or the iteration / function-evaluation budget.
//!
//! The objective may return a non-finite value (e.g. a failed Cholesky
//! factorisation); the line search then backtracks.

use crate::error::{Result, RustMlError};
use std::collections::VecDeque;

/// Options for [`minimize_lbfgsb`]. Defaults follow
/// `scipy.optimize.minimize(method="L-BFGS-B")`.
#[derive(Debug, Clone)]
pub struct LbfgsbOptions {
    /// Number of stored correction pairs (SciPy `maxcor`, default 10).
    pub m: usize,
    /// Maximum number of iterations (SciPy `maxiter`, default 15000).
    pub max_iter: usize,
    /// Maximum number of objective evaluations (SciPy `maxfun`, default 15000).
    pub max_fun: usize,
    /// Projected-gradient sup-norm tolerance (SciPy `gtol`, default 1e-5).
    pub pgtol: f64,
    /// Relative function-reduction tolerance (SciPy `ftol`, default 2.22e-9).
    pub ftol: f64,
    /// Maximum objective evaluations per line search (SciPy `maxls`, default 20).
    pub max_ls: usize,
}

impl Default for LbfgsbOptions {
    fn default() -> Self {
        Self {
            m: 10,
            max_iter: 15000,
            max_fun: 15000,
            pgtol: 1e-5,
            ftol: 2.220_446_049_250_313e-9,
            max_ls: 20,
        }
    }
}

/// Why [`minimize_lbfgsb`] stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LbfgsbStatus {
    /// Projected gradient sup-norm fell below `pgtol`.
    ProjectedGradient,
    /// Relative reduction of the objective fell below `ftol`.
    RelativeReduction,
    /// `max_iter` iterations were performed.
    MaxIter,
    /// `max_fun` objective evaluations were performed.
    MaxFun,
    /// No acceptable step could be found along a steepest-descent direction.
    LineSearchFailed,
    /// The objective is not finite at the starting point.
    NonFiniteStart,
}

/// Result of [`minimize_lbfgsb`].
#[derive(Debug, Clone)]
pub struct LbfgsbResult {
    /// Best point found.
    pub x: Vec<f64>,
    /// Objective value at `x`.
    pub fun: f64,
    /// Gradient at `x`.
    pub grad: Vec<f64>,
    /// Number of completed iterations.
    pub n_iter: usize,
    /// Number of objective evaluations.
    pub n_fun: usize,
    /// Termination reason.
    pub status: LbfgsbStatus,
}

impl LbfgsbResult {
    /// `true` if a tolerance-based stopping rule fired.
    pub fn converged(&self) -> bool {
        matches!(
            self.status,
            LbfgsbStatus::ProjectedGradient | LbfgsbStatus::RelativeReduction
        )
    }
}

struct Problem<'a, O> {
    obj: O,
    lo: &'a [f64],
    hi: &'a [f64],
    n_fun: usize,
}

#[derive(Clone)]
struct Trial {
    a: f64,
    f: f64,
    dphi: f64,
    x: Vec<f64>,
    g: Vec<f64>,
}

impl<O: FnMut(&[f64], &mut [f64]) -> f64> Problem<'_, O> {
    fn eval(&mut self, x: &[f64], g: &mut [f64]) -> f64 {
        self.n_fun += 1;
        let f = (self.obj)(x, g);
        if f.is_finite() && g.iter().all(|v| v.is_finite()) {
            f
        } else {
            f64::INFINITY
        }
    }

    /// Evaluate at `x + a·d`, clamped into the box (snapping onto bounds
    /// within rounding distance).
    fn trial(&mut self, x: &[f64], d: &[f64], a: f64) -> Trial {
        let n = x.len();
        let mut xt = vec![0.0; n];
        for i in 0..n {
            let mut v = x[i] + a * d[i];
            let (lo, hi) = (self.lo[i], self.hi[i]);
            if v <= lo
                || (lo.is_finite() && d[i] < 0.0 && (v - lo).abs() <= 1e-12 * (1.0 + lo.abs()))
            {
                v = lo;
            }
            if v >= hi
                || (hi.is_finite() && d[i] > 0.0 && (v - hi).abs() <= 1e-12 * (1.0 + hi.abs()))
            {
                v = hi;
            }
            xt[i] = v;
        }
        let mut g = vec![0.0; n];
        let f = self.eval(&xt, &mut g);
        let dphi = dot(&g, d);
        Trial {
            a,
            f,
            dphi,
            x: xt,
            g,
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn dot_masked(a: &[f64], b: &[f64], mask: &[bool]) -> f64 {
    let mut s = 0.0;
    for i in 0..a.len() {
        if mask[i] {
            s += a[i] * b[i];
        }
    }
    s
}

/// Cubic interpolation minimiser of the Hermite cubic through
/// `(a, fa, da)` and `(b, fb, db)`; `None` if degenerate.
fn cubic_min(a: f64, fa: f64, da: f64, b: f64, fb: f64, db: f64) -> Option<f64> {
    let d1 = da + db - 3.0 * (fa - fb) / (a - b);
    let disc = d1 * d1 - da * db;
    if disc < 0.0 {
        return None;
    }
    let d2 = (b - a).signum() * disc.sqrt();
    let denom = db - da + 2.0 * d2;
    if denom == 0.0 {
        return None;
    }
    let t = b - (b - a) * (db + d2 - d1) / denom;
    if t.is_finite() {
        Some(t)
    } else {
        None
    }
}

/// Minimise a smooth function with optional box constraints.
///
/// `obj(x, grad)` must return `f(x)` and write `∇f(x)` into `grad`.
/// `bounds[i] = (lo_i, hi_i)`; use `f64::NEG_INFINITY` / `f64::INFINITY`
/// for one-sided or free variables. `None` means unconstrained.
///
/// The starting point is projected into the box.
///
/// # Errors
///
/// Returns [`RustMlError::InvalidParameter`] if `bounds` has the wrong
/// length or contains `lo > hi` / NaN.
///
/// # Examples
///
/// ```
/// use anofox_ml_core::optim::{minimize_lbfgsb, LbfgsbOptions};
///
/// // Minimise (x-3)^2 + (y+1)^2 subject to x <= 2.
/// let res = minimize_lbfgsb(
///     |x, g| {
///         g[0] = 2.0 * (x[0] - 3.0);
///         g[1] = 2.0 * (x[1] + 1.0);
///         (x[0] - 3.0).powi(2) + (x[1] + 1.0).powi(2)
///     },
///     &[0.0, 0.0],
///     Some(&[(f64::NEG_INFINITY, 2.0), (f64::NEG_INFINITY, f64::INFINITY)]),
///     &LbfgsbOptions::default(),
/// )
/// .unwrap();
/// assert!((res.x[0] - 2.0).abs() < 1e-8);
/// assert!((res.x[1] + 1.0).abs() < 1e-5);
/// ```
pub fn minimize_lbfgsb<O>(
    obj: O,
    x0: &[f64],
    bounds: Option<&[(f64, f64)]>,
    opts: &LbfgsbOptions,
) -> Result<LbfgsbResult>
where
    O: FnMut(&[f64], &mut [f64]) -> f64,
{
    let n = x0.len();
    let (lo, hi): (Vec<f64>, Vec<f64>) = match bounds {
        Some(b) => {
            if b.len() != n {
                return Err(RustMlError::InvalidParameter(format!(
                    "bounds has length {} but x0 has length {n}",
                    b.len()
                )));
            }
            for (i, &(l, h)) in b.iter().enumerate() {
                if l.is_nan() || h.is_nan() || l > h {
                    return Err(RustMlError::InvalidParameter(format!(
                        "invalid bounds ({l}, {h}) for variable {i}"
                    )));
                }
            }
            b.iter().copied().unzip()
        }
        None => (vec![f64::NEG_INFINITY; n], vec![f64::INFINITY; n]),
    };
    let mut prob = Problem {
        obj,
        lo: &lo,
        hi: &hi,
        n_fun: 0,
    };
    let mut x: Vec<f64> = x0
        .iter()
        .enumerate()
        .map(|(i, &v)| v.max(lo[i]).min(hi[i]))
        .collect();
    let mut g = vec![0.0; n];
    let mut f = prob.eval(&x, &mut g);
    if !f.is_finite() {
        return Ok(LbfgsbResult {
            x,
            fun: f,
            grad: g,
            n_iter: 0,
            n_fun: prob.n_fun,
            status: LbfgsbStatus::NonFiniteStart,
        });
    }
    if n == 0 {
        return Ok(LbfgsbResult {
            x,
            fun: f,
            grad: g,
            n_iter: 0,
            n_fun: prob.n_fun,
            status: LbfgsbStatus::ProjectedGradient,
        });
    }

    let m = opts.m.max(1);
    let mut s_hist: VecDeque<Vec<f64>> = VecDeque::with_capacity(m);
    let mut y_hist: VecDeque<Vec<f64>> = VecDeque::with_capacity(m);
    let mut free = vec![true; n];
    let mut d = vec![0.0; n];
    let mut n_iter = 0;
    let c1 = 1e-4;
    let c2 = 0.9;

    let status = loop {
        // Projected gradient sup-norm: ||P(x - g) - x||_inf.
        let pg = (0..n)
            .map(|i| ((x[i] - g[i]).max(lo[i]).min(hi[i]) - x[i]).abs())
            .fold(0.0_f64, f64::max);
        if pg <= opts.pgtol {
            break LbfgsbStatus::ProjectedGradient;
        }
        if n_iter >= opts.max_iter {
            break LbfgsbStatus::MaxIter;
        }
        if prob.n_fun >= opts.max_fun {
            break LbfgsbStatus::MaxFun;
        }

        for i in 0..n {
            free[i] = !((x[i] <= lo[i] && g[i] > 0.0) || (x[i] >= hi[i] && g[i] < 0.0));
        }
        loop {
            two_loop(&g, &free, &s_hist, &y_hist, &mut d);
            let mut changed = false;
            for i in 0..n {
                if free[i] && ((x[i] <= lo[i] && d[i] < 0.0) || (x[i] >= hi[i] && d[i] > 0.0)) {
                    free[i] = false;
                    d[i] = 0.0;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        let mut gd = dot(&g, &d);
        if gd.is_nan() || gd >= 0.0 {
            // Not a descent direction: fall back to projected steepest descent.
            s_hist.clear();
            y_hist.clear();
            for i in 0..n {
                d[i] = if free[i] { -g[i] } else { 0.0 };
            }
            gd = dot(&g, &d);
            if gd.is_nan() || gd >= 0.0 {
                break LbfgsbStatus::ProjectedGradient;
            }
        }

        // Largest feasible step.
        let mut a_max = f64::INFINITY;
        for i in 0..n {
            if d[i] < 0.0 && lo[i].is_finite() {
                a_max = a_max.min((lo[i] - x[i]) / d[i]);
            } else if d[i] > 0.0 && hi[i].is_finite() {
                a_max = a_max.min((hi[i] - x[i]) / d[i]);
            }
        }
        let a_max = a_max.max(0.0);
        if a_max <= 0.0 {
            break LbfgsbStatus::ProjectedGradient;
        }
        let a0 = if s_hist.is_empty() {
            let dn = dot(&d, &d).sqrt();
            (1.0 / dn).min(1.0)
        } else {
            1.0
        }
        .min(a_max);

        let accepted = line_search(&mut prob, &x, &d, f, gd, a0, a_max, c1, c2, opts.max_ls);
        let t = match accepted {
            Some(t) => t,
            None => {
                if s_hist.is_empty() {
                    break LbfgsbStatus::LineSearchFailed;
                }
                s_hist.clear();
                y_hist.clear();
                continue;
            }
        };
        n_iter += 1;

        let s: Vec<f64> = t.x.iter().zip(&x).map(|(a, b)| a - b).collect();
        let yv: Vec<f64> = t.g.iter().zip(&g).map(|(a, b)| a - b).collect();
        let sy = dot(&s, &yv);
        let yy = dot(&yv, &yv);
        if sy > f64::EPSILON * yy {
            if s_hist.len() == m {
                s_hist.pop_front();
                y_hist.pop_front();
            }
            s_hist.push_back(s);
            y_hist.push_back(yv);
        }
        let f_old = f;
        x = t.x;
        g = t.g;
        f = t.f;
        if (f_old - f) / f_old.abs().max(f.abs()).max(1.0) <= opts.ftol {
            break LbfgsbStatus::RelativeReduction;
        }
    };

    Ok(LbfgsbResult {
        x,
        fun: f,
        grad: g,
        n_iter,
        n_fun: prob.n_fun,
        status,
    })
}

/// `d = -H g` restricted to `free` (zero elsewhere).
fn two_loop(
    g: &[f64],
    free: &[bool],
    s_hist: &VecDeque<Vec<f64>>,
    y_hist: &VecDeque<Vec<f64>>,
    d: &mut [f64],
) {
    let n = g.len();
    let k = s_hist.len();
    let mut q: Vec<f64> = (0..n).map(|i| if free[i] { g[i] } else { 0.0 }).collect();
    let mut rho = vec![0.0; k];
    let mut alpha = vec![0.0; k];
    let mut gamma = 1.0;
    let mut have_gamma = false;
    for j in (0..k).rev() {
        let sy = dot_masked(&s_hist[j], &y_hist[j], free);
        if sy <= 1e-300 {
            continue;
        }
        if !have_gamma {
            let yy = dot_masked(&y_hist[j], &y_hist[j], free);
            if yy > 0.0 {
                gamma = sy / yy;
                have_gamma = true;
            }
        }
        rho[j] = 1.0 / sy;
        alpha[j] = rho[j] * dot_masked(&s_hist[j], &q, free);
        for i in 0..n {
            if free[i] {
                q[i] -= alpha[j] * y_hist[j][i];
            }
        }
    }
    for v in q.iter_mut() {
        *v *= gamma;
    }
    for j in 0..k {
        if rho[j] == 0.0 {
            continue;
        }
        let b = rho[j] * dot_masked(&y_hist[j], &q, free);
        for i in 0..n {
            if free[i] {
                q[i] += (alpha[j] - b) * s_hist[j][i];
            }
        }
    }
    for i in 0..n {
        d[i] = if free[i] { -q[i] } else { 0.0 };
    }
}

/// Strong-Wolfe line search on `phi(a) = f(x + a d)`, `a ∈ (0, a_max]`.
/// Falls back to the best Armijo point seen if the budget runs out.
#[allow(clippy::too_many_arguments)]
fn line_search<O: FnMut(&[f64], &mut [f64]) -> f64>(
    prob: &mut Problem<'_, O>,
    x: &[f64],
    d: &[f64],
    f0: f64,
    dphi0: f64,
    a0: f64,
    a_max: f64,
    c1: f64,
    c2: f64,
    max_ls: usize,
) -> Option<Trial> {
    let armijo = |t: &Trial| t.f.is_finite() && t.f <= f0 + c1 * t.a * dphi0;
    let mut best: Option<Trial> = None;
    let keep_best = |t: &Trial, best: &mut Option<Trial>| {
        if armijo(t) && best.as_ref().is_none_or(|b| t.f < b.f) {
            *best = Some(t.clone());
        }
    };

    let mut prev = Trial {
        a: 0.0,
        f: f0,
        dphi: dphi0,
        x: x.to_vec(),
        g: Vec::new(),
    };
    let mut a = a0;
    let mut evals = 0;
    // Bracketing phase.
    let (mut lo, mut hi) = loop {
        if evals >= max_ls {
            return best;
        }
        let t = prob.trial(x, d, a);
        evals += 1;
        keep_best(&t, &mut best);
        if !t.f.is_finite() || t.f > f0 + c1 * a * dphi0 || (evals > 1 && t.f >= prev.f) {
            break (prev, t);
        }
        if t.dphi.abs() <= -c2 * dphi0 {
            return Some(t);
        }
        if t.dphi >= 0.0 {
            break (t, prev);
        }
        if a >= a_max {
            // Sufficient decrease at the feasible boundary: accept.
            return Some(t);
        }
        let a_next = (2.0 * a).min(a_max);
        prev = t;
        a = a_next;
    };

    // Zoom phase: `lo` satisfies Armijo with the lowest value seen so far,
    // the minimiser lies between `lo.a` and `hi.a`.
    while evals < max_ls {
        let (a_l, a_h) = (lo.a, hi.a);
        let width = (a_h - a_l).abs();
        if width <= 1e-14 * a_l.abs().max(a_h.abs()).max(1e-300) {
            break;
        }
        let mut a_j = if hi.f.is_finite() {
            cubic_min(a_l, lo.f, lo.dphi, a_h, hi.f, hi.dphi).unwrap_or(0.5 * (a_l + a_h))
        } else {
            0.5 * (a_l + a_h)
        };
        // Safeguard: stay well inside the bracket.
        let (left, right) = if a_l < a_h { (a_l, a_h) } else { (a_h, a_l) };
        let margin = 0.1 * width;
        if !(a_j > left + margin && a_j < right - margin) {
            a_j = 0.5 * (a_l + a_h);
        }
        let t = prob.trial(x, d, a_j);
        evals += 1;
        keep_best(&t, &mut best);
        if !t.f.is_finite() || t.f > f0 + c1 * a_j * dphi0 || t.f >= lo.f {
            hi = t;
        } else {
            if t.dphi.abs() <= -c2 * dphi0 {
                return Some(t);
            }
            if t.dphi * (hi.a - lo.a) >= 0.0 {
                hi = lo;
            }
            lo = t;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rosen(x: &[f64], g: &mut [f64]) -> f64 {
        let n = x.len();
        let mut f = 0.0;
        for v in g.iter_mut() {
            *v = 0.0;
        }
        for i in 0..n - 1 {
            let a = x[i + 1] - x[i] * x[i];
            let b = 1.0 - x[i];
            f += 100.0 * a * a + b * b;
            g[i] += -400.0 * x[i] * a - 2.0 * b;
            g[i + 1] += 200.0 * a;
        }
        f
    }

    #[test]
    fn rosenbrock_unbounded() {
        let opts = LbfgsbOptions {
            pgtol: 1e-8,
            ftol: 0.0,
            ..Default::default()
        };
        let res = minimize_lbfgsb(rosen, &[-1.2, 1.0, -0.5, 0.3, 0.8], None, &opts).unwrap();
        assert!(res.converged(), "{:?}", res.status);
        for v in &res.x {
            assert!((v - 1.0).abs() < 1e-6, "{:?}", res.x);
        }
    }

    #[test]
    fn rosenbrock_upper_bound_active() {
        // min rosen s.t. x0 <= 0.5 → (0.5, 0.25) (x1 free inside its box).
        let b = [(-2.0, 0.5), (-2.0, 2.0)];
        let opts = LbfgsbOptions {
            pgtol: 1e-9,
            ftol: 0.0,
            ..Default::default()
        };
        let res = minimize_lbfgsb(rosen, &[-1.2, 1.0], Some(&b), &opts).unwrap();
        assert!(res.converged(), "{:?}", res.status);
        assert!((res.x[0] - 0.5).abs() < 1e-12, "{:?}", res.x);
        assert!((res.x[1] - 0.25).abs() < 1e-7, "{:?}", res.x);
    }

    #[test]
    fn bounded_quadratic_matches_scipy() {
        // scipy.optimize.minimize(f, [0,0,0], jac=True, method="L-BFGS-B",
        //   bounds=[(1,3), (-inf, -2), (-0.5, 0.5)])
        // f = sum((x - c)^2 * w) with c = [0, 1, 2], w = [1, 2, 3]
        // → x* = [1, -2, 0.5]
        let c = [0.0, 1.0, 2.0];
        let w = [1.0, 2.0, 3.0];
        let res = minimize_lbfgsb(
            |x, g| {
                let mut f = 0.0;
                for i in 0..3 {
                    f += w[i] * (x[i] - c[i]).powi(2);
                    g[i] = 2.0 * w[i] * (x[i] - c[i]);
                }
                f
            },
            &[0.0, 0.0, 0.0],
            Some(&[(1.0, 3.0), (f64::NEG_INFINITY, -2.0), (-0.5, 0.5)]),
            &LbfgsbOptions::default(),
        )
        .unwrap();
        assert_eq!(res.x, vec![1.0, -2.0, 0.5]);
        assert_eq!(res.status, LbfgsbStatus::ProjectedGradient);
    }

    #[test]
    fn non_finite_region_is_avoided() {
        // f = -log(x) + x, minimum at x = 1; f = +inf for x <= 0.
        let res = minimize_lbfgsb(
            |x, g| {
                if x[0] <= 0.0 {
                    g[0] = 0.0;
                    return f64::INFINITY;
                }
                g[0] = -1.0 / x[0] + 1.0;
                -x[0].ln() + x[0]
            },
            &[5.0],
            None,
            &LbfgsbOptions {
                pgtol: 1e-10,
                ftol: 0.0,
                ..Default::default()
            },
        )
        .unwrap();
        assert!((res.x[0] - 1.0).abs() < 1e-8, "{:?}", res);
    }

    #[test]
    fn respects_max_fun_and_max_iter() {
        let opts = LbfgsbOptions {
            max_iter: 3,
            ..Default::default()
        };
        let res = minimize_lbfgsb(rosen, &[-1.2, 1.0], None, &opts).unwrap();
        assert_eq!(res.n_iter, 3);
        assert_eq!(res.status, LbfgsbStatus::MaxIter);
        let opts = LbfgsbOptions {
            max_fun: 5,
            ..Default::default()
        };
        let res = minimize_lbfgsb(rosen, &[-1.2, 1.0], None, &opts).unwrap();
        assert_eq!(res.status, LbfgsbStatus::MaxFun);
    }

    #[test]
    fn invalid_bounds_error() {
        assert!(minimize_lbfgsb(
            rosen,
            &[0.0, 0.0],
            Some(&[(1.0, 0.0), (0.0, 1.0)]),
            &LbfgsbOptions::default()
        )
        .is_err());
        assert!(minimize_lbfgsb(
            rosen,
            &[0.0, 0.0],
            Some(&[(0.0, 1.0)]),
            &LbfgsbOptions::default()
        )
        .is_err());
    }
}
