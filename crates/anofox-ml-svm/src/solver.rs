//! libsvm-style SMO solver shared by the kernel SVMs.
//!
//! A port of libsvm's `Solver` and `Solver_NU` (Chang & Lin 2011, §4;
//! Fan, Chen & Lin 2005): it solves
//!
//! ```text
//! min_α  ½ αᵀ Q α + pᵀ α    s.t.  yᵀ α = Δ,  0 ≤ αᵢ ≤ Cᵢ
//! ```
//!
//! with `yᵢ ∈ {+1, −1}`, keeping the gradient `G = Qα + p` up to date in
//! `O(l)` per step, choosing the working pair by second-order selection
//! (WSS3), stopping when the maximal violating pair's gap drops below `eps`,
//! shrinking bounded variables that are unlikely to move (with gradient
//! reconstruction before the final optimality check) and computing `ρ` from
//! the free variables. `Q` columns come from an LRU cache of kernel columns
//! bounded by a memory budget, so the full `l × l` Gram matrix is never
//! required.
//!
//! The NU variant (`nu = true`) additionally keeps the two equality
//! constraints of ν-SVC / ν-SVR (`Σ_{y=+1} α = const`, `Σ_{y=−1} α = const`)
//! by only pairing variables with the same `y`.

use ndarray::{Array2, ArrayView1};
use rayon::prelude::*;

use crate::kernel::SvmKernel;

const TAU: f64 = 1e-12;

pub(crate) fn default_true() -> bool {
    true
}

pub(crate) fn default_cache_size() -> f64 {
    200.0
}

/// Where kernel values come from.
pub(crate) enum KernelSource<'a> {
    /// Kernel function on the rows of the training matrix.
    Data {
        x: &'a Array2<f64>,
        kernel: &'a SvmKernel,
    },
    /// A user-supplied (training) Gram matrix.
    Precomputed(&'a Array2<f64>),
}

impl KernelSource<'_> {
    fn n(&self) -> usize {
        match self {
            KernelSource::Data { x, .. } => x.nrows(),
            KernelSource::Precomputed(k) => k.nrows(),
        }
    }

    fn value(&self, i: usize, j: usize) -> f64 {
        match self {
            KernelSource::Data { x, kernel } => kernel.compute(&x.row(i), &x.row(j)),
            KernelSource::Precomputed(k) => k[[i, j]],
        }
    }

    fn column(&self, i: usize) -> Vec<f64> {
        let n = self.n();
        match self {
            KernelSource::Data { x, kernel } => {
                let xi: ArrayView1<f64> = x.row(i);
                if n >= 2048 {
                    (0..n)
                        .into_par_iter()
                        .map(|j| kernel.compute(&xi, &x.row(j)))
                        .collect()
                } else {
                    (0..n).map(|j| kernel.compute(&xi, &x.row(j))).collect()
                }
            }
            KernelSource::Precomputed(k) => k.row(i).to_vec(),
        }
    }
}

/// LRU cache of kernel columns (libsvm's `Cache`), bounded by bytes.
struct KernelCache<'a> {
    source: KernelSource<'a>,
    columns: Vec<Option<Vec<f64>>>,
    /// Least-recently used first.
    order: std::collections::VecDeque<usize>,
    capacity: usize,
}

impl<'a> KernelCache<'a> {
    fn new(source: KernelSource<'a>, cache_size_mb: f64) -> Self {
        let n = source.n();
        let bytes = (cache_size_mb.max(0.0) * 1024.0 * 1024.0) as usize;
        let capacity = (bytes / (n.max(1) * std::mem::size_of::<f64>())).max(2);
        Self {
            source,
            columns: vec![None; n],
            order: std::collections::VecDeque::new(),
            capacity,
        }
    }

    fn column(&mut self, i: usize) -> &[f64] {
        if self.columns[i].is_some() {
            if let Some(pos) = self.order.iter().position(|&c| c == i) {
                self.order.remove(pos);
            }
        } else {
            if self.order.len() >= self.capacity {
                if let Some(old) = self.order.pop_front() {
                    self.columns[old] = None;
                }
            }
            self.columns[i] = Some(self.source.column(i));
        }
        self.order.push_back(i);
        self.columns[i].as_deref().expect("column cached")
    }
}

/// How the solver's `Q` relates to the kernel.
#[derive(Clone, Copy)]
pub(crate) enum QKind {
    /// C-SVC / ν-SVC: `Q_ij = y_i y_j K_ij` over `l` variables.
    Classification,
    /// One-class SVM: `Q_ij = K_ij`.
    OneClass,
    /// ε-SVR / ν-SVR: `2l` variables, `Q_ij = s_i s_j K(i mod l, j mod l)`
    /// with `s = +1` for the first `l` and `−1` for the last `l`.
    Regression,
}

pub(crate) struct QMatrix<'a> {
    kind: QKind,
    cache: KernelCache<'a>,
    /// Labels / signs of the solver variables.
    y: Vec<f64>,
    qd: Vec<f64>,
    l: usize,
}

impl<'a> QMatrix<'a> {
    /// `y` are the solver signs (`len = l` for classification/one-class,
    /// `2l` for regression).
    pub fn new(kind: QKind, source: KernelSource<'a>, y: Vec<f64>, cache_size_mb: f64) -> Self {
        let n = source.n();
        let base_diag: Vec<f64> = (0..n).map(|i| source.value(i, i)).collect();
        let l = y.len();
        let qd = (0..l).map(|k| base_diag[k % n]).collect();
        Self {
            kind,
            cache: KernelCache::new(source, cache_size_mb),
            y,
            qd,
            l,
        }
    }

    fn column(&mut self, i: usize) -> Vec<f64> {
        let n = self.cache.source.n();
        let yi = self.y[i];
        let kind = self.kind;
        let l = self.l;
        let y = &self.y;
        let base = self.cache.column(i % n);
        match kind {
            QKind::Classification => (0..l).map(|k| yi * y[k] * base[k]).collect(),
            QKind::OneClass => base.to_vec(),
            QKind::Regression => (0..l).map(|k| yi * y[k] * base[k % n]).collect(),
        }
    }
}

/// Solver output.
pub(crate) struct SolverResult {
    pub alpha: Vec<f64>,
    pub rho: f64,
    /// ν solver only: the `r` of libsvm's `SolutionInfo` (scale of the ν
    /// problem); 0 otherwise.
    pub r: f64,
}

/// Solver settings.
#[derive(Clone, Copy)]
pub(crate) struct SolverParams {
    pub eps: f64,
    pub shrinking: bool,
    pub max_iter: Option<usize>,
    pub nu: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum Status {
    Lower,
    Upper,
    Free,
}

struct Solver<'q, 'a> {
    q: &'q mut QMatrix<'a>,
    y: Vec<f64>,
    p: Vec<f64>,
    c: Vec<f64>,
    alpha: Vec<f64>,
    status: Vec<Status>,
    g: Vec<f64>,
    g_bar: Vec<f64>,
    /// Active (unshrunk) variables; `active[..active_size]` is the set.
    active: Vec<usize>,
    active_size: usize,
    unshrink: bool,
    eps: f64,
    nu: bool,
}

impl Solver<'_, '_> {
    fn status_of(alpha: f64, c: f64) -> Status {
        if alpha >= c {
            Status::Upper
        } else if alpha <= 0.0 {
            Status::Lower
        } else {
            Status::Free
        }
    }

    fn update_status(&mut self, i: usize) {
        self.status[i] = Self::status_of(self.alpha[i], self.c[i]);
    }

    fn is_upper(&self, i: usize) -> bool {
        self.status[i] == Status::Upper
    }

    fn is_lower(&self, i: usize) -> bool {
        self.status[i] == Status::Lower
    }

    fn is_free(&self, i: usize) -> bool {
        self.status[i] == Status::Free
    }

    fn reconstruct_gradient(&mut self) {
        let l = self.y.len();
        if self.active_size == l {
            return;
        }
        let inactive: Vec<usize> = self.active[self.active_size..].to_vec();
        for &j in &inactive {
            self.g[j] = self.g_bar[j] + self.p[j];
        }
        let free: Vec<usize> = self.active[..self.active_size]
            .iter()
            .copied()
            .filter(|&i| self.is_free(i))
            .collect();
        for i in free {
            let qi = self.q.column(i);
            let ai = self.alpha[i];
            for &j in &inactive {
                self.g[j] += ai * qi[j];
            }
        }
    }

    /// WSS3 (standard solver). `None` when optimal.
    fn select_working_set(&mut self) -> Option<(usize, usize)> {
        let mut gmax = f64::NEG_INFINITY;
        let mut gmax2 = f64::NEG_INFINITY;
        let mut gmax_idx = usize::MAX;
        for &t in &self.active[..self.active_size] {
            if self.y[t] > 0.0 {
                if !self.is_upper(t) && -self.g[t] >= gmax {
                    gmax = -self.g[t];
                    gmax_idx = t;
                }
            } else if !self.is_lower(t) && self.g[t] >= gmax {
                gmax = self.g[t];
                gmax_idx = t;
            }
        }
        let i = gmax_idx;
        let qi = if i != usize::MAX {
            Some(self.q.column(i))
        } else {
            None
        };
        let mut gmin_idx = usize::MAX;
        let mut obj_diff_min = f64::INFINITY;
        for &j in &self.active[..self.active_size] {
            if self.y[j] > 0.0 {
                if !self.is_lower(j) {
                    let grad_diff = gmax + self.g[j];
                    if self.g[j] >= gmax2 {
                        gmax2 = self.g[j];
                    }
                    if grad_diff > 0.0 {
                        let qi = qi.as_ref().expect("i selected");
                        let quad = self.q.qd[i] + self.q.qd[j] - 2.0 * self.y[i] * qi[j];
                        let obj = -(grad_diff * grad_diff) / if quad > 0.0 { quad } else { TAU };
                        if obj <= obj_diff_min {
                            gmin_idx = j;
                            obj_diff_min = obj;
                        }
                    }
                }
            } else if !self.is_upper(j) {
                let grad_diff = gmax - self.g[j];
                if -self.g[j] >= gmax2 {
                    gmax2 = -self.g[j];
                }
                if grad_diff > 0.0 {
                    let qi = qi.as_ref().expect("i selected");
                    let quad = self.q.qd[i] + self.q.qd[j] + 2.0 * self.y[i] * qi[j];
                    let obj = -(grad_diff * grad_diff) / if quad > 0.0 { quad } else { TAU };
                    if obj <= obj_diff_min {
                        gmin_idx = j;
                        obj_diff_min = obj;
                    }
                }
            }
        }
        if gmax + gmax2 < self.eps || gmin_idx == usize::MAX {
            None
        } else {
            Some((gmax_idx, gmin_idx))
        }
    }

    /// WSS3 for the ν solver (pairs share the same `y`).
    fn select_working_set_nu(&mut self) -> Option<(usize, usize)> {
        let (mut gmaxp, mut gmaxp2, mut gmaxp_idx) =
            (f64::NEG_INFINITY, f64::NEG_INFINITY, usize::MAX);
        let (mut gmaxn, mut gmaxn2, mut gmaxn_idx) =
            (f64::NEG_INFINITY, f64::NEG_INFINITY, usize::MAX);
        for &t in &self.active[..self.active_size] {
            if self.y[t] > 0.0 {
                if !self.is_upper(t) && -self.g[t] >= gmaxp {
                    gmaxp = -self.g[t];
                    gmaxp_idx = t;
                }
            } else if !self.is_lower(t) && self.g[t] >= gmaxn {
                gmaxn = self.g[t];
                gmaxn_idx = t;
            }
        }
        let qip = (gmaxp_idx != usize::MAX).then(|| self.q.column(gmaxp_idx));
        let qin = (gmaxn_idx != usize::MAX).then(|| self.q.column(gmaxn_idx));
        let mut gmin_idx = usize::MAX;
        let mut obj_diff_min = f64::INFINITY;
        for &j in &self.active[..self.active_size] {
            if self.y[j] > 0.0 {
                if !self.is_lower(j) {
                    let grad_diff = gmaxp + self.g[j];
                    if self.g[j] >= gmaxp2 {
                        gmaxp2 = self.g[j];
                    }
                    if grad_diff > 0.0 {
                        let q = qip.as_ref().expect("ip selected");
                        let quad = self.q.qd[gmaxp_idx] + self.q.qd[j] - 2.0 * q[j];
                        let obj = -(grad_diff * grad_diff) / if quad > 0.0 { quad } else { TAU };
                        if obj <= obj_diff_min {
                            gmin_idx = j;
                            obj_diff_min = obj;
                        }
                    }
                }
            } else if !self.is_upper(j) {
                let grad_diff = gmaxn - self.g[j];
                if -self.g[j] >= gmaxn2 {
                    gmaxn2 = -self.g[j];
                }
                if grad_diff > 0.0 {
                    let q = qin.as_ref().expect("in selected");
                    let quad = self.q.qd[gmaxn_idx] + self.q.qd[j] - 2.0 * q[j];
                    let obj = -(grad_diff * grad_diff) / if quad > 0.0 { quad } else { TAU };
                    if obj <= obj_diff_min {
                        gmin_idx = j;
                        obj_diff_min = obj;
                    }
                }
            }
        }
        if (gmaxp + gmaxp2).max(gmaxn + gmaxn2) < self.eps || gmin_idx == usize::MAX {
            return None;
        }
        let i = if self.y[gmin_idx] > 0.0 {
            gmaxp_idx
        } else {
            gmaxn_idx
        };
        Some((i, gmin_idx))
    }

    fn be_shrunk(&self, i: usize, gmax1: f64, gmax2: f64) -> bool {
        if self.is_upper(i) {
            if self.y[i] > 0.0 {
                -self.g[i] > gmax1
            } else {
                -self.g[i] > gmax2
            }
        } else if self.is_lower(i) {
            if self.y[i] > 0.0 {
                self.g[i] > gmax2
            } else {
                self.g[i] > gmax1
            }
        } else {
            false
        }
    }

    fn be_shrunk_nu(&self, i: usize, g1: f64, g2: f64, g3: f64, g4: f64) -> bool {
        if self.is_upper(i) {
            if self.y[i] > 0.0 {
                -self.g[i] > g1
            } else {
                -self.g[i] > g4
            }
        } else if self.is_lower(i) {
            if self.y[i] > 0.0 {
                self.g[i] > g2
            } else {
                self.g[i] > g3
            }
        } else {
            false
        }
    }

    fn do_shrinking(&mut self) {
        let l = self.y.len();
        if self.nu {
            let (mut g1, mut g2, mut g3, mut g4) = (
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
            );
            for &i in &self.active[..self.active_size] {
                if !self.is_upper(i) {
                    if self.y[i] > 0.0 {
                        g1 = g1.max(-self.g[i]);
                    } else {
                        g4 = g4.max(-self.g[i]);
                    }
                }
                if !self.is_lower(i) {
                    if self.y[i] > 0.0 {
                        g2 = g2.max(self.g[i]);
                    } else {
                        g3 = g3.max(self.g[i]);
                    }
                }
            }
            if !self.unshrink && (g1 + g2).max(g3 + g4) <= self.eps * 10.0 {
                self.unshrink = true;
                self.reconstruct_gradient();
                self.active_size = l;
            }
            self.shrink_where(|s, i| s.be_shrunk_nu(i, g1, g2, g3, g4));
        } else {
            let (mut g1, mut g2) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
            for &i in &self.active[..self.active_size] {
                if self.y[i] > 0.0 {
                    if !self.is_upper(i) {
                        g1 = g1.max(-self.g[i]);
                    }
                    if !self.is_lower(i) {
                        g2 = g2.max(self.g[i]);
                    }
                } else {
                    if !self.is_upper(i) {
                        g2 = g2.max(-self.g[i]);
                    }
                    if !self.is_lower(i) {
                        g1 = g1.max(self.g[i]);
                    }
                }
            }
            if !self.unshrink && g1 + g2 <= self.eps * 10.0 {
                self.unshrink = true;
                self.reconstruct_gradient();
                self.active_size = l;
            }
            self.shrink_where(|s, i| s.be_shrunk(i, g1, g2));
        }
    }

    fn shrink_where(&mut self, pred: impl Fn(&Self, usize) -> bool) {
        let mut k = 0;
        while k < self.active_size {
            let i = self.active[k];
            if pred(self, i) {
                self.active_size -= 1;
                self.active.swap(k, self.active_size);
            } else {
                k += 1;
            }
        }
    }

    fn update_pair(&mut self, i: usize, j: usize) {
        let qi = self.q.column(i);
        let qj = self.q.column(j);
        let (ci, cj) = (self.c[i], self.c[j]);
        let (old_ai, old_aj) = (self.alpha[i], self.alpha[j]);
        let (mut ai, mut aj) = (old_ai, old_aj);
        if self.y[i] != self.y[j] {
            let mut quad = self.q.qd[i] + self.q.qd[j] + 2.0 * qi[j];
            if quad <= 0.0 {
                quad = TAU;
            }
            let delta = (-self.g[i] - self.g[j]) / quad;
            let diff = ai - aj;
            ai += delta;
            aj += delta;
            if diff > 0.0 {
                if aj < 0.0 {
                    aj = 0.0;
                    ai = diff;
                }
            } else if ai < 0.0 {
                ai = 0.0;
                aj = -diff;
            }
            if diff > ci - cj {
                if ai > ci {
                    ai = ci;
                    aj = ci - diff;
                }
            } else if aj > cj {
                aj = cj;
                ai = cj + diff;
            }
        } else {
            let mut quad = self.q.qd[i] + self.q.qd[j] - 2.0 * qi[j];
            if quad <= 0.0 {
                quad = TAU;
            }
            let delta = (self.g[i] - self.g[j]) / quad;
            let sum = ai + aj;
            ai -= delta;
            aj += delta;
            if sum > ci {
                if ai > ci {
                    ai = ci;
                    aj = sum - ci;
                }
            } else if aj < 0.0 {
                aj = 0.0;
                ai = sum;
            }
            if sum > cj {
                if aj > cj {
                    aj = cj;
                    ai = sum - cj;
                }
            } else if ai < 0.0 {
                ai = 0.0;
                aj = sum;
            }
        }
        self.alpha[i] = ai;
        self.alpha[j] = aj;
        let (dai, daj) = (ai - old_ai, aj - old_aj);
        for &k in &self.active[..self.active_size] {
            self.g[k] += qi[k] * dai + qj[k] * daj;
        }
        // Maintain G_bar for variables entering / leaving the upper bound.
        let (ui, uj) = (self.is_upper(i), self.is_upper(j));
        self.update_status(i);
        self.update_status(j);
        let l = self.y.len();
        if ui != self.is_upper(i) {
            let s = if ui { -ci } else { ci };
            for k in 0..l {
                self.g_bar[k] += s * qi[k];
            }
        }
        if uj != self.is_upper(j) {
            let s = if uj { -cj } else { cj };
            for k in 0..l {
                self.g_bar[k] += s * qj[k];
            }
        }
    }

    fn calculate_rho(&self) -> f64 {
        let (mut nr_free, mut sum_free) = (0usize, 0.0);
        let (mut ub, mut lb) = (f64::INFINITY, f64::NEG_INFINITY);
        for &i in &self.active[..self.active_size] {
            let yg = self.y[i] * self.g[i];
            if self.is_upper(i) {
                if self.y[i] < 0.0 {
                    ub = ub.min(yg);
                } else {
                    lb = lb.max(yg);
                }
            } else if self.is_lower(i) {
                if self.y[i] > 0.0 {
                    ub = ub.min(yg);
                } else {
                    lb = lb.max(yg);
                }
            } else {
                nr_free += 1;
                sum_free += yg;
            }
        }
        if nr_free > 0 {
            sum_free / nr_free as f64
        } else {
            (ub + lb) / 2.0
        }
    }

    /// `(rho, r)` of the ν solver.
    fn calculate_rho_nu(&self) -> (f64, f64) {
        let mut acc = [(0usize, 0.0f64, f64::INFINITY, f64::NEG_INFINITY); 2];
        for &i in &self.active[..self.active_size] {
            let a = &mut acc[usize::from(self.y[i] < 0.0)];
            if self.is_upper(i) {
                a.3 = a.3.max(self.g[i]);
            } else if self.is_lower(i) {
                a.2 = a.2.min(self.g[i]);
            } else {
                a.0 += 1;
                a.1 += self.g[i];
            }
        }
        let r = |(n, s, ub, lb): (usize, f64, f64, f64)| {
            if n > 0 {
                s / n as f64
            } else {
                (ub + lb) / 2.0
            }
        };
        let (r1, r2) = (r(acc[0]), r(acc[1]));
        ((r1 - r2) / 2.0, (r1 + r2) / 2.0)
    }
}

/// Solve the dual. `alpha0` must be feasible; `c` holds the per-variable
/// upper bounds.
pub(crate) fn solve(
    q: &mut QMatrix<'_>,
    p: Vec<f64>,
    alpha0: Vec<f64>,
    c: Vec<f64>,
    params: &SolverParams,
) -> SolverResult {
    let l = p.len();
    let y = q.y.clone();
    let status: Vec<Status> = alpha0
        .iter()
        .zip(&c)
        .map(|(&a, &ci)| Solver::status_of(a, ci))
        .collect();
    let mut g = p.clone();
    let mut g_bar = vec![0.0; l];
    for i in 0..l {
        if status[i] != Status::Lower {
            let qi = q.column(i);
            let ai = alpha0[i];
            for k in 0..l {
                g[k] += ai * qi[k];
            }
            if status[i] == Status::Upper {
                for k in 0..l {
                    g_bar[k] += c[i] * qi[k];
                }
            }
        }
    }
    let mut s = Solver {
        q,
        y,
        p,
        c,
        alpha: alpha0,
        status,
        g,
        g_bar,
        active: (0..l).collect(),
        active_size: l,
        unshrink: false,
        eps: params.eps,
        nu: params.nu,
    };
    let max_iter = params
        .max_iter
        .unwrap_or_else(|| 10_000_000usize.max(l.saturating_mul(100)));
    let mut counter = l.min(1000) + 1;
    let mut iter = 0usize;
    while iter < max_iter {
        counter -= 1;
        if counter == 0 {
            counter = l.min(1000);
            if params.shrinking {
                s.do_shrinking();
            }
        }
        let pick = if s.nu {
            s.select_working_set_nu()
        } else {
            s.select_working_set()
        };
        let (i, j) = match pick {
            Some(pair) => pair,
            None => {
                // Re-check optimality on the whole problem.
                s.reconstruct_gradient();
                s.active_size = l;
                let again = if s.nu {
                    s.select_working_set_nu()
                } else {
                    s.select_working_set()
                };
                match again {
                    Some(pair) => {
                        counter = 1;
                        pair
                    }
                    None => break,
                }
            }
        };
        iter += 1;
        s.update_pair(i, j);
    }
    if s.active_size < l {
        s.reconstruct_gradient();
        s.active_size = l;
    }
    let (rho, r) = if s.nu {
        s.calculate_rho_nu()
    } else {
        (s.calculate_rho(), 0.0)
    };
    SolverResult {
        alpha: s.alpha,
        rho,
        r,
    }
}

// ---------------------------------------------------------------------------
// libsvm formulations
// ---------------------------------------------------------------------------

/// ε-SVR (libsvm `solve_epsilon_svr`). Returns `(α − α*, b)` with
/// `f(x) = Σ (αᵢ − αᵢ*) K(xᵢ, x) + b`.
pub(crate) fn solve_epsilon_svr(
    source: KernelSource<'_>,
    y: &[f64],
    c: f64,
    epsilon: f64,
    params: &SolverParams,
    cache_size_mb: f64,
) -> (Vec<f64>, f64) {
    let l = y.len();
    let signs: Vec<f64> = (0..2 * l).map(|k| if k < l { 1.0 } else { -1.0 }).collect();
    let p: Vec<f64> = (0..2 * l)
        .map(|k| {
            if k < l {
                epsilon - y[k]
            } else {
                epsilon + y[k - l]
            }
        })
        .collect();
    let mut q = QMatrix::new(QKind::Regression, source, signs, cache_size_mb);
    let res = solve(&mut q, p, vec![0.0; 2 * l], vec![c; 2 * l], params);
    let coef = (0..l).map(|i| res.alpha[i] - res.alpha[i + l]).collect();
    (coef, -res.rho)
}

/// ν-SVR (libsvm `solve_nu_svr`). Returns `(α − α*, b, ε)` where `ε` is the
/// tube width found by the ν formulation.
pub(crate) fn solve_nu_svr(
    source: KernelSource<'_>,
    y: &[f64],
    c: f64,
    nu: f64,
    params: &SolverParams,
    cache_size_mb: f64,
) -> (Vec<f64>, f64, f64) {
    let l = y.len();
    let mut sum = c * nu * l as f64 / 2.0;
    let mut alpha0 = vec![0.0; 2 * l];
    for i in 0..l {
        let a = sum.min(c);
        alpha0[i] = a;
        alpha0[i + l] = a;
        sum -= a;
    }
    let signs: Vec<f64> = (0..2 * l).map(|k| if k < l { 1.0 } else { -1.0 }).collect();
    let p: Vec<f64> = (0..2 * l)
        .map(|k| if k < l { -y[k] } else { y[k - l] })
        .collect();
    let mut q = QMatrix::new(QKind::Regression, source, signs, cache_size_mb);
    let params = SolverParams {
        nu: true,
        ..*params
    };
    let res = solve(&mut q, p, alpha0, vec![c; 2 * l], &params);
    let coef = (0..l).map(|i| res.alpha[i] - res.alpha[i + l]).collect();
    (coef, -res.rho, -res.r)
}

/// One-class SVM (libsvm `solve_one_class`). Returns `(α, ρ)` with
/// `0 ≤ αᵢ ≤ 1`, `Σ α = ν l` and decision `Σ αᵢ K(xᵢ, x) − ρ`.
pub(crate) fn solve_one_class(
    source: KernelSource<'_>,
    nu: f64,
    params: &SolverParams,
    cache_size_mb: f64,
) -> (Vec<f64>, f64) {
    let l = source.n();
    let total = nu * l as f64;
    let n_full = total.floor() as usize;
    let mut alpha0 = vec![0.0; l];
    for a in alpha0.iter_mut().take(n_full.min(l)) {
        *a = 1.0;
    }
    if n_full < l {
        alpha0[n_full] = total - n_full as f64;
    }
    let mut q = QMatrix::new(QKind::OneClass, source, vec![1.0; l], cache_size_mb);
    let res = solve(&mut q, vec![0.0; l], alpha0, vec![1.0; l], params);
    (res.alpha, res.rho)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn linear_separable_c_svc_dual() {
        // Two points per class on a line; hard-margin-ish with C = 10.
        let x = array![[-2.0], [-1.0], [1.0], [2.0]];
        let y = vec![-1.0, -1.0, 1.0, 1.0];
        let kernel = SvmKernel::Linear;
        let mut q = QMatrix::new(
            QKind::Classification,
            KernelSource::Data {
                x: &x,
                kernel: &kernel,
            },
            y.clone(),
            1.0,
        );
        let res = solve(
            &mut q,
            vec![-1.0; 4],
            vec![0.0; 4],
            vec![10.0; 4],
            &SolverParams {
                eps: 1e-6,
                shrinking: true,
                max_iter: None,
                nu: false,
            },
        );
        // Margin support vectors x = ±1 with α = 0.5 each, w = 1, b = 0.
        assert!((res.alpha[1] - 0.5).abs() < 1e-6);
        assert!((res.alpha[2] - 0.5).abs() < 1e-6);
        assert!(res.alpha[0].abs() < 1e-9 && res.alpha[3].abs() < 1e-9);
        assert!(res.rho.abs() < 1e-6);
        let sum: f64 = res.alpha.iter().zip(&y).map(|(a, y)| a * y).sum();
        assert!(sum.abs() < 1e-9);
    }

    #[test]
    fn tiny_cache_gives_same_solution() {
        let x = Array2::from_shape_fn((40, 2), |(i, j)| ((i * 7 + j * 3) % 11) as f64 / 3.0);
        let y: Vec<f64> = (0..40)
            .map(|i| if (i * 5) % 7 < 3 { 1.0 } else { -1.0 })
            .collect();
        let kernel = SvmKernel::rbf(0.5);
        let run = |mb: f64, shrinking: bool| {
            let mut q = QMatrix::new(
                QKind::Classification,
                KernelSource::Data {
                    x: &x,
                    kernel: &kernel,
                },
                y.clone(),
                mb,
            );
            solve(
                &mut q,
                vec![-1.0; 40],
                vec![0.0; 40],
                vec![1.0; 40],
                &SolverParams {
                    eps: 1e-8,
                    shrinking,
                    max_iter: None,
                    nu: false,
                },
            )
        };
        let a = run(100.0, true);
        let b = run(0.0, false);
        for (u, v) in a.alpha.iter().zip(&b.alpha) {
            assert!((u - v).abs() < 1e-6);
        }
        assert!((a.rho - b.rho).abs() < 1e-6);
    }
}
