//! t-distributed Stochastic Neighbor Embedding.
//!
//! Mirrors `sklearn.manifold.TSNE` (van der Maaten & Hinton 2008; Barnes–Hut
//! after van der Maaten 2014):
//!
//! 1. Conditional affinities `p_{j|i} ∝ exp(-β_i d²_ij)` with `β_i` found by
//!    binary search so each row has the requested perplexity. `Exact` uses
//!    all pairs; `BarnesHut` only the `⌊3·perplexity + 1⌋` nearest
//!    neighbours (from the shared neighbour index), giving a sparse P.
//!    Squared distances enter the search in single precision, as in sklearn.
//! 2. Joint `P = (P_cond + P_condᵀ) / sum`.
//! 3. Initial embedding: PCA scaled to a first-coordinate standard deviation
//!    of 1e-4 (default), N(0, 1e-4²), or a given array.
//! 4. Gradient descent on KL(P‖Q) with Student-t `Q` (`max(n_components-1,1)`
//!    degrees of freedom), momentum and per-coordinate gains (+0.2 / ×0.8,
//!    minimum 0.01): 250 exploration iterations with momentum 0.5 and `P`
//!    multiplied by `early_exaggeration`, then momentum 0.8 up to `max_iter`.
//!    Every 50 iterations the KL divergence is checked; optimisation stops
//!    without progress for `n_iter_without_progress` iterations (250 in the
//!    exploration phase) or once the gradient norm drops below
//!    `min_grad_norm`.
//!
//! The exact gradient costs O(n²) per iteration. Barnes–Hut summarises the
//! repulsive forces with a 2^d-tree (quad- / octree, `n_components ≤ 3`) and
//! the angle `theta`, for O(n log n) per iteration.
//!
//! sklearn keeps the embedding in single precision for PCA and random
//! initialisation and computes the Barnes–Hut forces in single precision;
//! this implementation works in double precision throughout.

use anofox_ml_core::{FitUnsupervised, Result, RustMlError};
use anofox_ml_neighbors::{DistanceMetric, NeighborAlgorithm, NeighborIndex};
use faer::linalg::solvers::SelfAdjointEigen;
use faer::{Mat, Side};
use ndarray::{Array2, Axis};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

const MACHINE_EPSILON: f64 = f64::EPSILON;
const FLOAT32_TINY: f64 = f32::MIN_POSITIVE as f64;
const EXPLORATION_ITERS: usize = 250;
const N_ITER_CHECK: usize = 50;

/// Gradient algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum TSneMethod {
    /// All pairs, O(n²) per iteration.
    Exact,
    /// Sparse nearest-neighbour P and a space-partitioning tree for the
    /// repulsive forces (needs `n_components ≤ 3`). sklearn's default.
    #[default]
    BarnesHut,
}

/// Initial embedding.
#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub enum TSneInit {
    /// First principal components, scaled so the first coordinate has
    /// standard deviation 1e-4 (sklearn's default).
    #[default]
    Pca,
    /// Independent N(0, 1e-4²) coordinates.
    Random,
    /// A given `(n_samples, n_components)` embedding.
    Array(Array2<f64>),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TSne {
    pub n_components: usize,
    /// Effective number of neighbours; must be below `n_samples`.
    pub perplexity: f64,
    /// `None` is sklearn's `"auto"`: `max(n / early_exaggeration / 4, 50)`.
    pub learning_rate: Option<f64>,
    /// Maximum number of iterations, including the 250 exploration ones.
    #[serde(alias = "n_iter")]
    pub max_iter: usize,
    pub early_exaggeration: f64,
    /// Stop after this many iterations without a lower KL divergence.
    #[serde(default = "default_n_iter_without_progress")]
    pub n_iter_without_progress: usize,
    /// Stop once the gradient norm falls below this.
    #[serde(default = "default_min_grad_norm")]
    pub min_grad_norm: f64,
    #[serde(default)]
    pub init: TSneInit,
    pub seed: u64,
    pub method: TSneMethod,
    /// Barnes–Hut angle: a cell is summarised when width / distance < θ.
    pub theta: f64,
}

fn default_n_iter_without_progress() -> usize {
    300
}
fn default_min_grad_norm() -> f64 {
    1e-7
}

impl TSne {
    /// sklearn's defaults: perplexity 30, early exaggeration 12, automatic
    /// learning rate, 1000 iterations, PCA init, Barnes–Hut with θ = 0.5.
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            perplexity: 30.0,
            learning_rate: None,
            max_iter: 1000,
            early_exaggeration: 12.0,
            n_iter_without_progress: 300,
            min_grad_norm: 1e-7,
            init: TSneInit::Pca,
            seed: 0,
            method: TSneMethod::BarnesHut,
            theta: 0.5,
        }
    }
    pub fn with_perplexity(mut self, p: f64) -> Self {
        self.perplexity = p;
        self
    }
    /// A fixed learning rate (instead of `"auto"`).
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = Some(lr);
        self
    }
    /// Maximum number of iterations.
    pub fn with_n_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }
    pub fn with_early_exaggeration(mut self, e: f64) -> Self {
        self.early_exaggeration = e;
        self
    }
    pub fn with_init(mut self, init: TSneInit) -> Self {
        self.init = init;
        self
    }
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }
    pub fn with_method(mut self, m: TSneMethod) -> Self {
        self.method = m;
        self
    }
    pub fn with_theta(mut self, t: f64) -> Self {
        self.theta = t;
        self
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedTSne {
    pub embedding: Array2<f64>,
    /// KL divergence after optimisation (for Barnes–Hut, the approximation
    /// sklearn reports).
    pub kl_divergence: f64,
    /// Number of iterations run (sklearn's `n_iter_`, zero-based index of
    /// the last iteration).
    pub n_iter: usize,
}

// ───── Affinities ────────────────────────────────────────────────────────

/// sklearn's `_binary_search_perplexity` for one row: `sqd` holds squared
/// distances (already rounded to single precision), `skip` the column of
/// the sample itself in the exact case.
fn conditional_row(sqd: &[f64], skip: Option<usize>, desired_entropy: f64) -> Vec<f64> {
    let (mut beta_min, mut beta_max, mut beta) = (f64::NEG_INFINITY, f64::INFINITY, 1.0_f64);
    let mut p = vec![0.0_f64; sqd.len()];
    for _ in 0..100 {
        let mut sum_p = 0.0;
        for (j, (pj, &d)) in p.iter_mut().zip(sqd).enumerate() {
            if Some(j) != skip {
                *pj = (-d * beta).exp();
                sum_p += *pj;
            }
        }
        if sum_p == 0.0 {
            sum_p = 1e-8;
        }
        let mut sum_dp = 0.0;
        for (pj, &d) in p.iter_mut().zip(sqd) {
            *pj /= sum_p;
            sum_dp += d * *pj;
        }
        let diff = sum_p.ln() + beta * sum_dp - desired_entropy;
        if diff.abs() <= 1e-5 {
            break;
        }
        if diff > 0.0 {
            beta_min = beta;
            beta = if beta_max == f64::INFINITY {
                beta * 2.0
            } else {
                (beta + beta_max) / 2.0
            };
        } else {
            beta_max = beta;
            beta = if beta_min == f64::NEG_INFINITY {
                beta / 2.0
            } else {
                (beta + beta_min) / 2.0
            };
        }
    }
    p
}

/// Squared Euclidean distance rounded to single precision (sklearn casts
/// the distances to float32 before the perplexity search).
fn sq_dist32(a: ndarray::ArrayView1<'_, f64>, b: ndarray::ArrayView1<'_, f64>) -> f64 {
    let d: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    d as f32 as f64
}

/// Dense joint probabilities (exact method), `max(P / sum, ε)`.
fn joint_dense(x: &Array2<f64>, perplexity: f64) -> Array2<f64> {
    let n = x.nrows();
    let desired_entropy = (perplexity as f32 as f64).ln();
    let rows: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let sqd: Vec<f64> = (0..n).map(|j| sq_dist32(x.row(i), x.row(j))).collect();
            conditional_row(&sqd, Some(i), desired_entropy)
        })
        .collect();
    let mut p = Array2::from_shape_fn((n, n), |(i, j)| rows[i][j] + rows[j][i]);
    let sum = p.sum().max(MACHINE_EPSILON);
    p.mapv_inplace(|v| (v / sum).max(MACHINE_EPSILON));
    p.diag_mut().fill(0.0);
    p
}

/// Symmetric sparse P in CSR form.
struct SparseP {
    indptr: Vec<usize>,
    indices: Vec<usize>,
    values: Vec<f64>,
}

/// Sparse joint probabilities from the `k` nearest neighbours (Barnes–Hut).
fn joint_sparse(x: &Array2<f64>, perplexity: f64, k: usize) -> Result<SparseP> {
    let n = x.nrows();
    let desired_entropy = (perplexity as f32 as f64).ln();
    let index = NeighborIndex::new(
        x,
        DistanceMetric::Euclidean,
        NeighborAlgorithm::Auto,
        Some(k),
    )?;
    let (dist, idx) = index.kneighbors_self(k)?;
    // Per row: neighbours sorted by index (sklearn's CSR order).
    let rows: Vec<Vec<(usize, f64)>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut nb: Vec<(usize, f64)> = idx
                .row(i)
                .iter()
                .zip(dist.row(i))
                .map(|(&j, &d)| (j, (d * d) as f32 as f64))
                .collect();
            nb.sort_unstable_by_key(|p| p.0);
            let sqd: Vec<f64> = nb.iter().map(|p| p.1).collect();
            let p = conditional_row(&sqd, None, desired_entropy);
            nb.iter().map(|q| q.0).zip(p).collect()
        })
        .collect();
    // P + Pᵀ.
    let mut triplets: Vec<(usize, usize, f64)> = Vec::with_capacity(2 * n * k);
    for (i, row) in rows.iter().enumerate() {
        for &(j, v) in row {
            triplets.push((i, j, v));
            triplets.push((j, i, v));
        }
    }
    triplets.sort_unstable_by_key(|t| (t.0, t.1));
    let mut indptr = vec![0usize; n + 1];
    let mut indices = Vec::with_capacity(triplets.len());
    let mut values: Vec<f64> = Vec::with_capacity(triplets.len());
    let mut last: Option<(usize, usize)> = None;
    for (i, j, v) in triplets {
        if last == Some((i, j)) {
            *values.last_mut().expect("merged entry") += v;
        } else {
            indices.push(j);
            values.push(v);
            indptr[i + 1] += 1;
            last = Some((i, j));
        }
    }
    for i in 0..n {
        indptr[i + 1] += indptr[i];
    }
    let sum = values.iter().sum::<f64>().max(MACHINE_EPSILON);
    values.iter_mut().for_each(|v| *v /= sum);
    Ok(SparseP {
        indptr,
        indices,
        values,
    })
}

/// Joint probabilities of either method.
enum Affinities {
    Dense(Array2<f64>),
    Sparse(SparseP),
}

impl Affinities {
    /// Multiply by `f` in place (sklearn's `P *= early_exaggeration` and
    /// `P /= early_exaggeration`).
    fn scale(&mut self, f: f64) {
        match self {
            Affinities::Dense(p) => p.mapv_inplace(|v| v * f),
            Affinities::Sparse(p) => p.values.iter_mut().for_each(|v| *v *= f),
        }
    }

    fn divide(&mut self, f: f64) {
        match self {
            Affinities::Dense(p) => p.mapv_inplace(|v| v / f),
            Affinities::Sparse(p) => p.values.iter_mut().for_each(|v| *v /= f),
        }
    }
}

// ───── Initialisation ────────────────────────────────────────────────────

/// PCA projection with sklearn's sign convention (largest |loading| of each
/// component positive), scaled to std 1e-4 in the first coordinate.
fn pca_init(x: &Array2<f64>, k: usize) -> Result<Array2<f64>> {
    let (n, d) = x.dim();
    if k > d.min(n) {
        return Err(RustMlError::InvalidParameter(format!(
            "PCA initialisation needs n_components ({k}) <= min(n_samples, n_features) ({})",
            d.min(n)
        )));
    }
    let mean = x.mean_axis(Axis(0)).expect("non-empty");
    let xc = x - &mean;
    let cov = xc.t().dot(&xc);
    let m = Mat::<f64>::from_fn(d, d, |i, j| cov[[i, j]]);
    let eig = SelfAdjointEigen::new(m.as_ref(), Side::Lower)
        .map_err(|e| RustMlError::InvalidParameter(format!("eigen failed: {e:?}")))?;
    let u = eig.U();
    let mut v = Array2::<f64>::zeros((d, k));
    for c in 0..k {
        let src = d - 1 - c; // eigenvalues ascending
        let col: Vec<f64> = (0..d).map(|r| u[(r, src)]).collect();
        let big = col
            .iter()
            .copied()
            .fold(0.0_f64, |b, val| if val.abs() > b.abs() { val } else { b });
        let sign = if big < 0.0 { -1.0 } else { 1.0 };
        for r in 0..d {
            v[[r, c]] = sign * col[r];
        }
    }
    let mut y = xc.dot(&v);
    let std = y.column(0).std(0.0);
    if std > 0.0 {
        y.mapv_inplace(|val| val / std * 1e-4);
    }
    Ok(y)
}

fn random_init(n: usize, k: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array2::from_shape_fn((n, k), |_| {
        // Box–Muller standard normal.
        let u1: f64 = rng.gen::<f64>().max(f64::MIN_POSITIVE);
        let u2: f64 = rng.gen();
        1e-4 * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    })
}

// ───── Barnes–Hut space-partitioning tree (sklearn's `_QuadTree`) ───────

const BH_EPSILON: f64 = 1e-6;
const NO_CHILD: usize = usize::MAX;

#[derive(Clone)]
struct Cell {
    min: [f64; 3],
    max: [f64; 3],
    center: [f64; 3],
    barycenter: [f64; 3],
    squared_max_width: f64,
    size: usize,
    is_leaf: bool,
    children: [usize; 8],
}

struct BhTree {
    d: usize,
    cells: Vec<Cell>,
}

impl BhTree {
    fn new_cell(d: usize, min: [f64; 3], max: [f64; 3]) -> Cell {
        let mut c = Cell {
            min,
            max,
            center: [0.0; 3],
            barycenter: [0.0; 3],
            squared_max_width: 0.0,
            size: 0,
            is_leaf: true,
            children: [NO_CHILD; 8],
        };
        for a in 0..d {
            c.center[a] = (min[a] + max[a]) / 2.0;
            let w = max[a] - min[a];
            c.squared_max_width = c.squared_max_width.max(w * w);
        }
        c
    }

    fn build(y: &Array2<f64>) -> Self {
        let d = y.ncols();
        let (mut min, mut max) = ([0.0; 3], [0.0; 3]);
        for a in 0..d {
            let col = y.column(a);
            let m = col.iter().copied().fold(f64::INFINITY, f64::min);
            let big = col.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            min[a] = m;
            // Strictly inside the box for positive, negative and small values.
            max[a] = (big * (1.0 + 1e-3 * big.signum())).max(big + 1e-3);
        }
        let mut tree = BhTree {
            d,
            cells: vec![Self::new_cell(d, min, max)],
        };
        for row in y.rows() {
            let mut p = [0.0; 3];
            for a in 0..d {
                p[a] = row[a];
            }
            tree.insert(p, 0);
        }
        tree
    }

    fn child_slot(&self, p: &[f64; 3], cell: usize) -> usize {
        let c = &self.cells[cell];
        (0..self.d).fold(0, |s, a| s * 2 + usize::from(p[a] >= c.center[a]))
    }

    fn new_child(&mut self, p: [f64; 3], parent: usize, size: usize) -> usize {
        let slot = self.child_slot(&p, parent);
        let (pc, mut min, mut max) = (
            self.cells[parent].center,
            self.cells[parent].min,
            self.cells[parent].max,
        );
        for a in 0..self.d {
            if p[a] >= pc[a] {
                min[a] = pc[a];
            } else {
                max[a] = pc[a];
            }
        }
        let mut child = Self::new_cell(self.d, min, max);
        child.barycenter = p;
        child.size = size;
        let id = self.cells.len();
        self.cells.push(child);
        let parent = &mut self.cells[parent];
        parent.is_leaf = false;
        parent.children[slot] = id;
        id
    }

    fn insert(&mut self, p: [f64; 3], cell: usize) {
        let d = self.d;
        let c = &mut self.cells[cell];
        if c.size == 0 {
            c.size = 1;
            c.barycenter = p;
            return;
        }
        if !c.is_leaf {
            let n = c.size as f64;
            for a in 0..d {
                c.barycenter[a] = (n * c.barycenter[a] + p[a]) / (n + 1.0);
            }
            c.size += 1;
            let slot = self.child_slot(&p, cell);
            match self.cells[cell].children[slot] {
                NO_CHILD => {
                    self.new_child(p, cell, 1);
                }
                child => self.insert(p, child),
            }
            return;
        }
        if (0..d).all(|a| (p[a] - c.barycenter[a]).abs() <= BH_EPSILON) {
            c.size += 1;
            return;
        }
        // Split the leaf: move its points into a child, then insert again.
        let (bc, size) = (c.barycenter, c.size);
        self.new_child(bc, cell, size);
        self.insert(p, cell);
    }

    /// Negative force numerator and Σ size·q of point `p` (sklearn's
    /// `summarize` + `compute_gradient_negative`, one degree of freedom
    /// `dof`).
    fn negative(&self, p: &[f64], theta_sq: f64, dof: f64, out: &mut [f64]) -> f64 {
        let d = self.d;
        let exponent = (dof + 1.0) / 2.0;
        let mut sum_q = 0.0;
        let mut stack = vec![0usize];
        out.fill(0.0);
        while let Some(id) = stack.pop() {
            let c = &self.cells[id];
            let mut delta = [0.0; 3];
            let mut dist2 = 0.0;
            let mut duplicate = true;
            for a in 0..d {
                delta[a] = p[a] - c.barycenter[a];
                dist2 += delta[a] * delta[a];
                duplicate &= delta[a].abs() <= BH_EPSILON;
            }
            if duplicate && c.is_leaf {
                continue;
            }
            if c.is_leaf || c.squared_max_width / dist2 < theta_sq {
                let mut q = dof / (dof + dist2);
                if dof != 1.0 {
                    q = q.powf(exponent);
                }
                let size = c.size as f64;
                sum_q += size * q;
                let mult = size * q * q;
                for a in 0..d {
                    out[a] += mult * delta[a];
                }
            } else {
                // Children in reverse so they pop in sklearn's order.
                for &ch in c.children[..1 << d].iter().rev() {
                    if ch != NO_CHILD {
                        stack.push(ch);
                    }
                }
            }
        }
        sum_q
    }
}

// ───── Objectives ────────────────────────────────────────────────────────

/// Exact KL divergence and gradient (sklearn's `_kl_divergence`).
fn objective_exact(
    p: &Array2<f64>,
    y: &Array2<f64>,
    dof: f64,
    compute_error: bool,
) -> (f64, Array2<f64>) {
    let (n, k) = y.dim();
    let exponent = (dof + 1.0) / -2.0;
    let w = |i: usize, j: usize| -> f64 {
        let d2: f64 = (0..k).map(|c| (y[[i, c]] - y[[j, c]]).powi(2)).sum();
        (d2 / dof + 1.0).powf(exponent)
    };
    // Σ over i<j of the unnormalised Student-t kernel.
    let half_sum: f64 = (0..n)
        .into_par_iter()
        .map(|i| ((i + 1)..n).map(|j| w(i, j)).sum::<f64>())
        .sum();
    let denom = 2.0 * half_sum;
    let c = 2.0 * (dof + 1.0) / dof;
    let rows: Vec<(Vec<f64>, f64)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut g = vec![0.0; k];
            let mut kl = 0.0;
            for j in 0..n {
                if j == i {
                    continue;
                }
                let wij = w(i, j);
                let q = (wij / denom).max(MACHINE_EPSILON);
                let pij = p[[i, j]];
                if compute_error && j > i {
                    kl += pij * (pij.max(MACHINE_EPSILON) / q).ln();
                }
                let f = (pij - q) * wij;
                for a in 0..k {
                    g[a] += f * (y[[i, a]] - y[[j, a]]);
                }
            }
            (g, kl)
        })
        .collect();
    let mut grad = Array2::zeros((n, k));
    let mut kl = 0.0;
    for (i, (g, e)) in rows.into_iter().enumerate() {
        for a in 0..k {
            grad[[i, a]] = c * g[a];
        }
        kl += e;
    }
    (2.0 * kl, grad)
}

/// Barnes–Hut KL approximation and gradient (sklearn's `_kl_divergence_bh`).
fn objective_bh(
    p: &SparseP,
    y: &Array2<f64>,
    dof: f64,
    theta: f64,
    compute_error: bool,
) -> (f64, Array2<f64>) {
    let (n, k) = y.dim();
    let tree = BhTree::build(y);
    let theta_sq = theta * theta;
    let neg: Vec<(Vec<f64>, f64)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut f = vec![0.0; k];
            let row = y.row(i);
            let q = tree.negative(row.as_slice().expect("row-major"), theta_sq, dof, &mut f);
            (f, q)
        })
        .collect();
    let sum_q: f64 = neg.iter().map(|v| v.1).sum();
    let exponent = (dof + 1.0) / 2.0;
    let pos: Vec<(Vec<f64>, f64)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut f = vec![0.0; k];
            let mut err = 0.0;
            let mut buff = vec![0.0; k];
            for e in p.indptr[i]..p.indptr[i + 1] {
                let j = p.indices[e];
                let pij = p.values[e];
                let mut dij = 0.0;
                for a in 0..k {
                    buff[a] = y[[i, a]] - y[[j, a]];
                    dij += buff[a] * buff[a];
                }
                let mut qij = dof / (dof + dij);
                if dof != 1.0 {
                    qij = qij.powf(exponent);
                }
                let m = pij * qij;
                if compute_error {
                    let qn = qij / sum_q;
                    err += pij * (pij.max(FLOAT32_TINY) / qn.max(FLOAT32_TINY)).ln();
                }
                for a in 0..k {
                    f[a] += m * buff[a];
                }
            }
            (f, err)
        })
        .collect();
    let c = 2.0 * (dof + 1.0) / dof;
    let mut grad = Array2::zeros((n, k));
    let mut err = 0.0;
    for i in 0..n {
        for a in 0..k {
            grad[[i, a]] = c * (pos[i].0[a] - neg[i].0[a] / sum_q);
        }
        err += pos[i].1;
    }
    (err, grad)
}

// ───── Optimiser (sklearn's `_gradient_descent`) ────────────────────────

struct DescentArgs {
    start: usize,
    max_iter: usize,
    momentum: f64,
    learning_rate: f64,
    n_iter_without_progress: usize,
    min_grad_norm: f64,
}

/// Runs iterations `start..max_iter`; returns (KL, last iteration).
fn gradient_descent(
    y: &mut Array2<f64>,
    args: &DescentArgs,
    objective: &dyn Fn(&Array2<f64>, bool) -> (f64, Array2<f64>),
) -> (f64, usize) {
    let mut update = Array2::<f64>::zeros(y.raw_dim());
    let mut gains = Array2::<f64>::ones(y.raw_dim());
    let mut error = f64::MAX;
    let mut best_error = f64::MAX;
    let mut best_iter = args.start;
    let mut i = args.start;
    while i < args.max_iter {
        let check = (i + 1) % N_ITER_CHECK == 0;
        let (e, mut grad) = objective(y, check || i + 1 == args.max_iter);
        error = e;
        ndarray::Zip::from(&mut gains)
            .and(&update)
            .and(&grad)
            .for_each(|g, &u, &gr| {
                *g = if u * gr < 0.0 { *g + 0.2 } else { *g * 0.8 };
                *g = g.max(0.01);
            });
        grad *= &gains;
        update = &update * args.momentum - &grad * args.learning_rate;
        *y += &update;
        if check {
            let grad_norm = grad.iter().map(|v| v * v).sum::<f64>().sqrt();
            if error < best_error {
                best_error = error;
                best_iter = i;
            } else if i - best_iter > args.n_iter_without_progress {
                break;
            }
            if grad_norm <= args.min_grad_norm {
                break;
            }
        }
        i += 1;
    }
    (
        error,
        i.min(args.max_iter.saturating_sub(1)).max(args.start),
    )
}

impl FitUnsupervised<f64> for TSne {
    type Fitted = FittedTSne;

    fn fit(&self, x: &Array2<f64>) -> Result<Self::Fitted> {
        let n = x.nrows();
        let k = self.n_components;
        if n < 2 {
            return Err(RustMlError::EmptyInput("need ≥ 2 samples".into()));
        }
        if k == 0 {
            return Err(RustMlError::InvalidParameter("n_components >= 1".into()));
        }
        if !(self.perplexity > 0.0 && self.perplexity < n as f64) {
            return Err(RustMlError::InvalidParameter(format!(
                "perplexity ({}) must be in (0, n_samples = {n})",
                self.perplexity
            )));
        }
        if self.early_exaggeration < 1.0 {
            return Err(RustMlError::InvalidParameter(
                "early_exaggeration must be >= 1".into(),
            ));
        }
        if self.method == TSneMethod::BarnesHut && k > 3 {
            return Err(RustMlError::InvalidParameter(
                "Barnes-Hut t-SNE needs n_components <= 3; use the exact method".into(),
            ));
        }
        let learning_rate = match self.learning_rate {
            Some(lr) if lr > 0.0 => lr,
            Some(lr) => {
                return Err(RustMlError::InvalidParameter(format!(
                    "learning_rate must be > 0, got {lr}"
                )))
            }
            None => (n as f64 / self.early_exaggeration / 4.0).max(50.0),
        };
        let x = x.as_standard_layout().into_owned();

        let mut y = match &self.init {
            TSneInit::Pca => pca_init(&x, k)?,
            TSneInit::Random => random_init(n, k, self.seed),
            TSneInit::Array(a) => {
                if a.dim() != (n, k) {
                    return Err(RustMlError::ShapeMismatch(format!(
                        "init has shape {:?}, expected ({n}, {k})",
                        a.dim()
                    )));
                }
                a.as_standard_layout().into_owned()
            }
        };

        let dof = (k as f64 - 1.0).max(1.0);
        let mut p = match self.method {
            TSneMethod::Exact => Affinities::Dense(joint_dense(&x, self.perplexity)),
            TSneMethod::BarnesHut => {
                let nn = (n - 1).min((3.0 * self.perplexity + 1.0) as usize);
                Affinities::Sparse(joint_sparse(&x, self.perplexity, nn)?)
            }
        };
        let theta = self.theta;
        let run = |p: &Affinities, y: &mut Array2<f64>, args: &DescentArgs| {
            gradient_descent(y, args, &|y: &Array2<f64>, err: bool| match p {
                Affinities::Dense(p) => objective_exact(p, y, dof, err),
                Affinities::Sparse(p) => objective_bh(p, y, dof, theta, err),
            })
        };

        // Exploration with early exaggeration and momentum 0.5.
        p.scale(self.early_exaggeration);
        let explore = DescentArgs {
            start: 0,
            max_iter: EXPLORATION_ITERS,
            momentum: 0.5,
            learning_rate,
            n_iter_without_progress: EXPLORATION_ITERS,
            min_grad_norm: self.min_grad_norm,
        };
        let (mut kl, mut it) = run(&p, &mut y, &explore);
        p.divide(self.early_exaggeration);
        if it < EXPLORATION_ITERS || self.max_iter > EXPLORATION_ITERS {
            let refine = DescentArgs {
                start: it + 1,
                max_iter: self.max_iter,
                momentum: 0.8,
                learning_rate,
                n_iter_without_progress: self.n_iter_without_progress,
                min_grad_norm: self.min_grad_norm,
            };
            (kl, it) = run(&p, &mut y, &refine);
        }

        Ok(FittedTSne {
            embedding: y,
            kl_divergence: kl,
            n_iter: it,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// On two well-separated clusters, t-SNE should put each cluster's points
    /// closer to its own centroid than to the other's.
    #[test]
    fn test_tsne_separates_two_blobs() {
        let n_per = 15;
        let mut x = Array2::<f64>::zeros((2 * n_per, 5));
        for i in 0..n_per {
            // Cluster A around origin, jitter via deterministic LCG.
            let s = (i as u64).wrapping_mul(6364136223846793005).wrapping_add(1);
            x[[i, 0]] = ((s >> 16) as f64 / u64::MAX as f64 - 0.5) * 0.3;
            x[[i, 1]] = (((s.wrapping_mul(13)) >> 16) as f64 / u64::MAX as f64 - 0.5) * 0.3;
            // Cluster B around (10, 10, ...).
            let s2 = ((i + 100) as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            x[[n_per + i, 0]] = 10.0 + ((s2 >> 16) as f64 / u64::MAX as f64 - 0.5) * 0.3;
            x[[n_per + i, 1]] =
                10.0 + (((s2.wrapping_mul(13)) >> 16) as f64 / u64::MAX as f64 - 0.5) * 0.3;
        }
        let fitted = TSne::new(2)
            .with_perplexity(5.0)
            .with_learning_rate(10.0)
            .with_n_iter(500)
            .with_seed(1)
            .fit(&x)
            .unwrap();
        let y = fitted.embedding;
        // Compute centroids.
        let mut ca = [0.0; 2];
        let mut cb = [0.0; 2];
        for i in 0..n_per {
            ca[0] += y[[i, 0]];
            ca[1] += y[[i, 1]];
            cb[0] += y[[n_per + i, 0]];
            cb[1] += y[[n_per + i, 1]];
        }
        for v in ca.iter_mut() {
            *v /= n_per as f64;
        }
        for v in cb.iter_mut() {
            *v /= n_per as f64;
        }
        // Linear-separability test: project each point onto the centroid-
        // difference direction; the two clusters should be on opposite sides
        // of the midpoint.
        let dir = [cb[0] - ca[0], cb[1] - ca[1]];
        let dir_norm = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt().max(1e-12);
        let mid = [(ca[0] + cb[0]) / 2.0, (ca[1] + cb[1]) / 2.0];
        let mut a_correct = 0;
        let mut b_correct = 0;
        for i in 0..n_per {
            let pa = ((y[[i, 0]] - mid[0]) * dir[0] + (y[[i, 1]] - mid[1]) * dir[1]) / dir_norm;
            let pb = ((y[[n_per + i, 0]] - mid[0]) * dir[0]
                + (y[[n_per + i, 1]] - mid[1]) * dir[1])
                / dir_norm;
            if pa < 0.0 {
                a_correct += 1;
            }
            if pb > 0.0 {
                b_correct += 1;
            }
        }
        // 80% of each cluster must land on the correct side of the
        // centroid-midpoint hyperplane. t-SNE on small datasets is finicky;
        // perfect linear separability isn't guaranteed.
        let thr = (n_per * 4) / 5;
        assert!(
            a_correct >= thr && b_correct >= thr,
            "linear-separability failed: A={}, B={}, threshold={}",
            a_correct,
            b_correct,
            thr,
        );
    }

    #[test]
    fn test_tsne_barnes_hut_separates_two_blobs() {
        // Larger dataset where BH actually helps. 60 points × 5 dims.
        let n_per = 30;
        let mut x = Array2::<f64>::zeros((2 * n_per, 5));
        for i in 0..n_per {
            let s = (i as u64).wrapping_mul(6364136223846793005).wrapping_add(1);
            x[[i, 0]] = ((s >> 16) as f64 / u64::MAX as f64 - 0.5) * 0.5;
            x[[i, 1]] = (((s.wrapping_mul(13)) >> 16) as f64 / u64::MAX as f64 - 0.5) * 0.5;
            let s2 = ((i + 200) as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            x[[n_per + i, 0]] = 15.0 + ((s2 >> 16) as f64 / u64::MAX as f64 - 0.5) * 0.5;
            x[[n_per + i, 1]] =
                15.0 + (((s2.wrapping_mul(13)) >> 16) as f64 / u64::MAX as f64 - 0.5) * 0.5;
        }
        let fitted = TSne::new(2)
            .with_perplexity(10.0)
            .with_learning_rate(15.0)
            .with_n_iter(500)
            .with_seed(7)
            .with_method(TSneMethod::BarnesHut)
            .with_theta(0.5)
            .fit(&x)
            .unwrap();
        let y = fitted.embedding;
        let mut ca = [0.0; 2];
        let mut cb = [0.0; 2];
        for i in 0..n_per {
            ca[0] += y[[i, 0]];
            ca[1] += y[[i, 1]];
            cb[0] += y[[n_per + i, 0]];
            cb[1] += y[[n_per + i, 1]];
        }
        for v in ca.iter_mut() {
            *v /= n_per as f64;
        }
        for v in cb.iter_mut() {
            *v /= n_per as f64;
        }
        let dir = [cb[0] - ca[0], cb[1] - ca[1]];
        let dir_norm = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt().max(1e-12);
        let mid = [(ca[0] + cb[0]) / 2.0, (ca[1] + cb[1]) / 2.0];
        let mut a_correct = 0;
        let mut b_correct = 0;
        for i in 0..n_per {
            let pa = ((y[[i, 0]] - mid[0]) * dir[0] + (y[[i, 1]] - mid[1]) * dir[1]) / dir_norm;
            let pb = ((y[[n_per + i, 0]] - mid[0]) * dir[0]
                + (y[[n_per + i, 1]] - mid[1]) * dir[1])
                / dir_norm;
            if pa < 0.0 {
                a_correct += 1;
            }
            if pb > 0.0 {
                b_correct += 1;
            }
        }
        let thr = (n_per * 4) / 5;
        assert!(
            a_correct >= thr && b_correct >= thr,
            "BH linear-separability failed: A={}, B={}, threshold={}",
            a_correct,
            b_correct,
            thr,
        );
    }
}
#[cfg(test)]
mod sklearn_parity {
    use super::*;

    /// The affinities, the first objective and the first ten optimiser
    /// steps reproduce sklearn 1.8 to rounding. Later iterates diverge
    /// chaotically (t-SNE amplifies rounding differences), so full runs are
    /// compared by KL divergence in `tests/tsne_parity.rs`.
    #[test]
    fn affinities_objective_and_first_steps_match_sklearn() {
        let x = Array2::from_shape_fn((40, 4), |(i, j)| {
            let (fi, fj) = (i as f64, j as f64);
            (fi * 0.6180339887 * (fj + 1.0) + fi * fi * 0.0137 * (fj + 3.0)) % 1.0
                + 3.0 * (i % 2) as f64
        });
        let y0 = Array2::from_shape_fn((40, 2), |(i, c)| {
            1e-4 * (((i as f64) * 0.7548776662 * (c as f64 + 1.0)) % 1.0 - 0.5)
        });
        let close =
            |a: f64, b: f64, tol: f64| assert!((a - b).abs() <= tol * b.abs(), "{a} vs {b}");

        // squareform(_joint_probabilities(pairwise_distances(X, squared=True), 8, 0))
        let p = joint_dense(&x, 8.0);
        close(p[[0, 2]], 3.679512701880893e-05, 1e-12);
        close(p[[0, 4]], 0.0004319271399449369, 1e-12);
        close(p[[1, 3]], 0.0001479462931127233, 1e-12);
        assert_eq!(p[[0, 1]], MACHINE_EPSILON);

        // PCA(2, svd_solver="randomized").fit_transform(X).astype(float32),
        // scaled to std 1e-4 in the first coordinate (single precision).
        let init = pca_init(&x, 2).unwrap();
        let expected = [
            -0.00012980977771803737,
            7.479491159756435e-07,
            0.00010744915925897658,
            -1.1319572877255268e-05,
            -9.105139906750992e-05,
            -5.342912118067034e-06,
        ];
        for (a, b) in init.iter().zip(expected) {
            close(*a, b, 1e-6);
        }

        // _kl_divergence(Y0.ravel(), P * 12, 1, 40, 2, compute_error=True)
        let pe = &p * 12.0;
        let (kl, g) = objective_exact(&pe, &y0, 1.0, true);
        close(kl, 48.069535258923395, 1e-12);
        let expected = [
            -2.3823892786931124e-05,
            -9.255942549921132e-06,
            3.9398799882380754e-05,
            -2.8679026457981637e-05,
        ];
        for (a, b) in g.iter().zip(expected) {
            close(*a, b, 1e-12);
        }

        // _gradient_descent(_kl_divergence, Y0, it=0, max_iter=m, momentum=0.5,
        //                   learning_rate=50.0, n_iter_check=50, ...)
        let steps: [(usize, [f64; 3]); 2] = [
            (
                1,
                [
                    0.000902955711477245,
                    0.0003202377019968453,
                    -0.0015504642286752303,
                ],
            ),
            (
                10,
                [3.487223458751857, 0.2239873268397652, -0.5775108711382306],
            ),
        ];
        for (m, first) in steps {
            let mut y = y0.clone();
            let args = DescentArgs {
                start: 0,
                max_iter: m,
                momentum: 0.5,
                learning_rate: 50.0,
                n_iter_without_progress: 250,
                min_grad_norm: 1e-7,
            };
            let (_, it) = gradient_descent(&mut y, &args, &|y: &Array2<f64>, e: bool| {
                objective_exact(&pe, y, 1.0, e)
            });
            assert_eq!(it, m - 1);
            for (a, b) in y.iter().zip(first) {
                close(*a, b, 1e-9);
            }
        }
    }
}
