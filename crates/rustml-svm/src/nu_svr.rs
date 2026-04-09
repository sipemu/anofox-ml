//! Nu-Support Vector Regression (nu-SVR).
//!
//! Direct port of libsvm's nu-SVR SMO (Chang & Lin 2011, "LIBSVM: A
//! Library for Support Vector Machines", and Chang & Lin 2002,
//! "Training ν-support vector classifiers: Theory and Algorithms").
//!
//! Solves the nu-SVR dual in libsvm's parameterization:
//!
//! ```text
//! min_β  (1/2) β^T Q β + p^T β
//!  s.t.  z^T β = 0                (bias constraint)
//!        e^T β = C · ν · ℓ        (nu constraint, libsvm convention)
//!        0 ≤ β_i ≤ C              (box)
//! ```
//!
//! where `β = [α_1, α_1*, …, α_ℓ, α_ℓ*]` is a 2ℓ-dimensional vector,
//! `z_i = +1` for α slots and `-1` for α* slots, `Q[i,j] = z_i z_j
//! K(x_{⌊i/2⌋}, x_{⌊j/2⌋})`, and `p_i = -z_i y_{⌊i/2⌋}`.
//!
//! The SMO working set selection restricts pairs to same-side (same
//! `z_i`) to keep both equality constraints satisfied. Within each side
//! the standard libsvm WSS criterion `-y·G` applies (where here
//! `y = z`), and the 2-variable update uses the closed-form Newton step
//! clipped to the box.

use ndarray::{Array1, Array2};
use rustml_core::{Fit, Float, Predict, Result, RustMlError};

use crate::kernel::SvmKernel;
use crate::svr;

/// Nu-Support Vector Regressor (unfitted state).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NuSvr {
    /// Nu parameter in (0, 1]. Upper bound on fraction of margin errors,
    /// lower bound on fraction of support vectors.
    pub nu: f64,
    /// Regularization parameter (box upper bound in the dual).
    pub c: f64,
    /// Kernel function.
    pub kernel: SvmKernel,
    /// Maximum number of SMO iterations.
    pub max_iter: usize,
    /// Stopping tolerance on the max KKT violation.
    pub tol: f64,
}

impl NuSvr {
    pub fn new() -> Self {
        Self {
            nu: 0.5,
            c: 1.0,
            kernel: SvmKernel::Rbf { gamma: 1.0 },
            max_iter: 10000,
            tol: 1e-6,
        }
    }

    pub fn with_nu(mut self, nu: f64) -> Self {
        self.nu = nu;
        self
    }

    pub fn with_c(mut self, c: f64) -> Self {
        self.c = c;
        self
    }

    pub fn with_kernel(mut self, kernel: SvmKernel) -> Self {
        self.kernel = kernel;
        self
    }

    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    fn validate(&self) -> Result<()> {
        if self.nu <= 0.0 || self.nu > 1.0 {
            return Err(RustMlError::InvalidParameter(
                "nu must be in (0, 1]".into(),
            ));
        }
        if self.c <= 0.0 {
            return Err(RustMlError::InvalidParameter("C must be positive".into()));
        }
        if self.max_iter == 0 {
            return Err(RustMlError::InvalidParameter(
                "max_iter must be at least 1".into(),
            ));
        }
        if self.tol <= 0.0 {
            return Err(RustMlError::InvalidParameter(
                "tol must be positive".into(),
            ));
        }
        match &self.kernel {
            SvmKernel::Rbf { gamma } if *gamma <= 0.0 => {
                return Err(RustMlError::InvalidParameter(
                    "gamma must be positive for RBF kernel".into(),
                ));
            }
            SvmKernel::Polynomial { degree, .. } if *degree == 0 => {
                return Err(RustMlError::InvalidParameter(
                    "degree must be at least 1 for polynomial kernel".into(),
                ));
            }
            _ => {}
        }
        Ok(())
    }
}

impl Default for NuSvr {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Nu-Support Vector Regressor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedNuSvr<F: Float> {
    inner: svr::FittedSvr<F>,
}

impl<F: Float> FittedNuSvr<F> {
    pub fn support_vectors(&self) -> &Array2<F> {
        self.inner.support_vectors()
    }

    pub fn n_support(&self) -> usize {
        self.inner.n_support()
    }

    pub fn bias(&self) -> F {
        self.inner.bias()
    }
}

impl<F: Float> Predict<F> for FittedNuSvr<F> {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        self.inner.predict(x)
    }
}

// ---------------------------------------------------------------------------
// nu-SVR SMO (direct port of libsvm's Solver_NU for SVR_NU)
// ---------------------------------------------------------------------------

/// Compute the symmetric kernel Gram matrix `K[i, j] = kernel(x_i, x_j)`.
fn compute_kernel_matrix<F: Float>(x: &Array2<F>, kernel: &SvmKernel) -> Vec<Vec<f64>> {
    let n = x.nrows();
    let mut k = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in i..n {
            let val = kernel.compute(&x.row(i), &x.row(j)).to_f64().unwrap();
            k[i][j] = val;
            k[j][i] = val;
        }
    }
    k
}

/// Solver state for the 2ℓ-variable nu-SVR dual.
struct NuSvrSolver<'a> {
    /// Number of original samples (ℓ).
    ell: usize,
    /// Number of dual variables (= 2ℓ).
    n: usize,
    /// Kernel Gram matrix on original samples, ℓ×ℓ.
    k: &'a [Vec<f64>],
    /// Target values, length ℓ.
    y: &'a [f64],
    /// Box upper bound (= C in libsvm's nu-SVR parameterization).
    upper: f64,
    /// Sum-of-variables target (= C · ν · ℓ in libsvm's convention).
    #[allow(dead_code)]
    c_nu: f64,
    /// Convergence tolerance.
    tol: f64,
    /// Max SMO iterations.
    max_iter: usize,

    /// Dual variables: `β[i] = α_{i}` for i<ℓ, `β[i] = α*_{i-ℓ}` for i≥ℓ.
    beta: Vec<f64>,
    /// Gradient: `g[i] = Σ_j Q[i,j] β_j + p_i`.
    g: Vec<f64>,
}

impl<'a> NuSvrSolver<'a> {
    fn new(k: &'a [Vec<f64>], y: &'a [f64], c: f64, nu: f64, max_iter: usize, tol: f64) -> Self {
        let ell = y.len();
        let n = 2 * ell;
        // In libsvm's nu-SVR parameterization the box upper bound is C
        // (not C/ℓ) and the sum-constraint is C·ν·ℓ (not C·ν).
        let upper = c;
        let mut c_nu = c * nu * ell as f64;
        // Feasibility: the sum constraint c_nu must be ≤ 2 · ℓ · upper = 2Cℓ.
        if c_nu > 2.0 * upper * ell as f64 {
            c_nu = 2.0 * upper * ell as f64;
        }

        // Initialize β to satisfy both equality constraints.
        //   Σ z β = 0: put equal mass in each (α, α*) pair.
        //   Σ β = c_nu: total mass c_nu split evenly across 2ℓ slots.
        // The even-split value is c_nu/(2ℓ); cap each at `upper` and, if
        // that saturates, fill the remaining mass as evenly as possible.
        let mut beta = vec![0.0_f64; n];
        let half = c_nu / 2.0;
        let per_slot = (half / ell as f64).min(upper);
        for i in 0..n {
            beta[i] = per_slot;
        }
        // If per-slot was capped by `upper` the sum may be below c_nu — that
        // is only possible when c_nu > 2ℓ·upper, which we've already clamped.

        let mut solver = Self {
            ell,
            n,
            k,
            y,
            upper,
            c_nu,
            tol,
            max_iter,
            beta,
            g: vec![0.0; n],
        };
        solver.compute_initial_gradient();
        solver
    }

    /// `z_i = +1` for i<ℓ (α slot), `-1` for i≥ℓ (α* slot).
    #[inline]
    fn z(&self, i: usize) -> f64 {
        if i < self.ell {
            1.0
        } else {
            -1.0
        }
    }

    /// Linear term `p_i = -z_i · y_{i mod ℓ}`.
    #[inline]
    fn p(&self, i: usize) -> f64 {
        -self.z(i) * self.y[i % self.ell]
    }

    /// Kernel-derived `Q[i, j]`: signed kernel on the underlying sample pair.
    #[inline]
    fn q_entry(&self, i: usize, j: usize) -> f64 {
        self.z(i) * self.z(j) * self.k[i % self.ell][j % self.ell]
    }

    fn compute_initial_gradient(&mut self) {
        for i in 0..self.n {
            let mut s = self.p(i);
            for j in 0..self.n {
                if self.beta[j] != 0.0 {
                    s += self.q_entry(i, j) * self.beta[j];
                }
            }
            self.g[i] = s;
        }
    }

    #[inline]
    fn is_upper_bound(&self, i: usize) -> bool {
        self.beta[i] >= self.upper - 1e-12
    }

    #[inline]
    fn is_lower_bound(&self, i: usize) -> bool {
        self.beta[i] <= 1e-12
    }

    /// Working set selection adapted from libsvm's `Solver_NU::select_working_set`.
    ///
    /// libsvm tracks four extrema (two per side) using the `-y · G` score
    /// where `y` is the (±1) slot label:
    ///
    /// - `Gmaxp` = max over I_up on z=+1 side of `-G`
    /// - `Gmaxn` = max over I_up on z=-1 side of `+G`
    ///
    /// with the sets
    ///
    /// - I_up for z=+1 = `{β_i < upper}`
    /// - I_up for z=-1 = `{β_i > 0}`
    /// - I_low for z=+1 = `{β_i > 0}`
    /// - I_low for z=-1 = `{β_i < upper}`
    fn select_working_set(&self) -> Option<(usize, usize, f64)> {
        // Per-side extrema.
        let mut gmaxp = f64::NEG_INFINITY;
        let mut gmaxp_idx: Option<usize> = None;
        let mut gmaxn = f64::NEG_INFINITY;
        let mut gmaxn_idx: Option<usize> = None;

        for t in 0..self.n {
            if t < self.ell {
                // z = +1 side. I_up = {β < upper}, score = -G.
                if !self.is_upper_bound(t) {
                    let score = -self.g[t];
                    if score > gmaxp {
                        gmaxp = score;
                        gmaxp_idx = Some(t);
                    }
                }
            } else {
                // z = -1 side. I_up = {β > 0}, score = +G.
                if !self.is_lower_bound(t) {
                    let score = self.g[t];
                    if score > gmaxn {
                        gmaxn = score;
                        gmaxn_idx = Some(t);
                    }
                }
            }
        }

        // Second-pass: for each side, find j in I_low on the same side that
        // maximizes the objective decrease.
        let mut gmin_plus = f64::INFINITY;
        let mut obj_diff_plus = f64::INFINITY;
        let mut gmin_plus_idx: Option<usize> = None;

        let mut gmin_minus = f64::INFINITY;
        let mut obj_diff_minus = f64::INFINITY;
        let mut gmin_minus_idx: Option<usize> = None;

        for t in 0..self.n {
            if t < self.ell {
                // z=+1 side. I_low = {β > 0}, score = -G.
                if !self.is_lower_bound(t) {
                    let score = -self.g[t];
                    if score < gmin_plus {
                        gmin_plus = score;
                    }
                    if let Some(i) = gmaxp_idx {
                        let grad_diff = gmaxp - score;
                        if grad_diff > 0.0 {
                            let qii = self.q_entry(i, i);
                            let qjj = self.q_entry(t, t);
                            let qij = self.q_entry(i, t);
                            let eta = (qii + qjj - 2.0 * qij).max(1e-12);
                            let obj_diff = -(grad_diff * grad_diff) / eta;
                            if obj_diff < obj_diff_plus {
                                obj_diff_plus = obj_diff;
                                gmin_plus_idx = Some(t);
                            }
                        }
                    }
                }
            } else {
                // z=-1 side. I_low = {β < upper}, score = +G.
                if !self.is_upper_bound(t) {
                    let score = self.g[t];
                    if score < gmin_minus {
                        gmin_minus = score;
                    }
                    if let Some(i) = gmaxn_idx {
                        let grad_diff = gmaxn - score;
                        if grad_diff > 0.0 {
                            let qii = self.q_entry(i, i);
                            let qjj = self.q_entry(t, t);
                            let qij = self.q_entry(i, t);
                            let eta = (qii + qjj - 2.0 * qij).max(1e-12);
                            let obj_diff = -(grad_diff * grad_diff) / eta;
                            if obj_diff < obj_diff_minus {
                                obj_diff_minus = obj_diff;
                                gmin_minus_idx = Some(t);
                            }
                        }
                    }
                }
            }
        }

        // Global KKT violation across both sides: libsvm's condition is
        // max(gmaxp - gmin_plus, gmaxn - gmin_minus) < tol → converged.
        let viol_plus = if gmaxp.is_finite() && gmin_plus.is_finite() {
            gmaxp - gmin_plus
        } else {
            f64::NEG_INFINITY
        };
        let viol_minus = if gmaxn.is_finite() && gmin_minus.is_finite() {
            gmaxn - gmin_minus
        } else {
            f64::NEG_INFINITY
        };
        let max_viol = viol_plus.max(viol_minus);
        if max_viol < self.tol {
            return None;
        }

        // Choose the side with the larger violation and return its best pair.
        if viol_plus >= viol_minus {
            match (gmaxp_idx, gmin_plus_idx) {
                (Some(i), Some(j)) => Some((i, j, viol_plus)),
                _ => None,
            }
        } else {
            match (gmaxn_idx, gmin_minus_idx) {
                (Some(i), Some(j)) => Some((i, j, viol_minus)),
                _ => None,
            }
        }
    }

    /// Perform a 2-variable update on a same-side pair `(i, j)`.
    ///
    /// The direction depends on the side:
    /// - z=+1 pair: Δβ_i = +ε, Δβ_j = -ε  (from y = z = +1)
    /// - z=-1 pair: Δβ_i = -ε, Δβ_j = +ε  (from y = z = -1)
    ///
    /// In both cases `β_i + β_j` is preserved, so we can parameterize the
    /// update via `sum = old_i + old_j` and the 1D QP along this constraint.
    fn update_pair(&mut self, i: usize, j: usize) {
        let qii = self.q_entry(i, i);
        let qjj = self.q_entry(j, j);
        let qij = self.q_entry(i, j);
        let eta = (qii + qjj - 2.0 * qij).max(1e-12);

        let old_i = self.beta[i];
        let old_j = self.beta[j];

        // Compute the step `delta` such that
        //   Δβ_i = +z_i · δ  (or equivalently:  y_i · δ)
        //   Δβ_j = −z_i · δ
        // subject to Δβ_i + Δβ_j = 0  (so sum is preserved).
        //
        // The 1D QP in δ along this direction is
        //   f(δ) = (1/2) η δ² + (z_i · g_i − z_i · g_j) · δ + const
        //        = (1/2) η δ² + z_i (g_i − g_j) δ + const
        // and its minimum is at δ = −z_i (g_i − g_j) / η.
        //
        // So new β_i = old_i + z_i · δ = old_i − (g_i − g_j) / η (z_i factors cancel)
        //    new β_j = old_j − z_i · δ = old_j + (g_i − g_j) / η
        //
        // which is the same update formula for both sides.
        let delta = (self.g[i] - self.g[j]) / eta;

        // We need new_i + new_j = old_i + old_j (constraint preserved).
        // Using the libsvm formulation, the new values are:
        //   new_i = old_i + (g_j − g_i) / η
        //   new_j = old_j + (g_i − g_j) / η
        let mut new_i = old_i - delta;
        let mut new_j = old_j + delta;

        // Clip to box [0, upper], re-enforcing the sum constraint afterwards.
        let sum = old_i + old_j;
        if new_i < 0.0 {
            new_i = 0.0;
            new_j = sum;
        }
        if new_j < 0.0 {
            new_j = 0.0;
            new_i = sum;
        }
        if new_i > self.upper {
            new_i = self.upper;
            new_j = sum - new_i;
        }
        if new_j > self.upper {
            new_j = self.upper;
            new_i = sum - new_j;
        }

        let di = new_i - old_i;
        let dj = new_j - old_j;
        if di.abs() < 1e-14 && dj.abs() < 1e-14 {
            return;
        }

        self.beta[i] = new_i;
        self.beta[j] = new_j;

        // Update the gradient: g_k += Q[k,i] · Δβ_i + Q[k,j] · Δβ_j.
        for k in 0..self.n {
            self.g[k] += self.q_entry(k, i) * di + self.q_entry(k, j) * dj;
        }
    }

    /// Run the SMO outer loop. Returns the number of iterations taken.
    fn solve(&mut self) -> usize {
        let mut iters = 0;
        for _ in 0..self.max_iter {
            iters += 1;
            match self.select_working_set() {
                Some((i, j, _)) => self.update_pair(i, j),
                None => break,
            }
        }
        iters
    }

    /// Recover `(w, bias)` from the dual solution.
    ///
    /// `w[i] = α_i - α*_i = β[i] - β[ℓ + i]`.
    ///
    /// Bias recovery is a direct port of libsvm's `Solver_NU::calculate_rho`.
    /// At the optimum, free variables on the z=+1 side satisfy
    ///   `g_i = rho + eps = r1`
    /// and on the z=-1 side
    ///   `g_i = eps - rho = r2`
    /// so `rho = (r1 - r2) / 2`. libsvm averages the free-variable gradients
    /// when any free variable exists; otherwise it uses the midpoint of the
    /// bound-variable extrema on that side.
    ///
    /// The SVR prediction is `f(x) = Σ w_i K(x, x_i) - rho`, so we return
    /// `bias = -rho`.
    fn finalize(self) -> (Vec<f64>, f64) {
        let ell = self.ell;
        let w: Vec<f64> = (0..ell).map(|i| self.beta[i] - self.beta[ell + i]).collect();

        // For each side, track:
        //   ub = min G[i] over i with β[i] = 0 (is_lower_bound)
        //   lb = max G[i] over i with β[i] = C (is_upper_bound)
        //   sum_free, nr_free over 0 < β[i] < C
        let mut ub1 = f64::INFINITY;
        let mut lb1 = f64::NEG_INFINITY;
        let mut sum_free1 = 0.0;
        let mut nr_free1 = 0usize;

        let mut ub2 = f64::INFINITY;
        let mut lb2 = f64::NEG_INFINITY;
        let mut sum_free2 = 0.0;
        let mut nr_free2 = 0usize;

        for t in 0..self.n {
            let gt = self.g[t];
            if t < ell {
                if self.is_lower_bound(t) {
                    if gt < ub1 {
                        ub1 = gt;
                    }
                } else if self.is_upper_bound(t) {
                    if gt > lb1 {
                        lb1 = gt;
                    }
                } else {
                    nr_free1 += 1;
                    sum_free1 += gt;
                }
            } else if self.is_lower_bound(t) {
                if gt < ub2 {
                    ub2 = gt;
                }
            } else if self.is_upper_bound(t) {
                if gt > lb2 {
                    lb2 = gt;
                }
            } else {
                nr_free2 += 1;
                sum_free2 += gt;
            }
        }

        let r1 = if nr_free1 > 0 {
            sum_free1 / nr_free1 as f64
        } else {
            0.5 * (ub1 + lb1)
        };
        let r2 = if nr_free2 > 0 {
            sum_free2 / nr_free2 as f64
        } else {
            0.5 * (ub2 + lb2)
        };

        // libsvm: si->rho = (r1 - r2) / 2, and the SVR predictor is
        //   f(x) = Σ w_i K(x, x_i) - rho
        // so bias (the additive constant used by FittedSvr::predict) is -rho.
        let rho = 0.5 * (r1 - r2);
        let bias = -rho;

        (w, bias)
    }
}

fn solve_nu_svr<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    kernel: &SvmKernel,
    c: f64,
    nu: f64,
    max_iter: usize,
    tol: f64,
) -> (Vec<f64>, f64) {
    let k = compute_kernel_matrix(x, kernel);
    let y_vec: Vec<f64> = y.iter().map(|v| v.to_f64().unwrap()).collect();
    let mut solver = NuSvrSolver::new(&k, &y_vec, c, nu, max_iter, tol);
    solver.solve();
    solver.finalize()
}

impl<F: Float> Fit<F> for NuSvr {
    type Fitted = FittedNuSvr<F>;

    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Self::Fitted> {
        self.validate()?;

        if x.is_empty() || y.is_empty() {
            return Err(RustMlError::EmptyInput(
                "training data must not be empty".into(),
            ));
        }
        if x.nrows() != y.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but y has {} elements",
                x.nrows(),
                y.len()
            )));
        }

        let (w, bias) =
            solve_nu_svr::<F>(x, y, &self.kernel, self.c, self.nu, self.max_iter, self.tol);

        // Extract support vectors (non-zero w).
        let sv_threshold = self.c * 1e-8;
        let mut sv_indices: Vec<usize> = (0..x.nrows())
            .filter(|&i| w[i].abs() > sv_threshold)
            .collect();
        if sv_indices.is_empty() {
            sv_indices = (0..x.nrows()).collect();
        }

        let n_features = x.ncols();
        let mut sv_matrix = Array2::<F>::zeros((sv_indices.len(), n_features));
        let mut dual_coefs = Array1::<F>::zeros(sv_indices.len());
        for (sv_pos, &orig_idx) in sv_indices.iter().enumerate() {
            for j in 0..n_features {
                sv_matrix[[sv_pos, j]] = x[[orig_idx, j]];
            }
            dual_coefs[sv_pos] = F::from_f64(w[orig_idx]).unwrap();
        }

        let inner = svr::FittedSvr::from_parts(
            sv_matrix,
            dual_coefs,
            F::from_f64(bias).unwrap(),
            self.kernel.clone(),
            n_features,
        );

        Ok(FittedNuSvr { inner })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_linear_regression() {
        let x = array![
            [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]
        ];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

        let model = NuSvr::new()
            .with_kernel(SvmKernel::Linear)
            .with_c(100.0)
            .with_nu(0.5);
        let fitted: FittedNuSvr<f64> = model.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(*p, *t, epsilon = 4.0);
        }
    }

    #[test]
    fn test_rbf_regression() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]];
        let y = array![1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0];

        let model = NuSvr::new()
            .with_kernel(SvmKernel::Rbf { gamma: 0.1 })
            .with_c(100.0)
            .with_nu(0.5);
        let fitted: FittedNuSvr<f64> = model.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(p.is_finite(), "prediction should be finite, got {}", p);
        }
    }

    #[test]
    fn test_small_nu_fewer_svs() {
        let x = array![
            [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]
        ];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

        let small = NuSvr::new()
            .with_kernel(SvmKernel::Linear)
            .with_c(100.0)
            .with_nu(0.1);
        let fitted_small: FittedNuSvr<f64> = small.fit(&x, &y).unwrap();

        let large = NuSvr::new()
            .with_kernel(SvmKernel::Linear)
            .with_c(100.0)
            .with_nu(0.9);
        let fitted_large: FittedNuSvr<f64> = large.fit(&x, &y).unwrap();

        assert!(
            fitted_small.n_support() <= fitted_large.n_support() + 1,
            "small nu ({} SVs) should have <= SVs than large nu ({} SVs)",
            fitted_small.n_support(),
            fitted_large.n_support()
        );
    }

    #[test]
    fn test_support_vectors_exist() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let model = NuSvr::new()
            .with_kernel(SvmKernel::Linear)
            .with_c(10.0)
            .with_nu(0.5);
        let fitted: FittedNuSvr<f64> = model.fit(&x, &y).unwrap();

        assert!(fitted.n_support() > 0);
        assert!(fitted.n_support() <= x.nrows());
    }

    #[test]
    fn test_constant_target() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![5.0, 5.0, 5.0, 5.0];

        let model = NuSvr::new()
            .with_kernel(SvmKernel::Linear)
            .with_c(1.0)
            .with_nu(0.5);
        let fitted: FittedNuSvr<f64> = model.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert_abs_diff_eq!(p, 5.0, epsilon = 1.0);
        }
    }

    #[test]
    fn test_empty_input_error() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<f64>::zeros(0);

        let model = NuSvr::new();
        let result: Result<FittedNuSvr<f64>> = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_shape_mismatch_fit() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 2.0, 3.0];

        let model = NuSvr::new();
        let result: Result<FittedNuSvr<f64>> = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_shape_mismatch_predict() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 2.0];

        let model = NuSvr::new().with_kernel(SvmKernel::Linear).with_c(10.0);
        let fitted: FittedNuSvr<f64> = model.fit(&x, &y).unwrap();

        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_invalid_nu_zero() {
        let x = array![[1.0], [2.0]];
        let y = array![1.0, 2.0];

        let model = NuSvr::new().with_nu(0.0);
        assert!(Fit::<f64>::fit(&model, &x, &y).is_err());
    }

    #[test]
    fn test_invalid_nu_negative() {
        let x = array![[1.0], [2.0]];
        let y = array![1.0, 2.0];

        let model = NuSvr::new().with_nu(-0.5);
        assert!(Fit::<f64>::fit(&model, &x, &y).is_err());
    }

    #[test]
    fn test_invalid_nu_above_one() {
        let x = array![[1.0], [2.0]];
        let y = array![1.0, 2.0];

        let model = NuSvr::new().with_nu(1.5);
        assert!(Fit::<f64>::fit(&model, &x, &y).is_err());
    }

    #[test]
    fn test_invalid_c() {
        let x = array![[1.0], [2.0]];
        let y = array![1.0, 2.0];

        let model = NuSvr::new().with_c(-1.0);
        assert!(Fit::<f64>::fit(&model, &x, &y).is_err());
    }

    #[test]
    fn test_builder_and_defaults() {
        let model = NuSvr::new()
            .with_nu(0.3)
            .with_c(5.0)
            .with_kernel(SvmKernel::Linear)
            .with_max_iter(500)
            .with_tol(1e-3);
        assert_eq!(model.nu, 0.3);
        assert_eq!(model.c, 5.0);
        assert_eq!(model.max_iter, 500);
        assert_eq!(model.tol, 1e-3);
        assert!(matches!(model.kernel, SvmKernel::Linear));
    }

    #[test]
    fn test_f32_support() {
        let x: Array2<f32> = array![[1.0f32], [2.0], [3.0], [4.0]];
        let y: Array1<f32> = array![2.0f32, 4.0, 6.0, 8.0];

        let model = NuSvr::new()
            .with_kernel(SvmKernel::Linear)
            .with_c(10.0)
            .with_nu(0.5);
        let fitted: FittedNuSvr<f32> = model.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(p.is_finite());
        }
    }
}
