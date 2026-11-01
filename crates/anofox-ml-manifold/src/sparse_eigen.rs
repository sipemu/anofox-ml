//! Bottom eigenpairs of a sparse symmetric positive semi-definite matrix
//! (the LLE matrix `(I − W)ᵀ(I − W)`, graph Laplacians).
//!
//! Small problems use a dense symmetric eigendecomposition. Larger ones use
//! shift-invert Lanczos, as sklearn does through ARPACK with `sigma`: faer's
//! thick-restart Lanczos finds the largest eigenvalues of `(M + εI)⁻¹`,
//! applied through a sparse Cholesky factorisation, which are the smallest
//! of `M`. Costs scale with the factorisation's fill-in rather than n³.

use anofox_ml_core::{Result, RustMlError};
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::solvers::{SelfAdjointEigen, Solve};
use faer::matrix_free::eigen::{
    partial_eigen_scratch, partial_self_adjoint_eigen, PartialEigenParams,
};
use faer::matrix_free::LinOp;
use faer::sparse::linalg::solvers::{Llt, SymbolicLlt};
use faer::sparse::{SparseColMat, Triplet};
use faer::{Col, Mat, MatMut, MatRef, Par, Side};
use ndarray::{Array1, Array2};

/// Up to this size the dense eigendecomposition is used.
pub const DENSE_LIMIT: usize = 500;

/// `(M + εI)⁻¹` through a sparse Cholesky factorisation.
#[derive(Debug)]
struct ShiftInvert {
    llt: Llt<usize, f64>,
    n: usize,
}

impl LinOp<f64> for ShiftInvert {
    fn apply_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
        StackReq::EMPTY
    }
    fn nrows(&self) -> usize {
        self.n
    }
    fn ncols(&self) -> usize {
        self.n
    }
    fn apply(&self, out: MatMut<'_, f64>, rhs: MatRef<'_, f64>, _par: Par, _stack: &mut MemStack) {
        let mut out = out;
        out.copy_from(rhs);
        self.llt.solve_in_place(out);
    }
    fn conj_apply(
        &self,
        out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        stack: &mut MemStack,
    ) {
        self.apply(out, rhs, par, stack);
    }
}

/// The `k` smallest eigenvalues (ascending) and their eigenvectors (as
/// columns) of the symmetric `n × n` matrix given by `entries` (both
/// triangles; repeated entries are summed).
pub fn smallest_eigenpairs(
    n: usize,
    entries: &[(usize, usize, f64)],
    k: usize,
) -> Result<(Array1<f64>, Array2<f64>)> {
    if k == 0 || k > n {
        return Err(RustMlError::InvalidParameter(format!(
            "need 1 <= k <= n, got k = {k}, n = {n}"
        )));
    }
    if n <= DENSE_LIMIT || 3 * k >= n {
        return dense(n, entries, k);
    }
    shift_invert(n, entries, k).or_else(|_| dense(n, entries, k))
}

fn dense(
    n: usize,
    entries: &[(usize, usize, f64)],
    k: usize,
) -> Result<(Array1<f64>, Array2<f64>)> {
    let mut m = Mat::<f64>::zeros(n, n);
    for &(i, j, v) in entries {
        m[(i, j)] += v;
    }
    let eig = SelfAdjointEigen::new(m.as_ref(), Side::Lower)
        .map_err(|e| RustMlError::InvalidParameter(format!("eigen failed: {e:?}")))?;
    let (s, u) = (eig.S(), eig.U());
    let vals = Array1::from_shape_fn(k, |c| s.column_vector()[c]);
    let vecs = Array2::from_shape_fn((n, k), |(i, c)| u[(i, c)]);
    Ok((vals, vecs))
}

fn shift_invert(
    n: usize,
    entries: &[(usize, usize, f64)],
    k: usize,
) -> Result<(Array1<f64>, Array2<f64>)> {
    let fail = |what: &str| RustMlError::InvalidParameter(format!("sparse eigensolver: {what}"));
    // Shift relative to the largest diagonal entry: keeps M + εI positive
    // definite in floating point while barely moving the spectrum.
    let mut diag = vec![0.0_f64; n];
    for &(i, j, v) in entries {
        if i == j {
            diag[i] += v;
        }
    }
    let eps = 1e-8 * diag.iter().copied().fold(0.0, f64::max).max(1e-300);
    let triplets: Vec<Triplet<usize, usize, f64>> = entries
        .iter()
        .map(|&(i, j, v)| Triplet::new(i, j, v))
        .chain((0..n).map(|i| Triplet::new(i, i, eps)))
        .collect();
    let a = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets)
        .map_err(|_| fail("bad matrix"))?;
    let symbolic =
        SymbolicLlt::try_new(a.symbolic(), Side::Lower).map_err(|_| fail("symbolic Cholesky"))?;
    let llt = Llt::try_new_with_symbolic(symbolic, a.as_ref(), Side::Lower)
        .map_err(|_| fail("Cholesky"))?;
    let op = ShiftInvert { llt, n };

    // A few extra Ritz vectors speed up convergence; keep the k largest.
    let k_ext = (2 * k + 1).min(n - 1).max(k);
    let params = PartialEigenParams::default();
    let mut vecs = Mat::<f64>::zeros(n, k_ext);
    let mut vals = vec![0.0_f64; k_ext];
    // Deterministic start vector with no special structure.
    let v0 = Col::<f64>::from_fn(n, |i| 1.0 + ((i as f64) * 0.6180339887).fract());
    let mut buf = MemBuffer::new(partial_eigen_scratch(&op, k_ext, Par::Seq, params));
    let info = partial_self_adjoint_eigen(
        vecs.as_mut(),
        &mut vals,
        &op,
        v0.as_ref(),
        1e-12,
        Par::Seq,
        MemStack::new(&mut buf),
        params,
    );
    if info.n_converged_eigen < k {
        return Err(fail("Lanczos did not converge"));
    }

    // Eigenvalues of M from Rayleigh quotients of the converged vectors.
    let mv = |col: usize| -> Vec<f64> {
        let mut out = vec![0.0; n];
        for &(i, j, v) in entries {
            out[i] += v * vecs[(j, col)];
        }
        out
    };
    let mut pairs: Vec<(f64, usize)> = (0..info.n_converged_eigen)
        .map(|c| {
            let m = mv(c);
            let num: f64 = (0..n).map(|i| vecs[(i, c)] * m[i]).sum();
            let den: f64 = (0..n).map(|i| vecs[(i, c)] * vecs[(i, c)]).sum();
            (num / den, c)
        })
        .collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let vals = Array1::from_shape_fn(k, |c| pairs[c].0);
    let vecs = Array2::from_shape_fn((n, k), |(i, c)| vecs[(i, pairs[c].1)]);
    Ok((vals, vecs))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Path-graph Laplacian: known spectrum 2 - 2 cos(πj/n).
    #[test]
    fn shift_invert_matches_dense_on_path_laplacian() {
        let n = 800;
        let mut e = Vec::new();
        for i in 0..n {
            let deg = if i == 0 || i == n - 1 { 1.0 } else { 2.0 };
            e.push((i, i, deg));
            if i + 1 < n {
                e.push((i, i + 1, -1.0));
                e.push((i + 1, i, -1.0));
            }
        }
        let (sv, svec) = shift_invert(n, &e, 4).unwrap();
        let (dv, dvec) = dense(n, &e, 4).unwrap();
        for c in 0..4 {
            let exact = 2.0 - 2.0 * (std::f64::consts::PI * c as f64 / n as f64).cos();
            assert!((sv[c] - exact).abs() < 1e-10, "{c}: {} vs {exact}", sv[c]);
            assert!((dv[c] - exact).abs() < 1e-10);
            // Same eigenvector up to sign.
            let dot: f64 = (0..n).map(|i| svec[[i, c]] * dvec[[i, c]]).sum();
            assert!((dot.abs() - 1.0).abs() < 1e-6, "{c}: {dot}");
        }
    }
}
