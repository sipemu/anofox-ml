//! sklearn's `randomized_svd` (Halko et al.): a Gaussian test matrix drawn
//! from numpy's `RandomState`, power iterations, then the SVD of the
//! projection, signs fixed by `svd_flip`.
//!
//! sklearn normalises the power iterations by LU; this uses QR. Both keep
//! the span of the iterates, and the result depends only on that span (the
//! SVD of `Qᵀ M` for an orthonormal basis `Q`), so the two agree up to
//! rounding. When `n_components + n_oversamples` reaches the rank, the span
//! is the whole range of `M` and the result is the exact truncated SVD.

use anofox_ml_core::numpy_random::RandomState;
use anofox_ml_core::Result;
use faer::Mat;
use ndarray::{s, Array2};

use crate::sparse_coding::thin_svd;

/// An orthonormal basis of the columns of `a` (thin QR's `Q`).
fn orthonormal(a: &Array2<f64>) -> Array2<f64> {
    let (n, r) = a.dim();
    let m = Mat::<f64>::from_fn(n, r, |i, j| a[[i, j]]);
    let q = m.qr().compute_thin_Q();
    Array2::from_shape_fn((q.nrows(), q.ncols()), |(i, j)| q[(i, j)])
}

/// The sign of each column's largest-magnitude entry (the first one on
/// ties; 0 for a zero column), as numpy's `sign(u[argmax(|u|)])`.
fn column_signs(u: &Array2<f64>) -> Vec<f64> {
    u.columns()
        .into_iter()
        .map(|c| {
            let (mut best, mut best_abs) = (0, f64::NEG_INFINITY);
            for (i, v) in c.iter().enumerate() {
                if v.abs() > best_abs {
                    (best, best_abs) = (i, v.abs());
                }
            }
            let v = c[best];
            if v > 0.0 {
                1.0
            } else if v < 0.0 {
                -1.0
            } else {
                0.0
            }
        })
        .collect()
}

/// sklearn's `randomized_svd(m, k, n_oversamples=10, n_iter="auto",
/// transpose="auto", flip_sign=True, random_state=rng)`: `(U, s, Vᵀ)` with
/// `k` components.
pub(crate) fn randomized_svd(
    m: &Array2<f64>,
    k: usize,
    rng: &mut RandomState,
) -> Result<(Array2<f64>, Vec<f64>, Array2<f64>)> {
    let (n, p) = m.dim();
    let n_random = k + 10;
    let n_iter = if (k as f64) < 0.1 * n.min(p) as f64 {
        7
    } else {
        4
    };
    let transpose = n < p;
    let a = if transpose {
        m.t().to_owned()
    } else {
        m.clone()
    };
    let mut q = Array2::from_shape_simple_fn((a.ncols(), n_random), || rng.standard_normal());
    for _ in 0..n_iter {
        q = orthonormal(&a.dot(&q));
        q = orthonormal(&a.t().dot(&q));
    }
    q = orthonormal(&a.dot(&q));
    let (uhat, sv, mut vt) = thin_svd(&q.t().dot(&a))?;
    let mut u = q.dot(&uhat);
    // svd_flip: u-based for the untransposed problem, v-based otherwise.
    let signs = if transpose {
        column_signs(&vt.t().to_owned())
    } else {
        column_signs(&u)
    };
    for (c, &sg) in signs.iter().enumerate() {
        u.column_mut(c).mapv_inplace(|v| v * sg);
        vt.row_mut(c).mapv_inplace(|v| v * sg);
    }
    let k = k.min(sv.len());
    let s_k = sv[..k].to_vec();
    Ok(if transpose {
        (
            vt.slice(s![..k, ..]).t().to_owned(),
            s_k,
            u.slice(s![.., ..k]).t().to_owned(),
        )
    } else {
        (
            u.slice(s![.., ..k]).to_owned(),
            s_k,
            vt.slice(s![..k, ..]).to_owned(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_rank_projection_is_the_exact_svd() {
        let m = Array2::from_shape_fn((9, 5), |(i, j)| ((i * 3 + j * 7) % 11) as f64 - 4.0);
        let mut rng = RandomState::new(0);
        let (u, s, vt) = randomized_svd(&m, 3, &mut rng).unwrap();
        let (_, exact, _) = thin_svd(&m).unwrap();
        for (a, b) in s.iter().zip(&exact) {
            assert!((a - b).abs() < 1e-10);
        }
        // U diag(s) Vᵀ is the best rank-3 approximation: its error is the
        // remaining singular values.
        let approx = (&u * &ndarray::Array1::from(s.clone())).dot(&vt);
        let err = (&m - &approx).mapv(|v| v * v).sum().sqrt();
        let rest = exact[3..].iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((err - rest).abs() < 1e-9);
    }
}
