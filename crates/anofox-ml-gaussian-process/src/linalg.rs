//! Small dense linear-algebra helpers (faer-backed) shared by GPR and GPC.

use anofox_ml_core::{Result, RustMlError};
use faer::linalg::solvers::Llt;
use faer::{Mat, MatRef, Par, Side};
use ndarray::{Array1, Array2};

pub(crate) fn to_mat(a: &Array2<f64>) -> Mat<f64> {
    Mat::from_fn(a.nrows(), a.ncols(), |i, j| a[[i, j]])
}

pub(crate) fn to_array2(m: MatRef<'_, f64>) -> Array2<f64> {
    Array2::from_shape_fn((m.nrows(), m.ncols()), |(i, j)| m[(i, j)])
}

/// Serde for the fitted models' faer matrices (the Cholesky factors), as
/// ndarray arrays: a round trip restores them exactly.
pub(crate) mod mat_serde {
    use super::{to_array2, to_mat};
    use faer::Mat;
    use ndarray::Array2;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(m: &Mat<f64>, s: S) -> Result<S::Ok, S::Error> {
        to_array2(m.as_ref()).serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Mat<f64>, D::Error> {
        Ok(to_mat(&Array2::<f64>::deserialize(d)?))
    }
}

pub(crate) fn col_to_array1(m: &Mat<f64>) -> Array1<f64> {
    Array1::from_shape_fn(m.nrows(), |i| m[(i, 0)])
}

pub(crate) fn vec_to_col(v: &Array1<f64>) -> Mat<f64> {
    Mat::from_fn(v.len(), 1, |i, _| v[i])
}

/// Plain Cholesky of a symmetric matrix (lower side); `None` if not PD.
pub(crate) fn cholesky(k: &Array2<f64>) -> Option<Llt<f64>> {
    Llt::new(to_mat(k).as_ref(), Side::Lower).ok()
}

/// Cholesky with jitter retry: on failure, add `jitter · mean(diag)` to
/// the diagonal with `jitter = 1e-10, 1e-9, …, 1e-5`.
/// Returns the factor and the jitter actually added (0 if none).
pub(crate) fn cholesky_with_jitter(k: &Array2<f64>) -> Result<(Llt<f64>, f64)> {
    if let Some(l) = cholesky(k) {
        return Ok((l, 0.0));
    }
    let n = k.nrows();
    let mean_diag = (0..n).map(|i| k[[i, i]].abs()).sum::<f64>() / n.max(1) as f64;
    let scale = if mean_diag > 0.0 { mean_diag } else { 1.0 };
    let mut jitter = 1e-10;
    while jitter <= 1e-5 * 1.0001 {
        let mut kj = k.clone();
        for i in 0..n {
            kj[[i, i]] += jitter * scale;
        }
        if let Some(l) = cholesky(&kj) {
            return Ok((l, jitter * scale));
        }
        jitter *= 10.0;
    }
    Err(RustMlError::InvalidParameter(
        "kernel matrix is not positive definite (Cholesky failed even with jitter); \
         try increasing alpha or adding a White kernel"
            .into(),
    ))
}

/// Solve `L V = B` for lower-triangular `L` in one batched call.
pub(crate) fn solve_lower(l: MatRef<'_, f64>, mut b: Mat<f64>) -> Mat<f64> {
    faer::linalg::triangular_solve::solve_lower_triangular_in_place(l, b.as_mut(), Par::Seq);
    b
}

/// Box–Muller standard normal.
pub(crate) fn std_normal(rng: &mut impl rand::Rng) -> f64 {
    let u1: f64 = 1.0 - rng.gen::<f64>();
    let u2: f64 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}
