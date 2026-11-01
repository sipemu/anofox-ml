//! Shared building blocks for [`GaussianMixture`](crate::GaussianMixture)
//! and [`BayesianGaussianMixture`](crate::BayesianGaussianMixture), ported
//! from `sklearn.mixture._gaussian_mixture`.
//!
//! # Parameter layout
//!
//! Covariances and precision Cholesky factors are stored as a
//! `Vec<Array2<f64>>` whose shape depends on the [`CovarianceType`]:
//!
//! | type        | entries | each entry |
//! |-------------|---------|------------|
//! | `Full`      | `k`     | `d × d`    |
//! | `Tied`      | `1`     | `d × d`    |
//! | `Diag`      | `k`     | `1 × d`    |
//! | `Spherical` | `k`     | `1 × 1`    |
//!
//! The precision Cholesky factor `U` satisfies `Λ = U Uᵀ` (sklearn's
//! `precisions_cholesky_`): upper triangular for full/tied, `1/σ` for
//! diag/spherical.

use anofox_ml_core::{Result, RustMlError};
use ndarray::{Array1, Array2, Axis};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::gmm::CovarianceType;
use crate::util::{kmeans_plusplus, logsumexp, sample_without_replacement};

/// How the responsibilities are initialised before the first M-step
/// (sklearn's `init_params`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MixtureInit {
    /// Hard assignments from a single [`KMeans`](crate::KMeans) run (sklearn
    /// default).
    #[default]
    KMeans,
    /// Hard assignments to `k` greedy k-means++ seeds.
    KMeansPlusPlus,
    /// Uniform random responsibilities, row-normalised.
    Random,
    /// Hard assignments to `k` random training rows.
    RandomFromData,
}

pub(crate) const CHOLESKY_ERROR: &str =
    "Fitting the mixture model failed because some components have \
     ill-defined empirical covariance (for instance caused by singleton or collapsed samples). \
     Try to decrease the number of components, increase reg_covar, or scale the input data.";

fn chol_err() -> RustMlError {
    RustMlError::InvalidParameter(CHOLESKY_ERROR.into())
}

/// Lower Cholesky factor of an SPD matrix; `None` if not positive definite.
pub(crate) fn cholesky_lower(a: &Array2<f64>) -> Option<Array2<f64>> {
    let n = a.nrows();
    let mut l = Array2::<f64>::zeros((n, n));
    for j in 0..n {
        let mut s = a[[j, j]];
        for p in 0..j {
            s -= l[[j, p]] * l[[j, p]];
        }
        if s <= 0.0 || !s.is_finite() {
            return None;
        }
        let ljj = s.sqrt();
        l[[j, j]] = ljj;
        for i in (j + 1)..n {
            let mut s = a[[i, j]];
            for p in 0..j {
                s -= l[[i, p]] * l[[j, p]];
            }
            l[[i, j]] = s / ljj;
        }
    }
    Some(l)
}

/// Inverse of a lower-triangular matrix (forward substitution).
fn invert_lower(l: &Array2<f64>) -> Array2<f64> {
    let n = l.nrows();
    let mut inv = Array2::<f64>::zeros((n, n));
    for c in 0..n {
        inv[[c, c]] = 1.0 / l[[c, c]];
        for i in (c + 1)..n {
            let mut s = 0.0;
            for p in c..i {
                s += l[[i, p]] * inv[[p, c]];
            }
            inv[[i, c]] = -s / l[[i, i]];
        }
    }
    inv
}

/// Number of covariance blocks for the given type.
pub(crate) fn n_blocks(ct: CovarianceType, k: usize) -> usize {
    if ct == CovarianceType::Tied {
        1
    } else {
        k
    }
}

/// Expected shape of each covariance / precision block.
pub(crate) fn block_shape(ct: CovarianceType, d: usize) -> (usize, usize) {
    match ct {
        CovarianceType::Full | CovarianceType::Tied => (d, d),
        CovarianceType::Diag => (1, d),
        CovarianceType::Spherical => (1, 1),
    }
}

/// Validate a user-provided covariance / precision layout.
pub(crate) fn check_blocks(
    blocks: &[Array2<f64>],
    ct: CovarianceType,
    k: usize,
    d: usize,
    what: &str,
) -> Result<()> {
    let nb = n_blocks(ct, k);
    let shape = block_shape(ct, d);
    if blocks.len() != nb || blocks.iter().any(|b| b.dim() != shape) {
        return Err(RustMlError::ShapeMismatch(format!(
            "{what} for covariance_type {ct:?} must be {nb} block(s) of shape {shape:?}"
        )));
    }
    Ok(())
}

/// `U` with `Λ = U Uᵀ` from covariances (sklearn `_compute_precision_cholesky`).
pub(crate) fn precision_cholesky(
    covariances: &[Array2<f64>],
    ct: CovarianceType,
) -> Result<Vec<Array2<f64>>> {
    match ct {
        CovarianceType::Full | CovarianceType::Tied => covariances
            .par_iter()
            .map(|cov| {
                let l = cholesky_lower(cov).ok_or_else(chol_err)?;
                Ok(invert_lower(&l).reversed_axes())
            })
            .collect(),
        CovarianceType::Diag | CovarianceType::Spherical => covariances
            .iter()
            .map(|cov| {
                if cov.iter().any(|&v| v <= 0.0 || !v.is_finite()) {
                    return Err(chol_err());
                }
                Ok(cov.mapv(|v| 1.0 / v.sqrt()))
            })
            .collect(),
    }
}

/// `U` with `Λ = U Uᵀ` directly from precision matrices
/// (sklearn `_compute_precision_cholesky_from_precisions`): the Cholesky
/// factor of the row/column-reversed precision, reversed back.
pub(crate) fn precision_cholesky_from_precisions(
    precisions: &[Array2<f64>],
    ct: CovarianceType,
) -> Result<Vec<Array2<f64>>> {
    match ct {
        CovarianceType::Full | CovarianceType::Tied => precisions
            .iter()
            .map(|p| {
                let d = p.nrows();
                let flipped = Array2::from_shape_fn((d, d), |(i, j)| p[[d - 1 - i, d - 1 - j]]);
                let l = cholesky_lower(&flipped).ok_or_else(|| {
                    RustMlError::InvalidParameter(
                        "precisions_init must be positive definite".into(),
                    )
                })?;
                Ok(Array2::from_shape_fn((d, d), |(i, j)| {
                    l[[d - 1 - i, d - 1 - j]]
                }))
            })
            .collect(),
        CovarianceType::Diag | CovarianceType::Spherical => precisions
            .iter()
            .map(|p| {
                if p.iter().any(|&v| v <= 0.0 || v.is_nan()) {
                    return Err(RustMlError::InvalidParameter(
                        "precisions_init must be positive".into(),
                    ));
                }
                Ok(p.mapv(f64::sqrt))
            })
            .collect(),
    }
}

/// Covariances from precision Cholesky factors: `Σ = (U Uᵀ)⁻¹`.
pub(crate) fn covariances_from_precision_cholesky(
    prec_chol: &[Array2<f64>],
    ct: CovarianceType,
) -> Vec<Array2<f64>> {
    match ct {
        CovarianceType::Full | CovarianceType::Tied => prec_chol
            .iter()
            .map(|u| {
                // Σ = U⁻ᵀ U⁻¹ ; Uᵀ is lower triangular.
                let inv_ut = invert_lower(&u.t().to_owned()); // = U⁻ᵀ
                inv_ut.dot(&inv_ut.t())
            })
            .collect(),
        CovarianceType::Diag | CovarianceType::Spherical => prec_chol
            .iter()
            .map(|u| u.mapv(|v| 1.0 / (v * v)))
            .collect(),
    }
}

/// `log|U|` per block (sklearn `_compute_log_det_cholesky`); for spherical
/// it is `d · log u` since the scalar stands for `u·I_d`.
pub(crate) fn log_det_cholesky(
    prec_chol: &[Array2<f64>],
    ct: CovarianceType,
    d: usize,
) -> Vec<f64> {
    prec_chol
        .iter()
        .map(|u| match ct {
            CovarianceType::Full | CovarianceType::Tied => {
                (0..u.nrows()).map(|i| u[[i, i]].ln()).sum()
            }
            CovarianceType::Diag => u.iter().map(|v| v.ln()).sum(),
            CovarianceType::Spherical => d as f64 * u[[0, 0]].ln(),
        })
        .collect()
}

/// `log N(x_i | μ_k, Λ_k⁻¹)` for all samples and components, batched with
/// one GEMM per component (sklearn `_estimate_log_gaussian_prob`).
pub(crate) fn estimate_log_gaussian_prob(
    x: &Array2<f64>,
    means: &Array2<f64>,
    prec_chol: &[Array2<f64>],
    ct: CovarianceType,
) -> Array2<f64> {
    let n = x.nrows();
    let d = x.ncols();
    let k = means.nrows();
    let log_det = log_det_cholesky(prec_chol, ct, d);
    let mut quad = Array2::<f64>::zeros((n, k));
    match ct {
        CovarianceType::Full | CovarianceType::Tied => {
            let cols: Vec<Array1<f64>> = (0..k)
                .into_par_iter()
                .map(|c| {
                    let u = &prec_chol[if ct == CovarianceType::Tied { 0 } else { c }];
                    let mut y = x.dot(u);
                    let mu_u = means.row(c).dot(u);
                    y -= &mu_u;
                    y.map_axis(Axis(1), |r| r.iter().map(|v| v * v).sum())
                })
                .collect();
            for (c, col) in cols.into_iter().enumerate() {
                quad.column_mut(c).assign(&col);
            }
        }
        CovarianceType::Diag => {
            // Σ_j μ²λ − 2 X·(μ∘λ)ᵀ + X²·λᵀ
            let prec = Array2::from_shape_fn((k, d), |(c, j)| prec_chol[c][[0, j]].powi(2));
            let mp = means * &prec;
            let a: Array1<f64> = (&mp * means).sum_axis(Axis(1));
            let x2 = x.mapv(|v| v * v);
            quad = x2.dot(&prec.t()) - x.dot(&mp.t()) * 2.0;
            quad += &a;
        }
        CovarianceType::Spherical => {
            let prec = Array1::from_shape_fn(k, |c| prec_chol[c][[0, 0]].powi(2));
            let m2: Array1<f64> = means.map_axis(Axis(1), |r| r.iter().map(|v| v * v).sum());
            let x2: Array1<f64> = x.map_axis(Axis(1), |r| r.iter().map(|v| v * v).sum());
            let xm = x.dot(&means.t());
            for i in 0..n {
                for c in 0..k {
                    quad[[i, c]] = (m2[c] - 2.0 * xm[[i, c]] + x2[i]) * prec[c];
                }
            }
        }
    }
    let ln2pi = (2.0 * std::f64::consts::PI).ln();
    let mut out = quad;
    for c in 0..k {
        let ld = log_det[if ct == CovarianceType::Tied { 0 } else { c }];
        out.column_mut(c)
            .mapv_inplace(|q| -0.5 * (d as f64 * ln2pi + q) + ld);
    }
    out
}

/// Row-wise `logsumexp` and log-responsibilities from weighted log-probs.
pub(crate) fn log_prob_resp(weighted: Array2<f64>) -> (Array1<f64>, Array2<f64>) {
    let norm: Array1<f64> = weighted
        .rows()
        .into_iter()
        .map(|r| logsumexp(r.as_slice().unwrap()))
        .collect();
    let mut log_resp = weighted;
    for (mut r, &z) in log_resp.rows_mut().into_iter().zip(norm.iter()) {
        r.mapv_inplace(|v| v - z);
    }
    (norm, log_resp)
}

/// Sufficient statistics of the M-step (sklearn
/// `_estimate_gaussian_parameters`): `(n_k, μ_k, Σ_k)` with
/// `n_k = Σ_i r_ik + 10·ε` and `reg_covar` added to the covariance diagonal.
pub(crate) fn estimate_gaussian_parameters(
    x: &Array2<f64>,
    resp: &Array2<f64>,
    reg_covar: f64,
    ct: CovarianceType,
) -> (Array1<f64>, Array2<f64>, Vec<Array2<f64>>) {
    let d = x.ncols();
    let k = resp.ncols();
    let nk: Array1<f64> = resp.sum_axis(Axis(0)) + 10.0 * f64::EPSILON;
    let mut means = resp.t().dot(x);
    for c in 0..k {
        let inv = 1.0 / nk[c];
        means.row_mut(c).mapv_inplace(|v| v * inv);
    }
    let covs = match ct {
        CovarianceType::Full => (0..k)
            .into_par_iter()
            .map(|c| {
                let diff = x - &means.row(c);
                let mut weighted = diff.clone();
                for (mut row, &r) in weighted.rows_mut().into_iter().zip(resp.column(c).iter()) {
                    row.mapv_inplace(|v| v * r);
                }
                let mut cov = weighted.t().dot(&diff) / nk[c];
                for j in 0..d {
                    cov[[j, j]] += reg_covar;
                }
                cov
            })
            .collect(),
        CovarianceType::Tied => {
            let avg_x2 = x.t().dot(x);
            let mut nm = means.t().to_owned();
            for c in 0..k {
                nm.column_mut(c).mapv_inplace(|v| v * nk[c]);
            }
            let avg_m2 = nm.dot(&means);
            let mut cov = (avg_x2 - avg_m2) / nk.sum();
            for j in 0..d {
                cov[[j, j]] += reg_covar;
            }
            vec![cov]
        }
        CovarianceType::Diag | CovarianceType::Spherical => {
            let x2 = x.mapv(|v| v * v);
            let mut avg_x2 = resp.t().dot(&x2);
            for c in 0..k {
                let inv = 1.0 / nk[c];
                avg_x2.row_mut(c).mapv_inplace(|v| v * inv);
            }
            let diag = avg_x2 - means.mapv(|v| v * v) + reg_covar;
            (0..k)
                .map(|c| {
                    let row = diag.row(c);
                    if ct == CovarianceType::Diag {
                        row.to_owned().insert_axis(Axis(0))
                    } else {
                        Array2::from_elem((1, 1), row.mean().unwrap())
                    }
                })
                .collect()
        }
    };
    (nk, means, covs)
}

/// Initial responsibilities (sklearn `_initialize_parameters`).
pub(crate) fn initial_resp(
    x: &Array2<f64>,
    k: usize,
    init: MixtureInit,
    rng: &mut StdRng,
) -> Result<Array2<f64>> {
    let n = x.nrows();
    let mut resp = Array2::<f64>::zeros((n, k));
    match init {
        MixtureInit::KMeans => {
            let km = crate::KMeans::new(k).with_n_init(1).with_seed(rng.gen());
            let fitted: crate::FittedKMeans<f64> = anofox_ml_core::FitUnsupervised::fit(&km, x)?;
            for (i, &l) in fitted.labels().iter().enumerate() {
                resp[[i, l as usize]] = 1.0;
            }
        }
        MixtureInit::KMeansPlusPlus => {
            let mut sub = StdRng::seed_from_u64(rng.gen());
            let (_, idx) = kmeans_plusplus(x, k, &mut sub, None);
            for (c, &i) in idx.iter().enumerate() {
                resp[[i, c]] = 1.0;
            }
        }
        MixtureInit::Random => {
            for mut row in resp.rows_mut() {
                row.mapv_inplace(|_| rng.gen::<f64>());
                let s = row.sum();
                row.mapv_inplace(|v| v / s);
            }
        }
        MixtureInit::RandomFromData => {
            let idx = sample_without_replacement::<f64>(n, k, rng, None);
            for (c, &i) in idx.iter().enumerate() {
                resp[[i, c]] = 1.0;
            }
        }
    }
    Ok(resp)
}

/// Number of free parameters (sklearn `_n_parameters`).
pub(crate) fn n_parameters(ct: CovarianceType, k: usize, d: usize) -> usize {
    let cov = match ct {
        CovarianceType::Full => k * d * (d + 1) / 2,
        CovarianceType::Diag => k * d,
        CovarianceType::Tied => d * (d + 1) / 2,
        CovarianceType::Spherical => k,
    };
    cov + d * k + k - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn precision_cholesky_roundtrip() {
        let cov = array![[2.0, 0.3, 0.1], [0.3, 1.0, -0.2], [0.1, -0.2, 0.5]];
        let u = precision_cholesky(std::slice::from_ref(&cov), CovarianceType::Full).unwrap();
        // U Uᵀ = Σ⁻¹  ⇒  Σ · U Uᵀ = I
        let prod = cov.dot(&u[0].dot(&u[0].t()));
        for i in 0..3 {
            for j in 0..3 {
                let e = if i == j { 1.0 } else { 0.0 };
                assert!((prod[[i, j]] - e).abs() < 1e-12);
            }
        }
        // Upper triangular, as sklearn's precisions_cholesky_.
        assert!(u[0][[1, 0]].abs() < 1e-15 && u[0][[2, 0]].abs() < 1e-15);
        let back = covariances_from_precision_cholesky(&u, CovarianceType::Full);
        for (a, b) in back[0].iter().zip(cov.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
        // From precisions: same U as from covariances (unique upper factor).
        let prec = u[0].dot(&u[0].t());
        let u2 = precision_cholesky_from_precisions(&[prec], CovarianceType::Full).unwrap();
        for (a, b) in u2[0].iter().zip(u[0].iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn singular_covariance_is_an_error() {
        let cov = array![[1.0, 1.0], [1.0, 1.0]];
        let err = precision_cholesky(&[cov], CovarianceType::Full).unwrap_err();
        assert!(err.to_string().contains("reg_covar"));
    }
}
