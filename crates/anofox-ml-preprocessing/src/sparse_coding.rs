//! Sparse coding and dictionary learning (sklearn's `SparseCoder` and
//! `DictionaryLearning`, and the loop behind [`SparsePca`](crate::SparsePca)).
//!
//! [`sparse_encode`] is sklearn's `sparse_encode`: each row of `x` coded on
//! the dictionary's atoms by OMP, LARS, LassoLars, Lasso (coordinate
//! descent) or soft thresholding. `dict_learning` is sklearn's
//! `_dict_learning`: start from the thin SVD, then alternate sparse codes
//! and the block-coordinate dictionary update until the cost
//! `½‖X − code·dict‖² + alpha·‖code‖₁` improves by less than `tol`
//! relatively. An atom that is never used is redrawn from a data row plus
//! noise (numpy's `RandomState`), as in sklearn.

use anofox_ml_core::numpy_random::RandomState;
use anofox_ml_core::{Fit, FitUnsupervised, Result, RustMlError, Transform};
use anofox_ml_regression::{Lars, LassoRegressor, OrthogonalMatchingPursuit};
use faer::linalg::solvers::Svd;
use faer::Mat;
use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};

/// How rows are coded on a dictionary (sklearn's `transform_algorithm` /
/// `algorithm`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CodingAlgorithm {
    /// Orthogonal matching pursuit with `n_nonzero_coefs` atoms.
    #[default]
    Omp,
    /// LARS stopped at `n_nonzero_coefs` atoms.
    Lars,
    /// The Lasso by LARS at `alpha / n_features`.
    LassoLars,
    /// The Lasso by coordinate descent at `alpha / n_features`.
    LassoCd,
    /// Soft thresholding of `dictionary · x` at `alpha`.
    Threshold,
}

/// sklearn's `sparse_encode(x, dictionary, algorithm, n_nonzero_coefs,
/// alpha, max_iter)`: the codes, `n_samples × n_components`.
/// `n_nonzero_coefs` (OMP, LARS) defaults to
/// `int(min(max(n_features / 10, 1), n_components))`, `alpha` to 1.
pub fn sparse_encode(
    x: ArrayView2<f64>,
    dictionary: ArrayView2<f64>,
    algorithm: CodingAlgorithm,
    n_nonzero_coefs: Option<usize>,
    alpha: Option<f64>,
    max_iter: usize,
) -> Result<Array2<f64>> {
    let (n, p) = x.dim();
    let k = dictionary.nrows();
    if dictionary.ncols() != p {
        return Err(RustMlError::ShapeMismatch(format!(
            "the dictionary has {} features, x has {p}",
            dictionary.ncols()
        )));
    }
    let n_nonzero =
        n_nonzero_coefs.unwrap_or_else(|| ((p as f64 / 10.0).max(1.0).min(k as f64)) as usize);
    let alpha = alpha.unwrap_or(1.0);
    if !(alpha >= 0.0 && alpha.is_finite()) {
        return Err(RustMlError::InvalidParameter(format!(
            "alpha must be finite and >= 0, got {alpha}"
        )));
    }
    let design = dictionary.t().to_owned(); // p × k
    let mut code = Array2::<f64>::zeros((n, k));
    match algorithm {
        CodingAlgorithm::Threshold => {
            let cov = x.dot(&design); // n × k
            code.assign(&cov.mapv(|c| c.signum() * (c.abs() - alpha).max(0.0)));
        }
        CodingAlgorithm::LassoLars => {
            let lars = Lars::lasso_lars(alpha / p as f64)
                .with_fit_intercept(false)
                .with_max_iter(max_iter);
            for (i, row) in x.rows().into_iter().enumerate() {
                code.row_mut(i)
                    .assign(&Fit::fit(&lars, &design, &row.to_owned())?.coef);
            }
        }
        CodingAlgorithm::Lars => {
            let lars = Lars::new(n_nonzero).with_fit_intercept(false);
            for (i, row) in x.rows().into_iter().enumerate() {
                code.row_mut(i)
                    .assign(&Fit::fit(&lars, &design, &row.to_owned())?.coef);
            }
        }
        CodingAlgorithm::LassoCd => {
            let lasso = LassoRegressor::new()
                .with_sklearn_alpha(alpha / p as f64)
                .with_intercept(false);
            for (i, row) in x.rows().into_iter().enumerate() {
                code.row_mut(i)
                    .assign(&Fit::fit(&lasso, &design, &row.to_owned())?.coefficients());
            }
        }
        CodingAlgorithm::Omp => {
            let omp = OrthogonalMatchingPursuit::new()
                .with_n_nonzero_coefs(n_nonzero)
                .with_fit_intercept(false);
            for (i, row) in x.rows().into_iter().enumerate() {
                code.row_mut(i)
                    .assign(&Fit::fit(&omp, &design, &row.to_owned())?.coef);
            }
        }
    }
    Ok(code)
}

/// sklearn's `split_sign`: `[max(code, 0) | -min(code, 0)]`.
fn split_sign(code: Array2<f64>) -> Array2<f64> {
    let (n, k) = code.dim();
    Array2::from_shape_fn((n, 2 * k), |(i, j)| {
        if j < k {
            code[[i, j]].max(0.0)
        } else {
            -code[[i, j - k]].min(0.0)
        }
    })
}

/// The transform settings shared by [`SparseCoder`] and
/// [`DictionaryLearning`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CodingParams {
    pub algorithm: CodingAlgorithm,
    pub n_nonzero_coefs: Option<usize>,
    pub alpha: Option<f64>,
    pub split_sign: bool,
    pub max_iter: usize,
}

impl Default for CodingParams {
    fn default() -> Self {
        Self {
            algorithm: CodingAlgorithm::Omp,
            n_nonzero_coefs: None,
            alpha: None,
            split_sign: false,
            max_iter: 1000,
        }
    }
}

impl CodingParams {
    fn encode(&self, x: ArrayView2<f64>, dictionary: &Array2<f64>) -> Result<Array2<f64>> {
        let code = sparse_encode(
            x,
            dictionary.view(),
            self.algorithm,
            self.n_nonzero_coefs,
            self.alpha,
            self.max_iter,
        )?;
        Ok(if self.split_sign {
            split_sign(code)
        } else {
            code
        })
    }
}

// ============================================================
// SparseCoder
// ============================================================

/// sklearn's `SparseCoder`: codes rows on a fixed `dictionary`
/// (`n_components × n_features`, atoms as rows). `fit` only checks it.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SparseCoder {
    pub dictionary: Array2<f64>,
    pub coding: CodingParams,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedSparseCoder {
    dictionary: Array2<f64>,
    coding: CodingParams,
}

impl FitUnsupervised<f64> for SparseCoder {
    type Fitted = FittedSparseCoder;

    fn fit_view(&self, x: ArrayView2<f64>) -> Result<FittedSparseCoder> {
        if self.dictionary.is_empty() {
            return Err(RustMlError::InvalidParameter(
                "SparseCoder needs a dictionary".into(),
            ));
        }
        if x.ncols() != self.dictionary.ncols() {
            return Err(RustMlError::ShapeMismatch(format!(
                "the dictionary has {} features, x has {}",
                self.dictionary.ncols(),
                x.ncols()
            )));
        }
        Ok(FittedSparseCoder {
            dictionary: self.dictionary.clone(),
            coding: self.coding.clone(),
        })
    }
}

impl Transform<f64> for FittedSparseCoder {
    fn transform_view(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        self.coding.encode(x, &self.dictionary)
    }
}

// ============================================================
// Dictionary learning
// ============================================================

/// Thin SVD of `x` with descending singular values: `(U, s, Vᵀ)`.
pub(crate) fn thin_svd(x: &Array2<f64>) -> Result<(Array2<f64>, Vec<f64>, Array2<f64>)> {
    let (n, p) = x.dim();
    let m = Mat::<f64>::from_fn(n, p, |i, j| x[[i, j]]);
    let svd = Svd::new_thin(m.as_ref())
        .map_err(|e| RustMlError::InvalidParameter(format!("SVD failed: {e:?}")))?;
    let (s, u, v) = (svd.S().column_vector(), svd.U(), svd.V());
    let r = s.nrows();
    let mut order: Vec<usize> = (0..r).collect();
    order.sort_by(|&a, &b| s[b].total_cmp(&s[a]));
    let uu = Array2::from_shape_fn((n, r), |(i, c)| u[(i, order[c])]);
    let vt = Array2::from_shape_fn((r, p), |(c, j)| v[(j, order[c])]);
    Ok((uu, order.iter().map(|&i| s[i]).collect(), vt))
}

/// sklearn's `_update_dict` (no positivity constraint).
fn update_dict(
    dictionary: &mut Array2<f64>,
    y: &Array2<f64>,
    code: &mut Array2<f64>,
    rng: &mut RandomState,
) {
    let n_samples = code.nrows();
    let a = code.t().dot(code);
    let b = y.t().dot(code);
    for k in 0..dictionary.nrows() {
        if a[[k, k]] > 1e-6 {
            let step = (&b.column(k) - &a.row(k).dot(dictionary)) / a[[k, k]];
            let mut row = dictionary.row_mut(k);
            row += &step;
        } else {
            let pick = rng.randint(n_samples as u64) as usize;
            let newd = y.row(pick).to_owned();
            let std = newd.std(0.0);
            let level = 0.01 * if std == 0.0 { 1.0 } else { std };
            let noisy = newd.mapv(|v| v + rng.normal(0.0, level));
            dictionary.row_mut(k).assign(&noisy);
            code.column_mut(k).fill(0.0);
        }
        let norm = dictionary.row(k).dot(&dictionary.row(k)).sqrt();
        dictionary.row_mut(k).mapv_inplace(|v| v / norm.max(1.0));
    }
}

/// The result of [`dict_learning`].
pub(crate) struct DictLearningFit {
    pub code: Array2<f64>,
    pub dictionary: Array2<f64>,
    pub errors: Vec<f64>,
}

/// sklearn's `_dict_learning(x, k, alpha, tol, max_iter, method)`, the
/// codes by `method` (LassoLars or LassoCd) at `alpha`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dict_learning(
    x: &Array2<f64>,
    k: usize,
    alpha: f64,
    tol: f64,
    max_iter: usize,
    method: CodingAlgorithm,
    method_max_iter: usize,
    seed: u32,
) -> Result<DictLearningFit> {
    let (n, p) = x.dim();
    if n == 0 || p == 0 {
        return Err(RustMlError::EmptyInput(
            "dictionary learning needs data".into(),
        ));
    }
    if k == 0 {
        return Err(RustMlError::InvalidParameter(
            "n_components must be at least 1".into(),
        ));
    }
    if !(alpha >= 0.0 && alpha.is_finite()) {
        return Err(RustMlError::InvalidParameter(format!(
            "alpha must be finite and >= 0, got {alpha}"
        )));
    }
    let (u, s, vt) = thin_svd(x)?;
    let mut code = Array2::<f64>::zeros((n, k));
    let mut dictionary = Array2::<f64>::zeros((k, p));
    for (c, &sc) in s.iter().enumerate().take(k) {
        code.column_mut(c).assign(&u.column(c));
        dictionary.row_mut(c).assign(&(&vt.row(c) * sc));
    }
    let mut rng = RandomState::new(seed);
    let mut errors: Vec<f64> = Vec::new();
    for ii in 0..max_iter {
        code = sparse_encode(
            x.view(),
            dictionary.view(),
            method,
            None,
            Some(alpha),
            method_max_iter,
        )?;
        update_dict(&mut dictionary, x, &mut code, &mut rng);
        let resid = x - &code.dot(&dictionary);
        let cost = 0.5 * resid.mapv(|v| v * v).sum() + alpha * code.mapv(f64::abs).sum();
        errors.push(cost);
        if ii > 0 && errors[ii - 1] - errors[ii] < tol * errors[ii] {
            break;
        }
    }
    Ok(DictLearningFit {
        code,
        dictionary,
        errors,
    })
}

/// How [`DictionaryLearning`] codes the data while it learns (sklearn's
/// `fit_algorithm`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FitAlgorithm {
    /// LassoLars.
    #[default]
    Lars,
    /// Lasso by coordinate descent.
    Cd,
}

/// sklearn's `DictionaryLearning` (no positivity constraints, no initial
/// code or dictionary): learns `n_components` atoms (default
/// `n_features`) with codes of sparsity `alpha`, then codes new rows with
/// the `transform` settings (`alpha` defaults to the fitting `alpha`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DictionaryLearning {
    pub n_components: Option<usize>,
    pub alpha: f64,
    pub max_iter: usize,
    pub tol: f64,
    pub fit_algorithm: FitAlgorithm,
    pub transform: CodingParams,
    pub seed: u32,
}

impl Default for DictionaryLearning {
    fn default() -> Self {
        Self {
            n_components: None,
            alpha: 1.0,
            max_iter: 1000,
            tol: 1e-8,
            fit_algorithm: FitAlgorithm::Lars,
            transform: CodingParams::default(),
            seed: 0,
        }
    }
}

/// A fitted [`DictionaryLearning`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedDictionaryLearning {
    /// The atoms, `n_components × n_features` (sklearn's `components_`).
    pub components: Array2<f64>,
    /// The cost after each iteration (sklearn's `error_`).
    pub error: Vec<f64>,
    pub n_iter: usize,
    /// The training rows' codes (what sklearn's `fit_transform` returns).
    #[serde(skip)]
    pub train_code: Array2<f64>,
    transform: CodingParams,
}

impl FitUnsupervised<f64> for DictionaryLearning {
    type Fitted = FittedDictionaryLearning;

    fn fit_view(&self, x: ArrayView2<f64>) -> Result<FittedDictionaryLearning> {
        let x = x.to_owned();
        let k = self.n_components.unwrap_or(x.ncols());
        let method = match self.fit_algorithm {
            FitAlgorithm::Lars => CodingAlgorithm::LassoLars,
            FitAlgorithm::Cd => CodingAlgorithm::LassoCd,
        };
        let fit = dict_learning(
            &x,
            k,
            self.alpha,
            self.tol,
            self.max_iter,
            method,
            self.transform.max_iter,
            self.seed,
        )?;
        let mut transform = self.transform.clone();
        transform.alpha = transform.alpha.or(Some(self.alpha));
        Ok(FittedDictionaryLearning {
            components: fit.dictionary,
            n_iter: fit.errors.len(),
            error: fit.errors,
            train_code: if transform.split_sign {
                split_sign(fit.code)
            } else {
                fit.code
            },
            transform,
        })
    }
}

impl Transform<f64> for FittedDictionaryLearning {
    fn transform_view(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        self.transform.encode(x, &self.components)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn threshold_and_split_sign() {
        let dict = array![[1.0, 0.0], [0.0, 1.0]];
        let x = array![[3.0, -0.5], [-2.0, 1.5]];
        let code = sparse_encode(
            x.view(),
            dict.view(),
            CodingAlgorithm::Threshold,
            None,
            Some(1.0),
            1000,
        )
        .unwrap();
        assert_eq!(code, array![[2.0, 0.0], [-1.0, 0.5]]);
        assert_eq!(
            split_sign(code),
            array![[2.0, 0.0, 0.0, 0.0], [0.0, 0.5, 1.0, 0.0]]
        );
    }

    #[test]
    fn omp_and_lars_recover_a_sparse_code() {
        // Orthonormal atoms: the code is the coordinates.
        let dict = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let x = array![[0.0, 4.0, 0.1]];
        // OMP refits the chosen atom; LARS stops where the next atom ties
        // (sklearn: [[0, 4, 0]] and [[0, 3.9, 0]]).
        for (algorithm, want) in [(CodingAlgorithm::Omp, 4.0), (CodingAlgorithm::Lars, 3.9)] {
            let code =
                sparse_encode(x.view(), dict.view(), algorithm, Some(1), None, 1000).unwrap();
            assert_eq!(
                code.mapv(|v| (v * 1e9).round() / 1e9),
                array![[0.0, want, 0.0]],
                "{algorithm:?}"
            );
        }
    }
}
