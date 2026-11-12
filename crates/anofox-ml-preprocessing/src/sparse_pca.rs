//! Sparse principal components (sklearn's `decomposition.SparsePCA`,
//! `method="lars"`): dictionary learning on the centred, transposed data,
//! whose sparse codes are the components.
//!
//! sklearn's `_dict_learning`: start from the thin SVD (`code = U`,
//! `dictionary = Σ Vᵀ`), then alternate LassoLars codes (`alpha / n`, no
//! intercept) and the block-coordinate dictionary update with in-place
//! atoms projected onto the unit ball, until the cost
//! `½‖X − code·dict‖² + alpha·‖code‖₁` improves by less than `tol`
//! relatively. The problem is equivariant under flipping a component's
//! sign, and sklearn's closing `svd_flip(code, dict, u_based_decision=True)`
//! fixes the signs, so the SVD's sign choice does not matter. An atom that
//! is never used is redrawn from a data row plus noise (numpy's
//! `RandomState`), as in sklearn.

use crate::sparse_coding::{
    CodingAlgorithm, CodingParams, FitAlgorithm, MiniBatchDictionaryLearning,
};
use anofox_ml_core::{FitUnsupervised, Result, RustMlError, Transform};
use ndarray::{Array1, Array2, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

/// sklearn's `SparsePCA(method="lars")`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SparsePca {
    /// `None`: `n_features`.
    pub n_components: Option<usize>,
    /// Sparsity of the components.
    pub alpha: f64,
    /// Ridge penalty of `transform`.
    pub ridge_alpha: f64,
    pub max_iter: usize,
    pub tol: f64,
    pub seed: u32,
}

impl Default for SparsePca {
    fn default() -> Self {
        Self {
            n_components: None,
            alpha: 1.0,
            ridge_alpha: 0.01,
            max_iter: 1000,
            tol: 1e-8,
            seed: 0,
        }
    }
}

/// A fitted [`SparsePca`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedSparsePca {
    /// Unit-norm sparse components, `n_components × n_features`
    /// (sklearn's `components_`).
    pub components: Array2<f64>,
    pub mean: Array1<f64>,
    /// The cost after each iteration (sklearn's `error_`).
    pub error: Vec<f64>,
    pub n_iter: usize,
    ridge_alpha: f64,
}

impl FitUnsupervised<f64> for SparsePca {
    type Fitted = FittedSparsePca;

    fn fit_view(&self, x: ArrayView2<f64>) -> Result<Self::Fitted> {
        let (n, p) = x.dim();
        if n == 0 || p == 0 {
            return Err(RustMlError::EmptyInput("SparsePCA needs data".into()));
        }
        let k = self.n_components.unwrap_or(p);
        if k == 0 {
            return Err(RustMlError::InvalidParameter(
                "n_components must be at least 1".into(),
            ));
        }
        if !(self.alpha >= 0.0 && self.alpha.is_finite()) {
            return Err(RustMlError::InvalidParameter(format!(
                "alpha must be finite and >= 0, got {}",
                self.alpha
            )));
        }
        let mean = x.mean_axis(Axis(0)).unwrap();
        // Dictionary learning on Xcᵀ (features as samples).
        let y = (&x - &mean).reversed_axes().to_owned(); // p × n
        let fit = crate::sparse_coding::dict_learning(
            &y,
            k,
            self.alpha,
            self.tol,
            self.max_iter,
            CodingAlgorithm::LassoLars,
            1000,
            self.seed,
        )?;
        let (mut code, errors) = (fit.code, fit.errors);
        // svd_flip(code, dictionary, u_based_decision=True): np.sign of each
        // code column's largest-magnitude entry (0 for a zero column).
        for c in 0..k {
            let col = code.column(c);
            let (mut best, mut best_abs) = (0, f64::NEG_INFINITY);
            for (i, v) in col.iter().enumerate() {
                if v.abs() > best_abs {
                    (best, best_abs) = (i, v.abs());
                }
            }
            let v = col[best];
            let sign = if v > 0.0 {
                1.0
            } else if v < 0.0 {
                -1.0
            } else {
                0.0
            };
            code.column_mut(c).mapv_inplace(|t| t * sign);
        }
        let mut components = code.reversed_axes();
        for mut row in components.rows_mut() {
            let norm = row.dot(&row).sqrt();
            let norm = if norm == 0.0 { 1.0 } else { norm };
            row.mapv_inplace(|v| v / norm);
        }
        Ok(FittedSparsePca {
            components,
            mean,
            n_iter: errors.len(),
            error: errors,
            ridge_alpha: self.ridge_alpha,
        })
    }
}

/// Solve the symmetric positive-definite `a z = b` by Cholesky (`b`:
/// columns).
fn cholesky_solve(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>> {
    let k = a.nrows();
    let mut l = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..=i {
            let s: f64 = a[[i, j]] - (0..j).map(|t| l[[i, t]] * l[[j, t]]).sum::<f64>();
            if i == j {
                if s <= 0.0 {
                    return Err(RustMlError::InvalidParameter(
                        "ridge system is not positive definite".into(),
                    ));
                }
                l[[i, i]] = s.sqrt();
            } else {
                l[[i, j]] = s / l[[j, j]];
            }
        }
    }
    let mut z = b.clone();
    for mut col in z.columns_mut() {
        for i in 0..k {
            let s: f64 = col[i] - (0..i).map(|t| l[[i, t]] * col[t]).sum::<f64>();
            col[i] = s / l[[i, i]];
        }
        for i in (0..k).rev() {
            let s: f64 = col[i] - (i + 1..k).map(|t| l[[t, i]] * col[t]).sum::<f64>();
            col[i] = s / l[[i, i]];
        }
    }
    Ok(z)
}

/// sklearn's `MiniBatchSparsePCA`: `MiniBatchDictionaryLearning` on the
/// centred, transposed data, whose LassoLars (or Lasso-CD) codes of the
/// data at `alpha` are the components (normalised); `transform` is [`SparsePca`]'s ridge
/// projection. The draws are numpy's `RandomState(seed)`, so the components
/// are sklearn's for the same `random_state`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MiniBatchSparsePca {
    /// `None`: `n_features`.
    pub n_components: Option<usize>,
    pub alpha: f64,
    pub ridge_alpha: f64,
    pub max_iter: usize,
    pub batch_size: usize,
    pub shuffle: bool,
    pub method: FitAlgorithm,
    pub tol: f64,
    pub max_no_improvement: Option<usize>,
    pub seed: u32,
}

impl Default for MiniBatchSparsePca {
    fn default() -> Self {
        Self {
            n_components: None,
            alpha: 1.0,
            ridge_alpha: 0.01,
            max_iter: 1000,
            batch_size: 3,
            shuffle: true,
            method: FitAlgorithm::Lars,
            tol: 1e-3,
            max_no_improvement: Some(10),
            seed: 0,
        }
    }
}

impl FitUnsupervised<f64> for MiniBatchSparsePca {
    type Fitted = FittedSparsePca;

    fn fit_view(&self, x: ArrayView2<f64>) -> Result<FittedSparsePca> {
        let (n, p) = x.dim();
        if n == 0 || p == 0 {
            return Err(RustMlError::EmptyInput(
                "MiniBatchSparsePCA needs data".into(),
            ));
        }
        let mean = x.mean_axis(Axis(0)).unwrap();
        let y = (&x - &mean).reversed_axes().to_owned(); // p × n
        let est = MiniBatchDictionaryLearning {
            n_components: Some(self.n_components.unwrap_or(p)),
            alpha: self.alpha,
            max_iter: self.max_iter,
            fit_algorithm: self.method,
            batch_size: self.batch_size,
            shuffle: self.shuffle,
            // sklearn codes the components by "lasso_" + method at alpha.
            transform: CodingParams {
                algorithm: match self.method {
                    FitAlgorithm::Lars => CodingAlgorithm::LassoLars,
                    FitAlgorithm::Cd => CodingAlgorithm::LassoCd,
                },
                alpha: Some(self.alpha),
                ..CodingParams::default()
            },
            tol: self.tol,
            max_no_improvement: self.max_no_improvement,
            seed: self.seed,
        };
        let fitted = est.learn(&y)?;
        let mut components = fitted.transform(&y)?.reversed_axes();
        for mut row in components.rows_mut() {
            let norm = row.dot(&row).sqrt();
            let norm = if norm == 0.0 { 1.0 } else { norm };
            row.mapv_inplace(|v| v / norm);
        }
        Ok(FittedSparsePca {
            components,
            mean,
            error: Vec::new(),
            n_iter: fitted.n_iter,
            ridge_alpha: self.ridge_alpha,
        })
    }
}

impl Transform<f64> for FittedSparsePca {
    /// sklearn's `ridge_regression(components_ᵀ, (X − mean)ᵀ, ridge_alpha)`:
    /// `(C Cᵀ + αI)⁻¹ C (x − mean)` per row.
    fn transform_view(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.mean.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.mean.len(),
                x.ncols()
            )));
        }
        let xc = &x - &self.mean;
        let c = &self.components;
        let mut gram = c.dot(&c.t());
        for i in 0..gram.nrows() {
            gram[[i, i]] += self.ridge_alpha;
        }
        let rhs = c.dot(&xc.t()); // k × n
        Ok(cholesky_solve(&gram, &rhs)?.reversed_axes())
    }
}
