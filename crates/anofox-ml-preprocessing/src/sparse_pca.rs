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

use anofox_ml_core::numpy_random::RandomState;
use anofox_ml_core::{Fit, FitUnsupervised, Result, RustMlError, Transform};
use anofox_ml_regression::Lars;
use faer::linalg::solvers::Svd;
use faer::Mat;
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

/// Thin SVD of `x` with descending singular values: `(U, s, Vᵀ)`.
fn thin_svd(x: &Array2<f64>) -> Result<(Array2<f64>, Vec<f64>, Array2<f64>)> {
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

/// sklearn's `_sparse_encode(algorithm="lasso_lars")`: one LassoLars fit
/// per row of `x` on the atoms.
fn sparse_encode(x: &Array2<f64>, dictionary: &Array2<f64>, alpha: f64) -> Result<Array2<f64>> {
    let design = dictionary.t().to_owned(); // n × k
    let lars = Lars::lasso_lars(alpha / x.ncols() as f64)
        .with_fit_intercept(false)
        .with_max_iter(1000);
    let mut code = Array2::<f64>::zeros((x.nrows(), dictionary.nrows()));
    for (i, row) in x.rows().into_iter().enumerate() {
        let fit = Fit::fit(&lars, &design, &row.to_owned())?;
        code.row_mut(i).assign(&fit.coef);
    }
    Ok(code)
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
        let (u, s, vt) = thin_svd(&y)?;
        let r = s.len();
        let mut code = Array2::<f64>::zeros((p, k));
        let mut dictionary = Array2::<f64>::zeros((k, n));
        for c in 0..k.min(r) {
            code.column_mut(c).assign(&u.column(c));
            dictionary.row_mut(c).assign(&(&vt.row(c) * s[c]));
        }
        let mut rng = RandomState::new(self.seed);
        let mut errors: Vec<f64> = Vec::new();
        for ii in 0..self.max_iter {
            code = sparse_encode(&y, &dictionary, self.alpha)?;
            update_dict(&mut dictionary, &y, &mut code, &mut rng);
            let resid = &y - &code.dot(&dictionary);
            let cost = 0.5 * resid.mapv(|v| v * v).sum() + self.alpha * code.mapv(f64::abs).sum();
            errors.push(cost);
            if ii > 0 {
                let de = errors[ii - 1] - errors[ii];
                if de < self.tol * errors[ii] {
                    break;
                }
            }
        }
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
