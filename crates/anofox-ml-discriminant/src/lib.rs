//! Linear and Quadratic Discriminant Analysis.
//!
//! Mirrors `sklearn.discriminant_analysis.{LinearDiscriminantAnalysis,
//! QuadraticDiscriminantAnalysis}`.
//!
//! - **LDA** assumes all classes share a common covariance Σ. Decision
//!   function is linear in `x`.
//! - **QDA** estimates a separate covariance Σ_k per class.

use anofox_ml_core::{Fit, Predict, PredictProba, Result, RustMlError, Transform};
use faer::linalg::solvers::{SelfAdjointEigen, Solve};
use faer::{Mat, Side};
use ndarray::{Array1, Array2};

/// Common helpers.
fn class_indices(y: &Array1<f64>) -> (Vec<f64>, Vec<Vec<usize>>) {
    let mut classes: Vec<f64> = y.iter().copied().collect();
    classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    classes.dedup();
    let groups: Vec<Vec<usize>> = classes
        .iter()
        .map(|&c| {
            y.iter()
                .enumerate()
                .filter(|(_, &v)| v == c)
                .map(|(i, _)| i)
                .collect()
        })
        .collect();
    (classes, groups)
}

fn class_mean(x: &Array2<f64>, idx: &[usize]) -> Array1<f64> {
    let d = x.ncols();
    let mut m = Array1::<f64>::zeros(d);
    for &i in idx {
        for j in 0..d {
            m[j] += x[[i, j]];
        }
    }
    let n = idx.len() as f64;
    m.mapv(|v| v / n)
}

fn outer_subtract_accum(x: &Array2<f64>, mu: &Array1<f64>, idx: &[usize], accum: &mut Array2<f64>) {
    let d = x.ncols();
    for &i in idx {
        let mut dv = vec![0.0; d];
        for j in 0..d {
            dv[j] = x[[i, j]] - mu[j];
        }
        for a in 0..d {
            for b in 0..d {
                accum[[a, b]] += dv[a] * dv[b];
            }
        }
    }
}

fn solve_psd(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
    let n = a.nrows();
    let am = Mat::from_fn(n, n, |i, j| a[[i, j]]);
    let llt = faer::linalg::solvers::Llt::new(am.as_ref(), Side::Lower)
        .map_err(|e| RustMlError::InvalidParameter(format!("LLT failed: {e:?}")))?;
    let bm = Mat::from_fn(n, 1, |i, _| b[i]);
    let s = llt.solve(&bm);
    Ok(Array1::from_vec((0..n).map(|i| s[(i, 0)]).collect()))
}

fn log_det_chol(a: &Array2<f64>) -> Result<f64> {
    let n = a.nrows();
    let am = Mat::from_fn(n, n, |i, j| a[[i, j]]);
    let llt = faer::linalg::solvers::Llt::new(am.as_ref(), Side::Lower)
        .map_err(|e| RustMlError::InvalidParameter(format!("LLT failed: {e:?}")))?;
    let lower = llt.L();
    let mut s = 0.0;
    for i in 0..n {
        s += lower[(i, i)].abs().ln();
    }
    Ok(2.0 * s)
}

/// Solve the generalised symmetric eigenproblem `Σ_b v = λ Σ_w v` via
/// Cholesky whitening (`Σ_w = L Lᵀ`): eigendecompose the symmetric matrix
/// `L⁻¹ Σ_b L⁻ᵀ = U Λ Uᵀ` and back-transform `V = L⁻ᵀ U`, which yields
/// `Vᵀ Σ_w V = I` (the same normalisation as `scipy.linalg.eigh(Sb, Sw)`).
///
/// Returns the top `n_proj` eigenvectors (descending eigenvalue order) and
/// their explained-variance ratios `λ_i / Σ_j λ_j`.
fn lda_projection(
    sw: &Mat<f64>,
    sb: &Mat<f64>,
    n_proj: usize,
) -> Result<(Array2<f64>, Array1<f64>)> {
    use faer::linalg::triangular_solve::{
        solve_lower_triangular_in_place, solve_upper_triangular_in_place,
    };
    use faer::Par;

    let d = sw.nrows();
    let llt = faer::linalg::solvers::Llt::new(sw.as_ref(), Side::Lower)
        .map_err(|e| RustMlError::InvalidParameter(format!("Σ_w Cholesky failed: {e:?}")))?;
    let lower = llt.L();

    // A = L⁻¹ Σ_b.
    let mut a = sb.clone();
    solve_lower_triangular_in_place(lower, a.as_mut(), Par::Seq);
    // B = L⁻¹ Aᵀ = L⁻¹ Σ_b L⁻ᵀ (Σ_b symmetric ⇒ Aᵀ = Σ_b L⁻ᵀ).
    let mut b = a.transpose().to_owned();
    solve_lower_triangular_in_place(lower, b.as_mut(), Par::Seq);
    // Symmetrise to wash out round-off before the self-adjoint solver.
    let b = Mat::from_fn(d, d, |i, j| 0.5 * (b[(i, j)] + b[(j, i)]));

    let eig = SelfAdjointEigen::new(b.as_ref(), Side::Lower)
        .map_err(|e| RustMlError::InvalidParameter(format!("eigen failed: {e:?}")))?;
    let evals = eig.S().column_vector();
    let u = eig.U();

    // Descending eigenvalue order.
    let mut order: Vec<usize> = (0..d).collect();
    order.sort_by(|&i, &j| {
        evals[j]
            .partial_cmp(&evals[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let total: f64 = (0..d).map(|i| evals[i]).sum();

    // V = L⁻ᵀ U for the retained columns.
    let mut v = Mat::from_fn(d, n_proj, |i, c| u[(i, order[c])]);
    solve_upper_triangular_in_place(lower.transpose(), v.as_mut(), Par::Seq);

    let scalings = Array2::from_shape_fn((d, n_proj), |(i, c)| v[(i, c)]);
    let evr = Array1::from_shape_fn(n_proj, |c| {
        if total > 0.0 {
            evals[order[c]] / total
        } else {
            0.0
        }
    });
    Ok((scalings, evr))
}

// ---------------------------------------------------------------------------
// LinearDiscriminantAnalysis (LDA)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct LinearDiscriminantAnalysis {
    /// Shrinkage on the within-class covariance toward `(tr(Σ)/d) I`.
    /// 0.0 = no shrinkage (sklearn default).
    pub shrinkage: f64,
    /// Reg term added to the diagonal of Σ for numerical stability.
    pub reg: f64,
}

impl LinearDiscriminantAnalysis {
    pub fn new() -> Self {
        Self {
            shrinkage: 0.0,
            reg: 1e-9,
        }
    }
    pub fn with_shrinkage(mut self, s: f64) -> Self {
        self.shrinkage = s;
        self
    }
}

impl Default for LinearDiscriminantAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedLinearDiscriminantAnalysis {
    pub classes: Vec<f64>,
    pub means: Vec<Array1<f64>>,
    pub priors: Vec<f64>,
    pub coef: Vec<Array1<f64>>, // sigma_inv @ mu_k
    pub intercept: Vec<f64>,    // -0.5 * mu_k^T sigma_inv mu_k + log(pi_k)
    /// Projection matrix for dimensionality reduction; columns are the
    /// generalised eigenvectors of `Σ_b v = λ Σ_w v` in descending eigenvalue
    /// order, normalised so that `Vᵀ Σ_w V = I` (sklearn `solver='eigen'`
    /// `scalings_`, with `Σ_w` the biased within-class covariance).
    /// Shape `(d, min(n_classes - 1, d))`.
    pub scalings: Array2<f64>,
    /// Fraction of between-class variance explained by each retained
    /// discriminant direction (`λ_i / Σ_j λ_j`), as sklearn's
    /// `explained_variance_ratio_` for `solver='eigen'`.
    #[serde(default)]
    pub explained_variance_ratio: Array1<f64>,
    /// Global feature mean (used to center before projecting).
    pub xbar: Array1<f64>,
    pub n_features: usize,
}

impl Fit<f64> for LinearDiscriminantAnalysis {
    type Fitted = FittedLinearDiscriminantAnalysis;

    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        if x.nrows() != y.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but y has {} elements",
                x.nrows(),
                y.len()
            )));
        }
        let (classes, groups) = class_indices(y);
        if classes.len() < 2 {
            return Err(RustMlError::InvalidParameter(
                "need at least 2 classes for LDA".into(),
            ));
        }
        let d = x.ncols();
        let n = x.nrows();

        let means: Vec<Array1<f64>> = groups.iter().map(|g| class_mean(x, g)).collect();
        let priors: Vec<f64> = groups.iter().map(|g| g.len() as f64 / n as f64).collect();

        // Pooled within-class scatter.
        let mut sigma = Array2::<f64>::zeros((d, d));
        for (mu, g) in means.iter().zip(groups.iter()) {
            outer_subtract_accum(x, mu, g, &mut sigma);
        }
        // sklearn divides by (n - n_classes) (unbiased).
        let denom = (n - classes.len()) as f64;
        sigma.mapv_inplace(|v| v / denom.max(1.0));

        // Optional shrinkage toward diagonal mean.
        if self.shrinkage > 0.0 {
            let trace = (0..d).map(|i| sigma[[i, i]]).sum::<f64>() / d as f64;
            for i in 0..d {
                for j in 0..d {
                    if i == j {
                        sigma[[i, j]] =
                            (1.0 - self.shrinkage) * sigma[[i, j]] + self.shrinkage * trace;
                    } else {
                        sigma[[i, j]] *= 1.0 - self.shrinkage;
                    }
                }
            }
        }
        for i in 0..d {
            sigma[[i, i]] += self.reg;
        }

        // For each class compute sigma_inv @ mu_k as the linear coef.
        let mut coef = Vec::with_capacity(classes.len());
        let mut intercept = Vec::with_capacity(classes.len());
        for (mu, pi) in means.iter().zip(priors.iter()) {
            let s_inv_mu = solve_psd(&sigma, mu)?;
            let q = mu.dot(&s_inv_mu); // mu^T sigma_inv mu
            coef.push(s_inv_mu);
            intercept.push(-0.5 * q + pi.ln());
        }

        // Build the LDA projection: solve the generalised eigenproblem
        // Σ_b v = λ Σ_w v by Cholesky-whitening Σ_w.
        let mut xbar = Array1::<f64>::zeros(d);
        for (mu, g) in means.iter().zip(groups.iter()) {
            let w = g.len() as f64 / n as f64;
            for j in 0..d {
                xbar[j] += w * mu[j];
            }
        }
        let mut s_b = Array2::<f64>::zeros((d, d));
        for (k_idx, mu) in means.iter().enumerate() {
            let nk = groups[k_idx].len() as f64;
            for a in 0..d {
                for b in 0..d {
                    s_b[[a, b]] += nk * (mu[a] - xbar[a]) * (mu[b] - xbar[b]);
                }
            }
        }
        // sklearn's eigen solver works with the *biased* covariances
        // (divided by n): Σ_w = Σ_k π_k Cov_k and Σ_b = Σ_t − Σ_w. Rescale
        // the (n − K)-normalised pooled covariance accordingly so that the
        // scalings / eigenvalues match `solver='eigen'`.
        let nf = n as f64;
        let cov_scale = denom.max(1.0) / nf;
        let (scalings, explained_variance_ratio) = lda_projection(
            &Mat::from_fn(d, d, |i, j| sigma[[i, j]] * cov_scale),
            &Mat::from_fn(d, d, |i, j| s_b[[i, j]] / nf),
            (classes.len() - 1).min(d),
        )?;
        Ok(FittedLinearDiscriminantAnalysis {
            classes,
            means,
            priors,
            coef,
            intercept,
            scalings,
            explained_variance_ratio,
            xbar,
            n_features: d,
        })
    }
}

impl Transform<f64> for FittedLinearDiscriminantAnalysis {
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        let n = x.nrows();
        let k = self.scalings.ncols();
        let mut out = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            for c in 0..k {
                let mut s = 0.0;
                for j in 0..self.n_features {
                    s += (x[[i, j]] - self.xbar[j]) * self.scalings[[j, c]];
                }
                out[[i, c]] = s;
            }
        }
        Ok(out)
    }
}

impl Predict<f64> for FittedLinearDiscriminantAnalysis {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        let n = x.nrows();
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let row = x.row(i);
            let mut best = f64::NEG_INFINITY;
            let mut best_k = 0usize;
            for (k, (c, b)) in self.coef.iter().zip(self.intercept.iter()).enumerate() {
                let score = row.dot(c) + b;
                if score > best {
                    best = score;
                    best_k = k;
                }
            }
            out[i] = self.classes[best_k];
        }
        Ok(out)
    }
}

impl PredictProba<f64> for FittedLinearDiscriminantAnalysis {
    fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        let n = x.nrows();
        let k = self.classes.len();
        let mut p = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            let row = x.row(i);
            let mut logits = vec![0.0_f64; k];
            let mut max_l = f64::NEG_INFINITY;
            for (c_i, (c, b)) in self.coef.iter().zip(self.intercept.iter()).enumerate() {
                let s = row.dot(c) + b;
                logits[c_i] = s;
                if s > max_l {
                    max_l = s;
                }
            }
            let mut z = 0.0;
            for c_i in 0..k {
                let e = (logits[c_i] - max_l).exp();
                p[[i, c_i]] = e;
                z += e;
            }
            for c_i in 0..k {
                p[[i, c_i]] /= z;
            }
        }
        Ok(p)
    }
}

// ---------------------------------------------------------------------------
// QuadraticDiscriminantAnalysis (QDA)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct QuadraticDiscriminantAnalysis {
    pub reg: f64,
}

impl QuadraticDiscriminantAnalysis {
    pub fn new() -> Self {
        Self { reg: 1e-9 }
    }
    pub fn with_reg(mut self, r: f64) -> Self {
        self.reg = r;
        self
    }
}

impl Default for QuadraticDiscriminantAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedQuadraticDiscriminantAnalysis {
    pub classes: Vec<f64>,
    pub means: Vec<Array1<f64>>,
    pub priors: Vec<f64>,
    pub sigmas: Vec<Array2<f64>>,
    pub log_det: Vec<f64>,
    pub n_features: usize,
}

impl Fit<f64> for QuadraticDiscriminantAnalysis {
    type Fitted = FittedQuadraticDiscriminantAnalysis;

    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        if x.nrows() != y.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but y has {} elements",
                x.nrows(),
                y.len()
            )));
        }
        let (classes, groups) = class_indices(y);
        if classes.len() < 2 {
            return Err(RustMlError::InvalidParameter(
                "need at least 2 classes for QDA".into(),
            ));
        }
        let d = x.ncols();
        let n = x.nrows();

        let means: Vec<Array1<f64>> = groups.iter().map(|g| class_mean(x, g)).collect();
        let priors: Vec<f64> = groups.iter().map(|g| g.len() as f64 / n as f64).collect();

        let mut sigmas = Vec::with_capacity(classes.len());
        let mut log_det = Vec::with_capacity(classes.len());
        for (k, g) in groups.iter().enumerate() {
            let mut s = Array2::<f64>::zeros((d, d));
            outer_subtract_accum(x, &means[k], g, &mut s);
            let denom = (g.len() as f64 - 1.0).max(1.0);
            s.mapv_inplace(|v| v / denom);
            for i in 0..d {
                s[[i, i]] += self.reg;
            }
            log_det.push(log_det_chol(&s)?);
            sigmas.push(s);
        }

        Ok(FittedQuadraticDiscriminantAnalysis {
            classes,
            means,
            priors,
            sigmas,
            log_det,
            n_features: d,
        })
    }
}

impl Predict<f64> for FittedQuadraticDiscriminantAnalysis {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        let n = x.nrows();
        let d = self.n_features;
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut best = f64::NEG_INFINITY;
            let mut best_k = 0usize;
            for k in 0..self.classes.len() {
                // discriminant_k(x) = -0.5 (x-mu)^T Σ_k^{-1} (x-mu) - 0.5 log|Σ_k| + log π_k
                let mut diff = Array1::<f64>::zeros(d);
                for j in 0..d {
                    diff[j] = x[[i, j]] - self.means[k][j];
                }
                let s_inv_diff = solve_psd(&self.sigmas[k], &diff)?;
                let m = diff.dot(&s_inv_diff);
                let score = -0.5 * m - 0.5 * self.log_det[k] + self.priors[k].ln();
                if score > best {
                    best = score;
                    best_k = k;
                }
            }
            out[i] = self.classes[best_k];
        }
        Ok(out)
    }
}

impl PredictProba<f64> for FittedQuadraticDiscriminantAnalysis {
    fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        let n = x.nrows();
        let k = self.classes.len();
        let d = self.n_features;
        let mut p = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            let mut logits = vec![0.0_f64; k];
            let mut max_l = f64::NEG_INFINITY;
            for c_i in 0..k {
                let mut diff = Array1::<f64>::zeros(d);
                for j in 0..d {
                    diff[j] = x[[i, j]] - self.means[c_i][j];
                }
                let s_inv_diff = solve_psd(&self.sigmas[c_i], &diff)?;
                let m = diff.dot(&s_inv_diff);
                let score = -0.5 * m - 0.5 * self.log_det[c_i] + self.priors[c_i].ln();
                logits[c_i] = score;
                if score > max_l {
                    max_l = score;
                }
            }
            let mut z = 0.0;
            for c_i in 0..k {
                let e = (logits[c_i] - max_l).exp();
                p[[i, c_i]] = e;
                z += e;
            }
            for c_i in 0..k {
                p[[i, c_i]] /= z;
            }
        }
        Ok(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_lda_two_well_separated_classes() {
        let x = array![
            [0.0, 0.0],
            [0.5, 0.1],
            [-0.3, -0.2],
            [0.2, -0.1],
            [5.0, 5.0],
            [5.1, 4.9],
            [4.7, 5.3],
            [5.0, 5.2],
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let fitted = LinearDiscriminantAnalysis::new().fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_eq!(*p, *t);
        }
    }

    #[test]
    fn test_lda_transform_separates() {
        // 3-class problem in 2D — projects to 2 dims (n_classes-1=2).
        let x = array![
            [0.0, 0.0],
            [0.5, 0.0],
            [0.0, 0.3],
            [4.0, 0.0],
            [4.2, 0.1],
            [4.0, 0.3],
            [0.0, 4.0],
            [0.1, 4.2],
            [-0.1, 4.0],
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];
        let fitted = LinearDiscriminantAnalysis::new().fit(&x, &y).unwrap();
        let t = fitted.transform(&x).unwrap();
        assert_eq!(t.shape(), &[9, 2]);
        // After projection, within-class points should be much closer than
        // between-class.
        let d_within: f64 = (0..3)
            .map(|c| {
                let base = 3 * c;
                ((t[[base, 0]] - t[[base + 1, 0]]).powi(2)
                    + (t[[base, 1]] - t[[base + 1, 1]]).powi(2))
                .sqrt()
            })
            .sum::<f64>()
            / 3.0;
        let d_between: f64 =
            ((t[[0, 0]] - t[[3, 0]]).powi(2) + (t[[0, 1]] - t[[3, 1]]).powi(2)).sqrt();
        assert!(
            d_between > 5.0 * d_within,
            "within={d_within}, between={d_between}"
        );
    }

    /// 4 classes × 6 samples × 3 correlated features. Reference values from
    /// sklearn 1.9.1 `LinearDiscriminantAnalysis(solver='eigen')`:
    /// `scalings_`, `explained_variance_ratio_` and `(X - X̄) @ scalings_`.
    fn lda_four_class_data() -> (Array2<f64>, Array1<f64>) {
        let x = array![
            [0.001, 0.24, -0.194],
            [-0.891, -0.898, -1.226],
            [0.06, 1.108, -0.156],
            [-0.62, 0.02, 0.326],
            [0.105, -0.681, -0.216],
            [0.695, -0.658, -0.6],
            [0.099, -1.672, -2.427],
            [1.765, -0.655, -0.074],
            [2.157, 0.445, -2.411],
            [1.461, 0.138, -0.011],
            [0.47, -0.8, -1.347],
            [1.191, 0.863, -0.693],
            [0.467, 2.688, 0.636],
            [0.388, 2.021, 1.063],
            [-0.725, 1.326, 2.068],
            [-1.047, 1.759, 0.998],
            [-0.141, 3.215, 2.049],
            [-0.699, 1.34, 1.327],
            [0.811, 1.433, -1.448],
            [1.667, 2.551, -1.687],
            [1.203, 0.751, -1.442],
            [-0.187, -0.176, -2.055],
            [1.899, 2.455, -2.325],
            [0.205, 1.041, -3.413],
        ];
        let y = Array1::from_iter((0..24).map(|i| (i / 6) as f64));
        (x, y)
    }

    #[test]
    fn test_lda_eigen_matches_sklearn_multiclass() {
        let (x, y) = lda_four_class_data();
        let fitted = LinearDiscriminantAnalysis::new().fit(&x, &y).unwrap();
        let sk_scalings = array![
            [1.4993785967073294, 0.5228310055497781, 1.2282762684511621],
            [
                -1.0147543973140523,
                -1.2583908947696176,
                0.12988828311400066
            ],
            [-0.8905420995009952, 1.03077519813169, 0.5294813006159078],
        ];
        let sk_evr = [
            0.7871788587221186,
            0.19000275099452799,
            0.022818390283353612,
        ];
        let sk_t3 = array![
            [
                -0.4519415329518326,
                0.7789716695189309,
                -0.40332496149225683
            ],
            [0.28444270961364804, 0.6808952463444498, -2.193184961370043],
            [
                -1.2781256123957354,
                -0.24329514028465624,
                -0.19799334248728126
            ],
        ];
        assert_eq!(fitted.scalings.shape(), &[3, 3]);
        let t = fitted.transform(&x).unwrap();
        for c in 0..3 {
            // Eigenvectors are defined up to sign.
            let sign = if fitted.scalings[[0, c]] * sk_scalings[[0, c]] < 0.0 {
                -1.0
            } else {
                1.0
            };
            for r in 0..3 {
                let got = sign * fitted.scalings[[r, c]];
                assert!(
                    (got - sk_scalings[[r, c]]).abs() < 1e-6,
                    "scalings[{r},{c}] = {got}, sklearn {}",
                    sk_scalings[[r, c]]
                );
                let got_t = sign * t[[r, c]];
                assert!(
                    (got_t - sk_t3[[r, c]]).abs() < 1e-6,
                    "transform[{r},{c}] = {got_t}, sklearn {}",
                    sk_t3[[r, c]]
                );
            }
            assert!(
                (fitted.explained_variance_ratio[c] - sk_evr[c]).abs() < 1e-8,
                "evr[{c}] = {}, sklearn {}",
                fitted.explained_variance_ratio[c],
                sk_evr[c]
            );
        }
    }

    #[test]
    fn test_lda_scalings_solve_generalised_eigenproblem() {
        let (x, y) = lda_four_class_data();
        let fitted = LinearDiscriminantAnalysis::new().fit(&x, &y).unwrap();
        let (n, d) = x.dim();
        // Biased within-class and between-class covariances.
        let mut sw = Array2::<f64>::zeros((d, d));
        let mut sb = Array2::<f64>::zeros((d, d));
        for (k, mu) in fitted.means.iter().enumerate() {
            let idx: Vec<usize> = (0..n).filter(|&i| y[i] == k as f64).collect();
            outer_subtract_accum(&x, mu, &idx, &mut sw);
            let diff = mu - &fitted.xbar;
            for a in 0..d {
                for b in 0..d {
                    sb[[a, b]] += idx.len() as f64 * diff[a] * diff[b];
                }
            }
        }
        sw.mapv_inplace(|v| v / n as f64);
        sb.mapv_inplace(|v| v / n as f64);
        let v = &fitted.scalings;
        // Vᵀ Σ_w V = I.
        let vt_sw_v = v.t().dot(&sw).dot(v);
        for i in 0..v.ncols() {
            for j in 0..v.ncols() {
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((vt_sw_v[[i, j]] - expect).abs() < 1e-8, "VᵀΣ_wV[{i},{j}]");
            }
        }
        // Σ_b v = λ Σ_w v with λ = vᵀ Σ_b v, descending.
        let mut prev = f64::INFINITY;
        for c in 0..v.ncols() {
            let vc = v.column(c).to_owned();
            let lam = vc.dot(&sb.dot(&vc));
            assert!(lam <= prev + 1e-12);
            prev = lam;
            let resid = sb.dot(&vc) - sw.dot(&vc) * lam;
            assert!(resid.iter().all(|r| r.abs() < 1e-8), "residual {resid:?}");
        }
        // Projected within-class covariance is the identity.
        let t = fitted.transform(&x).unwrap();
        let mut wc = Array2::<f64>::zeros((v.ncols(), v.ncols()));
        for k in 0..fitted.classes.len() {
            let idx: Vec<usize> = (0..n).filter(|&i| y[i] == k as f64).collect();
            let mu = class_mean(&t, &idx);
            outer_subtract_accum(&t, &mu, &idx, &mut wc);
        }
        wc.mapv_inplace(|v| v / n as f64);
        for i in 0..v.ncols() {
            for j in 0..v.ncols() {
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (wc[[i, j]] - expect).abs() < 1e-8,
                    "within-class cov [{i},{j}]"
                );
            }
        }
    }

    #[test]
    fn test_qda_two_well_separated_classes() {
        let x = array![
            [0.0, 0.0],
            [0.5, 0.1],
            [-0.3, -0.2],
            [0.2, -0.1],
            [0.1, 0.2],
            [-0.1, 0.0],
            [5.0, 5.0],
            [5.1, 4.9],
            [4.7, 5.3],
            [5.0, 5.2],
            [5.2, 5.1],
            [4.8, 5.0],
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let fitted = QuadraticDiscriminantAnalysis::new().fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_eq!(*p, *t);
        }
    }
}

impl anofox_ml_core::ClassifierScore<f64> for FittedLinearDiscriminantAnalysis {}
impl anofox_ml_core::ClassifierScore<f64> for FittedQuadraticDiscriminantAnalysis {}

impl anofox_ml_core::PredictLogProba<f64> for FittedLinearDiscriminantAnalysis {}
impl anofox_ml_core::PredictLogProba<f64> for FittedQuadraticDiscriminantAnalysis {}

impl anofox_ml_core::DecisionFunction<f64> for FittedLinearDiscriminantAnalysis {
    fn decision_function(
        &self,
        x: &ndarray::Array2<f64>,
    ) -> anofox_ml_core::Result<ndarray::Array2<f64>> {
        if x.ncols() != self.n_features {
            return Err(anofox_ml_core::RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        let n = x.nrows();
        let k = self.classes.len();
        let mut out = ndarray::Array2::<f64>::zeros((n, k));
        for i in 0..n {
            let row = x.row(i);
            for (c_i, (c, b)) in self.coef.iter().zip(self.intercept.iter()).enumerate() {
                out[[i, c_i]] = row.dot(c) + b;
            }
        }
        Ok(out)
    }
}
