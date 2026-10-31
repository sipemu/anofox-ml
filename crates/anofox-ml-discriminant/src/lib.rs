//! Linear and Quadratic Discriminant Analysis.
//!
//! Mirrors `sklearn.discriminant_analysis.{LinearDiscriminantAnalysis,
//! QuadraticDiscriminantAnalysis}`.
//!
//! - **LDA** assumes all classes share a common covariance Σ (decision
//!   function linear in `x`); solvers `svd` (default), `lsqr` and `eigen`,
//!   shrinkage (`Fixed` or `Auto` = Ledoit-Wolf) for `lsqr` / `eigen`,
//!   optional class `priors`.
//! - **QDA** estimates a separate covariance per class from the SVD of the
//!   centred class data, regularised as `(1 − r) S² + r` (sklearn
//!   `reg_param`).
//! - [`ledoit_wolf`] / [`oas`] covariance shrinkage estimators.

use anofox_ml_core::{Fit, Predict, PredictProba, Result, RustMlError, Transform};
use faer::linalg::solvers::{SelfAdjointEigen, Solve, Svd};
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
// Covariance estimators
// ---------------------------------------------------------------------------

/// Biased (maximum-likelihood) covariance of the rows of `x`.
fn empirical_covariance(x: &Array2<f64>) -> Array2<f64> {
    let n = x.nrows().max(1) as f64;
    let mean = x.mean_axis(ndarray::Axis(0)).expect("non-empty");
    let xc = x - &mean;
    xc.t().dot(&xc) / n
}

fn shrunk(cov: &Array2<f64>, shrinkage: f64) -> Array2<f64> {
    let d = cov.nrows();
    let mu = (0..d).map(|i| cov[[i, i]]).sum::<f64>() / d as f64;
    let mut out = cov * (1.0 - shrinkage);
    for i in 0..d {
        out[[i, i]] += shrinkage * mu;
    }
    out
}

/// Ledoit-Wolf shrunk covariance and its shrinkage coefficient
/// (sklearn `covariance.ledoit_wolf`, data centred).
pub fn ledoit_wolf(x: &Array2<f64>) -> (Array2<f64>, f64) {
    let (n, d) = x.dim();
    let mean = x.mean_axis(ndarray::Axis(0)).expect("non-empty");
    let xc = x - &mean;
    let x2 = xc.mapv(|v| v * v);
    let nf = n as f64;
    let emp_trace: Array1<f64> = x2.sum_axis(ndarray::Axis(0)) / nf;
    let mu = emp_trace.sum() / d as f64;
    let beta_ = x2.t().dot(&x2).sum();
    let delta_ = xc.t().dot(&xc).mapv(|v| v * v).sum() / (nf * nf);
    let beta = (beta_ / nf - delta_) / (d as f64 * nf);
    let delta = (delta_ - 2.0 * mu * emp_trace.sum() + d as f64 * mu * mu) / d as f64;
    let beta = beta.min(delta);
    let shrinkage = if beta == 0.0 { 0.0 } else { beta / delta };
    (shrunk(&empirical_covariance(x), shrinkage), shrinkage)
}

/// Oracle Approximating Shrinkage covariance and its coefficient (sklearn
/// `covariance.oas`).
pub fn oas(x: &Array2<f64>) -> (Array2<f64>, f64) {
    let (n, d) = x.dim();
    let emp = empirical_covariance(x);
    let alpha = emp.mapv(|v| v * v).mean().unwrap_or(0.0);
    let mu = (0..d).map(|i| emp[[i, i]]).sum::<f64>() / d as f64;
    let num = alpha + mu * mu;
    let den = (n as f64 + 1.0) * (alpha - mu * mu / d as f64);
    let shrinkage = if den == 0.0 {
        1.0
    } else {
        (num / den).min(1.0)
    };
    (shrunk(&emp, shrinkage), shrinkage)
}

// ---------------------------------------------------------------------------
// LinearDiscriminantAnalysis (LDA)
// ---------------------------------------------------------------------------

/// LDA solver (sklearn `solver`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum LdaSolver {
    /// SVD of the scaled within-class data: no covariance matrix, works with
    /// collinear / more features than samples (sklearn default).
    #[default]
    Svd,
    /// Least squares `Σ⁻¹ μ_k` on the (shrunk) pooled covariance.
    Lsqr,
    /// Generalised eigenproblem `Σ_b v = λ Σ_w v` (supports `transform`).
    Eigen,
}

/// Covariance shrinkage for the `Lsqr` / `Eigen` solvers.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub enum Shrinkage {
    #[default]
    None,
    /// Ledoit-Wolf on standardised features (sklearn `shrinkage='auto'`).
    Auto,
    /// Fixed shrinkage in `[0, 1]` toward `(tr Σ / d) I`.
    Fixed(f64),
}

#[derive(Debug, Clone)]
pub struct LinearDiscriminantAnalysis {
    pub solver: LdaSolver,
    pub shrinkage: Shrinkage,
    /// Class priors (in sorted class order); `None` = class frequencies.
    pub priors: Option<Vec<f64>>,
    /// Rank tolerance of the SVD solver (sklearn `tol`, default 1e-4).
    pub tol: f64,
}

impl LinearDiscriminantAnalysis {
    pub fn new() -> Self {
        Self {
            solver: LdaSolver::Svd,
            shrinkage: Shrinkage::None,
            priors: None,
            tol: 1e-4,
        }
    }
    pub fn with_solver(mut self, solver: LdaSolver) -> Self {
        self.solver = solver;
        self
    }
    /// Fixed shrinkage; switches the default `Svd` solver to `Lsqr` (sklearn
    /// rejects shrinkage with the SVD solver).
    pub fn with_shrinkage(mut self, s: f64) -> Self {
        self.shrinkage = Shrinkage::Fixed(s);
        if self.solver == LdaSolver::Svd {
            self.solver = LdaSolver::Lsqr;
        }
        self
    }
    /// Ledoit-Wolf shrinkage (`shrinkage='auto'`); switches `Svd` to `Lsqr`.
    pub fn with_shrinkage_auto(mut self) -> Self {
        self.shrinkage = Shrinkage::Auto;
        if self.solver == LdaSolver::Svd {
            self.solver = LdaSolver::Lsqr;
        }
        self
    }
    pub fn with_priors(mut self, priors: Vec<f64>) -> Self {
        self.priors = Some(priors);
        self
    }
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
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
    /// Per-class linear scores `x · coef_k + intercept_k` (softmax over
    /// classes gives the posterior).
    pub coef: Vec<Array1<f64>>,
    pub intercept: Vec<f64>,
    /// Projection used by `transform`: sklearn `scalings_` (`Svd`: whitened
    /// discriminant directions applied to `X − xbar`; `Eigen`: generalised
    /// eigenvectors applied to `X`, normalised so `Vᵀ Σ_w V = I`).
    pub scalings: Array2<f64>,
    /// sklearn `explained_variance_ratio_`.
    #[serde(default)]
    pub explained_variance_ratio: Array1<f64>,
    /// Prior-weighted mean of the class means (sklearn `xbar_`).
    pub xbar: Array1<f64>,
    #[serde(default)]
    pub solver: LdaSolver,
    pub n_features: usize,
}

/// sklearn's `_cov(X, shrinkage)`.
fn shrunk_class_cov(x: &Array2<f64>, shrinkage: Shrinkage) -> Result<Array2<f64>> {
    Ok(match shrinkage {
        Shrinkage::None => empirical_covariance(x),
        Shrinkage::Fixed(a) => {
            if !(0.0..=1.0).contains(&a) {
                return Err(RustMlError::InvalidParameter(
                    "shrinkage must be in [0, 1]".into(),
                ));
            }
            shrunk(&empirical_covariance(x), a)
        }
        Shrinkage::Auto => {
            let n = x.nrows() as f64;
            let mean = x.mean_axis(ndarray::Axis(0)).expect("non-empty");
            let sc: Array1<f64> = x
                .axis_iter(ndarray::Axis(1))
                .zip(mean.iter())
                .map(|(c, m)| {
                    let sd = (c.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n).sqrt();
                    if sd == 0.0 {
                        1.0
                    } else {
                        sd
                    }
                })
                .collect();
            let xs = (x - &mean) / &sc;
            let (s, _) = ledoit_wolf(&xs);
            let d = sc.len();
            Array2::from_shape_fn((d, d), |(i, j)| sc[i] * s[[i, j]] * sc[j])
        }
    })
}

fn to_mat(a: &Array2<f64>) -> Mat<f64> {
    Mat::from_fn(a.nrows(), a.ncols(), |i, j| a[[i, j]])
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
        let k = classes.len();
        if k < 2 {
            return Err(RustMlError::InvalidParameter(
                "need at least 2 classes for LDA".into(),
            ));
        }
        if self.solver == LdaSolver::Svd && self.shrinkage != Shrinkage::None {
            return Err(RustMlError::InvalidParameter(
                "shrinkage is not supported with the SVD solver".into(),
            ));
        }
        let (n, d) = x.dim();
        let means: Vec<Array1<f64>> = groups.iter().map(|g| class_mean(x, g)).collect();
        let priors: Vec<f64> = match &self.priors {
            Some(p) => {
                if p.len() != k || p.iter().any(|&v| !(v >= 0.0)) {
                    return Err(RustMlError::InvalidParameter(format!(
                        "priors must be {k} non-negative values"
                    )));
                }
                let s: f64 = p.iter().sum();
                p.iter().map(|v| v / s).collect()
            }
            None => groups.iter().map(|g| g.len() as f64 / n as f64).collect(),
        };
        let mut xbar = Array1::<f64>::zeros(d);
        for (mu, pi) in means.iter().zip(&priors) {
            xbar.scaled_add(*pi, mu);
        }
        let max_components = (k - 1).min(d);
        let (coef, intercept, scalings, evr) = match self.solver {
            LdaSolver::Svd => {
                // sklearn _solve_svd.
                let mut xc = Array2::<f64>::zeros((n, d));
                let mut row = 0;
                for (g, mu) in groups.iter().zip(&means) {
                    for &i in g {
                        xc.row_mut(row).assign(&(&x.row(i) - mu));
                        row += 1;
                    }
                }
                let std: Array1<f64> = xc
                    .axis_iter(ndarray::Axis(1))
                    .map(|c| {
                        let m = c.mean().unwrap_or(0.0);
                        let sd =
                            (c.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n as f64).sqrt();
                        if sd == 0.0 {
                            1.0
                        } else {
                            sd
                        }
                    })
                    .collect();
                let fac = 1.0 / (n as f64 - k as f64);
                let xs = (&xc / &std) * fac.sqrt();
                let svd = Svd::new_thin(to_mat(&xs).as_ref())
                    .map_err(|e| RustMlError::InvalidParameter(format!("SVD failed: {e:?}")))?;
                let s = svd.S().column_vector();
                let v = svd.V();
                let rank = (0..s.nrows()).filter(|&i| s[i] > self.tol).count();
                if rank == 0 {
                    return Err(RustMlError::InvalidParameter(
                        "within-class data has rank 0".into(),
                    ));
                }
                // scalings = (Vt[:rank] / std).T / S[:rank]   (d × rank)
                let sc1 = Array2::from_shape_fn((d, rank), |(j, r)| v[(j, r)] / std[j] / s[r]);
                let fac2 = if k == 1 { 1.0 } else { 1.0 / (k as f64 - 1.0) };
                let xb = Array2::from_shape_fn((k, d), |(c, j)| {
                    (n as f64 * priors[c] * fac2).sqrt() * (means[c][j] - xbar[j])
                });
                let m2 = xb.dot(&sc1);
                let svd2 = Svd::new_thin(to_mat(&m2).as_ref())
                    .map_err(|e| RustMlError::InvalidParameter(format!("SVD failed: {e:?}")))?;
                let s2 = svd2.S().column_vector();
                let v2 = svd2.V();
                let tot: f64 = (0..s2.nrows()).map(|i| s2[i] * s2[i]).sum();
                let evr = Array1::from_iter(
                    (0..s2.nrows().min(max_components)).map(|i| s2[i] * s2[i] / tot),
                );
                let rank2 = (0..s2.nrows())
                    .filter(|&i| s2[i] > self.tol * s2[0])
                    .count();
                let v2r = Array2::from_shape_fn((rank, rank2), |(a, b)| v2[(a, b)]);
                let scalings = sc1.dot(&v2r); // d × rank2
                let centered = Array2::from_shape_fn((k, d), |(c, j)| means[c][j] - xbar[j]);
                let cproj = centered.dot(&scalings); // k × rank2
                let coef_mat = cproj.dot(&scalings.t()); // k × d
                let coef: Vec<Array1<f64>> =
                    coef_mat.rows().into_iter().map(|r| r.to_owned()).collect();
                let intercept: Vec<f64> = (0..k)
                    .map(|c| {
                        -0.5 * cproj.row(c).mapv(|v| v * v).sum() + priors[c].ln()
                            - xbar.dot(&coef[c])
                    })
                    .collect();
                (coef, intercept, scalings, evr)
            }
            LdaSolver::Lsqr | LdaSolver::Eigen => {
                let mut sw = Array2::<f64>::zeros((d, d));
                for (g, pi) in groups.iter().zip(&priors) {
                    let xg = x.select(ndarray::Axis(0), g);
                    sw.scaled_add(*pi, &shrunk_class_cov(&xg, self.shrinkage)?);
                }
                let solve = |b: &Array1<f64>| -> Result<Array1<f64>> {
                    let m = to_mat(&sw);
                    let lu = m.partial_piv_lu();
                    let rhs = Mat::from_fn(d, 1, |i, _| b[i]);
                    let sol = lu.solve(&rhs);
                    let out = Array1::from_iter((0..d).map(|i| sol[(i, 0)]));
                    if out.iter().any(|v| !v.is_finite()) {
                        return Err(RustMlError::InvalidParameter(
                            "singular within-class covariance; use the Svd solver or shrinkage"
                                .into(),
                        ));
                    }
                    Ok(out)
                };
                if self.solver == LdaSolver::Lsqr {
                    let mut coef = Vec::with_capacity(k);
                    let mut intercept = Vec::with_capacity(k);
                    for (mu, pi) in means.iter().zip(&priors) {
                        let c = solve(mu)?;
                        intercept.push(-0.5 * mu.dot(&c) + pi.ln());
                        coef.push(c);
                    }
                    (coef, intercept, Array2::zeros((d, 0)), Array1::zeros(0))
                } else {
                    let st = shrunk_class_cov(x, self.shrinkage)?;
                    let sb = &st - &sw;
                    let (scalings, evr) = lda_projection(&to_mat(&sw), &to_mat(&sb), d)?;
                    let evr = evr.slice(ndarray::s![..max_components]).to_owned();
                    let means_mat = Array2::from_shape_fn((k, d), |(c, j)| means[c][j]);
                    let coef_mat = means_mat.dot(&scalings).dot(&scalings.t());
                    let coef: Vec<Array1<f64>> =
                        coef_mat.rows().into_iter().map(|r| r.to_owned()).collect();
                    let intercept: Vec<f64> = (0..k)
                        .map(|c| -0.5 * means[c].dot(&coef[c]) + priors[c].ln())
                        .collect();
                    (coef, intercept, scalings, evr)
                }
            }
        };
        Ok(FittedLinearDiscriminantAnalysis {
            classes,
            means,
            priors,
            coef,
            intercept,
            scalings,
            explained_variance_ratio: evr,
            xbar,
            solver: self.solver,
            n_features: d,
        })
    }
}

impl FittedLinearDiscriminantAnalysis {
    fn check(&self, x: &Array2<f64>) -> Result<()> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        Ok(())
    }

    /// Per-class scores `(n, K)`.
    fn scores(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.check(x)?;
        let k = self.coef.len();
        let w = Array2::from_shape_fn((self.n_features, k), |(j, c)| self.coef[c][j]);
        let mut s = x.dot(&w);
        for mut row in s.rows_mut() {
            for c in 0..k {
                row[c] += self.intercept[c];
            }
        }
        Ok(s)
    }
}

/// Row-wise softmax.
fn softmax_rows(mut s: Array2<f64>) -> Array2<f64> {
    for mut row in s.rows_mut() {
        let m = row.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        row.mapv_inplace(|v| (v - m).exp());
        let z = row.sum();
        row.mapv_inplace(|v| v / z);
    }
    s
}

fn argmax_labels(s: &Array2<f64>, classes: &[f64]) -> Array1<f64> {
    Array1::from_iter(s.rows().into_iter().map(|r| {
        let mut best = 0;
        for c in 1..r.len() {
            if r[c] > r[best] {
                best = c;
            }
        }
        classes[best]
    }))
}

impl Transform<f64> for FittedLinearDiscriminantAnalysis {
    /// sklearn `transform`: `(X − xbar) @ scalings` for `Svd`, `X @ scalings`
    /// for `Eigen` (first `min(K − 1, d)` components); `Lsqr` cannot
    /// transform.
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.check(x)?;
        let m = (self.classes.len() - 1)
            .min(self.n_features)
            .min(self.scalings.ncols());
        let sc = self.scalings.slice(ndarray::s![.., ..m]);
        match self.solver {
            LdaSolver::Svd => Ok((x - &self.xbar).dot(&sc)),
            LdaSolver::Eigen => Ok(x.dot(&sc)),
            LdaSolver::Lsqr => Err(RustMlError::InvalidParameter(
                "transform is not supported by the Lsqr solver".into(),
            )),
        }
    }
}

impl Predict<f64> for FittedLinearDiscriminantAnalysis {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        Ok(argmax_labels(&self.scores(x)?, &self.classes))
    }
}

impl PredictProba<f64> for FittedLinearDiscriminantAnalysis {
    fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        Ok(softmax_rows(self.scores(x)?))
    }
}

// ---------------------------------------------------------------------------
// QuadraticDiscriminantAnalysis (QDA)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct QuadraticDiscriminantAnalysis {
    /// sklearn `reg_param`: squared singular values become
    /// `(1 − r) S² + r`. Default 0.
    pub reg: f64,
    /// Class priors (sorted class order); `None` = class frequencies.
    pub priors: Option<Vec<f64>>,
    /// Rank tolerance (default 1e-4).
    pub tol: f64,
}

impl QuadraticDiscriminantAnalysis {
    pub fn new() -> Self {
        Self {
            reg: 0.0,
            priors: None,
            tol: 1e-4,
        }
    }
    pub fn with_reg(mut self, r: f64) -> Self {
        self.reg = r;
        self
    }
    pub fn with_priors(mut self, priors: Vec<f64>) -> Self {
        self.priors = Some(priors);
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
    /// Per class: right singular vectors of the centred class data
    /// (`d × r`, sklearn `rotations_`).
    pub rotations: Vec<Array2<f64>>,
    /// Per class: regularised variances along `rotations` (sklearn
    /// `scalings_`).
    pub scalings: Vec<Array1<f64>>,
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
        if !(0.0..=1.0).contains(&self.reg) {
            return Err(RustMlError::InvalidParameter(
                "reg_param must be in [0, 1]".into(),
            ));
        }
        let (classes, groups) = class_indices(y);
        let k = classes.len();
        if k < 2 {
            return Err(RustMlError::InvalidParameter(
                "need at least 2 classes for QDA".into(),
            ));
        }
        let (n, d) = x.dim();
        let means: Vec<Array1<f64>> = groups.iter().map(|g| class_mean(x, g)).collect();
        let priors: Vec<f64> = match &self.priors {
            Some(p) if p.len() == k && p.iter().all(|&v| v >= 0.0) => {
                let s: f64 = p.iter().sum();
                p.iter().map(|v| v / s).collect()
            }
            Some(_) => {
                return Err(RustMlError::InvalidParameter(format!(
                    "priors must be {k} non-negative values"
                )))
            }
            None => groups.iter().map(|g| g.len() as f64 / n as f64).collect(),
        };
        let mut rotations = Vec::with_capacity(k);
        let mut scalings = Vec::with_capacity(k);
        for (g, mu) in groups.iter().zip(&means) {
            if g.len() < 2 {
                return Err(RustMlError::InvalidParameter(
                    "every class needs at least 2 samples for QDA".into(),
                ));
            }
            let xc = &x.select(ndarray::Axis(0), g) - mu;
            let svd = Svd::new_thin(to_mat(&xc).as_ref())
                .map_err(|e| RustMlError::InvalidParameter(format!("SVD failed: {e:?}")))?;
            let s = svd.S().column_vector();
            let v = svd.V();
            let r = s.nrows();
            let denom = g.len() as f64 - 1.0;
            let s2 = Array1::from_iter(
                (0..r).map(|i| (1.0 - self.reg) * s[i] * s[i] / denom + self.reg),
            );
            if s2.iter().any(|&v| v <= 0.0) {
                return Err(RustMlError::InvalidParameter(
                    "a class covariance is singular; increase reg_param".into(),
                ));
            }
            rotations.push(Array2::from_shape_fn((d, r), |(a, b)| v[(a, b)]));
            scalings.push(s2);
        }
        Ok(FittedQuadraticDiscriminantAnalysis {
            classes,
            means,
            priors,
            rotations,
            scalings,
            n_features: d,
        })
    }
}

impl FittedQuadraticDiscriminantAnalysis {
    /// Per-class log posterior scores `(n, K)` (sklearn `_decision_function`).
    fn scores(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        let k = self.classes.len();
        let mut out = Array2::<f64>::zeros((x.nrows(), k));
        for c in 0..k {
            let w = &self.rotations[c] / &self.scalings[c].mapv(|s| s.sqrt());
            let z = (x - &self.means[c]).dot(&w);
            let logdet: f64 = self.scalings[c].mapv(f64::ln).sum();
            let col = z.mapv(|v| v * v).sum_axis(ndarray::Axis(1));
            for i in 0..x.nrows() {
                out[[i, c]] = -0.5 * (col[i] + logdet) + self.priors[c].ln();
            }
        }
        Ok(out)
    }
}

impl Predict<f64> for FittedQuadraticDiscriminantAnalysis {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        Ok(argmax_labels(&self.scores(x)?, &self.classes))
    }
}

impl PredictProba<f64> for FittedQuadraticDiscriminantAnalysis {
    fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        Ok(softmax_rows(self.scores(x)?))
    }
}

impl anofox_ml_core::ClassifierScore<f64> for FittedLinearDiscriminantAnalysis {}
impl anofox_ml_core::ClassifierScore<f64> for FittedQuadraticDiscriminantAnalysis {}

impl anofox_ml_core::PredictLogProba<f64> for FittedLinearDiscriminantAnalysis {}
impl anofox_ml_core::PredictLogProba<f64> for FittedQuadraticDiscriminantAnalysis {}

impl anofox_ml_core::DecisionFunction<f64> for FittedLinearDiscriminantAnalysis {
    fn decision_function(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.scores(x)
    }
}

impl anofox_ml_core::DecisionFunction<f64> for FittedQuadraticDiscriminantAnalysis {
    fn decision_function(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.scores(x)
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
    fn outer_subtract_accum(
        x: &Array2<f64>,
        mu: &Array1<f64>,
        idx: &[usize],
        acc: &mut Array2<f64>,
    ) {
        for &i in idx {
            let d = &x.row(i) - mu;
            for a in 0..d.len() {
                for b in 0..d.len() {
                    acc[[a, b]] += d[a] * d[b];
                }
            }
        }
    }

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
        let fitted = LinearDiscriminantAnalysis::new()
            .with_solver(LdaSolver::Eigen)
            .fit(&x, &y)
            .unwrap();
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
        // sklearn transform for solver='eigen' is X @ scalings_.
        let sk_t3 = array![
            [
                -0.06927650945547213,
                -0.5014613721767063,
                -0.0703179081036748
            ],
            [0.6671077331100084, -0.5995377953511876, -1.8601779079814609],
            [
                -0.8954605888993751,
                -1.5237281819802935,
                0.13501371090130085
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
        let fitted = LinearDiscriminantAnalysis::new()
            .with_solver(LdaSolver::Eigen)
            .fit(&x, &y)
            .unwrap();
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

    fn assert_rows(got: &Array2<f64>, want: &[[f64; 4]], tol: f64, ctx: &str) {
        for (i, row) in want.iter().enumerate() {
            for (j, &w) in row.iter().enumerate() {
                assert!(
                    (got[[i, j]] - w).abs() < tol,
                    "{ctx}[{i},{j}] {} vs {w}",
                    got[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_solvers_and_shrinkage_match_sklearn() {
        let (x, y) = lda_four_class_data();
        let x3 = x.slice(ndarray::s![..3, ..]).to_owned();
        // sklearn 1.8 LinearDiscriminantAnalysis() (svd solver)
        let svd = LinearDiscriminantAnalysis::new().fit(&x, &y).unwrap();
        assert_rows(
            &svd.predict_proba(&x3).unwrap(),
            &[
                [
                    0.9780605632366738,
                    0.011593410451172672,
                    0.0024586569071429463,
                    0.007887369405010548,
                ],
                [
                    0.9766208676052385,
                    0.01176264990445199,
                    5.424534271346096e-05,
                    0.011562237147596022,
                ],
                [
                    0.8572864267802185,
                    0.0017718407237539,
                    0.10508167073813209,
                    0.03586006175789555,
                ],
            ],
            1e-9,
            "svd proba",
        );
        let t = svd.transform(&x3).unwrap();
        let sk_t: [[f64; 3]; 2] = [
            [
                -0.41256428711863896,
                0.7111005917549632,
                -0.3681836323570192,
            ],
            [0.2596594806221435, 0.6215694762014881, -2.0020947935391153],
        ];
        for i in 0..2 {
            for c in 0..3 {
                assert!(
                    (t[[i, c]].abs() - sk_t[i][c].abs()).abs() < 1e-9,
                    "svd transform"
                );
            }
        }
        let sk_evr = [
            0.7871788587221183,
            0.19000275099452804,
            0.022818390283353578,
        ];
        for c in 0..3 {
            assert!((svd.explained_variance_ratio[c] - sk_evr[c]).abs() < 1e-9);
        }
        // lsqr + shrinkage='auto' (Ledoit-Wolf on standardised data)
        let lsqr = LinearDiscriminantAnalysis::new()
            .with_shrinkage_auto()
            .fit(&x, &y)
            .unwrap();
        assert_rows(
            &lsqr.predict_proba(&x3).unwrap(),
            &[
                [
                    0.9417371316258113,
                    0.048230613875241954,
                    0.005545289472407275,
                    0.004486965026539587,
                ],
                [
                    0.9686620539937553,
                    0.029247071134854866,
                    7.917035146385165e-06,
                    0.0020829578362434307,
                ],
                [
                    0.8357925089862442,
                    0.03170419253125628,
                    0.10765747823344439,
                    0.024845820249055062,
                ],
            ],
            1e-9,
            "lsqr auto proba",
        );
        assert!(LinearDiscriminantAnalysis::new()
            .with_solver(LdaSolver::Svd)
            .with_shrinkage(0.3)
            .fit(&x, &y)
            .is_ok()); // with_shrinkage switches Svd to Lsqr
        let mut bad = LinearDiscriminantAnalysis::new().with_shrinkage(0.3);
        bad.solver = LdaSolver::Svd;
        assert!(bad.fit(&x, &y).is_err());
        // QDA(reg_param=0.1)
        let q = QuadraticDiscriminantAnalysis::new()
            .with_reg(0.1)
            .fit(&x, &y)
            .unwrap();
        assert_rows(
            &q.predict_proba(&x3).unwrap(),
            &[
                [
                    0.9745002975245112,
                    0.018200345548335287,
                    0.0018739184336209098,
                    0.00542543849353244,
                ],
                [
                    0.9019246846118524,
                    0.03830137150943326,
                    9.80201661653137e-07,
                    0.05977296367705277,
                ],
                [
                    0.9100904850937166,
                    0.0061166761132903326,
                    0.07983296901423033,
                    0.003959869778762868,
                ],
            ],
            1e-9,
            "qda proba",
        );
        // Shrinkage estimators.
        assert!((ledoit_wolf(&x).1 - 0.3861610147387781).abs() < 1e-12);
        assert!((oas(&x).1 - 0.5087683839340374).abs() < 1e-12);
    }

    #[test]
    fn test_svd_solver_handles_collinear_features() {
        let (x, y) = lda_four_class_data();
        let dup = ndarray::concatenate![ndarray::Axis(1), x, x.slice(ndarray::s![.., 0..1])];
        let f = LinearDiscriminantAnalysis::new().fit(&dup, &y).unwrap();
        assert_eq!(
            f.predict(&dup).unwrap(),
            LinearDiscriminantAnalysis::new()
                .fit(&x, &y)
                .unwrap()
                .predict(&x)
                .unwrap()
        );
    }
}
