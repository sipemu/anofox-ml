//! Linear and Quadratic Discriminant Analysis.
//!
//! Mirrors `sklearn.discriminant_analysis.{LinearDiscriminantAnalysis,
//! QuadraticDiscriminantAnalysis}`.
//!
//! - **LDA** assumes all classes share a common covariance Σ. Decision
//!   function is linear in `x`.
//! - **QDA** estimates a separate covariance Σ_k per class.

use faer::linalg::solvers::Solve;
use faer::{Mat, Side};
use ndarray::{Array1, Array2};
use rustml_core::{Fit, Predict, Result, RustMlError};

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

fn outer_subtract_accum(
    x: &Array2<f64>,
    mu: &Array1<f64>,
    idx: &[usize],
    accum: &mut Array2<f64>,
) {
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
        Self { shrinkage: 0.0, reg: 1e-9 }
    }
    pub fn with_shrinkage(mut self, s: f64) -> Self { self.shrinkage = s; self }
}

impl Default for LinearDiscriminantAnalysis {
    fn default() -> Self { Self::new() }
}

#[derive(Debug, Clone)]
pub struct FittedLinearDiscriminantAnalysis {
    pub classes: Vec<f64>,
    pub means: Vec<Array1<f64>>,
    pub priors: Vec<f64>,
    pub coef: Vec<Array1<f64>>, // sigma_inv @ mu_k
    pub intercept: Vec<f64>,    // -0.5 * mu_k^T sigma_inv mu_k + log(pi_k)
    pub n_features: usize,
}

impl Fit<f64> for LinearDiscriminantAnalysis {
    type Fitted = FittedLinearDiscriminantAnalysis;

    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        if x.nrows() != y.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but y has {} elements", x.nrows(), y.len()
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
                        sigma[[i, j]] = (1.0 - self.shrinkage) * sigma[[i, j]] + self.shrinkage * trace;
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

        Ok(FittedLinearDiscriminantAnalysis {
            classes,
            means,
            priors,
            coef,
            intercept,
            n_features: d,
        })
    }
}

impl Predict<f64> for FittedLinearDiscriminantAnalysis {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}", self.n_features, x.ncols()
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

// ---------------------------------------------------------------------------
// QuadraticDiscriminantAnalysis (QDA)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct QuadraticDiscriminantAnalysis {
    pub reg: f64,
}

impl QuadraticDiscriminantAnalysis {
    pub fn new() -> Self { Self { reg: 1e-9 } }
    pub fn with_reg(mut self, r: f64) -> Self { self.reg = r; self }
}

impl Default for QuadraticDiscriminantAnalysis {
    fn default() -> Self { Self::new() }
}

#[derive(Debug, Clone)]
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
                "X has {} rows but y has {} elements", x.nrows(), y.len()
            )));
        }
        let (classes, groups) = class_indices(y);
        if classes.len() < 2 {
            return Err(RustMlError::InvalidParameter("need at least 2 classes for QDA".into()));
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
                "expected {} features, got {}", self.n_features, x.ncols()
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_lda_two_well_separated_classes() {
        let x = array![
            [0.0, 0.0], [0.5, 0.1], [-0.3, -0.2], [0.2, -0.1],
            [5.0, 5.0], [5.1, 4.9], [4.7, 5.3], [5.0, 5.2],
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let fitted = LinearDiscriminantAnalysis::new().fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_eq!(*p, *t);
        }
    }

    #[test]
    fn test_qda_two_well_separated_classes() {
        let x = array![
            [0.0, 0.0], [0.5, 0.1], [-0.3, -0.2], [0.2, -0.1], [0.1, 0.2], [-0.1, 0.0],
            [5.0, 5.0], [5.1, 4.9], [4.7, 5.3], [5.0, 5.2], [5.2, 5.1], [4.8, 5.0],
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let fitted = QuadraticDiscriminantAnalysis::new().fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_eq!(*p, *t);
        }
    }
}
