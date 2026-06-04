//! Orthogonal Matching Pursuit.
//!
//! Mirrors `sklearn.linear_model.OrthogonalMatchingPursuit`. Greedy sparse
//! regression: at each step, pick the feature most correlated with the
//! current residual, add it to the active set, then refit OLS on the active
//! set. Stop when `n_nonzero_coefs` features have been selected or the
//! residual norm falls below `tol`.

use faer::linalg::solvers::Solve;
use faer::{Mat, Side};
use ndarray::{Array1, Array2};
use rustml_core::{Fit, Predict, Result, RustMlError};

#[derive(Debug, Clone)]
pub struct OrthogonalMatchingPursuit {
    pub n_nonzero_coefs: Option<usize>,
    pub tol: Option<f64>,
    pub fit_intercept: bool,
}

impl OrthogonalMatchingPursuit {
    pub fn new() -> Self {
        Self {
            n_nonzero_coefs: None,
            tol: None,
            fit_intercept: true,
        }
    }
    pub fn with_n_nonzero_coefs(mut self, n: usize) -> Self {
        self.n_nonzero_coefs = Some(n);
        self
    }
    pub fn with_tol(mut self, t: f64) -> Self {
        self.tol = Some(t);
        self
    }
}

impl Default for OrthogonalMatchingPursuit {
    fn default() -> Self { Self::new() }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedOrthogonalMatchingPursuit {
    pub coef: Array1<f64>,
    pub intercept: f64,
    pub active_set: Vec<usize>,
    n_features: usize,
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

impl Fit<f64> for OrthogonalMatchingPursuit {
    type Fitted = FittedOrthogonalMatchingPursuit;

    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        if x.nrows() != y.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but y has {}", x.nrows(), y.len()
            )));
        }
        if x.is_empty() {
            return Err(RustMlError::EmptyInput("empty data".into()));
        }

        let d = x.ncols();
        let n = x.nrows();

        // Centre (and unit-normalize columns, matching sklearn's behavior).
        let (xc, yc, x_mean, y_mean) = {
            let n_f = n as f64;
            let mut x_mean = Array1::<f64>::zeros(d);
            for j in 0..d {
                x_mean[j] = x.column(j).sum() / n_f;
            }
            let y_mean = y.sum() / n_f;
            let mut xc = x.clone();
            if self.fit_intercept {
                for j in 0..d {
                    for i in 0..n {
                        xc[[i, j]] -= x_mean[j];
                    }
                }
            }
            let yc = if self.fit_intercept {
                y.mapv(|v| v - y_mean)
            } else {
                y.clone()
            };
            (xc, yc, x_mean, y_mean)
        };

        // Default n_nonzero = max(1, 0.1 * n_features) (sklearn rule).
        let target_k = self.n_nonzero_coefs.unwrap_or(((d as f64) * 0.1).ceil() as usize).max(1).min(d);

        let mut active: Vec<usize> = Vec::with_capacity(target_k);
        let mut residual = yc.clone();
        let mut coef_full = Array1::<f64>::zeros(d);

        for _step in 0..target_k {
            // Pick column with maximum |x_j' residual|.
            let mut best_j = 0usize;
            let mut best_abs = -1.0_f64;
            for j in 0..d {
                if active.contains(&j) {
                    continue;
                }
                let mut corr = 0.0;
                for i in 0..n {
                    corr += xc[[i, j]] * residual[i];
                }
                if corr.abs() > best_abs {
                    best_abs = corr.abs();
                    best_j = j;
                }
            }
            active.push(best_j);

            // Refit OLS on the active set.
            let m = active.len();
            let mut g = Array2::<f64>::zeros((m, m));
            let mut z = Array1::<f64>::zeros(m);
            for (ii, &a) in active.iter().enumerate() {
                let mut zi = 0.0;
                for k in 0..n {
                    zi += xc[[k, a]] * yc[k];
                }
                z[ii] = zi;
                for (jj, &b) in active.iter().enumerate() {
                    let mut g_ij = 0.0;
                    for k in 0..n {
                        g_ij += xc[[k, a]] * xc[[k, b]];
                    }
                    g[[ii, jj]] = g_ij;
                }
            }
            // Add tiny diagonal jitter for safety; OMP shouldn't need it
            // but ill-conditioned data can sneak in.
            for ii in 0..m {
                g[[ii, ii]] += 1e-12;
            }
            let beta_act = solve_psd(&g, &z)?;

            // Scatter into full coef.
            coef_full.fill(0.0);
            for (ii, &a) in active.iter().enumerate() {
                coef_full[a] = beta_act[ii];
            }

            // Update residual.
            for k in 0..n {
                let mut p = 0.0;
                for &a in &active {
                    p += xc[[k, a]] * coef_full[a];
                }
                residual[k] = yc[k] - p;
            }

            // Tolerance stop.
            if let Some(tol) = self.tol {
                let r2: f64 = residual.iter().map(|v| v * v).sum();
                if r2 < tol {
                    break;
                }
            }
        }

        let intercept = if self.fit_intercept {
            y_mean - x_mean.dot(&coef_full)
        } else {
            0.0
        };

        Ok(FittedOrthogonalMatchingPursuit {
            coef: coef_full,
            intercept,
            active_set: active,
            n_features: d,
        })
    }
}

impl Predict<f64> for FittedOrthogonalMatchingPursuit {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}", self.n_features, x.ncols()
            )));
        }
        Ok(x.dot(&self.coef).mapv(|v| v + self.intercept))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_omp_recovers_2_nonzero() {
        // y = 5*x0 + 3*x2 (features 1 and 3 are noise).
        let n = 60;
        let mut x = Array2::<f64>::zeros((n, 4));
        for i in 0..n {
            x[[i, 0]] = (i as f64) - 30.0;
            x[[i, 1]] = ((i * 7 % 11) as f64) - 5.0;
            x[[i, 2]] = ((i * 5 % 13) as f64) - 6.0;
            x[[i, 3]] = ((i * 3 % 7) as f64) - 3.0;
        }
        let y = x.column(0).mapv(|v| 5.0 * v) + x.column(2).mapv(|v| 3.0 * v);

        let fitted = OrthogonalMatchingPursuit::new()
            .with_n_nonzero_coefs(2)
            .fit(&x, &y).unwrap();
        // The two selected features must be 0 and 2 (in either order).
        let mut sel = fitted.active_set.clone();
        sel.sort();
        assert_eq!(sel, vec![0, 2]);
        assert!((fitted.coef[0] - 5.0).abs() < 0.1);
        assert!((fitted.coef[2] - 3.0).abs() < 0.1);
        assert_eq!(fitted.coef[1], 0.0);
        assert_eq!(fitted.coef[3], 0.0);
        let _ = array![1.0_f64];
    }
}
