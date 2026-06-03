//! Bayesian Ridge Regression and Automatic Relevance Determination (ARD).
//!
//! Mirrors `sklearn.linear_model.BayesianRidge` and `ARDRegression`. The model
//! is `y ~ N(Xβ + b, 1/α)` with prior `β_j ~ N(0, 1/λ_j)`. BayesianRidge ties
//! all `λ_j = λ`; ARD has a per-feature `λ_j`. Hyperparameters are estimated
//! by evidence (type-II) maximization.

use faer::linalg::solvers::Solve;
use faer::{Mat, Side};
use ndarray::{Array1, Array2};
use rustml_core::{Fit, Predict, Result, RustMlError};

fn center(x: &Array2<f64>, y: &Array1<f64>) -> (Array2<f64>, Array1<f64>, Array1<f64>, f64) {
    let n = x.nrows() as f64;
    let mut x_mean = Array1::<f64>::zeros(x.ncols());
    for j in 0..x.ncols() {
        x_mean[j] = x.column(j).sum() / n;
    }
    let y_mean = y.sum() / n;
    let mut xc = x.clone();
    for j in 0..x.ncols() {
        for i in 0..x.nrows() {
            xc[[i, j]] -= x_mean[j];
        }
    }
    let yc = y.mapv(|v| v - y_mean);
    (xc, yc, x_mean, y_mean)
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

fn invert_psd(a: &Array2<f64>) -> Result<Array2<f64>> {
    let n = a.nrows();
    let am = Mat::from_fn(n, n, |i, j| a[[i, j]]);
    let llt = faer::linalg::solvers::Llt::new(am.as_ref(), Side::Lower)
        .map_err(|e| RustMlError::InvalidParameter(format!("LLT failed: {e:?}")))?;
    // Solve A * X = I column by column.
    let mut out = Array2::<f64>::zeros((n, n));
    for col in 0..n {
        let mut e = Array1::<f64>::zeros(n);
        e[col] = 1.0;
        let em = Mat::from_fn(n, 1, |i, _| e[i]);
        let s = llt.solve(&em);
        for i in 0..n {
            out[[i, col]] = s[(i, 0)];
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// BayesianRidge
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct BayesianRidge {
    pub max_iter: usize,
    pub tol: f64,
    pub alpha_1: f64,
    pub alpha_2: f64,
    pub lambda_1: f64,
    pub lambda_2: f64,
}

impl BayesianRidge {
    pub fn new() -> Self {
        Self {
            max_iter: 300,
            tol: 1e-3,
            alpha_1: 1e-6,
            alpha_2: 1e-6,
            lambda_1: 1e-6,
            lambda_2: 1e-6,
        }
    }
}

impl Default for BayesianRidge {
    fn default() -> Self { Self::new() }
}

#[derive(Debug, Clone)]
pub struct FittedBayesianRidge {
    pub coef: Array1<f64>,
    pub intercept: f64,
    pub alpha: f64,
    pub lambda: f64,
    pub n_iter: usize,
    n_features: usize,
}

impl Fit<f64> for BayesianRidge {
    type Fitted = FittedBayesianRidge;

    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        if x.nrows() != y.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but y has {}", x.nrows(), y.len()
            )));
        }
        if x.is_empty() {
            return Err(RustMlError::EmptyInput("empty data".into()));
        }

        let n = x.nrows() as f64;
        let d = x.ncols();

        let (xc, yc, x_mean, y_mean) = center(x, y);

        // X'X and X'y once.
        let mut xtx = Array2::<f64>::zeros((d, d));
        for i in 0..d {
            for j in 0..d {
                let mut s = 0.0;
                for k in 0..xc.nrows() {
                    s += xc[[k, i]] * xc[[k, j]];
                }
                xtx[[i, j]] = s;
            }
        }
        let mut xty = Array1::<f64>::zeros(d);
        for j in 0..d {
            let mut s = 0.0;
            for k in 0..xc.nrows() {
                s += xc[[k, j]] * yc[k];
            }
            xty[j] = s;
        }

        // Initialize α, λ.
        let var_y = yc.iter().map(|v| v * v).sum::<f64>() / n.max(1.0);
        let mut alpha = 1.0 / var_y.max(1e-12);
        let mut lambda = 1.0_f64;

        let mut coef = Array1::<f64>::zeros(d);
        let mut prev_coef = coef.clone();
        let mut n_iter = 0;

        for iter in 0..self.max_iter {
            n_iter = iter + 1;
            // S^-1 = α X'X + λ I; μ = α S X'y
            let mut s_inv = xtx.clone();
            for i in 0..d {
                for j in 0..d {
                    s_inv[[i, j]] *= alpha;
                }
                s_inv[[i, i]] += lambda;
            }
            // Solve for μ from S^-1 μ = α X'y.
            let rhs = xty.mapv(|v| alpha * v);
            coef = solve_psd(&s_inv, &rhs)?;

            // γ = sum λ_i / (λ_i + λ) where λ_i are eigenvalues of α X'X.
            // We approximate γ via trace(S * α X'X) = d - λ tr(S).
            let s = invert_psd(&s_inv)?;
            let mut trace_s = 0.0;
            for i in 0..d {
                trace_s += s[[i, i]];
            }
            let gamma: f64 = (d as f64) - lambda * trace_s;

            // Update λ = (γ + 2λ₁) / (||μ||² + 2λ₂)
            let sq_coef: f64 = coef.iter().map(|v| v * v).sum();
            lambda = (gamma + 2.0 * self.lambda_1) / (sq_coef + 2.0 * self.lambda_2);

            // Compute residual ||y - Xμ||²
            let mut resid_sq = 0.0;
            for i in 0..xc.nrows() {
                let mut p = 0.0;
                for j in 0..d {
                    p += xc[[i, j]] * coef[j];
                }
                let r = yc[i] - p;
                resid_sq += r * r;
            }
            alpha = (n - gamma + 2.0 * self.alpha_1) / (resid_sq + 2.0 * self.alpha_2);

            // Convergence on coefficients.
            let dmax = coef
                .iter()
                .zip(prev_coef.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            if dmax < self.tol {
                break;
            }
            prev_coef = coef.clone();
        }

        let intercept = y_mean - x_mean.dot(&coef);
        Ok(FittedBayesianRidge {
            coef,
            intercept,
            alpha,
            lambda,
            n_iter,
            n_features: d,
        })
    }
}

impl Predict<f64> for FittedBayesianRidge {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}", self.n_features, x.ncols()
            )));
        }
        Ok(x.dot(&self.coef).mapv(|v| v + self.intercept))
    }
}

// ---------------------------------------------------------------------------
// ARDRegression (per-feature lambda; sparsity-inducing)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ARDRegression {
    pub max_iter: usize,
    pub tol: f64,
    pub alpha_1: f64,
    pub alpha_2: f64,
    pub lambda_1: f64,
    pub lambda_2: f64,
    /// Drop features whose λ_j exceeds this (so prior precision is huge).
    pub threshold_lambda: f64,
}

impl ARDRegression {
    pub fn new() -> Self {
        Self {
            max_iter: 300,
            tol: 1e-3,
            alpha_1: 1e-6,
            alpha_2: 1e-6,
            lambda_1: 1e-6,
            lambda_2: 1e-6,
            threshold_lambda: 1e4,
        }
    }
}

impl Default for ARDRegression {
    fn default() -> Self { Self::new() }
}

#[derive(Debug, Clone)]
pub struct FittedARDRegression {
    pub coef: Array1<f64>,
    pub intercept: f64,
    pub lambdas: Array1<f64>,
    pub alpha: f64,
    pub n_iter: usize,
    n_features: usize,
}

impl Fit<f64> for ARDRegression {
    type Fitted = FittedARDRegression;

    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        if x.nrows() != y.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but y has {}", x.nrows(), y.len()
            )));
        }
        let n = x.nrows() as f64;
        let d = x.ncols();
        let (xc, yc, x_mean, y_mean) = center(x, y);

        let mut alpha = 1.0_f64;
        let mut lambdas = Array1::<f64>::ones(d);
        let mut coef = Array1::<f64>::zeros(d);
        let mut n_iter = 0;

        // Pre-compute X'X (n × d × d) and X'y (d).
        let mut xtx = Array2::<f64>::zeros((d, d));
        for i in 0..d {
            for j in 0..d {
                let mut s = 0.0;
                for k in 0..xc.nrows() {
                    s += xc[[k, i]] * xc[[k, j]];
                }
                xtx[[i, j]] = s;
            }
        }
        let mut xty = Array1::<f64>::zeros(d);
        for j in 0..d {
            let mut s = 0.0;
            for k in 0..xc.nrows() {
                s += xc[[k, j]] * yc[k];
            }
            xty[j] = s;
        }

        let mut prev_coef = coef.clone();
        for iter in 0..self.max_iter {
            n_iter = iter + 1;

            // Keep only "active" features (λ_j < threshold).
            let active: Vec<usize> = (0..d)
                .filter(|&j| lambdas[j] < self.threshold_lambda)
                .collect();
            if active.is_empty() {
                break;
            }
            let m = active.len();

            // Build A = α X_act' X_act + diag(λ_act)
            let mut a = Array2::<f64>::zeros((m, m));
            for (ii, &i) in active.iter().enumerate() {
                for (jj, &j) in active.iter().enumerate() {
                    a[[ii, jj]] = alpha * xtx[[i, j]];
                }
                a[[ii, ii]] += lambdas[i];
            }
            let rhs: Array1<f64> = Array1::from_vec(active.iter().map(|&j| alpha * xty[j]).collect());
            let mu_act = solve_psd(&a, &rhs)?;
            let s = invert_psd(&a)?;

            // Scatter into full coef.
            coef.fill(0.0);
            for (ii, &i) in active.iter().enumerate() {
                coef[i] = mu_act[ii];
            }

            // Update λ_j for active features.
            for (ii, &i) in active.iter().enumerate() {
                let var = s[[ii, ii]];
                let denom = mu_act[ii].powi(2) + var + 2.0 * self.lambda_2;
                lambdas[i] = (1.0 + 2.0 * self.lambda_1) / denom.max(1e-12);
            }

            // Update α: γ = m - sum_i λ_i s_ii  (effective # parameters)
            let mut tr = 0.0;
            for ii in 0..m {
                tr += lambdas[active[ii]] * s[[ii, ii]];
            }
            let gamma = m as f64 - tr;

            // residual
            let mut rss = 0.0;
            for k in 0..xc.nrows() {
                let mut p = 0.0;
                for j in 0..d {
                    p += xc[[k, j]] * coef[j];
                }
                let r = yc[k] - p;
                rss += r * r;
            }
            alpha = (n - gamma + 2.0 * self.alpha_1) / (rss + 2.0 * self.alpha_2);

            // Convergence
            let dmax = coef
                .iter()
                .zip(prev_coef.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            if dmax < self.tol {
                break;
            }
            prev_coef = coef.clone();
        }

        let intercept = y_mean - x_mean.dot(&coef);
        Ok(FittedARDRegression {
            coef,
            intercept,
            lambdas,
            alpha,
            n_iter,
            n_features: d,
        })
    }
}

impl Predict<f64> for FittedARDRegression {
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
    fn test_bayesian_ridge_recovers_simple_line() {
        // y = 2 + 3x
        let x = Array2::from_shape_vec((10, 1), (0..10).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_vec((0..10).map(|i| 2.0 + 3.0 * i as f64).collect());

        let fitted = BayesianRidge::new().fit(&x, &y).unwrap();
        assert!((fitted.coef[0] - 3.0).abs() < 0.1);
        assert!((fitted.intercept - 2.0).abs() < 0.2);
    }

    #[test]
    fn test_ard_drops_irrelevant_features() {
        // y = 5*x0 + 0*x1 + 0*x2 — ARD should shrink x1, x2.
        use ndarray::array;
        let n = 80;
        let mut x = Array2::<f64>::zeros((n, 3));
        for i in 0..n {
            x[[i, 0]] = i as f64 - 40.0;
            x[[i, 1]] = ((i * 7 % 13) as f64) - 6.0;
            x[[i, 2]] = ((i * 5 % 11) as f64) - 5.0;
        }
        let y = x.column(0).mapv(|v| 5.0 * v);
        let fitted = ARDRegression::new().fit(&x, &y).unwrap();
        assert!((fitted.coef[0] - 5.0).abs() < 0.5);
        assert!(fitted.coef[1].abs() < 0.5);
        assert!(fitted.coef[2].abs() < 0.5);
        let _ = array![1.0]; // silence unused-import warning if any
    }
}
