//! Orthogonal Matching Pursuit.
//!
//! Mirrors `sklearn.linear_model.OrthogonalMatchingPursuit`. Greedy sparse
//! regression: at each step, pick the feature most correlated with the
//! current residual, add it to the active set, then refit least squares on
//! the active set. Columns are centred (with `fit_intercept`) but not
//! normalised, as in sklearn, so the selection depends on feature scale.
//!
//! The active-set Gram matrix is never rebuilt: its Cholesky factor grows
//! by one rank-1 row per step (sklearn's `_cholesky_omp`), so a step costs
//! one `Xᵀr` product plus `O(n·k + k²)`.

use anofox_ml_core::{Fit, Predict, Result, RustMlError};
use ndarray::{Array1, Array2, Axis};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OrthogonalMatchingPursuit {
    /// Number of non-zero coefficients. Default `max(⌊0.1·d⌋, 1)`.
    /// Ignored when `tol` is set.
    pub n_nonzero_coefs: Option<usize>,
    /// Maximum squared norm of the residual; overrides `n_nonzero_coefs`.
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
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
}

impl Default for OrthogonalMatchingPursuit {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedOrthogonalMatchingPursuit {
    pub coef: Array1<f64>,
    pub intercept: f64,
    /// Selected features, in selection order.
    pub active_set: Vec<usize>,
    n_features: usize,
}

/// Solve `L Lᵀ x = b` for lower-triangular `L` (row-major `m × m` prefix of
/// a `cap × cap` buffer).
fn cholesky_solve(l: &Array2<f64>, m: usize, b: &[f64]) -> Vec<f64> {
    let mut z = b.to_vec();
    for i in 0..m {
        let s: f64 = (0..i).map(|k| l[[i, k]] * z[k]).sum();
        z[i] = (z[i] - s) / l[[i, i]];
    }
    for i in (0..m).rev() {
        let s: f64 = (i + 1..m).map(|k| l[[k, i]] * z[k]).sum();
        z[i] = (z[i] - s) / l[[i, i]];
    }
    z
}

impl Fit<f64> for OrthogonalMatchingPursuit {
    type Fitted = FittedOrthogonalMatchingPursuit;

    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        crate::convert::check_xy(x, y)?;
        if let Some(t) = self.tol {
            if !(t >= 0.0) {
                return Err(RustMlError::InvalidParameter(
                    "tol must be non-negative".into(),
                ));
            }
        }
        let (n, d) = x.dim();

        let (x_mean, y_mean) = if self.fit_intercept {
            (x.mean_axis(Axis(0)).unwrap(), y.mean().unwrap())
        } else {
            (Array1::zeros(d), 0.0)
        };
        let xc = x - &x_mean;
        let yc = y - y_mean;

        let max_k = match (self.tol, self.n_nonzero_coefs) {
            (Some(_), _) => d,
            (None, Some(k)) => {
                if k == 0 || k > d {
                    return Err(RustMlError::InvalidParameter(format!(
                        "n_nonzero_coefs must be in 1..={d}, got {k}"
                    )));
                }
                k
            }
            (None, None) => (d / 10).max(1),
        }
        .min(n.max(1));

        let col_sq: Array1<f64> = xc.map_axis(Axis(0), |c| c.dot(&c));
        let xty = xc.t().dot(&yc);
        let mut l = Array2::<f64>::zeros((max_k, max_k));
        let mut active: Vec<usize> = Vec::with_capacity(max_k);
        let mut is_active = vec![false; d];
        let mut gamma: Vec<f64> = Vec::new();
        let mut residual = yc.clone();

        while active.len() < max_k {
            let corr = xc.t().dot(&residual);
            let mut best: Option<(usize, f64)> = None;
            for (j, &c) in corr.iter().enumerate() {
                if !is_active[j] && best.is_none_or(|(_, b)| c.abs() > b) {
                    best = Some((j, c.abs()));
                }
            }
            let Some((lam, c)) = best else { break };
            if c * c < f64::MIN_POSITIVE {
                break;
            }

            // Grow the Cholesky factor of the active Gram matrix by one row.
            let m = active.len();
            if m > 0 {
                let xl = xc.column(lam);
                let mut w: Vec<f64> = active.iter().map(|&a| xc.column(a).dot(&xl)).collect();
                for i in 0..m {
                    let s: f64 = (0..i).map(|k| l[[i, k]] * w[k]).sum();
                    w[i] = (w[i] - s) / l[[i, i]];
                }
                let v = col_sq[lam] - w.iter().map(|t| t * t).sum::<f64>();
                if v <= f64::EPSILON * col_sq[lam].max(f64::MIN_POSITIVE) {
                    // Linear dependence: sklearn stops prematurely here too.
                    break;
                }
                for (k, wk) in w.into_iter().enumerate() {
                    l[[m, k]] = wk;
                }
                l[[m, m]] = v.sqrt();
            } else {
                if col_sq[lam] <= 0.0 {
                    break;
                }
                l[[0, 0]] = col_sq[lam].sqrt();
            }
            active.push(lam);
            is_active[lam] = true;

            let rhs: Vec<f64> = active.iter().map(|&a| xty[a]).collect();
            gamma = cholesky_solve(&l, active.len(), &rhs);
            residual.assign(&yc);
            for (&a, &g) in active.iter().zip(&gamma) {
                residual.scaled_add(-g, &xc.column(a));
            }

            if let Some(tol) = self.tol {
                if residual.dot(&residual) <= tol {
                    break;
                }
            }
        }

        let mut coef = Array1::<f64>::zeros(d);
        for (&a, &g) in active.iter().zip(&gamma) {
            coef[a] = g;
        }
        let intercept = if self.fit_intercept {
            y_mean - x_mean.dot(&coef)
        } else {
            0.0
        };

        Ok(FittedOrthogonalMatchingPursuit {
            coef,
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
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
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
            .fit(&x, &y)
            .unwrap();
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

    #[test]
    fn test_omp_matches_sklearn() {
        // sklearn 1.8 OrthogonalMatchingPursuit; columns 1 and 4 rescaled
        // (x10, x0.1) so the unnormalised selection is exercised.
        let x = array![
            [1.764052, 4.001572, 0.978738, 2.240893, 0.186756, -0.977278],
            [0.950088, -1.513572, -0.103219, 0.410599, 0.014404, 1.454274],
            [0.761038, 1.21675, 0.443863, 0.333674, 0.149408, -0.205158],
            [0.313068, -8.540957, -2.55299, 0.653619, 0.086444, -0.742165],
            [2.269755, -14.543657, 0.045759, -0.187184, 0.153278, 1.469359],
            [0.154947, 3.781625, -0.887786, -1.980796, -0.034791, 0.156349],
            [1.230291, 12.023798, -0.387327, -0.302303, -0.104855, -1.420018],
            [-1.70627, 19.507754, -0.509652, -0.438074, -0.12528, 0.77749],
            [-1.613898, -2.127403, -0.895467, 0.386902, -0.051081, -1.180632],
            [-0.028182, 4.283319, 0.066517, 0.302472, -0.063432, -0.362741],
            [-0.67246, -3.595532, -0.813146, -1.726283, 0.017743, -0.401781],
            [-1.630198, 4.627823, -0.907298, 0.051945, 0.072909, 0.128983]
        ];
        let y = array![
            7.655502, -0.735587, 1.658285, -0.424832, -5.159026, -2.963866, 5.006542, 3.351169,
            0.392752, 2.328672, -4.419108, 0.394531
        ];
        let check = |f: &FittedOrthogonalMatchingPursuit, coef: [f64; 6], b: f64| {
            for j in 0..6 {
                assert!((f.coef[j] - coef[j]).abs() < 1e-10, "coef = {}", f.coef);
            }
            assert!((f.intercept - b).abs() < 1e-10);
        };

        let f = OrthogonalMatchingPursuit::new()
            .with_n_nonzero_coefs(3)
            .fit(&x, &y)
            .unwrap();
        check(
            &f,
            [
                0.0,
                0.2757988287605062,
                0.0,
                2.2022176684106043,
                0.0,
                -0.9360652578121931,
            ],
            0.0959911797203914,
        );

        let f = OrthogonalMatchingPursuit::new()
            .with_tol(0.5)
            .fit(&x, &y)
            .unwrap();
        check(
            &f,
            [
                0.5268342675681392,
                0.3037401246717109,
                0.0,
                2.0013465797043937,
                0.0,
                -1.0283617340637239,
            ],
            -0.04150126503461282,
        );

        let f = OrthogonalMatchingPursuit::new()
            .with_n_nonzero_coefs(2)
            .with_fit_intercept(false)
            .fit(&x, &y)
            .unwrap();
        check(
            &f,
            [0.0, 0.2997503622433381, 0.0, 2.390792528620986, 0.0, 0.0],
            0.0,
        );

        // Default: max(floor(0.1 * 6), 1) = 1 feature.
        let f = OrthogonalMatchingPursuit::new().fit(&x, &y).unwrap();
        assert_eq!(f.active_set.len(), 1);
    }

    #[test]
    fn test_omp_stops_on_linear_dependence() {
        let x = array![
            [1.0, 2.0, 0.5],
            [2.0, 4.0, -1.0],
            [3.0, 6.0, 0.2],
            [4.0, 8.0, 1.0]
        ];
        let y = array![1.0, 2.0, 3.5, 4.0];
        let f = OrthogonalMatchingPursuit::new()
            .with_n_nonzero_coefs(3)
            .fit(&x, &y)
            .unwrap();
        assert!(f.coef.iter().all(|c| c.is_finite()));
        assert!(f.active_set.len() <= 2);
    }
}

impl anofox_ml_core::RegressorScore<f64> for FittedOrthogonalMatchingPursuit {}
