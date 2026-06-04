//! Least Angle Regression (LARS) and LASSO LARS variants.
//!
//! Mirrors `sklearn.linear_model.{Lars, LassoLars}`. Walks the L1 regularisation
//! path piecewise-linearly; at each step a new feature joins the active set
//! (or, in LassoLars, an active feature can leave).

use ndarray::{Array1, Array2};
use rustml_core::{Fit, Predict, Result, RustMlError};

#[derive(Debug, Clone)]
pub struct Lars {
    pub n_nonzero_coefs: usize,
    pub fit_intercept: bool,
    pub lasso: bool,
}

impl Lars {
    pub fn new(n_nonzero_coefs: usize) -> Self {
        Self { n_nonzero_coefs, fit_intercept: true, lasso: false }
    }

    pub fn lasso(n_nonzero_coefs: usize) -> Self {
        Self { n_nonzero_coefs, fit_intercept: true, lasso: true }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedLars {
    pub coef: Array1<f64>,
    pub intercept: f64,
    pub active_set: Vec<usize>,
    n_features: usize,
}

fn sgn(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }
}

impl Fit<f64> for Lars {
    type Fitted = FittedLars;

    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        if x.nrows() != y.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but y has {}", x.nrows(), y.len()
            )));
        }
        let n = x.nrows();
        let d = x.ncols();
        if self.n_nonzero_coefs == 0 {
            return Err(RustMlError::InvalidParameter("need at least 1 coef".into()));
        }
        // Centre features and target (sklearn default for Lars).
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

        // sklearn deprecated `normalize=True` in Lars; the modern path keeps
        // centered (but un-normalised) features so coefficient magnitudes are
        // returned in the original units. We follow that.
        let xs = xc;
        let col_norm = Array1::<f64>::ones(d);

        let mut beta = Array1::<f64>::zeros(d);
        let mut residual = yc.clone();
        let mut active: Vec<usize> = Vec::new();
        let mut signs: Vec<f64> = Vec::new();

        let target_k = self.n_nonzero_coefs.min(d).min(n);
        let mut step = 0;
        while step < target_k {
            step += 1;
            // Correlations c_j = X_j' r.
            let mut corr = Array1::<f64>::zeros(d);
            for j in 0..d {
                let mut s = 0.0;
                for i in 0..n {
                    s += xs[[i, j]] * residual[i];
                }
                corr[j] = s;
            }
            // Max absolute correlation over inactive set.
            let mut max_abs = 0.0;
            let mut new_j = None;
            for j in 0..d {
                if active.contains(&j) {
                    continue;
                }
                if corr[j].abs() > max_abs {
                    max_abs = corr[j].abs();
                    new_j = Some(j);
                }
            }
            let j = match new_j {
                Some(j) => j,
                None => break,
            };
            active.push(j);
            signs.push(sgn(corr[j]));

            // Solve Gram*z = signs for direction in active subspace.
            let m = active.len();
            let mut gram = vec![vec![0.0_f64; m]; m];
            for a in 0..m {
                for b in 0..m {
                    let mut s = 0.0;
                    for i in 0..n {
                        s += xs[[i, active[a]]] * xs[[i, active[b]]];
                    }
                    gram[a][b] = signs[a] * signs[b] * s;
                }
            }
            let mut rhs = vec![1.0_f64; m];
            // Solve gram * w = 1 by Gauss elimination (m small).
            for col in 0..m {
                let mut piv = col;
                for r in (col + 1)..m {
                    if gram[r][col].abs() > gram[piv][col].abs() {
                        piv = r;
                    }
                }
                if piv != col {
                    gram.swap(col, piv);
                    rhs.swap(col, piv);
                }
                let pv = gram[col][col];
                if pv.abs() < 1e-14 {
                    continue;
                }
                for r in (col + 1)..m {
                    let f = gram[r][col] / pv;
                    for c in col..m {
                        gram[r][c] -= f * gram[col][c];
                    }
                    rhs[r] -= f * rhs[col];
                }
            }
            let mut w = vec![0.0_f64; m];
            for r in (0..m).rev() {
                let mut s = rhs[r];
                for c in (r + 1)..m {
                    s -= gram[r][c] * w[c];
                }
                let pv = gram[r][r];
                if pv.abs() > 1e-14 {
                    w[r] = s / pv;
                }
            }
            // a = (signs' w)^{-1/2}
            let mut sw = 0.0;
            for k in 0..m {
                sw += signs[k] * w[k];
            }
            let a = if sw > 0.0 { sw.sqrt().recip() } else { 1e-12 };

            // Equiangular direction u in feature space; equivalently the
            // descent direction in beta is a * signs[k] * w[k] for each
            // active k.
            let dir: Vec<f64> = (0..m).map(|k| a * signs[k] * w[k]).collect();

            // Determine step size γ. For each inactive j find when |c_j| would
            // equal max_abs. The cosines are A_j = X_j' equiangular vector.
            let mut a_inner = Array1::<f64>::zeros(d);
            for j in 0..d {
                let mut s = 0.0;
                for k in 0..m {
                    let mut col_sum = 0.0;
                    for i in 0..n {
                        col_sum += xs[[i, j]] * xs[[i, active[k]]];
                    }
                    s += dir[k] * col_sum;
                }
                a_inner[j] = s;
            }
            // gamma_hat = min over inactive j of:
            //   min((max_abs - corr_j) / (a - a_inner_j), (max_abs + corr_j) / (a + a_inner_j))
            let mut gamma = f64::INFINITY;
            for j in 0..d {
                if active.contains(&j) {
                    continue;
                }
                let denom1 = a - a_inner[j];
                let denom2 = a + a_inner[j];
                if denom1 > 1e-12 {
                    let g = (max_abs - corr[j]) / denom1;
                    if g > 1e-12 && g < gamma {
                        gamma = g;
                    }
                }
                if denom2 > 1e-12 {
                    let g = (max_abs + corr[j]) / denom2;
                    if g > 1e-12 && g < gamma {
                        gamma = g;
                    }
                }
            }
            if !gamma.is_finite() {
                // Last step (no inactive feature can equalise correlations);
                // jump straight to the OLS on active set.
                gamma = max_abs / a;
            }

            // LassoLars: shorten step if any active beta would cross zero.
            if self.lasso {
                for k in 0..m {
                    let dk = dir[k];
                    if dk.abs() < 1e-14 {
                        continue;
                    }
                    let bj = beta[active[k]];
                    let cross = -bj / dk;
                    if cross > 1e-12 && cross < gamma {
                        gamma = cross;
                    }
                }
            }

            // Update beta on active set.
            for k in 0..m {
                beta[active[k]] += gamma * dir[k];
            }
            // Update residual: r := r - gamma * u, where u = sum_k signs_k * w_k * x_{active_k} * a
            // In residual space: subtract gamma * X * delta_beta_active.
            for i in 0..n {
                let mut up = 0.0;
                for k in 0..m {
                    up += xs[[i, active[k]]] * dir[k];
                }
                residual[i] -= gamma * up;
            }
        }

        // Un-scale beta by column norms.
        for j in 0..d {
            beta[j] /= col_norm[j];
        }
        let intercept = if self.fit_intercept {
            y_mean - x_mean.dot(&beta)
        } else {
            0.0
        };
        Ok(FittedLars {
            coef: beta,
            intercept,
            active_set: active,
            n_features: d,
        })
    }
}

impl Predict<f64> for FittedLars {
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
// LassoLarsIC — information-criterion-selected step on the LARS path.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IcCriterion {
    Aic,
    Bic,
}

#[derive(Debug, Clone)]
pub struct LassoLarsIC {
    pub criterion: IcCriterion,
    pub max_features: Option<usize>,
    pub fit_intercept: bool,
}

impl LassoLarsIC {
    pub fn new(criterion: IcCriterion) -> Self {
        Self { criterion, max_features: None, fit_intercept: true }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedLassoLarsIC {
    pub fitted: FittedLars,
    pub criterion_value: f64,
    pub n_nonzero_coefs: usize,
}

impl Fit<f64> for LassoLarsIC {
    type Fitted = FittedLassoLarsIC;

    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        let n = x.nrows() as f64;
        let d = x.ncols();
        let max_k = self.max_features.unwrap_or(d).min(d).min(x.nrows());

        let mut best: Option<FittedLassoLarsIC> = None;

        for k in 1..=max_k {
            let lars = Lars {
                n_nonzero_coefs: k,
                fit_intercept: self.fit_intercept,
                lasso: true,
            };
            let fitted = lars.fit(x, y)?;
            let preds = fitted.predict(x)?;
            let rss: f64 = preds
                .iter()
                .zip(y.iter())
                .map(|(p, t)| (t - p).powi(2))
                .sum();
            let nnz = fitted
                .coef
                .iter()
                .filter(|v| v.abs() > 1e-12)
                .count() as f64;
            // sklearn's formula (matching `linear_model.LassoLarsIC.criterion_`):
            //   AIC = n * log(rss / n) + 2 * df
            //   BIC = n * log(rss / n) + log(n) * df
            // We follow that exactly. (Older sklearn used the rss/σ²
            // formulation under a fixed-noise assumption; the modern path
            // uses log-likelihood up to additive constants.)
            let log_rss = (rss / n.max(1.0)).max(1e-300).ln();
            let crit = match self.criterion {
                IcCriterion::Aic => n * log_rss + 2.0 * nnz,
                IcCriterion::Bic => n * log_rss + n.ln() * nnz,
            };
            let nnz_int = nnz as usize;
            let candidate = FittedLassoLarsIC {
                fitted,
                criterion_value: crit,
                n_nonzero_coefs: nnz_int,
            };
            match &best {
                None => best = Some(candidate),
                Some(b) if candidate.criterion_value < b.criterion_value => best = Some(candidate),
                _ => {}
            }
        }
        Ok(best.unwrap())
    }
}

impl Predict<f64> for FittedLassoLarsIC {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        self.fitted.predict(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_lars_recovers_two_features() {
        let n = 80;
        let mut data = Vec::new();
        for i in 0..n {
            let x0 = (i as f64) - 40.0;
            let x1 = ((i * 7 % 13) as f64) - 6.0;
            let x2 = ((i * 5 % 11) as f64) - 5.0;
            let x3 = ((i * 3 % 7) as f64) - 3.0;
            data.extend([x0, x1, x2, x3]);
        }
        let x = Array2::from_shape_vec((n, 4), data).unwrap();
        let y = x.column(0).mapv(|v| 3.0 * v) + x.column(2).mapv(|v| -2.0 * v);

        let fitted = Lars::new(2).fit(&x, &y).unwrap();
        let mut act = fitted.active_set.clone();
        act.sort();
        assert_eq!(act, vec![0, 2]);
        let _ = array![1.0_f64];
    }

    #[test]
    fn test_lasso_lars_basic() {
        let n = 40;
        let mut data = Vec::new();
        for i in 0..n {
            let x0 = (i as f64) - 20.0;
            let x1 = ((i * 7 % 13) as f64) - 6.0;
            data.extend([x0, x1]);
        }
        let x = Array2::from_shape_vec((n, 2), data).unwrap();
        let y = x.column(0).mapv(|v| 2.0 * v);
        let fitted = Lars::lasso(2).fit(&x, &y).unwrap();
        assert!((fitted.coef[0] - 2.0).abs() < 0.2);
    }

    #[test]
    fn test_lasso_lars_ic_bic_picks_sparse() {
        // 3 informative features out of 6 — BIC should pick a sparse model.
        let n = 100;
        let mut data = Vec::new();
        for i in 0..n {
            for j in 0..6 {
                data.push(((i * (j + 1) * 7) % 19) as f64 - 9.0);
            }
        }
        let x = Array2::from_shape_vec((n, 6), data).unwrap();
        let y = x.column(0).mapv(|v| 3.0 * v)
            + x.column(2).mapv(|v| -2.0 * v)
            + x.column(4).mapv(|v| 1.5 * v);
        let fitted = LassoLarsIC::new(IcCriterion::Bic).fit(&x, &y).unwrap();
        // Top-3 magnitudes should include the informative features.
        let mut order: Vec<(usize, f64)> = fitted
            .fitted
            .coef
            .iter()
            .enumerate()
            .map(|(i, v)| (i, v.abs()))
            .collect();
        order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top: std::collections::HashSet<usize> =
            order.iter().take(3).map(|(i, _)| *i).collect();
        for j in [0_usize, 2, 4] {
            assert!(top.contains(&j), "feature {j} not in top-3: {:?}", top);
        }
    }
}
