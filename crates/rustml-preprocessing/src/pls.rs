//! Partial Least Squares Regression (PLS1).
//!
//! Mirrors `sklearn.cross_decomposition.PLSRegression`. Implements the
//! NIPALS algorithm for `n_components` latent variables on 1-D `y` (PLS1).
//! 2-D `y` (PLS2) is not currently supported.

use ndarray::{Array1, Array2};
use rustml_core::{Fit, Predict, Result, RustMlError};

#[derive(Debug, Clone)]
pub struct PlsRegression {
    pub n_components: usize,
    pub max_iter: usize,
    pub tol: f64,
}

impl PlsRegression {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            max_iter: 500,
            tol: 1e-6,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedPlsRegression {
    pub x_mean: Array1<f64>,
    pub y_mean: f64,
    pub x_std: Array1<f64>,
    pub y_std: f64,
    /// Regression coefficients in centred+scaled space.
    pub coef: Array1<f64>,
    n_features: usize,
}

impl Fit<f64> for PlsRegression {
    type Fitted = FittedPlsRegression;

    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        if x.nrows() != y.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but y has {}",
                x.nrows(),
                y.len()
            )));
        }
        let n = x.nrows();
        let d = x.ncols();
        if self.n_components == 0 || self.n_components > d.min(n) {
            return Err(RustMlError::InvalidParameter(format!(
                "n_components must be in 1..={}",
                d.min(n)
            )));
        }

        // Standardize columns of X and y to unit variance (sklearn default).
        let n_f = n as f64;
        let mut x_mean = Array1::<f64>::zeros(d);
        for j in 0..d {
            x_mean[j] = x.column(j).sum() / n_f;
        }
        let y_mean = y.sum() / n_f;
        let mut x_std = Array1::<f64>::ones(d);
        for j in 0..d {
            let mut v = 0.0;
            for i in 0..n {
                let dv = x[[i, j]] - x_mean[j];
                v += dv * dv;
            }
            x_std[j] = (v / n_f).sqrt().max(1e-12);
        }
        let mut yv = 0.0;
        for i in 0..n {
            let dv = y[i] - y_mean;
            yv += dv * dv;
        }
        let y_std = (yv / n_f).sqrt().max(1e-12);

        let mut xs = Array2::<f64>::zeros((n, d));
        let mut ys = Array1::<f64>::zeros(n);
        for i in 0..n {
            for j in 0..d {
                xs[[i, j]] = (x[[i, j]] - x_mean[j]) / x_std[j];
            }
            ys[i] = (y[i] - y_mean) / y_std;
        }

        // NIPALS for PLS1 — Y is a single column.
        let k = self.n_components;
        let mut p_mat = Array2::<f64>::zeros((d, k));
        let mut w_mat = Array2::<f64>::zeros((d, k));
        let mut q_vec = Array1::<f64>::zeros(k);
        let mut x_def = xs.clone();
        let mut y_def = ys.clone();

        for comp in 0..k {
            // Weights w = X'y / ||X'y||
            let mut w = Array1::<f64>::zeros(d);
            for j in 0..d {
                let mut s = 0.0;
                for i in 0..n {
                    s += x_def[[i, j]] * y_def[i];
                }
                w[j] = s;
            }
            let nw = w.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
            for j in 0..d {
                w[j] /= nw;
            }
            // Scores t = X w
            let mut t = Array1::<f64>::zeros(n);
            for i in 0..n {
                let mut s = 0.0;
                for j in 0..d {
                    s += x_def[[i, j]] * w[j];
                }
                t[i] = s;
            }
            // Loadings p = X' t / (t' t)
            let tt: f64 = t.iter().map(|v| v * v).sum::<f64>().max(1e-12);
            let mut p = Array1::<f64>::zeros(d);
            for j in 0..d {
                let mut s = 0.0;
                for i in 0..n {
                    s += x_def[[i, j]] * t[i];
                }
                p[j] = s / tt;
            }
            // Regression coef on Y: q = y' t / (t' t)
            let mut q = 0.0;
            for i in 0..n {
                q += y_def[i] * t[i];
            }
            q /= tt;

            // Deflate.
            for i in 0..n {
                for j in 0..d {
                    x_def[[i, j]] -= t[i] * p[j];
                }
                y_def[i] -= t[i] * q;
            }

            for j in 0..d {
                p_mat[[j, comp]] = p[j];
                w_mat[[j, comp]] = w[j];
            }
            q_vec[comp] = q;
        }

        // Final regression coefficients in centred+scaled space:
        // beta = W (P' W)^{-1} q
        // For PLS1 this simplifies; we compute beta numerically.
        // PtW is k×k. Solve PtW * z = q, then beta = W z.
        let mut pt_w = Array2::<f64>::zeros((k, k));
        for a in 0..k {
            for b in 0..k {
                let mut s = 0.0;
                for j in 0..d {
                    s += p_mat[[j, a]] * w_mat[[j, b]];
                }
                pt_w[[a, b]] = s;
            }
        }
        // Forward solve PtW z = q (PtW is upper triangular for PLS1).
        let mut z = Array1::<f64>::zeros(k);
        // General solve via Gauss elimination (k is small).
        let mut m = pt_w.clone();
        let mut rhs = q_vec.clone();
        for col in 0..k {
            // Find pivot.
            let mut piv = col;
            for r in (col + 1)..k {
                if m[[r, col]].abs() > m[[piv, col]].abs() {
                    piv = r;
                }
            }
            if piv != col {
                for c in 0..k {
                    let tmp = m[[col, c]];
                    m[[col, c]] = m[[piv, c]];
                    m[[piv, c]] = tmp;
                }
                let tmp = rhs[col];
                rhs[col] = rhs[piv];
                rhs[piv] = tmp;
            }
            let pv = m[[col, col]];
            if pv.abs() < 1e-14 {
                continue;
            }
            for r in (col + 1)..k {
                let f = m[[r, col]] / pv;
                for c in col..k {
                    m[[r, c]] -= f * m[[col, c]];
                }
                rhs[r] -= f * rhs[col];
            }
        }
        // Back-substitution.
        for r in (0..k).rev() {
            let mut s = rhs[r];
            for c in (r + 1)..k {
                s -= m[[r, c]] * z[c];
            }
            let pv = m[[r, r]];
            if pv.abs() > 1e-14 {
                z[r] = s / pv;
            }
        }

        let mut coef = Array1::<f64>::zeros(d);
        for j in 0..d {
            let mut s = 0.0;
            for c in 0..k {
                s += w_mat[[j, c]] * z[c];
            }
            coef[j] = s;
        }

        Ok(FittedPlsRegression {
            x_mean,
            y_mean,
            x_std,
            y_std,
            coef,
            n_features: d,
        })
    }
}

impl Predict<f64> for FittedPlsRegression {
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
            let mut s = 0.0;
            for j in 0..self.n_features {
                let xs = (x[[i, j]] - self.x_mean[j]) / self.x_std[j];
                s += self.coef[j] * xs;
            }
            out[i] = s * self.y_std + self.y_mean;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_pls1_recovers_linear() {
        let rng_x: Vec<f64> = (0..40)
            .flat_map(|i| {
                let i = i as f64;
                vec![i, 0.5 * i, -0.3 * i + 1.0]
            })
            .collect();
        let x = Array2::from_shape_vec((40, 3), rng_x).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|v| 2.0 * v) + x.column(1).mapv(|v| 1.5 * v);
        let fitted = PlsRegression::new(2).fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        let rss: f64 = preds
            .iter()
            .zip(y.iter())
            .map(|(p, t)| (t - p).powi(2))
            .sum();
        let mean = y.iter().sum::<f64>() / y.len() as f64;
        let tss: f64 = y.iter().map(|t| (t - mean).powi(2)).sum();
        let r2 = 1.0 - rss / tss;
        assert!(r2 > 0.99, "R² too low: {r2}");
        let _ = array![1.0_f64];
    }
}
