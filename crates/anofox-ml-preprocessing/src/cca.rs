//! Canonical Correlation Analysis.
//!
//! Mirrors `sklearn.cross_decomposition.CCA` (and `PLSCanonical` for the
//! univariate case). Given paired (X, Y), CCA finds linear combinations
//! `Xa` and `Yb` that are maximally correlated.
//!
//! Closed form via SVD:
//! 1. Centre and whiten both X and Y (X_white = X · K_x, Y_white = Y · K_y).
//! 2. Compute the cross-covariance `C = X_whiteᵀ · Y_white / (n - 1)`.
//! 3. SVD `C = U Σ Vᵀ`. The first `k` columns of `K_x · U` and `K_y · V`
//!    are the loadings `x_weights_` and `y_weights_` (sklearn's naming).
//! 4. `Σ_ii` are the canonical correlations.

use anofox_ml_core::{Result, RustMlError};
use faer::linalg::solvers::Svd;
use faer::Mat;
use ndarray::{Array1, Array2};

pub struct Cca {
    pub n_components: usize,
}

impl Cca {
    pub fn new(n_components: usize) -> Self {
        Self { n_components }
    }

    pub fn fit(&self, x: &Array2<f64>, y: &Array2<f64>) -> Result<FittedCca> {
        let n = x.nrows();
        let dx = x.ncols();
        let dy = y.ncols();
        if y.nrows() != n {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but Y has {}",
                n,
                y.nrows()
            )));
        }
        let k = self.n_components.min(dx).min(dy);
        if k == 0 {
            return Err(RustMlError::InvalidParameter("n_components >= 1".into()));
        }
        let n_f = n as f64;
        let mut x_mean = Array1::<f64>::zeros(dx);
        for j in 0..dx {
            x_mean[j] = x.column(j).sum() / n_f;
        }
        let mut y_mean = Array1::<f64>::zeros(dy);
        for j in 0..dy {
            y_mean[j] = y.column(j).sum() / n_f;
        }

        let mut xc = x.clone();
        let mut yc = y.clone();
        for j in 0..dx {
            for i in 0..n {
                xc[[i, j]] -= x_mean[j];
            }
        }
        for j in 0..dy {
            for i in 0..n {
                yc[[i, j]] -= y_mean[j];
            }
        }

        // Whitening for X via SVD: X_centred = U_x Σ_x V_xᵀ.
        // K_x = V_x Σ_x⁻¹ √(n-1).
        let scale = (n as f64 - 1.0).sqrt();
        let kx = whitening(&xc, scale)?;
        let ky = whitening(&yc, scale)?;
        let x_white = xc.dot(&kx);
        let y_white = yc.dot(&ky);
        // C = X_white' Y_white / (n - 1).
        let c = x_white.t().dot(&y_white) / (n_f - 1.0).max(1.0);
        let nx = c.nrows();
        let ny = c.ncols();
        let cm = Mat::<f64>::from_fn(nx, ny, |i, j| c[[i, j]]);
        let svd = Svd::new(cm.as_ref())
            .map_err(|e| RustMlError::InvalidParameter(format!("SVD failed: {e:?}")))?;
        let u = svd.U();
        let s = svd.S();
        let v = svd.V();
        // x_weights = K_x · U[:, :k]; y_weights = K_y · V[:, :k].
        let k_real = k.min(nx).min(ny);
        let mut u_top = Array2::<f64>::zeros((nx, k_real));
        let mut v_top = Array2::<f64>::zeros((ny, k_real));
        let mut corrs = Array1::<f64>::zeros(k_real);
        for c_i in 0..k_real {
            corrs[c_i] = s.column_vector()[c_i];
            for i in 0..nx {
                u_top[[i, c_i]] = u[(i, c_i)];
            }
            for i in 0..ny {
                v_top[[i, c_i]] = v[(i, c_i)];
            }
        }
        let x_weights = kx.dot(&u_top);
        let y_weights = ky.dot(&v_top);
        Ok(FittedCca {
            x_mean,
            y_mean,
            x_weights,
            y_weights,
            canonical_correlations: corrs,
        })
    }
}

/// Returns whitening matrix `W` such that `X_centred · W` has identity covariance.
fn whitening(xc: &Array2<f64>, scale: f64) -> Result<Array2<f64>> {
    let n = xc.nrows();
    let d = xc.ncols();
    let m = Mat::<f64>::from_fn(n, d, |i, j| xc[[i, j]]);
    let svd = Svd::new(m.as_ref())
        .map_err(|e| RustMlError::InvalidParameter(format!("SVD failed: {e:?}")))?;
    let s = svd.S();
    let v = svd.V();
    let r = s.column_vector().nrows();
    let mut w = Array2::<f64>::zeros((d, r));
    for c in 0..r {
        let sigma = s.column_vector()[c].max(1e-12);
        for j in 0..d {
            w[[j, c]] = v[(j, c)] * scale / sigma;
        }
    }
    Ok(w)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedCca {
    pub x_mean: Array1<f64>,
    pub y_mean: Array1<f64>,
    /// X loadings, shape (n_features_x, n_components).
    pub x_weights: Array2<f64>,
    /// Y loadings, shape (n_features_y, n_components).
    pub y_weights: Array2<f64>,
    /// Canonical correlations (diagonal of the cross-covariance SVD).
    pub canonical_correlations: Array1<f64>,
}

impl FittedCca {
    /// Project X into canonical x-space: (X − x̄) · x_weights.
    pub fn transform_x(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.x_mean.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} X-features, got {}",
                self.x_mean.len(),
                x.ncols()
            )));
        }
        let mut xc = x.clone();
        for j in 0..x.ncols() {
            for i in 0..x.nrows() {
                xc[[i, j]] -= self.x_mean[j];
            }
        }
        Ok(xc.dot(&self.x_weights))
    }

    /// Project Y into canonical y-space: (Y − ȳ) · y_weights.
    pub fn transform_y(&self, y: &Array2<f64>) -> Result<Array2<f64>> {
        if y.ncols() != self.y_mean.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} Y-features, got {}",
                self.y_mean.len(),
                y.ncols()
            )));
        }
        let mut yc = y.clone();
        for j in 0..y.ncols() {
            for i in 0..y.nrows() {
                yc[[i, j]] -= self.y_mean[j];
            }
        }
        Ok(yc.dot(&self.y_weights))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_cca_finds_high_correlation() {
        // Construct X, Y where Y is essentially a noisy linear function of
        // X[:, 0]. CCA should find the first canonical correlation near 1.
        let n = 100;
        let mut x = Array2::<f64>::zeros((n, 3));
        let mut y = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let t = (i as f64) * 0.1;
            x[[i, 0]] = t.sin();
            x[[i, 1]] = (i as f64) - 50.0;
            x[[i, 2]] = ((i * 7) % 13) as f64;
            y[[i, 0]] = t.sin() + 0.01;
            y[[i, 1]] = -2.0 * t.sin();
        }
        let fitted = Cca::new(1).fit(&x, &y).unwrap();
        assert!(
            fitted.canonical_correlations[0] > 0.9,
            "first canonical correlation = {}",
            fitted.canonical_correlations[0]
        );
        let _ = array![1.0_f64];
    }

    #[test]
    fn test_cca_transform_shapes() {
        let x = array![[1.0_f64, 0.0], [0.0, 1.0], [2.0, 1.0], [1.0, 2.0]];
        let y = array![[1.0_f64, 0.0], [0.5, 0.5], [1.5, 0.5], [1.0, 1.0]];
        let fitted = Cca::new(1).fit(&x, &y).unwrap();
        let xt = fitted.transform_x(&x).unwrap();
        let yt = fitted.transform_y(&y).unwrap();
        assert_eq!(xt.shape(), &[4, 1]);
        assert_eq!(yt.shape(), &[4, 1]);
    }
}
