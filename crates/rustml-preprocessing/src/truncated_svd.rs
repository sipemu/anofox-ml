//! Truncated SVD (a.k.a. LSA when applied to a term-document matrix).
//!
//! Mirrors `sklearn.decomposition.TruncatedSVD`. Unlike PCA, the data is not
//! centered (so it can be applied to sparse inputs without densifying).
//!
//! Decomposes `X ≈ U Σ Vᵀ` and keeps the top `n_components` singular triplets.
//! The transform is `X V_k`, of shape `(n_samples, n_components)`.

use faer::linalg::solvers::Svd;
use faer::Mat;
use ndarray::{Array1, Array2};
use rustml_core::{FitUnsupervised, Result, RustMlError, Transform};

#[derive(Debug, Clone)]
pub struct TruncatedSvd {
    pub n_components: usize,
}

impl TruncatedSvd {
    pub fn new(n_components: usize) -> Self {
        Self { n_components }
    }
}

#[derive(Debug, Clone)]
pub struct FittedTruncatedSvd {
    /// Top-`k` right-singular vectors, shape (n_features, k).
    pub components: Array2<f64>,
    /// Top-`k` singular values.
    pub singular_values: Array1<f64>,
    /// Explained variance per component.
    pub explained_variance: Array1<f64>,
    n_features: usize,
}

impl FittedTruncatedSvd {
    pub fn n_components(&self) -> usize {
        self.components.ncols()
    }
}

impl FitUnsupervised<f64> for TruncatedSvd {
    type Fitted = FittedTruncatedSvd;

    fn fit(&self, x: &Array2<f64>) -> Result<Self::Fitted> {
        let (n, d) = x.dim();
        if n == 0 || d == 0 {
            return Err(RustMlError::EmptyInput("empty input".into()));
        }
        let k = self.n_components.min(d.min(n));
        if k == 0 {
            return Err(RustMlError::InvalidParameter(
                "n_components must be at least 1".into(),
            ));
        }

        let m = Mat::<f64>::from_fn(n, d, |i, j| x[[i, j]]);
        let svd = Svd::new(m.as_ref())
            .map_err(|e| RustMlError::InvalidParameter(format!("SVD failed: {e:?}")))?;
        let v = svd.V(); // d × d
        let s = svd.S(); // diag, length min(n, d)
        let sv_len = s.column_vector().nrows();

        // sklearn returns components_ shape (n_components, n_features) —
        // rows are right-singular vectors (V columns). For us: take first k
        // columns of V.
        let mut components = Array2::<f64>::zeros((d, k));
        let mut sv = Array1::<f64>::zeros(k);
        for j in 0..k {
            for i in 0..d {
                components[[i, j]] = v[(i, j)];
            }
            sv[j] = if j < sv_len { s.column_vector()[j] } else { 0.0 };
        }
        // Explained variance ≈ Var(X V_j) = (s_j^2) / (n - 1)
        let mut ev = Array1::<f64>::zeros(k);
        let denom = (n as f64 - 1.0).max(1.0);
        for j in 0..k {
            ev[j] = sv[j] * sv[j] / denom;
        }

        Ok(FittedTruncatedSvd {
            components,
            singular_values: sv,
            explained_variance: ev,
            n_features: d,
        })
    }
}

impl Transform<f64> for FittedTruncatedSvd {
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        Ok(x.dot(&self.components))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_truncated_svd_reduces_dim() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [2.0, 3.0, 5.0]];
        let svd = TruncatedSvd::new(2).fit(&x).unwrap();
        let t = svd.transform(&x).unwrap();
        assert_eq!(t.shape(), &[4, 2]);
        // First singular value should be much larger than second.
        assert!(svd.singular_values[0] > svd.singular_values[1]);
    }
}
