//! Truncated SVD (a.k.a. LSA when applied to a term-document matrix).
//!
//! Mirrors `sklearn.decomposition.TruncatedSVD`. Unlike PCA, the data is not
//! centered (so it can be applied to sparse inputs without densifying).
//!
//! Decomposes `X ≈ U Σ Vᵀ` and keeps the top `n_components` singular triplets.
//! The transform is `X V_k`, of shape `(n_samples, n_components)`.
//!
//! Uses faer's thin SVD (exact, like sklearn's `algorithm="arpack"` up to
//! numerical precision); component signs follow sklearn's `svd_flip`.

use anofox_ml_core::{FitUnsupervised, InverseTransform, Result, RustMlError, Transform};
use faer::linalg::solvers::Svd;
use faer::Mat;
use ndarray::{Array1, Array2, Axis};

use crate::pca::svd_flip_rows;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TruncatedSvd {
    pub n_components: usize,
}

impl TruncatedSvd {
    pub fn new(n_components: usize) -> Self {
        Self { n_components }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedTruncatedSvd {
    /// Top-`k` right-singular vectors, shape (n_features, k) — the transpose
    /// of sklearn's `components_`. Signs follow sklearn's `svd_flip`
    /// (largest-magnitude entry of each component is positive).
    pub components: Array2<f64>,
    /// Top-`k` singular values.
    pub singular_values: Array1<f64>,
    /// Variance of each projected component, `var(X V_k, ddof = 0)`
    /// (sklearn's `explained_variance_`).
    pub explained_variance: Array1<f64>,
    /// `explained_variance / var(X).sum()` (sklearn's
    /// `explained_variance_ratio_`).
    #[serde(default)]
    pub explained_variance_ratio: Array1<f64>,
    n_features: usize,
}

impl FittedTruncatedSvd {
    pub fn n_components(&self) -> usize {
        self.components.ncols()
    }
}

/// Column-wise population variance (`np.var(a, axis=0)`).
fn population_variance(a: &Array2<f64>) -> Array1<f64> {
    let n = a.nrows().max(1) as f64;
    let mean = a.sum_axis(Axis(0)) / n;
    let mut var = Array1::<f64>::zeros(a.ncols());
    for row in a.rows() {
        for (j, &v) in row.iter().enumerate() {
            let dv = v - mean[j];
            var[j] += dv * dv;
        }
    }
    var / n
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
        // Thin SVD: U is n × min(n, d) (never the full n × n factor).
        let svd = Svd::new_thin(m.as_ref())
            .map_err(|e| RustMlError::InvalidParameter(format!("SVD failed: {e:?}")))?;
        let v = svd.V(); // d × min(n, d)
        let s = svd.S().column_vector(); // length min(n, d)
        let mut order: Vec<usize> = (0..s.nrows()).collect();
        order.sort_by(|&a, &b| s[b].total_cmp(&s[a]));

        // Top-k right-singular vectors as rows (sklearn's `components_`
        // layout) so we can apply sklearn's sign convention.
        let mut vt = Array2::<f64>::from_shape_fn((k, d), |(c, j)| v[(j, order[c])]);
        let sv = Array1::<f64>::from_shape_fn(k, |c| s[order[c]]);
        // sklearn >= 1.5: `svd_flip(U, VT, u_based_decision=False)`.
        svd_flip_rows(&mut vt);
        let components = vt.t().to_owned();

        // sklearn: explained_variance_ = var(X_transformed, axis=0) (ddof=0),
        // explained_variance_ratio_ = explained_variance_ / var(X).sum().
        let ev = population_variance(&x.dot(&components));
        let total_var: f64 = population_variance(x).sum();
        let evr = if total_var > 0.0 {
            &ev / total_var
        } else {
            Array1::zeros(k)
        };

        Ok(FittedTruncatedSvd {
            components,
            singular_values: sv,
            explained_variance: ev,
            explained_variance_ratio: evr,
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

impl InverseTransform<f64> for FittedTruncatedSvd {
    /// Reconstruct the original-space representation from the projection
    /// `transform(X) = X V_k`. The inverse mapping is `T @ V_kᵀ`, valid up to
    /// the rank-`k` approximation error.
    fn inverse_transform(&self, t: &Array2<f64>) -> Result<Array2<f64>> {
        if t.ncols() != self.components.ncols() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} components, got {}",
                self.components.ncols(),
                t.ncols()
            )));
        }
        Ok(t.dot(&self.components.t()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_truncated_svd_reduces_dim() {
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [2.0, 3.0, 5.0]
        ];
        let svd = TruncatedSvd::new(2).fit(&x).unwrap();
        let t = svd.transform(&x).unwrap();
        assert_eq!(t.shape(), &[4, 2]);
        // First singular value should be much larger than second.
        assert!(svd.singular_values[0] > svd.singular_values[1]);
    }

    #[test]
    fn test_inverse_transform_reconstructs_full_rank() {
        let x = array![
            [1.0_f64, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [2.0, 3.0, 5.0]
        ];
        // Keep all components → inverse should reconstruct exactly.
        let svd = TruncatedSvd::new(3).fit(&x).unwrap();
        let t = svd.transform(&x).unwrap();
        let back = svd.inverse_transform(&t).unwrap();
        for ((i, j), &v) in x.indexed_iter() {
            assert!(
                (back[[i, j]] - v).abs() < 1e-9,
                "[{},{}]: {} vs {}",
                i,
                j,
                back[[i, j]],
                v
            );
        }
    }

    #[test]
    fn test_matches_sklearn_with_svd_flip() {
        // Reference: sklearn 1.8.0 TruncatedSVD(n_components=3, algorithm="arpack").
        let x = Array2::from_shape_fn((30, 5), |(i, j)| {
            let (fi, fj) = (i as f64, j as f64);
            (0.37 * (fi + 1.0) * (fj + 1.0)).sin()
                + 0.05 * (fj + 1.0) * (0.11 * fi).cos()
                + 0.3 * fj
        });
        let svd = TruncatedSvd::new(3).fit(&x).unwrap();
        let sk_components = [
            [
                0.05636160445980286,
                0.22755236252300534,
                0.38045806017760614,
                0.5354345635353492,
                0.716662252717777,
            ],
            [
                0.7897654759849791,
                -0.48282803667569774,
                -0.17809050830852308,
                0.3284411885975461,
                -0.05964676374417794,
            ],
            [
                0.10452402478774592,
                -0.4562200638603194,
                0.7039779658928075,
                -0.51365308402447,
                0.14667485941030473,
            ],
        ];
        for (c, sk) in sk_components.iter().enumerate() {
            for (j, &e) in sk.iter().enumerate() {
                assert!(
                    (svd.components[[j, c]] - e).abs() < 1e-8,
                    "component {c}[{j}]: {} vs {e}",
                    svd.components[[j, c]]
                );
            }
        }
        let sk_sv = [10.016892766348839, 4.133324915813744, 4.120841269584226];
        let sk_ev = [0.5450728005260516, 0.5689394863083795, 0.566043614605013];
        let sk_evr = [0.20575277800231487, 0.2147619175093368, 0.21366879077996975];
        for c in 0..3 {
            assert!((svd.singular_values[c] - sk_sv[c]).abs() < 1e-9);
            assert!((svd.explained_variance[c] - sk_ev[c]).abs() < 1e-9);
            assert!((svd.explained_variance_ratio[c] - sk_evr[c]).abs() < 1e-9);
        }
    }
}
