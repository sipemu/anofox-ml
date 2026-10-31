//! Principal Component Analysis.
//!
//! Mirrors `sklearn.decomposition.PCA` (sklearn 1.8, `svd_solver="auto"` for
//! exact solvers). The data is centred and decomposed with faer:
//!
//! * `n_samples >= 10 * n_features` (tall data, `n_features <= 1000`):
//!   self-adjoint eigendecomposition of the `d × d` covariance matrix
//!   (sklearn's `"covariance_eigh"` solver);
//! * otherwise: thin SVD of the centred `n × d` data matrix (sklearn's
//!   `"full"` solver).
//!
//! Component signs follow sklearn's `svd_flip(u_based_decision=False)`
//! convention (sklearn >= 1.5): in every component the entry with the largest
//! absolute value is positive. This makes the output deterministic and
//! directly comparable with sklearn's `components_`.

use anofox_ml_core::{FitUnsupervised, Float, InverseTransform, Result, RustMlError, Transform};
use faer::linalg::solvers::{SelfAdjointEigen, Svd};
use faer::{Mat, Side};
use ndarray::{Array1, Array2, Axis};

/// Parameters for PCA (unfitted state).
///
/// Principal Component Analysis reduces dimensionality by projecting data
/// onto the directions of maximum variance.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Pca {
    /// Number of principal components to keep.
    pub n_components: usize,
    /// When `true`, `transform` divides each projected component by
    /// `sqrt(explained_variance)` so the output has unit component-wise
    /// variance (sklearn's `whiten=True`).
    #[serde(default)]
    pub whiten: bool,
}

impl Pca {
    /// Create a new `Pca` with the given number of components (no whitening).
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            whiten: false,
        }
    }

    /// Enable / disable whitening (sklearn's `whiten`).
    pub fn with_whiten(mut self, whiten: bool) -> Self {
        self.whiten = whiten;
        self
    }
}

/// Fitted PCA — holds learned principal components, explained variance, and mean.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedPca<F: Float> {
    /// Principal component directions, shape (n_components, n_features).
    /// Each row is a unit eigenvector of the covariance matrix.
    components: Array2<F>,
    /// Variance explained by each component (eigenvalues), length n_components.
    explained_variance: Array1<F>,
    /// Fraction of the total variance explained by each component.
    #[serde(default)]
    explained_variance_ratio: Array1<F>,
    /// Singular values of the centred data for each component.
    #[serde(default)]
    singular_values: Array1<F>,
    /// Mean of the discarded eigenvalues (probabilistic-PCA noise variance).
    #[serde(default)]
    noise_variance: F,
    /// Per-feature mean used for centering, length n_features.
    mean: Array1<F>,
    /// Whether `transform` whitens its output.
    #[serde(default)]
    whiten: bool,
}

/// Use the covariance eigendecomposition when the data is this many times
/// taller than wide (sklearn's `"auto"` heuristic).
const COV_EIGH_RATIO: usize = 10;
/// ... and the covariance matrix stays small.
const COV_EIGH_MAX_FEATURES: usize = 1000;

/// Flip the sign of each row of `vt` (and nothing else) so that the entry of
/// largest absolute value is positive — `sklearn.utils.extmath.svd_flip`
/// with `u_based_decision=False`. Ties pick the first index (`np.argmax`).
pub(crate) fn svd_flip_rows(vt: &mut Array2<f64>) -> Vec<f64> {
    let mut signs = Vec::with_capacity(vt.nrows());
    for mut row in vt.rows_mut() {
        let mut best = 0usize;
        let mut best_abs = f64::NEG_INFINITY;
        for (j, &v) in row.iter().enumerate() {
            if v.abs() > best_abs {
                best_abs = v.abs();
                best = j;
            }
        }
        let sign = if row.is_empty() || row[best] >= 0.0 {
            1.0
        } else {
            -1.0
        };
        if sign < 0.0 {
            row.mapv_inplace(|v| -v);
        }
        signs.push(sign);
    }
    signs
}

impl<F: Float> FitUnsupervised<F> for Pca {
    type Fitted = FittedPca<F>;

    fn fit(&self, x: &Array2<F>) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(RustMlError::EmptyInput("input array is empty".into()));
        }

        if self.n_components == 0 {
            return Err(RustMlError::InvalidParameter(
                "n_components must be at least 1".into(),
            ));
        }

        if self.n_components > n_features {
            return Err(RustMlError::InvalidParameter(format!(
                "n_components ({}) must be <= n_features ({})",
                self.n_components, n_features
            )));
        }

        if n_samples < 2 {
            return Err(RustMlError::InvalidParameter(
                "PCA requires at least 2 samples to compute covariance".into(),
            ));
        }

        let k = self.n_components;
        let to_f64 = |v: F| v.to_f64().unwrap_or(f64::NAN);
        let xf: Array2<f64> = x.mapv(to_f64);
        if xf.iter().any(|v| !v.is_finite()) {
            return Err(RustMlError::InvalidParameter(
                "PCA input contains NaN or infinite values".into(),
            ));
        }

        // 1. Centre.
        let mean = xf.mean_axis(Axis(0)).expect("n_samples > 0 checked above");
        let xc = &xf - &mean;
        let dof = (n_samples - 1) as f64;

        // 2. Full spectrum (descending) and the matching right-singular
        //    vectors as rows of `vt` (only the first k are needed).
        let (spectrum, mut vt): (Vec<f64>, Array2<f64>) =
            if n_samples >= COV_EIGH_RATIO * n_features && n_features <= COV_EIGH_MAX_FEATURES {
                // Covariance eigendecomposition: C = Xcᵀ Xc / (n - 1).
                let cov = xc.t().dot(&xc) / dof;
                let cm = Mat::<f64>::from_fn(n_features, n_features, |i, j| cov[[i, j]]);
                let eig = SelfAdjointEigen::new(cm.as_ref(), Side::Lower).map_err(|e| {
                    RustMlError::InvalidParameter(format!("PCA eigendecomposition failed: {e:?}"))
                })?;
                let evals = eig.S().column_vector();
                let evecs = eig.U();
                let mut order: Vec<usize> = (0..n_features).collect();
                order.sort_by(|&a, &b| evals[b].total_cmp(&evals[a]));
                let spectrum: Vec<f64> = order.iter().map(|&i| evals[i].max(0.0)).collect();
                let vt = Array2::from_shape_fn((k, n_features), |(c, j)| evecs[(j, order[c])]);
                (spectrum, vt)
            } else {
                // Thin SVD of the centred data: Xc = U Σ Vᵀ.
                let m = Mat::<f64>::from_fn(n_samples, n_features, |i, j| xc[[i, j]]);
                let svd = Svd::new_thin(m.as_ref())
                    .map_err(|e| RustMlError::InvalidParameter(format!("PCA SVD failed: {e:?}")))?;
                let s = svd.S().column_vector();
                let v = svd.V();
                let r = s.nrows();
                let mut order: Vec<usize> = (0..r).collect();
                order.sort_by(|&a, &b| s[b].total_cmp(&s[a]));
                let spectrum: Vec<f64> = order.iter().map(|&i| s[i] * s[i] / dof).collect();
                // k <= n_features; if k > min(n, d) the trailing components are
                // zero-variance directions — complete them with an orthonormal
                // basis of the null space below.
                let kk = k.min(r);
                let mut vt = Array2::<f64>::zeros((k, n_features));
                for c in 0..kk {
                    for j in 0..n_features {
                        vt[[c, j]] = v[(j, order[c])];
                    }
                }
                if kk < k {
                    complete_orthonormal_rows(&mut vt, kk);
                }
                (spectrum, vt)
            };

        // 3. Deterministic signs (sklearn >= 1.5 convention).
        svd_flip_rows(&mut vt);

        let explained: Vec<f64> = (0..k)
            .map(|c| spectrum.get(c).copied().unwrap_or(0.0))
            .collect();
        let total_var: f64 = spectrum.iter().sum();
        let ratio: Vec<f64> = explained
            .iter()
            .map(|&e| if total_var > 0.0 { e / total_var } else { 0.0 })
            .collect();
        let singular: Vec<f64> = explained.iter().map(|&e| (e * dof).sqrt()).collect();
        let rank_cap = n_samples.min(n_features);
        let noise_variance = if k < rank_cap && spectrum.len() > k {
            let tail = &spectrum[k..rank_cap.min(spectrum.len())];
            tail.iter().sum::<f64>() / tail.len() as f64
        } else {
            0.0
        };

        let from = |v: f64| F::from_f64(v).unwrap();
        Ok(FittedPca {
            components: vt.mapv(from),
            explained_variance: explained.into_iter().map(from).collect(),
            explained_variance_ratio: ratio.into_iter().map(from).collect(),
            singular_values: singular.into_iter().map(from).collect(),
            noise_variance: from(noise_variance),
            mean: mean.mapv(from),
            whiten: self.whiten,
        })
    }
}

/// Fill rows `start..` of `rows` (whose first `start` rows are orthonormal)
/// with further orthonormal vectors via Gram–Schmidt over the standard basis.
fn complete_orthonormal_rows(rows: &mut Array2<f64>, start: usize) {
    let (k, d) = rows.dim();
    let mut filled = start;
    for basis in 0..d {
        if filled >= k {
            break;
        }
        let mut v = Array1::<f64>::zeros(d);
        v[basis] = 1.0;
        for _ in 0..2 {
            for p in 0..filled {
                let prev = rows.row(p);
                let dot = v.dot(&prev);
                v.scaled_add(-dot, &prev);
            }
        }
        let norm = v.dot(&v).sqrt();
        if norm > 1e-8 {
            v.mapv_inplace(|x| x / norm);
            rows.row_mut(filled).assign(&v);
            filled += 1;
        }
    }
}

impl<F: Float> FittedPca<F> {
    /// Per-component whitening scale `sqrt(explained_variance)`, clamped
    /// below at machine epsilon like sklearn.
    fn whiten_scale(&self) -> Array1<F> {
        let eps = F::epsilon();
        self.explained_variance.mapv(|v| {
            let s = v.max(F::zero()).sqrt();
            if s < eps {
                eps
            } else {
                s
            }
        })
    }
}

impl<F: Float> Transform<F> for FittedPca<F> {
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>> {
        let n_features = self.mean.len();
        if x.ncols() != n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                n_features,
                x.ncols()
            )));
        }

        // Center and project: (X - mean) @ components.T
        let centered = x - &self.mean;
        let mut out = centered.dot(&self.components.t());
        if self.whiten {
            out /= &self.whiten_scale();
        }
        Ok(out)
    }
}

impl<F: Float> InverseTransform<F> for FittedPca<F> {
    fn inverse_transform(&self, x: &Array2<F>) -> Result<Array2<F>> {
        let n_components = self.components.nrows();
        if x.ncols() != n_components {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} components, got {}",
                n_components,
                x.ncols()
            )));
        }

        // Reconstruct: X_reduced @ components + mean (undoing whitening first).
        if self.whiten {
            let scaled = x * &self.explained_variance.mapv(|v| v.max(F::zero()).sqrt());
            Ok(scaled.dot(&self.components) + &self.mean)
        } else {
            Ok(x.dot(&self.components) + &self.mean)
        }
    }
}

impl<F: Float> FittedPca<F> {
    /// Principal component directions, shape (n_components, n_features)
    /// (sklearn's `components_`).
    pub fn components(&self) -> &Array2<F> {
        &self.components
    }

    /// Variance explained by each component (sklearn's `explained_variance_`,
    /// i.e. eigenvalues of the covariance matrix with `ddof = 1`).
    pub fn explained_variance(&self) -> &Array1<F> {
        &self.explained_variance
    }

    /// Fraction of total variance explained by each component
    /// (sklearn's `explained_variance_ratio_`). All zeros for constant data.
    pub fn explained_variance_ratio(&self) -> &Array1<F> {
        &self.explained_variance_ratio
    }

    /// Singular values of the centred training data for each component
    /// (sklearn's `singular_values_`).
    pub fn singular_values(&self) -> &Array1<F> {
        &self.singular_values
    }

    /// Mean of the discarded eigenvalues (sklearn's `noise_variance_`);
    /// zero when all `min(n_samples, n_features)` components are kept.
    pub fn noise_variance(&self) -> F {
        self.noise_variance
    }

    /// Whether `transform` whitens its output.
    pub fn whiten(&self) -> bool {
        self.whiten
    }

    /// Per-feature mean used for centering.
    pub fn mean(&self) -> &Array1<F> {
        &self.mean
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_first_component_captures_most_variance() {
        // 2D data with a clear principal axis along (1, 1).
        // Variance along (1,1) is much larger than along (1,-1).
        let x = array![
            [1.0, 1.0],
            [2.0, 2.1],
            [3.0, 2.9],
            [4.0, 4.0],
            [5.0, 5.1],
            [6.0, 5.9],
            [7.0, 7.0],
            [8.0, 8.1],
        ];

        let pca = Pca::new(2);
        let fitted = FitUnsupervised::<f64>::fit(&pca, &x).unwrap();

        let var = fitted.explained_variance();

        // First component should capture the vast majority of variance.
        let total: f64 = var.iter().copied().sum();
        let ratio = var[0] / total;
        assert!(
            ratio > 0.95,
            "first component should capture >95% variance, got {:.4}",
            ratio
        );
    }

    #[test]
    fn test_transform_inverse_transform_roundtrip() {
        // With n_components == n_features, roundtrip should be exact.
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ];

        let pca = Pca::new(3);
        let fitted = FitUnsupervised::<f64>::fit(&pca, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();
        let recovered = fitted.inverse_transform(&transformed).unwrap();

        for (a, b) in x.iter().zip(recovered.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_transform_inverse_transform_lossy() {
        // With fewer components, roundtrip is approximate.
        let x = array![
            [1.0, 2.0, 0.5],
            [2.0, 4.0, 1.0],
            [3.0, 6.0, 1.5],
            [4.0, 8.0, 2.0],
            [5.0, 10.0, 2.5],
        ];

        let pca = Pca::new(1);
        let fitted = FitUnsupervised::<f64>::fit(&pca, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();
        let recovered = fitted.inverse_transform(&transformed).unwrap();

        // The data is nearly rank-1 (cols 2 and 3 are ~2x and ~0.5x col 1),
        // so even 1 component should give a reasonable reconstruction.
        for (a, b) in x.iter().zip(recovered.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 0.1);
        }
    }

    #[test]
    fn test_explained_variance_sorted_descending() {
        // Data with three genuinely distinct variance directions.
        let x = array![
            [1.0, 0.5, 0.1],
            [2.0, 1.0, 0.3],
            [3.0, 1.4, 0.2],
            [4.0, 2.1, 0.5],
            [5.0, 2.5, 0.8],
            [6.0, 3.2, 0.4],
            [7.0, 3.6, 0.9],
        ];

        let pca = Pca::new(3);
        let fitted = FitUnsupervised::<f64>::fit(&pca, &x).unwrap();
        let var = fitted.explained_variance();

        // All eigenvalues should be non-negative.
        for (i, &v) in var.iter().enumerate() {
            assert!(v >= 0.0, "explained_variance[{}] = {} is negative", i, v);
        }

        for i in 1..var.len() {
            assert!(
                var[i - 1] >= var[i],
                "explained_variance not sorted descending: var[{}]={} < var[{}]={}",
                i - 1,
                var[i - 1],
                i,
                var[i]
            );
        }
    }

    #[test]
    fn test_n_components_exceeds_n_features() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let pca = Pca::new(5);
        let result = FitUnsupervised::<f64>::fit(&pca, &x);
        assert!(result.is_err());

        let err = result.unwrap_err();
        match err {
            RustMlError::InvalidParameter(msg) => {
                assert!(
                    msg.contains("n_components"),
                    "error should mention n_components: {}",
                    msg
                );
            }
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }

    #[test]
    fn test_components_are_unit_vectors() {
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ];

        let pca = Pca::new(2);
        let fitted = FitUnsupervised::<f64>::fit(&pca, &x).unwrap();

        for row in fitted.components().rows() {
            let norm: f64 = row.iter().map(|&v| v * v).sum::<f64>().sqrt();
            assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_mean_is_correct() {
        let x = array![[1.0, 4.0], [3.0, 6.0]];

        let pca = Pca::new(2);
        let fitted = FitUnsupervised::<f64>::fit(&pca, &x).unwrap();

        assert_abs_diff_eq!(fitted.mean()[0], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(fitted.mean()[1], 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_shape_mismatch_on_transform() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let pca = Pca::new(1);
        let fitted = FitUnsupervised::<f64>::fit(&pca, &x).unwrap();

        let wrong = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&wrong).is_err());
    }

    #[test]
    fn test_empty_input() {
        let x = Array2::<f64>::zeros((0, 3));

        let pca = Pca::new(1);
        let result = FitUnsupervised::<f64>::fit(&pca, &x);
        assert!(result.is_err());
    }

    #[test]
    fn test_single_sample_error() {
        let x = array![[1.0, 2.0, 3.0]];

        let pca = Pca::new(1);
        let result = FitUnsupervised::<f64>::fit(&pca, &x);
        assert!(result.is_err());
    }

    #[test]
    fn test_constant_features() {
        // All features identical — zero variance.
        let x = array![[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]];

        let pca = Pca::new(2);
        let fitted = FitUnsupervised::<f64>::fit(&pca, &x).unwrap();

        // All eigenvalues should be zero (or near-zero).
        for &v in fitted.explained_variance().iter() {
            assert!(v.abs() < 1e-10, "expected near-zero variance, got {}", v);
        }
    }

    #[test]
    fn test_large_values() {
        // Large feature values should not produce NaN/Inf
        let x = array![[1e10, 2e10], [3e10, 4e10], [5e10, 6e10], [7e10, 8e10],];

        let pca = Pca::new(2);
        let fitted = FitUnsupervised::<f64>::fit(&pca, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        for &v in transformed.iter() {
            assert!(
                v.is_finite(),
                "PCA on large values produced non-finite: {}",
                v
            );
        }
        for &v in fitted.explained_variance().iter() {
            assert!(
                v.is_finite() && v >= 0.0,
                "variance should be finite and non-negative: {}",
                v
            );
        }
    }

    #[test]
    fn test_near_zero_variance_column() {
        // One column has near-zero variance, other column has real variance
        let x = array![
            [1.0, 5.0],
            [2.0, 5.0 + 1e-14],
            [3.0, 5.0 - 1e-14],
            [4.0, 5.0],
        ];

        let pca = Pca::new(2);
        let fitted = FitUnsupervised::<f64>::fit(&pca, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        for &v in transformed.iter() {
            assert!(
                v.is_finite(),
                "near-zero variance column produced non-finite: {}",
                v
            );
        }
        // First component should capture nearly all variance
        let var = fitted.explained_variance();
        assert!(var[0] > var[1] * 1e6, "first component should dominate");
    }

    #[test]
    fn test_collinear_features() {
        // Features 1 and 2 are perfectly collinear (col2 = 2*col1)
        // PCA should handle this gracefully
        let x = array![
            [1.0, 2.0, 0.5],
            [2.0, 4.0, 1.0],
            [3.0, 6.0, 1.5],
            [4.0, 8.0, 2.0],
            [5.0, 10.0, 2.5],
        ];

        let pca = Pca::new(3);
        let fitted = FitUnsupervised::<f64>::fit(&pca, &x).unwrap();
        let var = fitted.explained_variance();

        // All values should be finite and non-negative
        for &v in var.iter() {
            assert!(
                v.is_finite() && v >= -1e-10,
                "variance should be finite and non-negative: {}",
                v
            );
        }
        // With perfect collinearity, effective rank is 1, so at most 1 non-zero eigenvalue
        let nonzero_count = var.iter().filter(|&&v| v > 1e-8).count();
        assert!(
            nonzero_count <= 2,
            "collinear data should have rank <= 2, got {} non-zero eigenvalues",
            nonzero_count
        );
    }

    // ---- Regression tests for issue #51 (power iteration replaced by faer) ----
    // Reference values from sklearn 1.8.0 (`PCA(...).fit(X)`).

    fn assert_close(a: &[f64], b: &[f64], tol: f64, what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: length mismatch");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() < tol, "{what}[{i}]: got {x}, expected {y}");
        }
    }

    /// Same deterministic generator as the sklearn reference script.
    fn gen(n: usize, d: usize) -> Array2<f64> {
        Array2::from_shape_fn((n, d), |(i, j)| {
            let (fi, fj) = (i as f64, j as f64);
            (0.37 * (fi + 1.0) * (fj + 1.0)).sin()
                + 0.05 * (fj + 1.0) * (0.11 * fi).cos()
                + 0.3 * fj
        })
    }

    #[test]
    fn test_leading_eigvec_orthogonal_to_old_power_iteration_start() {
        // Leading eigenvector (2, -1)/√5 is exactly orthogonal to the old
        // power-iteration start vector (1, 2): the previous implementation
        // returned the wrong component here.
        let s5 = 5f64.sqrt();
        let t = [-9.0, -6.0, -3.0, 0.0, 3.0, 6.0, 9.0];
        let s = [1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0];
        let x = Array2::from_shape_fn((7, 2), |(i, j)| {
            let u1 = [2.0 / s5, -1.0 / s5];
            let u2 = [1.0 / s5, 2.0 / s5];
            t[i] * u1[j] + s[i] * u2[j] + [10.0, 5.0][j]
        });
        let fitted = FitUnsupervised::<f64>::fit(&Pca::new(2), &x).unwrap();
        let c = fitted.components();
        // sklearn: [[0.894427, -0.447214], [0.447214, 0.894427]] (svd_flip
        // makes the largest-|.| entry positive, so the sign matches exactly).
        assert_close(
            c.row(0).as_slice().unwrap(),
            &[2.0 / s5, -1.0 / s5],
            1e-12,
            "component 0",
        );
        assert_close(
            c.row(1).as_slice().unwrap(),
            &[1.0 / s5, 2.0 / s5],
            1e-12,
            "component 1",
        );
        assert_close(
            fitted.explained_variance().as_slice().unwrap(),
            &[42.0, 2.0 / 3.0],
            1e-10,
            "explained_variance",
        );
        assert_close(
            fitted.explained_variance_ratio().as_slice().unwrap(),
            &[0.984375, 0.015625],
            1e-12,
            "explained_variance_ratio",
        );
        assert_close(
            fitted.singular_values().as_slice().unwrap(),
            &[15.87450786638754, 2.0],
            1e-10,
            "singular_values",
        );
    }

    #[test]
    fn test_matches_sklearn_covariance_eigh_path() {
        // n >= 10 d -> covariance eigendecomposition (sklearn picks
        // `covariance_eigh` for this shape too). Eigenvalues are close
        // together, which defeated the old power iteration.
        let x = gen(60, 4);
        let fitted = FitUnsupervised::<f64>::fit(&Pca::new(3), &x).unwrap();
        let sk_components = [
            [
                0.43465036352959263,
                0.34690122652812294,
                0.4720103074709995,
                0.6840649604811003,
            ],
            [
                0.15277636253519716,
                -0.29047533450217244,
                0.8002206472583974,
                -0.5019266667529398,
            ],
            [
                -0.011357603288638895,
                0.8851536426385769,
                0.032418374467622524,
                -0.46402918309959057,
            ],
        ];
        for (c, sk) in sk_components.iter().enumerate() {
            assert_close(
                fitted.components().row(c).as_slice().unwrap(),
                sk,
                1e-8,
                "components",
            );
        }
        assert_close(
            fitted.explained_variance().as_slice().unwrap(),
            &[0.5687491695829472, 0.519341135444794, 0.5036883392722767],
            1e-10,
            "explained_variance",
        );
        assert_close(
            fitted.explained_variance_ratio().as_slice().unwrap(),
            &[0.27291051841484626, 0.2492024007917659, 0.24169151032098762],
            1e-10,
            "explained_variance_ratio",
        );
        assert_close(
            fitted.singular_values().as_slice().unwrap(),
            &[5.79277144425653, 5.53544279992512, 5.451386247282826],
            1e-9,
            "singular_values",
        );
        assert!((fitted.noise_variance() - 0.49223472713919486).abs() < 1e-10);

        let w = FitUnsupervised::<f64>::fit(&Pca::new(3).with_whiten(true), &x).unwrap();
        let t = w.transform(&x).unwrap();
        assert_close(
            t.row(0).as_slice().unwrap(),
            &[2.2432797526890393, 0.07212630178619289, 0.22495030606297606],
            1e-8,
            "whitened transform",
        );
    }

    #[test]
    fn test_matches_sklearn_full_svd_path() {
        // Wide-ish data (n < 10 d) -> thin SVD of the centred data.
        let x = gen(8, 5);
        let fitted = FitUnsupervised::<f64>::fit(&Pca::new(3), &x).unwrap();
        let sk_components = [
            [
                -0.025817769965272182,
                0.7109427605328162,
                0.06159113357117507,
                0.6312151348317079,
                0.3027669397817486,
            ],
            [
                -0.19256623031912473,
                0.25021945575636345,
                0.8306602688793396,
                -0.16563052203253772,
                -0.4276429805085115,
            ],
            [
                0.07506519464821175,
                -0.47919800117048234,
                0.08182192585857773,
                0.748295506883965,
                -0.44507695857312624,
            ],
        ];
        for (c, sk) in sk_components.iter().enumerate() {
            assert_close(
                fitted.components().row(c).as_slice().unwrap(),
                sk,
                1e-8,
                "components",
            );
        }
        assert_close(
            fitted.explained_variance().as_slice().unwrap(),
            &[0.6397690686207292, 0.6064827650042603, 0.6061044971247672],
            1e-10,
            "explained_variance",
        );
        assert_close(
            fitted.explained_variance_ratio().as_slice().unwrap(),
            &[0.2606505884301114, 0.24708929725515563, 0.24693518579492724],
            1e-10,
            "explained_variance_ratio",
        );
        assert_close(
            fitted.singular_values().as_slice().unwrap(),
            &[2.1162191475235037, 2.060431837025875, 2.0597891833567266],
            1e-9,
            "singular_values",
        );
        assert!((fitted.noise_variance() - 0.3010760535279718).abs() < 1e-10);

        let w = FitUnsupervised::<f64>::fit(&Pca::new(3).with_whiten(true), &x).unwrap();
        let t = w.transform(&x).unwrap();
        assert_close(
            t.row(0).as_slice().unwrap(),
            &[1.8292916744554633, 0.34294379475174414, 0.12221105200304454],
            1e-8,
            "whitened transform",
        );
        // Whitening round-trips through inverse_transform.
        let full = FitUnsupervised::<f64>::fit(&Pca::new(5).with_whiten(true), &x).unwrap();
        let back = full
            .inverse_transform(&full.transform(&x).unwrap())
            .unwrap();
        for (a, b) in x.iter().zip(back.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_components_orthonormal_when_k_exceeds_rank() {
        // n = 3 samples, d = 5 features -> rank <= 2; asking for 4 components
        // must still give an orthonormal basis.
        let x = gen(3, 5);
        let fitted = FitUnsupervised::<f64>::fit(&Pca::new(4), &x).unwrap();
        let c = fitted.components();
        let gram = c.dot(&c.t());
        for i in 0..4 {
            for j in 0..4 {
                let e = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(gram[[i, j]], e, epsilon = 1e-10);
            }
        }
        assert!(fitted.explained_variance()[2].abs() < 1e-12);
    }

    #[test]
    fn test_nan_input_is_rejected() {
        let x = array![[1.0, 2.0], [f64::NAN, 3.0], [2.0, 1.0]];
        assert!(FitUnsupervised::<f64>::fit(&Pca::new(1), &x).is_err());
    }

    #[test]
    fn test_f32_input() {
        let x = gen(20, 3).mapv(|v| v as f32);
        let fitted = FitUnsupervised::<f32>::fit(&Pca::new(2), &x).unwrap();
        assert_eq!(fitted.components().dim(), (2, 3));
        assert!(fitted.explained_variance()[0] >= fitted.explained_variance()[1]);
    }
}
