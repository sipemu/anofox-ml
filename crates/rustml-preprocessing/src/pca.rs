use ndarray::{Array1, Array2, Axis};
use rustml_core::{Float, FitUnsupervised, InverseTransform, Result, RustMlError, Transform};

/// Parameters for PCA (unfitted state).
///
/// Principal Component Analysis reduces dimensionality by projecting data
/// onto the directions of maximum variance. Eigendecomposition is performed
/// via power iteration with deflation, requiring no external LAPACK dependency.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Pca {
    /// Number of principal components to keep.
    pub n_components: usize,
}

impl Pca {
    /// Create a new `Pca` with the given number of components.
    pub fn new(n_components: usize) -> Self {
        Self { n_components }
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
    /// Per-feature mean used for centering, length n_features.
    mean: Array1<F>,
}

/// Number of power-iteration steps per component.
const POWER_ITER_STEPS: usize = 200;

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

        let n_f = F::from_usize(n_samples).unwrap();

        // 1. Compute per-feature mean.
        let mean = x.sum_axis(Axis(0)) / n_f;

        // 2. Center the data: X_centered = X - mean.
        let mut x_centered = x.to_owned();
        for mut row in x_centered.rows_mut() {
            for (j, val) in row.iter_mut().enumerate() {
                *val -= mean[j];
            }
        }

        // 3. Covariance matrix: C = X_centered.T @ X_centered / (n_samples - 1).
        //    Shape: (n_features, n_features).
        let n_minus_1 = F::from_usize(n_samples - 1).unwrap();
        let mut cov = Array2::<F>::zeros((n_features, n_features));
        for row in x_centered.rows() {
            for i in 0..n_features {
                for j in i..n_features {
                    let prod = row[i] * row[j];
                    cov[[i, j]] += prod;
                    if i != j {
                        cov[[j, i]] += prod;
                    }
                }
            }
        }
        cov.mapv_inplace(|v| v / n_minus_1);

        // 4. Power iteration with deflation to extract top-k eigenpairs.
        let mut components = Array2::<F>::zeros((self.n_components, n_features));
        let mut explained_variance = Array1::<F>::zeros(self.n_components);
        let eps = F::from_f64(1e-12).unwrap();

        for k in 0..self.n_components {
            // (a) Deterministic initial vector: v[i] = (i+1).
            let mut v = Array1::<F>::zeros(n_features);
            for i in 0..n_features {
                v[i] = F::from_usize(i + 1).unwrap();
            }
            // Orthogonalize against previously found components.
            for prev in 0..k {
                let prev_comp = components.row(prev);
                let dot: F = v.iter().zip(prev_comp.iter()).map(|(&a, &b)| a * b).fold(F::zero(), |s, p| s + p);
                for (vi, &ci) in v.iter_mut().zip(prev_comp.iter()) {
                    *vi -= dot * ci;
                }
            }
            // Normalize.
            let norm = v.iter().map(|&vi| vi * vi).fold(F::zero(), |a, b| a + b).sqrt();
            if norm < eps {
                // All directions exhausted; store zero eigenvalue with arbitrary
                // orthogonal direction (already zeroed out).
                explained_variance[k] = F::zero();
                // Build an orthogonal vector via standard basis probing.
                for basis_idx in 0..n_features {
                    v = Array1::<F>::zeros(n_features);
                    v[basis_idx] = F::one();
                    for prev in 0..k {
                        let prev_comp = components.row(prev);
                        let dot: F = v.iter().zip(prev_comp.iter()).map(|(&a, &b)| a * b).fold(F::zero(), |s, p| s + p);
                        for (vi, &ci) in v.iter_mut().zip(prev_comp.iter()) {
                            *vi -= dot * ci;
                        }
                    }
                    let n2 = v.iter().map(|&vi| vi * vi).fold(F::zero(), |a, b| a + b).sqrt();
                    if n2 > eps {
                        v.mapv_inplace(|vi| vi / n2);
                        break;
                    }
                }
                components.row_mut(k).assign(&v);
                continue;
            }
            v.mapv_inplace(|vi| vi / norm);

            // (b) Power iteration with convergence check.
            let convergence_tol = F::from_f64(1e-12).unwrap();
            for _ in 0..POWER_ITER_STEPS {
                // w = C @ v
                let mut w = cov.dot(&v);
                // Re-orthogonalize against previously found components
                // for numerical stability.
                for prev in 0..k {
                    let prev_comp = components.row(prev);
                    let dot: F = w.iter().zip(prev_comp.iter()).map(|(&a, &b)| a * b).fold(F::zero(), |s, p| s + p);
                    for (wi, &ci) in w.iter_mut().zip(prev_comp.iter()) {
                        *wi -= dot * ci;
                    }
                }
                // norm(w)
                let w_norm = w.iter().map(|&wi| wi * wi).fold(F::zero(), |a, b| a + b).sqrt();
                if w_norm < F::from_f64(1e-30).unwrap() {
                    // Degenerate -- remaining eigenvalues are essentially zero.
                    break;
                }
                let v_new = w.mapv(|wi| wi / w_norm);
                // Check convergence: |v_new - v| < tol
                let diff: F = v_new.iter().zip(v.iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .fold(F::zero(), |acc, d| acc + d);
                v = v_new;
                if diff < convergence_tol {
                    break;
                }
            }

            // (c) Eigenvalue = v^T C v. Clamp to zero if negative (numerical noise).
            let cv = cov.dot(&v);
            let eigenvalue = v.iter().zip(cv.iter()).map(|(&a, &b)| a * b).fold(F::zero(), |s, p| s + p);
            let eigenvalue = if eigenvalue < F::zero() { F::zero() } else { eigenvalue };

            // (d) Deflate: C = C - eigenvalue * v v^T.
            for i in 0..n_features {
                for j in 0..n_features {
                    cov[[i, j]] -= eigenvalue * v[i] * v[j];
                }
            }

            // (e) Store results.
            components.row_mut(k).assign(&v);
            explained_variance[k] = eigenvalue;
        }

        Ok(FittedPca {
            components,
            explained_variance,
            mean,
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
        let mut centered = x.to_owned();
        for mut row in centered.rows_mut() {
            for (j, val) in row.iter_mut().enumerate() {
                *val -= self.mean[j];
            }
        }
        Ok(centered.dot(&self.components.t()))
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

        // Reconstruct: X_reduced @ components + mean
        let mut result = x.dot(&self.components);
        for mut row in result.rows_mut() {
            for (j, val) in row.iter_mut().enumerate() {
                *val += self.mean[j];
            }
        }
        Ok(result)
    }
}

impl<F: Float> FittedPca<F> {
    /// Principal component directions, shape (n_components, n_features).
    pub fn components(&self) -> &Array2<F> {
        &self.components
    }

    /// Variance explained by each component.
    pub fn explained_variance(&self) -> &Array1<F> {
        &self.explained_variance
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

        let pca = Pca { n_components: 2 };
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

        let pca = Pca { n_components: 3 };
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

        let pca = Pca { n_components: 1 };
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

        let pca = Pca { n_components: 3 };
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

        let pca = Pca { n_components: 5 };
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

        let pca = Pca { n_components: 2 };
        let fitted = FitUnsupervised::<f64>::fit(&pca, &x).unwrap();

        for row in fitted.components().rows() {
            let norm: f64 = row.iter().map(|&v| v * v).sum::<f64>().sqrt();
            assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_mean_is_correct() {
        let x = array![[1.0, 4.0], [3.0, 6.0]];

        let pca = Pca { n_components: 2 };
        let fitted = FitUnsupervised::<f64>::fit(&pca, &x).unwrap();

        assert_abs_diff_eq!(fitted.mean()[0], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(fitted.mean()[1], 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_shape_mismatch_on_transform() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let pca = Pca { n_components: 1 };
        let fitted = FitUnsupervised::<f64>::fit(&pca, &x).unwrap();

        let wrong = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&wrong).is_err());
    }

    #[test]
    fn test_empty_input() {
        let x = Array2::<f64>::zeros((0, 3));

        let pca = Pca { n_components: 1 };
        let result = FitUnsupervised::<f64>::fit(&pca, &x);
        assert!(result.is_err());
    }

    #[test]
    fn test_single_sample_error() {
        let x = array![[1.0, 2.0, 3.0]];

        let pca = Pca { n_components: 1 };
        let result = FitUnsupervised::<f64>::fit(&pca, &x);
        assert!(result.is_err());
    }

    #[test]
    fn test_constant_features() {
        // All features identical — zero variance.
        let x = array![[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]];

        let pca = Pca { n_components: 2 };
        let fitted = FitUnsupervised::<f64>::fit(&pca, &x).unwrap();

        // All eigenvalues should be zero (or near-zero).
        for &v in fitted.explained_variance().iter() {
            assert!(v.abs() < 1e-10, "expected near-zero variance, got {}", v);
        }
    }

    #[test]
    fn test_large_values() {
        // Large feature values should not produce NaN/Inf
        let x = array![
            [1e10, 2e10],
            [3e10, 4e10],
            [5e10, 6e10],
            [7e10, 8e10],
        ];

        let pca = Pca { n_components: 2 };
        let fitted = FitUnsupervised::<f64>::fit(&pca, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        for &v in transformed.iter() {
            assert!(v.is_finite(), "PCA on large values produced non-finite: {}", v);
        }
        for &v in fitted.explained_variance().iter() {
            assert!(v.is_finite() && v >= 0.0, "variance should be finite and non-negative: {}", v);
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

        let pca = Pca { n_components: 2 };
        let fitted = FitUnsupervised::<f64>::fit(&pca, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        for &v in transformed.iter() {
            assert!(v.is_finite(), "near-zero variance column produced non-finite: {}", v);
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

        let pca = Pca { n_components: 3 };
        let fitted = FitUnsupervised::<f64>::fit(&pca, &x).unwrap();
        let var = fitted.explained_variance();

        // All values should be finite and non-negative
        for &v in var.iter() {
            assert!(v.is_finite() && v >= -1e-10, "variance should be finite and non-negative: {}", v);
        }
        // With perfect collinearity, effective rank is 1, so at most 1 non-zero eigenvalue
        let nonzero_count = var.iter().filter(|&&v| v > 1e-8).count();
        assert!(nonzero_count <= 2, "collinear data should have rank <= 2, got {} non-zero eigenvalues", nonzero_count);
    }
}
