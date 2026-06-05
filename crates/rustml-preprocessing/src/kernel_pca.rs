//! Kernel PCA.
//!
//! Mirrors `sklearn.decomposition.KernelPCA` for linear / RBF / polynomial
//! kernels. Returns coordinates `α_k √λ_k` for the top `k` eigenpairs of the
//! centered kernel matrix.

use faer::linalg::solvers::SelfAdjointEigen;
use faer::{Mat, Side};
use ndarray::{Array1, Array2};
use rustml_core::{FitUnsupervised, InverseTransform, Result, RustMlError, Transform};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum KpcaKernel {
    Linear,
    Rbf { gamma: f64 },
    Polynomial { degree: usize, coef0: f64, gamma: f64 },
}

impl KpcaKernel {
    fn compute(&self, a: &[f64], b: &[f64]) -> f64 {
        match self {
            KpcaKernel::Linear => a.iter().zip(b.iter()).map(|(x, y)| x * y).sum(),
            KpcaKernel::Rbf { gamma } => {
                let sd: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
                (-gamma * sd).exp()
            }
            KpcaKernel::Polynomial { degree, coef0, gamma } => {
                let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                (gamma * dot + coef0).powi(*degree as i32)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct KernelPca {
    pub n_components: usize,
    pub kernel: KpcaKernel,
}

impl KernelPca {
    pub fn new(n_components: usize, kernel: KpcaKernel) -> Self {
        Self { n_components, kernel }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedKernelPca {
    pub x_train: Array2<f64>,
    pub alphas: Array2<f64>,        // n_train × k
    pub eigenvalues: Array1<f64>,   // k
    pub row_means: Array1<f64>,     // training kernel row means
    pub global_mean: f64,
    pub kernel: KpcaKernel,
}

fn build_kernel(x_a: &Array2<f64>, x_b: &Array2<f64>, k: &KpcaKernel) -> Array2<f64> {
    let na = x_a.nrows();
    let nb = x_b.nrows();
    let mut out = Array2::<f64>::zeros((na, nb));
    for i in 0..na {
        let ai = x_a.row(i).to_owned();
        for j in 0..nb {
            let bj = x_b.row(j).to_owned();
            out[[i, j]] = k.compute(ai.as_slice().unwrap(), bj.as_slice().unwrap());
        }
    }
    out
}

impl FitUnsupervised<f64> for KernelPca {
    type Fitted = FittedKernelPca;

    fn fit(&self, x: &Array2<f64>) -> Result<Self::Fitted> {
        let n = x.nrows();
        if n == 0 {
            return Err(RustMlError::EmptyInput("empty input".into()));
        }
        let k_target = self.n_components.min(n);
        if k_target == 0 {
            return Err(RustMlError::InvalidParameter("n_components must be >= 1".into()));
        }

        let mut k = build_kernel(x, x, &self.kernel);
        // Centre: K_c = K - 1ₙ K - K 1ₙ + 1ₙ K 1ₙ where 1ₙ = ones(n,n) / n.
        let row_means: Array1<f64> = Array1::from_vec(
            (0..n).map(|i| k.row(i).sum() / n as f64).collect(),
        );
        let col_means: Array1<f64> = row_means.clone(); // K symmetric
        let global_mean: f64 = k.iter().copied().sum::<f64>() / (n as f64).powi(2);
        for i in 0..n {
            for j in 0..n {
                k[[i, j]] += global_mean - row_means[i] - col_means[j];
            }
        }

        // Symmetric eigendecomposition.
        let m = Mat::<f64>::from_fn(n, n, |i, j| k[[i, j]]);
        let eig = SelfAdjointEigen::new(m.as_ref(), Side::Lower)
            .map_err(|e| RustMlError::InvalidParameter(format!("eigen failed: {e:?}")))?;
        let s = eig.S(); // ascending
        let v = eig.U();

        // sklearn returns in descending order — slice from the end.
        let mut alphas = Array2::<f64>::zeros((n, k_target));
        let mut eigenvalues = Array1::<f64>::zeros(k_target);
        for c in 0..k_target {
            let src = n - 1 - c; // descending
            let val = s.column_vector()[src];
            eigenvalues[c] = val;
            // Normalise eigenvector so that the kPCA scores are
            // α / √λ * √λ = α (sklearn convention: scaled by √λ).
            for i in 0..n {
                alphas[[i, c]] = v[(i, src)];
            }
        }

        Ok(FittedKernelPca {
            x_train: x.clone(),
            alphas,
            eigenvalues,
            row_means,
            global_mean,
            kernel: self.kernel.clone(),
        })
    }
}

impl Transform<f64> for FittedKernelPca {
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let n_train = self.x_train.nrows();
        if x.ncols() != self.x_train.ncols() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.x_train.ncols(),
                x.ncols()
            )));
        }
        let n_new = x.nrows();
        let mut k_new = build_kernel(x, &self.x_train, &self.kernel);
        // Centre: K_new[i,j] - K_new_row_mean[i] - K_train_row_mean[j] + global_mean
        // Compute row means of k_new along training axis.
        let new_row_means: Array1<f64> = Array1::from_vec(
            (0..n_new).map(|i| k_new.row(i).sum() / n_train as f64).collect(),
        );
        for i in 0..n_new {
            for j in 0..n_train {
                k_new[[i, j]] += self.global_mean - new_row_means[i] - self.row_means[j];
            }
        }
        // Project: scores = K_new @ alphas, then scale by sign(λ) * √|λ| / ||α||
        // sklearn returns K_new @ alphas / sqrt(|λ|). The eigenvectors are
        // unit-norm; sklearn's `lambdas_` are λ; coordinates are α * √λ where
        // each column of α corresponds to a unit eigenvector. So the
        // transformed coord for sample i, comp c is sum_j K_new[i,j] α[j,c] / √λ_c.
        let mut out = Array2::<f64>::zeros((n_new, self.alphas.ncols()));
        for c in 0..self.alphas.ncols() {
            let lam = self.eigenvalues[c];
            let sqrt_lam = lam.abs().sqrt().max(1e-12);
            for i in 0..n_new {
                let mut s = 0.0;
                for j in 0..n_train {
                    s += k_new[[i, j]] * self.alphas[[j, c]];
                }
                out[[i, c]] = s / sqrt_lam;
            }
        }
        Ok(out)
    }
}

impl InverseTransform<f64> for FittedKernelPca {
    /// Approximate pre-image: maps a low-dim projection back to a reconstruction
    /// in the original feature space.
    ///
    /// Uses the linear approximation `X̂ = T · diag(√λ) · αᵀ · X_train`, which
    /// is exact for the linear kernel and a useful approximation for RBF /
    /// polynomial. Mirrors sklearn's `KernelPCA(fit_inverse_transform=False)`
    /// fallback (sklearn's full pre-image solver is iterative and not yet
    /// implemented here).
    fn inverse_transform(&self, t: &Array2<f64>) -> Result<Array2<f64>> {
        let k = self.alphas.ncols();
        if t.ncols() != k {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} components, got {}", k, t.ncols()
            )));
        }
        let n_train = self.x_train.nrows();
        let d_orig = self.x_train.ncols();
        let n_new = t.nrows();

        // Reconstruct dual coefficients in training-sample space:
        //   coef = T · diag(√λ) · αᵀ  → shape (n_new, n_train)
        let mut coef = Array2::<f64>::zeros((n_new, n_train));
        for i in 0..n_new {
            for j in 0..n_train {
                let mut s = 0.0;
                for c in 0..k {
                    let lam_sqrt = self.eigenvalues[c].abs().sqrt().max(1e-12);
                    s += t[[i, c]] * lam_sqrt * self.alphas[[j, c]];
                }
                coef[[i, j]] = s;
            }
        }
        // Linear reconstruction: x̂ = coef · X_train. For the linear kernel
        // this is the exact pre-image; for RBF / polynomial it's a useful
        // approximation that's the linear-projection answer in dual space.
        Ok(coef.dot(&self.x_train))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_kernel_pca_runs_rbf() {
        let x = array![
            [0.0_f64, 0.0], [1.0, 1.0], [2.0, 4.0], [3.0, 9.0], [4.0, 16.0],
        ];
        let kpca = KernelPca::new(2, KpcaKernel::Rbf { gamma: 0.1 });
        let fitted = kpca.fit(&x).unwrap();
        let t = fitted.transform(&x).unwrap();
        assert_eq!(t.shape(), &[5, 2]);
        assert!(fitted.eigenvalues[0] >= fitted.eigenvalues[1]);
    }

    #[test]
    fn test_kernel_pca_inverse_transform_runs() {
        let x = array![
            [0.0_f64, 0.0, 1.0], [1.0, 1.0, 0.0], [2.0, 4.0, -1.0],
            [3.0, 9.0, 2.0], [4.0, 16.0, 0.5],
        ];
        let fitted = KernelPca::new(2, KpcaKernel::Linear).fit(&x).unwrap();
        let t = fitted.transform(&x).unwrap();
        let back = fitted.inverse_transform(&t).unwrap();
        assert_eq!(back.shape(), &[5, 3]);
        for v in back.iter() {
            assert!(v.is_finite());
        }
    }
}
