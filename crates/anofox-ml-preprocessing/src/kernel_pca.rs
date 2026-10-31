//! Kernel PCA.
//!
//! Mirrors `sklearn.decomposition.KernelPCA` for linear / RBF / polynomial
//! kernels. Returns coordinates `α_k √λ_k` for the top `k` eigenpairs of the
//! centered kernel matrix.

use anofox_ml_core::{FitUnsupervised, InverseTransform, Result, RustMlError, Transform};
use faer::linalg::solvers::SelfAdjointEigen;
use faer::{Mat, Side};
use ndarray::{Array1, Array2};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum KpcaKernel {
    Linear,
    Rbf {
        gamma: f64,
    },
    Polynomial {
        degree: usize,
        coef0: f64,
        gamma: f64,
    },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KernelPca {
    pub n_components: usize,
    pub kernel: KpcaKernel,
    /// Drop components with zero (or negative) eigenvalue (sklearn
    /// `remove_zero_eig`). Default false.
    pub remove_zero_eig: bool,
}

impl KernelPca {
    pub fn new(n_components: usize, kernel: KpcaKernel) -> Self {
        Self {
            n_components,
            kernel,
            remove_zero_eig: false,
        }
    }
    pub fn with_remove_zero_eig(mut self, remove: bool) -> Self {
        self.remove_zero_eig = remove;
        self
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedKernelPca {
    pub x_train: Array2<f64>,
    pub alphas: Array2<f64>,      // n_train × k
    pub eigenvalues: Array1<f64>, // k
    pub row_means: Array1<f64>,   // training kernel row means
    pub global_mean: f64,
    pub kernel: KpcaKernel,
}

/// Kernel matrix via one GEMM: every kernel here is a function of `a·b`
/// (and of `‖a‖²`, `‖b‖²` for RBF).
fn build_kernel(x_a: &Array2<f64>, x_b: &Array2<f64>, k: &KpcaKernel) -> Array2<f64> {
    let g = x_a.dot(&x_b.t());
    match k {
        KpcaKernel::Linear => g,
        KpcaKernel::Polynomial {
            degree,
            coef0,
            gamma,
        } => g.mapv(|v| (gamma * v + coef0).powi(*degree as i32)),
        KpcaKernel::Rbf { gamma } => {
            let na: Array1<f64> = x_a.rows().into_iter().map(|r| r.dot(&r)).collect();
            let nb: Array1<f64> = x_b.rows().into_iter().map(|r| r.dot(&r)).collect();
            let mut out = g;
            for ((i, j), v) in out.indexed_iter_mut() {
                *v = (-gamma * (na[i] + nb[j] - 2.0 * *v).max(0.0)).exp();
            }
            out
        }
    }
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
            return Err(RustMlError::InvalidParameter(
                "n_components must be >= 1".into(),
            ));
        }

        let mut k = build_kernel(x, x, &self.kernel);
        // Centre: K_c = K - 1ₙ K - K 1ₙ + 1ₙ K 1ₙ where 1ₙ = ones(n,n) / n.
        let row_means: Array1<f64> =
            Array1::from_vec((0..n).map(|i| k.row(i).sum() / n as f64).collect());
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

        // sklearn: eigenvalues in descending order, tiny negatives clipped
        // to 0 (`_check_psd_eigenvalues`), eigenvector signs fixed so that
        // the largest-|·| entry of every column is positive (`svd_flip`).
        let scale = s.column_vector()[n - 1].abs().max(1.0);
        let mut cols: Vec<(f64, usize)> = (0..k_target)
            .map(|c| {
                let src = n - 1 - c;
                let val = s.column_vector()[src];
                (
                    if val < 1e-10 * scale {
                        0.0f64.max(val.max(0.0))
                    } else {
                        val
                    },
                    src,
                )
            })
            .collect();
        if self.remove_zero_eig {
            cols.retain(|&(v, _)| v > 0.0);
        }
        let kk = cols.len();
        let mut alphas = Array2::<f64>::zeros((n, kk));
        let mut eigenvalues = Array1::<f64>::zeros(kk);
        for (c, &(val, src)) in cols.iter().enumerate() {
            eigenvalues[c] = if val < 1e-10 * scale { 0.0 } else { val };
            let mut best = 0;
            for i in 1..n {
                if v[(i, src)].abs() > v[(best, src)].abs() {
                    best = i;
                }
            }
            let sign = if v[(best, src)] < 0.0 { -1.0 } else { 1.0 };
            for i in 0..n {
                alphas[[i, c]] = sign * v[(i, src)];
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
            (0..n_new)
                .map(|i| k_new.row(i).sum() / n_train as f64)
                .collect(),
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
        // Zero-eigenvalue components project to 0 (sklearn).
        let inv_sqrt = self
            .eigenvalues
            .mapv(|l| if l > 0.0 { 1.0 / l.sqrt() } else { 0.0 });
        let out = k_new.dot(&self.alphas) * &inv_sqrt;
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
                "expected {} components, got {}",
                k,
                t.ncols()
            )));
        }
        let n_train = self.x_train.nrows();
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
            [0.0_f64, 0.0],
            [1.0, 1.0],
            [2.0, 4.0],
            [3.0, 9.0],
            [4.0, 16.0],
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
            [0.0_f64, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [2.0, 4.0, -1.0],
            [3.0, 9.0, 2.0],
            [4.0, 16.0, 0.5],
        ];
        let fitted = KernelPca::new(2, KpcaKernel::Linear).fit(&x).unwrap();
        let t = fitted.transform(&x).unwrap();
        let back = fitted.inverse_transform(&t).unwrap();
        assert_eq!(back.shape(), &[5, 3]);
        for v in back.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_rbf_transform_matches_sklearn() {
        let xv = [
            1.624, -0.612, -0.528, -1.073, 0.865, -2.302, 1.745, -0.761, 0.319, -0.249, 1.462,
            -2.06, -0.322, -0.384, 1.134, -1.1, -0.172, -0.878, 0.042, 0.583, -1.101, 1.145, 0.902,
            0.502, 0.901, -0.684, -0.123, -0.936, -0.268, 0.53,
        ];
        let x = Array2::from_shape_vec((10, 3), xv.to_vec()).unwrap();
        // sklearn 1.8 KernelPCA(2, kernel='rbf', gamma=0.4)
        let f = KernelPca::new(2, KpcaKernel::Rbf { gamma: 0.4 })
            .fit(&x)
            .unwrap();
        let t = f
            .transform(&x.slice(ndarray::s![..3, ..]).to_owned())
            .unwrap();
        let sk = [
            [0.6244826994353455, -0.2987886641553929],
            [-0.5996187251420725, -0.3894362414355089],
            [0.6440933959886704, -0.24267860872085467],
        ];
        for i in 0..3 {
            for c in 0..2 {
                assert!(
                    (t[[i, c]] - sk[i][c]).abs() < 1e-9,
                    "[{i},{c}] {}",
                    t[[i, c]]
                );
            }
        }
        assert!((f.eigenvalues[0] - 2.197718892514534).abs() < 1e-9);
        assert!((f.eigenvalues[1] - 1.6941472173838554).abs() < 1e-9);
        // Constant data: all eigenvalues 0 -> removed / projected to 0.
        let c = Array2::from_elem((4, 2), 1.0);
        assert_eq!(
            KernelPca::new(2, KpcaKernel::Linear)
                .with_remove_zero_eig(true)
                .fit(&c)
                .unwrap()
                .eigenvalues
                .len(),
            0
        );
        let z = KernelPca::new(2, KpcaKernel::Linear).fit(&c).unwrap();
        assert!(z.transform(&c).unwrap().iter().all(|v| *v == 0.0));
    }
}
