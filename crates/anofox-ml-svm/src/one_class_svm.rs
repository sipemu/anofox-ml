//! One-Class SVM for unsupervised outlier / novelty detection.
//!
//! Learns a decision boundary in kernel space that encloses the training data.
//! New points are classified as inliers (+1) or outliers (-1).
//!
//! The dual optimisation problem:
//!   minimise  0.5 alpha^T K alpha
//!   subject to  0 <= alpha_i <= 1,  sum(alpha_i) = nu * n  (libsvm's scaling, as sklearn)
//!
//! Decision function: f(x) = sum_i alpha_i K(x_i, x) - rho
//! Prediction: sign(f(x)), +1 = inlier, -1 = outlier

use anofox_ml_core::{FitUnsupervised, Float, Predict, Result, RustMlError};
use ndarray::{Array1, Array2, ArrayView2, AsArray, Ix2};

use crate::kernel::SvmKernel;

/// One-Class SVM estimator (unfitted state).
///
/// Uses the nu-SVM formulation to learn a boundary around the training data
/// in kernel space. The parameter `nu` is an upper bound on the fraction of
/// outliers and a lower bound on the fraction of support vectors.
///
/// Uses the type-state pattern: call [`FitUnsupervised::fit`] to produce a
/// [`FittedOneClassSvm`] that can make predictions.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OneClassSvm {
    /// Fraction of outliers (0, 1]. Also a lower bound on the fraction of
    /// support vectors. Default: 0.5.
    pub nu: f64,
    /// Kernel function to use.
    pub kernel: SvmKernel,
    /// Maximum number of SMO iterations.
    pub max_iter: usize,
    /// Tolerance of the stopping criterion (libsvm / sklearn default 1e-3).
    pub tol: f64,
    /// Use libsvm's shrinking heuristic (default true).
    #[serde(default = "crate::solver::default_true")]
    pub shrinking: bool,
    /// Kernel cache size in MB (default 200).
    #[serde(default = "crate::solver::default_cache_size")]
    pub cache_size: f64,
}

impl OneClassSvm {
    /// Create a new `OneClassSvm` with default parameters.
    pub fn new() -> Self {
        Self {
            nu: 0.5,
            kernel: SvmKernel::default(),
            max_iter: 10_000_000,
            tol: 1e-3,
            shrinking: true,
            cache_size: 200.0,
        }
    }

    /// Set the nu parameter.
    pub fn with_nu(mut self, nu: f64) -> Self {
        self.nu = nu;
        self
    }

    /// Set the kernel function.
    pub fn with_kernel(mut self, kernel: SvmKernel) -> Self {
        self.kernel = kernel;
        self
    }

    /// Set the maximum number of solver iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the tolerance for the stopping criterion.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Enable or disable the shrinking heuristic.
    pub fn with_shrinking(mut self, shrinking: bool) -> Self {
        self.shrinking = shrinking;
        self
    }

    /// Set the kernel cache size in MB.
    pub fn with_cache_size(mut self, cache_size: f64) -> Self {
        self.cache_size = cache_size;
        self
    }

    /// Validate parameters before fitting.
    fn validate(&self) -> Result<()> {
        if self.nu <= 0.0 || self.nu > 1.0 {
            return Err(RustMlError::InvalidParameter("nu must be in (0, 1]".into()));
        }
        if self.max_iter == 0 {
            return Err(RustMlError::InvalidParameter(
                "max_iter must be at least 1".into(),
            ));
        }
        if self.tol <= 0.0 {
            return Err(RustMlError::InvalidParameter("tol must be positive".into()));
        }
        self.kernel.validate(false)?;
        Ok(())
    }
}

impl Default for OneClassSvm {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted One-Class SVM model.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedOneClassSvm<F: Float> {
    /// Support vectors (subset of training data with non-zero alpha).
    support_vectors: Array2<F>,
    /// Dual coefficients for each support vector.
    dual_coefs: Array1<F>,
    /// Offset term (rho).
    rho: F,
    /// Kernel used.
    kernel: SvmKernel,
    /// Number of features expected at prediction time.
    n_features: usize,
}

impl<F: Float> FittedOneClassSvm<F> {
    /// Returns the support vectors.
    pub fn support_vectors(&self) -> &Array2<F> {
        &self.support_vectors
    }

    /// Returns the number of support vectors.
    pub fn n_support(&self) -> usize {
        self.support_vectors.nrows()
    }

    /// Returns the offset rho.
    pub fn rho(&self) -> F {
        self.rho
    }

    /// Compute the raw decision function values for each sample.
    ///
    /// `f(x) = sum_i alpha_i K(x_i, x) - rho`
    ///
    /// Positive values indicate inliers, negative values indicate outliers.
    pub fn decision_function<'a>(&self, x: impl AsArray<'a, F, Ix2>) -> Result<Array1<F>> {
        let x: ArrayView2<F> = x.into();
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        if x.is_empty() {
            return Err(RustMlError::EmptyInput(
                "prediction input must not be empty".into(),
            ));
        }

        let scores: Vec<F> = x
            .rows()
            .into_iter()
            .map(|sample| {
                let mut result = F::zero();
                for (sv_idx, sv) in self.support_vectors.rows().into_iter().enumerate() {
                    result += self.dual_coefs[sv_idx] * self.kernel.compute(&sv, &sample);
                }
                result - self.rho
            })
            .collect();

        Ok(Array1::from_vec(scores))
    }

    /// Unshifted scores `sum_i alpha_i K(x_i, x)` (sklearn's `score_samples`
    /// = `decision_function + offset_`).
    pub fn score_samples<'a>(&self, x: impl AsArray<'a, F, Ix2>) -> Result<Array1<F>> {
        let x: ArrayView2<F> = x.into();
        Ok(self.decision_function(x.view())? + self.rho)
    }

    /// sklearn's `offset_` (= rho): `decision_function = score_samples - offset`.
    pub fn offset(&self) -> F {
        self.rho
    }
}

impl<F: Float> Predict<F> for FittedOneClassSvm<F> {
    fn predict_view(&self, x: ArrayView2<F>) -> Result<Array1<F>> {
        let scores = self.decision_function(x)?;
        let one = F::one();
        Ok(scores.mapv(|s| if s >= F::zero() { one } else { -one }))
    }
}

impl<F: Float> FitUnsupervised<F> for OneClassSvm {
    type Fitted = FittedOneClassSvm<F>;

    fn fit_view(&self, x: ArrayView2<F>) -> Result<Self::Fitted> {
        self.validate()?;

        if x.is_empty() {
            return Err(RustMlError::EmptyInput(
                "training data must not be empty".into(),
            ));
        }

        let kernel = self.kernel.resolve(x);
        let x64: Array2<f64> = x.mapv(|v| v.to_f64().unwrap());
        let (alpha, rho) = crate::solver::solve_one_class(
            crate::solver::KernelSource::Data {
                x: &x64,
                kernel: &kernel,
            },
            self.nu,
            &crate::solver::SolverParams {
                eps: self.tol,
                shrinking: self.shrinking,
                max_iter: Some(self.max_iter),
                nu: false,
            },
            self.cache_size,
        );
        // libsvm's dual (Σα = ν l), as sklearn's `dual_coef_` / `offset_`.
        let alphas: Vec<F> = alpha.iter().map(|&a| F::from_f64(a).unwrap()).collect();
        let rho = F::from_f64(rho).unwrap();

        let sv_indices: Vec<usize> = (0..x.nrows()).filter(|&i| alphas[i] > F::zero()).collect();

        if sv_indices.is_empty() {
            // Fallback: use all training points.
            let dual_coefs = Array1::from_vec(alphas);
            return Ok(FittedOneClassSvm {
                support_vectors: x.to_owned(),
                dual_coefs,
                rho,
                kernel: kernel.clone(),
                n_features: x.ncols(),
            });
        }

        let n_sv = sv_indices.len();
        let n_features = x.ncols();
        let mut support_vectors = Array2::zeros((n_sv, n_features));
        let mut dual_coefs = Array1::zeros(n_sv);

        for (sv_pos, &orig_idx) in sv_indices.iter().enumerate() {
            support_vectors.row_mut(sv_pos).assign(&x.row(orig_idx));
            dual_coefs[sv_pos] = alphas[orig_idx];
        }

        Ok(FittedOneClassSvm {
            support_vectors,
            dual_coefs,
            rho,
            kernel: kernel.clone(),
            n_features,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    /// 20 × 2 fixed dataset shared by the sklearn-parity tests (sklearn 1.9.1).
    fn parity_data() -> (Array2<f64>, Array1<f64>) {
        let x = array![
            [-1.486, -0.003],
            [0.406, -1.885],
            [-1.408, 1.713],
            [-1.718, -1.481],
            [1.793, 0.488],
            [-0.524, 0.046],
            [0.651, -0.899],
            [-1.448, 1.152],
            [0.681, 0.05],
            [1.267, 0.196],
            [1.924, -1.182],
            [0.215, -0.066],
            [-0.587, 0.366],
            [-1.059, 1.209],
            [1.469, -1.485],
            [-0.132, -0.891],
            [-1.668, 1.584],
            [-0.28, -1.409],
            [0.693, -1.191],
            [1.606, -1.131]
        ];
        let y = array![
            -0.963, -0.523, -0.02, -1.858, 1.153, -0.561, -0.017, -0.404, 0.707, 0.978, 0.486,
            0.263, -0.308, -0.227, 0.348, -0.71, -0.142, -0.921, -0.133, 0.469
        ];
        (x, y)
    }

    #[test]
    fn test_dual_is_feasible() {
        let (x, _) = parity_data();
        let kernel = SvmKernel::rbf(0.5);
        let n = x.nrows() as f64;
        for &(nu, max_iter) in &[(0.3, 50usize), (0.3, 20_000), (0.9, 500), (0.05, 500)] {
            let (alpha, _rho) = crate::solver::solve_one_class(
                crate::solver::KernelSource::Data {
                    x: &x,
                    kernel: &kernel,
                },
                nu,
                &crate::solver::SolverParams {
                    eps: 1e-12,
                    shrinking: true,
                    max_iter: Some(max_iter),
                    nu: false,
                },
                200.0,
            );
            // libsvm scaling: 0 <= α <= 1, Σα = ν n (kept by every SMO step).
            let sum: f64 = alpha.iter().sum();
            assert!((sum - nu * n).abs() < 1e-9, "Σα = {sum} (nu={nu})");
            assert!(alpha.iter().all(|&a| (0.0..=1.0).contains(&a)));
        }
    }

    #[test]
    fn test_rbf_matches_sklearn() {
        // sklearn OneClassSVM(kernel='rbf', gamma=0.5, nu=0.3, tol=1e-10):
        // decision_function and offset_ (libsvm scaling, Σα = νn).
        let sk_dec = [
            6.9503616195731865e-09,
            -9.859661975752942e-10,
            2.1139432426764415e-09,
            -0.0494922297759226,
            -2.6398514485492797e-09,
            0.01721861759923482,
            0.1708220770115747,
            0.09670733698392686,
            0.04353703818254151,
            0.12360942560747157,
            -1.7019230469372815e-09,
            -9.01498431460368e-10,
            3.08246628222264e-08,
            0.06233383172152385,
            0.05883642903452113,
            0.0677535427239575,
            2.6196671498723845e-09,
            0.039233360875097434,
            0.1780002101153637,
            0.13576351210873505,
        ];
        let sk_rho = 1.3410398139399482;
        let scale = 1.0;
        let (x, _) = parity_data();
        let model = OneClassSvm::new()
            .with_nu(0.3)
            .with_kernel(SvmKernel::rbf(0.5))
            .with_max_iter(100_000)
            .with_tol(1e-12);
        let fitted: FittedOneClassSvm<f64> = model.fit(&x).unwrap();
        assert!(
            (fitted.rho() - sk_rho * scale).abs() < 1e-5,
            "rho = {}, sklearn = {}",
            fitted.rho(),
            sk_rho * scale
        );
        let dec = fitted.decision_function(x.view()).unwrap();
        for (i, (d, e)) in dec.iter().zip(sk_dec.iter()).enumerate() {
            assert!(
                (d - e * scale).abs() < 1e-5,
                "dec[{i}] = {d}, sklearn = {}",
                e * scale
            );
        }
    }

    #[test]
    fn test_basic_inlier_detection() {
        // Train on a tight cluster; points in the cluster should be inliers.
        let x = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [-0.1, 0.0],
            [0.0, -0.1],
            [0.05, 0.05],
            [-0.05, -0.05],
        ];

        let ocsvm = OneClassSvm::new()
            .with_kernel(SvmKernel::rbf(10.0))
            .with_nu(0.1)
            .with_max_iter(5000);
        let fitted: FittedOneClassSvm<f64> = FitUnsupervised::fit(&ocsvm, &x).unwrap();

        let preds = fitted.predict(&x).unwrap();
        // Most training points should be classified as inliers (+1).
        let n_inliers = preds.iter().filter(|&&p| p > 0.0).count();
        assert!(
            n_inliers >= 6,
            "expected at least 6 inliers among 8 training points, got {}",
            n_inliers
        );
    }

    #[test]
    fn test_outlier_detection() {
        // Train on a cluster near the origin; far-away points should be outliers.
        let x_train = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [-0.1, 0.0],
            [0.0, -0.1],
            [0.05, 0.05],
            [-0.05, -0.05],
        ];

        let ocsvm = OneClassSvm::new()
            .with_kernel(SvmKernel::rbf(10.0))
            .with_nu(0.1)
            .with_max_iter(5000);
        let fitted: FittedOneClassSvm<f64> = FitUnsupervised::fit(&ocsvm, &x_train).unwrap();

        let x_outliers = array![[10.0, 10.0], [-10.0, -10.0], [5.0, -5.0]];
        let preds = fitted.predict(&x_outliers).unwrap();
        // All far-away points should be outliers (-1).
        for &p in preds.iter() {
            assert_abs_diff_eq!(p, -1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_nu_effect() {
        // Higher nu means more support vectors and a tighter boundary,
        // so more training points may be classified as outliers.
        let x = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
            [0.3, 0.7],
            [0.7, 0.3],
            [0.2, 0.2],
            [0.8, 0.8],
            [0.5, 0.1],
        ];

        let ocsvm_low = OneClassSvm::new()
            .with_kernel(SvmKernel::rbf(1.0))
            .with_nu(0.1)
            .with_max_iter(5000);
        let fitted_low: FittedOneClassSvm<f64> = FitUnsupervised::fit(&ocsvm_low, &x).unwrap();

        let ocsvm_high = OneClassSvm::new()
            .with_kernel(SvmKernel::rbf(1.0))
            .with_nu(0.9)
            .with_max_iter(5000);
        let fitted_high: FittedOneClassSvm<f64> = FitUnsupervised::fit(&ocsvm_high, &x).unwrap();

        let preds_low = fitted_low.predict(&x).unwrap();
        let preds_high = fitted_high.predict(&x).unwrap();

        let inliers_low = preds_low.iter().filter(|&&p| p > 0.0).count();
        let inliers_high = preds_high.iter().filter(|&&p| p > 0.0).count();

        // Low nu should classify at least as many inliers as high nu.
        assert!(
            inliers_low >= inliers_high,
            "low nu ({}) should have >= inliers than high nu ({})",
            inliers_low,
            inliers_high
        );
    }

    #[test]
    fn test_kernel_types() {
        let x = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [-0.1, 0.0],
            [0.0, -0.1],
        ];

        // Linear kernel
        let ocsvm = OneClassSvm::new()
            .with_kernel(SvmKernel::Linear)
            .with_nu(0.5)
            .with_max_iter(5000);
        let fitted: FittedOneClassSvm<f64> = FitUnsupervised::fit(&ocsvm, &x).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), x.nrows());

        // Polynomial kernel
        let ocsvm = OneClassSvm::new()
            .with_kernel(SvmKernel::poly(2, 1.0, 1.0))
            .with_nu(0.5)
            .with_max_iter(5000);
        let fitted: FittedOneClassSvm<f64> = FitUnsupervised::fit(&ocsvm, &x).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), x.nrows());

        // RBF kernel
        let ocsvm = OneClassSvm::new()
            .with_kernel(SvmKernel::rbf(1.0))
            .with_nu(0.5)
            .with_max_iter(5000);
        let fitted: FittedOneClassSvm<f64> = FitUnsupervised::fit(&ocsvm, &x).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), x.nrows());
    }

    #[test]
    fn test_predict_shape() {
        let x_train = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],];

        let ocsvm = OneClassSvm::new()
            .with_kernel(SvmKernel::rbf(1.0))
            .with_nu(0.5)
            .with_max_iter(5000);
        let fitted: FittedOneClassSvm<f64> = FitUnsupervised::fit(&ocsvm, &x_train).unwrap();

        let x_test = array![[0.5, 0.5], [2.0, 2.0], [-1.0, -1.0],];
        let preds = fitted.predict(&x_test).unwrap();
        assert_eq!(preds.len(), 3);

        // All predictions should be +1 or -1.
        for &p in preds.iter() {
            assert!(
                p == 1.0 || p == -1.0,
                "prediction should be +1 or -1, got {}",
                p
            );
        }
    }

    #[test]
    fn test_empty_input_errors() {
        let ocsvm = OneClassSvm::new();

        // Empty training data.
        let x_empty = Array2::<f64>::zeros((0, 2));
        let result: Result<FittedOneClassSvm<f64>> = FitUnsupervised::fit(&ocsvm, &x_empty);
        assert!(result.is_err());

        // Train on valid data, then predict on empty.
        let x_train = array![[0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [0.2, 0.8]];
        let fitted: FittedOneClassSvm<f64> = FitUnsupervised::fit(&ocsvm, &x_train).unwrap();

        let x_pred_empty = Array2::<f64>::zeros((0, 2));
        assert!(fitted.predict(&x_pred_empty).is_err());
    }

    #[test]
    fn test_decision_function_sign() {
        // Train on a cluster near origin; inliers should have positive
        // decision function values, outliers negative.
        let x_train = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [-0.1, 0.0],
            [0.0, -0.1],
            [0.05, 0.05],
            [-0.05, -0.05],
        ];

        let ocsvm = OneClassSvm::new()
            .with_kernel(SvmKernel::rbf(10.0))
            .with_nu(0.1)
            .with_max_iter(5000);
        let fitted: FittedOneClassSvm<f64> = FitUnsupervised::fit(&ocsvm, &x_train).unwrap();

        // Decision function for far-away outliers should be negative.
        let x_outliers = array![[10.0, 10.0], [-10.0, -10.0]];
        let scores = fitted.decision_function(x_outliers.view()).unwrap();
        for &s in scores.iter() {
            assert!(
                s < 0.0,
                "outlier should have negative decision function, got {}",
                s
            );
        }

        // score_samples = decision_function + offset (sklearn).
        let scores2 = fitted.score_samples(&x_outliers).unwrap();
        for (s1, s2) in scores.iter().zip(scores2.iter()) {
            assert_abs_diff_eq!(*s1 + fitted.offset(), *s2, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_f32_support() {
        let x: Array2<f32> = array![
            [0.0f32, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [-0.1, 0.0],
            [0.0, -0.1],
        ];

        let ocsvm = OneClassSvm::new()
            .with_kernel(SvmKernel::rbf(1.0))
            .with_nu(0.5)
            .with_max_iter(5000);
        let fitted: FittedOneClassSvm<f32> = FitUnsupervised::fit(&ocsvm, &x).unwrap();

        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), x.nrows());
        for &p in preds.iter() {
            assert!(p.is_finite());
            assert!(p == 1.0f32 || p == -1.0f32);
        }
    }

    #[test]
    fn test_invalid_parameters() {
        let x = array![[0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [0.2, 0.8]];

        // nu out of range
        let ocsvm = OneClassSvm::new().with_nu(0.0);
        assert!(FitUnsupervised::<f64>::fit(&ocsvm, &x).is_err());

        let ocsvm = OneClassSvm::new().with_nu(1.5);
        assert!(FitUnsupervised::<f64>::fit(&ocsvm, &x).is_err());

        // Invalid tol
        let ocsvm = OneClassSvm::new().with_tol(-1.0);
        assert!(FitUnsupervised::<f64>::fit(&ocsvm, &x).is_err());

        // Invalid max_iter
        let ocsvm = OneClassSvm::new().with_max_iter(0);
        assert!(FitUnsupervised::<f64>::fit(&ocsvm, &x).is_err());
    }

    #[test]
    fn test_feature_mismatch_predict() {
        let x_train = array![[0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [0.2, 0.8]];
        let ocsvm = OneClassSvm::new()
            .with_kernel(SvmKernel::Linear)
            .with_nu(0.5)
            .with_max_iter(5000);
        let fitted: FittedOneClassSvm<f64> = FitUnsupervised::fit(&ocsvm, &x_train).unwrap();

        // Wrong number of features.
        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_builder_pattern() {
        let ocsvm = OneClassSvm::new()
            .with_nu(0.3)
            .with_kernel(SvmKernel::Linear)
            .with_max_iter(500)
            .with_tol(1e-3);
        assert_eq!(ocsvm.nu, 0.3);
        assert_eq!(ocsvm.max_iter, 500);
        assert_eq!(ocsvm.tol, 1e-3);
    }
}
