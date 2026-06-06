//! One-Class SVM for unsupervised outlier / novelty detection.
//!
//! Learns a decision boundary in kernel space that encloses the training data.
//! New points are classified as inliers (+1) or outliers (-1).
//!
//! The dual optimisation problem:
//!   minimise  0.5 alpha^T K alpha
//!   subject to  0 <= alpha_i <= 1/(nu*n),  sum(alpha_i) = 1
//!
//! Decision function: f(x) = sum_i alpha_i K(x_i, x) - rho
//! Prediction: sign(f(x)), +1 = inlier, -1 = outlier

use ndarray::{Array1, Array2};
use rustml_core::{FitUnsupervised, Float, Predict, Result, RustMlError};

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
    /// Maximum number of solver iterations.
    pub max_iter: usize,
    /// Tolerance for the stopping criterion.
    pub tol: f64,
}

impl OneClassSvm {
    /// Create a new `OneClassSvm` with default parameters.
    pub fn new() -> Self {
        Self {
            nu: 0.5,
            kernel: SvmKernel::Rbf { gamma: 1.0 },
            max_iter: 1000,
            tol: 1e-4,
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
        match &self.kernel {
            SvmKernel::Rbf { gamma } if *gamma <= 0.0 => {
                return Err(RustMlError::InvalidParameter(
                    "gamma must be positive for RBF kernel".into(),
                ));
            }
            SvmKernel::Polynomial { degree, .. } if *degree == 0 => {
                return Err(RustMlError::InvalidParameter(
                    "degree must be at least 1 for polynomial kernel".into(),
                ));
            }
            _ => {}
        }
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
    pub fn decision_function(&self, x: &Array2<F>) -> Result<Array1<F>> {
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

    /// Returns the shifted decision function values (same as `decision_function`).
    ///
    /// Provided for API parity with scikit-learn's `score_samples`.
    pub fn score_samples(&self, x: &Array2<F>) -> Result<Array1<F>> {
        self.decision_function(x)
    }
}

impl<F: Float> Predict<F> for FittedOneClassSvm<F> {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        let scores = self.decision_function(x)?;
        let one = F::one();
        Ok(scores.mapv(|s| if s >= F::zero() { one } else { -one }))
    }
}

/// Precompute the symmetric kernel matrix.
fn precompute_kernel_matrix<F: Float>(x: &Array2<F>, kernel: &SvmKernel) -> Array2<F> {
    let n = x.nrows();
    let mut k = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in i..n {
            let val = kernel.compute(&x.row(i), &x.row(j));
            k[[i, j]] = val;
            k[[j, i]] = val;
        }
    }
    k
}

#[inline]
fn clip<F: Float>(value: F, lo: F, hi: F) -> F {
    if value > hi {
        hi
    } else if value < lo {
        lo
    } else {
        value
    }
}

/// Project a vector onto the probability simplex (sum = 1, each element >= 0)
/// using the efficient algorithm of Duchi et al. (2008).
fn project_simplex<F: Float>(v: &mut [F], upper: F) {
    let n = v.len();
    if n == 0 {
        return;
    }
    let one = F::one();

    // Scale down so box constraint is [0, 1], project, then scale back.
    // v_i' = v_i * upper, project onto simplex sum = 1 with [0, 1], then
    // v_i = v_i' / upper. But it is simpler to project with sum = 1 and
    // box [0, upper] directly.

    // Sort in descending order.
    let mut sorted: Vec<F> = v.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    // Find the threshold tau such that the projected values sum to 1.
    let mut cumsum = F::zero();
    let mut tau = F::zero();
    let mut found = false;
    for (j, &s_j) in sorted.iter().enumerate() {
        cumsum += s_j;
        let t = (cumsum - one) / F::from_usize(j + 1).unwrap();
        if s_j - t > F::zero() {
            tau = t;
        } else {
            found = true;
            break;
        }
    }
    let _ = found; // avoid unused warning

    // Project: v_i = clip(v_i - tau, 0, upper)
    for val in v.iter_mut() {
        *val = clip(*val - tau, F::zero(), upper);
    }
}

/// FISTA solver for the One-Class SVM dual.
///
/// Minimises 0.5 alpha^T K alpha
/// subject to  0 <= alpha_i <= 1/(nu*n),  sum(alpha_i) = 1.
///
/// Uses accelerated proximal gradient descent with Nesterov momentum,
/// projecting onto the constrained set (intersection of box and simplex)
/// at each step.
fn solve_one_class_svm<F: Float>(
    x: &Array2<F>,
    kernel: &SvmKernel,
    nu: F,
    max_iter: usize,
    tol: F,
) -> (Vec<F>, F) {
    let n = x.nrows();
    let zero = F::zero();
    let one = F::one();
    let two = one + one;
    let four = two + two;
    let n_f = F::from_usize(n).unwrap();

    let k_matrix = precompute_kernel_matrix(x, kernel);

    // Upper bound on each alpha_i.
    let alpha_upper = one / (nu * n_f);

    // Lipschitz constant L = max row-sum of |K|.
    let mut lipschitz = zero;
    for i in 0..n {
        let mut row_sum = zero;
        for j in 0..n {
            row_sum = row_sum + k_matrix[[i, j]].abs();
        }
        if row_sum > lipschitz {
            lipschitz = row_sum;
        }
    }
    if lipschitz < F::from_f64(1e-12).unwrap() {
        lipschitz = one;
    }
    let step = one / lipschitz;

    // Initialise alpha uniformly on the simplex: alpha_i = 1/n.
    let init_val = one / n_f;
    // Clamp initial value to the box constraint.
    let init_val = if init_val > alpha_upper {
        alpha_upper
    } else {
        init_val
    };
    let mut alpha = vec![init_val; n];
    // Re-project to ensure sum = 1 after clamping.
    project_simplex(&mut alpha, alpha_upper);

    let mut alpha_prev = alpha.clone();
    let mut v = alpha.clone(); // momentum point
    let mut t_k = one;

    for _iter in 0..max_iter {
        // Compute gradient: grad_i = sum_j K_ij * v_j
        let mut grad = vec![zero; n];
        for i in 0..n {
            let mut s = zero;
            for j in 0..n {
                s = s + k_matrix[[i, j]] * v[j];
            }
            grad[i] = s;
        }

        // Gradient step: z = v - step * grad
        for i in 0..n {
            alpha_prev[i] = alpha[i];
            alpha[i] = v[i] - step * grad[i];
        }

        // Project onto simplex intersected with box [0, alpha_upper].
        project_simplex(&mut alpha, alpha_upper);

        // Check convergence.
        let mut max_change = zero;
        for i in 0..n {
            let change = (alpha[i] - alpha_prev[i]).abs();
            if change > max_change {
                max_change = change;
            }
        }

        if max_change < tol {
            break;
        }

        // Nesterov momentum.
        let t_new = (one + (one + four * t_k * t_k).sqrt()) / two;
        let momentum = (t_k - one) / t_new;
        t_k = t_new;

        for i in 0..n {
            v[i] = alpha[i] + momentum * (alpha[i] - alpha_prev[i]);
        }
    }

    // Compute rho from free support vectors (0 < alpha_i < alpha_upper).
    let margin = F::from_f64(1e-8).unwrap();
    let mut rho_sum = zero;
    let mut rho_count = 0usize;

    for i in 0..n {
        if alpha[i] > margin && alpha[i] < alpha_upper - margin {
            // For a free SV: rho = sum_j alpha_j K(x_j, x_i)
            let mut score = zero;
            for j in 0..n {
                score = score + alpha[j] * k_matrix[[j, i]];
            }
            rho_sum = rho_sum + score;
            rho_count += 1;
        }
    }

    let rho = if rho_count > 0 {
        rho_sum / F::from_usize(rho_count).unwrap()
    } else {
        // Fallback: use any support vector (alpha > 0).
        let mut best_rho = zero;
        let mut found = false;
        for i in 0..n {
            if alpha[i] > margin {
                let mut score = zero;
                for j in 0..n {
                    score = score + alpha[j] * k_matrix[[j, i]];
                }
                best_rho = score;
                found = true;
                break;
            }
        }
        if !found {
            // Extreme fallback: use first sample.
            let mut score = zero;
            for j in 0..n {
                score = score + alpha[j] * k_matrix[[j, 0]];
            }
            best_rho = score;
        }
        best_rho
    };

    (alpha, rho)
}

impl<F: Float> FitUnsupervised<F> for OneClassSvm {
    type Fitted = FittedOneClassSvm<F>;

    fn fit(&self, x: &Array2<F>) -> Result<Self::Fitted> {
        self.validate()?;

        if x.is_empty() {
            return Err(RustMlError::EmptyInput(
                "training data must not be empty".into(),
            ));
        }

        let nu = F::from_f64(self.nu).unwrap();
        let tol = F::from_f64(self.tol).unwrap();

        let (alphas, rho) = solve_one_class_svm(x, &self.kernel, nu, self.max_iter, tol);

        // Extract support vectors (non-zero alphas).
        let sv_threshold = F::from_f64(1e-8).unwrap();
        let sv_indices: Vec<usize> = (0..x.nrows())
            .filter(|&i| alphas[i] > sv_threshold)
            .collect();

        if sv_indices.is_empty() {
            // Fallback: use all training points.
            let dual_coefs = Array1::from_vec(alphas);
            return Ok(FittedOneClassSvm {
                support_vectors: x.to_owned(),
                dual_coefs,
                rho,
                kernel: self.kernel.clone(),
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
            kernel: self.kernel.clone(),
            n_features,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

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
            .with_kernel(SvmKernel::Rbf { gamma: 10.0 })
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
            .with_kernel(SvmKernel::Rbf { gamma: 10.0 })
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
            .with_kernel(SvmKernel::Rbf { gamma: 1.0 })
            .with_nu(0.1)
            .with_max_iter(5000);
        let fitted_low: FittedOneClassSvm<f64> = FitUnsupervised::fit(&ocsvm_low, &x).unwrap();

        let ocsvm_high = OneClassSvm::new()
            .with_kernel(SvmKernel::Rbf { gamma: 1.0 })
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
            .with_kernel(SvmKernel::Polynomial {
                degree: 2,
                coef0: 1.0,
            })
            .with_nu(0.5)
            .with_max_iter(5000);
        let fitted: FittedOneClassSvm<f64> = FitUnsupervised::fit(&ocsvm, &x).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), x.nrows());

        // RBF kernel
        let ocsvm = OneClassSvm::new()
            .with_kernel(SvmKernel::Rbf { gamma: 1.0 })
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
            .with_kernel(SvmKernel::Rbf { gamma: 1.0 })
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
            .with_kernel(SvmKernel::Rbf { gamma: 10.0 })
            .with_nu(0.1)
            .with_max_iter(5000);
        let fitted: FittedOneClassSvm<f64> = FitUnsupervised::fit(&ocsvm, &x_train).unwrap();

        // Decision function for far-away outliers should be negative.
        let x_outliers = array![[10.0, 10.0], [-10.0, -10.0]];
        let scores = fitted.decision_function(&x_outliers).unwrap();
        for &s in scores.iter() {
            assert!(
                s < 0.0,
                "outlier should have negative decision function, got {}",
                s
            );
        }

        // score_samples should return the same values.
        let scores2 = fitted.score_samples(&x_outliers).unwrap();
        for (s1, s2) in scores.iter().zip(scores2.iter()) {
            assert_abs_diff_eq!(*s1, *s2, epsilon = 1e-10);
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
            .with_kernel(SvmKernel::Rbf { gamma: 1.0 })
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
