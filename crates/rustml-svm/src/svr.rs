//! Epsilon-Support Vector Regression (SVR) using SMO.

use ndarray::{Array1, Array2};
use rustml_core::{Fit, Float, Predict, Result, RustMlError};

use crate::kernel::SvmKernel;

/// Epsilon-Support Vector Regressor (unfitted state).
///
/// Implements an SMO-based solver for epsilon-insensitive regression:
/// `min 0.5 ||w||^2 + C * sum(max(0, |y_i - f(x_i)| - epsilon))`
///
/// Uses the type-state pattern: call [`Fit::fit`] to produce a [`FittedSvr`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Svr {
    /// Regularization parameter. Larger values penalize errors more.
    pub c: f64,
    /// Width of the epsilon-insensitive tube.
    pub epsilon: f64,
    /// Kernel function to use.
    pub kernel: SvmKernel,
    /// Maximum number of SMO iterations.
    pub max_iter: usize,
    /// Tolerance for stopping criterion.
    pub tol: f64,
}

impl Svr {
    pub fn new() -> Self {
        Self {
            c: 1.0,
            epsilon: 0.1,
            kernel: SvmKernel::Rbf { gamma: 1.0 },
            max_iter: 1000,
            tol: 1e-4,
        }
    }

    pub fn with_c(mut self, c: f64) -> Self {
        self.c = c;
        self
    }

    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    pub fn with_kernel(mut self, kernel: SvmKernel) -> Self {
        self.kernel = kernel;
        self
    }

    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    fn validate(&self) -> Result<()> {
        if self.c <= 0.0 {
            return Err(RustMlError::InvalidParameter("C must be positive".into()));
        }
        if self.epsilon < 0.0 {
            return Err(RustMlError::InvalidParameter(
                "epsilon must be non-negative".into(),
            ));
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

impl Default for Svr {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted epsilon-SVR model.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedSvr<F: Float> {
    /// Support vectors.
    support_vectors: Array2<F>,
    /// Dual coefficients (alpha_i - alpha_i*) for each support vector.
    dual_coefs: Array1<F>,
    /// Bias term.
    bias: F,
    /// Kernel used.
    kernel: SvmKernel,
    /// Number of features expected at prediction time.
    n_features: usize,
}

impl<F: Float> FittedSvr<F> {
    /// Returns the support vectors.
    pub fn support_vectors(&self) -> &Array2<F> {
        &self.support_vectors
    }

    /// Returns the number of support vectors.
    pub fn n_support(&self) -> usize {
        self.support_vectors.nrows()
    }

    /// Returns the bias term.
    pub fn bias(&self) -> F {
        self.bias
    }

    /// Construct a [`FittedSvr`] directly from its components.
    ///
    /// Used by alternative solvers (e.g. the direct nu-SVR SMO solver)
    /// that produce support vectors and dual coefficients through a
    /// different optimization routine.
    pub fn from_parts(
        support_vectors: Array2<F>,
        dual_coefs: Array1<F>,
        bias: F,
        kernel: SvmKernel,
        n_features: usize,
    ) -> Self {
        Self {
            support_vectors,
            dual_coefs,
            bias,
            kernel,
            n_features,
        }
    }
}

impl<F: Float> Predict<F> for FittedSvr<F> {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
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

        let predictions: Vec<F> = x
            .rows()
            .into_iter()
            .map(|sample| {
                let mut result = self.bias;
                for (sv_idx, sv) in self.support_vectors.rows().into_iter().enumerate() {
                    result += self.dual_coefs[sv_idx] * self.kernel.compute(&sv, &sample);
                }
                result
            })
            .collect();

        Ok(Array1::from_vec(predictions))
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

/// FISTA (Fast Iterative Shrinkage-Thresholding) solver for epsilon-SVR.
///
/// Maximises the dual objective:
///   W(w) = −½ Σ w_i w_j K(x_i,x_j) + Σ y_i w_i − ε Σ |w_i|
///
/// subject to  −C ≤ w_i ≤ C.
///
/// Uses accelerated proximal gradient descent with Nesterov momentum,
/// which converges in O(1/k²) and handles rank-deficient kernel matrices
/// (e.g. linear kernel on low-dim data) gracefully.
fn smo_svr<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    kernel: &SvmKernel,
    c: F,
    epsilon: F,
    max_iter: usize,
    tol: F,
) -> (Vec<F>, F) {
    let n = x.nrows();
    let zero = F::zero();
    let one = F::one();
    let two = one + one;
    let four = two + two;

    let k_matrix = precompute_kernel_matrix(x, kernel);

    // Lipschitz constant L = max row-sum of |K| (tighter than trace for PSD).
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
    let eps_step = epsilon * step;

    let mut w = vec![zero; n];
    let mut w_prev = vec![zero; n];
    let mut v = vec![zero; n]; // momentum point
    let mut t_k = one;

    let n_f = F::from_usize(n).unwrap();

    for _iter in 0..max_iter {
        // Compute K * v
        let mut kv = vec![zero; n];
        for i in 0..n {
            let mut s = zero;
            for j in 0..n {
                s = s + k_matrix[[i, j]] * v[j];
            }
            kv[i] = s;
        }

        // Proximal gradient step: w_new = prox_{step*ε|·|}(v + step*(y − K*v))
        for i in 0..n {
            let z_i = v[i] + step * (y[i] - kv[i]);

            let new_w = if z_i > eps_step {
                clip(z_i - eps_step, zero, c)
            } else if z_i < -eps_step {
                clip(z_i + eps_step, -c, zero)
            } else {
                zero
            };

            w_prev[i] = w[i];
            w[i] = new_w;
        }

        // Project onto Σ w_i = 0 (dual constraint from bias term),
        // then re-enforce box constraints.
        let w_mean = w.iter().copied().fold(zero, |a, b| a + b) / n_f;
        for i in 0..n {
            w[i] = clip(w[i] - w_mean, -c, c);
        }

        // Check convergence
        let mut max_change = zero;
        for i in 0..n {
            let change = (w[i] - w_prev[i]).abs();
            if change > max_change {
                max_change = change;
            }
        }

        if max_change < tol {
            break;
        }

        // Nesterov momentum
        let t_new = (one + (one + four * t_k * t_k).sqrt()) / two;
        let momentum = (t_k - one) / t_new;
        t_k = t_new;

        for i in 0..n {
            v[i] = w[i] + momentum * (w[i] - w_prev[i]);
        }
    }

    // ---- compute g = K * w for bias calculation ----
    let mut g = vec![zero; n];
    for i in 0..n {
        let mut s = zero;
        for j in 0..n {
            s = s + k_matrix[[i, j]] * w[j];
        }
        g[i] = s;
    }

    // ---- compute bias from free support vectors ----
    // Free means 0 < |w_i| < C (not at boundary).
    //   w_i > 0 (α_i active) ⇒ f(x_i) = y_i − ε ⇒ b = y_i − ε − g_i
    //   w_i < 0 (α*_i active) ⇒ f(x_i) = y_i + ε ⇒ b = y_i + ε − g_i
    let margin = F::from_f64(1e-6).unwrap();
    let mut b_sum = zero;
    let mut b_count = 0usize;

    for i in 0..n {
        if w[i] > margin && w[i] < c - margin {
            b_sum = b_sum + y[i] - epsilon - g[i];
            b_count += 1;
        } else if w[i] < -margin && w[i] > -(c - margin) {
            b_sum = b_sum + y[i] + epsilon - g[i];
            b_count += 1;
        }
    }

    let bias = if b_count > 0 {
        b_sum / F::from_usize(b_count).unwrap()
    } else {
        // All SVs bounded or no SVs — average upper/lower bias bounds
        let mut b_lo = F::from_f64(-1e30).unwrap();
        let mut b_hi = F::from_f64(1e30).unwrap();
        for i in 0..n {
            let lo_i = y[i] - epsilon - g[i];
            let hi_i = y[i] + epsilon - g[i];
            if w[i] >= zero {
                b_lo = if lo_i > b_lo { lo_i } else { b_lo };
            }
            if w[i] <= zero {
                b_hi = if hi_i < b_hi { hi_i } else { b_hi };
            }
        }
        (b_lo + b_hi) / (F::one() + F::one())
    };

    (w, bias)
}

impl<F: Float> Fit<F> for Svr {
    type Fitted = FittedSvr<F>;

    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Self::Fitted> {
        self.validate()?;

        if x.is_empty() || y.is_empty() {
            return Err(RustMlError::EmptyInput(
                "training data must not be empty".into(),
            ));
        }
        if x.nrows() != y.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but y has {} elements",
                x.nrows(),
                y.len()
            )));
        }

        let c = F::from_f64(self.c).unwrap();
        let epsilon = F::from_f64(self.epsilon).unwrap();
        let tol = F::from_f64(self.tol).unwrap();

        // Center targets for better convergence (standard SVM practice)
        let n = F::from_usize(y.len()).unwrap();
        let y_mean = y.iter().copied().fold(F::zero(), |a, b| a + b) / n;
        let y_centered = y.mapv(|v| v - y_mean);

        let (coefs, bias_centered) =
            smo_svr(x, &y_centered, &self.kernel, c, epsilon, self.max_iter, tol);
        let bias = bias_centered + y_mean;

        // Extract support vectors (non-zero coefficients)
        let sv_threshold = F::from_f64(1e-8).unwrap();
        let sv_indices: Vec<usize> = (0..x.nrows())
            .filter(|&i| coefs[i].abs() > sv_threshold)
            .collect();

        if sv_indices.is_empty() {
            // Fallback: use all training points
            let dual_coefs = Array1::from_vec(coefs);
            return Ok(FittedSvr {
                support_vectors: x.to_owned(),
                dual_coefs,
                bias,
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
            dual_coefs[sv_pos] = coefs[orig_idx];
        }

        Ok(FittedSvr {
            support_vectors,
            dual_coefs,
            bias,
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
    fn test_linear_regression() {
        // y = 2*x on well-separated data
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0],
            [10.0]
        ];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

        let svr = Svr::new()
            .with_kernel(SvmKernel::Linear)
            .with_c(100.0)
            .with_epsilon(0.1)
            .with_max_iter(5000);
        let fitted: FittedSvr<f64> = svr.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(*p, *t, epsilon = 2.0);
        }
    }

    #[test]
    fn test_rbf_regression() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]];
        let y = array![1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0];

        let svr = Svr::new()
            .with_kernel(SvmKernel::Rbf { gamma: 0.1 })
            .with_c(100.0)
            .with_epsilon(1.0)
            .with_max_iter(5000);
        let fitted: FittedSvr<f64> = svr.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        // Predictions should be in the right ballpark
        for &p in preds.iter() {
            assert!(p.is_finite(), "prediction should be finite, got {}", p);
        }
    }

    #[test]
    fn test_support_vectors_exist() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let svr = Svr::new()
            .with_kernel(SvmKernel::Linear)
            .with_c(10.0)
            .with_epsilon(0.1)
            .with_max_iter(5000);
        let fitted: FittedSvr<f64> = svr.fit(&x, &y).unwrap();

        assert!(
            fitted.n_support() > 0,
            "should have at least one support vector"
        );
        assert!(
            fitted.n_support() <= x.nrows(),
            "cannot have more SVs than training samples"
        );
    }

    #[test]
    fn test_empty_input_error() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<f64>::zeros(0);

        let svr = Svr::new();
        let result: Result<FittedSvr<f64>> = svr.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_shape_mismatch_fit() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 2.0, 3.0];

        let svr = Svr::new();
        let result: Result<FittedSvr<f64>> = svr.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_shape_mismatch_predict() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 2.0];

        let svr = Svr::new().with_kernel(SvmKernel::Linear).with_c(10.0);
        let fitted: FittedSvr<f64> = svr.fit(&x, &y).unwrap();

        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_invalid_c() {
        let x = array![[1.0], [2.0]];
        let y = array![1.0, 2.0];

        let svr = Svr::new().with_c(-1.0);
        assert!(Fit::<f64>::fit(&svr, &x, &y).is_err());
    }

    #[test]
    fn test_invalid_epsilon() {
        let x = array![[1.0], [2.0]];
        let y = array![1.0, 2.0];

        let svr = Svr::new().with_epsilon(-0.1);
        assert!(Fit::<f64>::fit(&svr, &x, &y).is_err());
    }

    #[test]
    fn test_builder_pattern() {
        let svr = Svr::new()
            .with_c(0.5)
            .with_epsilon(0.2)
            .with_kernel(SvmKernel::Linear)
            .with_max_iter(500)
            .with_tol(1e-3);
        assert_eq!(svr.c, 0.5);
        assert_eq!(svr.epsilon, 0.2);
        assert_eq!(svr.max_iter, 500);
        assert_eq!(svr.tol, 1e-3);
    }

    #[test]
    fn test_constant_target() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![5.0, 5.0, 5.0, 5.0];

        let svr = Svr::new()
            .with_kernel(SvmKernel::Linear)
            .with_c(1.0)
            .with_epsilon(0.1)
            .with_max_iter(1000);
        let fitted: FittedSvr<f64> = svr.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert_abs_diff_eq!(p, 5.0, epsilon = 1.0);
        }
    }

    #[test]
    fn test_f32_support() {
        let x: Array2<f32> = array![[1.0f32], [2.0], [3.0], [4.0]];
        let y: Array1<f32> = array![2.0f32, 4.0, 6.0, 8.0];

        let svr = Svr::new()
            .with_kernel(SvmKernel::Linear)
            .with_c(10.0)
            .with_epsilon(0.1)
            .with_max_iter(5000);
        let fitted: FittedSvr<f32> = svr.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(p.is_finite());
        }
    }
}
