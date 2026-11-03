//! Epsilon-Support Vector Regression (SVR) using an accelerated projected
//! (proximal) gradient dual solver (FISTA).

use anofox_ml_core::{Fit, Float, Predict, Result, RustMlError};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::kernel::SvmKernel;

/// Epsilon-Support Vector Regressor (unfitted state).
///
/// Solves the epsilon-insensitive dual with FISTA (accelerated proximal
/// gradient) and an exact projection onto the dual feasible set:
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
    /// Tolerance of the stopping criterion (libsvm / sklearn default 1e-3).
    pub tol: f64,
    /// Use libsvm's shrinking heuristic (default true).
    #[serde(default = "crate::solver::default_true")]
    pub shrinking: bool,
    /// Kernel cache size in MB (default 200).
    #[serde(default = "crate::solver::default_cache_size")]
    pub cache_size: f64,
}

impl Svr {
    pub fn new() -> Self {
        Self {
            c: 1.0,
            epsilon: 0.1,
            kernel: SvmKernel::default(),
            max_iter: 10_000_000,
            tol: 1e-3,
            shrinking: true,
            cache_size: 200.0,
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

    pub fn with_shrinking(mut self, shrinking: bool) -> Self {
        self.shrinking = shrinking;
        self
    }

    pub fn with_cache_size(mut self, cache_size: f64) -> Self {
        self.cache_size = cache_size;
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
        self.kernel.validate(false)?;
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
    fn predict_view(&self, x: ArrayView2<F>) -> Result<Array1<F>> {
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

impl<F: Float> Fit<F> for Svr {
    type Fitted = FittedSvr<F>;

    fn fit_view(&self, x: ArrayView2<F>, y: ArrayView1<F>) -> Result<Self::Fitted> {
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

        let kernel = self.kernel.resolve(x);
        let x64: Array2<f64> = x.mapv(|v| v.to_f64().unwrap());
        let y64: Vec<f64> = y.iter().map(|v| v.to_f64().unwrap()).collect();
        let (coefs, bias) = crate::solver::solve_epsilon_svr(
            crate::solver::KernelSource::Data {
                x: &x64,
                kernel: &kernel,
            },
            &y64,
            self.c,
            self.epsilon,
            &crate::solver::SolverParams {
                eps: self.tol,
                shrinking: self.shrinking,
                max_iter: Some(self.max_iter),
                nu: false,
            },
            self.cache_size,
        );
        let coefs: Vec<F> = coefs.iter().map(|&v| F::from_f64(v).unwrap()).collect();
        let bias = F::from_f64(bias).unwrap();

        // Extract support vectors (non-zero coefficients)
        let sv_indices: Vec<usize> = (0..x.nrows()).filter(|&i| coefs[i] != F::zero()).collect();

        if sv_indices.is_empty() {
            // Fallback: use all training points
            let dual_coefs = Array1::from_vec(coefs);
            return Ok(FittedSvr {
                support_vectors: x.to_owned(),
                dual_coefs,
                bias,
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
            dual_coefs[sv_pos] = coefs[orig_idx];
        }

        Ok(FittedSvr {
            support_vectors,
            dual_coefs,
            bias,
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
        let (x, y) = parity_data();
        let kernel = SvmKernel::rbf(0.5);
        for &(c, max_iter) in &[(1.0, 50usize), (1.0, 20_000), (0.05, 2_000)] {
            let x64 = x.clone();
            let (w, _b) = crate::solver::solve_epsilon_svr(
                crate::solver::KernelSource::Data {
                    x: &x64,
                    kernel: &kernel,
                },
                y.as_slice().unwrap(),
                c,
                0.1,
                &crate::solver::SolverParams {
                    eps: 1e-12,
                    shrinking: true,
                    max_iter: Some(max_iter),
                    nu: false,
                },
                200.0,
            );
            let sum: f64 = w.iter().sum();
            assert!(
                sum.abs() < 1e-6,
                "Σ(α − α*) = {sum} (C={c}, iters={max_iter})"
            );
            assert!(w.iter().all(|&wi| wi.abs() <= c + 1e-12));
        }
    }

    #[test]
    fn test_rbf_matches_sklearn() {
        // sklearn SVR(kernel='rbf', gamma=0.5, C=1, epsilon=0.1, tol=1e-10)
        let sk_pred = [
            -0.8629999947669329,
            -0.4229999800405266,
            -0.12000000600255367,
            -1.3537266608386191,
            1.0241806466400396,
            -0.4610000030940126,
            0.08300000989060297,
            -0.30399999659300175,
            0.6215526217944429,
            0.9906390972524501,
            0.410394264319056,
            0.16299999447920932,
            -0.36821779800692017,
            -0.24847108352010214,
            0.24799999641924758,
            -0.609999968434382,
            -0.15860107604854762,
            -0.8742795102520489,
            -0.04015272739990233,
            0.43214326812710635,
        ];
        let sk_bias = -0.158313369367737;
        let (x, y) = parity_data();
        let fitted: FittedSvr<f64> = Svr::new()
            .with_kernel(SvmKernel::rbf(0.5))
            .with_c(1.0)
            .with_epsilon(0.1)
            .with_max_iter(100_000)
            .with_tol(1e-10)
            .fit(&x, &y)
            .unwrap();
        let pred = fitted.predict(&x).unwrap();
        for (i, (p, e)) in pred.iter().zip(sk_pred.iter()).enumerate() {
            assert!((p - e).abs() < 1e-4, "pred[{i}] = {p}, sklearn {e}");
        }
        assert!(
            (fitted.bias() - sk_bias).abs() < 1e-4,
            "bias = {}, sklearn {sk_bias}",
            fitted.bias()
        );
    }

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
            .with_kernel(SvmKernel::rbf(0.1))
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
