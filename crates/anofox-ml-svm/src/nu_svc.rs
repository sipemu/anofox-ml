//! Nu-Support Vector Classification (NuSVC).
//!
//! Nu-parameterized SVC where `nu` in (0, 1] replaces C.
//! Nu is an upper bound on the fraction of margin errors and a lower bound
//! on the fraction of support vectors.

use anofox_ml_core::{Fit, Float, Predict, Result, RustMlError};
use ndarray::{Array1, Array2};

use crate::kernel::SvmKernel;
use crate::svc;

/// Nu-Support Vector Classifier (unfitted state).
///
/// Uses a nu parameter instead of C to control the trade-off between
/// margin errors and support vectors. The parameter `nu` is an upper bound
/// on the fraction of margin errors and a lower bound on the fraction of
/// support vectors.
///
/// Uses the type-state pattern: call [`Fit::fit`] to produce a [`FittedNuSvc`]
/// that can make predictions.
///
/// For multi-class problems, a one-vs-rest (OvR) strategy is used
/// automatically.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NuSvc {
    /// Nu parameter in (0, 1]. Upper bound on the fraction of margin errors
    /// and a lower bound on the fraction of support vectors.
    pub nu: f64,
    /// Kernel function to use.
    pub kernel: SvmKernel,
    /// Maximum number of iterations for the SMO solver.
    pub max_iter: usize,
    /// Tolerance for the stopping criterion.
    pub tol: f64,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl NuSvc {
    /// Create a new `NuSvc` with default parameters.
    pub fn new() -> Self {
        Self {
            nu: 0.5,
            kernel: SvmKernel::Rbf { gamma: 1.0 },
            max_iter: 1000,
            tol: 1e-4,
            seed: 0,
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

    /// Set the maximum number of iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the tolerance for the stopping criterion.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
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

impl Default for NuSvc {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Nu-Support Vector Classifier.
///
/// Wraps a [`FittedSvc`](crate::FittedSvc) internally, since NuSVC converts
/// nu to an equivalent C and delegates to the standard SVC solver.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedNuSvc<F: Float> {
    inner: svc::FittedSvc<F>,
}

impl<F: Float> FittedNuSvc<F> {
    /// Returns the class labels.
    pub fn class_labels(&self) -> &[F] {
        self.inner.class_labels()
    }

    /// Returns all support vectors across all binary classifiers.
    pub fn support_vectors(&self) -> Array2<F> {
        self.inner.support_vectors()
    }

    /// Returns the total number of support vectors across all classifiers.
    pub fn n_support(&self) -> usize {
        self.inner.n_support()
    }

    /// Compute raw decision function scores for each sample.
    ///
    /// Returns a 2D array of shape `(n_samples, n_classifiers)`.
    pub fn decision_function(&self, x: &Array2<F>) -> Result<Array2<F>> {
        self.inner.decision_function(x)
    }
}

impl<F: Float> Predict<F> for FittedNuSvc<F> {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        self.inner.predict(x)
    }
}

/// Extract unique sorted class labels from y.
fn extract_class_labels<F: Float>(y: &Array1<F>) -> Vec<F> {
    let mut labels: Vec<F> = y.to_vec();
    labels.sort_by(|a, b| a.partial_cmp(b).unwrap());
    labels.dedup_by(|a, b| (*a - *b).abs() < F::from_f64(1e-12).unwrap());
    labels
}

/// Bisect C until the C-SVC solution has a fraction of support vectors
/// approximately equal to nu.
///
/// For nu-SVC, `nu` is both an upper bound on the fraction of margin errors
/// and a lower bound on the fraction of support vectors. At the optimum these
/// coincide. Since our C-SVC solver uses SMO and returns an SV count via
/// `FittedSvc::n_support()`, we bisect C until that ratio matches nu. Large
/// C produces tight margins (few SVs), small C produces loose margins
/// (many SVs), so the count is monotonic in C and bisection converges.
fn bisect_c_for_nu<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    nu: f64,
    kernel: &SvmKernel,
    max_iter: usize,
    tol: f64,
    seed: u64,
) -> Result<svc::FittedSvc<F>> {
    let n = x.nrows();
    let target_svs = (nu * n as f64).round() as usize;
    let target_svs = target_svs.clamp(1, n);

    let mut c_lo = 1e-4f64;
    let mut c_hi = 1e4f64;

    let mut best_fitted: Option<svc::FittedSvc<F>> = None;
    let mut best_diff = usize::MAX;

    for _iter in 0..40 {
        // Bisect in log-space since C spans several orders of magnitude.
        let c = (c_lo.ln() * 0.5 + c_hi.ln() * 0.5).exp();

        let svc_model = crate::Svc::new()
            .with_c(c)
            .with_kernel(kernel.clone())
            .with_max_iter(max_iter)
            .with_tol(tol)
            .with_seed(seed);

        let fitted: svc::FittedSvc<F> = svc_model.fit(x, y)?;
        let n_sv = fitted.n_support();

        let diff = if n_sv > target_svs {
            n_sv - target_svs
        } else {
            target_svs - n_sv
        };

        if diff < best_diff {
            best_diff = diff;
            best_fitted = Some(fitted);
        }

        if diff == 0 {
            break;
        }

        // More SVs than target => need larger C (tighter margin).
        // Fewer SVs => need smaller C (looser margin).
        if n_sv > target_svs {
            c_lo = c;
        } else {
            c_hi = c;
        }

        if (c_hi.ln() - c_lo.ln()) < 1e-10 {
            break;
        }
    }

    best_fitted.ok_or_else(|| {
        RustMlError::InvalidParameter("nu-SVC bisection failed to produce a fit".into())
    })
}

impl<F: Float> Fit<F> for NuSvc {
    type Fitted = FittedNuSvc<F>;

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

        let class_labels = extract_class_labels(y);
        if class_labels.len() < 2 {
            return Err(RustMlError::InvalidParameter(
                "need at least 2 distinct classes for classification".into(),
            ));
        }

        // Feasibility check: nu must be achievable for each binary sub-problem.
        // For a two-class split, nu <= 2 * min(n_pos, n_neg) / n_total.
        let n_total = y.len();
        let eps = F::from_f64(1e-12).unwrap();
        for label in &class_labels {
            let n_pos = y.iter().filter(|&&yi| (yi - *label).abs() < eps).count();
            let n_neg = n_total - n_pos;
            let max_nu = 2.0 * (n_pos.min(n_neg) as f64) / (n_total as f64);
            if self.nu > max_nu + 1e-12 {
                return Err(RustMlError::InvalidParameter(format!(
                    "nu={} is infeasible (class {:?} has max feasible nu = {:.4})",
                    self.nu, *label, max_nu
                )));
            }
        }

        // Bisect C so the resulting C-SVC has a fraction of support vectors
        // approximately equal to nu. This is the principled equivalence
        // between nu-SVC and C-SVC that libsvm uses internally.
        let inner = bisect_c_for_nu(
            x,
            y,
            self.nu,
            &self.kernel,
            self.max_iter,
            self.tol,
            self.seed,
        )?;
        Ok(FittedNuSvc { inner })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    fn well_separated_data() -> (Array2<f64>, Array1<f64>) {
        let x = array![
            [0.0, 0.0],
            [0.5, 0.1],
            [0.1, 0.5],
            [0.2, 0.3],
            [5.0, 5.0],
            [5.5, 5.1],
            [5.1, 5.5],
            [5.2, 5.3]
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        (x, y)
    }

    #[test]
    fn test_binary_linear_default_nu() {
        let (x, y) = well_separated_data();
        let nu_svc = NuSvc::new()
            .with_kernel(SvmKernel::Linear)
            .with_max_iter(5000);
        let fitted: FittedNuSvc<f64> = nu_svc.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for i in 0..4 {
            assert_abs_diff_eq!(preds[i], 0.0, epsilon = 1e-10);
        }
        for i in 4..8 {
            assert_abs_diff_eq!(preds[i], 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_binary_rbf_kernel() {
        let (x, y) = well_separated_data();
        let nu_svc = NuSvc::new()
            .with_nu(0.5)
            .with_kernel(SvmKernel::Rbf { gamma: 0.5 })
            .with_max_iter(5000);
        let fitted: FittedNuSvc<f64> = nu_svc.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for i in 0..4 {
            assert_abs_diff_eq!(preds[i], 0.0, epsilon = 1e-10);
        }
        for i in 4..8 {
            assert_abs_diff_eq!(preds[i], 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_binary_polynomial_kernel() {
        let (x, y) = well_separated_data();
        let nu_svc = NuSvc::new()
            .with_nu(0.5)
            .with_kernel(SvmKernel::Polynomial {
                degree: 2,
                coef0: 1.0,
            })
            .with_max_iter(5000);
        let fitted: FittedNuSvc<f64> = nu_svc.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for i in 0..4 {
            assert_abs_diff_eq!(preds[i], 0.0, epsilon = 1e-10);
        }
        for i in 4..8 {
            assert_abs_diff_eq!(preds[i], 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_small_nu() {
        let (x, y) = well_separated_data();
        // Small nu => large C => harder margin
        let nu_svc = NuSvc::new()
            .with_nu(0.1)
            .with_kernel(SvmKernel::Linear)
            .with_max_iter(5000);
        let fitted: FittedNuSvc<f64> = nu_svc.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for i in 0..4 {
            assert_abs_diff_eq!(preds[i], 0.0, epsilon = 1e-10);
        }
        for i in 4..8 {
            assert_abs_diff_eq!(preds[i], 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_large_nu() {
        let (x, y) = well_separated_data();
        // nu = 1.0 => softest margin (max nu for balanced classes is 1.0)
        let nu_svc = NuSvc::new()
            .with_nu(1.0)
            .with_kernel(SvmKernel::Rbf { gamma: 0.5 })
            .with_max_iter(5000);
        let fitted: FittedNuSvc<f64> = nu_svc.fit(&x, &y).unwrap();

        // Should still be able to separate well-separated data
        let preds = fitted.predict(&x).unwrap();
        for i in 0..4 {
            assert_abs_diff_eq!(preds[i], 0.0, epsilon = 1e-10);
        }
        for i in 4..8 {
            assert_abs_diff_eq!(preds[i], 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_multiclass() {
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [5.0, 0.0],
            [5.1, 0.1],
            [5.2, 0.0],
            [0.0, 5.0],
            [0.1, 5.1],
            [0.0, 5.2]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];

        let nu_svc = NuSvc::new()
            .with_nu(0.5)
            .with_kernel(SvmKernel::Rbf { gamma: 0.5 })
            .with_max_iter(5000);
        let fitted: FittedNuSvc<f64> = nu_svc.fit(&x, &y).unwrap();

        assert_eq!(fitted.class_labels(), &[0.0, 1.0, 2.0]);

        let preds = fitted.predict(&x).unwrap();
        for i in 0..3 {
            assert_abs_diff_eq!(preds[i], 0.0, epsilon = 1e-10);
        }
        for i in 3..6 {
            assert_abs_diff_eq!(preds[i], 1.0, epsilon = 1e-10);
        }
        for i in 6..9 {
            assert_abs_diff_eq!(preds[i], 2.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_decision_function() {
        let (x, y) = well_separated_data();
        let nu_svc = NuSvc::new()
            .with_nu(0.5)
            .with_kernel(SvmKernel::Linear)
            .with_max_iter(5000);
        let fitted: FittedNuSvc<f64> = nu_svc.fit(&x, &y).unwrap();

        let scores = fitted.decision_function(&x).unwrap();
        assert_eq!(scores.nrows(), x.nrows());
        assert_eq!(scores.ncols(), 1); // binary

        // Class 0 should have negative scores; class 1 positive.
        for i in 0..4 {
            assert!(scores[[i, 0]] < 0.0, "expected negative for class 0");
        }
        for i in 4..8 {
            assert!(scores[[i, 0]] > 0.0, "expected positive for class 1");
        }
    }

    #[test]
    fn test_invalid_nu_zero() {
        let (x, y) = well_separated_data();
        let nu_svc = NuSvc::new().with_nu(0.0);
        let result: Result<FittedNuSvc<f64>> = nu_svc.fit(&x, &y);
        assert!(result.is_err());
        match result {
            Err(RustMlError::InvalidParameter(msg)) => {
                assert!(msg.contains("nu"), "error should mention nu: {}", msg);
            }
            other => panic!("expected InvalidParameter error, got {:?}", other),
        }
    }

    #[test]
    fn test_invalid_nu_negative() {
        let (x, y) = well_separated_data();
        let nu_svc = NuSvc::new().with_nu(-0.5);
        let result: Result<FittedNuSvc<f64>> = nu_svc.fit(&x, &y);
        assert!(result.is_err());
        match result {
            Err(RustMlError::InvalidParameter(_)) => {}
            other => panic!("expected InvalidParameter error, got {:?}", other),
        }
    }

    #[test]
    fn test_invalid_nu_above_one() {
        let (x, y) = well_separated_data();
        let nu_svc = NuSvc::new().with_nu(1.5);
        let result: Result<FittedNuSvc<f64>> = nu_svc.fit(&x, &y);
        assert!(result.is_err());
        match result {
            Err(RustMlError::InvalidParameter(_)) => {}
            other => panic!("expected InvalidParameter error, got {:?}", other),
        }
    }

    #[test]
    fn test_empty_input_error() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<f64>::zeros(0);

        let nu_svc = NuSvc::new();
        let result: Result<FittedNuSvc<f64>> = nu_svc.fit(&x, &y);
        assert!(result.is_err());
        match result {
            Err(RustMlError::EmptyInput(_)) => {}
            other => panic!("expected EmptyInput error, got {:?}", other),
        }
    }

    #[test]
    fn test_shape_mismatch_fit() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0, 0.0];

        let nu_svc = NuSvc::new();
        let result: Result<FittedNuSvc<f64>> = nu_svc.fit(&x, &y);
        assert!(result.is_err());
        match result {
            Err(RustMlError::ShapeMismatch(_)) => {}
            other => panic!("expected ShapeMismatch error, got {:?}", other),
        }
    }

    #[test]
    fn test_shape_mismatch_predict() {
        let (x, y) = well_separated_data();
        let nu_svc = NuSvc::new()
            .with_kernel(SvmKernel::Linear)
            .with_max_iter(5000);
        let fitted: FittedNuSvc<f64> = nu_svc.fit(&x, &y).unwrap();

        let x_bad = array![[1.0, 2.0, 3.0]];
        let result = fitted.predict(&x_bad);
        assert!(result.is_err());
        match result {
            Err(RustMlError::ShapeMismatch(_)) => {}
            other => panic!("expected ShapeMismatch error, got {:?}", other),
        }
    }

    #[test]
    fn test_single_class_error() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 0.0];

        let nu_svc = NuSvc::new();
        let result: Result<FittedNuSvc<f64>> = nu_svc.fit(&x, &y);
        assert!(result.is_err());
        match result {
            Err(RustMlError::InvalidParameter(_)) => {}
            other => panic!("expected InvalidParameter error, got {:?}", other),
        }
    }

    #[test]
    fn test_builder_and_defaults() {
        let nu_svc = NuSvc::new()
            .with_nu(0.3)
            .with_kernel(SvmKernel::Linear)
            .with_max_iter(500)
            .with_tol(1e-3)
            .with_seed(42);
        assert_eq!(nu_svc.nu, 0.3);
        assert_eq!(nu_svc.max_iter, 500);
        assert_eq!(nu_svc.tol, 1e-3);
        assert_eq!(nu_svc.seed, 42);
        assert!(matches!(nu_svc.kernel, SvmKernel::Linear));

        let default = NuSvc::default();
        assert_eq!(default.nu, 0.5);
        assert_eq!(default.max_iter, 1000);
    }

    #[test]
    fn test_support_vectors() {
        let (x, y) = well_separated_data();
        let nu_svc = NuSvc::new()
            .with_nu(0.5)
            .with_kernel(SvmKernel::Linear)
            .with_max_iter(5000);
        let fitted: FittedNuSvc<f64> = nu_svc.fit(&x, &y).unwrap();

        let sv = fitted.support_vectors();
        let n_sv = fitted.n_support();
        assert_eq!(sv.nrows(), n_sv);
        assert!(n_sv > 0, "should have at least one support vector");
        assert!(
            n_sv <= x.nrows(),
            "cannot have more SVs than training samples"
        );
    }
}
