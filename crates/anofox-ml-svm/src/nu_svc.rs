//! Nu-Support Vector Classification (NuSVC).
//!
//! Nu-parameterized SVC where `nu` in (0, 1] replaces C.
//! Nu is an upper bound on the fraction of margin errors and a lower bound
//! on the fraction of support vectors.

use anofox_ml_core::{Fit, Float, Predict, Result, RustMlError};
use ndarray::{Array1, Array2};

use crate::kernel::SvmKernel;
use crate::svc::{self, DecisionFunctionShape};

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
/// For multi-class problems, a one-vs-one strategy is used (as libsvm),
/// via the underlying [`Svc`](crate::Svc).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NuSvc {
    /// Nu parameter in (0, 1]. Upper bound on the fraction of margin errors
    /// and a lower bound on the fraction of support vectors.
    pub nu: f64,
    /// Kernel function to use.
    pub kernel: SvmKernel,
    /// Maximum number of SMO iterations per class pair.
    pub max_iter: usize,
    /// Tolerance of the stopping criterion (default 1e-3).
    pub tol: f64,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Fit Platt-scaling models so that `predict_proba` is available.
    #[serde(default)]
    pub probability: bool,
    /// Layout of `decision_function` for more than two classes.
    #[serde(default)]
    pub decision_function_shape: DecisionFunctionShape,
    /// Break voting ties by decision-function confidence (`Ovr` only).
    #[serde(default)]
    pub break_ties: bool,
}

impl NuSvc {
    /// Create a new `NuSvc` with default parameters.
    pub fn new() -> Self {
        Self {
            nu: 0.5,
            kernel: SvmKernel::default(),
            max_iter: 10_000_000,
            tol: 1e-3,
            seed: 0,
            probability: false,
            decision_function_shape: DecisionFunctionShape::Ovr,
            break_ties: false,
        }
    }

    /// Enable Platt-scaled probability estimates.
    pub fn with_probability(mut self, probability: bool) -> Self {
        self.probability = probability;
        self
    }

    /// Set the multi-class decision-function layout.
    pub fn with_decision_function_shape(mut self, shape: DecisionFunctionShape) -> Self {
        self.decision_function_shape = shape;
        self
    }

    /// Break voting ties by decision-function confidence.
    pub fn with_break_ties(mut self, break_ties: bool) -> Self {
        self.break_ties = break_ties;
        self
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
        self.kernel.validate(true)?;
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
/// Wraps a [`FittedSvc`](crate::FittedSvc): the one-vs-one sub-models come
/// from libsvm's ν-SVC dual (two equality constraints, ν solver).
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

    /// Decision function; see [`FittedSvc::decision_function`](crate::FittedSvc::decision_function)
    /// for the shapes.
    pub fn decision_function(&self, x: &Array2<F>) -> Result<Array2<F>> {
        self.inner.decision_function(x)
    }

    /// Platt-scaled class probabilities (requires `probability = true`).
    pub fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>> {
        self.inner.predict_proba(x)
    }

    /// The underlying fitted C-SVC.
    pub fn inner(&self) -> &svc::FittedSvc<F> {
        &self.inner
    }
}

impl<F: Float> anofox_ml_core::DecisionFunction<F> for FittedNuSvc<F> {
    fn decision_function(&self, x: &Array2<F>) -> Result<Array2<F>> {
        self.inner.decision_function(x)
    }
}

impl<F: Float> anofox_ml_core::PredictProba<F> for FittedNuSvc<F> {
    fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>> {
        self.inner.predict_proba(x)
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
    labels.sort_by(anofox_ml_core::labels::label_cmp);
    labels.dedup_by(|a, b| (*a - *b).abs() < F::from_f64(1e-12).unwrap());
    labels
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

        // Feasibility check (libsvm): for every one-vs-one pair (i, j),
        // nu <= 2 * min(n_i, n_j) / (n_i + n_j).
        let eps = F::from_f64(1e-12).unwrap();
        let counts: Vec<usize> = class_labels
            .iter()
            .map(|label| y.iter().filter(|&&yi| (yi - *label).abs() < eps).count())
            .collect();
        for i in 0..counts.len() {
            for j in (i + 1)..counts.len() {
                let max_nu =
                    2.0 * (counts[i].min(counts[j]) as f64) / ((counts[i] + counts[j]) as f64);
                if self.nu > max_nu + 1e-12 {
                    return Err(RustMlError::InvalidParameter(format!(
                        "nu={} is infeasible (classes {:?}/{:?} allow at most nu = {:.4})",
                        self.nu, class_labels[i], class_labels[j], max_nu
                    )));
                }
            }
        }

        // libsvm's ν-SVC dual for every one-vs-one pair.
        let template = crate::Svc::new()
            .with_kernel(self.kernel.clone())
            .with_max_iter(self.max_iter)
            .with_tol(self.tol)
            .with_seed(self.seed)
            .with_probability(self.probability)
            .with_decision_function_shape(self.decision_function_shape)
            .with_break_ties(self.break_ties);
        let inner = template.fit_nu(x, y, self.nu)?;
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
            .with_kernel(SvmKernel::rbf(0.5))
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
            .with_kernel(SvmKernel::poly(2, 1.0, 1.0))
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
        // nu = 1.0 makes libsvm's nu dual degenerate (r = 0): sklearn raises
        // "dual coefficients ... not finite", and so do we.
        assert!(NuSvc::new()
            .with_nu(1.0)
            .with_kernel(SvmKernel::rbf(0.5))
            .fit(&x, &y)
            .is_err());
        let nu_svc = NuSvc::new()
            .with_nu(0.9)
            .with_kernel(SvmKernel::rbf(0.5))
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
            .with_kernel(SvmKernel::rbf(0.5))
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
        assert_eq!(default.max_iter, 10_000_000);
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

    #[test]
    fn test_nu_dual_matches_sklearn() {
        // sklearn 1.8 NuSVC(nu=0.4, kernel='rbf', gamma=0.7, tol=1e-8) on
        // round(RandomState(0).randn(40, 2), 4), y = x0 + 0.5 noise > 0.
        let xv = [
            1.7641, 0.4002, 0.9787, 2.2409, 1.8676, -0.9773, 0.9501, -0.1514, -0.1032, 0.4106,
            0.144, 1.4543, 0.761, 0.1217, 0.4439, 0.3337, 1.4941, -0.2052, 0.3131, -0.8541, -2.553,
            0.6536, 0.8644, -0.7422, 2.2698, -1.4544, 0.0458, -0.1872, 1.5328, 1.4694, 0.1549,
            0.3782, -0.8878, -1.9808, -0.3479, 0.1563, 1.2303, 1.2024, -0.3873, -0.3023, -1.0486,
            -1.42, -1.7063, 1.9508, -0.5097, -0.4381, -1.2528, 0.7775, -1.6139, -0.2127, -0.8955,
            0.3869, -0.5108, -1.1806, -0.0282, 0.4283, 0.0665, 0.3025, -0.6343, -0.3627, -0.6725,
            -0.3596, -0.8131, -1.7263, 0.1774, -0.4018, -1.6302, 0.4628, -0.9073, 0.0519, 0.7291,
            0.129, 1.1394, -1.2348, 0.4023, -0.6848, -0.8708, -0.5788, -0.3116, 0.0562,
        ];
        let yv = [
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 0., 1., 1., 0., 0., 0.,
            0., 0., 0., 1., 0., 0., 1., 1., 1., 0., 0., 0., 0., 1., 1., 1., 0., 1.,
        ];
        let x = Array2::from_shape_vec((40, 2), xv.to_vec()).unwrap();
        let y = Array1::from_vec(yv.to_vec());
        let f: FittedNuSvc<f64> = NuSvc::new()
            .with_nu(0.4)
            .with_kernel(SvmKernel::rbf(0.7))
            .with_tol(1e-8)
            .fit(&x, &y)
            .unwrap();
        let sk = [
            0.9999999803120584,
            0.9999999886807078,
            1.1131058242079421,
            1.1298730763518703,
            1.1429350344296836,
        ];
        let d = anofox_ml_core::DecisionFunction::decision_function(
            &f,
            &x.slice(ndarray::s![..5, ..]).to_owned(),
        )
        .unwrap();
        let sign = if d[[0, 0]] * sk[0] > 0.0 { 1.0 } else { -1.0 };
        for i in 0..5 {
            assert!(
                (sign * d[[i, 0]] - sk[i]).abs() < 1e-6,
                "[{i}] {} vs {}",
                d[[i, 0]],
                sk[i]
            );
        }
    }
}
