use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rustml_core::{Fit, Float, Predict, Result, RustMlError};

use crate::kernel::SvmKernel;

/// Support Vector Classifier with kernel support (unfitted state).
///
/// Implements a simplified SMO (Sequential Minimal Optimization) algorithm.
/// Uses the type-state pattern: call [`Fit::fit`] to produce a [`FittedSvc`]
/// that can make predictions.
///
/// For multi-class problems, a one-vs-rest (OvR) strategy is used
/// automatically.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Svc {
    /// Regularization parameter. Larger values mean less regularization.
    pub c: f64,
    /// Kernel function to use.
    pub kernel: SvmKernel,
    /// Maximum number of iterations for the SMO solver.
    pub max_iter: usize,
    /// Tolerance for the stopping criterion.
    pub tol: f64,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Svc {
    /// Create a new `Svc` with default parameters.
    pub fn new() -> Self {
        Self {
            c: 1.0,
            kernel: SvmKernel::Rbf { gamma: 1.0 },
            max_iter: 1000,
            tol: 1e-4,
            seed: 0,
        }
    }

    /// Set the regularization parameter C.
    pub fn with_c(mut self, c: f64) -> Self {
        self.c = c;
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
        if self.c <= 0.0 {
            return Err(RustMlError::InvalidParameter(
                "C must be positive".into(),
            ));
        }
        if self.max_iter == 0 {
            return Err(RustMlError::InvalidParameter(
                "max_iter must be at least 1".into(),
            ));
        }
        if self.tol <= 0.0 {
            return Err(RustMlError::InvalidParameter(
                "tol must be positive".into(),
            ));
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

impl Default for Svc {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted binary SVC storing support vectors and dual coefficients.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
struct BinarySvc<F: Float> {
    /// Support vectors (subset of training data).
    support_vectors: Array2<F>,
    /// Dual coefficients (alpha_i * y_i) for each support vector.
    dual_coefs: Array1<F>,
    /// Bias term.
    bias: F,
    /// Kernel used for this classifier.
    kernel: SvmKernel,
}

impl<F: Float> BinarySvc<F> {
    /// Compute decision function for a single sample.
    fn decision_value(&self, sample: &ndarray::ArrayView1<F>) -> F {
        let mut result = self.bias;
        for (sv_idx, sv) in self.support_vectors.rows().into_iter().enumerate() {
            result += self.dual_coefs[sv_idx] * self.kernel.compute(&sv, sample);
        }
        result
    }
}

/// Fitted Support Vector Classifier.
///
/// For binary problems, contains a single set of support vectors + bias.
/// For multi-class problems, contains one binary classifier per class
/// (one-vs-rest).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedSvc<F: Float> {
    /// Unique sorted class labels.
    class_labels: Vec<F>,
    /// One binary classifier per class (OvR), or a single one for binary.
    classifiers: Vec<BinarySvc<F>>,
}

impl<F: Float> FittedSvc<F> {
    /// Returns the class labels.
    pub fn class_labels(&self) -> &[F] {
        &self.class_labels
    }

    /// Returns all support vectors across all binary classifiers.
    /// For binary classification, returns the single set of support vectors.
    /// For multi-class, concatenates support vectors from all OvR classifiers
    /// (may contain duplicates).
    pub fn support_vectors(&self) -> Array2<F> {
        if self.classifiers.len() == 1 {
            return self.classifiers[0].support_vectors.clone();
        }
        let n_features = self.classifiers[0].support_vectors.ncols();
        let total_rows: usize = self.classifiers.iter().map(|c| c.support_vectors.nrows()).sum();
        let mut result = Array2::zeros((total_rows, n_features));
        let mut offset = 0;
        for clf in &self.classifiers {
            let n = clf.support_vectors.nrows();
            result
                .slice_mut(ndarray::s![offset..offset + n, ..])
                .assign(&clf.support_vectors);
            offset += n;
        }
        result
    }

    /// Returns the total number of support vectors across all classifiers.
    pub fn n_support(&self) -> usize {
        self.classifiers.iter().map(|c| c.support_vectors.nrows()).sum()
    }

    /// Compute raw decision function scores for each sample.
    ///
    /// Returns a 2D array of shape `(n_samples, n_classifiers)`.
    pub fn decision_function(&self, x: &Array2<F>) -> Result<Array2<F>> {
        if x.is_empty() {
            return Err(RustMlError::EmptyInput(
                "prediction input must not be empty".into(),
            ));
        }
        let n_features = self.classifiers[0].support_vectors.ncols();
        if x.ncols() != n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                n_features,
                x.ncols()
            )));
        }

        let n_samples = x.nrows();
        let n_classifiers = self.classifiers.len();
        let mut scores = Array2::zeros((n_samples, n_classifiers));

        for (ci, clf) in self.classifiers.iter().enumerate() {
            for (i, sample) in x.rows().into_iter().enumerate() {
                scores[[i, ci]] = clf.decision_value(&sample);
            }
        }

        Ok(scores)
    }
}

impl<F: Float> Predict<F> for FittedSvc<F> {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        let scores = self.decision_function(x)?;
        let n_samples = x.nrows();
        let mut predictions = Array1::zeros(n_samples);

        if self.class_labels.len() == 2 {
            // Binary: positive score -> class_labels[1], negative -> class_labels[0]
            for i in 0..n_samples {
                if scores[[i, 0]] >= F::zero() {
                    predictions[i] = self.class_labels[1];
                } else {
                    predictions[i] = self.class_labels[0];
                }
            }
        } else {
            // Multi-class OvR: pick the class with the highest score.
            for i in 0..n_samples {
                let mut best_ci = 0;
                let mut best_score = scores[[i, 0]];
                for ci in 1..self.classifiers.len() {
                    if scores[[i, ci]] > best_score {
                        best_score = scores[[i, ci]];
                        best_ci = ci;
                    }
                }
                predictions[i] = self.class_labels[best_ci];
            }
        }

        Ok(predictions)
    }
}

/// Extract unique sorted class labels from y.
fn extract_class_labels<F: Float>(y: &Array1<F>) -> Vec<F> {
    let mut labels: Vec<F> = y.to_vec();
    labels.sort_by(|a, b| a.partial_cmp(b).unwrap());
    labels.dedup_by(|a, b| (*a - *b).abs() < F::from_f64(1e-12).unwrap());
    labels
}

/// Train a single binary SVC using simplified SMO.
///
/// Labels must be +1/-1 encoded.
fn fit_binary_svc<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    kernel: &SvmKernel,
    c: F,
    max_iter: usize,
    tol: F,
    seed: u64,
) -> BinarySvc<F> {
    let n_samples = x.nrows();
    let mut alpha = vec![F::zero(); n_samples];
    let mut bias = F::zero();

    let zero = F::zero();
    let two = F::from_f64(2.0).unwrap();

    // Precompute the kernel matrix for efficiency.
    let mut k_matrix = Array2::<F>::zeros((n_samples, n_samples));
    for i in 0..n_samples {
        for j in i..n_samples {
            let val = kernel.compute(&x.row(i), &x.row(j));
            k_matrix[[i, j]] = val;
            k_matrix[[j, i]] = val;
        }
    }

    let _rng = StdRng::seed_from_u64(seed);

    for _iter in 0..max_iter {
        let mut num_changed = 0usize;

        for i in 0..n_samples {
            // Compute error for sample i: E_i = f(x_i) - y_i
            let mut f_i = bias;
            for j in 0..n_samples {
                f_i += alpha[j] * y[j] * k_matrix[[j, i]];
            }
            let e_i = f_i - y[i];

            // Check KKT conditions (simplified)
            let yi_ei = y[i] * e_i;
            if (yi_ei < -tol && alpha[i] < c) || (yi_ei > tol && alpha[i] > zero) {
                // Select j != i (use simple heuristic: pick j with max |Ei - Ej|)
                let mut best_j = if i == 0 { 1 } else { 0 };
                let mut best_delta = F::zero();
                for j in 0..n_samples {
                    if j == i {
                        continue;
                    }
                    let mut f_j = bias;
                    for k in 0..n_samples {
                        f_j += alpha[k] * y[k] * k_matrix[[k, j]];
                    }
                    let e_j = f_j - y[j];
                    let delta = (e_i - e_j).abs();
                    if delta > best_delta {
                        best_delta = delta;
                        best_j = j;
                    }
                }

                let j = best_j;

                // Compute error for sample j
                let mut f_j = bias;
                for k in 0..n_samples {
                    f_j += alpha[k] * y[k] * k_matrix[[k, j]];
                }
                let e_j = f_j - y[j];

                let old_ai = alpha[i];
                let old_aj = alpha[j];

                // Compute bounds L and H
                let (l, h) = if (y[i] - y[j]).abs() > F::from_f64(1e-12).unwrap() {
                    // y_i != y_j
                    let l_val = if alpha[j] - alpha[i] > zero {
                        alpha[j] - alpha[i]
                    } else {
                        zero
                    };
                    let h_val = if c + alpha[j] - alpha[i] < c {
                        c + alpha[j] - alpha[i]
                    } else {
                        c
                    };
                    (l_val, h_val)
                } else {
                    // y_i == y_j
                    let l_val = if alpha[i] + alpha[j] - c > zero {
                        alpha[i] + alpha[j] - c
                    } else {
                        zero
                    };
                    let h_val = if alpha[i] + alpha[j] < c {
                        alpha[i] + alpha[j]
                    } else {
                        c
                    };
                    (l_val, h_val)
                };

                if (l - h).abs() < F::from_f64(1e-12).unwrap() {
                    continue;
                }

                // Compute eta = 2*K(i,j) - K(i,i) - K(j,j)
                let eta = two * k_matrix[[i, j]] - k_matrix[[i, i]] - k_matrix[[j, j]];
                if eta >= zero {
                    continue;
                }

                // Update alpha_j
                let mut new_aj = old_aj - y[j] * (e_i - e_j) / eta;
                // Clip to [L, H]
                if new_aj > h {
                    new_aj = h;
                } else if new_aj < l {
                    new_aj = l;
                }

                if (new_aj - old_aj).abs() < F::from_f64(1e-8).unwrap() {
                    continue;
                }

                alpha[j] = new_aj;
                alpha[i] = old_ai + y[i] * y[j] * (old_aj - new_aj);

                // Update bias
                let b1 = bias - e_i
                    - y[i] * (alpha[i] - old_ai) * k_matrix[[i, i]]
                    - y[j] * (alpha[j] - old_aj) * k_matrix[[i, j]];
                let b2 = bias - e_j
                    - y[i] * (alpha[i] - old_ai) * k_matrix[[i, j]]
                    - y[j] * (alpha[j] - old_aj) * k_matrix[[j, j]];

                if alpha[i] > zero && alpha[i] < c {
                    bias = b1;
                } else if alpha[j] > zero && alpha[j] < c {
                    bias = b2;
                } else {
                    bias = (b1 + b2) / two;
                }

                num_changed += 1;
            }
        }

        if num_changed == 0 {
            break;
        }
    }

    // Extract support vectors (alpha > 0)
    let sv_threshold = F::from_f64(1e-8).unwrap();
    let sv_indices: Vec<usize> = (0..n_samples)
        .filter(|&i| alpha[i] > sv_threshold)
        .collect();

    let n_sv = sv_indices.len();
    let n_features = x.ncols();

    if n_sv == 0 {
        // Fallback: no support vectors found, use all points.
        let dual_coefs = Array1::from_vec(
            (0..n_samples).map(|i| alpha[i] * y[i]).collect(),
        );
        return BinarySvc {
            support_vectors: x.to_owned(),
            dual_coefs,
            bias,
            kernel: kernel.clone(),
        };
    }

    let mut support_vectors = Array2::zeros((n_sv, n_features));
    let mut dual_coefs = Array1::zeros(n_sv);

    for (sv_pos, &orig_idx) in sv_indices.iter().enumerate() {
        support_vectors.row_mut(sv_pos).assign(&x.row(orig_idx));
        dual_coefs[sv_pos] = alpha[orig_idx] * y[orig_idx];
    }

    BinarySvc {
        support_vectors,
        dual_coefs,
        bias,
        kernel: kernel.clone(),
    }
}

impl<F: Float> Fit<F> for Svc {
    type Fitted = FittedSvc<F>;

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

        let c = F::from_f64(self.c).unwrap();
        let tol = F::from_f64(self.tol).unwrap();
        let eps = F::from_f64(1e-12).unwrap();

        if class_labels.len() == 2 {
            let y_binary = y.mapv(|yi| {
                if (yi - class_labels[1]).abs() < eps {
                    F::one()
                } else {
                    -F::one()
                }
            });

            let clf = fit_binary_svc(
                x,
                &y_binary,
                &self.kernel,
                c,
                self.max_iter,
                tol,
                self.seed,
            );
            Ok(FittedSvc {
                class_labels,
                classifiers: vec![clf],
            })
        } else {
            let mut classifiers = Vec::with_capacity(class_labels.len());

            for (ci, &label) in class_labels.iter().enumerate() {
                let y_binary = y.mapv(|yi| {
                    if (yi - label).abs() < eps {
                        F::one()
                    } else {
                        -F::one()
                    }
                });

                let seed_offset = ci as u64;
                let clf = fit_binary_svc(
                    x,
                    &y_binary,
                    &self.kernel,
                    c,
                    self.max_iter,
                    tol,
                    self.seed.wrapping_add(seed_offset),
                );
                classifiers.push(clf);
            }

            Ok(FittedSvc {
                class_labels,
                classifiers,
            })
        }
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
    fn test_binary_linear_kernel_f64() {
        let (x, y) = well_separated_data();
        let svc = Svc::new()
            .with_kernel(SvmKernel::Linear)
            .with_c(10.0)
            .with_max_iter(5000);
        let fitted: FittedSvc<f64> = svc.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for i in 0..4 {
            assert_abs_diff_eq!(preds[i], 0.0, epsilon = 1e-10);
        }
        for i in 4..8 {
            assert_abs_diff_eq!(preds[i], 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_binary_rbf_kernel_f64() {
        let (x, y) = well_separated_data();
        let svc = Svc::new()
            .with_kernel(SvmKernel::Rbf { gamma: 0.5 })
            .with_c(10.0)
            .with_max_iter(5000);
        let fitted: FittedSvc<f64> = svc.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for i in 0..4 {
            assert_abs_diff_eq!(preds[i], 0.0, epsilon = 1e-10);
        }
        for i in 4..8 {
            assert_abs_diff_eq!(preds[i], 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_binary_polynomial_kernel_f64() {
        let (x, y) = well_separated_data();
        let svc = Svc::new()
            .with_kernel(SvmKernel::Polynomial {
                degree: 2,
                coef0: 1.0,
            })
            .with_c(10.0)
            .with_max_iter(5000);
        let fitted: FittedSvc<f64> = svc.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for i in 0..4 {
            assert_abs_diff_eq!(preds[i], 0.0, epsilon = 1e-10);
        }
        for i in 4..8 {
            assert_abs_diff_eq!(preds[i], 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_binary_rbf_kernel_f32() {
        let x: Array2<f32> = array![
            [0.0, 0.0],
            [0.5, 0.1],
            [0.1, 0.5],
            [0.2, 0.3],
            [5.0, 5.0],
            [5.5, 5.1],
            [5.1, 5.5],
            [5.2, 5.3]
        ];
        let y: Array1<f32> = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let svc = Svc::new()
            .with_kernel(SvmKernel::Rbf { gamma: 0.5 })
            .with_c(10.0)
            .with_max_iter(5000);
        let fitted: FittedSvc<f32> = svc.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for i in 0..4 {
            assert_abs_diff_eq!(preds[i], 0.0_f32, epsilon = 1e-5);
        }
        for i in 4..8 {
            assert_abs_diff_eq!(preds[i], 1.0_f32, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_support_vectors() {
        let (x, y) = well_separated_data();
        let svc = Svc::new()
            .with_kernel(SvmKernel::Linear)
            .with_c(10.0)
            .with_max_iter(5000);
        let fitted: FittedSvc<f64> = svc.fit(&x, &y).unwrap();

        let sv = fitted.support_vectors();
        let n_sv = fitted.n_support();
        assert_eq!(sv.nrows(), n_sv);
        assert!(n_sv > 0, "should have at least one support vector");
        assert!(n_sv <= x.nrows(), "cannot have more SVs than training samples");
    }

    #[test]
    fn test_decision_function() {
        let (x, y) = well_separated_data();
        let svc = Svc::new()
            .with_kernel(SvmKernel::Linear)
            .with_c(10.0)
            .with_max_iter(5000);
        let fitted: FittedSvc<f64> = svc.fit(&x, &y).unwrap();

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
    fn test_multiclass_svc() {
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

        let svc = Svc::new()
            .with_kernel(SvmKernel::Rbf { gamma: 0.5 })
            .with_c(10.0)
            .with_max_iter(5000);
        let fitted: FittedSvc<f64> = svc.fit(&x, &y).unwrap();

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
    fn test_empty_input_error() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<f64>::zeros(0);

        let svc = Svc::new();
        let result: Result<FittedSvc<f64>> = svc.fit(&x, &y);
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

        let svc = Svc::new();
        let result: Result<FittedSvc<f64>> = svc.fit(&x, &y);
        assert!(result.is_err());
        match result {
            Err(RustMlError::ShapeMismatch(_)) => {}
            other => panic!("expected ShapeMismatch error, got {:?}", other),
        }
    }

    #[test]
    fn test_shape_mismatch_predict() {
        let (x, y) = well_separated_data();
        let svc = Svc::new().with_kernel(SvmKernel::Linear).with_c(10.0);
        let fitted: FittedSvc<f64> = svc.fit(&x, &y).unwrap();

        let x_bad = array![[1.0, 2.0, 3.0]];
        let result = fitted.predict(&x_bad);
        assert!(result.is_err());
        match result {
            Err(RustMlError::ShapeMismatch(_)) => {}
            other => panic!("expected ShapeMismatch error, got {:?}", other),
        }
    }

    #[test]
    fn test_invalid_c() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0];

        let svc = Svc::new().with_c(-1.0);
        let result: Result<FittedSvc<f64>> = svc.fit(&x, &y);
        assert!(result.is_err());
        match result {
            Err(RustMlError::InvalidParameter(_)) => {}
            other => panic!("expected InvalidParameter error, got {:?}", other),
        }
    }

    #[test]
    fn test_invalid_gamma() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0];

        let svc = Svc::new().with_kernel(SvmKernel::Rbf { gamma: -0.5 });
        let result: Result<FittedSvc<f64>> = svc.fit(&x, &y);
        assert!(result.is_err());
        match result {
            Err(RustMlError::InvalidParameter(_)) => {}
            other => panic!("expected InvalidParameter error, got {:?}", other),
        }
    }

    #[test]
    fn test_single_class_error() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 0.0];

        let svc = Svc::new();
        let result: Result<FittedSvc<f64>> = svc.fit(&x, &y);
        assert!(result.is_err());
        match result {
            Err(RustMlError::InvalidParameter(_)) => {}
            other => panic!("expected InvalidParameter error, got {:?}", other),
        }
    }

    #[test]
    fn test_builder_pattern() {
        let svc = Svc::new()
            .with_c(0.5)
            .with_kernel(SvmKernel::Linear)
            .with_max_iter(500)
            .with_tol(1e-3)
            .with_seed(42);
        assert_eq!(svc.c, 0.5);
        assert_eq!(svc.max_iter, 500);
        assert_eq!(svc.tol, 1e-3);
        assert_eq!(svc.seed, 42);
        assert!(matches!(svc.kernel, SvmKernel::Linear));
    }

    #[test]
    fn test_default() {
        let svc = Svc::default();
        assert_eq!(svc.c, 1.0);
        assert_eq!(svc.max_iter, 1000);
        assert_eq!(svc.tol, 1e-4);
        assert_eq!(svc.seed, 0);
        assert!(matches!(svc.kernel, SvmKernel::Rbf { gamma } if (gamma - 1.0).abs() < 1e-10));
    }

    #[test]
    fn test_decision_function_empty_input() {
        let (x, y) = well_separated_data();
        let svc = Svc::new().with_kernel(SvmKernel::Linear).with_c(10.0);
        let fitted: FittedSvc<f64> = svc.fit(&x, &y).unwrap();

        let x_empty = Array2::<f64>::zeros((0, 2));
        let result = fitted.decision_function(&x_empty);
        assert!(result.is_err());
        match result {
            Err(RustMlError::EmptyInput(_)) => {}
            other => panic!("expected EmptyInput error, got {:?}", other),
        }
    }
}
