use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rustml_core::{Fit, Float, Predict, Result, RustMlError};

/// Linear Support Vector Classifier parameters (unfitted state).
///
/// Uses hinge loss with L2 regularization, solved via coordinate descent
/// (similar to sklearn's LinearSVC with liblinear). Uses the type-state
/// pattern: call [`Fit::fit`] to produce a [`FittedLinearSvc`] that can
/// make predictions.
///
/// For multi-class problems, a one-vs-rest (OvR) strategy is used
/// automatically.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LinearSvc {
    /// Regularization parameter (inverse of regularization strength).
    /// Larger values mean less regularization.
    pub c: f64,
    /// Maximum number of iterations for the solver.
    pub max_iter: usize,
    /// Tolerance for the stopping criterion.
    pub tol: f64,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl LinearSvc {
    /// Create a new `LinearSvc` with default parameters.
    pub fn new() -> Self {
        Self {
            c: 1.0,
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
        Ok(())
    }
}

impl Default for LinearSvc {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Linear SVC for a single binary classifier.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
struct BinaryLinearSvc<F: Float> {
    /// Weight vector, shape `(n_features,)`.
    weights: Array1<F>,
    /// Bias term.
    bias: F,
}

/// Fitted Linear Support Vector Classifier.
///
/// For binary problems, contains a single weight vector + bias.
/// For multi-class problems, contains one binary classifier per class
/// (one-vs-rest).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedLinearSvc<F: Float> {
    /// Unique sorted class labels.
    class_labels: Vec<F>,
    /// One binary classifier per class (OvR).
    classifiers: Vec<BinaryLinearSvc<F>>,
}

impl<F: Float> FittedLinearSvc<F> {
    /// Returns the weight matrix. For binary classification this is a single
    /// row vector; for multi-class it has one row per class.
    pub fn weights(&self) -> Array2<F> {
        let n_features = self.classifiers[0].weights.len();
        let n_classifiers = self.classifiers.len();
        let mut w = Array2::zeros((n_classifiers, n_features));
        for (i, clf) in self.classifiers.iter().enumerate() {
            w.row_mut(i).assign(&clf.weights);
        }
        w
    }

    /// Returns the bias terms. One per binary classifier.
    pub fn bias(&self) -> Array1<F> {
        Array1::from_vec(self.classifiers.iter().map(|c| c.bias).collect())
    }

    /// Returns the unique sorted class labels.
    pub fn class_labels(&self) -> &[F] {
        &self.class_labels
    }

    /// Compute raw decision function scores for each sample.
    ///
    /// Returns an array of shape `(n_samples,)` for binary classification
    /// (positive = class 1, negative = class 0) or `(n_samples, n_classes)`
    /// scores for multi-class. For consistency, this always returns a 2D array
    /// with shape `(n_samples, n_classifiers)`.
    pub fn decision_function(&self, x: &Array2<F>) -> Result<Array2<F>> {
        if x.is_empty() {
            return Err(RustMlError::EmptyInput(
                "prediction input must not be empty".into(),
            ));
        }
        let n_features = self.classifiers[0].weights.len();
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
                scores[[i, ci]] = sample.dot(&clf.weights) + clf.bias;
            }
        }

        Ok(scores)
    }
}

impl<F: Float> Predict<F> for FittedLinearSvc<F> {
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

/// Clamp `val` to the interval `[zero, c]`.
#[inline]
fn clamp_alpha<F: Float>(val: F, zero: F, c: F) -> F {
    if val < zero {
        zero
    } else if val > c {
        c
    } else {
        val
    }
}

/// Perform a single coordinate descent update for sample `i`.
///
/// Returns `Some((new_alpha, delta))` when an update is possible,
/// or `None` when the denominator is near-zero and the step must be
/// skipped.
#[inline]
fn coordinate_descent_step<F: Float>(
    old_alpha: F,
    xi: ndarray::ArrayView1<'_, F>,
    yi: F,
    w: &Array1<F>,
    bias: F,
    sq_norm: F,
    c: F,
) -> Option<(F, F)> {
    let one = F::one();
    let zero = F::zero();

    let denom = sq_norm + one / c;
    if denom.abs() < F::from_f64(1e-15).unwrap() {
        return None;
    }

    let prediction = xi.dot(w) + bias;
    let new_alpha_unclamped = old_alpha + (one - yi * prediction) / denom;
    let new_alpha = clamp_alpha(new_alpha_unclamped, zero, c);
    let delta = new_alpha - old_alpha;

    Some((new_alpha, delta))
}

/// Apply one coordinate descent update for sample `i`.
///
/// Computes the step, clamps alpha, and updates weight vector + bias
/// in-place. Returns the absolute change in alpha (zero if the step
/// was skipped).
#[inline]
#[allow(clippy::too_many_arguments)]
fn apply_cd_update<F: Float>(
    i: usize,
    alpha: &mut [F],
    w: &mut Array1<F>,
    bias: &mut F,
    x_row: ndarray::ArrayView1<'_, F>,
    y_i: F,
    sq_norm: F,
    c: F,
) -> F {
    let Some((new_alpha, delta)) =
        coordinate_descent_step(alpha[i], x_row, y_i, w, *bias, sq_norm, c)
    else {
        return F::zero();
    };

    let abs_delta = delta.abs();
    if abs_delta > F::from_f64(1e-15).unwrap() {
        alpha[i] = new_alpha;
        let scaled = &x_row * (delta * y_i);
        *w += &scaled.to_owned();
        *bias += delta * y_i;
    }

    abs_delta
}

/// Train a single binary linear SVC using coordinate descent on
/// the dual formulation of hinge loss + L2 regularization.
///
/// Labels must be +1/-1 encoded.
fn fit_binary_linear_svc<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    c: F,
    max_iter: usize,
    tol: F,
    seed: u64,
) -> BinaryLinearSvc<F> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    let mut alpha = vec![F::zero(); n_samples];
    let mut w = Array1::<F>::zeros(n_features);
    let mut bias = F::zero();

    let sq_norms: Vec<F> = x
        .rows()
        .into_iter()
        .map(|row| row.dot(&row))
        .collect();

    let mut rng = StdRng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..n_samples).collect();

    for _ in 0..max_iter {
        indices.shuffle(&mut rng);

        let max_change = indices.iter().fold(F::zero(), |mc, &i| {
            let change = apply_cd_update(i, &mut alpha, &mut w, &mut bias, x.row(i), y[i], sq_norms[i], c);
            if change > mc { change } else { mc }
        });

        if max_change < tol {
            break;
        }
    }

    BinaryLinearSvc { weights: w, bias }
}

impl<F: Float> Fit<F> for LinearSvc {
    type Fitted = FittedLinearSvc<F>;

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
            // Binary classification: encode as +1/-1 where
            // class_labels[1] -> +1, class_labels[0] -> -1
            let y_binary = y.mapv(|yi| {
                if (yi - class_labels[1]).abs() < eps {
                    F::one()
                } else {
                    -F::one()
                }
            });

            let clf = fit_binary_linear_svc(x, &y_binary, c, self.max_iter, tol, self.seed);
            Ok(FittedLinearSvc {
                class_labels,
                classifiers: vec![clf],
            })
        } else {
            // Multi-class: one-vs-rest
            let mut classifiers = Vec::with_capacity(class_labels.len());

            for (ci, &label) in class_labels.iter().enumerate() {
                // +1 for this class, -1 for all others
                let y_binary = y.mapv(|yi| {
                    if (yi - label).abs() < eps {
                        F::one()
                    } else {
                        -F::one()
                    }
                });

                let seed_offset = ci as u64;
                let clf = fit_binary_linear_svc(
                    x,
                    &y_binary,
                    c,
                    self.max_iter,
                    tol,
                    self.seed.wrapping_add(seed_offset),
                );
                classifiers.push(clf);
            }

            Ok(FittedLinearSvc {
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

    fn linearly_separable_data() -> (Array2<f64>, Array1<f64>) {
        let x = array![
            [0.0, 0.0],
            [0.5, 0.1],
            [0.1, 0.5],
            [0.2, 0.3],
            [3.0, 3.0],
            [3.5, 3.1],
            [3.1, 3.5],
            [3.2, 3.3]
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        (x, y)
    }

    #[test]
    fn test_binary_classification_f64() {
        let (x, y) = linearly_separable_data();
        let svc = LinearSvc::new().with_c(1.0).with_max_iter(2000);
        let fitted: FittedLinearSvc<f64> = svc.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for i in 0..4 {
            assert_abs_diff_eq!(preds[i], 0.0, epsilon = 1e-10);
        }
        for i in 4..8 {
            assert_abs_diff_eq!(preds[i], 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_binary_classification_f32() {
        let x: Array2<f32> = array![
            [0.0, 0.0],
            [0.5, 0.1],
            [0.1, 0.5],
            [0.2, 0.3],
            [3.0, 3.0],
            [3.5, 3.1],
            [3.1, 3.5],
            [3.2, 3.3]
        ];
        let y: Array1<f32> = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let svc = LinearSvc::new().with_c(1.0).with_max_iter(2000);
        let fitted: FittedLinearSvc<f32> = svc.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for i in 0..4 {
            assert_abs_diff_eq!(preds[i], 0.0_f32, epsilon = 1e-5);
        }
        for i in 4..8 {
            assert_abs_diff_eq!(preds[i], 1.0_f32, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_decision_function() {
        let (x, y) = linearly_separable_data();
        let svc = LinearSvc::new().with_max_iter(2000);
        let fitted: FittedLinearSvc<f64> = svc.fit(&x, &y).unwrap();

        let scores = fitted.decision_function(&x).unwrap();
        assert_eq!(scores.nrows(), x.nrows());
        assert_eq!(scores.ncols(), 1); // binary => 1 classifier

        // Class 0 samples should have negative scores; class 1 should have positive.
        for i in 0..4 {
            assert!(scores[[i, 0]] < 0.0, "expected negative score for class 0");
        }
        for i in 4..8 {
            assert!(scores[[i, 0]] > 0.0, "expected positive score for class 1");
        }
    }

    #[test]
    fn test_weights_and_bias() {
        let (x, y) = linearly_separable_data();
        let svc = LinearSvc::new();
        let fitted: FittedLinearSvc<f64> = svc.fit(&x, &y).unwrap();

        let w = fitted.weights();
        assert_eq!(w.nrows(), 1); // binary => 1 classifier
        assert_eq!(w.ncols(), 2); // 2 features

        let b = fitted.bias();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_multiclass_classification() {
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [3.0, 0.0],
            [3.1, 0.1],
            [3.2, 0.0],
            [0.0, 3.0],
            [0.1, 3.1],
            [0.0, 3.2]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];

        let svc = LinearSvc::new().with_c(10.0).with_max_iter(5000);
        let fitted: FittedLinearSvc<f64> = svc.fit(&x, &y).unwrap();

        assert_eq!(fitted.class_labels(), &[0.0, 1.0, 2.0]);

        // Weights should have 3 rows (one per class) for OvR
        let w = fitted.weights();
        assert_eq!(w.nrows(), 3);
        assert_eq!(w.ncols(), 2);

        // Predict training data (well-separated, should be correct)
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

        let svc = LinearSvc::new();
        let result: Result<FittedLinearSvc<f64>> = svc.fit(&x, &y);
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

        let svc = LinearSvc::new();
        let result: Result<FittedLinearSvc<f64>> = svc.fit(&x, &y);
        assert!(result.is_err());
        match result {
            Err(RustMlError::ShapeMismatch(_)) => {}
            other => panic!("expected ShapeMismatch error, got {:?}", other),
        }
    }

    #[test]
    fn test_shape_mismatch_predict() {
        let (x, y) = linearly_separable_data();
        let svc = LinearSvc::new();
        let fitted: FittedLinearSvc<f64> = svc.fit(&x, &y).unwrap();

        let x_bad = array![[1.0, 2.0, 3.0]]; // 3 features instead of 2
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

        let svc = LinearSvc::new().with_c(-1.0);
        let result: Result<FittedLinearSvc<f64>> = svc.fit(&x, &y);
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

        let svc = LinearSvc::new();
        let result: Result<FittedLinearSvc<f64>> = svc.fit(&x, &y);
        assert!(result.is_err());
        match result {
            Err(RustMlError::InvalidParameter(_)) => {}
            other => panic!("expected InvalidParameter error, got {:?}", other),
        }
    }

    #[test]
    fn test_decision_function_empty_input() {
        let (x, y) = linearly_separable_data();
        let svc = LinearSvc::new();
        let fitted: FittedLinearSvc<f64> = svc.fit(&x, &y).unwrap();

        let x_empty = Array2::<f64>::zeros((0, 2));
        let result = fitted.decision_function(&x_empty);
        assert!(result.is_err());
        match result {
            Err(RustMlError::EmptyInput(_)) => {}
            other => panic!("expected EmptyInput error, got {:?}", other),
        }
    }

    #[test]
    fn test_builder_pattern() {
        let svc = LinearSvc::new()
            .with_c(0.5)
            .with_max_iter(500)
            .with_tol(1e-3)
            .with_seed(42);
        assert_eq!(svc.c, 0.5);
        assert_eq!(svc.max_iter, 500);
        assert_eq!(svc.tol, 1e-3);
        assert_eq!(svc.seed, 42);
    }

    #[test]
    fn test_default() {
        let svc = LinearSvc::default();
        assert_eq!(svc.c, 1.0);
        assert_eq!(svc.max_iter, 1000);
        assert_eq!(svc.tol, 1e-4);
        assert_eq!(svc.seed, 0);
    }

    #[test]
    fn test_reproducibility_with_seed() {
        let (x, y) = linearly_separable_data();
        let svc = LinearSvc::new().with_seed(42);

        let fitted1: FittedLinearSvc<f64> = svc.fit(&x, &y).unwrap();
        let fitted2: FittedLinearSvc<f64> = svc.fit(&x, &y).unwrap();

        let w1 = fitted1.weights();
        let w2 = fitted2.weights();
        for i in 0..w1.nrows() {
            for j in 0..w1.ncols() {
                assert_abs_diff_eq!(w1[[i, j]], w2[[i, j]], epsilon = 1e-12);
            }
        }
    }

    mod prop_tests {
        use super::*;
        use proptest::prelude::*;
        use rustml_core::Fit;
        use rustml_core::Predict;

        /// Generate well-separated 2D binary classification data using
        /// a deterministic hash-based noise generator.
        fn make_well_separated_binary(
            n_per_class: usize,
            seed: u64,
        ) -> (Array2<f64>, Array1<f64>) {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let n = n_per_class * 2;
            let mut x_data = Vec::with_capacity(n * 2);
            let mut y_data = Vec::with_capacity(n);

            for i in 0..n {
                let class = if i < n_per_class { 0.0 } else { 1.0 };
                let offset = if class == 0.0 { -5.0 } else { 5.0 };

                let mut h = DefaultHasher::new();
                seed.hash(&mut h);
                (i as u64).hash(&mut h);
                let bits = h.finish();
                let noise = (bits as f64 / u64::MAX as f64) * 2.0 - 1.0; // [-1, 1]
                x_data.push(offset + noise);

                let mut h2 = DefaultHasher::new();
                seed.hash(&mut h2);
                (i as u64).hash(&mut h2);
                1u64.hash(&mut h2);
                let bits2 = h2.finish();
                let noise2 = (bits2 as f64 / u64::MAX as f64) * 2.0 - 1.0;
                x_data.push(noise2);

                y_data.push(class);
            }

            let x = Array2::from_shape_vec((n, 2), x_data).unwrap();
            let y = Array1::from_vec(y_data);
            (x, y)
        }

        proptest! {
            /// For well-separated data, every prediction must be a valid
            /// class label from the training set.
            #[test]
            fn predictions_are_valid_class_labels(
                n_per_class in 5_usize..50,
                seed in 0_u64..10_000,
            ) {
                let (x, y) = make_well_separated_binary(n_per_class, seed);

                let svc = LinearSvc::new()
                    .with_c(1.0)
                    .with_max_iter(1000)
                    .with_seed(seed);

                let fitted: FittedLinearSvc<f64> = svc.fit(&x, &y).unwrap();
                let preds = fitted.predict(&x).unwrap();
                let labels = fitted.class_labels();

                for &p in preds.iter() {
                    prop_assert!(
                        labels.contains(&p),
                        "prediction {} is not a valid class label (valid: {:?})",
                        p,
                        labels,
                    );
                }
            }
        }
    }
}
