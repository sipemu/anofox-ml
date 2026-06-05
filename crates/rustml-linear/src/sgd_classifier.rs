//! SGD-based linear classifier.
//!
//! Supports hinge (linear SVM), log_loss (logistic regression), and
//! modified_huber loss functions, trained with stochastic gradient descent.
//! Multi-class uses a one-vs-rest strategy.

use crate::sgd_common::{compute_lr, penalty_gradient, LearningRate, Penalty};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rustml_core::{Fit, Float, Predict, Result, RustMlError};

/// Loss function for SGD classifier.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ClassifierLoss {
    /// Hinge loss (linear SVM). Not differentiable at 1, but subgradient works.
    Hinge,
    /// Log loss (logistic regression).
    Log,
    /// Modified Huber: smooth hinge that gives probability estimates.
    ModifiedHuber,
    /// Perceptron loss: hinge with zero threshold.
    Perceptron,
}

impl Default for ClassifierLoss {
    fn default() -> Self {
        ClassifierLoss::Hinge
    }
}

/// Stochastic Gradient Descent classifier.
///
/// Linear classifier trained via SGD, supporting multiple loss functions.
/// For multi-class problems, uses one-vs-rest decomposition.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SgdClassifier {
    pub loss: ClassifierLoss,
    pub penalty: Penalty,
    pub alpha: f64,
    pub l1_ratio: f64,
    pub max_iter: usize,
    pub tol: f64,
    pub eta0: f64,
    pub power_t: f64,
    pub learning_rate: LearningRate,
    pub shuffle: bool,
    pub seed: u64,
}

impl SgdClassifier {
    pub fn new() -> Self {
        Self {
            loss: ClassifierLoss::Hinge,
            penalty: Penalty::L2,
            alpha: 0.0001,
            l1_ratio: 0.15,
            max_iter: 1000,
            tol: 1e-3,
            eta0: 0.01,
            power_t: 0.5,
            learning_rate: LearningRate::Optimal,
            shuffle: true,
            seed: 0,
        }
    }

    pub fn with_loss(mut self, loss: ClassifierLoss) -> Self {
        self.loss = loss;
        self
    }
    pub fn with_penalty(mut self, penalty: Penalty) -> Self {
        self.penalty = penalty;
        self
    }
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }
    pub fn with_l1_ratio(mut self, l1_ratio: f64) -> Self {
        self.l1_ratio = l1_ratio;
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
    pub fn with_eta0(mut self, eta0: f64) -> Self {
        self.eta0 = eta0;
        self
    }
    pub fn with_power_t(mut self, power_t: f64) -> Self {
        self.power_t = power_t;
        self
    }
    pub fn with_learning_rate(mut self, lr: LearningRate) -> Self {
        self.learning_rate = lr;
        self
    }
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl Default for SgdClassifier {
    fn default() -> Self {
        Self::new()
    }
}

/// A fitted SGD linear classifier.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedSgdClassifier<F: Float> {
    /// One weight vector per class (OvR). Shape: (n_classes, n_features).
    weights: Vec<Array1<F>>,
    /// One bias per class.
    biases: Vec<F>,
    /// Sorted unique class labels.
    classes: Vec<F>,
    n_features: usize,
}

impl<F: Float> FittedSgdClassifier<F> {
    /// Return the unique class labels.
    pub fn classes(&self) -> &[F] {
        &self.classes
    }

    /// Return weight vectors (one per class in OvR).
    pub fn weights(&self) -> &[Array1<F>] {
        &self.weights
    }

    /// Return biases (one per class in OvR).
    pub fn biases(&self) -> &[F] {
        &self.biases
    }

    /// Compute raw decision function values for each class.
    pub fn decision_function(&self, x: &Array2<F>) -> Result<Array2<F>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        let n = x.nrows();
        let k = self.classes.len();
        let mut scores = Array2::zeros((n, k));
        for (c, (w, &b)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            for i in 0..n {
                let mut s = b;
                for j in 0..self.n_features {
                    s = s + x[[i, j]] * w[j];
                }
                scores[[i, c]] = s;
            }
        }
        Ok(scores)
    }
}

/// Train a single binary OvR classifier via SGD.
/// Returns (weights, bias).
fn train_binary_sgd(
    x: &Array2<f64>,
    y_binary: &[f64], // +1 or -1
    loss: ClassifierLoss,
    penalty: Penalty,
    alpha: f64,
    l1_ratio: f64,
    max_iter: usize,
    tol: f64,
    eta0: f64,
    power_t: f64,
    learning_rate: LearningRate,
    shuffle: bool,
    seed: u64,
) -> (Array1<f64>, f64) {
    let n = x.nrows();
    let p = x.ncols();
    let mut w = Array1::zeros(p);
    let mut b = 0.0;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..n).collect();
    let mut t: usize = 1;

    for _epoch in 0..max_iter {
        if shuffle {
            indices.shuffle(&mut rng);
        }

        let mut total_loss = 0.0;

        for &i in &indices {
            let eta = compute_lr(learning_rate, eta0, alpha, t, power_t);
            t += 1;

            // Compute decision: z = w · x_i + b
            let mut z = b;
            for j in 0..p {
                z += w[j] * x[[i, j]];
            }
            let yi = y_binary[i];
            let margin = yi * z;

            // Compute loss gradient w.r.t. z
            let dloss = match loss {
                ClassifierLoss::Hinge => {
                    if margin < 1.0 {
                        total_loss += (1.0 - margin).max(0.0);
                        -yi
                    } else {
                        0.0
                    }
                }
                ClassifierLoss::Log => {
                    let sig = 1.0 / (1.0 + (-z).exp());
                    let label01 = (yi + 1.0) / 2.0; // convert {-1,1} to {0,1}
                    total_loss += -(label01 * sig.ln() + (1.0 - label01) * (1.0 - sig).ln());
                    sig - label01
                }
                ClassifierLoss::ModifiedHuber => {
                    if margin >= 1.0 {
                        0.0
                    } else if margin >= -1.0 {
                        total_loss += (1.0 - margin).powi(2);
                        -2.0 * yi * (1.0 - margin)
                    } else {
                        total_loss += -4.0 * margin;
                        -4.0 * yi
                    }
                }
                ClassifierLoss::Perceptron => {
                    if margin <= 0.0 {
                        total_loss += -margin;
                        -yi
                    } else {
                        0.0
                    }
                }
            };

            // Update weights: w -= eta * (dloss * x_i + penalty_grad)
            if dloss != 0.0 {
                for j in 0..p {
                    w[j] -= eta * (dloss * x[[i, j]] + penalty_gradient(w[j], alpha, penalty, l1_ratio));
                }
                b -= eta * dloss;
            } else {
                // Still apply regularization
                for j in 0..p {
                    w[j] -= eta * penalty_gradient(w[j], alpha, penalty, l1_ratio);
                }
            }
        }

        // Check convergence
        let avg_loss = total_loss / n as f64;
        if avg_loss < tol {
            break;
        }
    }

    (w, b)
}

impl Fit<f64> for SgdClassifier {
    type Fitted = FittedSgdClassifier<f64>;

    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        if x.nrows() != y.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} rows but y has {} elements",
                x.nrows(),
                y.len()
            )));
        }
        if x.is_empty() {
            return Err(RustMlError::EmptyInput("training data is empty".into()));
        }
        if self.alpha < 0.0 {
            return Err(RustMlError::InvalidParameter(
                "alpha must be non-negative".into(),
            ));
        }

        // Collect unique classes
        let mut classes: Vec<f64> = y.iter().copied().collect();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        classes.dedup();

        if classes.len() < 2 {
            return Err(RustMlError::InvalidParameter(
                "need at least 2 classes".into(),
            ));
        }

        let n_features = x.ncols();

        // One-vs-rest: train one binary classifier per class
        let mut weights = Vec::with_capacity(classes.len());
        let mut biases = Vec::with_capacity(classes.len());

        for (c_idx, &cls) in classes.iter().enumerate() {
            let y_binary: Vec<f64> = y.iter().map(|&v| if v == cls { 1.0 } else { -1.0 }).collect();

            let (w, b) = train_binary_sgd(
                x,
                &y_binary,
                self.loss,
                self.penalty,
                self.alpha,
                self.l1_ratio,
                self.max_iter,
                self.tol,
                self.eta0,
                self.power_t,
                self.learning_rate,
                self.shuffle,
                self.seed.wrapping_add(c_idx as u64),
            );
            weights.push(w);
            biases.push(b);
        }

        Ok(FittedSgdClassifier {
            weights,
            biases,
            classes,
            n_features,
        })
    }
}

impl Predict<f64> for FittedSgdClassifier<f64> {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let scores = self.decision_function(x)?;
        let n = x.nrows();
        let mut preds = Array1::zeros(n);

        for i in 0..n {
            let mut best_c = 0;
            let mut best_s = scores[[i, 0]];
            for c in 1..self.classes.len() {
                if scores[[i, c]] > best_s {
                    best_s = scores[[i, c]];
                    best_c = c;
                }
            }
            preds[i] = self.classes[best_c];
        }
        Ok(preds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn make_binary_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec(
            (12, 2),
            vec![
                0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 0.0, 1.0,
                3.0, 3.0, 3.5, 3.5, 4.0, 3.0, 3.0, 4.0,
                0.5, 0.0, 0.0, 0.5,
                3.5, 3.0, 3.0, 3.5,
            ],
        )
        .unwrap();
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        (x, y)
    }

    #[test]
    fn test_sgd_classifier_hinge() {
        let (x, y) = make_binary_data();
        let clf = SgdClassifier::new()
            .with_loss(ClassifierLoss::Hinge)
            .with_max_iter(500)
            .with_alpha(0.001);
        let fitted = clf.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        let correct: usize = preds.iter().zip(y.iter()).filter(|(&p, &t)| p == t).count();
        assert!(correct >= 8, "should classify most points correctly, got {}/12", correct);
    }

    #[test]
    fn test_sgd_classifier_log_loss() {
        let (x, y) = make_binary_data();
        let clf = SgdClassifier::new()
            .with_loss(ClassifierLoss::Log)
            .with_max_iter(500)
            .with_alpha(0.001);
        let fitted = clf.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        let correct: usize = preds.iter().zip(y.iter()).filter(|(&p, &t)| p == t).count();
        assert!(correct >= 8, "should classify most points correctly, got {}/12", correct);
    }

    #[test]
    fn test_sgd_classifier_modified_huber() {
        let (x, y) = make_binary_data();
        let clf = SgdClassifier::new()
            .with_loss(ClassifierLoss::ModifiedHuber)
            .with_max_iter(500);
        let fitted = clf.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(p == 0.0 || p == 1.0);
        }
    }

    #[test]
    fn test_sgd_classifier_perceptron() {
        let (x, y) = make_binary_data();
        let clf = SgdClassifier::new()
            .with_loss(ClassifierLoss::Perceptron)
            .with_penalty(Penalty::None)
            .with_max_iter(500);
        let fitted = clf.fit(&x, &y).unwrap();

        assert_eq!(fitted.classes().len(), 2);
    }

    #[test]
    fn test_sgd_classifier_multiclass() {
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.5, 0.5, 0.0, 0.5,
                3.0, 0.0, 3.5, 0.5, 3.0, 0.5,
                1.5, 3.0, 1.0, 3.5, 2.0, 3.0,
            ],
        )
        .unwrap();
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];

        let clf = SgdClassifier::new()
            .with_loss(ClassifierLoss::Hinge)
            .with_max_iter(1000)
            .with_alpha(0.001);
        let fitted = clf.fit(&x, &y).unwrap();

        assert_eq!(fitted.classes().len(), 3);
        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(p == 0.0 || p == 1.0 || p == 2.0);
        }
    }

    #[test]
    fn test_sgd_classifier_decision_function() {
        let (x, y) = make_binary_data();
        let clf = SgdClassifier::new().with_max_iter(100);
        let fitted = clf.fit(&x, &y).unwrap();

        let scores = fitted.decision_function(&x).unwrap();
        assert_eq!(scores.nrows(), 12);
        assert_eq!(scores.ncols(), 2);
    }

    #[test]
    fn test_sgd_classifier_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 2), vec![0.0; 6]).unwrap();
        let y = array![0.0, 1.0];
        assert!(SgdClassifier::new().fit(&x, &y).is_err());
    }

    #[test]
    fn test_sgd_classifier_empty_input() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<f64>::zeros(0);
        assert!(SgdClassifier::new().fit(&x, &y).is_err());
    }

    #[test]
    fn test_sgd_classifier_single_class() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![0.0, 0.0, 0.0];
        assert!(SgdClassifier::new().fit(&x, &y).is_err());
    }

    #[test]
    fn test_sgd_classifier_elastic_net() {
        let (x, y) = make_binary_data();
        let clf = SgdClassifier::new()
            .with_penalty(Penalty::ElasticNet)
            .with_l1_ratio(0.5)
            .with_max_iter(500);
        let fitted = clf.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 12);
    }

    #[test]
    fn test_sgd_classifier_constant_lr() {
        let (x, y) = make_binary_data();
        let clf = SgdClassifier::new()
            .with_learning_rate(LearningRate::Constant)
            .with_eta0(0.01)
            .with_max_iter(500);
        let fitted = clf.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 12);
    }
}

impl rustml_core::ClassifierScore<f64> for FittedSgdClassifier<f64> {}
