use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rustml_core::{Fit, Float, Predict, Result, RustMlError};

use crate::activation::Activation;
use crate::network::{
    adam_update, backward_pass, forward_pass, sgd_update, DenseLayer,
};
use crate::solver::{AdamState, Solver};
use crate::utils::{
    cross_entropy_loss, one_hot_encode, select_rows, shuffle_indices, softmax,
};

/// Multi-layer perceptron classifier (unfitted).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MlpClassifier {
    /// Sizes of hidden layers, e.g. `vec![100]` for one hidden layer of 100 units.
    pub hidden_layer_sizes: Vec<usize>,
    /// Activation function for hidden layers.
    pub activation: Activation,
    /// Optimization solver.
    pub solver: Solver,
    /// Learning rate.
    pub learning_rate: f64,
    /// Maximum number of epochs.
    pub max_iter: usize,
    /// Tolerance for early stopping based on loss improvement.
    pub tol: f64,
    /// Random seed for weight initialization and shuffling.
    pub seed: u64,
    /// Mini-batch size. `None` means full batch.
    pub batch_size: Option<usize>,
    /// L2 regularization strength.
    pub alpha: f64,
}

impl Default for MlpClassifier {
    fn default() -> Self {
        Self {
            hidden_layer_sizes: vec![100],
            activation: Activation::Relu,
            solver: Solver::Adam,
            learning_rate: 0.001,
            max_iter: 200,
            tol: 1e-4,
            seed: 0,
            batch_size: Some(200),
            alpha: 1e-4,
        }
    }
}

/// Fitted multi-layer perceptron classifier.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedMlpClassifier<F: Float> {
    layers: Vec<DenseLayer<F>>,
    activation: Activation,
    class_labels: Vec<F>,
    n_features: usize,
}

impl<F: Float> FittedMlpClassifier<F> {
    /// Return class probability estimates (softmax output).
    pub fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        let (logits, _) = forward_pass(&self.layers, self.activation, x);
        Ok(softmax(logits))
    }
}

impl<F: Float> Fit<F> for MlpClassifier {
    type Fitted = FittedMlpClassifier<F>;

    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Self::Fitted> {
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
        if self.hidden_layer_sizes.is_empty() {
            return Err(RustMlError::InvalidParameter(
                "hidden_layer_sizes must not be empty".into(),
            ));
        }
        for (i, &s) in self.hidden_layer_sizes.iter().enumerate() {
            if s == 0 {
                return Err(RustMlError::InvalidParameter(format!(
                    "hidden_layer_sizes[{i}] must be > 0"
                )));
            }
        }

        let n_features = x.ncols();
        let n_samples = x.nrows();

        // Discover unique class labels
        let mut class_labels: Vec<F> = {
            let mut seen = std::collections::BTreeMap::new();
            for &v in y.iter() {
                let key = v.to_f64().unwrap().to_bits();
                seen.entry(key).or_insert(v);
            }
            seen.into_values().collect()
        };
        class_labels.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n_classes = class_labels.len();

        if n_classes < 2 {
            return Err(RustMlError::InvalidParameter(
                "need at least 2 classes for classification".into(),
            ));
        }

        // Build layer sizes: [n_features, hidden..., n_classes]
        let mut layer_sizes = Vec::with_capacity(self.hidden_layer_sizes.len() + 2);
        layer_sizes.push(n_features);
        layer_sizes.extend_from_slice(&self.hidden_layer_sizes);
        layer_sizes.push(n_classes);

        let mut rng = StdRng::seed_from_u64(self.seed);

        // Initialize layers
        let mut layers: Vec<DenseLayer<F>> = Vec::with_capacity(layer_sizes.len() - 1);
        for i in 0..layer_sizes.len() - 1 {
            layers.push(DenseLayer::new(
                layer_sizes[i],
                layer_sizes[i + 1],
                self.activation,
                &mut rng,
            ));
        }

        // Adam optimizer state
        let mut w_states: Vec<AdamState<F>> = layers
            .iter()
            .map(|l| AdamState::new(l.weights.nrows(), l.weights.ncols()))
            .collect();
        let mut b_states: Vec<AdamState<F>> = layers
            .iter()
            .map(|l| AdamState::new(1, l.biases.ncols()))
            .collect();

        let lr = F::from_f64(self.learning_rate).unwrap();
        let alpha = F::from_f64(self.alpha).unwrap();
        let tol = F::from_f64(self.tol).unwrap();
        let batch_size = self.batch_size.unwrap_or(n_samples).min(n_samples);
        let full_batch = batch_size >= n_samples;

        let y_onehot = one_hot_encode(y, &class_labels);
        let mut indices: Vec<usize> = (0..n_samples).collect();
        let mut prev_loss = F::infinity();

        for _epoch in 0..self.max_iter {
            shuffle_indices(&mut indices, &mut rng);

            let mut epoch_loss = F::zero();
            let mut n_batches = 0;

            for chunk in indices.chunks(batch_size) {
                // When the batch covers the full dataset, avoid copying.
                let (x_batch_owned, y_batch_owned);
                let (x_b, y_b) = if full_batch {
                    (x, &y_onehot)
                } else {
                    x_batch_owned = select_rows(x, chunk);
                    y_batch_owned = select_rows(&y_onehot, chunk);
                    (&x_batch_owned, &y_batch_owned)
                };

                // Forward
                let (logits, caches) = forward_pass(&layers, self.activation, x_b);
                let probs = softmax(logits);

                // Loss
                epoch_loss += cross_entropy_loss(&probs, y_b);
                n_batches += 1;

                // Backward: output delta = probs - y_onehot
                let output_delta = &probs - y_b;
                let gradients = backward_pass(&layers, &caches, self.activation, output_delta, alpha);

                // Update
                match self.solver {
                    Solver::Sgd => sgd_update(&mut layers, &gradients, lr),
                    Solver::Adam => {
                        adam_update(&mut layers, &gradients, &mut w_states, &mut b_states, lr)
                    }
                }
            }

            let avg_loss = epoch_loss / F::from_usize(n_batches).unwrap();
            if (prev_loss - avg_loss).abs() < tol {
                break;
            }
            prev_loss = avg_loss;
        }

        Ok(FittedMlpClassifier {
            layers,
            activation: self.activation,
            class_labels,
            n_features,
        })
    }
}

impl<F: Float> Predict<F> for FittedMlpClassifier<F> {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        let probs = self.predict_proba(x)?;
        let mut predictions = Vec::with_capacity(x.nrows());
        for row in probs.rows() {
            let (max_idx, _) = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();
            predictions.push(self.class_labels[max_idx]);
        }
        Ok(Array1::from_vec(predictions))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn xor_classification() {
        // XOR is not linearly separable, requires hidden layer
        let x = array![
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ];
        let y = array![0.0, 1.0, 1.0, 0.0];

        let mlp = MlpClassifier {
            hidden_layer_sizes: vec![10, 10],
            activation: Activation::Relu,
            solver: Solver::Adam,
            learning_rate: 0.01,
            max_iter: 500,
            tol: 1e-6,
            seed: 42,
            batch_size: None, // full batch
            alpha: 0.0,
        };

        let fitted: FittedMlpClassifier<f64> = mlp.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(*p, *t, epsilon = 1e-10);
        }
    }

    #[test]
    fn simple_three_class() {
        let x = array![
            [0.0, 0.0], [0.1, 0.1], [0.2, 0.0],
            [5.0, 5.0], [5.1, 5.1], [5.2, 5.0],
            [10.0, 0.0], [10.1, 0.1], [10.2, 0.0],
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];

        let mlp = MlpClassifier {
            hidden_layer_sizes: vec![20],
            learning_rate: 0.01,
            max_iter: 300,
            seed: 123,
            batch_size: None,
            alpha: 0.0,
            ..Default::default()
        };

        let fitted: FittedMlpClassifier<f64> = mlp.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        // Should get most training samples correct
        let correct: usize = preds
            .iter()
            .zip(y.iter())
            .filter(|(p, t)| (**p - **t).abs() < 1e-10)
            .count();
        assert!(correct >= 7, "expected >= 7 correct, got {correct}");
    }

    #[test]
    fn predict_proba_sums_to_one() {
        let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let mlp = MlpClassifier {
            hidden_layer_sizes: vec![5],
            max_iter: 10,
            seed: 7,
            ..Default::default()
        };
        let fitted: FittedMlpClassifier<f64> = mlp.fit(&x, &y).unwrap();
        let proba = fitted.predict_proba(&x).unwrap();

        for row in proba.rows() {
            assert_abs_diff_eq!(row.sum(), 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn shape_mismatch_error() {
        let x = array![[1.0, 2.0]];
        let y = array![0.0, 1.0];

        let mlp = MlpClassifier::default();
        let result: std::result::Result<FittedMlpClassifier<f64>, _> = mlp.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn empty_input_error() {
        let x: Array2<f64> = Array2::zeros((0, 2));
        let y: Array1<f64> = Array1::zeros(0);

        let mlp = MlpClassifier::default();
        let result: std::result::Result<FittedMlpClassifier<f64>, _> = mlp.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn single_class_error() {
        let x = array![[1.0], [2.0], [3.0]];
        let y = array![0.0, 0.0, 0.0];

        let mlp = MlpClassifier::default();
        let result: std::result::Result<FittedMlpClassifier<f64>, _> = mlp.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn predict_wrong_features_error() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0];

        let mlp = MlpClassifier {
            hidden_layer_sizes: vec![5],
            max_iter: 5,
            seed: 0,
            ..Default::default()
        };
        let fitted: FittedMlpClassifier<f64> = mlp.fit(&x, &y).unwrap();

        let x_bad = array![[1.0]];
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn reproducibility() {
        let x = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let y = array![0.0, 1.0, 1.0, 0.0];

        let mlp = MlpClassifier {
            hidden_layer_sizes: vec![10],
            max_iter: 50,
            seed: 999,
            batch_size: None,
            ..Default::default()
        };

        let fitted1: FittedMlpClassifier<f64> = mlp.fit(&x, &y).unwrap();
        let fitted2: FittedMlpClassifier<f64> = mlp.fit(&x, &y).unwrap();

        let p1 = fitted1.predict_proba(&x).unwrap();
        let p2 = fitted2.predict_proba(&x).unwrap();

        for (a, b) in p1.iter().zip(p2.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-12);
        }
    }
}
