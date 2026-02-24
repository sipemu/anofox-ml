use ndarray::Array2;
use rustml_core::Float;
use serde::{Deserialize, Serialize};

use crate::activation::Activation;
use crate::solver::AdamState;

/// A single fully-connected layer: z = input @ W + b, a = activation(z).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct DenseLayer<F: Float> {
    /// Weight matrix: (n_in, n_out)
    pub weights: Array2<F>,
    /// Bias vector stored as (1, n_out)
    pub biases: Array2<F>,
}

impl<F: Float> DenseLayer<F> {
    /// Create a new layer with He initialization (good for ReLU) or Xavier.
    pub fn new(
        n_in: usize,
        n_out: usize,
        activation: Activation,
        rng: &mut impl rand::Rng,
    ) -> Self {
        let std_dev = match activation {
            Activation::Relu => (2.0 / n_in as f64).sqrt(),
            _ => (1.0 / n_in as f64).sqrt(), // Xavier
        };

        let weights = Array2::from_shape_fn((n_in, n_out), |_| {
            let u1: f64 = rng.gen_range(0.0001..1.0);
            let u2: f64 = rng.gen_range(0.0..1.0);
            let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            F::from_f64(normal * std_dev).unwrap()
        });

        let biases = Array2::zeros((1, n_out));

        Self { weights, biases }
    }

    /// Forward pass: z = input @ W + b.
    pub fn linear(&self, input: &Array2<F>) -> Array2<F> {
        input.dot(&self.weights) + &self.biases
    }
}

/// Cache for one layer during forward pass (needed for backward).
pub(crate) struct LayerCache<F: Float> {
    pub input: Array2<F>,
    pub z: Array2<F>,
}

/// Run forward pass through all layers, returning (output, caches).
pub(crate) fn forward_pass<F: Float>(
    layers: &[DenseLayer<F>],
    activation: Activation,
    x: &Array2<F>,
) -> (Array2<F>, Vec<LayerCache<F>>) {
    let mut current = x.clone();
    let mut caches = Vec::with_capacity(layers.len());

    for (i, layer) in layers.iter().enumerate() {
        let z = layer.linear(&current);
        let a = if i < layers.len() - 1 {
            // Hidden layers use the activation function
            activation.forward(&z)
        } else {
            // Output layer: raw logits (caller applies softmax / identity)
            z.clone()
        };

        caches.push(LayerCache {
            input: current,
            z,
        });
        current = a;
    }

    (current, caches)
}

/// Weight gradients for a single layer.
pub(crate) struct LayerGradients<F: Float> {
    pub dw: Array2<F>,
    pub db: Array2<F>,
}

/// Backward pass for classification (softmax + cross-entropy).
/// `output_delta` is (probs - y_onehot) for the output layer.
pub(crate) fn backward_pass<F: Float>(
    layers: &[DenseLayer<F>],
    caches: &[LayerCache<F>],
    activation: Activation,
    output_delta: Array2<F>,
    alpha: F,
) -> Vec<LayerGradients<F>> {
    let n_layers = layers.len();
    let batch_f = F::from_usize(output_delta.nrows()).unwrap();
    let mut grads = Vec::with_capacity(n_layers);
    let mut delta = output_delta;

    for i in (0..n_layers).rev() {
        let cache = &caches[i];

        // Weight gradient: (1/batch) * input^T @ delta + alpha * W
        let dw = cache.input.t().dot(&delta) / batch_f + &layers[i].weights * alpha;
        let db = delta.sum_axis(ndarray::Axis(0)).insert_axis(ndarray::Axis(0)) / batch_f;

        grads.push(LayerGradients { dw, db });

        if i > 0 {
            // Propagate delta to previous layer:
            // delta_prev = (delta @ W_i^T) * f'(z_{i-1})
            let raw_delta = delta.dot(&layers[i].weights.t());
            let act_deriv = activation.backward(&caches[i - 1].z);
            delta = raw_delta * act_deriv;
        }
    }

    grads.reverse();
    grads
}

/// Apply weight updates using SGD.
pub(crate) fn sgd_update<F: Float>(
    layers: &mut [DenseLayer<F>],
    gradients: &[LayerGradients<F>],
    lr: F,
) {
    for (layer, grad) in layers.iter_mut().zip(gradients.iter()) {
        layer.weights = &layer.weights - &grad.dw * lr;
        layer.biases = &layer.biases - &grad.db * lr;
    }
}

/// Apply weight updates using Adam.
pub(crate) fn adam_update<F: Float>(
    layers: &mut [DenseLayer<F>],
    gradients: &[LayerGradients<F>],
    weight_states: &mut [AdamState<F>],
    bias_states: &mut [AdamState<F>],
    lr: F,
) {
    for (i, (layer, grad)) in layers.iter_mut().zip(gradients.iter()).enumerate() {
        let w_delta = weight_states[i].step(&grad.dw, lr);
        layer.weights = &layer.weights + &w_delta;

        let b_delta = bias_states[i].step(&grad.db, lr);
        layer.biases = &layer.biases + &b_delta;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn dense_layer_shapes() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let layer = DenseLayer::<f64>::new(3, 5, Activation::Relu, &mut rng);
        assert_eq!(layer.weights.shape(), &[3, 5]);
        assert_eq!(layer.biases.shape(), &[1, 5]);
    }

    #[test]
    fn forward_pass_output_shape() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let layers = vec![
            DenseLayer::<f64>::new(4, 10, Activation::Relu, &mut rng),
            DenseLayer::new(10, 3, Activation::Relu, &mut rng),
        ];
        let x = Array2::<f64>::zeros((5, 4));
        let (output, caches) = forward_pass(&layers, Activation::Relu, &x);
        assert_eq!(output.shape(), &[5, 3]);
        assert_eq!(caches.len(), 2);
    }
}
