use anofox_ml_core::{Fit, Float, Predict, Result, RustMlError};
use ndarray::{Array1, Array2, Axis};
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::activation::Activation;
use crate::network::{adam_update, backward_pass, forward_pass, sgd_update, DenseLayer};
use crate::solver::{AdamState, Solver};
use crate::utils::{mse_loss, select_rows, shuffle_indices};

/// Multi-layer perceptron regressor (unfitted).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MlpRegressor {
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

impl Default for MlpRegressor {
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

/// Fitted multi-layer perceptron regressor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedMlpRegressor<F: Float> {
    layers: Vec<DenseLayer<F>>,
    activation: Activation,
    n_features: usize,
}

impl<F: Float> Fit<F> for MlpRegressor {
    type Fitted = FittedMlpRegressor<F>;

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
        let n_outputs = 1;

        // Build layer sizes: [n_features, hidden..., 1]
        let mut layer_sizes = Vec::with_capacity(self.hidden_layer_sizes.len() + 2);
        layer_sizes.push(n_features);
        layer_sizes.extend_from_slice(&self.hidden_layer_sizes);
        layer_sizes.push(n_outputs);

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

        // Reshape y to (n_samples, 1)
        let y_2d = y.clone().insert_axis(Axis(1));
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
                    (x, &y_2d)
                } else {
                    x_batch_owned = select_rows(x, chunk);
                    y_batch_owned = select_rows(&y_2d, chunk);
                    (&x_batch_owned, &y_batch_owned)
                };

                // Forward
                let (output, caches) = forward_pass(&layers, self.activation, x_b);

                // Loss
                epoch_loss += mse_loss(&output, y_b);
                n_batches += 1;

                // Backward: MSE gradient: (y_pred - y_true) / n_outputs
                let output_delta = (&output - y_b)
                    * (F::from_f64(2.0).unwrap() / F::from_usize(output.nrows()).unwrap());
                let gradients =
                    backward_pass(&layers, &caches, self.activation, output_delta, alpha);

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

        Ok(FittedMlpRegressor {
            layers,
            activation: self.activation,
            n_features,
        })
    }
}

impl<F: Float> Predict<F> for FittedMlpRegressor<F> {
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }

        let (output, _) = forward_pass(&self.layers, self.activation, x);
        // output is (n_samples, 1), flatten to 1D
        Ok(output.column(0).to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn simple_linear_regression() {
        // y = 2*x + 1
        let x = array![
            [0.0],
            [0.25],
            [0.5],
            [0.75],
            [1.0],
            [1.25],
            [1.5],
            [1.75],
            [2.0],
            [2.25],
        ];
        let y = array![1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5];

        let mlp = MlpRegressor {
            hidden_layer_sizes: vec![20],
            activation: Activation::Relu,
            solver: Solver::Adam,
            learning_rate: 0.01,
            max_iter: 500,
            tol: 1e-7,
            seed: 42,
            batch_size: None,
            alpha: 0.0,
        };

        let fitted: FittedMlpRegressor<f64> = mlp.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(*p, *t, epsilon = 0.5);
        }
    }

    #[test]
    fn quadratic_regression() {
        // y = x^2
        let x = array![
            [0.0],
            [0.5],
            [1.0],
            [1.5],
            [2.0],
            [2.5],
            [3.0],
            [3.5],
            [4.0],
            [4.5],
        ];
        let y = array![0.0, 0.25, 1.0, 2.25, 4.0, 6.25, 9.0, 12.25, 16.0, 20.25];

        let mlp = MlpRegressor {
            hidden_layer_sizes: vec![20, 10],
            learning_rate: 0.005,
            max_iter: 1000,
            tol: 1e-8,
            seed: 123,
            batch_size: None,
            alpha: 0.0,
            ..Default::default()
        };

        let fitted: FittedMlpRegressor<f64> = mlp.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        // Should get a reasonable approximation
        let mse: f64 = preds
            .iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / y.len() as f64;
        assert!(mse < 5.0, "MSE too high: {mse}");
    }

    #[test]
    fn shape_mismatch_error() {
        let x = array![[1.0, 2.0]];
        let y = array![0.0, 1.0];

        let mlp = MlpRegressor::default();
        let result: std::result::Result<FittedMlpRegressor<f64>, _> = mlp.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn empty_input_error() {
        let x: Array2<f64> = Array2::zeros((0, 2));
        let y: Array1<f64> = Array1::zeros(0);

        let mlp = MlpRegressor::default();
        let result: std::result::Result<FittedMlpRegressor<f64>, _> = mlp.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn predict_wrong_features_error() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 2.0];

        let mlp = MlpRegressor {
            hidden_layer_sizes: vec![5],
            max_iter: 5,
            seed: 0,
            ..Default::default()
        };
        let fitted: FittedMlpRegressor<f64> = mlp.fit(&x, &y).unwrap();

        let x_bad = array![[1.0]];
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn reproducibility() {
        let x = array![[0.0], [1.0], [2.0], [3.0]];
        let y = array![0.0, 1.0, 4.0, 9.0];

        let mlp = MlpRegressor {
            hidden_layer_sizes: vec![10],
            max_iter: 50,
            seed: 42,
            batch_size: None,
            ..Default::default()
        };

        let fitted1: FittedMlpRegressor<f64> = mlp.fit(&x, &y).unwrap();
        let fitted2: FittedMlpRegressor<f64> = mlp.fit(&x, &y).unwrap();

        let p1 = fitted1.predict(&x).unwrap();
        let p2 = fitted2.predict(&x).unwrap();

        for (a, b) in p1.iter().zip(p2.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-12);
        }
    }
}
