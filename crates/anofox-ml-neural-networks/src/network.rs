use anofox_ml_core::Float;
use ndarray::linalg::general_mat_mul;
use ndarray::{Array2, ArrayView2, AsArray, Axis, Ix2, Zip};
use serde::{Deserialize, Serialize};

use crate::activation::Activation;

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
    /// Create a new layer with sklearn's Glorot-uniform initialisation:
    /// weights and biases `~ U(-b, b)` with `b = √(6 / (n_in + n_out))`
    /// (factor 2 instead of 6 for the logistic/sigmoid activation).
    pub fn new(
        n_in: usize,
        n_out: usize,
        activation: Activation,
        rng: &mut impl rand::Rng,
    ) -> Self {
        let factor = if activation == Activation::Sigmoid {
            2.0
        } else {
            6.0
        };
        let bound = (factor / (n_in + n_out) as f64).sqrt();
        let weights = Array2::from_shape_fn((n_in, n_out), |_| {
            F::from_f64(rng.gen_range(-bound..bound)).unwrap()
        });
        let biases = Array2::from_shape_fn((1, n_out), |_| {
            F::from_f64(rng.gen_range(-bound..bound)).unwrap()
        });
        Self { weights, biases }
    }

    /// Forward pass: z = input @ W + b.
    pub fn linear<'a>(&self, input: impl AsArray<'a, F, Ix2>) -> Array2<F> {
        let input: ArrayView2<F> = input.into();
        input.dot(&self.weights) + &self.biases
    }
}

/// Output-layer activation (sklearn `out_activation_`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OutputKind {
    /// Regression, squared loss.
    Identity,
    /// Binary classification (single unit), log loss.
    Logistic,
    /// Multi-class classification, cross-entropy.
    Softmax,
}

/// Forward pass. Returns the cached hidden activations (one per hidden
/// layer, post-activation) and the output-layer logits. `x` is borrowed,
/// never copied.
pub(crate) fn forward<F: Float>(
    layers: &[DenseLayer<F>],
    activation: Activation,
    x: ArrayView2<F>,
) -> (Vec<Array2<F>>, Array2<F>) {
    let last = layers.len() - 1;
    let mut hidden: Vec<Array2<F>> = Vec::with_capacity(last);
    for (i, layer) in layers.iter().enumerate() {
        let mut z = if i == 0 {
            x.dot(&layer.weights)
        } else {
            hidden[i - 1].dot(&layer.weights)
        };
        z += &layer.biases;
        if i < last {
            activation.forward_inplace(&mut z);
            hidden.push(z);
        } else {
            return (hidden, z);
        }
    }
    unreachable!("layers must not be empty")
}

/// Logits → network output (identity / logistic / softmax).
pub(crate) fn output_activation<F: Float>(kind: OutputKind, mut z: Array2<F>) -> Array2<F> {
    match kind {
        OutputKind::Identity => {}
        OutputKind::Logistic => z.mapv_inplace(|v| F::one() / (F::one() + (-v).exp())),
        OutputKind::Softmax => {
            for mut row in z.rows_mut() {
                let max = row.fold(F::neg_infinity(), |a, &b| if b > a { b } else { a });
                row.mapv_inplace(|v| (v - max).exp());
                let s = row.sum();
                row.mapv_inplace(|v| v / s);
            }
        }
    }
    z
}

#[inline]
fn softplus(z: f64) -> f64 {
    z.max(0.0) + (-z.abs()).exp().ln_1p()
}

/// Data loss (without L2) from the logits, summed over samples (divide by
/// the batch size for the mean):
///
/// * identity — `½ Σ_j (z_j - y_j)²`
/// * logistic — `y softplus(-z) + (1-y) softplus(z)`
/// * softmax  — `logsumexp(z) - Σ_k y_k z_k` (log-softmax, no clipping)
pub(crate) fn data_loss_sum<F: Float>(
    kind: OutputKind,
    logits: &Array2<F>,
    y: &ArrayView2<F>,
) -> f64 {
    let mut total = 0.0;
    match kind {
        OutputKind::Identity => {
            Zip::from(logits).and(y).for_each(|&z, &t| {
                let d = (z - t).to_f64().unwrap();
                total += 0.5 * d * d;
            });
        }
        OutputKind::Logistic => {
            Zip::from(logits).and(y).for_each(|&z, &t| {
                let (z, t) = (z.to_f64().unwrap(), t.to_f64().unwrap());
                total += t * softplus(-z) + (1.0 - t) * softplus(z);
            });
        }
        OutputKind::Softmax => {
            for (zr, yr) in logits.rows().into_iter().zip(y.rows()) {
                let max = zr
                    .iter()
                    .fold(f64::NEG_INFINITY, |a, &b| a.max(b.to_f64().unwrap()));
                let lse = max
                    + zr.iter()
                        .map(|&v| (v.to_f64().unwrap() - max).exp())
                        .sum::<f64>()
                        .ln();
                let mut yz = 0.0;
                let mut ysum = 0.0;
                for (&zv, &yv) in zr.iter().zip(yr.iter()) {
                    let yv = yv.to_f64().unwrap();
                    yz += yv * zv.to_f64().unwrap();
                    ysum += yv;
                }
                total += ysum * lse - yz;
            }
        }
    }
    total
}

/// `Σ ||W||²` over all weight matrices (biases excluded, as in sklearn).
pub(crate) fn squared_weight_norm<F: Float>(layers: &[DenseLayer<F>]) -> f64 {
    layers
        .iter()
        .map(|l| {
            l.weights
                .iter()
                .map(|&w| {
                    let w = w.to_f64().unwrap();
                    w * w
                })
                .sum::<f64>()
        })
        .sum()
}

/// Weight gradients for a single layer.
#[derive(Debug, Clone)]
pub(crate) struct LayerGradients<F: Float> {
    pub dw: Array2<F>,
    pub db: Array2<F>,
}

impl<F: Float> LayerGradients<F> {
    pub fn zeros_like(layers: &[DenseLayer<F>]) -> Vec<Self> {
        layers
            .iter()
            .map(|l| LayerGradients {
                dw: Array2::zeros(l.weights.raw_dim()),
                db: Array2::zeros(l.biases.raw_dim()),
            })
            .collect()
    }
}

/// Loss and gradients for one batch (sklearn `_backprop`):
///
/// `loss = data_loss / n + alpha/(2n) Σ||W||²`,
/// `∂/∂W = (inputᵀ δ + alpha W) / n`, `∂/∂b = Σ δ / n`,
/// with `δ_out = output(z) - y`. Gradients are written into `grads`.
pub(crate) fn backprop<F: Float>(
    layers: &[DenseLayer<F>],
    activation: Activation,
    kind: OutputKind,
    x: ArrayView2<F>,
    y: ArrayView2<F>,
    alpha: f64,
    grads: &mut [LayerGradients<F>],
) -> f64 {
    let n = x.nrows();
    let n_f = n as f64;
    let (hidden, logits) = forward(layers, activation, x);
    let loss =
        data_loss_sum(kind, &logits, &y) / n_f + 0.5 * alpha * squared_weight_norm(layers) / n_f;

    let mut delta = output_activation(kind, logits);
    delta -= &y;
    let inv_n = F::from_f64(1.0 / n_f).unwrap();
    let alpha_n = F::from_f64(alpha / n_f).unwrap();
    for i in (0..layers.len()).rev() {
        let g = &mut grads[i];
        if i == 0 {
            general_mat_mul(inv_n, &x.t(), &delta, F::zero(), &mut g.dw);
        } else {
            general_mat_mul(inv_n, &hidden[i - 1].t(), &delta, F::zero(), &mut g.dw);
        }
        g.dw.scaled_add(alpha_n, &layers[i].weights);
        let db = delta.sum_axis(Axis(0));
        Zip::from(g.db.row_mut(0))
            .and(&db)
            .for_each(|o, &v| *o = v * inv_n);
        if i > 0 {
            let mut prev = delta.dot(&layers[i].weights.t());
            activation.backward_from_activation_inplace(&hidden[i - 1], &mut prev);
            delta = prev;
        }
    }
    loss
}

/// Network output (after the output activation) for prediction.
pub(crate) fn predict_output<F: Float>(
    layers: &[DenseLayer<F>],
    activation: Activation,
    kind: OutputKind,
    x: ArrayView2<F>,
) -> Array2<F> {
    let (_, logits) = forward(layers, activation, x);
    output_activation(kind, logits)
}

/// Total number of parameters (all weights then all biases).
pub(crate) fn n_params<F: Float>(layers: &[DenseLayer<F>]) -> usize {
    layers
        .iter()
        .map(|l| l.weights.len() + l.biases.len())
        .sum()
}

/// Flatten parameters (sklearn `_pack`: all coefs, then all intercepts).
pub(crate) fn pack<F: Float>(layers: &[DenseLayer<F>], out: &mut Vec<f64>) {
    out.clear();
    for l in layers {
        out.extend(l.weights.iter().map(|v| v.to_f64().unwrap()));
    }
    for l in layers {
        out.extend(l.biases.iter().map(|v| v.to_f64().unwrap()));
    }
}

pub(crate) fn pack_grads<F: Float>(grads: &[LayerGradients<F>], out: &mut [f64]) {
    let mut k = 0;
    for g in grads {
        for v in g.dw.iter() {
            out[k] = v.to_f64().unwrap();
            k += 1;
        }
    }
    for g in grads {
        for v in g.db.iter() {
            out[k] = v.to_f64().unwrap();
            k += 1;
        }
    }
}

pub(crate) fn unpack<F: Float>(layers: &mut [DenseLayer<F>], p: &[f64]) {
    let mut k = 0;
    for l in layers.iter_mut() {
        for v in l.weights.iter_mut() {
            *v = F::from_f64(p[k]).unwrap();
            k += 1;
        }
    }
    for l in layers.iter_mut() {
        for v in l.biases.iter_mut() {
            *v = F::from_f64(p[k]).unwrap();
            k += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand::SeedableRng;

    #[test]
    fn dense_layer_shapes() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let layer = DenseLayer::<f64>::new(3, 5, Activation::Relu, &mut rng);
        assert_eq!(layer.weights.shape(), &[3, 5]);
        assert_eq!(layer.biases.shape(), &[1, 5]);
    }

    #[test]
    fn glorot_uniform_bounds_and_nonzero_biases() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let l = DenseLayer::<f64>::new(30, 20, Activation::Relu, &mut rng);
        let b = (6.0_f64 / 50.0).sqrt();
        assert!(l.weights.iter().all(|w| w.abs() <= b));
        assert!(l.biases.iter().all(|w| w.abs() <= b));
        assert!(l.biases.iter().any(|w| w.abs() > 0.0));
        // Roughly uniform: variance ≈ b²/3.
        let var = l.weights.iter().map(|w| w * w).sum::<f64>() / l.weights.len() as f64;
        assert!((var - b * b / 3.0).abs() < 0.2 * b * b / 3.0, "{var}");
        let ls = DenseLayer::<f64>::new(30, 20, Activation::Sigmoid, &mut rng);
        let bs = (2.0_f64 / 50.0).sqrt();
        assert!(ls.weights.iter().all(|w| w.abs() <= bs));
        assert!(ls.weights.iter().any(|w| w.abs() > 0.5 * bs));
    }

    #[test]
    fn forward_output_shape() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let layers = vec![
            DenseLayer::<f64>::new(4, 10, Activation::Relu, &mut rng),
            DenseLayer::new(10, 3, Activation::Relu, &mut rng),
        ];
        let x = Array2::<f64>::zeros((5, 4));
        let (hidden, out) = forward(&layers, Activation::Relu, x.view());
        assert_eq!(out.shape(), &[5, 3]);
        assert_eq!(hidden.len(), 1);
    }

    /// Analytic gradients (incl. the sklearn-scaled L2 term) agree with
    /// central finite differences of the loss, for every output kind.
    #[test]
    fn backprop_matches_finite_differences() {
        let x = array![
            [0.2, -1.0, 0.5],
            [1.5, 0.3, -0.7],
            [-0.4, 0.8, 1.1],
            [0.0, -0.2, 0.9]
        ];
        let cases = [
            (
                OutputKind::Identity,
                array![[0.5, -1.0], [1.2, 0.3], [-0.7, 0.0], [0.1, 2.0]],
            ),
            (OutputKind::Logistic, array![[1.0], [0.0], [1.0], [0.0]]),
            (
                OutputKind::Softmax,
                array![
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0]
                ],
            ),
        ];
        for act in [Activation::Tanh, Activation::Sigmoid, Activation::Relu] {
            for (kind, y) in cases.iter() {
                let mut rng = rand::rngs::StdRng::seed_from_u64(3);
                let mut layers = vec![
                    DenseLayer::<f64>::new(3, 4, act, &mut rng),
                    DenseLayer::new(4, y.ncols(), act, &mut rng),
                ];
                let alpha = 0.3;
                let mut grads = LayerGradients::zeros_like(&layers);
                backprop(&layers, act, *kind, x.view(), y.view(), alpha, &mut grads);
                let mut g = vec![0.0; n_params(&layers)];
                pack_grads(&grads, &mut g);
                let mut p = Vec::new();
                pack(&layers, &mut p);
                let h = 1e-6;
                for k in 0..p.len() {
                    let mut pp = p.clone();
                    pp[k] += h;
                    unpack(&mut layers, &pp);
                    let mut tmp = LayerGradients::zeros_like(&layers);
                    let fp = backprop(&layers, act, *kind, x.view(), y.view(), alpha, &mut tmp);
                    pp[k] -= 2.0 * h;
                    unpack(&mut layers, &pp);
                    let fm = backprop(&layers, act, *kind, x.view(), y.view(), alpha, &mut tmp);
                    unpack(&mut layers, &p);
                    let fd = (fp - fm) / (2.0 * h);
                    assert!(
                        (fd - g[k]).abs() < 1e-6,
                        "{act:?} {kind:?} param {k}: {fd} vs {}",
                        g[k]
                    );
                }
            }
        }
    }

    #[test]
    fn softmax_loss_is_finite_for_extreme_logits() {
        let z = array![[1000.0, -1000.0], [-800.0, 800.0]];
        let y = array![[0.0, 1.0], [0.0, 1.0]];
        let l = data_loss_sum(OutputKind::Softmax, &z, &y.view());
        assert!((l - 2000.0).abs() < 1e-9, "{l}");
        let l = data_loss_sum(
            OutputKind::Logistic,
            &array![[-1000.0]],
            &array![[1.0]].view(),
        );
        assert!((l - 1000.0).abs() < 1e-9);
    }
}
