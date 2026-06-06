use anofox_ml_core::Float;
use ndarray::Array2;

/// Optimization algorithm for weight updates.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Solver {
    Sgd,
    #[default]
    Adam,
}

/// Per-parameter Adam optimizer state.
pub(crate) struct AdamState<F: Float> {
    pub m: Array2<F>, // first moment
    pub v: Array2<F>, // second moment
    pub t: usize,     // timestep
}

impl<F: Float> AdamState<F> {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            m: Array2::zeros((rows, cols)),
            v: Array2::zeros((rows, cols)),
            t: 0,
        }
    }

    /// Compute the Adam update and return the parameter delta.
    pub fn step(&mut self, grad: &Array2<F>, lr: F) -> Array2<F> {
        let beta1 = F::from_f64(0.9).unwrap();
        let beta2 = F::from_f64(0.999).unwrap();
        let eps = F::from_f64(1e-8).unwrap();

        self.t += 1;
        let t_f = F::from_usize(self.t).unwrap();

        // Update biased first and second moment estimates
        self.m = &self.m * beta1 + grad * (F::one() - beta1);
        self.v = &self.v * beta2 + &grad.mapv(|g| g * g) * (F::one() - beta2);

        // Bias-corrected estimates
        let m_hat = &self.m / (F::one() - beta1.powf(t_f));
        let v_hat = &self.v / (F::one() - beta2.powf(t_f));

        // Parameter update
        m_hat / v_hat.mapv(|v| v.sqrt() + eps) * (-lr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn adam_step_reduces_gradient() {
        let mut state = AdamState::<f64>::new(1, 2);
        let grad = array![[1.0, -2.0]];
        let delta = state.step(&grad, 0.001);

        // Adam should produce a non-zero update in the opposite direction of the gradient
        assert!(delta[[0, 0]] < 0.0); // gradient is positive, delta should be negative
        assert!(delta[[0, 1]] > 0.0); // gradient is negative, delta should be positive
    }
}
