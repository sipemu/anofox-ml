use anofox_ml_core::Float;
use ndarray::{Array2, Zip};

/// Activation function used in hidden layers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Activation {
    #[default]
    Relu,
    Tanh,
    Sigmoid,
    Identity,
}

impl Activation {
    /// Apply activation element-wise: a = f(z).
    pub fn forward<F: Float>(&self, z: &Array2<F>) -> Array2<F> {
        match self {
            Activation::Relu => z.mapv(|v| if v > F::zero() { v } else { F::zero() }),
            Activation::Tanh => z.mapv(|v| v.tanh()),
            Activation::Sigmoid => z.mapv(|v| F::one() / (F::one() + (-v).exp())),
            Activation::Identity => z.clone(),
        }
    }

    /// Apply the activation in place.
    pub fn forward_inplace<F: Float>(&self, z: &mut Array2<F>) {
        match self {
            Activation::Relu => z.mapv_inplace(|v| if v > F::zero() { v } else { F::zero() }),
            Activation::Tanh => z.mapv_inplace(|v| v.tanh()),
            Activation::Sigmoid => z.mapv_inplace(|v| F::one() / (F::one() + (-v).exp())),
            Activation::Identity => {}
        }
    }

    /// Multiply `delta` in place by `f'(z)`, expressed through the cached
    /// activation `a = f(z)` (no recomputation of `tanh` / `sigmoid`):
    /// ReLU `a > 0`, tanh `1 - a²`, sigmoid `a (1 - a)`, identity `1`.
    pub fn backward_from_activation_inplace<F: Float>(&self, a: &Array2<F>, delta: &mut Array2<F>) {
        match self {
            Activation::Relu => Zip::from(delta).and(a).for_each(|d, &av| {
                if av <= F::zero() {
                    *d = F::zero();
                }
            }),
            Activation::Tanh => Zip::from(delta)
                .and(a)
                .for_each(|d, &av| *d *= F::one() - av * av),
            Activation::Sigmoid => Zip::from(delta)
                .and(a)
                .for_each(|d, &av| *d *= av * (F::one() - av)),
            Activation::Identity => {}
        }
    }

    /// Compute activation derivative element-wise given the pre-activation z.
    pub fn backward<F: Float>(&self, z: &Array2<F>) -> Array2<F> {
        match self {
            Activation::Relu => z.mapv(|v| if v > F::zero() { F::one() } else { F::zero() }),
            Activation::Tanh => z.mapv(|v| {
                let t = v.tanh();
                F::one() - t * t
            }),
            Activation::Sigmoid => z.mapv(|v| {
                let s = F::one() / (F::one() + (-v).exp());
                s * (F::one() - s)
            }),
            Activation::Identity => Array2::ones(z.raw_dim()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn relu_forward() {
        let z = array![[-1.0, 0.0, 2.0]];
        let a = Activation::Relu.forward(&z);
        assert_abs_diff_eq!(a[[0, 0]], 0.0);
        assert_abs_diff_eq!(a[[0, 1]], 0.0);
        assert_abs_diff_eq!(a[[0, 2]], 2.0);
    }

    #[test]
    fn relu_backward() {
        let z = array![[-1.0, 0.0, 2.0]];
        let d = Activation::Relu.backward(&z);
        assert_abs_diff_eq!(d[[0, 0]], 0.0);
        assert_abs_diff_eq!(d[[0, 1]], 0.0);
        assert_abs_diff_eq!(d[[0, 2]], 1.0);
    }

    #[test]
    fn sigmoid_forward() {
        let z = array![[0.0_f64]];
        let a = Activation::Sigmoid.forward(&z);
        assert_abs_diff_eq!(a[[0, 0]], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn tanh_forward() {
        let z = array![[0.0_f64]];
        let a = Activation::Tanh.forward(&z);
        assert_abs_diff_eq!(a[[0, 0]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn cached_derivative_matches_pre_activation_derivative() {
        let z = array![[-1.5, -0.2, 0.0, 0.3, 2.0]];
        for act in [
            Activation::Relu,
            Activation::Tanh,
            Activation::Sigmoid,
            Activation::Identity,
        ] {
            let mut a = z.clone();
            act.forward_inplace(&mut a);
            assert_eq!(a, act.forward(&z));
            let mut d = Array2::<f64>::ones(z.raw_dim());
            act.backward_from_activation_inplace(&a, &mut d);
            for (x, y) in d.iter().zip(act.backward(&z).iter()) {
                assert_abs_diff_eq!(*x, *y, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn identity_forward() {
        let z = array![[3.125_f64, -2.7]];
        let a = Activation::Identity.forward(&z);
        for (ai, zi) in a.iter().zip(z.iter()) {
            assert_abs_diff_eq!(*ai, *zi, epsilon = 1e-15);
        }
    }
}
