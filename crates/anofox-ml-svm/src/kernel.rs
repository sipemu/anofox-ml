use anofox_ml_core::Float;
use ndarray::ArrayView1;

/// Kernel functions for the Support Vector Classifier.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum SvmKernel {
    /// Linear kernel: K(x, y) = x . y
    Linear,
    /// Radial Basis Function kernel: K(x, y) = exp(-gamma * ||x - y||^2)
    Rbf {
        /// Controls how far the influence of a single training example reaches.
        gamma: f64,
    },
    /// Polynomial kernel: K(x, y) = (x . y + coef0)^degree
    Polynomial {
        /// Degree of the polynomial kernel function.
        degree: usize,
        /// Independent term in the polynomial kernel function.
        coef0: f64,
    },
}

impl Default for SvmKernel {
    fn default() -> Self {
        SvmKernel::Rbf { gamma: 1.0 }
    }
}

impl SvmKernel {
    /// Compute the kernel value between two feature vectors.
    pub fn compute<F: Float>(&self, x: &ArrayView1<F>, y: &ArrayView1<F>) -> F {
        match self {
            SvmKernel::Linear => x.dot(y),
            SvmKernel::Rbf { gamma } => {
                let gamma_f = F::from_f64(*gamma).unwrap();
                let sq_dist = x.iter().zip(y.iter()).fold(F::zero(), |acc, (&a, &b)| {
                    let d = a - b;
                    acc + d * d
                });
                (-gamma_f * sq_dist).exp()
            }
            SvmKernel::Polynomial { degree, coef0 } => {
                let coef0_f = F::from_f64(*coef0).unwrap();
                let dot = x.dot(y) + coef0_f;
                let mut result = F::one();
                for _ in 0..*degree {
                    result *= dot;
                }
                result
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_linear_kernel() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![4.0, 5.0, 6.0];
        let kernel = SvmKernel::Linear;
        let result: f64 = kernel.compute(&x.view(), &y.view());
        // 1*4 + 2*5 + 3*6 = 32
        assert_abs_diff_eq!(result, 32.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rbf_kernel_same_point() {
        let x = array![1.0, 2.0];
        let kernel = SvmKernel::Rbf { gamma: 0.5 };
        let result: f64 = kernel.compute(&x.view(), &x.view());
        // exp(-0.5 * 0) = 1.0
        assert_abs_diff_eq!(result, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rbf_kernel_different_points() {
        let x = array![0.0, 0.0];
        let y = array![1.0, 1.0];
        let kernel = SvmKernel::Rbf { gamma: 0.5 };
        let result: f64 = kernel.compute(&x.view(), &y.view());
        // exp(-0.5 * 2) = exp(-1)
        assert_abs_diff_eq!(result, (-1.0_f64).exp(), epsilon = 1e-10);
    }

    #[test]
    fn test_polynomial_kernel() {
        let x = array![1.0, 2.0];
        let y = array![3.0, 4.0];
        let kernel = SvmKernel::Polynomial {
            degree: 2,
            coef0: 1.0,
        };
        let result: f64 = kernel.compute(&x.view(), &y.view());
        // (1*3 + 2*4 + 1)^2 = (12)^2 = 144
        assert_abs_diff_eq!(result, 144.0, epsilon = 1e-10);
    }

    #[test]
    fn test_polynomial_kernel_degree_one() {
        let x = array![1.0, 2.0];
        let y = array![3.0, 4.0];
        let kernel = SvmKernel::Polynomial {
            degree: 1,
            coef0: 0.0,
        };
        let result: f64 = kernel.compute(&x.view(), &y.view());
        // (1*3 + 2*4 + 0)^1 = 11
        assert_abs_diff_eq!(result, 11.0, epsilon = 1e-10);
    }

    #[test]
    fn test_linear_kernel_f32() {
        let x = array![1.0_f32, 2.0, 3.0];
        let y = array![4.0_f32, 5.0, 6.0];
        let kernel = SvmKernel::Linear;
        let result: f32 = kernel.compute(&x.view(), &y.view());
        assert_abs_diff_eq!(result, 32.0_f32, epsilon = 1e-5);
    }

    #[test]
    fn test_rbf_kernel_f32() {
        let x = array![0.0_f32, 0.0];
        let y = array![1.0_f32, 1.0];
        let kernel = SvmKernel::Rbf { gamma: 0.5 };
        let result: f32 = kernel.compute(&x.view(), &y.view());
        assert_abs_diff_eq!(result, (-1.0_f32).exp(), epsilon = 1e-5);
    }
}
