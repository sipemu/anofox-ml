use anofox_ml_core::{FitUnsupervised, Float, InverseTransform, Result, RustMlError, Transform};
use ndarray::{Array1, Array2};

/// Parameters for PowerTransformer (unfitted state).
///
/// Applies a Yeo-Johnson power transform to each feature to make the data
/// more Gaussian-like, then optionally standardizes to zero mean and unit
/// variance.
///
/// The optimal lambda per feature is found via grid search (maximizing the
/// log-likelihood of the resulting normal distribution).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PowerTransformer {
    /// If true, standardize the transformed data to zero mean, unit variance.
    pub standardize: bool,
}

impl PowerTransformer {
    /// Create a new `PowerTransformer` with defaults (standardization enabled).
    pub fn new() -> Self {
        Self { standardize: true }
    }

    /// Set whether to standardize after the power transform.
    pub fn standardize(mut self, standardize: bool) -> Self {
        self.standardize = standardize;
        self
    }
}

impl Default for PowerTransformer {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted PowerTransformer -- holds learned lambdas per feature and
/// optional standardization parameters (mean and std).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedPowerTransformer<F: Float> {
    lambdas: Array1<F>,
    means: Array1<F>,
    stds: Array1<F>,
    standardize: bool,
}

/// Apply the Yeo-Johnson transform to a single value.
fn yeo_johnson<F: Float>(x: F, lam: F) -> F {
    let zero = F::zero();
    let one = F::one();
    let two = F::from_f64(2.0).unwrap();
    let eps = F::from_f64(1e-10).unwrap();

    if x >= zero {
        if (lam - zero).abs() > eps {
            // ((x + 1)^lambda - 1) / lambda
            ((x + one).powf(lam) - one) / lam
        } else {
            // ln(x + 1)
            (x + one).ln()
        }
    } else {
        // x < 0
        if (lam - two).abs() > eps {
            // -((-x + 1)^(2 - lambda) - 1) / (2 - lambda)
            -((-x + one).powf(two - lam) - one) / (two - lam)
        } else {
            // -ln(-x + 1)
            -(-x + one).ln()
        }
    }
}

/// Apply the inverse Yeo-Johnson transform to a single value.
fn yeo_johnson_inverse<F: Float>(y: F, lam: F) -> F {
    let zero = F::zero();
    let one = F::one();
    let two = F::from_f64(2.0).unwrap();
    let eps = F::from_f64(1e-10).unwrap();

    if y >= zero {
        if (lam - zero).abs() > eps {
            // x = (y * lambda + 1)^(1/lambda) - 1
            (y * lam + one).powf(one / lam) - one
        } else {
            // x = exp(y) - 1
            y.exp() - one
        }
    } else {
        // y < 0
        if (lam - two).abs() > eps {
            // x = 1 - (-(2 - lambda) * y + 1)^(1/(2-lambda))
            one - (-(two - lam) * y + one).powf(one / (two - lam))
        } else {
            // x = 1 - exp(-y)
            one - (-y).exp()
        }
    }
}

/// Compute the negative log-likelihood for a candidate lambda on a column.
/// The log-likelihood of the Yeo-Johnson transformed data under a normal
/// model includes the Jacobian term.
fn neg_log_likelihood<F: Float>(col: &[F], lam: F) -> f64 {
    let n = col.len() as f64;
    // Transform all values
    let transformed: Vec<f64> = col
        .iter()
        .map(|&x| yeo_johnson(x, lam).to_f64().unwrap())
        .collect();

    // Mean
    let mean: f64 = transformed.iter().sum::<f64>() / n;

    // Variance
    let var: f64 = transformed
        .iter()
        .map(|&t| (t - mean) * (t - mean))
        .sum::<f64>()
        / n;
    let var = var.max(1e-30); // avoid log(0)

    // Log-likelihood = -n/2 * ln(2*pi*var) + (lam - 1) * sum(sign(x) * ln(|x| + 1))
    // We only need to maximize, so we can drop the constant -n/2 * ln(2*pi) part
    // nll = n/2 * ln(var) - (lam - 1) * sum(sign(x) * ln(|x| + 1))
    let lam_f64 = lam.to_f64().unwrap();
    let jacobian_sum: f64 = col
        .iter()
        .map(|&x| {
            let x_f64 = x.to_f64().unwrap();
            let sign = if x_f64 >= 0.0 { 1.0 } else { -1.0 };
            sign * (x_f64.abs() + 1.0).ln()
        })
        .sum();

    n / 2.0 * var.ln() - (lam_f64 - 1.0) * jacobian_sum
}

impl<F: Float> FitUnsupervised<F> for PowerTransformer {
    type Fitted = FittedPowerTransformer<F>;

    fn fit(&self, x: &Array2<F>) -> Result<Self::Fitted> {
        if x.is_empty() {
            return Err(RustMlError::EmptyInput("input array is empty".into()));
        }

        let ncols = x.ncols();
        let mut lambdas = Array1::<F>::zeros(ncols);

        // For each feature, find the best lambda by grid search
        for j in 0..ncols {
            let col: Vec<F> = x.column(j).to_vec();
            let mut best_lam = F::zero();
            let mut best_nll = f64::INFINITY;

            // Grid search over lambda in [-2.0, 2.0] with step 0.1
            let mut lam_val = -20i32; // represents -2.0
            while lam_val <= 20 {
                let lam = F::from_f64(lam_val as f64 / 10.0).unwrap();
                let nll = neg_log_likelihood(&col, lam);
                if nll < best_nll {
                    best_nll = nll;
                    best_lam = lam;
                }
                lam_val += 1;
            }

            lambdas[j] = best_lam;
        }

        // Transform the data to compute standardization parameters
        let n = F::from_usize(x.nrows()).unwrap();
        let mut means = Array1::<F>::zeros(ncols);
        let mut stds = Array1::<F>::ones(ncols);

        if self.standardize {
            // Compute the transformed data's mean and std per column
            for j in 0..ncols {
                let lam = lambdas[j];
                let mut sum = F::zero();
                for &val in x.column(j).iter() {
                    sum = sum + yeo_johnson(val, lam);
                }
                let mean = sum / n;
                means[j] = mean;

                let mut var_sum = F::zero();
                for &val in x.column(j).iter() {
                    let t = yeo_johnson(val, lam) - mean;
                    var_sum = var_sum + t * t;
                }
                stds[j] = (var_sum / n).sqrt();
            }
        }

        Ok(FittedPowerTransformer {
            lambdas,
            means,
            stds,
            standardize: self.standardize,
        })
    }
}

impl<F: Float> Transform<F> for FittedPowerTransformer<F> {
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>> {
        if x.ncols() != self.lambdas.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.lambdas.len(),
                x.ncols()
            )));
        }

        let mut result = Array2::<F>::zeros(x.raw_dim());
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                let mut val = yeo_johnson(x[[i, j]], self.lambdas[j]);
                if self.standardize {
                    val = val - self.means[j];
                    if self.stds[j] > F::from_f64(1e-15).unwrap() {
                        val = val / self.stds[j];
                    }
                }
                result[[i, j]] = val;
            }
        }
        Ok(result)
    }
}

impl<F: Float> InverseTransform<F> for FittedPowerTransformer<F> {
    fn inverse_transform(&self, x: &Array2<F>) -> Result<Array2<F>> {
        if x.ncols() != self.lambdas.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.lambdas.len(),
                x.ncols()
            )));
        }

        let mut result = Array2::<F>::zeros(x.raw_dim());
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                let mut val = x[[i, j]];
                // Undo standardization
                if self.standardize {
                    if self.stds[j] > F::from_f64(1e-15).unwrap() {
                        val = val * self.stds[j];
                    }
                    val = val + self.means[j];
                }
                // Undo Yeo-Johnson
                result[[i, j]] = yeo_johnson_inverse(val, self.lambdas[j]);
            }
        }
        Ok(result)
    }
}

impl<F: Float> FittedPowerTransformer<F> {
    /// Return the fitted lambda per feature.
    pub fn lambdas(&self) -> &Array1<F> {
        &self.lambdas
    }

    /// Return the mean per feature (after Yeo-Johnson, before standardization).
    pub fn means(&self) -> &Array1<F> {
        &self.means
    }

    /// Return the std per feature (after Yeo-Johnson, before standardization).
    pub fn stds(&self) -> &Array1<F> {
        &self.stds
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_fit_transform_basic() {
        let x = array![
            [1.0, -1.0],
            [2.0, -2.0],
            [3.0, -3.0],
            [4.0, -4.0],
            [5.0, -5.0],
        ];
        let pt = PowerTransformer::default();
        let fitted = FitUnsupervised::<f64>::fit(&pt, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        // Each column should have mean ~0 and std ~1
        let n = x.nrows() as f64;
        for j in 0..x.ncols() {
            let col_mean: f64 = transformed.column(j).sum() / n;
            assert_abs_diff_eq!(col_mean, 0.0, epsilon = 1e-8);

            let col_std: f64 = (transformed
                .column(j)
                .iter()
                .map(|&v| (v - col_mean).powi(2))
                .sum::<f64>()
                / n)
                .sqrt();
            assert_abs_diff_eq!(col_std, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_inverse_transform_roundtrip() {
        let x = array![[0.5, 1.0], [1.5, 2.0], [2.5, 3.0], [3.5, 4.0], [4.5, 5.0],];
        let pt = PowerTransformer::default();
        let fitted = FitUnsupervised::<f64>::fit(&pt, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();
        let recovered = fitted.inverse_transform(&transformed).unwrap();

        for (a, b) in x.iter().zip(recovered.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_without_standardize() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let pt = PowerTransformer::new().standardize(false);
        let fitted = FitUnsupervised::<f64>::fit(&pt, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        // Without standardization, the transform is just Yeo-Johnson
        let lam = fitted.lambdas()[0];
        for i in 0..x.nrows() {
            let expected = yeo_johnson(x[[i, 0]], lam);
            assert_abs_diff_eq!(transformed[[i, 0]], expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_yeo_johnson_identity_at_lambda_one() {
        // When lambda = 1, Yeo-Johnson for x >= 0 is ((x+1)^1 - 1)/1 = x
        let one = 1.0_f64;
        for &x in &[0.0, 1.0, 2.0, 5.0, 10.0] {
            let result = yeo_johnson(x, one);
            assert_abs_diff_eq!(result, x, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_yeo_johnson_lambda_zero() {
        // When lambda = 0, Yeo-Johnson for x >= 0 is ln(x + 1)
        let zero = 0.0_f64;
        for &x in &[0.0, 1.0, 2.0, 5.0] {
            let result = yeo_johnson(x, zero);
            let expected = (x + 1.0).ln();
            assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_yeo_johnson_negative_values() {
        // Verify transform and inverse for negative values
        let lam = 0.5_f64;
        for &x in &[-1.0, -2.0, -5.0, -0.5] {
            let y = yeo_johnson(x, lam);
            let x_back = yeo_johnson_inverse(y, lam);
            assert_abs_diff_eq!(x, x_back, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_lambda_two_negative_branch() {
        // When lambda = 2 and x < 0: -ln(-x + 1)
        let two = 2.0_f64;
        for &x in &[-1.0, -2.0, -0.5] {
            let result = yeo_johnson(x, two);
            let expected = -(-x + 1.0).ln();
            assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_empty_input() {
        let x: Array2<f64> = Array2::zeros((0, 0));
        let pt = PowerTransformer::default();
        assert!(FitUnsupervised::<f64>::fit(&pt, &x).is_err());
    }

    #[test]
    fn test_shape_mismatch() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let pt = PowerTransformer::default();
        let fitted = FitUnsupervised::<f64>::fit(&pt, &x).unwrap();

        let x_wrong = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&x_wrong).is_err());
        assert!(fitted.inverse_transform(&x_wrong).is_err());
    }

    #[test]
    fn test_mixed_positive_negative() {
        let x = array![
            [-3.0, 10.0],
            [-1.0, 20.0],
            [0.0, 30.0],
            [1.0, 40.0],
            [3.0, 50.0],
        ];
        let pt = PowerTransformer::default();
        let fitted = FitUnsupervised::<f64>::fit(&pt, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();
        let recovered = fitted.inverse_transform(&transformed).unwrap();

        for (a, b) in x.iter().zip(recovered.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-5);
        }
    }
}
