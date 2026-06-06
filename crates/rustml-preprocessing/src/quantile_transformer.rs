use ndarray::Array2;
use rustml_core::{FitUnsupervised, Float, Result, RustMlError, Transform};

/// Output distribution for the quantile transformer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum OutputDistribution {
    /// Map to a uniform distribution on [0, 1].
    Uniform,
    /// Map to a standard normal distribution.
    Normal,
}

/// Parameters for QuantileTransformer (unfitted state).
///
/// Transforms features to follow a uniform or normal distribution by
/// estimating the cumulative distribution function (CDF) via quantiles.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QuantileTransformer {
    /// Number of quantiles to compute. Clamped to n_samples if larger.
    pub n_quantiles: usize,
    /// Target output distribution.
    pub output_distribution: OutputDistribution,
}

impl QuantileTransformer {
    /// Create a new `QuantileTransformer` with defaults (1000 quantiles, uniform output).
    pub fn new() -> Self {
        Self {
            n_quantiles: 1000,
            output_distribution: OutputDistribution::Uniform,
        }
    }

    /// Set the number of quantiles to compute.
    pub fn n_quantiles(mut self, n_quantiles: usize) -> Self {
        self.n_quantiles = n_quantiles;
        self
    }

    /// Set the output distribution.
    pub fn output_distribution(mut self, output_distribution: OutputDistribution) -> Self {
        self.output_distribution = output_distribution;
        self
    }
}

impl Default for QuantileTransformer {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted QuantileTransformer -- holds quantile references per feature.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedQuantileTransformer<F: Float> {
    /// For each feature, sorted quantile values (length = effective n_quantiles).
    quantiles: Vec<Vec<F>>,
    /// The corresponding CDF positions for each quantile, in [0, 1].
    references: Vec<f64>,
    output_distribution: OutputDistribution,
}

/// Approximate the inverse of the standard normal CDF (probit function)
/// using the rational approximation by Peter Acklam.
fn inverse_normal_cdf(p: f64) -> f64 {
    if p <= 0.0 {
        return -8.0; // clamp
    }
    if p >= 1.0 {
        return 8.0; // clamp
    }

    // Coefficients for the rational approximation
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        // Rational approximation for lower region
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= p_high {
        // Rational approximation for central region
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        // Rational approximation for upper region
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Linear interpolation: given sorted `xp` with corresponding `fp`,
/// find the interpolated value at `x`.
fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    debug_assert_eq!(xp.len(), fp.len());
    let n = xp.len();
    if n == 0 {
        return 0.0;
    }
    if x <= xp[0] {
        return fp[0];
    }
    if x >= xp[n - 1] {
        return fp[n - 1];
    }

    // Binary search for the interval
    let mut lo = 0;
    let mut hi = n - 1;
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if xp[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    let dx = xp[hi] - xp[lo];
    if dx.abs() < 1e-30 {
        return fp[lo];
    }
    let t = (x - xp[lo]) / dx;
    fp[lo] + t * (fp[hi] - fp[lo])
}

impl<F: Float> FitUnsupervised<F> for QuantileTransformer {
    type Fitted = FittedQuantileTransformer<F>;

    fn fit(&self, x: &Array2<F>) -> Result<Self::Fitted> {
        if x.is_empty() {
            return Err(RustMlError::EmptyInput("input array is empty".into()));
        }

        let n_samples = x.nrows();
        let ncols = x.ncols();
        let effective_n = self.n_quantiles.min(n_samples);

        // Compute reference positions in [0, 1]
        let references: Vec<f64> = if effective_n == 1 {
            vec![0.5]
        } else {
            (0..effective_n)
                .map(|i| i as f64 / (effective_n - 1) as f64)
                .collect()
        };

        let mut quantiles = Vec::with_capacity(ncols);

        for j in 0..ncols {
            let mut col: Vec<F> = x.column(j).to_vec();
            col.sort_by(|a, b| a.partial_cmp(b).unwrap());

            // Compute quantiles at the reference positions
            let q: Vec<F> = references
                .iter()
                .map(|&p| percentile_sorted(&col, p))
                .collect();

            quantiles.push(q);
        }

        Ok(FittedQuantileTransformer {
            quantiles,
            references,
            output_distribution: self.output_distribution,
        })
    }
}

/// Compute a percentile from a sorted slice using linear interpolation.
fn percentile_sorted<F: Float>(sorted: &[F], p: f64) -> F {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let idx = p * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil().min((n - 1) as f64) as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = F::from_f64(idx - lo as f64).unwrap();
        sorted[lo] * (F::one() - frac) + sorted[hi] * frac
    }
}

impl<F: Float> Transform<F> for FittedQuantileTransformer<F> {
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>> {
        let expected_cols = self.quantiles.len();
        if x.ncols() != expected_cols {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                expected_cols,
                x.ncols()
            )));
        }

        let mut result = Array2::<F>::zeros(x.raw_dim());

        for j in 0..x.ncols() {
            let q = &self.quantiles[j];
            // Build xp (quantile values as f64) and fp (references)
            let xp: Vec<f64> = q.iter().map(|&v| v.to_f64().unwrap()).collect();
            let fp = &self.references;

            for i in 0..x.nrows() {
                let val = x[[i, j]].to_f64().unwrap();
                // Interpolate: map from data space to [0, 1]
                let mut u = interp(val, &xp, fp);

                // Clip to (epsilon, 1 - epsilon) to avoid infinities in normal transform
                let eps = 1e-7;
                u = u.max(eps).min(1.0 - eps);

                let out = match self.output_distribution {
                    OutputDistribution::Uniform => u,
                    OutputDistribution::Normal => inverse_normal_cdf(u),
                };

                result[[i, j]] = F::from_f64(out).unwrap();
            }
        }

        Ok(result)
    }
}

impl<F: Float> FittedQuantileTransformer<F> {
    /// Return the quantile values per feature.
    pub fn quantiles(&self) -> &Vec<Vec<F>> {
        &self.quantiles
    }

    /// Return the reference positions used for interpolation.
    pub fn references(&self) -> &Vec<f64> {
        &self.references
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_uniform_output() {
        let x = array![
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
        ];
        let qt = QuantileTransformer::new()
            .n_quantiles(5)
            .output_distribution(OutputDistribution::Uniform);
        let fitted = FitUnsupervised::<f64>::fit(&qt, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        // With 5 samples and 5 quantiles, the result should be approximately
        // [0, 0.25, 0.5, 0.75, 1.0] clipped to (eps, 1-eps)
        let eps = 1e-7;
        assert_abs_diff_eq!(transformed[[0, 0]], eps, epsilon = 1e-6);
        assert_abs_diff_eq!(transformed[[1, 0]], 0.25, epsilon = 1e-6);
        assert_abs_diff_eq!(transformed[[2, 0]], 0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(transformed[[3, 0]], 0.75, epsilon = 1e-6);
        assert_abs_diff_eq!(transformed[[4, 0]], 1.0 - eps, epsilon = 1e-6);
    }

    #[test]
    fn test_normal_output() {
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0],
            [10.0],
        ];
        let qt = QuantileTransformer::new()
            .n_quantiles(10)
            .output_distribution(OutputDistribution::Normal);
        let fitted = FitUnsupervised::<f64>::fit(&qt, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        // The median value (5.5) should map to approximately 0
        // Values below median should be negative, above should be positive
        assert!(transformed[[0, 0]] < 0.0);
        assert!(transformed[[9, 0]] > 0.0);

        // The output should be symmetric around the median
        assert_abs_diff_eq!(transformed[[0, 0]], -transformed[[9, 0]], epsilon = 1e-6);
    }

    #[test]
    fn test_output_range_uniform() {
        let x = array![
            [10.0],
            [20.0],
            [30.0],
            [40.0],
            [50.0],
            [60.0],
            [70.0],
            [80.0],
            [90.0],
            [100.0],
        ];
        let qt = QuantileTransformer::new()
            .n_quantiles(10)
            .output_distribution(OutputDistribution::Uniform);
        let fitted = FitUnsupervised::<f64>::fit(&qt, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        // All values should be in (0, 1)
        for &v in transformed.iter() {
            assert!(v > 0.0 && v < 1.0, "value out of range: {}", v);
        }
    }

    #[test]
    fn test_empty_input() {
        let x: Array2<f64> = Array2::zeros((0, 0));
        let qt = QuantileTransformer::default();
        assert!(FitUnsupervised::<f64>::fit(&qt, &x).is_err());
    }

    #[test]
    fn test_shape_mismatch() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let qt = QuantileTransformer::default();
        let fitted = FitUnsupervised::<f64>::fit(&qt, &x).unwrap();

        let x_wrong = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&x_wrong).is_err());
    }

    #[test]
    fn test_n_quantiles_larger_than_samples() {
        // n_quantiles > n_samples should be clamped
        let x = array![[1.0], [2.0], [3.0]];
        let qt = QuantileTransformer::new()
            .n_quantiles(1000)
            .output_distribution(OutputDistribution::Uniform);
        let fitted = FitUnsupervised::<f64>::fit(&qt, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        // Should still produce valid output
        for &v in transformed.iter() {
            assert!(v.is_finite(), "non-finite value: {}", v);
        }
    }

    #[test]
    fn test_monotonicity_preserved() {
        // Transform should preserve ordering
        let x = array![[1.0], [3.0], [5.0], [7.0], [9.0]];
        let qt = QuantileTransformer::new()
            .n_quantiles(5)
            .output_distribution(OutputDistribution::Uniform);
        let fitted = FitUnsupervised::<f64>::fit(&qt, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        for i in 1..x.nrows() {
            assert!(
                transformed[[i, 0]] >= transformed[[i - 1, 0]],
                "monotonicity violated at row {}",
                i
            );
        }
    }

    #[test]
    fn test_inverse_normal_cdf_symmetry() {
        // inverse_normal_cdf(0.5) should be 0
        assert_abs_diff_eq!(inverse_normal_cdf(0.5), 0.0, epsilon = 1e-10);
        // Symmetry: inv_cdf(p) = -inv_cdf(1-p)
        for &p in &[0.1, 0.2, 0.3, 0.4] {
            assert_abs_diff_eq!(
                inverse_normal_cdf(p),
                -inverse_normal_cdf(1.0 - p),
                epsilon = 1e-10
            );
        }
    }
}
