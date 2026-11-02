use crate::quantile::{percentile_sorted, sorted_non_nan};
use crate::special::{norm_cdf, norm_ppf};
use anofox_ml_core::{FitUnsupervised, Float, InverseTransform, Result, RustMlError, Transform};
use ndarray::{Array2, ArrayView2};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Output distribution for the quantile transformer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum OutputDistribution {
    /// Map to a uniform distribution on [0, 1].
    Uniform,
    /// Map to a standard normal distribution.
    Normal,
}

fn default_subsample() -> Option<usize> {
    Some(10_000)
}

/// Parameters for QuantileTransformer (unfitted state).
///
/// Transforms features to follow a uniform or normal distribution by
/// estimating the cumulative distribution function (CDF) via quantiles.
/// Mirrors `sklearn.preprocessing.QuantileTransformer`, including the
/// two-sided interpolation that averages the forward and reversed
/// interpolation so repeated quantiles (discrete / zero-inflated data) map
/// to the centre of their CDF plateau, and the `subsample` of rows used to
/// estimate the quantiles. NaNs are ignored at fit time and passed through.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QuantileTransformer {
    /// Number of quantiles to compute. Clamped to n_samples if larger.
    pub n_quantiles: usize,
    /// Target output distribution.
    pub output_distribution: OutputDistribution,
    /// Maximum number of rows used to estimate the quantiles (rows are
    /// drawn without replacement when `n_samples > subsample`). `None`
    /// uses all rows. Default `Some(10_000)` like sklearn.
    #[serde(default = "default_subsample")]
    pub subsample: Option<usize>,
    /// Seed for the row subsample.
    #[serde(default)]
    pub random_state: u64,
}

impl QuantileTransformer {
    /// Create a new `QuantileTransformer` with defaults (1000 quantiles,
    /// uniform output, subsample 10 000).
    pub fn new() -> Self {
        Self {
            n_quantiles: 1000,
            output_distribution: OutputDistribution::Uniform,
            subsample: default_subsample(),
            random_state: 0,
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

    /// Set the maximum number of rows used to estimate quantiles
    /// (`None` = all rows).
    pub fn with_subsample(mut self, subsample: Option<usize>) -> Self {
        self.subsample = subsample;
        self
    }

    /// Set the seed for the row subsample.
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = seed;
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

/// sklearn's `BOUNDS_THRESHOLD`.
const BOUNDS_THRESHOLD: f64 = 1e-7;

/// Kept for backwards compatibility of the (private) test helper name; the
/// accurate quantile function lives in `crate::special`.
#[cfg(test)]
fn inverse_normal_cdf(p: f64) -> f64 {
    norm_ppf(p)
}

/// `np.interp(x, xp, fp)` for ascending (possibly repeated) `xp`.
///
/// Matches numpy exactly, including for ties: the interval is the last `j`
/// with `xp[j] <= x`, so for a run of equal `xp` values the *last* `fp` of
/// the run is returned.
fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    debug_assert_eq!(xp.len(), fp.len());
    let n = xp.len();
    if n == 0 {
        return f64::NAN;
    }
    if x < xp[0] {
        return fp[0];
    }
    if x > xp[n - 1] {
        return fp[n - 1];
    }
    // last j with xp[j] <= x
    let j = xp.partition_point(|&v| v <= x).saturating_sub(1);
    if j >= n - 1 {
        return fp[n - 1];
    }
    if x == xp[j] {
        return fp[j];
    }
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    slope * (x - xp[j]) + fp[j]
}

impl<F: Float> FitUnsupervised<F> for QuantileTransformer {
    type Fitted = FittedQuantileTransformer<F>;

    fn fit_view(&self, x: ArrayView2<F>) -> Result<Self::Fitted> {
        if x.is_empty() {
            return Err(RustMlError::EmptyInput("input array is empty".into()));
        }
        if self.n_quantiles == 0 {
            return Err(RustMlError::InvalidParameter(
                "n_quantiles must be at least 1".into(),
            ));
        }
        if let Some(sub) = self.subsample {
            if sub == 0 {
                return Err(RustMlError::InvalidParameter(
                    "subsample must be at least 1".into(),
                ));
            }
            if self.n_quantiles > sub {
                return Err(RustMlError::InvalidParameter(format!(
                    "the number of quantiles cannot be greater than the number of samples \
                     used: got {} quantiles and subsample = {}",
                    self.n_quantiles, sub
                )));
            }
        }

        let n_samples = x.nrows();
        let ncols = x.ncols();
        let effective_n = self.n_quantiles.min(n_samples).max(1);

        // np.linspace(0, 1, n_quantiles_)
        let references: Vec<f64> = if effective_n == 1 {
            vec![0.0]
        } else {
            let step = 1.0 / (effective_n - 1) as f64;
            let mut r: Vec<f64> = (0..effective_n).map(|i| i as f64 * step).collect();
            r[effective_n - 1] = 1.0;
            r
        };

        // Row subsample (without replacement), shared by all columns.
        let rows: Option<Vec<usize>> = match self.subsample {
            Some(sub) if sub < n_samples => {
                let mut rng = StdRng::seed_from_u64(self.random_state);
                let mut idx = rand::seq::index::sample(&mut rng, n_samples, sub).into_vec();
                idx.sort_unstable();
                Some(idx)
            }
            _ => None,
        };

        let mut quantiles = Vec::with_capacity(ncols);

        for j in 0..ncols {
            // NaNs are ignored at fit time (sklearn uses `np.nanpercentile`)
            // and pass through `transform` unchanged.
            let column = x.column(j);
            let col = match &rows {
                Some(idx) => sorted_non_nan(idx.iter().map(|&i| column[i])),
                None => sorted_non_nan(column.iter().copied()),
            };

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

impl<F: Float> FittedQuantileTransformer<F> {
    fn apply(&self, x: ArrayView2<F>, inverse: bool) -> Result<Array2<F>> {
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
            let q: Vec<f64> = self.quantiles[j]
                .iter()
                .map(|v| v.to_f64().unwrap())
                .collect();
            for i in 0..x.nrows() {
                let val = x[[i, j]].to_f64().unwrap();
                let out =
                    transform_value(val, &q, &self.references, self.output_distribution, inverse);
                result[[i, j]] = F::from_f64(out).unwrap();
            }
        }
        Ok(result)
    }
}

/// Port of sklearn's `QuantileTransformer._transform_col` for a single
/// value. Maps through one feature's quantile function (forward:
/// data → uniform/normal; inverse: uniform/normal → data).
fn transform_value(
    value: f64,
    quantiles: &[f64],
    references: &[f64],
    dist: OutputDistribution,
    inverse: bool,
) -> f64 {
    // NaN passes through; an all-NaN training column has NaN quantiles and
    // maps everything to NaN.
    if value.is_nan() || quantiles.first().is_none_or(|v| v.is_nan()) {
        return f64::NAN;
    }
    let n = quantiles.len();
    let (lower_x, upper_x, lower_y, upper_y) = if inverse {
        (0.0, 1.0, quantiles[0], quantiles[n - 1])
    } else {
        (quantiles[0], quantiles[n - 1], 0.0, 1.0)
    };
    let mut v = value;
    if inverse && dist == OutputDistribution::Normal {
        v = norm_cdf(v);
    }
    let (is_lower, is_upper) = match dist {
        OutputDistribution::Normal => (
            v - BOUNDS_THRESHOLD < lower_x,
            v + BOUNDS_THRESHOLD > upper_x,
        ),
        OutputDistribution::Uniform => (v == lower_x, v == upper_x),
    };
    let mut out = if !inverse {
        // Average of the forward and the reversed interpolation, so that a
        // run of repeated quantiles maps to the middle of its plateau.
        let rq: Vec<f64> = quantiles.iter().rev().map(|&t| -t).collect();
        let rr: Vec<f64> = references.iter().rev().map(|&t| -t).collect();
        0.5 * (interp(v, quantiles, references) - interp(-v, &rq, &rr))
    } else {
        interp(v, references, quantiles)
    };
    if is_upper {
        out = upper_y;
    }
    if is_lower {
        out = lower_y;
    }
    if !inverse && dist == OutputDistribution::Normal {
        let clip_min = norm_ppf(BOUNDS_THRESHOLD - f64::EPSILON);
        let clip_max = norm_ppf(1.0 - (BOUNDS_THRESHOLD - f64::EPSILON));
        out = norm_ppf(out).clamp(clip_min, clip_max);
    }
    out
}

impl<F: Float> Transform<F> for FittedQuantileTransformer<F> {
    fn transform_view(&self, x: ArrayView2<F>) -> Result<Array2<F>> {
        self.apply(x, false)
    }
}

impl<F: Float> InverseTransform<F> for FittedQuantileTransformer<F> {
    /// Map uniform (or standard-normal) values back to the data scale by
    /// interpolating the stored quantiles (sklearn `inverse_transform`).
    fn inverse_transform_view(&self, x: ArrayView2<F>) -> Result<Array2<F>> {
        self.apply(x, true)
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
    fn test_nan_ignored_at_fit_and_passed_through() {
        // Regression for #42: NaN used to panic in `partial_cmp().unwrap()`.
        let x = array![[1.0], [f64::NAN], [2.0], [3.0], [4.0], [5.0]];
        let qt = QuantileTransformer::new()
            .n_quantiles(5)
            .output_distribution(OutputDistribution::Uniform);
        let fitted = FitUnsupervised::<f64>::fit(&qt, &x).unwrap();
        assert_eq!(fitted.quantiles()[0], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let t = fitted.transform(&x).unwrap();
        assert!(t[[1, 0]].is_nan());
        assert_abs_diff_eq!(t[[3, 0]], 0.5, epsilon = 1e-9);

        // All-NaN column: no panic, NaN out.
        let all_nan = array![[f64::NAN], [f64::NAN]];
        let fitted = FitUnsupervised::<f64>::fit(&qt, &all_nan).unwrap();
        assert!(fitted.transform(&array![[1.0]]).unwrap()[[0, 0]].is_nan());
    }

    fn zero_inflated() -> Array2<f64> {
        array![
            [0.0, 1.0],
            [0.0, 2.0],
            [0.0, 3.0],
            [0.0, 4.0],
            [1.0, 5.0],
            [2.0, 6.0],
            [0.0, 7.0],
            [3.0, 8.0],
            [0.0, 9.0],
            [5.0, 10.0]
        ]
    }

    fn assert_rows(actual: &Array2<f64>, expected: &[[f64; 2]], tol: f64) {
        for (i, row) in expected.iter().enumerate() {
            for (j, &e) in row.iter().enumerate() {
                assert_abs_diff_eq!(actual[[i, j]], e, epsilon = tol);
            }
        }
    }

    #[test]
    fn test_repeated_quantiles_match_sklearn() {
        // sklearn 1.8: QuantileTransformer(n_quantiles=6).fit(X)
        let x = zero_inflated();
        let qt = QuantileTransformer::new().n_quantiles(6);
        let fitted = FitUnsupervised::<f64>::fit(&qt, &x).unwrap();
        assert_eq!(fitted.quantiles()[0][..3], [0.0, 0.0, 0.0]);
        let t = fitted.transform(&x).unwrap();
        let expected_col0 = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.6666666666666667,
            0.7777777777777778,
            0.0,
            0.8571428571428572,
            0.0,
            1.0,
        ];
        for (i, &e) in expected_col0.iter().enumerate() {
            assert_abs_diff_eq!(t[[i, 0]], e, epsilon = 1e-12);
            assert_abs_diff_eq!(t[[i, 1]], i as f64 / 9.0, epsilon = 1e-12);
        }
        // New points, including a value *inside* the plateau of repeated
        // zero quantiles -> 0.5 * (forward + reversed) interpolation.
        let xt = array![[-1.0, 0.5], [0.5, 2.5], [4.0, 11.0], [0.0, 1.0]];
        let t = fitted.transform(&xt).unwrap();
        assert_rows(
            &t,
            &[
                [0.0, 0.0],
                [0.6111111111111112, 0.16666666666666669],
                [0.9285714285714286, 1.0],
                [0.0, 0.0],
            ],
            1e-12,
        );

        let qn = QuantileTransformer::new()
            .n_quantiles(6)
            .output_distribution(OutputDistribution::Normal);
        let fitted_n = FitUnsupervised::<f64>::fit(&qn, &x).unwrap();
        let t = fitted_n.transform(&xt).unwrap();
        assert_rows(
            &t,
            &[
                [-5.199337582605575, -5.199337582605575],
                [0.28221614706250825, -0.967421566101701],
                [1.465233792685523, 5.19933758270342],
                [-5.199337582605575, -5.199337582605575],
            ],
            1e-9,
        );
    }

    #[test]
    fn test_inverse_transform_matches_sklearn() {
        let x = zero_inflated();
        let fitted =
            FitUnsupervised::<f64>::fit(&QuantileTransformer::new().n_quantiles(6), &x).unwrap();
        let u = array![[0.0, 0.0], [0.1, 0.35], [0.5, 0.9], [1.0, 1.0]];
        let inv = fitted.inverse_transform(&u).unwrap();
        assert_rows(
            &inv,
            &[
                [0.0, 1.0],
                [0.0, 4.1499999999999995],
                [0.20000000000000004, 9.1],
                [5.0, 10.0],
            ],
            1e-12,
        );
        let fitted_n = FitUnsupervised::<f64>::fit(
            &QuantileTransformer::new()
                .n_quantiles(6)
                .output_distribution(OutputDistribution::Normal),
            &x,
        )
        .unwrap();
        let z = array![[-6.0, -0.3], [0.2, 1.1], [0.0, 6.0]];
        let inv = fitted_n.inverse_transform(&z).unwrap();
        assert_rows(
            &inv,
            &[
                [0.0, 4.438797200299426],
                [0.3585194188782061, 8.779005451482556],
                [0.20000000000000004, 10.0],
            ],
            1e-9,
        );
    }

    #[test]
    fn test_subsample_limits_rows_used() {
        // 2000 rows, subsample 500: quantiles come from 500 rows only, are
        // deterministic for a fixed seed, and still approximate the CDF.
        let n = 2000;
        let x = Array2::from_shape_fn((n, 1), |(i, _)| i as f64);
        let qt = QuantileTransformer::new()
            .n_quantiles(100)
            .with_subsample(Some(500))
            .with_random_state(7);
        let a = FitUnsupervised::<f64>::fit(&qt, &x).unwrap();
        let b = FitUnsupervised::<f64>::fit(&qt, &x).unwrap();
        assert_eq!(a.quantiles(), b.quantiles());
        let full = FitUnsupervised::<f64>::fit(&qt.clone().with_subsample(None), &x).unwrap();
        assert_ne!(a.quantiles(), full.quantiles());
        let t = a.transform(&array![[1000.0]]).unwrap();
        assert!((t[[0, 0]] - 0.5).abs() < 0.05);
        // n_quantiles > subsample is rejected like sklearn.
        let bad = QuantileTransformer::new()
            .n_quantiles(100)
            .with_subsample(Some(50));
        assert!(FitUnsupervised::<f64>::fit(&bad, &x).is_err());
    }

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

        // With 5 samples and 5 quantiles the result is exactly
        // [0, 0.25, 0.5, 0.75, 1.0] (sklearn does not clip uniform output).
        assert_abs_diff_eq!(transformed[[0, 0]], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(transformed[[1, 0]], 0.25, epsilon = 1e-12);
        assert_abs_diff_eq!(transformed[[2, 0]], 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(transformed[[3, 0]], 0.75, epsilon = 1e-12);
        assert_abs_diff_eq!(transformed[[4, 0]], 1.0, epsilon = 1e-12);
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

        // All values should be in [0, 1]
        for &v in transformed.iter() {
            assert!((0.0..=1.0).contains(&v), "value out of range: {}", v);
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
