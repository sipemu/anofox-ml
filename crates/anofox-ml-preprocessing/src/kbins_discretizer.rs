use anofox_ml_core::{FitUnsupervised, Float, Result, RustMlError, Transform};
use ndarray::Array2;

/// Strategy for computing bin edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BinStrategy {
    /// All bins have equal width: `(max - min) / n_bins`.
    Uniform,
    /// All bins have approximately the same number of samples (quantile-based).
    Quantile,
}

/// Encoding strategy for transformed output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum EncodeStrategy {
    /// Each value is replaced by its integer bin index (0-based).
    Ordinal,
    /// Each feature is expanded into `n_bins` binary columns (one-hot encoding).
    Onehot,
}

/// Parameters for KBinsDiscretizer (unfitted state).
///
/// Bins continuous features into discrete intervals. Two binning strategies
/// are supported: uniform-width and quantile-based. Output can be ordinal
/// (bin indices) or one-hot encoded.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KBinsDiscretizer {
    /// Number of bins per feature.
    pub n_bins: usize,
    /// Strategy for computing bin edges.
    pub strategy: BinStrategy,
    /// Encoding strategy for the output.
    pub encode: EncodeStrategy,
}

impl KBinsDiscretizer {
    /// Create a new `KBinsDiscretizer` with defaults (5 bins, quantile strategy, ordinal encoding).
    pub fn new() -> Self {
        Self {
            n_bins: 5,
            strategy: BinStrategy::Quantile,
            encode: EncodeStrategy::Ordinal,
        }
    }

    /// Set the number of bins.
    pub fn n_bins(mut self, n_bins: usize) -> Self {
        self.n_bins = n_bins;
        self
    }

    /// Set the binning strategy.
    pub fn strategy(mut self, strategy: BinStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set the encoding strategy.
    pub fn encode(mut self, encode: EncodeStrategy) -> Self {
        self.encode = encode;
        self
    }
}

impl Default for KBinsDiscretizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted KBinsDiscretizer -- holds bin edges per feature.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "F: serde::de::DeserializeOwned"))]
pub struct FittedKBinsDiscretizer<F: Float> {
    /// Bin edges per feature. Each inner vec has `n_bins + 1` values.
    bin_edges: Vec<Vec<F>>,
    n_bins: usize,
    encode: EncodeStrategy,
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

impl<F: Float> FitUnsupervised<F> for KBinsDiscretizer {
    type Fitted = FittedKBinsDiscretizer<F>;

    fn fit(&self, x: &Array2<F>) -> Result<Self::Fitted> {
        if x.is_empty() {
            return Err(RustMlError::EmptyInput("input array is empty".into()));
        }
        if self.n_bins < 2 {
            return Err(RustMlError::InvalidParameter(
                "n_bins must be at least 2".into(),
            ));
        }

        let ncols = x.ncols();
        let mut bin_edges = Vec::with_capacity(ncols);

        for j in 0..ncols {
            let mut col: Vec<F> = x.column(j).to_vec();
            col.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let edges = match self.strategy {
                BinStrategy::Uniform => {
                    let min_val = col[0];
                    let max_val = col[col.len() - 1];
                    let range = max_val - min_val;
                    let step = range / F::from_usize(self.n_bins).unwrap();
                    let mut e = Vec::with_capacity(self.n_bins + 1);
                    for i in 0..=self.n_bins {
                        e.push(min_val + step * F::from_usize(i).unwrap());
                    }
                    e
                }
                BinStrategy::Quantile => {
                    let mut e = Vec::with_capacity(self.n_bins + 1);
                    for i in 0..=self.n_bins {
                        let p = i as f64 / self.n_bins as f64;
                        e.push(percentile_sorted(&col, p));
                    }
                    e
                }
            };

            bin_edges.push(edges);
        }

        Ok(FittedKBinsDiscretizer {
            bin_edges,
            n_bins: self.n_bins,
            encode: self.encode,
        })
    }
}

/// Find the bin index for a value given bin edges.
/// Returns a 0-based bin index in [0, n_bins - 1].
fn find_bin<F: Float>(val: F, edges: &[F], n_bins: usize) -> usize {
    // Binary search: find the rightmost edge <= val
    let mut lo = 0;
    let mut hi = edges.len() - 1;

    // Clamp to first/last bin for out-of-range values
    if val <= edges[0] {
        return 0;
    }
    if val >= edges[edges.len() - 1] {
        return n_bins - 1;
    }

    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if edges[mid] <= val {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    // lo is the index of the left edge of the bin, bin index = lo
    // Clamp to [0, n_bins - 1]
    lo.min(n_bins - 1)
}

impl<F: Float> Transform<F> for FittedKBinsDiscretizer<F> {
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>> {
        let expected_cols = self.bin_edges.len();
        if x.ncols() != expected_cols {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                expected_cols,
                x.ncols()
            )));
        }

        match self.encode {
            EncodeStrategy::Ordinal => {
                let mut result = Array2::<F>::zeros(x.raw_dim());
                for i in 0..x.nrows() {
                    for j in 0..x.ncols() {
                        let bin = find_bin(x[[i, j]], &self.bin_edges[j], self.n_bins);
                        result[[i, j]] = F::from_usize(bin).unwrap();
                    }
                }
                Ok(result)
            }
            EncodeStrategy::Onehot => {
                let out_cols = expected_cols * self.n_bins;
                let mut result = Array2::<F>::zeros((x.nrows(), out_cols));
                for i in 0..x.nrows() {
                    for j in 0..x.ncols() {
                        let bin = find_bin(x[[i, j]], &self.bin_edges[j], self.n_bins);
                        let col_offset = j * self.n_bins + bin;
                        result[[i, col_offset]] = F::one();
                    }
                }
                Ok(result)
            }
        }
    }
}

impl<F: Float> FittedKBinsDiscretizer<F> {
    /// Return the bin edges per feature.
    pub fn bin_edges(&self) -> &Vec<Vec<F>> {
        &self.bin_edges
    }

    /// Return the number of bins.
    pub fn n_bins(&self) -> usize {
        self.n_bins
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_uniform_ordinal() {
        let x = array![
            [0.0, 0.0],
            [2.5, 5.0],
            [5.0, 10.0],
            [7.5, 15.0],
            [10.0, 20.0],
        ];
        let kbd = KBinsDiscretizer::new()
            .n_bins(4)
            .strategy(BinStrategy::Uniform)
            .encode(EncodeStrategy::Ordinal);
        let fitted = FitUnsupervised::<f64>::fit(&kbd, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        // Uniform bins for col 0: [0, 2.5, 5, 7.5, 10]
        // 0.0 -> bin 0, 2.5 -> bin 1, 5.0 -> bin 2, 7.5 -> bin 3, 10.0 -> bin 3
        assert_abs_diff_eq!(transformed[[0, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[1, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[2, 0]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[4, 0]], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_quantile_ordinal() {
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
        let kbd = KBinsDiscretizer::new()
            .n_bins(5)
            .strategy(BinStrategy::Quantile)
            .encode(EncodeStrategy::Ordinal);
        let fitted = FitUnsupervised::<f64>::fit(&kbd, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        // All bin indices should be in [0, 4]
        for &v in transformed.iter() {
            assert!(v >= 0.0 && v <= 4.0, "bin index out of range: {}", v);
        }

        // Values should be non-decreasing (monotonic)
        for i in 1..x.nrows() {
            assert!(
                transformed[[i, 0]] >= transformed[[i - 1, 0]],
                "monotonicity violated at row {}",
                i
            );
        }
    }

    #[test]
    fn test_onehot_encoding() {
        let x = array![[1.0], [3.0], [5.0], [7.0], [9.0]];
        let kbd = KBinsDiscretizer::new()
            .n_bins(3)
            .strategy(BinStrategy::Uniform)
            .encode(EncodeStrategy::Onehot);
        let fitted = FitUnsupervised::<f64>::fit(&kbd, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        // Output should have 3 columns (1 feature * 3 bins)
        assert_eq!(transformed.ncols(), 3);

        // Each row should have exactly one 1.0 and two 0.0
        for i in 0..transformed.nrows() {
            let row_sum: f64 = transformed.row(i).sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_onehot_multiple_features() {
        let x = array![[1.0, 10.0], [5.0, 50.0], [9.0, 90.0]];
        let kbd = KBinsDiscretizer::new()
            .n_bins(3)
            .strategy(BinStrategy::Uniform)
            .encode(EncodeStrategy::Onehot);
        let fitted = FitUnsupervised::<f64>::fit(&kbd, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        // Output should have 6 columns (2 features * 3 bins)
        assert_eq!(transformed.ncols(), 6);

        // Each row: exactly two 1.0 values (one per feature)
        for i in 0..transformed.nrows() {
            let row_sum: f64 = transformed.row(i).sum();
            assert_abs_diff_eq!(row_sum, 2.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_empty_input() {
        let x: Array2<f64> = Array2::zeros((0, 0));
        let kbd = KBinsDiscretizer::default();
        assert!(FitUnsupervised::<f64>::fit(&kbd, &x).is_err());
    }

    #[test]
    fn test_invalid_n_bins() {
        let x = array![[1.0], [2.0], [3.0]];
        let kbd = KBinsDiscretizer::new().n_bins(1);
        assert!(FitUnsupervised::<f64>::fit(&kbd, &x).is_err());
    }

    #[test]
    fn test_shape_mismatch() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let kbd = KBinsDiscretizer::default();
        let fitted = FitUnsupervised::<f64>::fit(&kbd, &x).unwrap();

        let x_wrong = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&x_wrong).is_err());
    }

    #[test]
    fn test_out_of_range_values() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let kbd = KBinsDiscretizer::new()
            .n_bins(3)
            .strategy(BinStrategy::Uniform)
            .encode(EncodeStrategy::Ordinal);
        let fitted = FitUnsupervised::<f64>::fit(&kbd, &x).unwrap();

        // Transform values outside the fitted range
        let x_test = array![[-10.0], [0.0], [3.0], [6.0], [100.0]];
        let transformed = fitted.transform(&x_test).unwrap();

        // Out-of-range should clamp to first/last bin
        assert_abs_diff_eq!(transformed[[0, 0]], 0.0, epsilon = 1e-10); // below min
        assert_abs_diff_eq!(transformed[[4, 0]], 2.0, epsilon = 1e-10); // above max
    }

    #[test]
    fn test_constant_feature() {
        let x = array![[5.0], [5.0], [5.0], [5.0]];
        let kbd = KBinsDiscretizer::new()
            .n_bins(3)
            .strategy(BinStrategy::Uniform)
            .encode(EncodeStrategy::Ordinal);
        let fitted = FitUnsupervised::<f64>::fit(&kbd, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        // All values should map to the same bin (or at least be finite)
        for &v in transformed.iter() {
            assert!(v.is_finite(), "constant feature produced non-finite: {}", v);
        }
    }

    #[test]
    fn test_f32() {
        let x = array![[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let kbd = KBinsDiscretizer::new()
            .n_bins(3)
            .strategy(BinStrategy::Quantile)
            .encode(EncodeStrategy::Ordinal);
        let fitted = FitUnsupervised::<f32>::fit(&kbd, &x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        for &v in transformed.iter() {
            assert!(v.is_finite());
            assert!(v >= 0.0 && v < 3.0);
        }
    }
}
