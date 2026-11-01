use crate::quantile::{percentile_sorted, sorted_non_nan};
use anofox_ml_core::{FitUnsupervised, Float, InverseTransform, Result, RustMlError, Transform};
use ndarray::Array2;

/// Strategy for computing bin edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BinStrategy {
    /// All bins have equal width: `(max - min) / n_bins`.
    Uniform,
    /// All bins have approximately the same number of samples (quantile-based).
    Quantile,
    /// Bin edges are midpoints between the centres of a 1-D k-means
    /// clustering of the feature (sklearn `strategy="kmeans"`: Lloyd's
    /// algorithm initialised at the uniform-bin midpoints).
    KMeans,
}

/// Encoding strategy for transformed output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum EncodeStrategy {
    /// Each value is replaced by its integer bin index (0-based).
    Ordinal,
    /// Each feature is expanded into `n_bins_[j]` binary columns (one-hot encoding).
    Onehot,
}

/// Parameters for KBinsDiscretizer (unfitted state).
///
/// Bins continuous features into discrete intervals, mirroring
/// `sklearn.preprocessing.KBinsDiscretizer`:
///
/// * `Uniform`, `Quantile` (numpy "linear" percentiles) and `KMeans`
///   strategies;
/// * for `Quantile` / `KMeans`, bins narrower than `1e-8` (duplicate edges)
///   are removed, so the number of bins can differ per feature
///   ([`FittedKBinsDiscretizer::n_bins_per_feature`]);
/// * constant features get a single bin `(-inf, inf)`;
/// * values are assigned with `searchsorted(inner_edges, x, side="right")`;
/// * [`InverseTransform`] maps bin indices back to bin centres.
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
    /// Bin edges per feature. Each inner vec has `n_bins_[j] + 1` values.
    bin_edges: Vec<Vec<F>>,
    /// Requested number of bins.
    n_bins: usize,
    encode: EncodeStrategy,
    /// Effective number of bins per feature (sklearn `n_bins_`).
    #[serde(default)]
    n_bins_per_feature: Vec<usize>,
}

/// `np.linspace(lo, hi, n + 1)`.
fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    let step = (hi - lo) / n as f64;
    let mut e: Vec<f64> = (0..=n).map(|i| lo + i as f64 * step).collect();
    e[n] = hi;
    e
}

/// 1-D k-means (Lloyd) as run by sklearn's KBinsDiscretizer: `n_init = 1`,
/// `init` = midpoints of the uniform bins, `max_iter = 300`,
/// `tol = 1e-4 * var(x)`, empty clusters relocated to the farthest points.
/// Returns the sorted cluster centres.
fn kmeans_1d(x: &[f64], k: usize, lo: f64, hi: f64) -> Vec<f64> {
    let uniform = linspace(lo, hi, k);
    let mut centers: Vec<f64> = (0..k)
        .map(|i| (uniform[i + 1] + uniform[i]) * 0.5)
        .collect();
    let n = x.len();
    let mean = x.iter().sum::<f64>() / n as f64;
    let tol = 1e-4 * x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
    let assign = |centers: &[f64], labels: &mut [usize]| {
        for (i, &v) in x.iter().enumerate() {
            let mut best = 0;
            let mut best_d = f64::INFINITY;
            for (c, &m) in centers.iter().enumerate() {
                let d = (v - m) * (v - m);
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            labels[i] = best;
        }
    };
    let mut labels = vec![0usize; n];
    let mut labels_old = vec![usize::MAX; n];
    for _ in 0..300 {
        assign(&centers, &mut labels);
        // M-step with empty-cluster relocation.
        let mut sums = vec![0.0; k];
        let mut counts = vec![0.0; k];
        for (i, &l) in labels.iter().enumerate() {
            sums[l] += x[i];
            counts[l] += 1.0;
        }
        let empty: Vec<usize> = (0..k).filter(|&c| counts[c] == 0.0).collect();
        if !empty.is_empty() {
            let mut far: Vec<usize> = (0..n).collect();
            let dist = |i: usize| (x[i] - centers[labels[i]]).powi(2);
            far.sort_by(|&a, &b| dist(b).total_cmp(&dist(a)));
            for (e, &c) in empty.iter().enumerate() {
                let i = far[e];
                let old = labels[i];
                sums[old] -= x[i];
                counts[old] -= 1.0;
                sums[c] = x[i];
                counts[c] = 1.0;
            }
        }
        let new_centers: Vec<f64> = (0..k)
            .map(|c| {
                if counts[c] > 0.0 {
                    sums[c] / counts[c]
                } else {
                    centers[c]
                }
            })
            .collect();
        let shift: f64 = centers
            .iter()
            .zip(&new_centers)
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        centers = new_centers;
        if labels == labels_old {
            break;
        }
        if shift <= tol {
            break;
        }
        labels_old.copy_from_slice(&labels);
    }
    centers.sort_by(f64::total_cmp);
    centers
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
        let mut n_bins_per_feature = Vec::with_capacity(ncols);

        for j in 0..ncols {
            // NaNs are ignored when computing the bin edges; at transform
            // time a NaN input yields NaN (ordinal) / an all-zero block (one-hot).
            let col = sorted_non_nan(x.column(j).iter().copied());
            if col.is_empty() {
                return Err(RustMlError::InvalidParameter(format!(
                    "feature {j} contains only NaN values; cannot compute bin edges"
                )));
            }
            let min_val = col[0].to_f64().unwrap();
            let max_val = col[col.len() - 1].to_f64().unwrap();
            if min_val == max_val {
                // sklearn: constant feature -> a single bin (-inf, inf).
                bin_edges.push(vec![F::neg_infinity(), F::infinity()]);
                n_bins_per_feature.push(1);
                continue;
            }

            let mut edges: Vec<f64> = match self.strategy {
                BinStrategy::Uniform => linspace(min_val, max_val, self.n_bins),
                BinStrategy::Quantile => (0..=self.n_bins)
                    .map(|i| {
                        let p = i as f64 / self.n_bins as f64;
                        percentile_sorted(&col, p).to_f64().unwrap()
                    })
                    .collect(),
                BinStrategy::KMeans => {
                    let colf: Vec<f64> = col.iter().map(|v| v.to_f64().unwrap()).collect();
                    let centers = kmeans_1d(&colf, self.n_bins, min_val, max_val);
                    let mut e = Vec::with_capacity(self.n_bins + 1);
                    e.push(min_val);
                    for w in centers.windows(2) {
                        e.push((w[1] + w[0]) * 0.5);
                    }
                    e.push(max_val);
                    e
                }
            };

            // Remove bins whose width is too small (<= 1e-8), as sklearn
            // does for the quantile and kmeans strategies.
            if matches!(self.strategy, BinStrategy::Quantile | BinStrategy::KMeans) {
                let mut kept = Vec::with_capacity(edges.len());
                let mut prev = f64::NAN;
                for (i, &e) in edges.iter().enumerate() {
                    if i == 0 || e - prev > 1e-8 {
                        kept.push(e);
                    }
                    prev = e;
                }
                edges = kept;
            }

            n_bins_per_feature.push(edges.len() - 1);
            bin_edges.push(edges.into_iter().map(|e| F::from_f64(e).unwrap()).collect());
        }

        Ok(FittedKBinsDiscretizer {
            bin_edges,
            n_bins: self.n_bins,
            encode: self.encode,
            n_bins_per_feature,
        })
    }
}

/// Bin index of `val`: `np.searchsorted(edges[1:-1], val, side="right")`.
fn find_bin<F: Float>(val: F, edges: &[F]) -> usize {
    let inner = &edges[1..edges.len() - 1];
    inner.partition_point(|&e| e <= val)
}

impl<F: Float> FittedKBinsDiscretizer<F> {
    /// Effective number of bins per feature (sklearn `n_bins_`). May be
    /// smaller than the requested `n_bins` when duplicate edges were removed
    /// or the feature is constant.
    pub fn n_bins_per_feature(&self) -> Vec<usize> {
        if self.n_bins_per_feature.len() == self.bin_edges.len() {
            self.n_bins_per_feature.clone()
        } else {
            // Models serialised before `n_bins_` existed.
            self.bin_edges.iter().map(|e| e.len() - 1).collect()
        }
    }
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
                        let v = x[[i, j]];
                        if v.is_nan() {
                            result[[i, j]] = F::nan();
                            continue;
                        }
                        let bin = find_bin(v, &self.bin_edges[j]);
                        result[[i, j]] = F::from_usize(bin).unwrap();
                    }
                }
                Ok(result)
            }
            EncodeStrategy::Onehot => {
                let nb = self.n_bins_per_feature();
                let offsets: Vec<usize> = nb
                    .iter()
                    .scan(0, |acc, &b| {
                        let o = *acc;
                        *acc += b;
                        Some(o)
                    })
                    .collect();
                let out_cols: usize = nb.iter().sum();
                let mut result = Array2::<F>::zeros((x.nrows(), out_cols));
                for i in 0..x.nrows() {
                    for j in 0..x.ncols() {
                        let v = x[[i, j]];
                        if v.is_nan() {
                            continue;
                        }
                        let bin = find_bin(v, &self.bin_edges[j]);
                        result[[i, offsets[j] + bin]] = F::one();
                    }
                }
                Ok(result)
            }
        }
    }
}

impl<F: Float> InverseTransform<F> for FittedKBinsDiscretizer<F> {
    /// Map bin indices (ordinal) or one-hot blocks back to the bin centres
    /// (`(edge[k] + edge[k+1]) / 2`). Like sklearn, a constant feature's
    /// single `(-inf, inf)` bin maps to NaN.
    fn inverse_transform(&self, x: &Array2<F>) -> Result<Array2<F>> {
        let nb = self.n_bins_per_feature();
        let n_features = self.bin_edges.len();
        let ordinal: Array2<F> = match self.encode {
            EncodeStrategy::Ordinal => {
                if x.ncols() != n_features {
                    return Err(RustMlError::ShapeMismatch(format!(
                        "expected {} features, got {}",
                        n_features,
                        x.ncols()
                    )));
                }
                x.to_owned()
            }
            EncodeStrategy::Onehot => {
                let total: usize = nb.iter().sum();
                if x.ncols() != total {
                    return Err(RustMlError::ShapeMismatch(format!(
                        "expected {} one-hot columns, got {}",
                        total,
                        x.ncols()
                    )));
                }
                let mut out = Array2::<F>::zeros((x.nrows(), n_features));
                for i in 0..x.nrows() {
                    let mut off = 0;
                    for (j, &b) in nb.iter().enumerate() {
                        let mut best = 0;
                        for k in 1..b {
                            if x[[i, off + k]] > x[[i, off + best]] {
                                best = k;
                            }
                        }
                        out[[i, j]] = F::from_usize(best).unwrap();
                        off += b;
                    }
                }
                out
            }
        };
        let mut result = Array2::<F>::zeros(ordinal.raw_dim());
        let two = F::from_f64(2.0).unwrap();
        for i in 0..ordinal.nrows() {
            for j in 0..n_features {
                let v = ordinal[[i, j]];
                if v.is_nan() {
                    result[[i, j]] = F::nan();
                    continue;
                }
                let k = v.to_f64().unwrap();
                if k < 0.0 || k.fract() != 0.0 || k as usize >= nb[j] {
                    return Err(RustMlError::InvalidParameter(format!(
                        "bin index {k} out of range for feature {j} ({} bins)",
                        nb[j]
                    )));
                }
                let k = k as usize;
                let e = &self.bin_edges[j];
                result[[i, j]] = (e[k] + e[k + 1]) / two;
            }
        }
        Ok(result)
    }
}

impl<F: Float> FittedKBinsDiscretizer<F> {
    /// Return the bin edges per feature.
    pub fn bin_edges(&self) -> &Vec<Vec<F>> {
        &self.bin_edges
    }

    /// Return the requested number of bins (see
    /// [`n_bins_per_feature`](Self::n_bins_per_feature) for the effective
    /// per-feature count).
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
    fn test_nan_ignored_at_fit_and_passed_through() {
        // Regression for #42: NaN used to panic in `partial_cmp().unwrap()`.
        let x = array![[0.0], [f64::NAN], [1.0], [2.0], [3.0], [4.0]];
        for strategy in [BinStrategy::Uniform, BinStrategy::Quantile] {
            let kbd = KBinsDiscretizer::new().n_bins(4).strategy(strategy);
            let fitted = FitUnsupervised::<f64>::fit(&kbd, &x).unwrap();
            assert_eq!(fitted.bin_edges()[0], vec![0.0, 1.0, 2.0, 3.0, 4.0]);
            let t = fitted.transform(&x).unwrap();
            assert!(t[[1, 0]].is_nan());
            assert_abs_diff_eq!(t[[3, 0]], 2.0, epsilon = 1e-12);

            let onehot = KBinsDiscretizer::new()
                .n_bins(4)
                .strategy(strategy)
                .encode(EncodeStrategy::Onehot);
            let fitted = FitUnsupervised::<f64>::fit(&onehot, &x).unwrap();
            let t = fitted.transform(&x).unwrap();
            assert_eq!(t.row(1).sum(), 0.0, "NaN row should be all zeros");
            assert_eq!(t.row(0).sum(), 1.0);
        }
        // An all-NaN column is an error, not a panic.
        let all_nan = array![[f64::NAN], [f64::NAN]];
        assert!(FitUnsupervised::<f64>::fit(&KBinsDiscretizer::new(), &all_nan).is_err());
    }

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

    fn sk_x() -> Array2<f64> {
        array![
            [0.0, 1.0],
            [0.0, 1.2],
            [0.0, 1.1],
            [0.0, 5.0],
            [0.0, 5.3],
            [1.0, 9.0],
            [2.0, 9.4],
            [0.0, 9.9],
            [3.0, 20.0],
            [0.0, 5.1],
            [5.0, 1.05]
        ]
    }

    fn check(
        fitted: &FittedKBinsDiscretizer<f64>,
        edges: &[&[f64]],
        t: &[[f64; 2]],
        inv: &[[f64; 2]],
    ) {
        for (j, e) in edges.iter().enumerate() {
            assert_eq!(fitted.bin_edges()[j].len(), e.len(), "feature {j}");
            for (a, b) in fitted.bin_edges()[j].iter().zip(e.iter()) {
                assert_abs_diff_eq!(*a, *b, epsilon = 1e-10);
            }
        }
        let x = sk_x();
        let out = fitted.transform(&x).unwrap();
        for (i, row) in t.iter().enumerate() {
            assert_eq!(out[[i, 0]], row[0], "row {i}");
            assert_eq!(out[[i, 1]], row[1], "row {i}");
        }
        let back = fitted
            .inverse_transform(&array![[0.0, 0.0], [1.0, 3.0]])
            .unwrap();
        for (i, row) in inv.iter().enumerate() {
            for j in 0..2 {
                assert_abs_diff_eq!(back[[i, j]], row[j], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_quantile_duplicate_edges_removed_like_sklearn() {
        // sklearn 1.8: KBinsDiscretizer(4, strategy="quantile",
        //   quantile_method="linear", encode="ordinal").fit(X)
        let kbd = KBinsDiscretizer::new()
            .n_bins(4)
            .strategy(BinStrategy::Quantile);
        let fitted = FitUnsupervised::<f64>::fit(&kbd, &sk_x()).unwrap();
        assert_eq!(fitted.n_bins_per_feature(), vec![2, 4]);
        check(
            &fitted,
            &[&[0.0, 1.5, 5.0], &[1.0, 1.15, 5.1, 9.2, 20.0]],
            &[
                [0.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [0.0, 2.0],
                [0.0, 2.0],
                [1.0, 3.0],
                [0.0, 3.0],
                [1.0, 3.0],
                [0.0, 2.0],
                [1.0, 0.0],
            ],
            &[[0.75, 1.075], [3.25, 14.6]],
        );
        // One-hot output has sum(n_bins_) columns, no all-zero columns.
        let oh = FitUnsupervised::<f64>::fit(&kbd.clone().encode(EncodeStrategy::Onehot), &sk_x())
            .unwrap();
        let t = oh.transform(&sk_x()).unwrap();
        assert_eq!(t.ncols(), 6);
        for c in 0..6 {
            assert!(t.column(c).sum() > 0.0, "empty one-hot column {c}");
        }
        let back = oh.inverse_transform(&t).unwrap();
        assert_abs_diff_eq!(back[[0, 1]], 1.075, epsilon = 1e-12);
    }

    #[test]
    fn test_kmeans_strategy_matches_sklearn() {
        let kbd = KBinsDiscretizer::new()
            .n_bins(4)
            .strategy(BinStrategy::KMeans);
        let fitted = FitUnsupervised::<f64>::fit(&kbd, &sk_x()).unwrap();
        assert_eq!(fitted.n_bins_per_feature(), vec![4, 4]);
        check(
            &fitted,
            &[
                &[0.0, 1.0625, 2.5, 4.0, 5.0],
                &[
                    1.0,
                    3.1104166666666666,
                    7.283333333333333,
                    14.716666666666667,
                    20.0,
                ],
            ],
            &[
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 2.0],
                [1.0, 2.0],
                [0.0, 2.0],
                [2.0, 3.0],
                [0.0, 1.0],
                [3.0, 0.0],
            ],
            &[[0.53125, 2.0552083333333333], [1.78125, 17.358333333333334]],
        );
    }

    #[test]
    fn test_uniform_strategy_matches_sklearn() {
        let kbd = KBinsDiscretizer::new()
            .n_bins(4)
            .strategy(BinStrategy::Uniform);
        let fitted = FitUnsupervised::<f64>::fit(&kbd, &sk_x()).unwrap();
        check(
            &fitted,
            &[
                &[0.0, 1.25, 2.5, 3.75, 5.0],
                &[1.0, 5.75, 10.5, 15.25, 20.0],
            ],
            &[
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [2.0, 3.0],
                [0.0, 0.0],
                [3.0, 0.0],
            ],
            &[[0.625, 3.375], [1.875, 17.625]],
        );
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

        // sklearn: a constant feature gets a single bin, everything -> 0.
        for &v in transformed.iter() {
            assert_eq!(v, 0.0);
        }
        assert_eq!(fitted.n_bins_per_feature(), vec![1]);
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
