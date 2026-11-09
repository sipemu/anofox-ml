//! LightGBM-like gradient boosting booster.
//!
//! A pure-Rust implementation of the key LightGBM algorithms:
//!
//! - **Leaf-wise tree growth** with `num_leaves` (best-first expansion via priority queue)
//! - **NaN handling**: learns the best missing direction per split
//! - **GOSS** (Gradient-based One-Side Sampling)
//! - **L1 + L2 regularization**
//! - **Bagging and feature fractions** (`subsample` with `subsample_freq`,
//!   `colsample_bytree`, `colsample_bynode`), drawn from LightGBM's own
//!   generator and seeds
//! - **Early stopping** with an evaluation set
//! - **Categorical features** (Fisher method: sort by grad/hess, then split)
//! - **Custom objectives** (user-provided gradient/hessian)
//! - **Feature importance** (split count and total gain)
//!
//! Not included: distributed training.

use crate::hist_gradient_boosting::EarlyStoppingScoring;
use anofox_ml_core::numpy_random::Generator;
use anofox_ml_core::{Fit, FitWeighted, Predict, Result, RustMlError};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, AsArray, Ix1, Ix2, ShapeBuilder};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

const DEFAULT_MAX_BINS: usize = 255;
/// Special bin index reserved for NaN / missing values.
const MISSING_BIN: u8 = 255;

// ============================================================
// Binning
// ============================================================

/// How features are binned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum BinningMethod {
    /// LightGBM's greedy binning with `min_data_in_bin`; categories with
    /// few samples share the "other" bin.
    #[default]
    LightGbm,
    /// sklearn's `_BinMapper`: midpoints between consecutive distinct values
    /// when there are at most `max_bins` of them, otherwise the
    /// `method="midpoint"` percentiles at `max_bins` equal steps (from a
    /// subsample of 200 000 rows on larger data). Every category gets its own
    /// bin (the first `max_bins` in sorted order; later ones are treated as
    /// missing).
    Sklearn,
}

/// LightGBM's zero threshold (`kZeroThreshold`, the float `1e-35f`):
/// smaller magnitudes count as zero.
const LGBM_ZERO_THRESHOLD: f64 = 1e-35_f32 as f64;

/// LightGBM's `kEpsilon` (the float `1e-15f`).
const LGBM_EPSILON: f64 = 1e-15_f32 as f64;

/// `std::nextafter(x, +inf)` (LightGBM's `GetDoubleUpperBound`).
fn next_up(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        x
    } else if x == 0.0 {
        f64::from_bits(1)
    } else if x > 0.0 {
        f64::from_bits(x.to_bits() + 1)
    } else {
        f64::from_bits(x.to_bits() - 1)
    }
}

/// LightGBM's `GreedyFindBin`: bin upper bounds (the last is `+inf`) for
/// sorted distinct values with their counts.
fn lgbm_greedy_find_bin(
    values: &[f64],
    counts: &[usize],
    max_bin: usize,
    total_cnt: usize,
    min_data_in_bin: usize,
) -> Vec<f64> {
    let n = values.len();
    let mut bounds = Vec::new();
    let push = |bounds: &mut Vec<f64>, v: f64| {
        // CheckDoubleEqualOrdered: skip a bound within one ulp of the last.
        if bounds.last().is_none_or(|&b: &f64| v > next_up(b)) {
            bounds.push(v);
        }
    };
    if n <= max_bin {
        let mut cur = 0;
        for i in 0..n.saturating_sub(1) {
            cur += counts[i];
            if cur >= min_data_in_bin {
                let v = next_up((values[i] + values[i + 1]) / 2.0);
                if bounds.last().is_none_or(|&b: &f64| v > next_up(b)) {
                    bounds.push(v);
                    cur = 0;
                }
            }
        }
        bounds.push(f64::INFINITY);
        return bounds;
    }
    let mut max_bin = max_bin;
    if let Some(limit) = total_cnt.checked_div(min_data_in_bin) {
        max_bin = max_bin.min(limit).max(1);
    }
    let mut mean_bin_size = total_cnt as f64 / max_bin as f64;
    let mut rest_bin_cnt = max_bin as i64;
    let mut rest_sample_cnt = total_cnt as i64;
    let is_big: Vec<bool> = counts.iter().map(|&c| c as f64 >= mean_bin_size).collect();
    for (k, &big) in is_big.iter().enumerate() {
        if big {
            rest_bin_cnt -= 1;
            rest_sample_cnt -= counts[k] as i64;
        }
    }
    mean_bin_size = rest_sample_cnt as f64 / rest_bin_cnt as f64;
    let mut upper = vec![f64::INFINITY; max_bin];
    let mut lower = vec![f64::INFINITY; max_bin];
    let mut bin_cnt = 0;
    lower[0] = values[0];
    let mut cur = 0usize;
    for i in 0..n - 1 {
        if !is_big[i] {
            rest_sample_cnt -= counts[i] as i64;
        }
        cur += counts[i];
        if is_big[i]
            || cur as f64 >= mean_bin_size
            || (is_big[i + 1] && cur as f64 >= (mean_bin_size * 0.5).max(1.0))
        {
            upper[bin_cnt] = values[i];
            bin_cnt += 1;
            lower[bin_cnt] = values[i + 1];
            if bin_cnt >= max_bin - 1 {
                break;
            }
            cur = 0;
            if !is_big[i] {
                rest_bin_cnt -= 1;
                mean_bin_size = rest_sample_cnt as f64 / rest_bin_cnt as f64;
            }
        }
    }
    bin_cnt += 1;
    for i in 0..bin_cnt - 1 {
        push(&mut bounds, next_up((upper[i] + lower[i + 1]) / 2.0));
    }
    bounds.push(f64::INFINITY);
    bounds
}

/// LightGBM's `FindBinWithZeroAsOneBin`: negative and positive values are
/// binned separately around a bin for (near-)zero.
fn lgbm_zero_as_one_bin(
    values: &[f64],
    counts: &[usize],
    max_bin: usize,
    total_cnt: usize,
    min_data_in_bin: usize,
) -> Vec<f64> {
    let (mut left_cnt_data, mut cnt_zero, mut right_cnt_data) = (0, 0, 0);
    for (&v, &c) in values.iter().zip(counts) {
        if v <= -LGBM_ZERO_THRESHOLD {
            left_cnt_data += c;
        } else if v > LGBM_ZERO_THRESHOLD {
            right_cnt_data += c;
        } else {
            cnt_zero += c;
        }
    }
    let left_cnt = values
        .iter()
        .position(|&v| v > -LGBM_ZERO_THRESHOLD)
        .unwrap_or(values.len());
    let mut bounds = Vec::new();
    if left_cnt > 0 && max_bin > 1 {
        let left_max_bin = ((left_cnt_data as f64 / (total_cnt - cnt_zero) as f64)
            * (max_bin - 1) as f64) as usize;
        bounds = lgbm_greedy_find_bin(
            &values[..left_cnt],
            &counts[..left_cnt],
            left_max_bin.max(1),
            left_cnt_data,
            min_data_in_bin,
        );
        if let Some(last) = bounds.last_mut() {
            *last = -LGBM_ZERO_THRESHOLD;
        }
    }
    let right_start = (left_cnt..values.len()).find(|&i| values[i] > LGBM_ZERO_THRESHOLD);
    let right_max_bin = max_bin as i64 - 1 - bounds.len() as i64;
    match right_start {
        Some(rs) if right_max_bin > 0 => {
            let right = lgbm_greedy_find_bin(
                &values[rs..],
                &counts[rs..],
                right_max_bin as usize,
                right_cnt_data,
                min_data_in_bin,
            );
            bounds.push(LGBM_ZERO_THRESHOLD);
            bounds.extend(right);
        }
        _ => bounds.push(f64::INFINITY),
    }
    bounds
}

/// LightGBM's numerical `BinMapper::FindBin` on a dense column: bin edges
/// (the upper bounds without the final `+inf` and the NaN bin). As with
/// dense input to LightGBM, (near-)zeros are counted rather than sampled
/// and re-inserted as one distinct value at their place.
fn lightgbm_bin_edges(col: ArrayView1<f64>, max_bin: usize, min_data_in_bin: usize) -> Vec<f64> {
    let total = col.len();
    let na_cnt = col.iter().filter(|v| v.is_nan()).count();
    let mut values: Vec<f64> = col
        .iter()
        .copied()
        .filter(|v| !v.is_nan() && v.abs() > LGBM_ZERO_THRESHOLD)
        .collect();
    let zero_cnt = total - values.len() - na_cnt;
    values.sort_by(f64::total_cmp);

    let (mut distinct, mut counts): (Vec<f64>, Vec<usize>) = (Vec::new(), Vec::new());
    if values.is_empty() || (values[0] > 0.0 && zero_cnt > 0) {
        distinct.push(0.0);
        counts.push(zero_cnt);
    }
    if let Some(&first) = values.first() {
        distinct.push(first);
        counts.push(1);
    }
    for w in values.windows(2) {
        let (prev, v) = (w[0], w[1]);
        if v > next_up(prev) {
            if prev < 0.0 && v > 0.0 {
                distinct.push(0.0);
                counts.push(zero_cnt);
            }
            distinct.push(v);
            counts.push(1);
        } else {
            // Within an ulp: keep the larger value.
            *distinct.last_mut().unwrap() = v;
            *counts.last_mut().unwrap() += 1;
        }
    }
    if values.last().is_some_and(|&v| v < 0.0) && zero_cnt > 0 {
        distinct.push(0.0);
        counts.push(zero_cnt);
    }
    let mut bounds = if na_cnt == 0 {
        lgbm_zero_as_one_bin(&distinct, &counts, max_bin, total, min_data_in_bin)
    } else {
        lgbm_zero_as_one_bin(
            &distinct,
            &counts,
            max_bin - 1,
            total - na_cnt,
            min_data_in_bin,
        )
    };
    bounds.pop(); // +inf
    bounds
}

/// sklearn's `_find_binning_thresholds` for one column (non-NaN values,
/// sorted).
fn sklearn_thresholds(sorted: &[f64], max_bins: usize) -> Vec<f64> {
    let mut distinct = sorted.to_vec();
    distinct.dedup();
    if distinct.len() <= max_bins {
        return distinct.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect();
    }
    // np.percentile(col, linspace(0, 100, max_bins + 1)[1:-1], method="midpoint"),
    // in numpy's arithmetic: the level `k * (100 / max_bins) / 100`, the
    // virtual index `(floor(i) + ceil(i)) / 2` of `i = (n - 1) q`, and
    // numpy's interpolation `b - (b - a) / 2` between neighbours (whether
    // `i` lands exactly on an integer decides between a value and a
    // midpoint).
    let n = sorted.len();
    let step = 100.0 / max_bins as f64;
    (1..max_bins)
        .map(|k| {
            let q = k as f64 * step / 100.0;
            let i = (n - 1) as f64 * q;
            let virtual_index = 0.5 * (i.floor() + i.ceil());
            let v = if virtual_index >= (n - 1) as f64 {
                sorted[n - 1]
            } else {
                let below = virtual_index.floor();
                let (a, b) = (sorted[below as usize], sorted[below as usize + 1]);
                if virtual_index == below {
                    a
                } else {
                    b - (b - a) * 0.5
                }
            };
            v.min(f64::MAX)
        })
        .collect()
}

#[cfg(test)]
pub(crate) fn sklearn_thresholds_for_test(sorted: &[f64], max_bins: usize) -> Vec<f64> {
    sklearn_thresholds(sorted, max_bins)
}

/// Rows used to place sklearn's bin thresholds: all of them up to 200 000,
/// else a random subset of that size (sklearn's `subsample`).
const SKLEARN_BIN_SUBSAMPLE: usize = 200_000;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct FeatureBins {
    /// Sorted bin edges. For non-NaN values.
    edges: Vec<f64>,
    /// Whether this feature is categorical (different split logic).
    is_categorical: bool,
}

/// Bin features into u8 indices. NaN maps to `MISSING_BIN` (255).
///
/// `min_data_in_bin` controls the minimum number of samples per bin (LightGBM
/// default: 3).  Bins with fewer samples are merged with their neighbours,
/// reducing the effective split resolution and preventing overfitting on small
/// datasets.
fn compute_bins(
    x: ArrayView2<f64>,
    max_bins: usize,
    categorical_features: &[usize],
    min_data_in_bin: usize,
    method: BinningMethod,
    seed: u64,
) -> (Array2<u8>, Vec<FeatureBins>) {
    let n = x.nrows();
    let subsample: Option<Vec<usize>> =
        (method == BinningMethod::Sklearn && n > SKLEARN_BIN_SUBSAMPLE).then(|| {
            let mut rng = StdRng::seed_from_u64(seed);
            rand::seq::index::sample(&mut rng, n, SKLEARN_BIN_SUBSAMPLE).into_vec()
        });
    let p = x.ncols();
    let cat_set: std::collections::HashSet<usize> = categorical_features.iter().copied().collect();

    // The columns are binned in parallel.
    use rayon::prelude::*;
    let columns: Vec<(Vec<u8>, FeatureBins)> = (0..p)
        .into_par_iter()
        .map(|j| {
            let is_categorical = cat_set.contains(&j);

            let mut edges = Vec::new();

            if is_categorical && method == BinningMethod::Sklearn {
                // sklearn ordinal-encodes every category (its estimators reject
                // more than `max_bins` of them).
                edges = x
                    .column(j)
                    .iter()
                    .copied()
                    .filter(|v| !v.is_nan())
                    .collect();
                edges.sort_by(f64::total_cmp);
                edges.dedup();
                edges.truncate(max_bins);
            } else if is_categorical {
                edges = categorical_edges(x.column(j), max_bins, min_data_in_bin);
            } else if method == BinningMethod::Sklearn {
                let mut raw: Vec<f64> = match &subsample {
                    Some(rows) => rows.iter().map(|&i| x[[i, j]]).collect(),
                    None => x.column(j).to_vec(),
                };
                raw.retain(|v| !v.is_nan());
                raw.sort_by(f64::total_cmp);
                if !raw.is_empty() {
                    edges = sklearn_thresholds(&raw, max_bins);
                }
            } else {
                edges = lightgbm_bin_edges(x.column(j), max_bins, min_data_in_bin);
            }

            let column: Vec<u8> = x
                .column(j)
                .iter()
                .map(|&v| {
                    if v.is_nan() {
                        MISSING_BIN
                    } else if is_categorical {
                        // Categories without a bin share the "other" (missing) bin.
                        categorical_bin(v, &edges).map_or(MISSING_BIN, |b| b as u8)
                    } else {
                        numeric_bin(v, &edges)
                    }
                })
                .collect();
            (
                column,
                FeatureBins {
                    edges,
                    is_categorical,
                },
            )
        })
        .collect();

    // Column-major: histograms of bundles and row partitions read one
    // feature at a time.
    let mut binned = Array2::zeros((n, p).f());
    let mut all_bins = Vec::with_capacity(p);
    for (j, (column, bins)) in columns.into_iter().enumerate() {
        binned
            .column_mut(j)
            .assign(&ndarray::ArrayView1::from(&column));
        all_bins.push(bins);
    }

    (binned, all_bins)
}

/// The categories of a categorical feature that get their own bin, sorted by
/// value (the bin index is the position in this list).
///
/// Follows LightGBM's categorical binning: categories are ranked by
/// frequency, and the most frequent ones are kept until a category is rarer
/// than `min_data_in_bin` (the two most frequent are always kept) or all
/// `max_bins - 1` value bins are used. Every other category, like a category
/// unseen in training, falls into the "other" bin, which is the missing bin
/// and always goes right at a categorical split.
fn categorical_edges(
    col: ndarray::ArrayView1<'_, f64>,
    max_bins: usize,
    min_data_in_bin: usize,
) -> Vec<f64> {
    let mut values: Vec<f64> = col.iter().copied().filter(|v| !v.is_nan()).collect();
    values.sort_by(f64::total_cmp);
    let mut counts: Vec<(f64, usize)> = Vec::new();
    for v in values {
        match counts.last_mut() {
            Some((last, c)) if *last == v => *c += 1,
            _ => counts.push((v, 1)),
        }
    }
    // Most frequent first; ties keep the smaller value first.
    counts.sort_by_key(|&(_, c)| std::cmp::Reverse(c));
    let mut edges: Vec<f64> = counts
        .iter()
        .enumerate()
        .take_while(|&(k, &(_, c))| k < 2 || c >= min_data_in_bin)
        .take(max_bins - 1)
        .map(|(_, &(v, _))| v)
        .collect();
    edges.sort_by(f64::total_cmp);
    edges
}

/// Bin index of a non-NaN numeric value. The index is clamped *before* the
/// narrowing cast so it can never wrap around or collide with `MISSING_BIN`.
#[inline]
fn numeric_bin(v: f64, edges: &[f64]) -> u8 {
    edges
        .partition_point(|&e| e < v)
        .min(MISSING_BIN as usize - 1) as u8
}

/// Validate `max_bins` (LightGBM-style: at most 255 value bins; the missing
/// bin is extra). Larger values used to wrap silently in the `u8` bins.
fn validate_max_bins(max_bins: usize) -> Result<()> {
    if !(2..=DEFAULT_MAX_BINS).contains(&max_bins) {
        return Err(RustMlError::InvalidParameter(format!(
            "max_bins must be in [2, {DEFAULT_MAX_BINS}], got {max_bins}"
        )));
    }
    Ok(())
}

/// Index of the category `v` among the sorted category values `edges`
/// (values within 1e-12 match).
#[inline]
fn categorical_bin(v: f64, edges: &[f64]) -> Option<usize> {
    let pos = edges.partition_point(|&e| e < v - 1e-12);
    (pos < edges.len() && (edges[pos] - v).abs() < 1e-12).then_some(pos)
}

/// Bin of one value with existing (training) bin edges.
#[inline]
fn bin_value(v: f64, bins: &FeatureBins) -> u8 {
    if v.is_nan() {
        MISSING_BIN
    } else if bins.is_categorical {
        categorical_bin(v, &bins.edges)
            .map_or(MISSING_BIN, |b| b.min(MISSING_BIN as usize - 1) as u8)
    } else {
        numeric_bin(v, &bins.edges)
    }
}

/// Bin every row of `x` with *existing* (training) bin edges, row-major
/// (prediction walks one row at a time), in parallel over blocks of rows.
/// Works for any layout of `x`.
fn bin_matrix(x: ArrayView2<f64>, all_bins: &[FeatureBins]) -> Array2<u8> {
    use rayon::prelude::*;
    let (n, p) = x.dim();
    let mut out = Array2::<u8>::zeros((n, p));
    if p > 0 {
        out.as_slice_mut()
            .expect("row-major")
            .par_chunks_mut(PREDICT_BLOCK_ROWS * p)
            .enumerate()
            .for_each(|(block, out)| {
                let first = block * PREDICT_BLOCK_ROWS;
                for (j, bins) in all_bins.iter().enumerate() {
                    for (k, o) in out.iter_mut().skip(j).step_by(p).enumerate() {
                        *o = bin_value(x[[first + k, j]], bins);
                    }
                }
            });
    }
    out
}

/// Run a fit on `n_jobs` threads (`None`: the physical cores, at most the
/// current pool's threads). Inside a rayon worker (e.g. parallel
/// cross-validation) the fit stays in the caller's pool.
pub(crate) fn with_fit_threads<T: Send>(
    n_jobs: Option<usize>,
    fit: impl FnOnce() -> T + Send,
) -> T {
    let current = rayon::current_num_threads();
    let threads = n_jobs
        .unwrap_or_else(|| num_cpus::get_physical().min(current))
        .max(1);
    if rayon::current_thread_index().is_some() || threads == current {
        return fit();
    }
    match rayon::ThreadPoolBuilder::new().num_threads(threads).build() {
        Ok(pool) => pool.install(fit),
        Err(_) => fit(),
    }
}

/// Check that a fit can continue a previous one (`warm_start`): what the
/// continuation cannot reproduce is an error.
fn check_warm_start(
    prev_stages: usize,
    prev_learning_rate: f64,
    n_features: usize,
    p: usize,
    model: WarmStartModel,
    opts: &LgbmFitOptions<'_>,
    stopping: bool,
) -> Result<()> {
    let invalid = |m: &str| Err(RustMlError::InvalidParameter(format!("warm start: {m}")));
    if n_features != p {
        return invalid(&format!(
            "X has {p} features, the fitted model {n_features}"
        ));
    }
    if model.n_estimators < prev_stages {
        return invalid(&format!(
            "n_estimators={} is fewer than the {prev_stages} fitted stages",
            model.n_estimators
        ));
    }
    if model.learning_rate != prev_learning_rate {
        return invalid("the learning rate cannot change");
    }
    if model.boosting_type != BoostingType::Gbdt {
        return invalid("only gbdt boosting can be continued (not GOSS or DART)");
    }
    if stopping || opts.eval_set.is_some() || model.early_stopping {
        return invalid("early stopping cannot be continued");
    }
    if opts.init_score.is_some() {
        return invalid("init_score is not supported");
    }
    if model.feature_fraction_per_split < 1.0 {
        return invalid("max_features < 1 is not supported");
    }
    if model.colsample_bynode < 1.0 {
        return invalid("colsample_bynode < 1 is not supported");
    }
    Ok(())
}

/// The configuration a warm start is checked against.
struct WarmStartModel {
    n_estimators: usize,
    learning_rate: f64,
    boosting_type: BoostingType,
    early_stopping: bool,
    feature_fraction_per_split: f64,
    colsample_bynode: f64,
}

/// Bin `x` with the bins of a fitted model, column-major (as
/// `compute_bins`).
fn bin_with(x: ArrayView2<f64>, bins: &[FeatureBins]) -> Array2<u8> {
    let rows = bin_matrix(x, bins);
    let mut binned = Array2::zeros(rows.raw_dim().f());
    binned.assign(&rows);
    binned
}

/// Rows per parallel block of predictions.
const PREDICT_BLOCK_ROWS: usize = 1024;

/// `scores[i] += factor · tree(row i)` for every row of the row-major
/// `binned_rows`, in parallel.
fn add_tree_scores(scores: &mut [f64], tree: &TreeNode, binned_rows: &Array2<u8>, factor: f64) {
    use rayon::prelude::*;
    scores
        .par_iter_mut()
        .with_min_len(4096)
        .enumerate()
        .for_each(|(i, s)| *s += factor * tree.predict_bins(row_of(binned_rows, i)));
}

/// `add_tree_scores` from the rows of the tree's leaves (as built), when
/// they hold every row (no bagging or GOSS): each leaf's value is read once.
fn add_leaf_scores(
    scores: &mut [f64],
    tree: &TreeNode,
    leaf_rows: &[Vec<usize>],
    binned_rows: &Array2<u8>,
    factor: f64,
) {
    if leaf_rows.iter().map(Vec::len).sum::<usize>() != scores.len() {
        return add_tree_scores(scores, tree, binned_rows, factor);
    }
    for rows in leaf_rows {
        let Some(&first) = rows.first() else { continue };
        let value = tree.predict_bins(row_of(binned_rows, first));
        for &i in rows {
            scores[i] += factor * value;
        }
    }
}

/// Row `i` of a row-major binned matrix.
#[inline]
fn row_of(binned: &Array2<u8>, i: usize) -> &[u8] {
    let p = binned.ncols();
    &binned.as_slice().expect("row-major binned matrix")[i * p..(i + 1) * p]
}

/// Number of bins a feature can occupy (missing excluded).
#[inline]
fn n_value_bins(bins: &FeatureBins) -> usize {
    if bins.is_categorical {
        bins.edges.len().min(MISSING_BIN as usize)
    } else {
        (bins.edges.len() + 1).min(MISSING_BIN as usize)
    }
}

/// Validate the eval-set part of [`LgbmFitOptions`] against the training data.
fn validate_eval_opts(opts: &LgbmFitOptions<'_>, n_features: usize) -> Result<()> {
    if let Some((xe, ye)) = opts.eval_set {
        if xe.nrows() != ye.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "eval X has {} rows but eval y has {} elements",
                xe.nrows(),
                ye.len()
            )));
        }
        if xe.ncols() != n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "eval X has {} features but training X has {}",
                xe.ncols(),
                n_features
            )));
        }
        if xe.nrows() == 0 {
            return Err(RustMlError::EmptyInput("eval set is empty".into()));
        }
        if let Some(w) = opts.eval_sample_weight {
            if w.len() != ye.len() {
                return Err(RustMlError::ShapeMismatch(format!(
                    "eval_sample_weight has {} entries but eval y has {}",
                    w.len(),
                    ye.len()
                )));
            }
        }
    }
    Ok(())
}

// ============================================================
// Histograms
// ============================================================

/// Histogram for one feature at one node.
/// Accumulates per-bin gradient + hessian sums.
#[derive(Clone)]
struct Histogram {
    grad_sum: Vec<f64>,
    hess_sum: Vec<f64>,
    count: Vec<u32>,
    /// Gradient/hessian sums for samples with NaN in this feature.
    missing_grad: f64,
    missing_hess: f64,
    missing_count: u32,
}

impl Histogram {
    /// A copy with the counts LightGBM would estimate from the hessians,
    /// `round(hess · cnt_factor)`.
    fn with_estimated_counts(&self, cnt_factor: f64) -> Self {
        let est = |h: f64| (h * cnt_factor + 0.5) as u32;
        Self {
            grad_sum: self.grad_sum.clone(),
            hess_sum: self.hess_sum.clone(),
            count: self.hess_sum.iter().map(|&h| est(h)).collect(),
            missing_grad: self.missing_grad,
            missing_hess: self.missing_hess,
            missing_count: est(self.missing_hess),
        }
    }

    fn new(n_bins: usize) -> Self {
        Self {
            grad_sum: vec![0.0; n_bins],
            hess_sum: vec![0.0; n_bins],
            count: vec![0; n_bins],
            missing_grad: 0.0,
            missing_hess: 0.0,
            missing_count: 0,
        }
    }

    /// `self -= child` (histogram subtraction: the larger child's
    /// histogram from the parent's and the smaller child's).
    fn subtract(&mut self, child: &Histogram) {
        let sub = |a: &mut [f64], b: &[f64]| a.iter_mut().zip(b).for_each(|(a, b)| *a -= b);
        sub(&mut self.grad_sum, &child.grad_sum);
        sub(&mut self.hess_sum, &child.hess_sum);
        self.count
            .iter_mut()
            .zip(&child.count)
            .for_each(|(a, b)| *a -= b);
        self.missing_grad -= child.missing_grad;
        self.missing_hess -= child.missing_hess;
        self.missing_count -= child.missing_count;
    }
}

// ============================================================
// Split finding
// ============================================================

#[derive(Clone)]
#[allow(dead_code)]
struct BestSplit {
    feature: usize,
    bin_threshold: u8,
    gain: f64,
    left_value: f64,
    right_value: f64,
    left_count: usize,
    right_count: usize,
    /// true = missing values go left, false = right.
    missing_left: bool,
    /// For categorical features: the ordered permutation used.
    /// For numeric: None.
    categorical_order: Option<Vec<u8>>,
    /// LightGBM's numeric scans: the left child's gradient and hessian sums
    /// as scanned (the hessian with `kEpsilon`), from which LightGBM derives
    /// the children's totals.
    lgbm_left_sums: Option<(f64, f64)>,
}

/// Compute the leaf value with L1 + L2 regularization.
/// Uses the proximal operator for the L1 term: soft-threshold then divide by (H + L2).
#[inline]
/// sklearn's node value `-G / (H + l2 + 1e-15)`, clipped to `bounds`
/// (`compute_node_value`; no L1 penalty).
fn sklearn_node_value(grad_sum: f64, hess_sum: f64, l2: f64, (lo, hi): OutputBounds) -> f64 {
    (-grad_sum / (hess_sum + l2 + 1e-15)).clamp(lo, hi)
}

fn leaf_value(grad_sum: f64, hess_sum: f64, l1: f64, l2: f64) -> f64 {
    if grad_sum.abs() <= l1 {
        0.0
    } else {
        -(grad_sum.signum() * (grad_sum.abs() - l1)) / (hess_sum + l2)
    }
}

/// Gain for a given (grad_sum, hess_sum) pair with L1 + L2 regularization.
#[inline]
fn leaf_gain(grad_sum: f64, hess_sum: f64, l1: f64, l2: f64) -> f64 {
    if grad_sum.abs() <= l1 {
        0.0
    } else {
        let thresholded = grad_sum.abs() - l1;
        thresholded * thresholded / (hess_sum + l2)
    }
}

/// L1 soft-thresholded gradient sum.
#[inline]
fn threshold_l1(grad_sum: f64, l1: f64) -> f64 {
    grad_sum.signum() * (grad_sum.abs() - l1).max(0.0)
}

/// Gain of a leaf with a *given* (possibly clamped) output:
/// `-(2 T(G) w + (H + l2) w^2)` (LightGBM's `GetLeafGainGivenOutput`).
/// Equals [`leaf_gain`] when `w` is the unconstrained optimum.
#[inline]
fn leaf_gain_given_output(grad_sum: f64, hess_sum: f64, l1: f64, l2: f64, output: f64) -> f64 {
    -(2.0 * threshold_l1(grad_sum, l1) * output + (hess_sum + l2) * output * output)
}

/// Output bounds `(lo, hi)` a leaf must respect for monotone constraints.
type OutputBounds = (f64, f64);

const UNBOUNDED: OutputBounds = (f64::NEG_INFINITY, f64::INFINITY);

/// Child output bounds after splitting a leaf with bounds `parent` on a
/// feature with monotone constraint `mono` (LightGBM "basic" method): for a
/// constrained feature the bounds are split at the midpoint of the two
/// child outputs, otherwise both children inherit the parent's bounds.
fn child_bounds(
    parent: OutputBounds,
    mono: i8,
    left_value: f64,
    right_value: f64,
) -> (OutputBounds, OutputBounds) {
    let (lo, hi) = parent;
    let mid = 0.5 * (left_value + right_value);
    match mono {
        1 => ((lo, mid), (mid, hi)),
        -1 => ((mid, hi), (lo, mid)),
        _ => (parent, parent),
    }
}

/// Monotone constraint of `feat` (categorical features are never constrained).
#[inline]
fn monotone_of(feat: usize, all_bins: &[FeatureBins], monotone_constraints: &[i8]) -> i8 {
    if all_bins[feat].is_categorical {
        0
    } else {
        monotone_constraints.get(feat).copied().unwrap_or(0)
    }
}

/// From this many bins (over the candidate features) a node's splits are
/// searched in parallel.
const PARALLEL_SPLIT_BINS: usize = 4096;

/// Sums per bin value of one feature at one node: gradients and hessians,
/// and (in sklearn mode) counts, indexed by the `u8` bin (the missing bin
/// last).
struct BinSums {
    gh: [(f64, f64); 256],
    count: [u32; 256],
}

/// Bin sums of the bin column `col` over the rows `rows` (`None`: every row,
/// in order), `gh` holding their gradients and hessians in that order.
/// Counts are kept with `COUNT`.
fn bin_sums<const COUNT: bool>(col: &[u8], rows: Option<&[u32]>, gh: &[(f32, f32)]) -> BinSums {
    let mut s = BinSums {
        gh: [(0.0, 0.0); 256],
        count: [0; 256],
    };
    let mut add = |b: u8, (g, h): (f32, f32)| {
        let a = &mut s.gh[b as usize];
        a.0 += f64::from(g);
        a.1 += f64::from(h);
        if COUNT {
            s.count[b as usize] += 1;
        }
    };
    match rows {
        Some(rows) => {
            for (&i, &gh) in rows.iter().zip(gh) {
                add(col[i as usize], gh);
            }
        }
        None => {
            for (&b, &gh) in col.iter().zip(gh) {
                add(b, gh);
            }
        }
    }
    s
}

/// The histogram of a feature with `n_bins` value bins from its bin sums.
fn histogram_of(s: &BinSums, n_bins: usize) -> Histogram {
    let m = MISSING_BIN as usize;
    Histogram {
        grad_sum: s.gh[..n_bins].iter().map(|a| a.0).collect(),
        hess_sum: s.gh[..n_bins].iter().map(|a| a.1).collect(),
        count: s.count[..n_bins].to_vec(),
        missing_grad: s.gh[m].0,
        missing_hess: s.gh[m].1,
        missing_count: s.count[m],
    }
}

/// LightGBM's `FixHistogram`: the sums of bin `bin` (`MISSING_BIN` for the
/// missing values) as the node's totals minus the other bins, in bin order.
fn fix_histogram(hist: &mut Histogram, bin: u8, (grad, hess): (f64, f64)) {
    let (mut g, mut h) = (grad, hess);
    for (i, (bg, bh)) in hist.grad_sum.iter().zip(&hist.hess_sum).enumerate() {
        if i != bin as usize {
            g -= bg;
            h -= bh;
        }
    }
    if bin == MISSING_BIN {
        (hist.missing_grad, hist.missing_hess) = (g, h);
    } else {
        g -= hist.missing_grad;
        h -= hist.missing_hess;
        (hist.grad_sum[bin as usize], hist.hess_sum[bin as usize]) = (g, h);
    }
}

/// The bins LightGBM derives with `FixHistogram`: per numeric feature its
/// `most_freq_bin` (the bin of 0 unless another bin holds at least
/// `kSparseThreshold` = 70% of the rows), unless that is its first bin.
fn lgbm_fixed_bins(
    binned_x: &Array2<u8>,
    all_bins: &[FeatureBins],
    has_missing: &[bool],
) -> Vec<Option<u8>> {
    let n = binned_x.nrows();
    all_bins
        .iter()
        .enumerate()
        .map(|(j, b)| {
            if b.is_categorical || n == 0 {
                return None;
            }
            let n_bins = n_value_bins(b);
            // LightGBM's bin order: the value bins, then the missing one.
            let index = |bin: u8| {
                if bin == MISSING_BIN {
                    n_bins
                } else {
                    bin as usize
                }
            };
            let mut counts = vec![0usize; n_bins + usize::from(has_missing[j])];
            for &bin in binned_x.column(j) {
                if let Some(c) = counts.get_mut(index(bin)) {
                    *c += 1;
                }
            }
            let mut most = 0;
            for (k, &c) in counts.iter().enumerate() {
                if c > counts[most] {
                    most = k;
                }
            }
            let default = numeric_bin(0.0, &b.edges) as usize;
            if most != default && (counts[most] as f64 / n as f64) < 0.7 {
                most = default;
            }
            (most > 0).then(|| {
                if most == n_bins {
                    MISSING_BIN
                } else {
                    most as u8
                }
            })
        })
        .collect()
}

/// Rows per chunk of a large node's parallel partition (LightGBM mode; see
/// `build_lgbm_tree`).
const PARTITION_CHUNK_ROWS: usize = 16_384;

/// Gradient / hessian totals of `rows`, summed in row order (as
/// `node_totals`).
fn row_totals(rows: &[usize], gradients: &[f64], hessians: &[f64]) -> (f64, f64) {
    rows.iter().fold((-0.0, -0.0), |t, &i| {
        (t.0 + gradients[i], t.1 + hessians[i])
    })
}

/// Partitions `rows` in place and stably, the rows going left first, and
/// returns their number. Branch-free: every row is written to both sides
/// (the right ones to `scratch`), and only the side it goes to advances.
fn partition_in_place(
    rows: &mut [usize],
    goes_left: impl Fn(usize) -> bool,
    scratch: &mut Vec<usize>,
) -> usize {
    scratch.clear();
    scratch.resize(rows.len(), 0);
    let (mut n_left, mut n_right) = (0, 0);
    for k in 0..rows.len() {
        let i = rows[k];
        let l = goes_left(i);
        rows[n_left] = i;
        scratch[n_right] = i;
        n_left += usize::from(l);
        n_right += usize::from(!l);
    }
    rows[n_left..].copy_from_slice(&scratch[..n_right]);
    n_left
}

/// Below this many (sample, feature) pairs histograms are built serially.
const PARALLEL_HISTOGRAM_WORK: usize = 1 << 16;

/// Per-feature gradient/hessian histograms of the rows `indices` (whose
/// gradient / hessian totals are `totals`), one per entry of
/// `feature_indices`, in parallel over the features for large nodes. Each feature sums its rows in order. Counts are kept only with
/// `with_counts` (sklearn mode; LightGBM estimates them from the hessians,
/// and the counts are then 0).
///
/// The node's gradients and hessians are gathered once, in row order and as
/// float32 (they are float32 values, as in LightGBM and sklearn), so that
/// every feature reads them sequentially; a node holding every row in order
/// (the root without bagging) reads the bin columns sequentially too.
#[allow(clippy::too_many_arguments)]
fn build_histograms(
    binned_x: &Array2<u8>,
    all_bins: &[FeatureBins],
    bundles: Option<&FeatureBundles>,
    with_counts: bool,
    gradients: &[f64],
    hessians: &[f64],
    indices: &[usize],
    totals: (f64, f64),
    feature_indices: &[usize],
    fixed_bins: &[Option<u8>],
) -> Vec<Histogram> {
    use rayon::prelude::*;
    let n_rows = binned_x.nrows();
    let parallel = indices.len() * feature_indices.len() >= PARALLEL_HISTOGRAM_WORK;
    let all_in_order = indices.len() == n_rows && indices.is_sorted();
    let to_f32 = |v: f64| {
        let r = v as f32;
        debug_assert!(
            f64::from(r) == v || v.is_nan(),
            "gradient {v} is not a float32"
        );
        r
    };
    let ordered: Vec<(f32, f32)> = indices
        .iter()
        .map(|&i| (to_f32(gradients[i]), to_f32(hessians[i])))
        .collect();
    let rows32: Option<Vec<u32>> =
        (!all_in_order).then(|| indices.iter().map(|&i| i as u32).collect());
    let sums = |col: &[u8]| {
        if with_counts {
            bin_sums::<true>(col, rows32.as_deref(), &ordered)
        } else {
            bin_sums::<false>(col, rows32.as_deref(), &ordered)
        }
    };

    // The bin sums of the bundles holding any of the features.
    let mut needed: Vec<usize> = bundles.map_or_else(Vec::new, |b| {
        feature_indices
            .iter()
            .filter_map(|&f| b.bundle_of[f].map(|(id, _)| id))
            .collect()
    });
    needed.sort_unstable();
    needed.dedup();
    let bundle_sums: Vec<BinSums> = if parallel {
        needed
            .par_iter()
            .map(|&id| sums(&bundles.unwrap().columns[id]))
            .collect()
    } else {
        needed
            .iter()
            .map(|&id| sums(&bundles.unwrap().columns[id]))
            .collect()
    };
    let totals = (
        totals.0,
        totals.1,
        if with_counts { indices.len() as u32 } else { 0 },
    );

    let one = |feat: usize| {
        let n_bins = n_value_bins(&all_bins[feat]);
        if let Some(at) = bundles.and_then(|b| b.bundle_of[feat]) {
            let k = needed.binary_search(&at.0).unwrap();
            return bundles
                .unwrap()
                .member_histogram(at, n_bins, &bundle_sums[k], totals);
        }
        let col = binned_x.column(feat);
        let owned;
        let col = match col.as_slice() {
            Some(c) => c,
            None => {
                owned = col.to_vec();
                &owned
            }
        };
        let mut hist = histogram_of(&sums(col), n_bins);
        if let Some(bin) = fixed_bins.get(feat).copied().flatten() {
            fix_histogram(&mut hist, bin, (totals.0, totals.1));
        }
        hist
    };
    if parallel {
        feature_indices.par_iter().map(|&f| one(f)).collect()
    } else {
        feature_indices.iter().map(|&f| one(f)).collect()
    }
}

// ============================================================
// Exclusive feature bundling
// ============================================================

/// Groups of features searched for a bundle a feature can join (the most
/// recent ones, as LightGBM's `max_search_group`).
const MAX_SEARCH_BUNDLES: usize = 100;

/// One feature of a bundle.
struct BundleMember {
    /// The feature's most frequent bin (`MISSING_BIN` for missing), which
    /// is not stored in the bundle.
    default_bin: u8,
    /// The feature's other bins present in the training data, ascending
    /// (`MISSING_BIN` last); the k-th is bundle bin `offset + k`.
    bins: Vec<u8>,
    offset: usize,
}

/// LightGBM's exclusive feature bundling (EFB, `enable_bundle`). Features
/// that are never off their most frequent bin in the same row (typically
/// one-hot or sparse columns) share one bin column: bundle bin 0 means
/// every member is at its most frequent bin, the others name one member's
/// bin. A node then takes one pass over its rows per bundle instead of one
/// per feature, and each member's histogram is read off the bundle's,
/// its most frequent bin being the node totals minus its other bins (as
/// LightGBM's `FixHistogram`).
///
/// Bundling is lossless: only exactly exclusive features are bundled
/// (LightGBM tolerates `n / 10000` conflicting rows, so from 10 000 rows on
/// it can bundle features this does not). Splits, row partitions and
/// predictions keep using the per-feature bins.
struct FeatureBundles {
    /// Per bundle: the bundle bin of every row.
    columns: Vec<Vec<u8>>,
    members: Vec<Vec<BundleMember>>,
    /// Per feature: its bundle and its position there, `None` when it is
    /// on its own.
    bundle_of: Vec<Option<(usize, usize)>>,
}

/// Rows of one feature off its most frequent bin, and that bin.
fn non_default_rows(col: ArrayView1<u8>) -> (u8, Vec<usize>) {
    let mut counts = [0usize; 256];
    for &b in col {
        counts[b as usize] += 1;
    }
    // The first of the most frequent bins.
    let default_bin = (0..256).rev().max_by_key(|&b| counts[b]).unwrap() as u8;
    let rows = (0..col.len()).filter(|&i| col[i] != default_bin).collect();
    (default_bin, rows)
}

/// Greedy bundles of features in the order `order`: each feature joins the
/// most recent bundle (of the last `MAX_SEARCH_BUNDLES`) it conflicts with
/// in no row and whose bins it fits into.
fn greedy_bundles(
    order: &[usize],
    rows: &[Vec<usize>],
    n_bins: &[usize],
    n: usize,
) -> Vec<Vec<usize>> {
    let words = n.div_ceil(64);
    let mut bundles: Vec<Vec<usize>> = Vec::new();
    let mut used: Vec<Vec<u64>> = Vec::new();
    let mut total_rows: Vec<usize> = Vec::new();
    let mut total_bins: Vec<usize> = Vec::new();
    for &f in order {
        let fits = |g: usize| {
            total_rows[g] + rows[f].len() <= n
                && total_bins[g] + n_bins[f] <= 256
                && rows[f]
                    .iter()
                    .all(|&i| used[g][i / 64] & (1 << (i % 64)) == 0)
        };
        let target = (0..bundles.len())
            .rev()
            .take(MAX_SEARCH_BUNDLES)
            .find(|&g| fits(g));
        let g = target.unwrap_or_else(|| {
            bundles.push(Vec::new());
            used.push(vec![0; words]);
            total_rows.push(0);
            total_bins.push(1);
            bundles.len() - 1
        });
        bundles[g].push(f);
        total_rows[g] += rows[f].len();
        total_bins[g] += n_bins[f];
        for &i in &rows[f] {
            used[g][i / 64] |= 1 << (i % 64);
        }
    }
    bundles
}

impl FeatureBundles {
    /// Bundles of the columns of `binned_x`, `None` when no two features
    /// can share one. As LightGBM, features are bundled in their own order
    /// and in descending order of their non-default counts, and the
    /// grouping with fewer bundles is kept.
    fn find(binned_x: &Array2<u8>) -> Option<Self> {
        let (n, p) = binned_x.dim();
        if p < 2 {
            return None;
        }
        let (defaults, rows): (Vec<u8>, Vec<Vec<usize>>) =
            (0..p).map(|f| non_default_rows(binned_x.column(f))).unzip();
        let mut present: Vec<Vec<u8>> = (0..p)
            .map(|f| {
                let col = binned_x.column(f);
                let mut seen = [false; 256];
                for &i in &rows[f] {
                    seen[col[i] as usize] = true;
                }
                (0..=255u8).filter(|&b| seen[b as usize]).collect()
            })
            .collect();
        let n_bins: Vec<usize> = present.iter().map(Vec::len).collect();
        let natural: Vec<usize> = (0..p).collect();
        let mut by_count = natural.clone();
        by_count.sort_by_key(|&f| std::cmp::Reverse(rows[f].len()));
        let first = greedy_bundles(&natural, &rows, &n_bins, n);
        let second = greedy_bundles(&by_count, &rows, &n_bins, n);
        let groups = if second.len() < first.len() {
            second
        } else {
            first
        };
        if groups.len() == p {
            return None;
        }

        let mut bundle_of = vec![None; p];
        let mut columns = Vec::new();
        let mut members = Vec::new();
        for group in groups.into_iter().filter(|g| g.len() > 1) {
            let id = columns.len();
            let mut column = vec![0u8; n];
            let mut offset = 1;
            let mut group_members = Vec::with_capacity(group.len());
            for f in group {
                bundle_of[f] = Some((id, group_members.len()));
                let bins = std::mem::take(&mut present[f]);
                let mut bundle_bin = [0u8; 256];
                for (k, &b) in bins.iter().enumerate() {
                    bundle_bin[b as usize] = (offset + k) as u8;
                }
                let col = binned_x.column(f);
                for &i in &rows[f] {
                    column[i] = bundle_bin[col[i] as usize];
                }
                let width = bins.len();
                group_members.push(BundleMember {
                    default_bin: defaults[f],
                    bins,
                    offset,
                });
                offset += width;
            }
            columns.push(column);
            members.push(group_members);
        }
        Some(Self {
            columns,
            members,
            bundle_of,
        })
    }

    /// Histogram of the `member`-th feature of bundle `bundle` from the
    /// bundle's bin sums and the node totals.
    fn member_histogram(
        &self,
        (bundle, member): (usize, usize),
        n_bins: usize,
        sums: &BinSums,
        totals: (f64, f64, u32),
    ) -> Histogram {
        let m = &self.members[bundle][member];
        let mut hist = Histogram::new(n_bins);
        let (mut g_def, mut h_def, mut c_def) = totals;
        for (k, &b) in m.bins.iter().enumerate() {
            let bb = m.offset + k;
            let ((g, h), c) = (sums.gh[bb], sums.count[bb]);
            g_def -= g;
            h_def -= h;
            c_def -= c;
            set_bin(&mut hist, b, g, h, c);
        }
        set_bin(&mut hist, m.default_bin, g_def, h_def, c_def);
        hist
    }
}

fn set_bin(hist: &mut Histogram, bin: u8, grad: f64, hess: f64, count: u32) {
    if bin == MISSING_BIN {
        hist.missing_grad = grad;
        hist.missing_hess = hess;
        hist.missing_count = count;
    } else {
        let b = bin as usize;
        hist.grad_sum[b] = grad;
        hist.hess_sum[b] = hess;
        hist.count[b] = count;
    }
}

/// Gradient / hessian totals of a node.
fn node_totals(gradients: &[f64], hessians: &[f64], indices: &[usize]) -> (f64, f64) {
    (
        indices.iter().map(|&i| gradients[i]).sum(),
        indices.iter().map(|&i| hessians[i]).sum(),
    )
}

/// Best split of categorical feature `feat` (LightGBM's
/// `FindBestThresholdCategoricalInner`). The left child is a set of
/// categories; the "other"/missing bin always goes right.
///
/// With at most `max_cat_to_onehot` bins (the "other" bin included) each
/// category is tried alone against the rest. Otherwise the categories with
/// at least `cat_smooth` samples are sorted by `G / (H + cat_smooth)` and
/// prefixes of up to `max_cat_threshold` categories are tried from both ends
/// of that order, with `reg_lambda + cat_l2` as the children's L2 penalty and
/// at least `min_data_per_group` samples per added group and on the right.
/// `bounds` is `Some` in monotone-constrained mode.
#[allow(clippy::too_many_arguments)]
fn best_categorical_split(
    hist: &Histogram,
    totals: (f64, f64),
    total_count: usize,
    feat: usize,
    n_categories: usize,
    params: &LgbmParams,
    bounds: Option<OutputBounds>,
    parent_gain: f64,
) -> Option<BestSplit> {
    let cat = &params.categorical;
    let (total_grad, total_hess) = totals;
    let (l1, mcs, mcw) = (
        params.reg_alpha,
        params.min_child_samples,
        params.min_child_weight,
    );
    let one_hot = n_categories < cat.max_cat_to_onehot;
    let l2 = params.reg_lambda + if one_hot { 0.0 } else { cat.cat_l2 };

    // Gain and child outputs of sending the categories summed in `left` left.
    let evaluate = |lg: f64, lh: f64| -> Option<(f64, f64, f64)> {
        let (rg, rh) = (total_grad - lg, total_hess - lh);
        let mut lv = leaf_value(lg, lh, l1, l2);
        let mut rv = leaf_value(rg, rh, l1, l2);
        let gain = if let Some((lo, hi)) = bounds {
            lv = lv.clamp(lo, hi);
            rv = rv.clamp(lo, hi);
            0.5 * (leaf_gain_given_output(lg, lh, l1, l2, lv)
                + leaf_gain_given_output(rg, rh, l1, l2, rv)
                - parent_gain)
        } else {
            0.5 * (leaf_gain(lg, lh, l1, l2) + leaf_gain(rg, rh, l1, l2) - parent_gain)
        };
        (gain > params.min_split_gain).then_some((gain, lv, rv))
    };

    let mut best: Option<(f64, f64, f64, Vec<u8>, usize)> = None;
    let mut consider = |cand: Option<(f64, f64, f64)>, left: &[u8], left_count: usize| {
        if let Some((gain, lv, rv)) = cand {
            if gain > best.as_ref().map_or(f64::NEG_INFINITY, |b| b.0) {
                best = Some((gain, lv, rv, left.to_vec(), left_count));
            }
        }
    };

    let bins = 0..n_categories.min(MISSING_BIN as usize) as u8;
    if one_hot {
        for b in bins {
            let k = b as usize;
            let (lc, lh) = (hist.count[k] as usize, hist.hess_sum[k]);
            let rc = total_count - lc;
            if lc < mcs || lh < mcw || rc < mcs || total_hess - lh < mcw {
                continue;
            }
            consider(evaluate(hist.grad_sum[k], lh), &[b], lc);
        }
    } else {
        let ctr = |b: u8| hist.grad_sum[b as usize] / (hist.hess_sum[b as usize] + cat.cat_smooth);
        let mut sorted: Vec<u8> = bins
            .filter(|&b| hist.count[b as usize] as f64 >= cat.cat_smooth)
            .collect();
        sorted.sort_by(|&a, &b| ctr(a).total_cmp(&ctr(b)));
        let max_num_cat = cat.max_cat_threshold.min(sorted.len().div_ceil(2));
        let reversed: Vec<u8> = sorted.iter().rev().copied().collect();
        for order in [&sorted, &reversed] {
            let (mut lg, mut lh, mut lc, mut group) = (0.0, 0.0, 0usize, 0usize);
            for (i, &b) in order.iter().take(max_num_cat).enumerate() {
                let k = b as usize;
                lg += hist.grad_sum[k];
                lh += hist.hess_sum[k];
                lc += hist.count[k] as usize;
                group += hist.count[k] as usize;
                if lc < mcs || lh < mcw {
                    continue;
                }
                let rc = total_count - lc;
                if rc < mcs || rc < cat.min_data_per_group || total_hess - lh < mcw {
                    break;
                }
                if group < cat.min_data_per_group {
                    continue;
                }
                group = 0;
                consider(evaluate(lg, lh), &order[..=i], lc);
            }
        }
    }

    best.map(
        |(gain, left_value, right_value, left, left_count)| BestSplit {
            feature: feat,
            bin_threshold: *left.last().expect("non-empty left set"),
            gain,
            left_value,
            right_value,
            left_count,
            right_count: total_count - left_count,
            missing_left: false,
            categorical_order: Some(left),
            lgbm_left_sums: None,
        },
    )
}

/// sklearn's categorical split (`_find_best_bin_to_split_category`, Fisher's
/// grouping): the categories with a support of at least 10 (in hessian
/// units scaled to samples) are sorted by `G / (H + 10)`, the missing values
/// counting as a category when the training data had any, and prefixes of up
/// to half of them are tried from both ends of that order. Low-support
/// categories always go right. Gains use sklearn's `_split_gain`.
#[allow(clippy::too_many_arguments)]
fn sklearn_categorical_split(
    hist: &Histogram,
    (total_grad, total_hess): (f64, f64),
    total_count: usize,
    feat: usize,
    n_categories: usize,
    params: &LgbmParams,
    bounds: OutputBounds,
    node_loss: f64,
) -> Option<BestSplit> {
    const MIN_CAT_SUPPORT: f64 = 10.0;
    let (mcs, mcw, l2) = (
        params.min_child_samples,
        params.min_child_weight,
        params.reg_lambda,
    );
    let support_factor = total_count as f64 / total_hess;
    // (bin, gradient sum, hessian sum, count, sort value)
    let mut cats: Vec<(u8, f64, f64, usize, f64)> = (0..n_categories.min(MISSING_BIN as usize))
        .map(|b| {
            (
                b as u8,
                hist.grad_sum[b],
                hist.hess_sum[b],
                hist.count[b] as usize,
            )
        })
        .chain(params.has_missing[feat].then_some((
            MISSING_BIN,
            hist.missing_grad,
            hist.missing_hess,
            hist.missing_count as usize,
        )))
        .filter(|&(_, _, h, _)| h * support_factor >= MIN_CAT_SUPPORT)
        .map(|(b, g, h, c)| (b, g, h, c, g / (h + MIN_CAT_SUPPORT)))
        .collect();
    if cats.len() <= 1 {
        return None;
    }
    cats.sort_by(|a, b| a.4.total_cmp(&b.4));

    let n_used = cats.len();
    // (gain, left value, right value, left count, forward scan, index of
    // the last category sent left)
    let mut best: Option<(f64, f64, f64, usize, bool, usize)> = None;
    for forward in [true, false] {
        let middle = n_used.div_ceil(2) - usize::from(!forward);
        let (mut lg, mut lh, mut lc) = (0.0, 0.0, 0usize);
        for i in 0..middle {
            let k = if forward { i } else { n_used - 1 - i };
            let (_, g, h, c, _) = cats[k];
            lc += c;
            lh += h;
            lg += g;
            let (rc, rh, rg) = (total_count - lc, total_hess - lh, total_grad - lg);
            if lc < mcs || lh < mcw {
                continue;
            }
            if rc < mcs || rh < mcw {
                break;
            }
            let lv = sklearn_node_value(lg, lh, l2, bounds);
            let rv = sklearn_node_value(rg, rh, l2, bounds);
            let mut gain = node_loss;
            gain -= lg * lv;
            gain -= rg * rv;
            let gain = 0.5 * gain;
            if gain > params.min_split_gain && gain > best.as_ref().map_or(-1.0, |b| b.0) {
                best = Some((gain, lv, rv, lc, forward, k));
            }
        }
    }

    best.map(|(gain, left_value, right_value, left_count, forward, k)| {
        let left = if forward { &cats[..=k] } else { &cats[k..] };
        let missing_left = left.iter().any(|c| c.0 == MISSING_BIN);
        let mut order: Vec<u8> = left
            .iter()
            .map(|c| c.0)
            .filter(|&b| b != MISSING_BIN)
            .collect();
        order.sort_unstable();
        BestSplit {
            feature: feat,
            bin_threshold: order.last().copied().unwrap_or(MISSING_BIN),
            gain,
            left_value,
            right_value,
            left_count,
            right_count: total_count - left_count,
            missing_left,
            categorical_order: Some(order),
            lgbm_left_sums: None,
        }
    })
}

/// Best split of a node from its histograms (`hists[k]` belongs to
/// `feature_indices[k]`).
///
/// With monotone constraints, child outputs are clamped to the node's
/// `bounds`, the gain is computed for the clamped outputs, and splits whose
/// outputs violate the constraint direction are rejected.
///
/// In sklearn mode (`params.sklearn_compat`) numeric candidates are scanned
/// in sklearn's order (missing values right with the bins ascending, then
/// missing values left with the bins descending), gains use sklearn's
/// `_split_gain` arithmetic, and missing values the training data did not
/// have go to the larger child. A candidate replaces the best one only when
/// strictly better, so all of this decides ties exactly as sklearn does.
#[allow(clippy::too_many_arguments)]
fn find_best_split(
    hists: &[Histogram],
    totals: (f64, f64),
    total_count: usize,
    feature_indices: &[usize],
    all_bins: &[FeatureBins],
    params: &LgbmParams,
    bounds: OutputBounds,
    parent_value: f64,
    groups: &[usize],
    feature_rng: Option<&mut Generator>,
    node_features: Option<&[usize]>,
) -> Option<BestSplit> {
    let LgbmParams {
        min_child_samples,
        min_child_weight,
        min_split_gain,
        reg_alpha: l1,
        reg_lambda: l2,
        ref monotone_constraints,
        ..
    } = *params;
    let mut best: Option<BestSplit> = None;
    let (total_grad, total_hess) = totals;

    // Constrained mode: any monotone constraint is active. Outputs are then
    // clamped to `bounds` and gains evaluated for the clamped outputs.
    let constrained = monotone_constraints.iter().any(|&m| m != 0);
    let (lo, hi) = bounds;
    // LightGBM's gain shift is the unconstrained gain of the node.
    let parent_gain = if params.sklearn_compat && constrained {
        leaf_gain_given_output(total_grad, total_hess, l1, l2, parent_value)
    } else if params.sklearn_compat {
        leaf_gain(total_grad, total_hess, l1, l2)
    } else {
        leaf_gain(total_grad, total_hess + 2.0 * LGBM_EPSILON, l1, l2)
    };
    // sklearn's loss of the current node, `G * value` with the node's
    // (bounded) value.
    let sklearn_node_loss = total_grad * parent_value;
    let allowed = |feat: usize| {
        params.interaction_cst.is_empty()
            || groups
                .iter()
                .any(|&g| params.interaction_cst[g].contains(&feat))
    };

    // sklearn moves the categorical features in front of the numeric ones,
    // so they win ties.
    let mut positions: Vec<usize> = (0..feature_indices.len()).collect();
    if params.sklearn_compat {
        positions.sort_by_key(|&k| !all_bins[feature_indices[k]].is_categorical);
    }
    positions.retain(|&k| allowed(feature_indices[k]));
    // LightGBM's `feature_fraction_bynode` (drawn by the caller).
    if let Some(sub) = node_features {
        positions.retain(|&k| sub.contains(&feature_indices[k]));
    }
    // sklearn's `max_features`: a random subset of the allowed features for
    // every node it evaluates (a shuffled mask, drawn from the fit's
    // generator).
    if let Some(rng) = feature_rng {
        let n_allowed = positions.len();
        let n_sub = ((params.feature_fraction_per_split * n_allowed as f64).ceil() as usize).max(1);
        let mut mask: Vec<bool> = (0..n_allowed).map(|k| k < n_sub).collect();
        rng.shuffle(&mut mask);
        let mut keep = mask.into_iter();
        positions.retain(|_| keep.next().unwrap_or(false));
    }
    // LightGBM keeps no counts in its histograms: it estimates them from the
    // hessians, `round(hess · n / (H + 2 kEpsilon))`.
    let cnt_factor = total_count as f64 / (total_hess + 2.0 * LGBM_EPSILON);
    let est_count = |h: f64| (h * cnt_factor + 0.5) as usize;
    // Each feature's first best candidate (in parallel for many features),
    // then the first best over the features in order: the outcome of one
    // sequential scan.
    let feature_best = |feat: usize, hist: &Histogram| -> Option<BestSplit> {
        let best: std::cell::RefCell<Option<BestSplit>> = std::cell::RefCell::new(None);
        // Gain of the best candidate so far (read by the scans' fast path).
        let best_gain = std::cell::Cell::new(0.0f64);
        // LightGBM compares the unshifted gains `G_L²/H_L + G_R²/H_R` within
        // a scan (`scan_raw`: the scan's best so far) with `lgbm_shift`
        // (its `min_gain_shift`; the engine's `min_split_gain` is halved).
        let scan_raw = std::cell::Cell::new(f64::NEG_INFINITY);
        let lgbm_shift = parent_gain + 2.0 * min_split_gain;
        if all_bins[feat].is_categorical {
            let estimated;
            let hist = if params.sklearn_compat {
                hist
            } else {
                estimated = hist.with_estimated_counts(cnt_factor);
                &estimated
            };
            let n_categories = all_bins[feat].edges.len();
            let candidate = if params.sklearn_compat {
                sklearn_categorical_split(
                    hist,
                    totals,
                    total_count,
                    feat,
                    n_categories,
                    params,
                    bounds,
                    sklearn_node_loss,
                )
            } else {
                best_categorical_split(
                    hist,
                    totals,
                    total_count,
                    feat,
                    n_categories,
                    params,
                    constrained.then_some(bounds),
                    parent_gain,
                )
            };
            if let Some(split) = candidate {
                if split.gain > best.borrow().as_ref().map_or(0.0, |b| b.gain) {
                    *best.borrow_mut() = Some(split);
                }
            }
            return best.into_inner();
        }
        // Gain of sending `left` left (`right = totals - left`); keeps the
        // candidate if it is strictly better than the best so far.
        let consider = |(left_grad, left_hess, left_count): (f64, f64, usize),
                        (right_grad, right_hess, right_count): (f64, f64, usize),
                        bin: u8,
                        missing_left: bool| {
            if left_count < min_child_samples || left_hess < min_child_weight {
                return;
            }
            if right_count < min_child_samples || right_hess < min_child_weight {
                return;
            }

            let (left_val, right_val, gain);
            if params.sklearn_compat {
                // sklearn's `_split_gain`, in its order of operations: with
                // integer gradient sums (L1, quantile) mirrored splits tie
                // mathematically and the rounding decides. Halved to the
                // scale of the other branches (exact).
                left_val = sklearn_node_value(left_grad, left_hess, l2, bounds);
                right_val = sklearn_node_value(right_grad, right_hess, l2, bounds);
                match monotone_of(feat, all_bins, monotone_constraints) {
                    1 if left_val > right_val => return,
                    -1 if left_val < right_val => return,
                    _ => {}
                }
                let mut g = sklearn_node_loss;
                g -= left_grad * left_val;
                g -= right_grad * right_val;
                gain = 0.5 * g;
            } else if constrained {
                left_val = leaf_value(left_grad, left_hess, l1, l2).clamp(lo, hi);
                right_val = leaf_value(right_grad, right_hess, l1, l2).clamp(lo, hi);
                // Monotone constraint check: feature with monotone=+1
                // requires left_value <= right_value (predictions
                // increase as bin increases), -1 the opposite.
                match monotone_of(feat, all_bins, monotone_constraints) {
                    1 if left_val > right_val => return,
                    -1 if left_val < right_val => return,
                    _ => {}
                }
                let left_g = leaf_gain_given_output(left_grad, left_hess, l1, l2, left_val);
                let right_g = leaf_gain_given_output(right_grad, right_hess, l1, l2, right_val);
                let raw = left_g + right_g;
                if !(raw > lgbm_shift && raw > scan_raw.get()) {
                    return;
                }
                scan_raw.set(raw);
                gain = 0.5 * (raw - parent_gain);
            } else {
                let raw = leaf_gain(left_grad, left_hess, l1, l2)
                    + leaf_gain(right_grad, right_hess, l1, l2);
                // The outputs only for a new best candidate.
                if !(raw > lgbm_shift && raw > scan_raw.get()) {
                    return;
                }
                scan_raw.set(raw);
                gain = 0.5 * (raw - parent_gain);
                left_val = leaf_value(left_grad, left_hess, l1, l2);
                right_val = leaf_value(right_grad, right_hess, l1, l2);
            }

            if !params.sklearn_compat
                || (gain > min_split_gain && gain > best.borrow().as_ref().map_or(0.0, |b| b.gain))
            {
                best_gain.set(gain);
                *best.borrow_mut() = Some(BestSplit {
                    feature: feat,
                    bin_threshold: bin,
                    gain,
                    left_value: left_val,
                    right_value: right_val,
                    left_count,
                    right_count,
                    missing_left,
                    categorical_order: None,
                    lgbm_left_sums: (!params.sklearn_compat).then_some((left_grad, left_hess)),
                });
            }
        };
        // Left sums of the bins up to `bin`, plus the missing values when
        // they go left.
        let scan_ascending = |missing_left: bool, thresholds: &[u8]| {
            let mut left = if missing_left {
                (
                    hist.missing_grad,
                    hist.missing_hess,
                    hist.missing_count as usize,
                )
            } else {
                (0.0, 0.0, 0)
            };
            for &bin in thresholds {
                let b = bin as usize;
                left.0 += hist.grad_sum[b];
                left.1 += hist.hess_sum[b];
                left.2 += hist.count[b] as usize;
                let right = (
                    total_grad - left.0,
                    total_hess - left.1,
                    total_count - left.2,
                );
                consider(left, right, bin, missing_left);
            }
        };

        if params.sklearn_compat {
            // sklearn: missing values right with the bins ascending, then (if
            // the training data had any) missing values left with the bins
            // descending, accumulating the right child as sklearn does. With
            // missing values, the last value bin is a threshold too (all
            // values left, the missing ones right).
            let n_bins = n_value_bins(&all_bins[feat]);
            let has_missing = params.has_missing[feat];
            let thresholds: Vec<u8> = (0..n_bins - 1 + usize::from(has_missing))
                .map(|b| b as u8)
                .collect();
            scan_ascending(false, &thresholds);
            if has_missing {
                let mut right = (0.0, 0.0, 0usize);
                for bin in (0..n_bins.saturating_sub(1)).rev() {
                    let b = bin + 1;
                    right.0 += hist.grad_sum[b];
                    right.1 += hist.hess_sum[b];
                    right.2 += hist.count[b] as usize;
                    let left = (
                        total_grad - right.0,
                        total_hess - right.1,
                        total_count - right.2,
                    );
                    consider(left, right, bin as u8, true);
                }
            }
        } else {
            // LightGBM's `FindBestThresholdSequentially`: the right child
            // accumulated from the top bin down (missing values left), then,
            // for features with missing values, the left child from the
            // bottom up (missing values right). The hessian sums start at
            // `kEpsilon`, the node's at `2 kEpsilon`, and the counts are
            // the estimates of `hist` (see above).
            let n_bins = n_value_bins(&all_bins[feat]);
            let has_nan = params.has_missing[feat];
            let sum_hess = total_hess + 2.0 * LGBM_EPSILON;
            let other = |(g, h, c): (f64, f64, usize)| {
                (total_grad - g, sum_hess - h, total_count.saturating_sub(c))
            };
            // Without constraints, a candidate goes to `consider` only when
            // it beats the best one (the same gain, computed here inline).
            let promising = |(lg, lh, lc): (f64, f64, usize), (rg, rh, rc): (f64, f64, usize)| {
                if constrained {
                    return true;
                }
                if lc < min_child_samples
                    || lh < min_child_weight
                    || rc < min_child_samples
                    || rh < min_child_weight
                {
                    return false;
                }
                let raw = leaf_gain(lg, lh, l1, l2) + leaf_gain(rg, rh, l1, l2);
                raw > lgbm_shift && raw > scan_raw.get()
            };
            // LightGBM's `num_bin` counts the missing bin.
            if has_nan && n_bins < 2 {
                // Only the missing values on the right.
                let right = (
                    hist.missing_grad,
                    LGBM_EPSILON + hist.missing_hess,
                    est_count(hist.missing_hess),
                );
                if right.2 <= total_count {
                    consider(other(right), right, 0, false);
                }
            } else {
                let mut right = (0.0, LGBM_EPSILON, 0usize);
                for b in (1..n_bins).rev() {
                    right.0 += hist.grad_sum[b];
                    right.1 += hist.hess_sum[b];
                    right.2 += est_count(hist.hess_sum[b]);
                    if right.2 > total_count {
                        break;
                    }
                    let left = other(right);
                    if promising(left, right) {
                        consider(left, right, (b - 1) as u8, true);
                    }
                }
                if has_nan {
                    // A scan of its own; it replaces the first one's split
                    // only if better than that one's shifted gain shifted
                    // back (LightGBM's `best_gain > output->gain +
                    // min_gain_shift`, which rounds).
                    let first = best.take();
                    let first_raw = scan_raw.replace(f64::NEG_INFINITY);
                    let mut left = (0.0, LGBM_EPSILON, 0usize);
                    for b in 0..n_bins {
                        left.0 += hist.grad_sum[b];
                        left.1 += hist.hess_sum[b];
                        left.2 += est_count(hist.hess_sum[b]);
                        if left.2 > total_count {
                            break;
                        }
                        let right = other(left);
                        if promising(left, right) {
                            consider(left, right, b as u8, false);
                        }
                    }
                    let second = best.take();
                    *best.borrow_mut() = match (first, second) {
                        (Some(f), Some(_))
                            if scan_raw.get() <= (first_raw - lgbm_shift) + lgbm_shift =>
                        {
                            Some(f)
                        }
                        (f, s) => s.or(f),
                    };
                }
            }
        }
        best.into_inner()
    };
    let n_bins: usize = positions
        .iter()
        .map(|&k| n_value_bins(&all_bins[feature_indices[k]]))
        .sum();
    let per_feature: Vec<Option<BestSplit>> = if n_bins >= PARALLEL_SPLIT_BINS {
        use rayon::prelude::*;
        positions
            .par_iter()
            .with_min_len(4)
            .map(|&k| feature_best(feature_indices[k], &hists[k]))
            .collect()
    } else {
        positions
            .iter()
            .map(|&k| feature_best(feature_indices[k], &hists[k]))
            .collect()
    };
    for split in per_feature.into_iter().flatten() {
        if split.gain > best.as_ref().map_or(0.0, |b| b.gain) {
            best = Some(split);
        }
    }

    // Missing values the training data did not have: sklearn sends them to
    // the child with more samples, LightGBM treats them as 0.
    if let Some(split) = best.as_mut() {
        if !params.has_missing[split.feature] {
            if params.sklearn_compat {
                split.missing_left = split.left_count > split.right_count;
            } else if split.categorical_order.is_none() {
                split.missing_left =
                    numeric_bin(0.0, &all_bins[split.feature].edges) <= split.bin_threshold;
            }
        }
    }

    best
}

// ============================================================
// Tree structure
// ============================================================

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
enum TreeNode {
    Leaf {
        value: f64,
    },
    Internal {
        feature: usize,
        bin_threshold: u8,
        /// true = missing goes left.
        missing_left: bool,
        /// For categorical features: ordered permutation (bins <= threshold go left).
        categorical_order: Option<Vec<u8>>,
        /// Bins that go left, derived from `categorical_order` on first use.
        #[serde(skip)]
        categorical_left: Box<std::sync::OnceLock<Vec<bool>>>,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

/// Bins routed left by a categorical split: those up to and including the
/// threshold bin in `order`.
fn categorical_left_mask(order: &[u8], threshold: u8) -> Vec<bool> {
    let mut mask = vec![false; MISSING_BIN as usize + 1];
    let thresh_pos = order.iter().position(|&b| b == threshold).unwrap_or(0);
    for &b in order.iter().take(thresh_pos + 1) {
        mask[b as usize] = true;
    }
    mask
}

/// Whether `bin` goes left at a split.
#[inline]
fn goes_left(
    bin: u8,
    bin_threshold: u8,
    missing_left: bool,
    categorical_left: Option<&[bool]>,
) -> bool {
    if bin == MISSING_BIN {
        missing_left
    } else if let Some(mask) = categorical_left {
        mask[bin as usize]
    } else {
        bin <= bin_threshold
    }
}

impl TreeNode {
    /// Multiply every leaf value by `f` (LightGBM's `Tree::Shrinkage`).
    fn scale(&mut self, f: f64) {
        match self {
            TreeNode::Leaf { value } => *value *= f,
            TreeNode::Internal { left, right, .. } => {
                left.scale(f);
                right.scale(f);
            }
        }
    }

    /// Add `b` to every leaf value (LightGBM's `Tree::AddBias`).
    fn add_bias(&mut self, b: f64) {
        match self {
            TreeNode::Leaf { value } => *value += b,
            TreeNode::Internal { left, right, .. } => {
                left.add_bias(b);
                right.add_bias(b);
            }
        }
    }

    /// Prediction for one binned row.
    fn predict_bins(&self, bins: &[u8]) -> f64 {
        let mut node = self;
        loop {
            match node {
                TreeNode::Leaf { value } => return *value,
                TreeNode::Internal {
                    feature,
                    bin_threshold,
                    missing_left,
                    categorical_order,
                    categorical_left,
                    left,
                    right,
                } => {
                    let mask = categorical_order.as_ref().map(|order| {
                        categorical_left
                            .get_or_init(|| categorical_left_mask(order, *bin_threshold))
                            .as_slice()
                    });
                    node = if goes_left(bins[*feature], *bin_threshold, *missing_left, mask) {
                        left
                    } else {
                        right
                    };
                }
            }
        }
    }
}

// ============================================================
// Leaf-wise tree builder
// ============================================================

/// Candidate leaf for priority queue. Higher gain = higher priority.
struct LeafCandidate {
    node_id: usize,
    gain: f64,
    split: BestSplit,
    indices: Vec<usize>,
    /// The node's histograms (for the children's histogram subtraction).
    hists: Vec<Histogram>,
    depth: usize,
    /// Output bounds of this leaf (monotone constraints).
    bounds: OutputBounds,
    /// Interaction groups still open to this node's descendants.
    groups: Vec<usize>,
    /// The node's gradient and hessian totals.
    totals: (f64, f64),
}

impl PartialEq for LeafCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.gain == other.gain
    }
}
impl Eq for LeafCandidate {}
impl PartialOrd for LeafCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for LeafCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.gain
            .partial_cmp(&other.gain)
            .unwrap_or(Ordering::Equal)
    }
}

/// Flat node storage during construction.
enum BuildNode {
    Leaf {
        value: f64,
    },
    Internal {
        feature: usize,
        bin_threshold: u8,
        missing_left: bool,
        categorical_order: Option<Vec<u8>>,
        left: usize,
        right: usize,
    },
}

/// Build a tree using leaf-wise (best-first) expansion.
///
/// `feature_gain_out` and `feature_split_count_out` are accumulators that get
/// updated with per-feature gain and split counts respectively. Returns the
/// tree and the rows of `root_indices` in each of its leaves.
#[allow(clippy::too_many_arguments)]
fn build_lgbm_tree(
    binned_x: &Array2<u8>,
    bundles: Option<&FeatureBundles>,
    gradients: &[f64],
    hessians: &[f64],
    root_indices: Vec<usize>,
    feature_indices: &[usize],
    all_bins: &[FeatureBins],
    params: &LgbmParams,
    feature_gain_out: &mut [f64],
    feature_split_count_out: &mut [u32],
    feature_rng: &mut Option<Generator>,
    mut col_sampler: Option<&mut LgbmColSampler>,
) -> (TreeNode, Vec<Vec<usize>>) {
    let mut nodes: Vec<BuildNode> = Vec::new();

    // Root is initially a leaf with its value (0 in sklearn, whose root
    // value only enters the split gains).
    let root_totals = node_totals(gradients, hessians, &root_indices);
    let root_value = if params.sklearn_compat {
        0.0
    } else {
        leaf_value(
            root_totals.0,
            root_totals.1,
            params.reg_alpha,
            params.reg_lambda,
        )
    };
    nodes.push(BuildNode::Leaf { value: root_value });

    let mut best_for = |hists: &[Histogram],
                        totals,
                        count,
                        bounds,
                        value,
                        groups: &[usize],
                        node_features: Option<&[usize]>| {
        find_best_split(
            hists,
            totals,
            count,
            feature_indices,
            all_bins,
            params,
            bounds,
            value,
            groups,
            feature_rng.as_mut(),
            node_features,
        )
    };
    let mut draw_node_features = || col_sampler.as_mut().and_then(|c| c.node_features());
    let root_groups: Vec<usize> = (0..params.interaction_cst.len()).collect();

    // Try to split the root.
    let root_hists = build_histograms(
        binned_x,
        all_bins,
        bundles,
        params.sklearn_compat,
        gradients,
        hessians,
        &root_indices,
        root_totals,
        feature_indices,
        &params.fixed_bins,
    );
    // Neither sklearn nor LightGBM evaluates a root it cannot split (which
    // matters for the draws of their feature subsampling).
    let root_splittable = root_indices.len() >= 2 * params.min_child_samples
        && (!params.sklearn_compat || root_totals.1 >= params.min_child_weight);
    let root_split = root_splittable
        .then(|| {
            let node_features = draw_node_features();
            best_for(
                &root_hists,
                root_totals,
                root_indices.len(),
                UNBOUNDED,
                root_value,
                &root_groups,
                node_features.as_deref(),
            )
        })
        .flatten();

    // The rows of the finished leaves.
    let mut leaf_rows: Vec<Vec<usize>> = Vec::new();
    let mut heap: BinaryHeap<LeafCandidate> = BinaryHeap::new();
    if let Some(split) = root_split {
        heap.push(LeafCandidate {
            node_id: 0,
            gain: split.gain,
            split,
            indices: root_indices,
            hists: root_hists,
            depth: 0,
            bounds: UNBOUNDED,
            groups: root_groups,
            totals: root_totals,
        });
    } else {
        leaf_rows.push(root_indices);
    }

    // Number of leaves = number of leaf nodes, starts at 1 (root as leaf).
    let mut n_leaves = 1;

    while n_leaves < params.num_leaves {
        let candidate = match heap.pop() {
            Some(c) => c,
            None => break,
        };

        // Depth check
        if let Some(max_d) = params.max_depth {
            if candidate.depth >= max_d {
                leaf_rows.push(candidate.indices);
                continue;
            }
        }

        let split = candidate.split;
        let indices = candidate.indices;

        // Partition samples
        let mask = split
            .categorical_order
            .as_ref()
            .map(|order| categorical_left_mask(order, split.bin_threshold));
        let col = binned_x.column(split.feature);
        let col = col.as_slice().expect("column-major binned matrix");
        let left = |i: usize| {
            goes_left(
                col[i],
                split.bin_threshold,
                split.missing_left,
                mask.as_deref(),
            )
        };
        // LightGBM's numeric scans give the children's totals (below).
        let with_totals = split.lgbm_left_sums.is_none();
        let (left_idx, right_idx, left_totals, right_totals) = {
            let mut indices = indices;
            let totals = |rows: &[usize]| {
                if with_totals {
                    row_totals(rows, gradients, hessians)
                } else {
                    (-0.0, -0.0)
                }
            };
            if params.sklearn_compat || indices.len() <= PARTITION_CHUNK_ROWS {
                let n_left = partition_in_place(&mut indices, left, &mut Vec::new());
                let right = indices.split_off(n_left);
                let (lt, rt) = (totals(&indices), totals(&right));
                (indices, right, lt, rt)
            } else {
                // Large nodes (LightGBM numerics): fixed chunks in parallel,
                // joined in order; the totals are the chunks' sums added in
                // chunk order (independent of the thread count).
                use rayon::prelude::*;
                let parts: Vec<_> = indices
                    .par_chunks_mut(PARTITION_CHUNK_ROWS)
                    .map_init(Vec::new, |scratch, chunk| {
                        let n_left = partition_in_place(chunk, left, scratch);
                        let (l, r) = chunk.split_at(n_left);
                        (n_left, totals(l), totals(r))
                    })
                    .collect();
                // The left rows are compacted in place (each chunk's move
                // only overwrites rows already copied), the right ones
                // copied out.
                let n_left = parts.iter().map(|p| p.0).sum();
                let mut right = Vec::with_capacity(indices.len() - n_left);
                let (mut lt, mut rt) = ((-0.0f64, -0.0f64), (-0.0f64, -0.0f64));
                let mut at = 0;
                for (k, (nl, ls, rs)) in parts.into_iter().enumerate() {
                    let start = k * PARTITION_CHUNK_ROWS;
                    let end = (start + PARTITION_CHUNK_ROWS).min(indices.len());
                    right.extend_from_slice(&indices[start + nl..end]);
                    indices.copy_within(start..start + nl, at);
                    at += nl;
                    lt = (lt.0 + ls.0, lt.1 + ls.1);
                    rt = (rt.0 + rs.0, rt.1 + rs.1);
                }
                indices.truncate(n_left);
                (indices, right, lt, rt)
            }
        };

        // LightGBM takes the children's totals from its scan
        // (`SplitInfo::left_sum_gradient`, ...).
        let (left_totals, right_totals) = match split.lgbm_left_sums {
            Some((lg, lh)) => {
                let (g, h) = candidate.totals;
                (
                    (lg, lh - LGBM_EPSILON),
                    (g - lg, h + 2.0 * LGBM_EPSILON - lh - LGBM_EPSILON),
                )
            }
            None => (left_totals, right_totals),
        };

        // Create left and right leaf nodes
        let left_id = nodes.len();
        nodes.push(BuildNode::Leaf {
            value: split.left_value,
        });
        let right_id = nodes.len();
        nodes.push(BuildNode::Leaf {
            value: split.right_value,
        });

        // Convert the parent from Leaf to Internal
        nodes[candidate.node_id] = BuildNode::Internal {
            feature: split.feature,
            bin_threshold: split.bin_threshold,
            missing_left: split.missing_left,
            categorical_order: split.categorical_order.clone(),
            left: left_id,
            right: right_id,
        };

        // Track split gains for feature importance (accumulated across trees)
        feature_gain_out[split.feature] += split.gain;
        feature_split_count_out[split.feature] += 1;

        n_leaves += 1; // One leaf became two leaves (net +1)

        // Propagate output bounds to the children (LightGBM "basic"
        // monotone method) so that deeper splits — on any feature — can
        // never break global monotonicity.
        let (left_bounds, right_bounds) = child_bounds(
            candidate.bounds,
            monotone_of(split.feature, all_bins, &params.monotone_constraints),
            split.left_value,
            split.right_value,
        );

        // The children keep the interaction groups containing the split
        // feature.
        let groups: Vec<usize> = candidate
            .groups
            .iter()
            .copied()
            .filter(|&g| params.interaction_cst[g].contains(&split.feature))
            .collect();

        let depth = candidate.depth + 1;
        let splittable = |n: usize| {
            n >= 2 * params.min_child_samples && params.max_depth.is_none_or(|d| depth < d)
        };
        let (mut left_ok, mut right_ok) = (splittable(left_idx.len()), splittable(right_idx.len()));
        // Both finalize the children once the leaf budget is spent, without
        // evaluating them.
        if (!left_ok && !right_ok) || n_leaves >= params.num_leaves {
            leaf_rows.extend([left_idx, right_idx]);
            continue;
        }
        // LightGBM evaluates both children once either has the data to be
        // split (its counts in the split search are estimated).
        if !params.sklearn_compat {
            (left_ok, right_ok) = (true, true);
        }
        // LightGBM draws the node features of the smaller child first.
        let (left_features, right_features) = if left_idx.len() < right_idx.len() {
            let l = draw_node_features();
            (l, draw_node_features())
        } else {
            let r = draw_node_features();
            (draw_node_features(), r)
        };

        // Histogram subtraction: build the smaller child's histograms and
        // derive the larger child's from the parent's.
        let left_smaller = left_idx.len() <= right_idx.len();
        let (small_idx, small_totals) = if left_smaller {
            (&left_idx, left_totals)
        } else {
            (&right_idx, right_totals)
        };
        let small = build_histograms(
            binned_x,
            all_bins,
            bundles,
            params.sklearn_compat,
            gradients,
            hessians,
            small_idx,
            small_totals,
            feature_indices,
            &params.fixed_bins,
        );
        let mut large = candidate.hists;
        for (parent, s) in large.iter_mut().zip(&small) {
            parent.subtract(s);
        }
        let (left_hists, right_hists) = if left_smaller {
            (small, large)
        } else {
            (large, small)
        };

        // Try to split the new children
        for (ok, child_id, child_indices, hists, bounds, value, totals, node_features) in [
            (
                left_ok,
                left_id,
                left_idx,
                left_hists,
                left_bounds,
                split.left_value,
                left_totals,
                left_features,
            ),
            (
                right_ok,
                right_id,
                right_idx,
                right_hists,
                right_bounds,
                split.right_value,
                right_totals,
                right_features,
            ),
        ] {
            if !ok {
                leaf_rows.push(child_indices);
                continue;
            }
            if let Some(s) = best_for(
                &hists,
                totals,
                child_indices.len(),
                bounds,
                value,
                &groups,
                node_features.as_deref(),
            ) {
                heap.push(LeafCandidate {
                    node_id: child_id,
                    gain: s.gain,
                    split: s,
                    indices: child_indices,
                    hists,
                    depth,
                    bounds,
                    groups: groups.clone(),
                    totals,
                });
            } else {
                leaf_rows.push(child_indices);
            }
        }
    }
    leaf_rows.extend(heap.into_iter().map(|c| c.indices));

    // Convert flat BuildNode tree to TreeNode tree
    (flatten_tree(&nodes, 0), leaf_rows)
}

fn flatten_tree(nodes: &[BuildNode], id: usize) -> TreeNode {
    match &nodes[id] {
        BuildNode::Leaf { value } => TreeNode::Leaf { value: *value },
        BuildNode::Internal {
            feature,
            bin_threshold,
            missing_left,
            categorical_order,
            left,
            right,
        } => TreeNode::Internal {
            feature: *feature,
            bin_threshold: *bin_threshold,
            missing_left: *missing_left,
            categorical_order: categorical_order.clone(),
            categorical_left: Default::default(),
            left: Box::new(flatten_tree(nodes, *left)),
            right: Box::new(flatten_tree(nodes, *right)),
        },
    }
}

// ============================================================
// GOSS sampling
// ============================================================

// ============================================================
// Parameters
// ============================================================

/// Boosting type.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum BoostingType {
    /// Standard gradient boosting.
    Gbdt,
    /// Gradient-based One-Side Sampling.
    Goss,
    /// Dropouts meet Multiple Additive Regression Trees (LightGBM's
    /// `boosting_type="dart"`, parameters in `dart`).
    Dart,
}

/// LightGBM's DART parameters (names and defaults of LightGBM).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DartParams {
    /// Share of the earlier trees dropped per iteration (default 0.1).
    pub drop_rate: f64,
    /// At most this many trees dropped per iteration; `<= 0` = no limit
    /// (default 50).
    pub max_drop: i64,
    /// Probability of dropping nothing in an iteration (default 0.5).
    pub skip_drop: f64,
    /// Drop every tree with the same probability instead of in proportion
    /// to its weight (default false).
    pub uniform_drop: bool,
    /// XGBoost's normalisation (default false).
    pub xgboost_dart_mode: bool,
    /// Seed of the dropping (default 4).
    pub drop_seed: u32,
}

impl Default for DartParams {
    fn default() -> Self {
        Self {
            drop_rate: 0.1,
            max_drop: 50,
            skip_drop: 0.5,
            uniform_drop: false,
            xgboost_dart_mode: false,
            drop_seed: 4,
        }
    }
}

/// LightGBM's `Random` (utils/random.h): the LCG behind its bagging, GOSS,
/// feature fractions and DART's dropping.
#[derive(Debug, Clone)]
struct LgbmRandom(u32);

impl LgbmRandom {
    fn step(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(214_013).wrapping_add(2_531_011);
        self.0
    }

    fn rand_int16(&mut self) -> usize {
        ((self.step() >> 16) & 0x7fff) as usize
    }

    /// `NextShort(0, upper)`.
    fn next_short(&mut self, upper: usize) -> usize {
        self.rand_int16() % upper
    }

    /// `NextInt(0, upper)`.
    fn next_int(&mut self, upper: usize) -> usize {
        (self.step() & 0x7fff_ffff) as usize % upper
    }

    /// `NextFloat`: a float in [0, 1).
    fn next_float(&mut self) -> f32 {
        self.rand_int16() as f32 / 32768.0
    }

    /// `Sample(n, k)`: `k` of `0..n`, ascending.
    fn sample(&mut self, n: usize, k: usize) -> Vec<usize> {
        if k > n || k == 0 {
            return Vec::new();
        }
        if k == n {
            return (0..n).collect();
        }
        if k > 1 && k as f64 > n as f64 / (k as f64).log2() {
            let mut picked = Vec::with_capacity(k);
            for i in 0..n {
                let prob = (k - picked.len()) as f64 / (n - i) as f64;
                if f64::from(self.next_float()) < prob {
                    picked.push(i);
                }
            }
            picked
        } else {
            let mut set = std::collections::BTreeSet::new();
            for r in n - k..n {
                let v = self.next_int(r + 1);
                if !set.insert(v) {
                    set.insert(r);
                }
            }
            set.into_iter().collect()
        }
    }
}

/// The seeds LightGBM derives from `seed` (its `Config::Set`); without a
/// seed its defaults (bagging 3, feature fraction 2, DART's `drop_seed`).
#[derive(Debug, Clone, Copy)]
struct LgbmSeeds {
    bagging: u32,
    feature_fraction: u32,
    drop: u32,
}

impl LgbmSeeds {
    fn new(seed: Option<u64>, drop_seed: u32) -> Self {
        let Some(seed) = seed else {
            return Self {
                bagging: 3,
                feature_fraction: 2,
                drop: drop_seed,
            };
        };
        let mut rng = LgbmRandom(seed as u32);
        let mut next = || rng.next_short(i16::MAX as usize) as u32;
        let _data_random_seed = next();
        let bagging = next();
        let drop = next();
        let feature_fraction = next();
        Self {
            bagging,
            feature_fraction,
            drop,
        }
    }
}

/// Rows per generator of LightGBM's bagging and GOSS (`bagging_rand_block_`).
const LGBM_BAGGING_BLOCK: usize = 1024;

/// One LightGBM generator per block of rows, seeded `seed + block`.
fn block_generators(seed: u32, n: usize) -> Vec<LgbmRandom> {
    (0..n.div_ceil(LGBM_BAGGING_BLOCK))
        .map(|b| LgbmRandom(seed.wrapping_add(b as u32)))
        .collect()
}

/// LightGBM's bagging (`BaggingSampleStrategy`): every `freq` iterations
/// each row is kept with probability `fraction`; with balanced bagging
/// (binary labels) the positive and negative rows with their own fractions.
struct LgbmBagging {
    rands: Vec<LgbmRandom>,
    fraction: f64,
    freq: usize,
    /// `(is positive, pos_fraction, neg_fraction)` for balanced bagging.
    balanced: Option<(Vec<bool>, f64, f64)>,
    bag: Option<Vec<usize>>,
}

impl LgbmBagging {
    /// `None` when LightGBM does not bag.
    fn new(
        seed: u32,
        n: usize,
        fraction: f64,
        freq: usize,
        balanced: Option<(Vec<bool>, f64, f64)>,
    ) -> Option<Self> {
        let balanced =
            balanced.filter(|(pos, p, q)| (*p < 1.0 || *q < 1.0) && pos.iter().any(|&b| b));
        (freq > 0 && (fraction < 1.0 || balanced.is_some())).then(|| Self {
            rands: block_generators(seed, n),
            fraction,
            freq,
            balanced,
            bag: None,
        })
    }

    /// The bag of iteration `iter` (re-drawn on the first iteration and
    /// every `freq` iterations while it is not the whole data).
    fn rows(&mut self, iter: usize, n: usize) -> Vec<usize> {
        let redraw = match &self.bag {
            None => true,
            Some(bag) => bag.len() < n && iter % self.freq == 0,
        };
        if redraw {
            let mut bag = Vec::new();
            for i in 0..n {
                let fraction = match &self.balanced {
                    Some((pos, p, q)) => {
                        if pos[i] {
                            *p
                        } else {
                            *q
                        }
                    }
                    None => self.fraction,
                };
                if f64::from(self.rands[i / LGBM_BAGGING_BLOCK].next_float()) < fraction {
                    bag.push(i);
                }
            }
            self.bag = Some(bag);
        }
        self.bag.clone().unwrap_or_default()
    }
}

/// LightGBM's GOSS (`GOSSStrategy`, one thread): after the first
/// `1 / learning_rate` iterations, keeps the rows of the `top_rate` largest
/// `Σ |g·h|` (over the trees of the iteration) and samples `other_rate · n`
/// of the others, whose gradients and hessians it amplifies by
/// `(n - top_k) / other_k` (float32). Returns the rows, ascending.
fn lgbm_goss(
    rands: &mut [LgbmRandom],
    gradients: &mut [&mut [f64]],
    hessians: &mut [&mut [f64]],
    top_rate: f64,
    other_rate: f64,
) -> Vec<usize> {
    let n = gradients[0].len();
    let weight = |g: &[&mut [f64]], h: &[&mut [f64]], i: usize| {
        g.iter().zip(h).fold(0.0f32, |acc, (g, h)| {
            acc + (g[i] as f32 * h[i] as f32).abs()
        })
    };
    let mut weights: Vec<f32> = (0..n).map(|i| weight(gradients, hessians, i)).collect();
    let top_k = ((n as f64 * top_rate) as usize).max(1);
    let other_k = (n as f64 * other_rate) as usize;
    // LightGBM reads the threshold at `top_k - 1` after its partial sort
    // (not always the `top_k`-th largest weight).
    lgbm_arg_max_at_k(&mut weights, top_k - 1);
    let threshold = weights[top_k - 1];
    let multiply = (n - top_k) as f32 / other_k as f32;
    let mut rows = Vec::new();
    let mut big = 0usize;
    for i in 0..n {
        if weight(gradients, hessians, i) >= threshold {
            rows.push(i);
            big += 1;
            continue;
        }
        let rest_need = other_k as f64 - (rows.len() - big) as f64;
        let rest_all = (n - i) as f64 - (top_k as f64 - big as f64);
        if f64::from(rands[i / LGBM_BAGGING_BLOCK].next_float()) < rest_need / rest_all {
            rows.push(i);
            for (g, h) in gradients.iter_mut().zip(hessians.iter_mut()) {
                g[i] = f64::from(g[i] as f32 * multiply);
                h[i] = f64::from(h[i] as f32 * multiply);
            }
        }
    }
    rows
}

/// LightGBM's `ArrayArgs::Partition`: a three-way partition of
/// `a[start..end]` around its last value, descending; returns the ends of
/// the larger and the smaller part.
fn lgbm_partition(a: &mut [f32], start: isize, end: isize) -> (isize, isize) {
    if start >= end - 1 {
        return (start - 1, end);
    }
    let at = |k: isize| k as usize;
    let (mut i, mut j) = (start - 1, end - 1);
    let (mut p, mut q) = (i, j);
    let v = a[at(end - 1)];
    loop {
        loop {
            i += 1;
            if a[at(i)] <= v {
                break;
            }
        }
        loop {
            j -= 1;
            if v <= a[at(j)] || j == start {
                break;
            }
        }
        if i >= j {
            break;
        }
        a.swap(at(i), at(j));
        if a[at(i)] == v {
            p += 1;
            a.swap(at(p), at(i));
        }
        if v == a[at(j)] {
            q -= 1;
            a.swap(at(j), at(q));
        }
    }
    a.swap(at(i), at(end - 1));
    j = i - 1;
    i += 1;
    for k in start..=p {
        a.swap(at(k), at(j));
        j -= 1;
    }
    let mut k = end - 2;
    while k >= q {
        a.swap(at(i), at(k));
        k -= 1;
        i += 1;
    }
    (j, i)
}

/// LightGBM's `ArrayArgs::ArgMaxAtK`: partially sorts `a` (descending)
/// towards index `k`.
fn lgbm_arg_max_at_k(a: &mut [f32], k: usize) {
    let k = k as isize;
    let (mut start, mut end) = (0isize, a.len() as isize);
    while start < end - 1 {
        let (l, r) = lgbm_partition(a, start, end);
        if (k > l && k < r) || (l == start - 1 && r == end - 1) {
            return;
        } else if k <= l {
            end = l + 1;
        } else {
            start = r;
        }
    }
}

/// LightGBM's `ColSampler`: the features of a tree (`feature_fraction`, a
/// subset of the non-trivial features) and of a node
/// (`feature_fraction_bynode`, a subset of the tree's), from one generator.
#[derive(Debug, Clone)]
struct LgbmColSampler {
    rng: LgbmRandom,
    bytree: f64,
    bynode: f64,
    /// The features LightGBM can split on (more than one bin).
    valid: Vec<usize>,
    /// The tree's features (`valid` without `feature_fraction`).
    used: Vec<usize>,
}

impl LgbmColSampler {
    /// Like LightGBM, the constructor draws a first (unused) tree subset.
    fn new(seed: u32, bytree: f64, bynode: f64, valid: Vec<usize>) -> Self {
        let mut s = Self {
            rng: LgbmRandom(seed),
            bytree,
            bynode,
            used: valid.clone(),
            valid,
        };
        s.reset_by_tree();
        s
    }

    /// `GetCnt`: `round(total · fraction)`, at least one.
    fn count(total: usize, fraction: f64) -> usize {
        ((total as f64 * fraction + 0.5) as usize).max(total.min(1))
    }

    /// `ResetByTree`: draw the next tree's features.
    fn reset_by_tree(&mut self) {
        if self.bytree < 1.0 {
            let k = Self::count(self.valid.len(), self.bytree);
            let valid = &self.valid;
            self.used = self
                .rng
                .sample(valid.len(), k)
                .into_iter()
                .map(|i| valid[i])
                .collect();
        }
    }

    /// The features of a tree: every feature without `feature_fraction`.
    fn tree_features(&self, p: usize) -> Vec<usize> {
        if self.bytree < 1.0 {
            self.used.clone()
        } else {
            (0..p).collect()
        }
    }

    /// `GetByNode`: the features a node may split on, if subsampled.
    fn node_features(&mut self) -> Option<Vec<usize>> {
        (self.bynode < 1.0).then(|| {
            let k = Self::count(self.used.len(), self.bynode);
            let used = &self.used;
            self.rng
                .sample(used.len(), k)
                .into_iter()
                .map(|i| used[i])
                .collect()
        })
    }
}

/// The sampling parameters of a fit.
#[derive(Debug, Clone, Copy)]
struct SamplingParams {
    boosting_type: BoostingType,
    learning_rate: f64,
    subsample: f64,
    subsample_freq: usize,
    colsample_bytree: f64,
    colsample_bynode: f64,
    goss_top_rate: f64,
    goss_other_rate: f64,
    seeds: LgbmSeeds,
}

impl SamplingParams {
    fn validate(&self) -> Result<()> {
        let invalid = |m: &str| Err(RustMlError::InvalidParameter(m.into()));
        let fraction = |f: f64| f > 0.0 && f <= 1.0;
        if !fraction(self.subsample) {
            return invalid("subsample must be in (0, 1]");
        }
        if !fraction(self.colsample_bytree) || !fraction(self.colsample_bynode) {
            return invalid("colsample_bytree and colsample_bynode must be in (0, 1]");
        }
        if self.boosting_type == BoostingType::Goss {
            if !(self.goss_top_rate > 0.0 && self.goss_other_rate > 0.0)
                || self.goss_top_rate + self.goss_other_rate > 1.0
            {
                return invalid("GOSS needs top_rate, other_rate > 0 with a sum of at most 1");
            }
            if self.subsample_freq > 0 && self.subsample < 1.0 {
                return invalid("GOSS cannot be combined with bagging");
            }
        }
        Ok(())
    }
}

/// LightGBM's row and feature samplers of a fit.
struct LgbmSamplers {
    bagging: Option<LgbmBagging>,
    goss: Option<Vec<LgbmRandom>>,
    goss_warmup: usize,
    cols: LgbmColSampler,
    p: SamplingParams,
}

impl LgbmSamplers {
    /// `positives`: the binary labels, for balanced bagging with
    /// `(pos, neg)` fractions.
    fn new(
        p: SamplingParams,
        binned_x: &Array2<u8>,
        balanced: Option<(Vec<bool>, f64, f64)>,
    ) -> Self {
        let n = binned_x.nrows();
        let bagging = (p.boosting_type != BoostingType::Goss)
            .then(|| LgbmBagging::new(p.seeds.bagging, n, p.subsample, p.subsample_freq, balanced))
            .flatten();
        let goss =
            (p.boosting_type == BoostingType::Goss).then(|| block_generators(p.seeds.bagging, n));
        let valid = if p.colsample_bytree < 1.0 || p.colsample_bynode < 1.0 {
            valid_features(binned_x)
        } else {
            Vec::new()
        };
        Self {
            bagging,
            goss,
            // LightGBM's `static_cast<int>(1.0f / learning_rate)`.
            goss_warmup: (1.0 / p.learning_rate) as usize,
            cols: LgbmColSampler::new(
                p.seeds.feature_fraction,
                p.colsample_bytree,
                p.colsample_bynode,
                valid,
            ),
            p,
        }
    }

    /// The rows of iteration `iter`; GOSS amplifies the gradients and
    /// hessians (of all trees of the iteration) of its sampled rows.
    fn rows(
        &mut self,
        iter: usize,
        n: usize,
        gradients: &mut [&mut [f64]],
        hessians: &mut [&mut [f64]],
    ) -> Vec<usize> {
        if let Some(b) = self.bagging.as_mut() {
            return b.rows(iter, n);
        }
        match self.goss.as_mut() {
            Some(rands) if iter >= self.goss_warmup => lgbm_goss(
                rands,
                gradients,
                hessians,
                self.p.goss_top_rate,
                self.p.goss_other_rate,
            ),
            _ => (0..n).collect(),
        }
    }

    /// The features of the next tree.
    fn tree_features(&mut self, n_features: usize) -> Vec<usize> {
        self.cols.reset_by_tree();
        self.cols.tree_features(n_features)
    }

    /// The node sampler for the tree builder, if nodes are subsampled.
    fn node_sampler(&mut self) -> Option<&mut LgbmColSampler> {
        (self.p.colsample_bynode < 1.0).then_some(&mut self.cols)
    }

    /// Replay the draws of `rounds` fitted rounds of `trees_per_round`
    /// trees (a warm start; GOSS and node sampling, whose draws depend on
    /// the data, cannot be continued).
    fn replay(&mut self, rounds: usize, trees_per_round: usize, n: usize) {
        for iter in 0..rounds {
            if let Some(b) = self.bagging.as_mut() {
                b.rows(iter, n);
            }
            for _ in 0..trees_per_round {
                self.cols.reset_by_tree();
            }
        }
    }
}

/// The features of `binned_x` with more than one bin (LightGBM's
/// `ValidFeatureIndices`; it drops the trivial ones).
fn valid_features(binned_x: &Array2<u8>) -> Vec<usize> {
    (0..binned_x.ncols())
        .filter(|&j| {
            let col = binned_x.column(j);
            col.iter().any(|&b| b != col[0])
        })
        .collect()
}

/// DART's book-keeping (LightGBM's `dart.hpp`). Tree leaves carry their
/// current weight, scaled in LightGBM's order of operations.
struct DartState {
    p: DartParams,
    learning_rate: f64,
    rng: LgbmRandom,
    tree_weight: Vec<f64>,
    sum_weight: f64,
}

impl DartState {
    fn new(p: DartParams, learning_rate: f64, drop_seed: u32) -> Self {
        Self {
            p,
            learning_rate,
            rng: LgbmRandom(drop_seed),
            tree_weight: Vec::new(),
            sum_weight: 0.0,
        }
    }

    /// `DroppingTrees` over the `n_iters` earlier iterations: the dropped
    /// iterations and the shrinkage of the next tree.
    fn drop_trees(&mut self, n_iters: usize) -> (Vec<usize>, f64) {
        let mut drops = Vec::new();
        let max_drop = usize::try_from(self.p.max_drop).ok().filter(|&m| m > 0);
        let is_skip = f64::from(self.rng.next_float()) < self.p.skip_drop;
        if !is_skip {
            let mut drop_rate = self.p.drop_rate;
            if !self.p.uniform_drop {
                let inv_average_weight = self.tree_weight.len() as f64 / self.sum_weight;
                if let Some(m) = max_drop {
                    drop_rate = drop_rate.min(m as f64 * inv_average_weight / self.sum_weight);
                }
                for i in 0..n_iters {
                    let r = f64::from(self.rng.next_float());
                    if r < drop_rate * self.tree_weight[i] * inv_average_weight {
                        drops.push(i);
                        if max_drop.is_some_and(|m| drops.len() >= m) {
                            break;
                        }
                    }
                }
            } else {
                if let Some(m) = max_drop {
                    drop_rate = drop_rate.min(m as f64 / n_iters as f64);
                }
                for i in 0..n_iters {
                    if f64::from(self.rng.next_float()) < drop_rate {
                        drops.push(i);
                        if max_drop.is_some_and(|m| drops.len() >= m) {
                            break;
                        }
                    }
                }
            }
        }
        let k = drops.len() as f64;
        let shrinkage = if !self.p.xgboost_dart_mode {
            self.learning_rate / (1.0 + k)
        } else if drops.is_empty() {
            self.learning_rate
        } else {
            self.learning_rate / (self.learning_rate + k)
        };
        (drops, shrinkage)
    }

    /// `Normalize`: the factors the dropped trees are scaled by (for the
    /// validation scores, then for the training scores), updating the
    /// tree weights.
    fn normalize(&mut self, drops: &[usize], shrinkage: f64) -> (f64, f64) {
        let k = drops.len() as f64;
        let (lr, xgb) = (self.learning_rate, self.p.xgboost_dart_mode);
        if !self.p.uniform_drop {
            for &i in drops {
                if xgb {
                    self.sum_weight -= self.tree_weight[i] * (1.0 / (k + lr));
                    self.tree_weight[i] *= k / (k + lr);
                } else {
                    self.sum_weight -= self.tree_weight[i] * (1.0 / (k + 1.0));
                    self.tree_weight[i] *= k / (k + 1.0);
                }
            }
        }
        if xgb {
            (shrinkage, -k / lr)
        } else {
            (1.0 / (k + 1.0), -k)
        }
    }

    /// After a new tree: record its weight (non-uniform dropping).
    fn push(&mut self, shrinkage: f64) {
        if !self.p.uniform_drop {
            self.tree_weight.push(shrinkage);
            self.sum_weight += shrinkage;
        }
    }
}

impl Default for BoostingType {
    fn default() -> Self {
        BoostingType::Gbdt
    }
}

/// Objective / loss function.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum LgbmObjective {
    /// Regression with squared error loss.
    Regression,
    /// Regression with mean absolute error (L1) loss.
    RegressionL1,
    /// Regression with Huber loss.
    Huber,
    /// Quantile regression: pinball loss at level `alpha` in (0, 1)
    /// (LightGBM's `objective="quantile"`). Leaf outputs are refit to the
    /// `alpha`-percentile of their residuals.
    Quantile { alpha: f64 },
    /// Poisson regression with a log link (LightGBM's `objective="poisson"`).
    /// Targets must be non-negative with a positive sum; predictions are
    /// `exp` of the raw score.
    Poisson,
    /// Gamma regression with a log link (LightGBM's `objective="gamma"`).
    /// Targets must be positive; predictions are `exp` of the raw score.
    Gamma,
    /// Binary classification with log-loss.
    Binary,
    /// Multi-class classification with softmax.
    Multiclass,
}

#[derive(Clone)]
struct LgbmParams {
    num_leaves: usize,
    max_depth: Option<usize>,
    min_child_samples: usize,
    min_child_weight: f64,
    min_split_gain: f64,
    reg_alpha: f64,
    reg_lambda: f64,
    /// Per-feature monotone constraint: -1 = decreasing, 0 = none, +1 = increasing.
    /// Length must match number of features (empty = all zero).
    monotone_constraints: Vec<i8>,
    categorical: CategoricalSplitParams,
    /// Evaluate and scan split candidates as sklearn does (see
    /// `find_best_split`).
    sklearn_compat: bool,
    /// Per feature: whether the training data has missing values (used in
    /// sklearn mode; empty otherwise).
    has_missing: Vec<bool>,
    /// Interaction constraints: groups of features (empty = none). A branch
    /// may only split on features of the groups that contain every feature
    /// split on above it (sklearn's / LightGBM's `interaction_cst`).
    interaction_cst: Vec<Vec<usize>>,
    /// sklearn's `max_features`: share of the allowed features tried at
    /// every split (1 = all).
    feature_fraction_per_split: f64,
    /// Per feature, the bin LightGBM derives from the node totals (see
    /// `lgbm_fixed_bins`; empty in sklearn mode).
    fixed_bins: Vec<Option<u8>>,
}

/// How categorical features are split (LightGBM's parameters of the same
/// names and defaults).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CategoricalSplitParams {
    /// Up to this many bins (the "other" bin included) every category is
    /// tried alone against the rest (one-vs-rest). Default 4.
    pub max_cat_to_onehot: usize,
    /// Smoothing added to the hessian when ordering categories by
    /// `G / (H + cat_smooth)`; categories with fewer samples are not
    /// candidates for the left set. Default 10.
    pub cat_smooth: f64,
    /// Extra L2 penalty on the children of a many-vs-many split. Default 10.
    pub cat_l2: f64,
    /// Maximum number of categories in the left set. Default 32.
    pub max_cat_threshold: usize,
    /// Minimum samples added per group of categories and on the right side
    /// of a many-vs-many split. Default 100.
    pub min_data_per_group: usize,
}

impl Default for CategoricalSplitParams {
    fn default() -> Self {
        Self {
            max_cat_to_onehot: 4,
            cat_smooth: 10.0,
            cat_l2: 10.0,
            max_cat_threshold: 32,
            min_data_per_group: 100,
        }
    }
}

// ============================================================
// LgbmRegressor
// ============================================================

/// LightGBM-style gradient boosting regressor.
///
/// Uses leaf-wise tree growth with `num_leaves`, histogram-based splits,
/// learned missing directions, L1+L2 regularization, and optional GOSS sampling.
///
/// Most parameters mirror LightGBM's Python API: `num_leaves`, `max_depth`,
/// `learning_rate`, `n_estimators`, `min_child_samples`, `min_child_weight`,
/// `reg_alpha`, `reg_lambda`, `subsample`, `colsample_bytree`, `min_split_gain`,
/// `boosting_type`, `objective`, `monotone_constraints`, `categorical_features`.
///
/// Supports sample weights, init_score, and early stopping via the
/// [`FitWithEval`](LgbmRegressor::fit_with_eval) method. The standard
/// [`Fit::fit`] call does plain batch training without early stopping.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LgbmRegressor {
    pub n_estimators: usize,
    pub num_leaves: usize,
    pub max_depth: Option<usize>,
    pub learning_rate: f64,
    pub min_child_samples: usize,
    pub min_child_weight: f64,
    pub min_split_gain: f64,
    pub reg_alpha: f64,
    pub reg_lambda: f64,
    /// Share of the rows of a bag (LightGBM's `bagging_fraction`); only
    /// used with `subsample_freq > 0`.
    pub subsample: f64,
    /// Re-bag every this many iterations; 0 (default) = no bagging
    /// (LightGBM's `bagging_freq`).
    #[serde(default)]
    pub subsample_freq: usize,
    /// Share of the features of a tree (LightGBM's `feature_fraction`).
    pub colsample_bytree: f64,
    /// Share of the tree's features a node may split on (LightGBM's
    /// `feature_fraction_bynode`).
    #[serde(default = "default_one")]
    pub colsample_bynode: f64,
    pub max_bins: usize,
    /// Minimum number of training samples per histogram bin.  Bins with fewer
    /// samples are merged with their neighbours, reducing the effective split
    /// resolution.  Matches LightGBM's `min_data_in_bin` (default: 3).
    pub min_data_in_bin: usize,
    /// Numeric binning: LightGBM's (default) or sklearn's.
    #[serde(default)]
    pub binning: BinningMethod,
    pub boosting_type: BoostingType,
    /// DART parameters (with `boosting_type = Dart`).
    #[serde(default)]
    pub dart: DartParams,
    pub goss_top_rate: f64,
    pub goss_other_rate: f64,
    pub objective: LgbmObjective,
    /// Huber transition point (LightGBM's `alpha`, default 0.9).
    pub huber_delta: f64,
    /// Added to the raw score in the Poisson hessian, `exp(score +
    /// max_delta_step)`, to safeguard the optimisation (LightGBM's
    /// `poisson_max_delta_step`, default 0.7; 0 gives the exact hessian).
    #[serde(default = "default_poisson_max_delta_step")]
    pub poisson_max_delta_step: f64,
    pub early_stopping_rounds: Option<usize>,
    pub categorical_features: Vec<usize>,
    /// How categorical features are split.
    #[serde(default)]
    pub categorical_split: CategoricalSplitParams,
    /// Per-feature monotone constraint: -1 (decreasing), 0 (none), +1 (increasing).
    /// Length must match `n_features` (or be empty for no constraints).
    pub monotone_constraints: Vec<i8>,
    /// Exclusive feature bundling (LightGBM's `enable_bundle`, default
    /// true): see `FeatureBundles`. It changes the speed, not the model
    /// (up to rounding in the histograms).
    #[serde(default = "default_true")]
    pub enable_bundle: bool,
    /// Threads of a fit (LightGBM's `n_jobs`): `None` = the physical
    /// cores (at most the current rayon pool's threads), which on machines
    /// with SMT or efficiency cores is faster than all logical ones. The
    /// model does not depend on it.
    #[serde(default)]
    pub n_jobs: Option<usize>,
    /// LightGBM's `seed` (`random_state`): derives the seeds of bagging,
    /// GOSS, the feature fractions and DART's dropping as LightGBM does.
    /// `None` (default): LightGBM's default seeds (and `dart.drop_seed`).
    #[serde(default)]
    pub seed: Option<u64>,
    /// sklearn's numerics instead of LightGBM's: float32 gradients and
    /// hessians, its L1 / quantile gradient at ties (prediction == target
    /// counts as below), split search (see `find_best_split`), weighted
    /// percentiles and given unit weights. The HistGradientBoosting facade
    /// turns it on.
    #[serde(skip)]
    pub(crate) sklearn_compat: bool,
    /// Interaction constraints (see `LgbmParams::interaction_cst`); set by
    /// the HistGradientBoosting facade.
    #[serde(skip)]
    pub(crate) interaction_cst: Vec<Vec<usize>>,
    /// sklearn's `max_features` (share of features per split) and the seed
    /// of its `default_rng`; set by the HistGradientBoosting facade.
    #[serde(skip, default = "default_one")]
    pub(crate) feature_fraction_per_split: f64,
    #[serde(skip)]
    pub(crate) feature_subsample_seed: u32,
}

impl LgbmRegressor {
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            num_leaves: 31,
            max_depth: None,
            learning_rate: 0.1,
            min_child_samples: 20,
            min_child_weight: 1e-3,
            min_split_gain: 0.0,
            reg_alpha: 0.0,
            reg_lambda: 0.0,
            subsample: 1.0,
            subsample_freq: 0,
            colsample_bytree: 1.0,
            colsample_bynode: 1.0,
            max_bins: DEFAULT_MAX_BINS,
            min_data_in_bin: 3,
            binning: BinningMethod::LightGbm,
            boosting_type: BoostingType::Gbdt,
            dart: DartParams::default(),
            goss_top_rate: 0.2,
            goss_other_rate: 0.1,
            objective: LgbmObjective::Regression,
            huber_delta: 0.9,
            poisson_max_delta_step: default_poisson_max_delta_step(),
            early_stopping_rounds: None,
            categorical_features: Vec::new(),
            categorical_split: CategoricalSplitParams::default(),
            monotone_constraints: Vec::new(),
            enable_bundle: true,
            n_jobs: None,
            seed: None,
            sklearn_compat: false,
            interaction_cst: Vec::new(),
            feature_fraction_per_split: 1.0,
            feature_subsample_seed: 0,
        }
    }

    // Builder methods
    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }
    pub fn with_num_leaves(mut self, n: usize) -> Self {
        self.num_leaves = n;
        self
    }
    pub fn with_max_depth(mut self, d: Option<usize>) -> Self {
        self.max_depth = d;
        self
    }
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }
    pub fn with_min_child_samples(mut self, n: usize) -> Self {
        self.min_child_samples = n;
        self
    }
    pub fn with_min_child_weight(mut self, w: f64) -> Self {
        self.min_child_weight = w;
        self
    }
    pub fn with_min_split_gain(mut self, g: f64) -> Self {
        self.min_split_gain = g;
        self
    }
    pub fn with_reg_alpha(mut self, a: f64) -> Self {
        self.reg_alpha = a;
        self
    }
    pub fn with_reg_lambda(mut self, l: f64) -> Self {
        self.reg_lambda = l;
        self
    }
    pub fn with_subsample(mut self, s: f64) -> Self {
        self.subsample = s;
        self
    }
    pub fn with_subsample_freq(mut self, f: usize) -> Self {
        self.subsample_freq = f;
        self
    }
    pub fn with_colsample_bynode(mut self, c: f64) -> Self {
        self.colsample_bynode = c;
        self
    }
    pub fn with_colsample_bytree(mut self, c: f64) -> Self {
        self.colsample_bytree = c;
        self
    }
    /// Maximum number of bins for non-missing values (2..=255, default 255).
    pub fn with_max_bins(mut self, b: usize) -> Self {
        self.max_bins = b;
        self
    }
    pub fn with_min_data_in_bin(mut self, n: usize) -> Self {
        self.min_data_in_bin = n;
        self
    }
    /// Numeric binning method (LightGBM's by default).
    pub fn with_binning(mut self, method: BinningMethod) -> Self {
        self.binning = method;
        self
    }
    pub fn with_boosting_type(mut self, b: BoostingType) -> Self {
        self.boosting_type = b;
        self
    }
    /// DART (sets `boosting_type = Dart`).
    pub fn with_dart(mut self, p: DartParams) -> Self {
        self.boosting_type = BoostingType::Dart;
        self.dart = p;
        self
    }
    pub fn with_goss_rates(mut self, top: f64, other: f64) -> Self {
        self.goss_top_rate = top;
        self.goss_other_rate = other;
        self
    }
    pub fn with_objective(mut self, o: LgbmObjective) -> Self {
        self.objective = o;
        self
    }
    pub fn with_huber_delta(mut self, d: f64) -> Self {
        self.huber_delta = d;
        self
    }
    pub fn with_poisson_max_delta_step(mut self, d: f64) -> Self {
        self.poisson_max_delta_step = d;
        self
    }
    pub fn with_early_stopping(mut self, r: Option<usize>) -> Self {
        self.early_stopping_rounds = r;
        self
    }
    pub fn with_categorical_features(mut self, feats: Vec<usize>) -> Self {
        self.categorical_features = feats;
        self
    }
    /// Categorical split parameters (LightGBM's `max_cat_to_onehot`,
    /// `cat_smooth`, `cat_l2`, `max_cat_threshold`, `min_data_per_group`).
    pub fn with_categorical_split(mut self, p: CategoricalSplitParams) -> Self {
        self.categorical_split = p;
        self
    }
    pub fn with_monotone_constraints(mut self, constraints: Vec<i8>) -> Self {
        self.monotone_constraints = constraints;
        self
    }
    /// Exclusive feature bundling on or off (LightGBM's `enable_bundle`).
    pub fn with_enable_bundle(mut self, on: bool) -> Self {
        self.enable_bundle = on;
        self
    }
    /// Threads of a fit (see the `n_jobs` field).
    pub fn with_n_jobs(mut self, n_jobs: Option<usize>) -> Self {
        self.n_jobs = n_jobs;
        self
    }
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = Some(s);
        self
    }

    fn warm_start_model(&self) -> WarmStartModel {
        WarmStartModel {
            n_estimators: self.n_estimators,
            learning_rate: self.learning_rate,
            boosting_type: self.boosting_type,
            early_stopping: self.early_stopping_rounds.is_some(),
            feature_fraction_per_split: self.feature_fraction_per_split,
            colsample_bynode: self.colsample_bynode,
        }
    }

    fn sampling(&self) -> SamplingParams {
        SamplingParams {
            boosting_type: self.boosting_type,
            learning_rate: self.learning_rate,
            subsample: self.subsample,
            subsample_freq: self.subsample_freq,
            colsample_bytree: self.colsample_bytree,
            colsample_bynode: self.colsample_bynode,
            goss_top_rate: self.goss_top_rate,
            goss_other_rate: self.goss_other_rate,
            seeds: LgbmSeeds::new(self.seed, self.dart.drop_seed),
        }
    }

    fn params(&self) -> LgbmParams {
        LgbmParams {
            num_leaves: self.num_leaves,
            max_depth: self.max_depth,
            min_child_samples: self.min_child_samples,
            min_child_weight: self.min_child_weight,
            // LightGBM compares its (unhalved) gain with min_split_gain; the
            // engine's gains are halved.
            min_split_gain: if self.sklearn_compat {
                self.min_split_gain
            } else {
                0.5 * self.min_split_gain
            },
            reg_alpha: self.reg_alpha,
            reg_lambda: self.reg_lambda,
            monotone_constraints: self.monotone_constraints.clone(),
            categorical: self.categorical_split,
            sklearn_compat: self.sklearn_compat,
            has_missing: Vec::new(),
            interaction_cst: self.interaction_cst.clone(),
            feature_fraction_per_split: self.feature_fraction_per_split,
            fixed_bins: Vec::new(),
        }
    }
}

impl Default for LgbmRegressor {
    fn default() -> Self {
        Self::new()
    }
}

/// Options for training with an evaluation set.
#[derive(Debug, Clone)]
pub struct LgbmFitOptions<'a> {
    /// Optional per-sample weights (length = n_samples).
    pub sample_weight: Option<ArrayView1<'a, f64>>,
    /// Optional initial scores / baseline predictions (length = n_samples).
    /// Used instead of the default baseline.
    pub init_score: Option<ArrayView1<'a, f64>>,
    /// Optional evaluation set (X_eval, y_eval) for early stopping.
    pub eval_set: Option<(&'a Array2<f64>, &'a Array1<f64>)>,
    /// Optional per-sample weights for the eval set.
    pub eval_sample_weight: Option<ArrayView1<'a, f64>>,
    /// Print training progress every N iterations (0 = silent).
    pub verbose: usize,
}

impl<'a> Default for LgbmFitOptions<'a> {
    fn default() -> Self {
        Self {
            sample_weight: None,
            init_score: None,
            eval_set: None,
            eval_sample_weight: None,
            verbose: 0,
        }
    }
}

/// Fitted LightGBM-style regressor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedLgbmRegressor {
    trees: Vec<TreeNode>,
    bins: Vec<FeatureBins>,
    baseline: f64,
    learning_rate: f64,
    n_features: usize,
    /// Per-feature cumulative split gain (feature importance).
    feature_gain: Vec<f64>,
    /// Per-feature split count.
    feature_split_count: Vec<u32>,
    best_iteration: usize,
    /// Predictions are `exp` of the raw score (Poisson, Gamma).
    #[serde(default)]
    log_link: bool,
}

impl FittedLgbmRegressor {
    pub fn n_estimators(&self) -> usize {
        self.trees.len()
    }
    pub fn best_iteration(&self) -> usize {
        self.best_iteration
    }

    /// Feature importance by total gain (normalized to sum to 1).
    pub fn feature_importances(&self) -> Array1<f64> {
        let sum: f64 = self.feature_gain.iter().sum();
        if sum > 0.0 {
            Array1::from_vec(self.feature_gain.iter().map(|g| g / sum).collect())
        } else {
            Array1::zeros(self.n_features)
        }
    }

    /// Feature importance by split count.
    pub fn feature_split_counts(&self) -> Array1<u32> {
        Array1::from_vec(self.feature_split_count.clone())
    }
}

/// Per column of `x`: whether it has a missing value (all `false` unless
/// `needed`).
fn columns_with_missing(x: ArrayView2<f64>, needed: bool) -> Vec<bool> {
    x.columns()
        .into_iter()
        .map(|c| needed && c.iter().any(|v| v.is_nan()))
        .collect()
}

/// Round to float32, the precision sklearn keeps gradients and hessians in
/// (their sums are float64 there too).
fn round_to_f32(values: &mut [f64]) {
    for v in values {
        *v = f64::from(*v as f32);
    }
}

fn default_one() -> f64 {
    1.0
}

/// The generator of sklearn's per-split feature subsampling, if any.
fn feature_generator(fraction: f64, seed: u32) -> Option<Generator> {
    (fraction < 1.0).then(|| Generator::new(seed))
}

fn default_poisson_max_delta_step() -> f64 {
    0.7
}

/// Whether the objective models `log E[y]` (predictions are `exp` of the
/// raw score).
fn has_log_link(objective: LgbmObjective) -> bool {
    matches!(objective, LgbmObjective::Poisson | LgbmObjective::Gamma)
}

fn compute_regression_gradients(
    y: &[f64],
    preds: &[f64],
    model: &LgbmRegressor,
) -> (Vec<f64>, Vec<f64>) {
    let LgbmRegressor {
        objective,
        huber_delta,
        poisson_max_delta_step,
        sklearn_compat,
        ..
    } = *model;
    let n = y.len();
    let mut grad = vec![0.0; n];
    let mut hess = vec![1.0; n];

    match objective {
        LgbmObjective::Regression => {
            for i in 0..n {
                grad[i] = preds[i] - y[i];
                hess[i] = 1.0;
            }
        }
        LgbmObjective::RegressionL1 => {
            // LightGBM's sign has sign(0) = 0; leaf outputs are refit to
            // residual medians afterwards (`renew_leaf_outputs`).
            for i in 0..n {
                let r = preds[i] - y[i];
                grad[i] = match (r == 0.0, sklearn_compat) {
                    (false, _) => r.signum(),
                    (true, false) => 0.0,
                    (true, true) => -1.0,
                };
                hess[i] = 1.0;
            }
        }
        LgbmObjective::Huber => {
            for i in 0..n {
                let r = preds[i] - y[i];
                if r.abs() <= huber_delta {
                    grad[i] = r;
                } else {
                    grad[i] = huber_delta * r.signum();
                }
                hess[i] = 1.0;
            }
        }
        LgbmObjective::Quantile { alpha } => {
            // LightGBM: 1 − alpha above the target, −alpha below; sklearn
            // counts a tie as below.
            for i in 0..n {
                let above = if sklearn_compat {
                    preds[i] > y[i]
                } else {
                    preds[i] >= y[i]
                };
                grad[i] = if above { 1.0 - alpha } else { -alpha };
                hess[i] = 1.0;
            }
        }
        LgbmObjective::Poisson => {
            for i in 0..n {
                grad[i] = preds[i].exp() - y[i];
                hess[i] = (preds[i] + poisson_max_delta_step).exp();
            }
        }
        LgbmObjective::Gamma => {
            for i in 0..n {
                let e = y[i] * (-preds[i]).exp();
                grad[i] = 1.0 - e;
                hess[i] = e;
            }
        }
        _ => unreachable!("regressor only supports regression objectives"),
    }

    (grad, hess)
}

fn validate_regression_objective(objective: LgbmObjective, y: ArrayView1<f64>) -> Result<()> {
    match objective {
        LgbmObjective::Poisson if y.iter().any(|&v| v < 0.0) || y.sum() <= 0.0 => {
            Err(RustMlError::InvalidParameter(
                "the Poisson objective needs non-negative targets with a positive sum".into(),
            ))
        }
        LgbmObjective::Gamma if y.iter().any(|&v| v <= 0.0) => Err(RustMlError::InvalidParameter(
            "the Gamma objective needs positive targets".into(),
        )),
        LgbmObjective::Binary | LgbmObjective::Multiclass => Err(RustMlError::InvalidParameter(
            format!("{objective:?} is a classification objective; use LgbmClassifier"),
        )),
        LgbmObjective::Quantile { alpha } if !(alpha > 0.0 && alpha < 1.0) => Err(
            RustMlError::InvalidParameter(format!("quantile alpha must be in (0, 1), got {alpha}")),
        ),
        _ => Ok(()),
    }
}

/// Level of the residual percentile that leaf outputs and the baseline are
/// refit to (L1: the median), or `None` for Newton leaves.
fn refit_percentile(objective: LgbmObjective) -> Option<f64> {
    match objective {
        LgbmObjective::RegressionL1 => Some(0.5),
        LgbmObjective::Quantile { alpha } => Some(alpha),
        _ => None,
    }
}

/// `alpha`-percentile of `values` with linear interpolation between order
/// statistics (LightGBM's `PercentileFun`, numpy's default method).
fn percentile(values: &mut [f64], alpha: f64) -> f64 {
    let n = values.len();
    if n <= 1 {
        return values.first().copied().unwrap_or(0.0);
    }
    // LightGBM's `PercentileFun`: interpolated from the descending order.
    values.sort_by(|a, b| b.total_cmp(a));
    let float_pos = (n - 1) as f64 * (1.0 - alpha);
    let pos = float_pos as usize + 1;
    if pos >= n {
        return values[n - 1];
    }
    let bias = float_pos - (pos - 1) as f64;
    let (v1, v2) = (values[pos - 1], values[pos]);
    v1 - (v1 - v2) * bias
}

/// Weighted `alpha`-percentile (LightGBM 4.7's `WeightedPercentileFun`):
/// the first value whose cumulative weight exceeds `alpha` times the total,
/// interpolated from the previous value when that step's weight is at
/// least 1.
fn weighted_percentile(values: &[f64], weights: &[f64], alpha: f64) -> f64 {
    let n = values.len();
    if n <= 1 {
        return values.first().copied().unwrap_or(0.0);
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let cdf: Vec<f64> = order
        .iter()
        .scan(0.0, |acc, &i| {
            *acc += weights[i];
            Some(*acc)
        })
        .collect();
    let threshold = cdf[n - 1] * alpha;
    let pos = cdf.partition_point(|&c| c <= threshold).min(n - 1);
    if pos == 0 || pos == n - 1 {
        return values[order[pos]];
    }
    let (v1, v2) = (values[order[pos - 1]], values[order[pos]]);
    let step = cdf[pos] - cdf[pos - 1];
    if step >= 1.0 {
        (threshold - cdf[pos - 1]) / step * (v2 - v1) + v1
    } else {
        v1
    }
}

/// sklearn's `_weighted_percentile` (numpy's "inverted_cdf"): the first
/// value whose cumulative weight reaches `alpha` times the total.
fn sklearn_weighted_percentile(values: &[f64], weights: &[f64], alpha: f64) -> f64 {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let cdf: Vec<f64> = order
        .iter()
        .scan(0.0, |acc, &i| {
            *acc += weights[i];
            Some(*acc)
        })
        .collect();
    let Some(&total) = cdf.last() else {
        return 0.0;
    };
    // sklearn passes the level in percent: `(100 alpha) / 100 * total`.
    let mut target = 100.0 * alpha / 100.0 * total;
    if target == 0.0 {
        target = f64::from_bits(1);
    }
    let pos = cdf.partition_point(|&c| c < target).min(values.len() - 1);
    values[order[pos]]
}

/// `alpha`-percentile of `values[rows]`, weighted when `weights` is given
/// (with sklearn's weighted formula when `sklearn` is set).
fn rows_percentile(
    values: &[f64],
    rows: &[usize],
    weights: Option<ArrayView1<f64>>,
    alpha: f64,
    sklearn: bool,
) -> f64 {
    let mut v: Vec<f64> = rows.iter().map(|&i| values[i]).collect();
    match weights {
        Some(w) => {
            let wv: Vec<f64> = rows.iter().map(|&i| w[i]).collect();
            if sklearn {
                sklearn_weighted_percentile(&v, &wv, alpha)
            } else {
                weighted_percentile(&v, &wv, alpha)
            }
        }
        None if sklearn => numpy_percentile(&mut v, alpha),
        None => percentile(&mut v, alpha),
    }
}

/// numpy's default (linear) percentile, with numpy's interpolation
/// `b - (b - a)(1 - t)` for `t >= 0.5` (it differs from [`percentile`] in
/// the last bits).
fn numpy_percentile(values: &mut [f64], alpha: f64) -> f64 {
    let n = values.len();
    if n <= 1 {
        return values.first().copied().unwrap_or(0.0);
    }
    values.sort_by(f64::total_cmp);
    let pos = (n - 1) as f64 * alpha;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let (a, b, t) = (values[lo], values[hi], pos - lo as f64);
    let diff = b - a;
    if t >= 0.5 {
        b - diff * (1.0 - t)
    } else {
        a + diff * t
    }
}

/// Replace every leaf output with the `alpha`-percentile residual of the
/// training rows `rows` that reach it (LightGBM's `RenewTreeOutput` for the
/// L1 and quantile objectives). `binned_x` is column-major; leaves no row
/// reaches keep their value.
fn renew_leaf_outputs(
    node: &mut TreeNode,
    binned_x: &Array2<u8>,
    rows: Vec<usize>,
    residuals: &[f64],
    weights: Option<ArrayView1<f64>>,
    alpha: f64,
    sklearn: bool,
) {
    match node {
        TreeNode::Leaf { value } => {
            if !rows.is_empty() {
                *value = rows_percentile(residuals, &rows, weights, alpha, sklearn);
            }
        }
        TreeNode::Internal {
            feature,
            bin_threshold,
            missing_left,
            categorical_order,
            left,
            right,
            ..
        } => {
            let mask = categorical_order
                .as_ref()
                .map(|order| categorical_left_mask(order, *bin_threshold));
            let col = binned_x.column(*feature);
            let (l, r): (Vec<usize>, Vec<usize>) = rows
                .into_iter()
                .partition(|&i| goes_left(col[i], *bin_threshold, *missing_left, mask.as_deref()));
            renew_leaf_outputs(left, binned_x, l, residuals, weights, alpha, sklearn);
            renew_leaf_outputs(right, binned_x, r, residuals, weights, alpha, sklearn);
        }
    }
}

/// LightGBM's Python package drops sample weights that are all 1, which
/// selects the unweighted percentile formulas; this keeps that behaviour.
fn effective_sample_weight(w: Option<ArrayView1<f64>>) -> Option<ArrayView1<f64>> {
    w.filter(|w| w.iter().any(|&v| v != 1.0))
}

/// Initial score of a regression objective: the weighted mean (squared
/// error, Huber), the weighted median / `alpha`-percentile (L1, quantile)
/// or the log of the weighted mean (Poisson, Gamma), as LightGBM's
/// `BoostFromScore`.
fn regression_baseline(
    y: &[f64],
    weights: Option<ArrayView1<f64>>,
    objective: LgbmObjective,
    sklearn: bool,
) -> f64 {
    if let Some(alpha) = refit_percentile(objective) {
        let rows: Vec<usize> = (0..y.len()).collect();
        return rows_percentile(y, &rows, weights, alpha, sklearn);
    }
    let mean = match weights {
        Some(w) => y.iter().zip(w).map(|(&v, &wi)| v * wi).sum::<f64>() / w.sum(),
        None => y.iter().sum::<f64>() / y.len() as f64,
    };
    if has_log_link(objective) {
        mean.ln()
    } else {
        mean
    }
}

/// sklearn's early stopping rule (`scoring="loss"`): stop once none of the
/// last `n_iter_no_change` scores beats the one before them by more than
/// `tol`; all trees are kept.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SklearnStopping {
    pub n_iter_no_change: usize,
    pub tol: f64,
    pub scoring: EarlyStoppingScoring,
    /// Training rows the scorer is evaluated on (sklearn's small trainset;
    /// `None` = all). The loss is always taken over all of them.
    pub small_train: Option<Vec<usize>>,
}

/// Weighted mean `Σ w v / Σ w` over `rows` (all `0..n` when `None`).
fn weighted_mean(
    n: usize,
    rows: Option<&[usize]>,
    w: &dyn Fn(usize) -> f64,
    v: &dyn Fn(usize) -> f64,
) -> f64 {
    let (mut total, mut wsum) = (0.0, 0.0);
    let mut add = |i: usize| {
        total += w(i) * v(i);
        wsum += w(i);
    };
    match rows {
        Some(r) => r.iter().copied().for_each(&mut add),
        None => (0..n).for_each(&mut add),
    }
    if wsum > 0.0 {
        total / wsum
    } else {
        total
    }
}

/// sklearn's regression scorers on predictions `pred` (non-loss scorings).
fn regression_score(
    scoring: EarlyStoppingScoring,
    y: &[f64],
    pred: &dyn Fn(usize) -> f64,
    w: &dyn Fn(usize) -> f64,
    rows: Option<&[usize]>,
) -> f64 {
    let n = y.len();
    match scoring {
        EarlyStoppingScoring::NegMeanSquaredError => {
            -weighted_mean(n, rows, w, &|i| (y[i] - pred(i)).powi(2))
        }
        EarlyStoppingScoring::NegMeanAbsoluteError => {
            -weighted_mean(n, rows, w, &|i| (y[i] - pred(i)).abs())
        }
        _ => {
            // r2_score (also the regressor's default score).
            let avg = weighted_mean(n, rows, w, &|i| y[i]);
            let (mut num, mut den) = (0.0, 0.0);
            let mut add = |i: usize| {
                num += w(i) * (y[i] - pred(i)).powi(2);
                den += w(i) * (y[i] - avg).powi(2);
            };
            match rows {
                Some(r) => r.iter().copied().for_each(&mut add),
                None => (0..n).for_each(&mut add),
            }
            if den == 0.0 {
                if num == 0.0 {
                    1.0
                } else {
                    0.0
                }
            } else {
                1.0 - num / den
            }
        }
    }
}

/// sklearn's classification scorers on raw scores (one column for a
/// binary model, else one per class; non-loss scorings).
fn classification_score(
    scoring: EarlyStoppingScoring,
    raw: &[Vec<f64>],
    class_of: &dyn Fn(usize) -> usize,
    w: &dyn Fn(usize) -> f64,
    rows: Option<&[usize]>,
) -> f64 {
    let n = raw.first().map_or(0, Vec::len);
    let predicted = |i: usize| {
        if raw.len() == 1 {
            usize::from(raw[0][i] > 0.0)
        } else {
            let mut best = 0;
            for c in 1..raw.len() {
                if raw[c][i] > raw[best][i] {
                    best = c;
                }
            }
            best
        }
    };
    let proba_of_true = |i: usize| {
        let k = class_of(i);
        if raw.len() == 1 {
            let p1 = 1.0 / (1.0 + (-raw[0][i]).exp());
            if k == 1 {
                p1
            } else {
                1.0 - p1
            }
        } else {
            let max_s = raw.iter().map(|s| s[i]).fold(f64::NEG_INFINITY, f64::max);
            let sum: f64 = raw.iter().map(|s| (s[i] - max_s).exp()).sum();
            (raw[k][i] - max_s).exp() / sum
        }
    };
    match scoring {
        EarlyStoppingScoring::NegLogLoss => {
            let eps = f64::EPSILON;
            -weighted_mean(n, rows, w, &|i| {
                -proba_of_true(i).clamp(eps, 1.0 - eps).ln()
            })
        }
        // accuracy (also the classifier's default score)
        _ => weighted_mean(n, rows, w, &|i| {
            f64::from(u8::from(predicted(i) == class_of(i)))
        }),
    }
}

/// Scores recorded by sklearn-style early stopping: the negative mean loss
/// before the first tree and after every iteration.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct EarlyStoppingScores {
    pub train: Vec<f64>,
    /// Empty when the training data is the validation data.
    pub validation: Vec<f64>,
}

struct ScoreTracker {
    rule: SklearnStopping,
    scores: EarlyStoppingScores,
}

impl ScoreTracker {
    fn new(rule: SklearnStopping) -> Self {
        Self {
            rule,
            scores: EarlyStoppingScores::default(),
        }
    }

    /// Record the scores after an iteration (or of the baseline); returns
    /// sklearn's `_should_stop` on the validation scores, or on the training
    /// scores without validation data.
    fn record(&mut self, train: f64, validation: Option<f64>) -> bool {
        self.scores.train.push(train);
        let scores = match validation {
            Some(v) => {
                self.scores.validation.push(v);
                &self.scores.validation
            }
            None => &self.scores.train,
        };
        let reference_position = self.rule.n_iter_no_change + 1;
        if scores.len() < reference_position {
            return false;
        }
        let reference = scores[scores.len() - reference_position] + self.rule.tol;
        !scores[scores.len() - reference_position + 1..]
            .iter()
            .any(|&s| s > reference)
    }
}

/// sklearn's numerically stable `log(1 + exp(x))`.
fn sklearn_log1pexp(x: f64) -> f64 {
    if x <= -37.0 {
        x.exp()
    } else if x <= -2.0 {
        x.exp().ln_1p()
    } else if x <= 18.0 {
        (1.0 + x.exp()).ln()
    } else if x <= 33.3 {
        x + (-x).exp()
    } else {
        x
    }
}

impl LgbmRegressor {
    /// Fit with additional options (sample weights, init score, eval set, early stopping).
    pub fn fit_with_eval<'a, 'b>(
        &self,
        x: impl AsArray<'a, f64, Ix2>,
        y: impl AsArray<'b, f64, Ix1>,
        opts: &LgbmFitOptions<'_>,
    ) -> Result<FittedLgbmRegressor> {
        Ok(self.fit_scored(x.into(), y.into(), opts, None)?.0)
    }

    /// [`fit_with_eval`](Self::fit_with_eval), optionally with sklearn's
    /// early stopping (on `opts.eval_set` if given, else on the training
    /// data), returning its scores.
    pub(crate) fn fit_scored(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        opts: &LgbmFitOptions<'_>,
        stopping: Option<SklearnStopping>,
    ) -> Result<(FittedLgbmRegressor, EarlyStoppingScores)> {
        with_fit_threads(self.n_jobs, || {
            self.fit_scored_inner(x, y, opts, stopping, None)
        })
    }

    /// Continue `prev` on `(x, y)` up to `n_estimators` trees (sklearn's
    /// `warm_start=True`): its bins and trees are kept and the bagging draws
    /// of its rounds replayed, so on the same data the result equals a single
    /// fit with `n_estimators` trees. GOSS, DART, early stopping,
    /// `init_score` and a changed learning rate are errors.
    pub fn fit_warm_start<'a, 'b>(
        &self,
        prev: &FittedLgbmRegressor,
        x: impl AsArray<'a, f64, Ix2>,
        y: impl AsArray<'b, f64, Ix1>,
        opts: &LgbmFitOptions<'_>,
    ) -> Result<FittedLgbmRegressor> {
        let (x, y) = (x.into(), y.into());
        with_fit_threads(self.n_jobs, || {
            self.fit_scored_inner(x, y, opts, None, Some(prev))
        })
        .map(|(f, _)| f)
    }

    fn fit_scored_inner(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        opts: &LgbmFitOptions<'_>,
        stopping: Option<SklearnStopping>,
        prev: Option<&FittedLgbmRegressor>,
    ) -> Result<(FittedLgbmRegressor, EarlyStoppingScores)> {
        // LightGBM keeps the quantile level as a float32.
        if let LgbmObjective::Quantile { alpha } = self.objective {
            let alpha32 = f64::from(alpha as f32);
            if !self.sklearn_compat && alpha32 != alpha {
                let mut m = self.clone();
                m.objective = LgbmObjective::Quantile { alpha: alpha32 };
                return m.fit_scored_inner(x, y, opts, stopping, prev);
            }
        }
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
        if let Some(sw) = opts.sample_weight {
            if sw.len() != y.len() {
                return Err(RustMlError::ShapeMismatch(format!(
                    "sample_weight has {} entries but y has {}",
                    sw.len(),
                    y.len()
                )));
            }
        }
        if let Some(init) = opts.init_score {
            if init.len() != y.len() {
                return Err(RustMlError::ShapeMismatch(format!(
                    "init_score has {} entries but y has {}",
                    init.len(),
                    y.len()
                )));
            }
        }
        if !self.monotone_constraints.is_empty() && self.monotone_constraints.len() != x.ncols() {
            return Err(RustMlError::InvalidParameter(format!(
                "monotone_constraints has {} entries but X has {} features",
                self.monotone_constraints.len(),
                x.ncols()
            )));
        }
        validate_eval_opts(opts, x.ncols())?;
        validate_max_bins(self.max_bins)?;
        self.sampling().validate()?;
        validate_regression_objective(self.objective, y)?;
        if let Some(f) = prev {
            check_warm_start(
                f.trees.len(),
                f.learning_rate,
                f.n_features,
                x.ncols(),
                self.warm_start_model(),
                opts,
                stopping.is_some(),
            )?;
        }

        let n = x.nrows();
        let p = x.ncols();
        // LightGBM stores labels and weights as float32 (`label_t`).
        let lgbm_f32 = |v: f64| {
            if self.sklearn_compat {
                v
            } else {
                f64::from(v as f32)
            }
        };
        let y_vec: Vec<f64> = y.iter().map(|&v| lgbm_f32(v)).collect();
        let (binned_x, bins) = match prev {
            Some(f) => (bin_with(x, &f.bins), f.bins.clone()),
            None => compute_bins(
                x,
                self.max_bins,
                &self.categorical_features,
                self.min_data_in_bin,
                self.binning,
                self.seed.unwrap_or(0),
            ),
        };
        // Row-major copy for the per-row score updates.
        let binned_rows = binned_x.as_standard_layout().into_owned();
        // sklearn has no feature bundling (and its parity needs the plain
        // histogram sums).
        let bundles = (self.enable_bundle && !self.sklearn_compat)
            .then(|| FeatureBundles::find(&binned_x))
            .flatten();

        // sklearn treats any given weights as weights (its weighted
        // percentiles differ from the unweighted ones even for unit weights).
        let weights: Option<Array1<f64>> = if self.sklearn_compat {
            opts.sample_weight.map(|w| w.to_owned())
        } else {
            effective_sample_weight(opts.sample_weight).map(|w| w.mapv(lgbm_f32))
        };
        let sample_weight = weights.as_ref().map(|w| w.view());

        // Baseline / init_score (a warm start: the previous model's scores,
        // summed as during its fit)
        let (baseline, mut preds): (f64, Vec<f64>) = if let Some(f) = prev {
            let mut preds = vec![f.baseline; n];
            for tree in &f.trees {
                add_tree_scores(&mut preds[..], tree, &binned_rows, f.learning_rate);
            }
            (f.baseline, preds)
        } else if let Some(init) = opts.init_score {
            // init_score replaces the baseline entirely; we still store a
            // scalar "baseline" = mean(init) so predict() has a consistent
            // starting value for new rows.
            let m = init.iter().sum::<f64>() / n as f64;
            (m, init.to_vec())
        } else {
            let b = regression_baseline(&y_vec, sample_weight, self.objective, self.sklearn_compat);
            (b, vec![b; n])
        };

        let start = prev.map_or(0, |f| f.trees.len());
        let mut trees: Vec<TreeNode> = Vec::with_capacity(self.n_estimators);
        trees.extend(prev.into_iter().flat_map(|f| f.trees.iter().cloned()));
        let mut feature_gain = prev.map_or_else(|| vec![0.0f64; p], |f| f.feature_gain.clone());
        let mut feature_split_count =
            prev.map_or_else(|| vec![0u32; p], |f| f.feature_split_count.clone());

        let sampling = self.sampling();
        let mut samplers = LgbmSamplers::new(sampling, &binned_x, None);
        samplers.replay(start, 1, n);
        let mut feature_rng =
            feature_generator(self.feature_fraction_per_split, self.feature_subsample_seed);
        let has_missing = columns_with_missing(x, true);
        let fixed_bins = if self.sklearn_compat {
            Vec::new()
        } else {
            lgbm_fixed_bins(&binned_x, &bins, &has_missing)
        };
        let params = LgbmParams {
            has_missing,
            fixed_bins,
            ..self.params()
        };

        // Prepare eval set predictions if present. The eval rows MUST be
        // binned with the training bin edges: trees split on training bin
        // indices, so bins computed from the eval data would mean something
        // else entirely.
        let eval_binned = opts
            .eval_set
            .as_ref()
            .map(|(xe, _)| bin_matrix(xe.view(), &bins));
        let eval_y: Option<Vec<f64>> = opts.eval_set.as_ref().map(|(_, ye)| ye.to_vec());
        let mut eval_preds: Option<Vec<f64>> = opts
            .eval_set
            .as_ref()
            .map(|(xe, _)| vec![baseline; xe.nrows()]);

        let (scoring, small_train) = stopping
            .as_ref()
            .map_or((EarlyStoppingScoring::Loss, None), |s| {
                (s.scoring, s.small_train.clone())
            });
        let mut tracker = stopping.map(ScoreTracker::new);
        let log_link = has_log_link(self.objective);
        // sklearn's score of raw predictions: the negative mean loss, or a
        // scorer on the predictions.
        let score = |y: &[f64], raw: &[f64], w: Option<ArrayView1<f64>>, rows: Option<&[usize]>| {
            if scoring == EarlyStoppingScoring::Loss {
                return -eval_loss(y, raw, w, self.objective, self.huber_delta);
            }
            let pred = |i: usize| if log_link { raw[i].exp() } else { raw[i] };
            let weight = |i: usize| w.map_or(1.0, |w| w[i]);
            regression_score(scoring, y, &pred, &weight, rows)
        };
        let losses = |preds: &[f64], eval_preds: Option<&Vec<f64>>| {
            let train = score(&y_vec, preds, sample_weight, small_train.as_deref());
            let validation = eval_y
                .as_ref()
                .zip(eval_preds)
                .map(|(ye, ep)| score(ye, ep, opts.eval_sample_weight, None));
            (train, validation)
        };
        if let Some(t) = tracker.as_mut() {
            let (train, validation) = losses(&preds, eval_preds.as_ref());
            t.record(train, validation);
        }

        let mut best_eval_loss = f64::INFINITY;
        let mut best_iteration = start;
        let mut best_feature_gain = feature_gain.clone();
        let mut best_feature_split_count = feature_split_count.clone();
        let mut stale_rounds = 0usize;

        if self.boosting_type == BoostingType::Dart
            && (opts.init_score.is_some()
                || tracker.is_some()
                || self.early_stopping_rounds.is_some())
        {
            return Err(RustMlError::InvalidParameter(
                "DART supports neither init_score nor early stopping".into(),
            ));
        }
        let mut dart = (self.boosting_type == BoostingType::Dart)
            .then(|| DartState::new(self.dart, self.learning_rate, sampling.seeds.drop));

        for iter in start..self.n_estimators {
            // DART: drop earlier trees before the gradients are computed.
            let dart_step = dart.as_mut().map(|d| {
                let (drops, shrinkage) = d.drop_trees(trees.len());
                for &k in &drops {
                    trees[k].scale(-1.0);
                    add_tree_scores(&mut preds[..], &trees[k], &binned_rows, 1.0);
                }
                (drops, shrinkage)
            });
            let (mut gradients, mut hessians) = compute_regression_gradients(&y_vec, &preds, self);

            // Apply sample weights (multiply into gradients and hessians)
            if let Some(sw) = sample_weight {
                // LightGBM's weighted L2 rounds the residual to float32
                // before weighting it.
                if !self.sklearn_compat && matches!(self.objective, LgbmObjective::Regression) {
                    round_to_f32(&mut gradients);
                }
                for i in 0..n {
                    gradients[i] *= sw[i];
                    hessians[i] *= sw[i];
                }
            }
            // sklearn and LightGBM both keep gradients in float32.
            round_to_f32(&mut gradients);
            round_to_f32(&mut hessians);

            // Row selection: bagging or GOSS (which amplifies the
            // gradients of its sampled rows).
            let row_indices = samplers.rows(iter, n, &mut [&mut gradients], &mut [&mut hessians]);
            let feature_indices = samplers.tree_features(p);
            let refit = refit_percentile(self.objective).map(|a| (a, row_indices.clone()));
            let (mut tree, leaf_rows) = build_lgbm_tree(
                &binned_x,
                bundles.as_ref(),
                &gradients,
                &hessians,
                row_indices,
                &feature_indices,
                &bins,
                &params,
                &mut feature_gain,
                &mut feature_split_count,
                &mut feature_rng,
                samplers.node_sampler(),
            );
            if let Some((alpha, rows)) = refit {
                let residuals: Vec<f64> = y_vec.iter().zip(&preds).map(|(y, p)| y - p).collect();
                renew_leaf_outputs(
                    &mut tree,
                    &binned_x,
                    rows,
                    &residuals,
                    sample_weight,
                    alpha,
                    self.sklearn_compat,
                );
            }

            if let (Some(d), Some((drops, shrinkage))) = (dart.as_mut(), dart_step) {
                // The tree's leaves carry its weight; the first tree also
                // carries the initial score (LightGBM's `AddBias`).
                tree.scale(shrinkage);
                add_leaf_scores(&mut preds[..], &tree, &leaf_rows, &binned_rows, 1.0);
                if iter == 0 && baseline.abs() > LGBM_EPSILON {
                    tree.add_bias(baseline);
                }
                let (f_valid, f_train) = d.normalize(&drops, shrinkage);
                for &k in &drops {
                    trees[k].scale(f_valid);
                    trees[k].scale(f_train);
                    add_tree_scores(&mut preds[..], &trees[k], &binned_rows, 1.0);
                }
                d.push(shrinkage);
                trees.push(tree);
                best_iteration = iter + 1;
                continue;
            }
            // Update predictions (on ALL rows, not just sampled)
            add_leaf_scores(
                &mut preds[..],
                &tree,
                &leaf_rows,
                &binned_rows,
                self.learning_rate,
            );

            if let Some(t) = tracker.as_mut() {
                if let (Some(eb), Some(ep)) = (eval_binned.as_ref(), eval_preds.as_mut()) {
                    for (i, e) in ep.iter_mut().enumerate() {
                        *e += self.learning_rate * tree.predict_bins(row_of(eb, i));
                    }
                }
                let (train, validation) = losses(&preds, eval_preds.as_ref());
                trees.push(tree);
                best_iteration = iter + 1;
                if t.record(train, validation) {
                    break;
                }
                continue;
            }

            // Update eval predictions and check early stopping
            if let (Some(eb), Some(ye), Some(ep)) =
                (eval_binned.as_ref(), eval_y.as_ref(), eval_preds.as_mut())
            {
                for i in 0..eb.nrows() {
                    ep[i] += self.learning_rate * tree.predict_bins(row_of(eb, i));
                }
                let loss = eval_loss(
                    ye,
                    ep,
                    opts.eval_sample_weight,
                    self.objective,
                    self.huber_delta,
                );
                if loss < best_eval_loss - 1e-12 {
                    best_eval_loss = loss;
                    best_iteration = iter + 1;
                    best_feature_gain.clone_from(&feature_gain);
                    best_feature_split_count.clone_from(&feature_split_count);
                    stale_rounds = 0;
                } else {
                    stale_rounds += 1;
                }

                if opts.verbose > 0 && (iter + 1) % opts.verbose == 0 {
                    eprintln!(
                        "[lgbm] iter {}: eval_loss={:.6} best={:.6}",
                        iter + 1,
                        loss,
                        best_eval_loss
                    );
                }

                if let Some(rounds) = self.early_stopping_rounds {
                    if stale_rounds >= rounds {
                        trees.push(tree);
                        break;
                    }
                }
            } else {
                best_iteration = iter + 1;
            }

            trees.push(tree);
        }

        // With early stopping, keep only the trees up to the best iteration
        // (LightGBM's `best_iteration` semantics) so `predict` uses the best
        // model rather than the over-fitted tail.
        if opts.eval_set.is_some() && self.early_stopping_rounds.is_some() && best_iteration > 0 {
            trees.truncate(best_iteration);
            feature_gain = best_feature_gain;
            feature_split_count = best_feature_split_count;
        }

        // DART's leaves carry the tree weights and the initial score.
        let (baseline, learning_rate) = if dart.is_some() {
            (0.0, 1.0)
        } else {
            (baseline, self.learning_rate)
        };
        let fitted = FittedLgbmRegressor {
            trees,
            bins,
            baseline,
            learning_rate,
            n_features: p,
            feature_gain,
            feature_split_count,
            best_iteration,
            log_link: has_log_link(self.objective),
        };
        Ok((fitted, tracker.map(|t| t.scores).unwrap_or_default()))
    }
}

/// Compute evaluation loss for the current objective.
fn eval_loss(
    y: &[f64],
    preds: &[f64],
    weights: Option<ArrayView1<f64>>,
    objective: LgbmObjective,
    huber_delta: f64,
) -> f64 {
    let n = y.len();
    let mut total = 0.0;
    let mut wsum = 0.0;

    for i in 0..n {
        let w = weights.map_or(1.0, |w| w[i]);
        let r = preds[i] - y[i];
        let loss_i = match objective {
            LgbmObjective::Regression => 0.5 * r * r,
            LgbmObjective::RegressionL1 => r.abs(),
            // Pinball loss; r = prediction − target.
            LgbmObjective::Quantile { alpha } => {
                if r >= 0.0 {
                    (1.0 - alpha) * r
                } else {
                    -alpha * r
                }
            }
            LgbmObjective::Huber => {
                if r.abs() <= huber_delta {
                    0.5 * r * r
                } else {
                    huber_delta * (r.abs() - 0.5 * huber_delta)
                }
            }
            // Negative log-likelihoods of the raw (log-scale) score, up to
            // terms constant in the prediction.
            LgbmObjective::Poisson => preds[i].exp() - y[i] * preds[i],
            LgbmObjective::Gamma => preds[i] + y[i] * (-preds[i]).exp(),
            _ => 0.5 * r * r,
        };
        total += w * loss_i;
        wsum += w;
    }

    if wsum > 0.0 {
        total / wsum
    } else {
        total
    }
}

impl Fit<f64> for LgbmRegressor {
    type Fitted = FittedLgbmRegressor;

    fn fit_view(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<Self::Fitted> {
        self.fit_with_eval(x, y, &LgbmFitOptions::default())
    }
}

impl FitWeighted<f64> for LgbmRegressor {
    type Fitted = FittedLgbmRegressor;

    fn fit_weighted_view(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        sample_weight: Option<ArrayView1<f64>>,
    ) -> Result<Self::Fitted> {
        let opts = LgbmFitOptions {
            sample_weight,
            ..LgbmFitOptions::default()
        };
        self.fit_with_eval(x, y, &opts)
    }
}

impl FittedLgbmRegressor {
    /// Predictions after each boosting iteration (sklearn's
    /// `staged_predict`); the last stage equals `predict`.
    pub fn staged_predict<'a>(
        &self,
        x: impl AsArray<'a, f64, Ix2>,
    ) -> Result<impl Iterator<Item = Array1<f64>> + '_> {
        let x: ArrayView2<f64> = x.into();
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        let binned = bin_matrix(x, &self.bins);
        let mut raw = Array1::from_elem(x.nrows(), self.baseline);
        Ok(self.trees.iter().map(move |tree| {
            for (i, v) in raw.iter_mut().enumerate() {
                *v += self.learning_rate * tree.predict_bins(row_of(&binned, i));
            }
            if self.log_link {
                raw.mapv(f64::exp)
            } else {
                raw.clone()
            }
        }))
    }
}

impl Predict<f64> for FittedLgbmRegressor {
    fn predict_view(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }

        let n = x.nrows();
        let mut preds = Array1::from_elem(n, self.baseline);

        let binned = bin_matrix(x, &self.bins);
        let p = binned.ncols();
        let flat = binned.as_slice().expect("row-major binned matrix");
        {
            use rayon::prelude::*;
            preds
                .as_slice_mut()
                .unwrap()
                .par_iter_mut()
                .with_min_len(PREDICT_BLOCK_ROWS)
                .zip(flat.par_chunks_exact(p.max(1)))
                .for_each(|(pred, bins)| {
                    for tree in &self.trees {
                        *pred += self.learning_rate * tree.predict_bins(bins);
                    }
                });
        }
        if self.log_link {
            preds.mapv_inplace(f64::exp);
        }

        Ok(preds)
    }
}

// ============================================================
// LgbmClassifier
// ============================================================

/// Class weight strategy for LgbmClassifier (the shared
/// [`anofox_ml_core::ClassWeight`]).
pub type LgbmClassWeight = anofox_ml_core::ClassWeight;

/// LightGBM-style gradient boosting classifier.
///
/// Supports binary and multi-class classification with leaf-wise tree growth,
/// learned missing directions, L1+L2 regularization, and optional GOSS sampling.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LgbmClassifier {
    pub n_estimators: usize,
    pub num_leaves: usize,
    pub max_depth: Option<usize>,
    pub learning_rate: f64,
    pub min_child_samples: usize,
    pub min_child_weight: f64,
    pub min_split_gain: f64,
    pub reg_alpha: f64,
    pub reg_lambda: f64,
    /// Share of the rows of a bag (LightGBM's `bagging_fraction`); only
    /// used with `subsample_freq > 0`.
    pub subsample: f64,
    /// Re-bag every this many iterations; 0 (default) = no bagging
    /// (LightGBM's `bagging_freq`).
    #[serde(default)]
    pub subsample_freq: usize,
    /// Share of the features of a tree (LightGBM's `feature_fraction`).
    pub colsample_bytree: f64,
    /// Share of the tree's features a node may split on (LightGBM's
    /// `feature_fraction_bynode`).
    #[serde(default = "default_one")]
    pub colsample_bynode: f64,
    pub max_bins: usize,
    pub min_data_in_bin: usize,
    /// Numeric binning: LightGBM's (default) or sklearn's.
    #[serde(default)]
    pub binning: BinningMethod,
    pub boosting_type: BoostingType,
    /// DART parameters (with `boosting_type = Dart`).
    #[serde(default)]
    pub dart: DartParams,
    pub goss_top_rate: f64,
    pub goss_other_rate: f64,
    pub categorical_features: Vec<usize>,
    /// How categorical features are split.
    #[serde(default)]
    pub categorical_split: CategoricalSplitParams,
    pub monotone_constraints: Vec<i8>,
    pub class_weight: Option<LgbmClassWeight>,
    /// Balanced bagging of binary labels (LightGBM's `pos_bagging_fraction`
    /// and `neg_bagging_fraction`, default 1): the shares of the positive
    /// and the negative rows of a bag (with `subsample_freq > 0`), in place
    /// of `subsample`.
    #[serde(default = "default_one")]
    pub pos_bagging_fraction: f64,
    #[serde(default = "default_one")]
    pub neg_bagging_fraction: f64,
    pub early_stopping_rounds: Option<usize>,
    /// Exclusive feature bundling (LightGBM's `enable_bundle`, default
    /// true): see `FeatureBundles`. It changes the speed, not the model
    /// (up to rounding in the histograms).
    #[serde(default = "default_true")]
    pub enable_bundle: bool,
    /// Threads of a fit (LightGBM's `n_jobs`): `None` = the physical
    /// cores (at most the current rayon pool's threads), which on machines
    /// with SMT or efficiency cores is faster than all logical ones. The
    /// model does not depend on it.
    #[serde(default)]
    pub n_jobs: Option<usize>,
    /// LightGBM's `seed` (`random_state`): derives the seeds of bagging,
    /// GOSS, the feature fractions and DART's dropping as LightGBM does.
    /// `None` (default): LightGBM's default seeds (and `dart.drop_seed`).
    #[serde(default)]
    pub seed: Option<u64>,
    /// Scale the softmax hessian by K / (K - 1) as LightGBM does; the
    /// HistGradientBoosting facade turns it off (sklearn's hessian).
    #[serde(skip, default = "default_true")]
    pub(crate) softmax_hessian_factor: bool,
    /// sklearn's numerics: float32 gradients and hessians and its split
    /// search (see `find_best_split`); the HistGradientBoosting facade
    /// turns it on.
    #[serde(skip)]
    pub(crate) sklearn_compat: bool,
    /// Interaction constraints (see `LgbmParams::interaction_cst`); set by
    /// the HistGradientBoosting facade.
    #[serde(skip)]
    pub(crate) interaction_cst: Vec<Vec<usize>>,
    /// sklearn's `max_features` (share of features per split) and the seed
    /// of its `default_rng`; set by the HistGradientBoosting facade.
    #[serde(skip, default = "default_one")]
    pub(crate) feature_fraction_per_split: f64,
    #[serde(skip)]
    pub(crate) feature_subsample_seed: u32,
}

fn default_true() -> bool {
    true
}

impl LgbmClassifier {
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            num_leaves: 31,
            max_depth: None,
            learning_rate: 0.1,
            min_child_samples: 20,
            min_child_weight: 1e-3,
            min_split_gain: 0.0,
            reg_alpha: 0.0,
            reg_lambda: 0.0,
            subsample: 1.0,
            subsample_freq: 0,
            colsample_bytree: 1.0,
            colsample_bynode: 1.0,
            max_bins: DEFAULT_MAX_BINS,
            min_data_in_bin: 3,
            binning: BinningMethod::LightGbm,
            boosting_type: BoostingType::Gbdt,
            dart: DartParams::default(),
            goss_top_rate: 0.2,
            goss_other_rate: 0.1,
            categorical_features: Vec::new(),
            categorical_split: CategoricalSplitParams::default(),
            monotone_constraints: Vec::new(),
            class_weight: None,
            pos_bagging_fraction: 1.0,
            neg_bagging_fraction: 1.0,
            early_stopping_rounds: None,
            enable_bundle: true,
            n_jobs: None,
            seed: None,
            softmax_hessian_factor: true,
            sklearn_compat: false,
            interaction_cst: Vec::new(),
            feature_fraction_per_split: 1.0,
            feature_subsample_seed: 0,
        }
    }

    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }
    pub fn with_num_leaves(mut self, n: usize) -> Self {
        self.num_leaves = n;
        self
    }
    pub fn with_max_depth(mut self, d: Option<usize>) -> Self {
        self.max_depth = d;
        self
    }
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }
    pub fn with_min_child_samples(mut self, n: usize) -> Self {
        self.min_child_samples = n;
        self
    }
    pub fn with_min_child_weight(mut self, w: f64) -> Self {
        self.min_child_weight = w;
        self
    }
    pub fn with_reg_alpha(mut self, a: f64) -> Self {
        self.reg_alpha = a;
        self
    }
    pub fn with_reg_lambda(mut self, l: f64) -> Self {
        self.reg_lambda = l;
        self
    }
    pub fn with_subsample(mut self, s: f64) -> Self {
        self.subsample = s;
        self
    }
    pub fn with_subsample_freq(mut self, f: usize) -> Self {
        self.subsample_freq = f;
        self
    }
    pub fn with_colsample_bynode(mut self, c: f64) -> Self {
        self.colsample_bynode = c;
        self
    }
    pub fn with_colsample_bytree(mut self, c: f64) -> Self {
        self.colsample_bytree = c;
        self
    }
    /// Maximum number of bins for non-missing values (2..=255, default 255).
    pub fn with_max_bins(mut self, b: usize) -> Self {
        self.max_bins = b;
        self
    }
    pub fn with_min_data_in_bin(mut self, n: usize) -> Self {
        self.min_data_in_bin = n;
        self
    }
    /// Numeric binning method (LightGBM's by default).
    pub fn with_binning(mut self, method: BinningMethod) -> Self {
        self.binning = method;
        self
    }
    pub fn with_boosting_type(mut self, b: BoostingType) -> Self {
        self.boosting_type = b;
        self
    }
    /// DART (sets `boosting_type = Dart`).
    pub fn with_dart(mut self, p: DartParams) -> Self {
        self.boosting_type = BoostingType::Dart;
        self.dart = p;
        self
    }
    pub fn with_goss_rates(mut self, top: f64, other: f64) -> Self {
        self.goss_top_rate = top;
        self.goss_other_rate = other;
        self
    }
    pub fn with_categorical_features(mut self, feats: Vec<usize>) -> Self {
        self.categorical_features = feats;
        self
    }
    /// Categorical split parameters (LightGBM's `max_cat_to_onehot`,
    /// `cat_smooth`, `cat_l2`, `max_cat_threshold`, `min_data_per_group`).
    pub fn with_categorical_split(mut self, p: CategoricalSplitParams) -> Self {
        self.categorical_split = p;
        self
    }
    pub fn with_monotone_constraints(mut self, c: Vec<i8>) -> Self {
        self.monotone_constraints = c;
        self
    }
    /// Balanced bagging: the shares of the positive and the negative rows.
    pub fn with_bagging_fractions(mut self, pos: f64, neg: f64) -> Self {
        self.pos_bagging_fraction = pos;
        self.neg_bagging_fraction = neg;
        self
    }
    pub fn with_class_weight(mut self, w: Option<LgbmClassWeight>) -> Self {
        self.class_weight = w;
        self
    }
    pub fn with_early_stopping(mut self, r: Option<usize>) -> Self {
        self.early_stopping_rounds = r;
        self
    }
    /// Exclusive feature bundling on or off (LightGBM's `enable_bundle`).
    pub fn with_enable_bundle(mut self, on: bool) -> Self {
        self.enable_bundle = on;
        self
    }
    /// Threads of a fit (see the `n_jobs` field).
    pub fn with_n_jobs(mut self, n_jobs: Option<usize>) -> Self {
        self.n_jobs = n_jobs;
        self
    }
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = Some(s);
        self
    }

    fn warm_start_model(&self) -> WarmStartModel {
        WarmStartModel {
            n_estimators: self.n_estimators,
            learning_rate: self.learning_rate,
            boosting_type: self.boosting_type,
            early_stopping: self.early_stopping_rounds.is_some(),
            feature_fraction_per_split: self.feature_fraction_per_split,
            colsample_bynode: self.colsample_bynode,
        }
    }

    fn sampling(&self) -> SamplingParams {
        SamplingParams {
            boosting_type: self.boosting_type,
            learning_rate: self.learning_rate,
            subsample: self.subsample,
            subsample_freq: self.subsample_freq,
            colsample_bytree: self.colsample_bytree,
            colsample_bynode: self.colsample_bynode,
            goss_top_rate: self.goss_top_rate,
            goss_other_rate: self.goss_other_rate,
            seeds: LgbmSeeds::new(self.seed, self.dart.drop_seed),
        }
    }

    fn params(&self) -> LgbmParams {
        LgbmParams {
            num_leaves: self.num_leaves,
            max_depth: self.max_depth,
            min_child_samples: self.min_child_samples,
            min_child_weight: self.min_child_weight,
            // LightGBM compares its (unhalved) gain with min_split_gain; the
            // engine's gains are halved.
            min_split_gain: if self.sklearn_compat {
                self.min_split_gain
            } else {
                0.5 * self.min_split_gain
            },
            reg_alpha: self.reg_alpha,
            reg_lambda: self.reg_lambda,
            monotone_constraints: self.monotone_constraints.clone(),
            categorical: self.categorical_split,
            sklearn_compat: self.sklearn_compat,
            has_missing: Vec::new(),
            interaction_cst: self.interaction_cst.clone(),
            feature_fraction_per_split: self.feature_fraction_per_split,
            fixed_bins: Vec::new(),
        }
    }

    /// Compute per-sample class weights from the class_weight strategy.
    fn compute_class_weights(&self, y: ArrayView1<f64>) -> Result<Vec<f64>> {
        match &self.class_weight {
            None => Ok(vec![1.0; y.len()]),
            Some(cw) => Ok(anofox_ml_core::compute_sample_weight(cw, y)?.to_vec()),
        }
    }
}

impl Default for LgbmClassifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted LightGBM-style classifier.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedLgbmClassifier {
    /// Binary: single tree set. Multiclass: one set per class.
    tree_sets: Vec<Vec<TreeNode>>,
    bins: Vec<FeatureBins>,
    baselines: Vec<f64>,
    classes: Vec<f64>,
    learning_rate: f64,
    n_features: usize,
    feature_gain: Vec<f64>,
    feature_split_count: Vec<u32>,
    /// Number of boosting rounds kept (the best round when early stopping
    /// was used with an eval set).
    #[serde(default)]
    best_iteration: usize,
}

impl FittedLgbmClassifier {
    pub fn classes(&self) -> &[f64] {
        &self.classes
    }
    /// Best boosting round on the eval set (1-based). Equals the number of
    /// rounds trained when no eval set / early stopping was used.
    pub fn best_iteration(&self) -> usize {
        self.best_iteration
    }
    pub fn n_estimators(&self) -> usize {
        self.tree_sets.first().map_or(0, |ts| ts.len())
    }

    pub fn feature_importances(&self) -> Array1<f64> {
        let sum: f64 = self.feature_gain.iter().sum();
        if sum > 0.0 {
            Array1::from_vec(self.feature_gain.iter().map(|g| g / sum).collect())
        } else {
            Array1::zeros(self.n_features)
        }
    }

    pub fn predict_proba<'a>(&self, x: impl AsArray<'a, f64, Ix2>) -> Result<Array2<f64>> {
        let x: ArrayView2<f64> = x.into();
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }

        use rayon::prelude::*;
        let n = x.nrows();
        let n_classes = self.classes.len();

        if n_classes == 2 {
            let mut proba = Array2::zeros((n, 2));
            let binned = bin_matrix(x, &self.bins);
            proba
                .as_slice_mut()
                .unwrap()
                .par_chunks_mut(2)
                .with_min_len(PREDICT_BLOCK_ROWS)
                .enumerate()
                .for_each(|(i, row)| {
                    let mut score = self.baselines[0];
                    for tree in &self.tree_sets[0] {
                        score += self.learning_rate * tree.predict_bins(row_of(&binned, i));
                    }
                    let p1 = 1.0 / (1.0 + (-score).exp());
                    row[0] = 1.0 - p1;
                    row[1] = p1;
                });
            Ok(proba)
        } else {
            let mut proba = Array2::zeros((n, n_classes));
            let binned = bin_matrix(x, &self.bins);
            proba
                .as_slice_mut()
                .unwrap()
                .par_chunks_mut(n_classes)
                .with_min_len(PREDICT_BLOCK_ROWS)
                .enumerate()
                .for_each_init(
                    || vec![0.0; n_classes],
                    |scores, (i, row)| {
                        for (c, ts) in self.tree_sets.iter().enumerate() {
                            scores[c] = self.baselines[c];
                            for tree in ts {
                                scores[c] +=
                                    self.learning_rate * tree.predict_bins(row_of(&binned, i));
                            }
                        }
                        let max_s = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        let exp_sum: f64 = scores.iter().map(|&s| (s - max_s).exp()).sum();
                        for c in 0..n_classes {
                            row[c] = (scores[c] - max_s).exp() / exp_sum;
                        }
                    },
                );
            Ok(proba)
        }
    }
}

impl LgbmClassifier {
    /// Fit with additional options (sample_weight, init_score, eval_set, early stopping).
    pub fn fit_with_eval<'a, 'b>(
        &self,
        x: impl AsArray<'a, f64, Ix2>,
        y: impl AsArray<'b, f64, Ix1>,
        opts: &LgbmFitOptions<'_>,
    ) -> Result<FittedLgbmClassifier> {
        Ok(self.fit_scored(x.into(), y.into(), opts, None)?.0)
    }

    /// [`fit_with_eval`](Self::fit_with_eval), optionally with sklearn's
    /// early stopping (on `opts.eval_set` if given, else on the training
    /// data), returning its scores.
    pub(crate) fn fit_scored(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        opts: &LgbmFitOptions<'_>,
        stopping: Option<SklearnStopping>,
    ) -> Result<(FittedLgbmClassifier, EarlyStoppingScores)> {
        with_fit_threads(self.n_jobs, || {
            self.fit_scored_inner(x, y, opts, stopping, None)
        })
    }

    /// Continue `prev` on `(x, y)` up to `n_estimators` rounds (sklearn's
    /// `warm_start=True`; see [`LgbmRegressor::fit_warm_start`]). The
    /// classes must be those of `prev`.
    pub fn fit_warm_start<'a, 'b>(
        &self,
        prev: &FittedLgbmClassifier,
        x: impl AsArray<'a, f64, Ix2>,
        y: impl AsArray<'b, f64, Ix1>,
        opts: &LgbmFitOptions<'_>,
    ) -> Result<FittedLgbmClassifier> {
        let (x, y) = (x.into(), y.into());
        with_fit_threads(self.n_jobs, || {
            self.fit_scored_inner(x, y, opts, None, Some(prev))
        })
        .map(|(f, _)| f)
    }

    fn fit_scored_inner(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        opts: &LgbmFitOptions<'_>,
        stopping: Option<SklearnStopping>,
        prev: Option<&FittedLgbmClassifier>,
    ) -> Result<(FittedLgbmClassifier, EarlyStoppingScores)> {
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
        if !self.monotone_constraints.is_empty() && self.monotone_constraints.len() != x.ncols() {
            return Err(RustMlError::InvalidParameter(format!(
                "monotone_constraints has {} entries but X has {} features",
                self.monotone_constraints.len(),
                x.ncols()
            )));
        }
        if let Some(sw) = opts.sample_weight {
            if sw.len() != y.len() {
                return Err(RustMlError::ShapeMismatch(format!(
                    "sample_weight has {} entries but y has {}",
                    sw.len(),
                    y.len()
                )));
            }
        }
        if let Some(init) = opts.init_score {
            if init.len() != y.len() {
                return Err(RustMlError::ShapeMismatch(format!(
                    "init_score has {} entries but y has {}",
                    init.len(),
                    y.len()
                )));
            }
        }
        validate_eval_opts(opts, x.ncols())?;
        validate_max_bins(self.max_bins)?;
        self.sampling().validate()?;
        if let Some(f) = prev {
            check_warm_start(
                f.n_estimators(),
                f.learning_rate,
                f.n_features,
                x.ncols(),
                self.warm_start_model(),
                opts,
                stopping.is_some(),
            )?;
        }

        let n = x.nrows();
        let p = x.ncols();
        let (binned_x, bins) = match prev {
            Some(f) => (bin_with(x, &f.bins), f.bins.clone()),
            None => compute_bins(
                x,
                self.max_bins,
                &self.categorical_features,
                self.min_data_in_bin,
                self.binning,
                self.seed.unwrap_or(0),
            ),
        };
        // Row-major copy for the per-row score updates.
        let binned_rows = binned_x.as_standard_layout().into_owned();
        // sklearn has no feature bundling (and its parity needs the plain
        // histogram sums).
        let bundles = (self.enable_bundle && !self.sklearn_compat)
            .then(|| FeatureBundles::find(&binned_x))
            .flatten();

        let mut classes: Vec<f64> = y.iter().copied().collect();
        classes.sort_by(f64::total_cmp);
        classes.dedup();
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(RustMlError::InvalidParameter(
                "need at least 2 classes".into(),
            ));
        }
        if prev.is_some_and(|f| f.classes != classes) {
            return Err(RustMlError::InvalidParameter(
                "warm start: y has other classes than the fitted model".into(),
            ));
        }

        // Combine class_weight + sample_weight into a single effective weight per row.
        let class_weights = self.compute_class_weights(y)?;
        // LightGBM stores weights as float32 (`label_t`).
        let effective_weights: Vec<f64> = (0..n)
            .map(|i| class_weights[i] * opts.sample_weight.map_or(1.0, |sw| sw[i]))
            .map(|w| {
                if self.sklearn_compat {
                    w
                } else {
                    f64::from(w as f32)
                }
            })
            .collect();

        // Eval set: binned with the TRAINING bin edges, labels mapped to
        // class indices.
        let eval_binned = opts.eval_set.map(|(xe, _)| bin_matrix(xe.view(), &bins));
        let eval_class_idx: Option<Vec<usize>> = match opts.eval_set {
            Some((_, ye)) => Some(
                ye.iter()
                    .map(|&v| {
                        classes.iter().position(|&c| c == v).ok_or_else(|| {
                            RustMlError::InvalidParameter(format!(
                                "eval_set contains label {v} not seen in training"
                            ))
                        })
                    })
                    .collect::<Result<Vec<usize>>>()?,
            ),
            None => None,
        };
        let eval_w = |i: usize| opts.eval_sample_weight.map_or(1.0, |w| w[i]);
        let mut es = EarlyStopState::new(self.early_stopping_rounds, opts.verbose);
        let (scoring, small_train) = stopping
            .as_ref()
            .map_or((EarlyStoppingScoring::Loss, None), |s| {
                (s.scoring, s.small_train.clone())
            });
        let mut tracker = stopping.map(ScoreTracker::new);
        // sklearn's mean losses (weighted by `w`) of raw scores.
        let mean = |total: f64, wsum: f64| if wsum > 0.0 { total / wsum } else { total };
        let binary_loss =
            |raw: &[f64], is_pos: &dyn Fn(usize) -> bool, w: &dyn Fn(usize) -> f64| {
                let (mut total, mut wsum) = (0.0, 0.0);
                for (i, &r) in raw.iter().enumerate() {
                    let yi = if is_pos(i) { 1.0 } else { 0.0 };
                    total += w(i) * (sklearn_log1pexp(r) - yi * r);
                    wsum += w(i);
                }
                mean(total, wsum)
            };
        let multi_loss =
            |raw: &[Vec<f64>], class_of: &dyn Fn(usize) -> usize, w: &dyn Fn(usize) -> f64| {
                let (mut total, mut wsum) = (0.0, 0.0);
                for i in 0..raw.first().map_or(0, Vec::len) {
                    let max_s = raw.iter().map(|s| s[i]).fold(f64::NEG_INFINITY, f64::max);
                    let sum_exps: f64 = raw.iter().map(|s| (s[i] - max_s).exp()).sum();
                    total += w(i) * (sum_exps.ln() + max_s - raw[class_of(i)][i]);
                    wsum += w(i);
                }
                mean(total, wsum)
            };
        let train_class = |i: usize| classes.iter().position(|&c| c == y[i]).unwrap_or(0);
        let train_w = |i: usize| effective_weights[i];

        // Balanced bagging (binary labels): its own fractions for the
        // positive and the negative rows.
        let balanced = (n_classes == 2).then(|| {
            let positives = (0..n).map(|i| train_class(i) == 1).collect();
            (
                positives,
                self.pos_bagging_fraction,
                self.neg_bagging_fraction,
            )
        });
        let sampling = self.sampling();
        let mut samplers = LgbmSamplers::new(sampling, &binned_x, balanced);
        // A warm start replays the draws of the fitted rounds (one tree per
        // class and round for more than two classes).
        let start = prev.map_or(0, |f| f.n_estimators());
        let trees_per_round = if n_classes == 2 { 1 } else { n_classes };
        samplers.replay(start, trees_per_round, n);
        if self.boosting_type == BoostingType::Dart
            && (opts.init_score.is_some()
                || tracker.is_some()
                || self.early_stopping_rounds.is_some())
        {
            return Err(RustMlError::InvalidParameter(
                "DART supports neither init_score nor early stopping".into(),
            ));
        }
        let mut dart = (self.boosting_type == BoostingType::Dart)
            .then(|| DartState::new(self.dart, self.learning_rate, sampling.seeds.drop));
        let mut feature_rng =
            feature_generator(self.feature_fraction_per_split, self.feature_subsample_seed);
        let has_missing = columns_with_missing(x, true);
        let fixed_bins = if self.sklearn_compat {
            Vec::new()
        } else {
            lgbm_fixed_bins(&binned_x, &bins, &has_missing)
        };
        let params = LgbmParams {
            has_missing,
            fixed_bins,
            ..self.params()
        };
        let mut feature_gain = prev.map_or_else(|| vec![0.0f64; p], |f| f.feature_gain.clone());
        let mut feature_split_count =
            prev.map_or_else(|| vec![0u32; p], |f| f.feature_split_count.clone());
        // The previous model's raw scores of tree set `c`, summed as during
        // its fit.
        let prev_scores = |f: &FittedLgbmClassifier, c: usize| {
            let mut raw = vec![f.baselines[c]; n];
            for tree in &f.tree_sets[c] {
                add_tree_scores(&mut raw[..], tree, &binned_rows, f.learning_rate);
            }
            raw
        };

        if n_classes == 2 {
            let pos_class = classes[1];
            let labels: Vec<f64> = y
                .iter()
                .map(|&v| if v == pos_class { 1.0 } else { 0.0 })
                .collect();
            // Weighted positive rate (LightGBM's BoostFromScore).
            let pos_frac: f64 = labels
                .iter()
                .zip(&effective_weights)
                .map(|(l, w)| l * w)
                .sum::<f64>()
                / effective_weights.iter().sum::<f64>();
            let baseline = prev.map_or((pos_frac / (1.0 - pos_frac + 1e-15)).ln(), |f| {
                f.baselines[0]
            });
            let mut raw = if let Some(f) = prev {
                prev_scores(f, 0)
            } else if let Some(init) = opts.init_score {
                init.to_vec()
            } else {
                vec![baseline; n]
            };
            let mut trees: Vec<TreeNode> = Vec::with_capacity(self.n_estimators);
            trees.extend(
                prev.into_iter()
                    .flat_map(|f| f.tree_sets[0].iter().cloned()),
            );
            let mut eval_raw: Option<Vec<f64>> =
                eval_binned.as_ref().map(|eb| vec![baseline; eb.nrows()]);
            let binary_losses = |raw: &[f64], eval_raw: Option<&Vec<f64>>| {
                let train_class = |i: usize| usize::from(labels[i] == 1.0);
                if scoring != EarlyStoppingScoring::Loss {
                    let train = classification_score(
                        scoring,
                        &[raw.to_vec()],
                        &train_class,
                        &train_w,
                        small_train.as_deref(),
                    );
                    let validation = eval_raw.zip(eval_class_idx.as_ref()).map(|(er, ci)| {
                        classification_score(
                            scoring,
                            std::slice::from_ref(er),
                            &|i| ci[i],
                            &eval_w,
                            None,
                        )
                    });
                    return (train, validation);
                }
                let train = -binary_loss(raw, &|i| labels[i] == 1.0, &train_w);
                let validation = eval_raw
                    .zip(eval_class_idx.as_ref())
                    .map(|(er, ci)| -binary_loss(er, &|i| ci[i] == 1, &eval_w));
                (train, validation)
            };
            if let Some(t) = tracker.as_mut() {
                let (train, validation) = binary_losses(&raw, eval_raw.as_ref());
                t.record(train, validation);
            }

            for iter in start..self.n_estimators {
                // DART: drop earlier trees before the gradients are computed.
                let dart_step = dart.as_mut().map(|d| {
                    let (drops, shrinkage) = d.drop_trees(trees.len());
                    for &k in &drops {
                        trees[k].scale(-1.0);
                        add_tree_scores(&mut raw[..], &trees[k], &binned_rows, 1.0);
                    }
                    (drops, shrinkage)
                });
                let mut gradients: Vec<f64> = (0..n)
                    .map(|i| {
                        let pr = 1.0 / (1.0 + (-raw[i]).exp());
                        (pr - labels[i]) * effective_weights[i]
                    })
                    .collect();
                let mut hessians: Vec<f64> = (0..n)
                    .map(|i| {
                        let pr = 1.0 / (1.0 + (-raw[i]).exp());
                        (pr * (1.0 - pr)).max(1e-12) * effective_weights[i]
                    })
                    .collect();

                round_to_f32(&mut gradients);
                round_to_f32(&mut hessians);

                let row_indices =
                    samplers.rows(iter, n, &mut [&mut gradients], &mut [&mut hessians]);
                let feature_indices = samplers.tree_features(p);

                let (tree, leaf_rows) = build_lgbm_tree(
                    &binned_x,
                    bundles.as_ref(),
                    &gradients,
                    &hessians,
                    row_indices,
                    &feature_indices,
                    &bins,
                    &params,
                    &mut feature_gain,
                    &mut feature_split_count,
                    &mut feature_rng,
                    samplers.node_sampler(),
                );

                if let (Some(d), Some((drops, shrinkage))) = (dart.as_mut(), dart_step) {
                    let mut tree = tree;
                    tree.scale(shrinkage);
                    add_leaf_scores(&mut raw[..], &tree, &leaf_rows, &binned_rows, 1.0);
                    if iter == 0 && baseline.abs() > LGBM_EPSILON {
                        tree.add_bias(baseline);
                    }
                    let (f_valid, f_train) = d.normalize(&drops, shrinkage);
                    for &k in &drops {
                        trees[k].scale(f_valid);
                        trees[k].scale(f_train);
                        add_tree_scores(&mut raw[..], &trees[k], &binned_rows, 1.0);
                    }
                    d.push(shrinkage);
                    trees.push(tree);
                    continue;
                }
                add_leaf_scores(
                    &mut raw[..],
                    &tree,
                    &leaf_rows,
                    &binned_rows,
                    self.learning_rate,
                );

                if let Some(t) = tracker.as_mut() {
                    if let (Some(eb), Some(er)) = (eval_binned.as_ref(), eval_raw.as_mut()) {
                        for (i, e) in er.iter_mut().enumerate() {
                            *e += self.learning_rate * tree.predict_bins(row_of(eb, i));
                        }
                    }
                    let (train, validation) = binary_losses(&raw, eval_raw.as_ref());
                    trees.push(tree);
                    if t.record(train, validation) {
                        break;
                    }
                    continue;
                }

                let mut stop = false;
                if let (Some(eb), Some(er), Some(ci)) = (
                    eval_binned.as_ref(),
                    eval_raw.as_mut(),
                    eval_class_idx.as_ref(),
                ) {
                    let mut total = 0.0;
                    let mut wsum = 0.0;
                    for i in 0..eb.nrows() {
                        er[i] += self.learning_rate * tree.predict_bins(row_of(eb, i));
                        let p1 = (1.0 / (1.0 + (-er[i]).exp())).clamp(1e-15, 1.0 - 1e-15);
                        let li = if ci[i] == 1 {
                            -p1.ln()
                        } else {
                            -(1.0 - p1).ln()
                        };
                        total += eval_w(i) * li;
                        wsum += eval_w(i);
                    }
                    let loss = if wsum > 0.0 { total / wsum } else { total };
                    stop = es.update(iter, loss, &feature_gain, &feature_split_count);
                }

                trees.push(tree);
                if stop {
                    break;
                }
            }

            let (keep, best_iteration) =
                es.finish(trees.len(), &mut feature_gain, &mut feature_split_count);
            trees.truncate(keep);

            // DART's leaves carry the tree weights and the initial score.
            let (baseline, learning_rate) = if dart.is_some() {
                (0.0, 1.0)
            } else {
                (baseline, self.learning_rate)
            };
            let fitted = FittedLgbmClassifier {
                tree_sets: vec![trees],
                bins,
                baselines: vec![baseline],
                classes,
                learning_rate,
                n_features: p,
                feature_gain,
                feature_split_count,
                best_iteration,
            };
            Ok((fitted, tracker.map(|t| t.scores).unwrap_or_default()))
        } else {
            // Multiclass: softmax with one tree per class per round.
            let mut tree_sets: Vec<Vec<TreeNode>> = match prev {
                Some(f) => f.tree_sets.clone(),
                None => vec![Vec::with_capacity(self.n_estimators); n_classes],
            };
            let mut baselines = Vec::with_capacity(n_classes);
            let mut raw_scores = vec![vec![0.0; n]; n_classes];

            let total_weight: f64 = effective_weights.iter().sum();
            for (c, &cls) in classes.iter().enumerate() {
                let class_weight: f64 = y
                    .iter()
                    .zip(&effective_weights)
                    .filter(|(&v, _)| v == cls)
                    .map(|(_, w)| w)
                    .sum();
                let prior = (class_weight / total_weight).max(1e-15);
                let bl = prev.map_or(prior.ln(), |f| f.baselines[c]);
                baselines.push(bl);
                raw_scores[c] = match prev {
                    Some(f) => prev_scores(f, c),
                    None => vec![bl; n],
                };
            }
            let mut eval_raw: Option<Vec<Vec<f64>>> = eval_binned
                .as_ref()
                .map(|eb| baselines.iter().map(|&bl| vec![bl; eb.nrows()]).collect());
            let multi_losses = |raw: &[Vec<f64>], eval_raw: Option<&Vec<Vec<f64>>>| {
                if scoring != EarlyStoppingScoring::Loss {
                    let train = classification_score(
                        scoring,
                        raw,
                        &train_class,
                        &train_w,
                        small_train.as_deref(),
                    );
                    let validation = eval_raw.zip(eval_class_idx.as_ref()).map(|(er, ci)| {
                        classification_score(scoring, er, &|i| ci[i], &eval_w, None)
                    });
                    return (train, validation);
                }
                let train = -multi_loss(raw, &train_class, &train_w);
                let validation = eval_raw
                    .zip(eval_class_idx.as_ref())
                    .map(|(er, ci)| -multi_loss(er, &|i| ci[i], &eval_w));
                (train, validation)
            };
            if let Some(t) = tracker.as_mut() {
                let (train, validation) = multi_losses(&raw_scores, eval_raw.as_ref());
                t.record(train, validation);
            }

            for iter in start..self.n_estimators {
                // DART: drop earlier trees (of every class) first.
                let dart_step = dart.as_mut().map(|d| {
                    let (drops, shrinkage) = d.drop_trees(tree_sets[0].len());
                    for &k in &drops {
                        for (ts, rs) in tree_sets.iter_mut().zip(raw_scores.iter_mut()) {
                            ts[k].scale(-1.0);
                            add_tree_scores(&mut rs[..], &ts[k], &binned_rows, 1.0);
                        }
                    }
                    (drops, shrinkage)
                });
                // Softmax probabilities.
                let mut probas = vec![vec![0.0; n_classes]; n];
                for i in 0..n {
                    let max_s = raw_scores
                        .iter()
                        .map(|s| s[i])
                        .fold(f64::NEG_INFINITY, f64::max);
                    let exp_sum: f64 = raw_scores.iter().map(|s| (s[i] - max_s).exp()).sum();
                    for c in 0..n_classes {
                        probas[i][c] = (raw_scores[c][i] - max_s).exp() / exp_sum;
                    }
                }

                // LightGBM scales the softmax hessian by K / (K - 1).
                let hess_factor = if self.softmax_hessian_factor {
                    n_classes as f64 / (n_classes as f64 - 1.0)
                } else {
                    1.0
                };
                // The gradients of every class, then one row sample for the
                // iteration's trees.
                let (mut all_gradients, mut all_hessians): (Vec<Vec<f64>>, Vec<Vec<f64>>) = classes
                    .iter()
                    .enumerate()
                    .map(|(c, &cls)| {
                        let mut gradients: Vec<f64> = (0..n)
                            .map(|i| {
                                let lbl = if y[i] == cls { 1.0 } else { 0.0 };
                                (probas[i][c] - lbl) * effective_weights[i]
                            })
                            .collect();
                        let mut hessians: Vec<f64> = (0..n)
                            .map(|i| {
                                // LightGBM's order: `factor * p * (1 - p)`.
                                (hess_factor * probas[i][c] * (1.0 - probas[i][c])).max(1e-12)
                                    * effective_weights[i]
                            })
                            .collect();
                        round_to_f32(&mut gradients);
                        round_to_f32(&mut hessians);
                        (gradients, hessians)
                    })
                    .unzip();
                let row_indices = {
                    let mut g: Vec<&mut [f64]> =
                        all_gradients.iter_mut().map(|v| v.as_mut_slice()).collect();
                    let mut h: Vec<&mut [f64]> =
                        all_hessians.iter_mut().map(|v| v.as_mut_slice()).collect();
                    samplers.rows(iter, n, &mut g, &mut h)
                };
                for (c, (gradients, hessians)) in
                    all_gradients.iter().zip(&all_hessians).enumerate()
                {
                    let row_indices = row_indices.clone();
                    let feature_indices = samplers.tree_features(p);

                    let (tree, leaf_rows) = build_lgbm_tree(
                        &binned_x,
                        bundles.as_ref(),
                        gradients,
                        hessians,
                        row_indices,
                        &feature_indices,
                        &bins,
                        &params,
                        &mut feature_gain,
                        &mut feature_split_count,
                        &mut feature_rng,
                        samplers.node_sampler(),
                    );

                    let mut tree = tree;
                    if let Some((_, shrinkage)) = &dart_step {
                        tree.scale(*shrinkage);
                        add_leaf_scores(
                            &mut raw_scores[c][..],
                            &tree,
                            &leaf_rows,
                            &binned_rows,
                            1.0,
                        );
                        if iter == 0 && baselines[c].abs() > LGBM_EPSILON {
                            tree.add_bias(baselines[c]);
                        }
                    } else {
                        add_leaf_scores(
                            &mut raw_scores[c][..],
                            &tree,
                            &leaf_rows,
                            &binned_rows,
                            self.learning_rate,
                        );
                    }
                    if let (Some(eb), Some(er)) = (eval_binned.as_ref(), eval_raw.as_mut()) {
                        for i in 0..eb.nrows() {
                            er[c][i] += self.learning_rate * tree.predict_bins(row_of(eb, i));
                        }
                    }

                    tree_sets[c].push(tree);
                }

                if let (Some(d), Some((drops, shrinkage))) = (dart.as_mut(), dart_step) {
                    let (f_valid, f_train) = d.normalize(&drops, shrinkage);
                    for &k in &drops {
                        for (ts, rs) in tree_sets.iter_mut().zip(raw_scores.iter_mut()) {
                            ts[k].scale(f_valid);
                            ts[k].scale(f_train);
                            add_tree_scores(&mut rs[..], &ts[k], &binned_rows, 1.0);
                        }
                    }
                    d.push(shrinkage);
                    continue;
                }

                if let Some(t) = tracker.as_mut() {
                    let (train, validation) = multi_losses(&raw_scores, eval_raw.as_ref());
                    if t.record(train, validation) {
                        break;
                    }
                    continue;
                }

                // Multiclass log-loss on the eval set after this round.
                if let (Some(er), Some(ci)) = (eval_raw.as_ref(), eval_class_idx.as_ref()) {
                    let mut total = 0.0;
                    let mut wsum = 0.0;
                    for (i, &yi) in ci.iter().enumerate() {
                        let max_s = er.iter().map(|s| s[i]).fold(f64::NEG_INFINITY, f64::max);
                        let lse = max_s + er.iter().map(|s| (s[i] - max_s).exp()).sum::<f64>().ln();
                        total += eval_w(i) * (lse - er[yi][i]);
                        wsum += eval_w(i);
                    }
                    let loss = if wsum > 0.0 { total / wsum } else { total };
                    if es.update(iter, loss, &feature_gain, &feature_split_count) {
                        break;
                    }
                }
            }

            let n_rounds = tree_sets.first().map_or(0, |ts| ts.len());
            let (keep, best_iteration) =
                es.finish(n_rounds, &mut feature_gain, &mut feature_split_count);
            for ts in &mut tree_sets {
                ts.truncate(keep);
            }

            let (baselines, learning_rate) = if dart.is_some() {
                (vec![0.0; baselines.len()], 1.0)
            } else {
                (baselines, self.learning_rate)
            };
            let fitted = FittedLgbmClassifier {
                tree_sets,
                bins,
                baselines,
                classes,
                learning_rate,
                n_features: p,
                feature_gain,
                feature_split_count,
                best_iteration,
            };
            Ok((fitted, tracker.map(|t| t.scores).unwrap_or_default()))
        }
    }
}

/// Book-keeping for eval-set tracking / early stopping in the classifier.
struct EarlyStopState {
    rounds: Option<usize>,
    verbose: usize,
    tracking: bool,
    best_loss: f64,
    best_iteration: usize,
    stale: usize,
    best_gain: Vec<f64>,
    best_split_count: Vec<u32>,
}

impl EarlyStopState {
    fn new(rounds: Option<usize>, verbose: usize) -> Self {
        Self {
            rounds,
            verbose,
            tracking: false,
            best_loss: f64::INFINITY,
            best_iteration: 0,
            stale: 0,
            best_gain: Vec::new(),
            best_split_count: Vec::new(),
        }
    }

    /// Record the eval loss after round `iter` (0-based). Returns `true` when
    /// training should stop.
    fn update(&mut self, iter: usize, loss: f64, gain: &[f64], split_count: &[u32]) -> bool {
        self.tracking = true;
        if loss < self.best_loss - 1e-12 {
            self.best_loss = loss;
            self.best_iteration = iter + 1;
            self.best_gain = gain.to_vec();
            self.best_split_count = split_count.to_vec();
            self.stale = 0;
        } else {
            self.stale += 1;
        }
        if self.verbose > 0 && (iter + 1).is_multiple_of(self.verbose) {
            eprintln!(
                "[lgbm] iter {}: eval_loss={:.6} best={:.6}",
                iter + 1,
                loss,
                self.best_loss
            );
        }
        self.rounds.is_some_and(|r| self.stale >= r)
    }

    /// Returns `(rounds_to_keep, best_iteration)`. With early stopping the
    /// model is truncated to the best round (and the importances are rolled
    /// back to it); otherwise all rounds are kept. `best_iteration` is the
    /// best eval round when an eval set was tracked, else `n_rounds`.
    fn finish(
        self,
        n_rounds: usize,
        gain: &mut Vec<f64>,
        split_count: &mut Vec<u32>,
    ) -> (usize, usize) {
        if !self.tracking || self.best_iteration == 0 {
            return (n_rounds, n_rounds);
        }
        if self.rounds.is_some() {
            *gain = self.best_gain;
            *split_count = self.best_split_count;
            (self.best_iteration, self.best_iteration)
        } else {
            (n_rounds, self.best_iteration)
        }
    }
}

impl Fit<f64> for LgbmClassifier {
    type Fitted = FittedLgbmClassifier;

    fn fit_view(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<Self::Fitted> {
        self.fit_with_eval(x, y, &LgbmFitOptions::default())
    }
}

impl FitWeighted<f64> for LgbmClassifier {
    type Fitted = FittedLgbmClassifier;

    fn fit_weighted_view(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        sample_weight: Option<ArrayView1<f64>>,
    ) -> Result<Self::Fitted> {
        let opts = LgbmFitOptions {
            sample_weight,
            ..LgbmFitOptions::default()
        };
        self.fit_with_eval(x, y, &opts)
    }
}

impl Predict<f64> for FittedLgbmClassifier {
    fn predict_view(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        Ok(self.labels_of(&self.predict_proba(x)?))
    }
}

impl FittedLgbmClassifier {
    /// The most probable class of every row of `proba`.
    fn labels_of(&self, proba: &Array2<f64>) -> Array1<f64> {
        proba
            .rows()
            .into_iter()
            .map(|row| {
                let mut best_c = 0;
                for c in 1..row.len() {
                    if row[c] > row[best_c] {
                        best_c = c;
                    }
                }
                self.classes[best_c]
            })
            .collect()
    }

    /// Class probabilities after each boosting iteration (sklearn's
    /// `staged_predict_proba`); the last stage equals
    /// [`predict_proba`](Self::predict_proba).
    pub fn staged_predict_proba<'a>(
        &self,
        x: impl AsArray<'a, f64, Ix2>,
    ) -> Result<impl Iterator<Item = Array2<f64>> + '_> {
        let x: ArrayView2<f64> = x.into();
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        let binned = bin_matrix(x, &self.bins);
        let n = x.nrows();
        let binary = self.classes.len() == 2;
        let n_sets = if binary { 1 } else { self.tree_sets.len() };
        let mut scores: Vec<Vec<f64>> = (0..n_sets).map(|c| vec![self.baselines[c]; n]).collect();
        Ok((0..self.n_estimators()).map(move |stage| {
            for (c, s) in scores.iter_mut().enumerate() {
                let tree = &self.tree_sets[c][stage];
                for (i, v) in s.iter_mut().enumerate() {
                    *v += self.learning_rate * tree.predict_bins(row_of(&binned, i));
                }
            }
            if binary {
                Array2::from_shape_fn((n, 2), |(i, c)| {
                    let p1 = 1.0 / (1.0 + (-scores[0][i]).exp());
                    if c == 1 {
                        p1
                    } else {
                        1.0 - p1
                    }
                })
            } else {
                let mut proba = Array2::zeros((n, n_sets));
                for i in 0..n {
                    let max_s = scores
                        .iter()
                        .map(|s| s[i])
                        .fold(f64::NEG_INFINITY, f64::max);
                    let exp_sum: f64 = scores.iter().map(|s| (s[i] - max_s).exp()).sum();
                    for c in 0..n_sets {
                        proba[[i, c]] = (scores[c][i] - max_s).exp() / exp_sum;
                    }
                }
                proba
            }
        }))
    }

    /// Predicted classes after each boosting iteration (sklearn's
    /// `staged_predict`).
    pub fn staged_predict<'a>(
        &self,
        x: impl AsArray<'a, f64, Ix2>,
    ) -> Result<impl Iterator<Item = Array1<f64>> + '_> {
        Ok(self
            .staged_predict_proba(x)?
            .map(move |proba| self.labels_of(&proba)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn dart_rejects_early_stopping_and_init_scores() {
        let x = Array2::from_shape_fn((60, 2), |(i, j)| ((i * (j + 3)) % 17) as f64);
        let y = x.column(0).to_owned();
        let dart = LgbmRegressor::new().with_dart(DartParams::default());
        let init = Array1::zeros(60);
        let opts = LgbmFitOptions {
            init_score: Some(init.view()),
            ..LgbmFitOptions::default()
        };
        assert!(dart.fit_with_eval(&x, &y, &opts).is_err());
        assert!(dart
            .clone()
            .with_early_stopping(Some(5))
            .fit(&x, &y)
            .is_err());
        // Without drops (skip_drop = 1) DART is GBDT.
        let no_drop = LgbmRegressor::new().with_dart(DartParams {
            skip_drop: 1.0,
            ..DartParams::default()
        });
        let a = no_drop.fit(&x, &y).unwrap().predict(&x).unwrap();
        let b = LgbmRegressor::new()
            .fit(&x, &y)
            .unwrap()
            .predict(&x)
            .unwrap();
        for (u, v) in a.iter().zip(&b) {
            assert!((u - v).abs() < 1e-9, "{u} vs {v}");
        }
    }

    #[test]
    fn test_lgbm_regressor_basic() {
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
            [10.0]
        ];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

        let model = LgbmRegressor::new()
            .with_n_estimators(50)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .with_learning_rate(0.1);

        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(*p, *t, epsilon = 3.0);
        }
    }

    #[test]
    fn test_lgbm_regressor_nan_handling() {
        // Data with NaN values
        let x = array![
            [1.0],
            [2.0],
            [f64::NAN],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [f64::NAN],
            [9.0],
            [10.0]
        ];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

        let fitted = LgbmRegressor::new()
            .with_n_estimators(20)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .fit(&x, &y)
            .unwrap();

        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(p.is_finite(), "prediction should be finite");
        }
    }

    #[test]
    fn test_lgbm_classifier_binary() {
        let x = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0],
            [10.0, 1.0],
            [11.0, 1.0],
            [12.0, 1.0],
            [13.0, 1.0],
            [14.0, 1.0]
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let fitted = LgbmClassifier::new()
            .with_n_estimators(30)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .fit(&x, &y)
            .unwrap();

        let preds = fitted.predict(&x).unwrap();
        let correct: usize = preds.iter().zip(y.iter()).filter(|(&p, &t)| p == t).count();
        assert!(
            correct >= 9,
            "should classify most correctly, got {}/10",
            correct
        );
    }

    #[test]
    fn test_lgbm_classifier_multiclass() {
        let x = array![
            [0.0, 0.0],
            [0.5, 0.5],
            [0.0, 0.5],
            [5.0, 0.0],
            [5.5, 0.5],
            [5.0, 0.5],
            [0.0, 10.0],
            [0.5, 10.5],
            [0.0, 10.5]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];

        let fitted = LgbmClassifier::new()
            .with_n_estimators(30)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .fit(&x, &y)
            .unwrap();

        assert_eq!(fitted.classes(), &[0.0, 1.0, 2.0]);
        let proba = fitted.predict_proba(&x).unwrap();
        assert_eq!(proba.ncols(), 3);

        // Rows should sum to 1
        for i in 0..x.nrows() {
            let row_sum: f64 = (0..3).map(|c| proba[[i, c]]).sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_lgbm_regressor_goss() {
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
            [10.0]
        ];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

        let fitted = LgbmRegressor::new()
            .with_n_estimators(30)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .with_boosting_type(BoostingType::Goss)
            .fit(&x, &y)
            .unwrap();

        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 10);
        for &p in preds.iter() {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn test_lgbm_regressor_l1_l2() {
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
            [10.0]
        ];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

        // With strong L2, leaf values shrink toward zero
        let fitted = LgbmRegressor::new()
            .with_n_estimators(20)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .with_reg_lambda(100.0)
            .fit(&x, &y)
            .unwrap();

        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn test_lgbm_regressor_subsample() {
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
            [10.0]
        ];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

        let fitted = LgbmRegressor::new()
            .with_n_estimators(20)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .with_subsample(0.5)
            .with_subsample_freq(1)
            .with_colsample_bytree(0.5)
            .fit(&x, &y)
            .unwrap();

        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 10);
    }

    #[test]
    fn test_lgbm_regressor_num_leaves_controls_complexity() {
        // y = 2*x; deeper trees (more leaves) should fit training data more closely.
        let x = Array2::from_shape_vec((20, 1), (0..20).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_vec((0..20).map(|i| 2.0 * i as f64).collect());

        let fitted_small = LgbmRegressor::new()
            .with_n_estimators(20)
            .with_num_leaves(2)
            .with_min_child_samples(1)
            .fit(&x, &y)
            .unwrap();

        let fitted_large = LgbmRegressor::new()
            .with_n_estimators(20)
            .with_num_leaves(16)
            .with_min_child_samples(1)
            .fit(&x, &y)
            .unwrap();

        let preds_small = fitted_small.predict(&x).unwrap();
        let preds_large = fitted_large.predict(&x).unwrap();

        let err_small: f64 = preds_small
            .iter()
            .zip(y.iter())
            .map(|(&p, &t)| (p - t).powi(2))
            .sum();
        let err_large: f64 = preds_large
            .iter()
            .zip(y.iter())
            .map(|(&p, &t)| (p - t).powi(2))
            .sum();

        assert!(
            err_large <= err_small + 1e-6,
            "larger num_leaves should fit better: small={}, large={}",
            err_small,
            err_large
        );
    }

    #[test]
    fn test_lgbm_feature_importances() {
        let x = array![
            [1.0, 100.0],
            [2.0, 100.0],
            [3.0, 100.0],
            [10.0, 100.0],
            [11.0, 100.0],
            [12.0, 100.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let fitted = LgbmClassifier::new()
            .with_n_estimators(10)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .fit(&x, &y)
            .unwrap();

        let imp = fitted.feature_importances();
        // Only feature 0 is informative, so it should get most importance.
        assert!(imp[0] > imp[1]);
    }

    #[test]
    fn test_lgbm_shape_mismatch() {
        let x = array![[1.0], [2.0], [3.0]];
        let y = array![1.0, 2.0];
        assert!(LgbmRegressor::new().fit(&x, &y).is_err());
    }

    #[test]
    fn test_lgbm_empty_input() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<f64>::zeros(0);
        assert!(LgbmRegressor::new().fit(&x, &y).is_err());
    }

    #[test]
    fn test_lgbm_classifier_single_class() {
        let x = array![[1.0], [2.0], [3.0]];
        let y = array![0.0, 0.0, 0.0];
        assert!(LgbmClassifier::new().fit(&x, &y).is_err());
    }

    // --- New feature tests ---

    #[test]
    fn test_lgbm_regressor_sample_weight() {
        // Weight the high-x rows heavily so the model focuses there.
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
            [10.0]
        ];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];
        let sw = Array1::from_vec(vec![
            0.01, 0.01, 0.01, 0.01, 0.01, 10.0, 10.0, 10.0, 10.0, 10.0,
        ]);

        let opts = LgbmFitOptions {
            sample_weight: Some(sw.view()),
            ..Default::default()
        };

        let fitted = LgbmRegressor::new()
            .with_n_estimators(30)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .fit_with_eval(&x, &y, &opts)
            .unwrap();

        let preds = fitted.predict(&x).unwrap();
        // High-weight rows (indices 5..10) should be fit reasonably well.
        for i in 5..10 {
            assert!((preds[i] - y[i]).abs() < 5.0);
            assert!(preds[i].is_finite());
        }
    }

    #[test]
    fn test_lgbm_regressor_init_score() {
        // If we pass init_score close to the target, training converges in few rounds.
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
            [10.0]
        ];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];
        let init = Array1::from_vec(y.to_vec());

        let opts = LgbmFitOptions {
            init_score: Some(init.view()),
            ..Default::default()
        };

        let fitted = LgbmRegressor::new()
            .with_n_estimators(10)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .with_learning_rate(0.01)
            .fit_with_eval(&x, &y, &opts)
            .unwrap();

        let preds = fitted.predict(&x).unwrap();
        // Predictions won't include init_score for new data (baseline = mean init),
        // so we just check they are finite and the fit worked.
        for &p in preds.iter() {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn test_lgbm_regressor_early_stopping() {
        let x_train = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0],
            [10.0]
        ];
        let y_train = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];
        let x_eval = array![[11.0], [12.0], [13.0]];
        let y_eval = array![22.0, 24.0, 26.0];

        let opts = LgbmFitOptions {
            eval_set: Some((&x_eval, &y_eval)),
            ..Default::default()
        };

        let fitted = LgbmRegressor::new()
            .with_n_estimators(100)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .with_early_stopping(Some(5))
            .fit_with_eval(&x_train, &y_train, &opts)
            .unwrap();

        // best_iteration should be <= n_estimators
        assert!(fitted.best_iteration() <= 100);
        // Some predictions should be finite
        let preds = fitted.predict(&x_train).unwrap();
        for &p in preds.iter() {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn test_lgbm_classifier_class_weight_balanced() {
        // Imbalanced data: 80% class 0, 20% class 1.
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
            [10.0]
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];

        let fitted = LgbmClassifier::new()
            .with_n_estimators(20)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .with_class_weight(Some(LgbmClassWeight::Balanced))
            .fit(&x, &y)
            .unwrap();

        let preds = fitted.predict(&x).unwrap();
        // With class balancing, the model should at least predict the minority
        // class somewhere.
        let n_minority: usize = preds.iter().filter(|&&v| v == 1.0).count();
        assert!(
            n_minority >= 1,
            "class balancing should predict minority class"
        );
    }

    #[test]
    fn test_lgbm_regressor_monotone_constraint_increasing() {
        // y increases with x[0], decreases with x[1].
        let x = array![
            [1.0, 10.0],
            [2.0, 9.0],
            [3.0, 8.0],
            [4.0, 7.0],
            [5.0, 6.0],
            [6.0, 5.0],
            [7.0, 4.0],
            [8.0, 3.0],
            [9.0, 2.0],
            [10.0, 1.0],
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        // Constrain: feature 0 increasing, feature 1 decreasing
        let fitted = LgbmRegressor::new()
            .with_n_estimators(30)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .with_monotone_constraints(vec![1, -1])
            .fit(&x, &y)
            .unwrap();

        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(p.is_finite());
        }
    }

    /// Deterministic LCG for test data.
    fn lcg_uniform(state: &mut u64) -> f64 {
        *state = (1_664_525 * *state + 1_013_904_223) % (1 << 32);
        *state as f64 / (1u64 << 32) as f64
    }

    /// y = 3 x0 + 4 x1 (0.5 - x0): increasing in x0 on average but
    /// *decreasing* in x0 for large x1, so trees must fight the data to stay
    /// monotone. With only local child checks, deeper splits on x1 broke
    /// global monotonicity in x0 (#47).
    fn monotone_conflict_data(n: usize) -> (Array2<f64>, Array1<f64>) {
        let mut st = 7u64;
        let mut x = Array2::<f64>::zeros((n, 2));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let x0 = lcg_uniform(&mut st);
            let x1 = lcg_uniform(&mut st);
            let noise = lcg_uniform(&mut st) - 0.5;
            x[[i, 0]] = x0;
            x[[i, 1]] = x1;
            y[i] = 3.0 * x0 + 4.0 * x1 * (0.5 - x0) + 0.3 * noise;
        }
        (x, y)
    }

    /// Predictions along x0 for every x1 on a grid; returns the largest
    /// violation of the requested direction (0 = perfectly monotone).
    fn max_monotone_violation(predict: impl Fn(&Array2<f64>) -> Array1<f64>, sign: f64) -> f64 {
        let n0 = 60;
        let mut worst: f64 = 0.0;
        for k in 0..=12 {
            let x1 = -0.1 + k as f64 * 0.1;
            let grid = Array2::from_shape_fn((n0, 2), |(i, j)| {
                if j == 0 {
                    -0.1 + 1.2 * i as f64 / (n0 - 1) as f64
                } else {
                    x1
                }
            });
            let p = predict(&grid);
            for w in p.as_slice().unwrap().windows(2) {
                worst = worst.max(sign * (w[0] - w[1]));
            }
        }
        worst
    }

    #[test]
    fn test_lgbm_monotone_increasing_is_global() {
        let (x, y) = monotone_conflict_data(400);
        let fitted = LgbmRegressor::new()
            .with_n_estimators(60)
            .with_num_leaves(31)
            .with_min_child_samples(5)
            .with_monotone_constraints(vec![1, 0])
            .fit(&x, &y)
            .unwrap();
        let viol = max_monotone_violation(|g| fitted.predict(g).unwrap(), 1.0);
        assert!(viol <= 1e-12, "monotone(+1) violated by {viol}");
        // The unconstrained model does follow the non-monotone data.
        let free = LgbmRegressor::new()
            .with_n_estimators(60)
            .with_num_leaves(31)
            .with_min_child_samples(5)
            .fit(&x, &y)
            .unwrap();
        let viol_free = max_monotone_violation(|g| free.predict(g).unwrap(), 1.0);
        assert!(
            viol_free > 0.05,
            "test data should be non-monotone, {viol_free}"
        );
    }

    #[test]
    fn test_lgbm_monotone_decreasing_is_global_classifier() {
        let (x, y) = monotone_conflict_data(400);
        // Flip: decreasing in x0 on average, increasing for large x1.
        let labels = y.mapv(|v| if v < 1.0 { 1.0 } else { 0.0 });
        let fitted = LgbmClassifier::new()
            .with_n_estimators(60)
            .with_num_leaves(15)
            .with_min_child_samples(5)
            .with_monotone_constraints(vec![-1, 0])
            .fit(&x, &labels)
            .unwrap();
        let viol = max_monotone_violation(
            |g| fitted.predict_proba(g).unwrap().column(1).to_owned(),
            -1.0,
        );
        assert!(viol <= 1e-12, "monotone(-1) violated by {viol}");
    }

    #[test]
    fn test_child_bounds_split_at_midpoint() {
        assert_eq!(
            child_bounds(UNBOUNDED, 1, 1.0, 3.0),
            ((f64::NEG_INFINITY, 2.0), (2.0, f64::INFINITY))
        );
        assert_eq!(
            child_bounds((0.0, 5.0), -1, 3.0, 1.0),
            ((2.0, 5.0), (0.0, 2.0))
        );
        assert_eq!(
            child_bounds((0.0, 5.0), 0, 3.0, 1.0),
            ((0.0, 5.0), (0.0, 5.0))
        );
        // Given-output gain equals the closed form at the optimum.
        let (g, h, l1, l2) = (-3.0, 2.0, 0.5, 1.0);
        let w = leaf_value(g, h, l1, l2);
        assert!((leaf_gain_given_output(g, h, l1, l2, w) - leaf_gain(g, h, l1, l2)).abs() < 1e-12);
    }

    #[test]
    fn test_lgbm_max_bins_validated() {
        let x = Array2::from_shape_fn((300, 1), |(i, _)| i as f64);
        let y = Array1::from_shape_fn(300, |i| i as f64);
        for bad in [0usize, 1, 256, 1000] {
            let r = LgbmRegressor::new().with_max_bins(bad).fit(&x, &y);
            assert!(matches!(r, Err(RustMlError::InvalidParameter(_))), "{bad}");
            let c = LgbmClassifier::new()
                .with_max_bins(bad)
                .fit(&x, &y.mapv(|v| (v > 150.0) as u8 as f64));
            assert!(matches!(c, Err(RustMlError::InvalidParameter(_))), "{bad}");
        }
        assert!(LgbmRegressor::new().with_max_bins(255).fit(&x, &y).is_ok());
        // Numeric bins are clamped before the narrowing cast.
        let edges: Vec<f64> = (0..400).map(|v| v as f64).collect();
        assert_eq!(numeric_bin(1e9, &edges), MISSING_BIN - 1);
        assert_eq!(numeric_bin(0.5, &edges), 1);
    }

    #[test]
    fn test_lgbm_monotone_constraints_wrong_length() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 2.0];
        // 3 constraints for 2 features — should error
        let result = LgbmRegressor::new()
            .with_monotone_constraints(vec![1, 0, -1])
            .fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_lgbm_feature_importance_reflects_gain() {
        // Feature 0 drives prediction, feature 1 is noise.
        let x = array![
            [1.0, 50.0],
            [2.0, 50.0],
            [3.0, 50.0],
            [4.0, 50.0],
            [10.0, 50.0],
            [11.0, 50.0],
            [12.0, 50.0],
            [13.0, 50.0],
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0];

        let fitted = LgbmRegressor::new()
            .with_n_estimators(20)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .fit(&x, &y)
            .unwrap();

        let imp = fitted.feature_importances();
        assert!(
            imp[0] > imp[1],
            "feature 0 should have higher importance: {:?}",
            imp
        );
        let sum: f64 = imp.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10 || sum == 0.0);
    }

    #[test]
    fn test_lgbm_goss_unbiased_vs_gbdt() {
        // On well-separated data, GOSS and GBDT should both converge to high accuracy.
        let x = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0],
            [10.0, 1.0],
            [11.0, 1.0],
            [12.0, 1.0],
            [13.0, 1.0],
            [14.0, 1.0]
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let gbdt = LgbmClassifier::new()
            .with_n_estimators(30)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .with_boosting_type(BoostingType::Gbdt)
            .fit(&x, &y)
            .unwrap();

        let goss = LgbmClassifier::new()
            .with_n_estimators(30)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .with_boosting_type(BoostingType::Goss)
            .fit(&x, &y)
            .unwrap();

        let gbdt_correct: usize = gbdt
            .predict(&x)
            .unwrap()
            .iter()
            .zip(y.iter())
            .filter(|(&p, &t)| p == t)
            .count();
        let goss_correct: usize = goss
            .predict(&x)
            .unwrap()
            .iter()
            .zip(y.iter())
            .filter(|(&p, &t)| p == t)
            .count();

        // Both should classify essentially everything correctly.
        assert!(gbdt_correct >= 9);
        assert!(goss_correct >= 9);
    }

    // --- Edge case tests ---

    #[test]
    fn test_lgbm_constant_y() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![7.0, 7.0, 7.0, 7.0, 7.0];

        let fitted = LgbmRegressor::new()
            .with_n_estimators(20)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .fit(&x, &y)
            .unwrap();

        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert_abs_diff_eq!(p, 7.0, epsilon = 0.1);
        }
    }

    #[test]
    fn test_lgbm_num_leaves_one_is_baseline_only() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let fitted = LgbmRegressor::new()
            .with_n_estimators(30)
            .with_num_leaves(1)
            .with_min_child_samples(1)
            .fit(&x, &y)
            .unwrap();

        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(p.is_finite());
            assert_abs_diff_eq!(p, 6.0, epsilon = 1.0);
        }
    }

    #[test]
    fn test_lgbm_zero_learning_rate() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let fitted = LgbmRegressor::new()
            .with_n_estimators(10)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .with_learning_rate(0.0)
            .fit(&x, &y)
            .unwrap();

        let preds = fitted.predict(&x).unwrap();
        let mean: f64 = y.iter().sum::<f64>() / y.len() as f64;
        for &p in preds.iter() {
            assert_abs_diff_eq!(p, mean, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_lgbm_single_estimator() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let fitted = LgbmRegressor::new()
            .with_n_estimators(1)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .fit(&x, &y)
            .unwrap();

        assert_eq!(fitted.n_estimators(), 1);
        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn test_lgbm_extreme_l2_regularization() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let fitted = LgbmRegressor::new()
            .with_n_estimators(10)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .with_reg_lambda(1e6)
            .fit(&x, &y)
            .unwrap();

        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert_abs_diff_eq!(p, 3.0, epsilon = 0.5);
        }
    }

    #[test]
    fn test_lgbm_extreme_l1_regularization() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let fitted = LgbmRegressor::new()
            .with_n_estimators(10)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .with_reg_alpha(1e6)
            .fit(&x, &y)
            .unwrap();

        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert_abs_diff_eq!(p, 3.0, epsilon = 0.5);
        }
    }

    #[test]
    fn test_lgbm_all_nan_row_predict() {
        let x_train = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]];
        let y_train = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let fitted = LgbmRegressor::new()
            .with_n_estimators(10)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .fit(&x_train, &y_train)
            .unwrap();

        let x_test = array![[f64::NAN, f64::NAN]];
        let preds = fitted.predict(&x_test).unwrap();
        assert!(preds[0].is_finite());
    }

    #[test]
    fn test_lgbm_nan_in_predict_only() {
        let x_train = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y_train = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let fitted = LgbmRegressor::new()
            .with_n_estimators(10)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .fit(&x_train, &y_train)
            .unwrap();

        let x_test = array![[2.5], [f64::NAN], [4.0]];
        let preds = fitted.predict(&x_test).unwrap();
        for &p in preds.iter() {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn test_lgbm_infinity_in_features() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [f64::INFINITY]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 10.0];

        let fitted = LgbmRegressor::new()
            .with_n_estimators(10)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .fit(&x, &y)
            .unwrap();

        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(!p.is_nan());
        }
    }

    #[test]
    fn test_lgbm_duplicate_rows() {
        let x = array![
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
            [3.0, 4.0],
            [3.0, 4.0],
            [3.0, 4.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let fitted = LgbmClassifier::new()
            .with_n_estimators(10)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .fit(&x, &y)
            .unwrap();

        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(p == 0.0 || p == 1.0);
        }
    }

    #[test]
    fn test_lgbm_highly_imbalanced_with_balanced_weight() {
        let x = Array2::from_shape_vec((20, 1), (0..20).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_vec((0..20).map(|i| if i >= 19 { 1.0 } else { 0.0 }).collect());

        let fitted = LgbmClassifier::new()
            .with_n_estimators(20)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .with_class_weight(Some(LgbmClassWeight::Balanced))
            .fit(&x, &y)
            .unwrap();

        let preds = fitted.predict(&x).unwrap();
        let n_minority: usize = preds.iter().filter(|&&v| v == 1.0).count();
        assert!(n_minority >= 1);
    }

    #[test]
    fn test_lgbm_negative_class_labels() {
        let x = array![[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]];
        let y = array![-1.0, -1.0, -1.0, 5.5, 5.5, 5.5];

        let fitted = LgbmClassifier::new()
            .with_n_estimators(15)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .fit(&x, &y)
            .unwrap();

        assert_eq!(fitted.classes(), &[-1.0, 5.5]);
        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(p == -1.0 || p == 5.5);
        }
    }

    #[test]
    fn test_lgbm_zero_sample_weights() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let sw = Array1::from_vec(vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);

        let opts = LgbmFitOptions {
            sample_weight: Some(sw.view()),
            ..Default::default()
        };

        let fitted = LgbmRegressor::new()
            .with_n_estimators(20)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .fit_with_eval(&x, &y, &opts)
            .unwrap();

        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn test_lgbm_goss_extreme_rates() {
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
            [10.0]
        ];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

        let goss = |top: f64, other: f64| {
            LgbmRegressor::new()
                .with_n_estimators(20)
                .with_num_leaves(4)
                .with_min_child_samples(1)
                .with_boosting_type(BoostingType::Goss)
                .with_goss_rates(top, other)
                .fit(&x, &y)
        };
        // LightGBM's checks: positive rates with a sum of at most 1.
        assert!(goss(0.6, 0.6).is_err());
        assert!(goss(0.5, 0.0).is_err());
        let preds = goss(0.5, 0.5).unwrap().predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn test_lgbm_max_depth_one_stumps() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let fitted = LgbmRegressor::new()
            .with_n_estimators(10)
            .with_num_leaves(2)
            .with_max_depth(Some(1))
            .with_min_child_samples(1)
            .fit(&x, &y)
            .unwrap();

        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn test_lgbm_feature_never_used_zero_importance() {
        let x = array![
            [1.0, 42.0],
            [2.0, 42.0],
            [3.0, 42.0],
            [4.0, 42.0],
            [5.0, 42.0]
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let fitted = LgbmRegressor::new()
            .with_n_estimators(20)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .fit(&x, &y)
            .unwrap();

        let imp = fitted.feature_importances();
        assert_eq!(imp[1], 0.0);
    }

    #[test]
    fn test_lgbm_min_split_gain_too_high() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let fitted = LgbmRegressor::new()
            .with_n_estimators(10)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .with_min_split_gain(1e12)
            .fit(&x, &y)
            .unwrap();

        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert_abs_diff_eq!(p, 3.0, epsilon = 0.5);
        }
    }

    #[test]
    fn test_lgbm_predict_wrong_n_features() {
        let x_train = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y_train = array![1.0, 2.0, 3.0];

        let fitted = LgbmRegressor::new()
            .with_n_estimators(5)
            .with_min_child_samples(1)
            .fit(&x_train, &y_train)
            .unwrap();

        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_lgbm_monotone_on_constant_feature() {
        let x = array![
            [1.0, 50.0],
            [2.0, 50.0],
            [3.0, 50.0],
            [4.0, 50.0],
            [5.0, 50.0]
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let fitted = LgbmRegressor::new()
            .with_n_estimators(10)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .with_monotone_constraints(vec![1, -1])
            .fit(&x, &y)
            .unwrap();

        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn test_lgbm_many_classes() {
        let x = array![
            [0.0],
            [0.1],
            [0.2],
            [1.0],
            [1.1],
            [1.2],
            [2.0],
            [2.1],
            [2.2],
            [3.0],
            [3.1],
            [3.2],
            [4.0],
            [4.1],
            [4.2]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0];

        let fitted = LgbmClassifier::new()
            .with_n_estimators(20)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .fit(&x, &y)
            .unwrap();

        assert_eq!(fitted.classes().len(), 5);
        let proba = fitted.predict_proba(&x).unwrap();
        assert_eq!(proba.ncols(), 5);
        for i in 0..x.nrows() {
            let row_sum: f64 = (0..5).map(|c| proba[[i, c]]).sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_lgbm_all_nan_column() {
        let x = array![
            [1.0, f64::NAN],
            [2.0, f64::NAN],
            [3.0, f64::NAN],
            [4.0, f64::NAN],
            [5.0, f64::NAN]
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let fitted = LgbmRegressor::new()
            .with_n_estimators(10)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .fit(&x, &y)
            .unwrap();

        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn test_lgbm_num_leaves_exceeds_n() {
        let x = array![[1.0], [2.0], [3.0]];
        let y = array![1.0, 2.0, 3.0];

        let fitted = LgbmRegressor::new()
            .with_n_estimators(5)
            .with_num_leaves(32)
            .with_min_child_samples(1)
            .fit(&x, &y)
            .unwrap();

        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn test_lgbm_predict_empty_matrix() {
        let x_train = array![[1.0], [2.0], [3.0]];
        let y_train = array![1.0, 2.0, 3.0];

        let fitted = LgbmRegressor::new()
            .with_n_estimators(5)
            .with_num_leaves(2)
            .with_min_child_samples(1)
            .fit(&x_train, &y_train)
            .unwrap();

        let x_empty = Array2::<f64>::zeros((0, 1));
        let preds = fitted.predict(&x_empty).unwrap();
        assert_eq!(preds.len(), 0);
    }

    #[test]
    fn test_lgbm_classifier_categorical_multiclass() {
        let x = array![
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [0.0, 2.0],
            [1.0, 2.0],
            [2.0, 2.0],
            [0.0, 3.0],
            [1.0, 3.0],
            [2.0, 3.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];

        let fitted = LgbmClassifier::new()
            .with_n_estimators(20)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .with_categorical_features(vec![1])
            .fit(&x, &y)
            .unwrap();

        assert_eq!(fitted.classes().len(), 3);
        let preds = fitted.predict(&x).unwrap();
        let correct: usize = preds.iter().zip(y.iter()).filter(|(&p, &t)| p == t).count();
        assert!(correct >= 7, "got {}/9", correct);
    }

    #[test]
    fn test_lgbm_sample_weight_wrong_length_errors() {
        let x = array![[1.0], [2.0], [3.0]];
        let y = array![1.0, 2.0, 3.0];
        let sw = Array1::from_vec(vec![1.0, 1.0]);

        let opts = LgbmFitOptions {
            sample_weight: Some(sw.view()),
            ..Default::default()
        };

        assert!(LgbmRegressor::new().fit_with_eval(&x, &y, &opts).is_err());
    }

    #[test]
    fn test_lgbm_init_score_wrong_length_errors() {
        let x = array![[1.0], [2.0], [3.0]];
        let y = array![1.0, 2.0, 3.0];
        let init = Array1::from_vec(vec![0.0, 0.0]);

        let opts = LgbmFitOptions {
            init_score: Some(init.view()),
            ..Default::default()
        };

        assert!(LgbmRegressor::new().fit_with_eval(&x, &y, &opts).is_err());
    }

    #[test]
    fn test_lgbm_stress_larger_dataset() {
        // 200 samples, 5 features, noisy linear target.
        let n = 200;
        let p = 5;
        let mut x_data = Vec::with_capacity(n * p);
        let mut y_data = Vec::with_capacity(n);
        let mut rng_state = 42u64;
        let mut next = || {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng_state >> 33) as f64) / (u32::MAX as f64) - 0.5
        };
        for i in 0..n {
            let f0 = next();
            let f1 = next();
            let f2 = next();
            let f3 = next();
            let f4 = next();
            x_data.extend([f0, f1, f2, f3, f4]);
            y_data.push(2.0 * f0 - 1.5 * f1 + 0.3 * f2 + 0.1 * next());
            let _ = i;
        }
        let x = Array2::from_shape_vec((n, p), x_data).unwrap();
        let y = Array1::from_vec(y_data);

        let fitted = LgbmRegressor::new()
            .with_n_estimators(100)
            .with_num_leaves(15)
            .with_learning_rate(0.05)
            .with_min_child_samples(5)
            .fit(&x, &y)
            .unwrap();

        let preds = fitted.predict(&x).unwrap();
        // R² should be reasonably high
        let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
        let ss_res: f64 = preds
            .iter()
            .zip(y.iter())
            .map(|(&p, &t)| (p - t).powi(2))
            .sum();
        let ss_tot: f64 = y.iter().map(|&t| (t - y_mean).powi(2)).sum();
        let r2 = 1.0 - ss_res / ss_tot;
        assert!(
            r2 > 0.5,
            "R² should be > 0.5 on noisy linear data, got {:.4}",
            r2
        );

        // Feature importance should reflect the true signal: f0 (coef 2.0) > f1 (coef 1.5) > others
        let imp = fitted.feature_importances();
        let sum: f64 = imp.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
        assert!(imp[0] > 0.0);
    }

    #[test]
    fn test_lgbm_stress_many_estimators() {
        // 500 estimators on small data — shouldn't overflow or NaN.
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
            [10.0]
        ];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

        let fitted = LgbmRegressor::new()
            .with_n_estimators(500)
            .with_num_leaves(4)
            .with_learning_rate(0.01)
            .with_min_child_samples(1)
            .fit(&x, &y)
            .unwrap();

        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(p.is_finite());
            assert!(!p.is_nan());
        }
        assert_eq!(fitted.n_estimators(), 500);
    }

    #[test]
    fn test_lgbm_reproducible_with_seed() {
        let x = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
            [13.0, 14.0],
            [15.0, 16.0]
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let m = LgbmRegressor::new()
            .with_n_estimators(20)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .with_subsample(0.5)
            .with_subsample_freq(1)
            .with_colsample_bytree(0.5)
            .with_seed(42);

        let f1 = m.fit(&x, &y).unwrap();
        let f2 = m.fit(&x, &y).unwrap();

        let p1 = f1.predict(&x).unwrap();
        let p2 = f2.predict(&x).unwrap();

        for (a, b) in p1.iter().zip(p2.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_lgbm_different_seeds_differ() {
        let x = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
            [13.0, 14.0],
            [15.0, 16.0]
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let m1 = LgbmRegressor::new()
            .with_n_estimators(20)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .with_subsample(0.5)
            .with_subsample_freq(1)
            .with_seed(1);
        let m2 = LgbmRegressor::new()
            .with_n_estimators(20)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .with_subsample(0.5)
            .with_subsample_freq(1)
            .with_seed(2);

        let f1 = m1.fit(&x, &y).unwrap();
        let f2 = m2.fit(&x, &y).unwrap();

        let p1 = f1.predict(&x).unwrap();
        let p2 = f2.predict(&x).unwrap();

        // At least one prediction should differ with different seeds
        let any_diff = p1.iter().zip(p2.iter()).any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(
            any_diff,
            "different seeds should give different predictions"
        );
    }

    #[test]
    fn test_lgbm_clone_preserves_predictions() {
        // Fitted models are Clone; cloned versions should behave identically.
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let fitted = LgbmRegressor::new()
            .with_n_estimators(10)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .fit(&x, &y)
            .unwrap();

        let cloned = fitted.clone();
        let orig = fitted.predict(&x).unwrap();
        let clon = cloned.predict(&x).unwrap();
        for (a, b) in orig.iter().zip(clon.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_lgbm_early_stopping_respects_rounds() {
        let x_train = Array2::from_shape_vec((20, 1), (0..20).map(|i| i as f64).collect()).unwrap();
        let y_train = Array1::from_vec((0..20).map(|i| 2.0 * i as f64).collect());
        let x_eval = array![[5.0], [10.0], [15.0]];
        let y_eval = array![10.0, 20.0, 30.0];

        let opts = LgbmFitOptions {
            eval_set: Some((&x_eval, &y_eval)),
            ..Default::default()
        };

        let fitted = LgbmRegressor::new()
            .with_n_estimators(500)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .with_early_stopping(Some(3))
            .fit_with_eval(&x_train, &y_train, &opts)
            .unwrap();

        assert!(fitted.best_iteration() > 0);
        assert!(fitted.best_iteration() <= 500);
    }

    #[test]
    fn test_categorical_edges_keep_most_frequent() {
        // 300 categories: 0..=99 appear 5 times, 100..=299 three times,
        // and 1000 once. Only 254 value bins exist: all frequent categories
        // and the 154 smallest 3-count categories keep a bin; the rest share
        // the "other" bin instead of collapsing into the last category.
        let mut col = Vec::new();
        for c in 0..300 {
            col.extend(std::iter::repeat(c as f64).take(if c < 100 { 5 } else { 3 }));
        }
        col.push(1000.0);
        let col = Array1::from_vec(col);
        let edges = categorical_edges(col.view(), DEFAULT_MAX_BINS, 3);
        assert_eq!(edges.len(), 254);
        assert_eq!(edges.first(), Some(&0.0));
        assert_eq!(edges.last(), Some(&253.0));
        let bins = FeatureBins {
            edges,
            is_categorical: true,
        };
        assert_eq!(bin_value(42.0, &bins), 42);
        assert_eq!(bin_value(299.0, &bins), MISSING_BIN);
        assert_eq!(bin_value(1000.0, &bins), MISSING_BIN);

        // Rarer than min_data_in_bin (after the two most frequent): "other".
        let col = Array1::from_vec(vec![1.0, 1.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(
            categorical_edges(col.view(), DEFAULT_MAX_BINS, 3),
            vec![1.0, 2.0]
        );
    }

    #[test]
    fn test_lgbm_regressor_categorical_features() {
        // x[:, 0] is numeric, x[:, 1] is categorical
        let x = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 1.0],
            [4.0, 1.0],
            [5.0, 2.0],
            [6.0, 2.0],
            [7.0, 0.0],
            [8.0, 1.0]
        ];
        let y = array![1.0, 2.0, 10.0, 11.0, 20.0, 21.0, 3.0, 12.0];

        let fitted = LgbmRegressor::new()
            .with_n_estimators(20)
            .with_num_leaves(8)
            .with_min_child_samples(1)
            .with_categorical_features(vec![1])
            .fit(&x, &y)
            .unwrap();

        let preds = fitted.predict(&x).unwrap();
        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(*p, *t, epsilon = 5.0);
        }
    }
}

#[cfg(test)]
mod early_stopping_regression_tests {
    use super::*;
    use ndarray::ShapeBuilder;

    fn lcg(s: &mut u64) -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*s >> 11) as f64 / (1u64 << 53) as f64
    }

    fn reg_data(n: usize, seed: u64, noise: f64) -> (Array2<f64>, Array1<f64>) {
        let mut s = seed;
        let x = Array2::from_shape_fn((n, 3), |_| lcg(&mut s) * 4.0 - 2.0);
        let y = Array1::from_iter(
            x.rows()
                .into_iter()
                .map(|r| (2.0 * r[0]).sin() + r[1] * r[1] + noise * (lcg(&mut s) - 0.5)),
        );
        (x, y)
    }

    fn clf_data(n: usize, seed: u64, n_classes: usize, flip: f64) -> (Array2<f64>, Array1<f64>) {
        let mut s = seed;
        let x = Array2::from_shape_fn((n, 3), |_| lcg(&mut s) * 4.0 - 2.0);
        let y = Array1::from_iter(x.rows().into_iter().map(|r| {
            let c = if n_classes == 2 {
                usize::from(r[0] + 0.5 * r[1] > 0.0)
            } else {
                (((r[0] + 2.0) / 4.0 * n_classes as f64) as usize).min(n_classes - 1)
            };
            let c = if lcg(&mut s) < flip {
                (c + 1) % n_classes
            } else {
                c
            };
            10.0 + c as f64
        }));
        (x, y)
    }

    fn log_loss(fitted: &FittedLgbmClassifier, x: &Array2<f64>, y: &Array1<f64>) -> f64 {
        let proba = fitted.predict_proba(x).unwrap();
        let classes = fitted.classes();
        let mut total = 0.0;
        for (i, &t) in y.iter().enumerate() {
            let c = classes.iter().position(|&c| c == t).unwrap();
            total -= proba[[i, c]].max(1e-15).ln();
        }
        total / y.len() as f64
    }

    /// #44: eval rows must be binned with the TRAINING edges.
    #[test]
    fn eval_rows_use_training_bins() {
        let (x, _) = reg_data(200, 1, 0.0);
        let (binned, bins) = compute_bins(
            x.view(),
            DEFAULT_MAX_BINS,
            &[],
            3,
            BinningMethod::LightGbm,
            0,
        );
        // A subset (e.g. an eval split) has a different distribution; with
        // its own edges its bins would not match the training bins.
        let rows: Vec<usize> = (0..200).filter(|&i| x[[i, 0]] > 0.5).collect();
        let sub = x.select(ndarray::Axis(0), &rows);
        let eb = bin_matrix(sub.view(), &bins);
        for (k, &i) in rows.iter().enumerate() {
            assert_eq!(eb.row(k), binned.row(i));
        }
        let (own, _) = compute_bins(
            sub.view(),
            DEFAULT_MAX_BINS,
            &[],
            3,
            BinningMethod::LightGbm,
            0,
        );
        assert_ne!(own, eb, "sanity: own-edge binning differs");
    }

    /// #44: eval == train -> training loss keeps decreasing, so the best
    /// iteration is the last one.
    #[test]
    fn regressor_eval_equals_train_best_iteration_is_last() {
        let (x, y) = reg_data(300, 2, 0.1);
        let opts = LgbmFitOptions {
            eval_set: Some((&x, &y)),
            ..Default::default()
        };
        let fitted = LgbmRegressor::new()
            .with_n_estimators(40)
            .with_num_leaves(8)
            .with_min_child_samples(5)
            .with_early_stopping(Some(5))
            .fit_with_eval(&x, &y, &opts)
            .unwrap();
        assert_eq!(fitted.best_iteration(), 40);
        assert_eq!(fitted.n_estimators(), 40);
    }

    /// #44: after early stopping, the model is truncated to best_iteration.
    #[test]
    fn regressor_truncates_to_best_iteration() {
        let (x, y) = reg_data(150, 3, 3.0);
        let (xe, ye) = reg_data(150, 4, 3.0);
        let opts = LgbmFitOptions {
            eval_set: Some((&xe, &ye)),
            ..Default::default()
        };
        let fitted = LgbmRegressor::new()
            .with_n_estimators(300)
            .with_num_leaves(31)
            .with_min_child_samples(2)
            .with_learning_rate(0.3)
            .with_early_stopping(Some(5))
            .fit_with_eval(&x, &y, &opts)
            .unwrap();
        assert!(
            fitted.best_iteration() < 300,
            "noise should trigger early stopping"
        );
        assert_eq!(fitted.n_estimators(), fitted.best_iteration());
    }

    #[test]
    fn regressor_eval_shape_is_validated() {
        let (x, y) = reg_data(50, 5, 0.1);
        let xe = Array2::<f64>::zeros((10, 2));
        let ye = Array1::<f64>::zeros(10);
        let opts = LgbmFitOptions {
            eval_set: Some((&xe, &ye)),
            ..Default::default()
        };
        assert!(LgbmRegressor::new().fit_with_eval(&x, &y, &opts).is_err());
    }

    /// #44: the classifier used to ignore eval_set / early_stopping_rounds.
    #[test]
    fn classifier_binary_eval_equals_train() {
        let (x, y) = clf_data(300, 6, 2, 0.05);
        let opts = LgbmFitOptions {
            eval_set: Some((&x, &y)),
            ..Default::default()
        };
        let fitted = LgbmClassifier::new()
            .with_n_estimators(30)
            .with_num_leaves(8)
            .with_min_child_samples(5)
            .with_early_stopping(Some(3))
            .fit_with_eval(&x, &y, &opts)
            .unwrap();
        assert_eq!(fitted.best_iteration(), 30);
        assert_eq!(fitted.n_estimators(), 30);

        // Eval (= train) log-loss decreases with more rounds.
        let mut prev = f64::INFINITY;
        for n in [5, 10, 20, 30] {
            let f = LgbmClassifier::new()
                .with_n_estimators(n)
                .with_num_leaves(8)
                .with_min_child_samples(5)
                .fit(&x, &y)
                .unwrap();
            let l = log_loss(&f, &x, &y);
            assert!(
                l < prev,
                "loss {l} did not decrease (prev {prev}) at {n} rounds"
            );
            prev = l;
        }
    }

    #[test]
    fn classifier_binary_early_stops_and_truncates() {
        let (x, y) = clf_data(150, 7, 2, 0.3);
        let (xe, ye) = clf_data(150, 8, 2, 0.3);
        let opts = LgbmFitOptions {
            eval_set: Some((&xe, &ye)),
            ..Default::default()
        };
        let fitted = LgbmClassifier::new()
            .with_n_estimators(300)
            .with_min_child_samples(2)
            .with_learning_rate(0.3)
            .with_early_stopping(Some(5))
            .fit_with_eval(&x, &y, &opts)
            .unwrap();
        assert!(fitted.best_iteration() < 300);
        assert_eq!(fitted.n_estimators(), fitted.best_iteration());
        let preds = fitted.predict(&xe).unwrap();
        assert!(preds.iter().all(|&p| p == 10.0 || p == 11.0));
    }

    #[test]
    fn classifier_multiclass_early_stops_and_truncates() {
        let (x, y) = clf_data(240, 9, 3, 0.3);
        let (xe, ye) = clf_data(240, 10, 3, 0.3);
        let opts = LgbmFitOptions {
            eval_set: Some((&xe, &ye)),
            ..Default::default()
        };
        let fitted = LgbmClassifier::new()
            .with_n_estimators(300)
            .with_min_child_samples(2)
            .with_learning_rate(0.3)
            .with_early_stopping(Some(5))
            .fit_with_eval(&x, &y, &opts)
            .unwrap();
        assert!(fitted.best_iteration() < 300);
        assert_eq!(fitted.n_estimators(), fitted.best_iteration());

        // Unknown eval label -> clear error.
        let ye_bad = ye.mapv(|v| if v == 12.0 { 99.0 } else { v });
        let opts_bad = LgbmFitOptions {
            eval_set: Some((&xe, &ye_bad)),
            ..Default::default()
        };
        assert!(LgbmClassifier::new()
            .with_n_estimators(5)
            .fit_with_eval(&x, &y, &opts_bad)
            .is_err());
    }

    #[test]
    fn classifier_validates_sample_weight_length() {
        let (x, y) = clf_data(40, 11, 2, 0.0);
        let sw = Array1::<f64>::ones(39);
        let opts = LgbmFitOptions {
            sample_weight: Some(sw.view()),
            ..Default::default()
        };
        match LgbmClassifier::new().fit_with_eval(&x, &y, &opts) {
            Err(RustMlError::ShapeMismatch(_)) => {}
            other => panic!("expected ShapeMismatch, got {:?}", other.map(|_| ())),
        }
    }

    #[test]
    fn fortran_order_input_works() {
        let (x, y) = reg_data(100, 12, 0.1);
        let mut xf = Array2::<f64>::zeros(x.raw_dim().f());
        xf.assign(&x);
        let m = LgbmRegressor::new()
            .with_n_estimators(10)
            .with_min_child_samples(5);
        let a = m.fit(&x, &y).unwrap();
        let b = m.fit(&xf, &y).unwrap();
        assert_eq!(a.predict(&x).unwrap(), b.predict(&xf).unwrap());
    }
}

#[cfg(test)]
mod bundling_tests {
    use super::*;

    fn lcg(s: &mut u64) -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*s >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Two dense columns, a one-hot block of 8, a categorical column that is
    /// mostly 0, and 4 sparse columns, one with missing values.
    fn sparse_data(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
        let mut s = seed;
        let mut x = Array2::<f64>::zeros((n, 15));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            x[[i, 0]] = lcg(&mut s) * 4.0 - 2.0;
            x[[i, 1]] = lcg(&mut s);
            let level = (lcg(&mut s) * 8.0) as usize;
            x[[i, 2 + level]] = 1.0;
            if lcg(&mut s) < 0.15 {
                x[[i, 10]] = 1.0 + (lcg(&mut s) * 4.0).floor();
            }
            for j in 11..15 {
                if lcg(&mut s) < 0.1 {
                    x[[i, j]] = lcg(&mut s) * 2.0 - 1.0;
                }
            }
            if lcg(&mut s) < 0.03 {
                x[[i, 14]] = f64::NAN;
            }
            y[i] =
                x[[i, 0]] + 0.3 * level as f64 - x[[i, 10]] + 3.0 * x[[i, 12]] + 0.2 * lcg(&mut s);
        }
        (x, y)
    }

    #[test]
    fn exclusive_columns_share_a_bundle() {
        let (x, _) = sparse_data(300, 1);
        let (binned, _) = compute_bins(x.view(), 255, &[10], 3, BinningMethod::LightGbm, 0);
        let b = FeatureBundles::find(&binned).expect("bundles");
        assert!(b.bundle_of[0].is_none() && b.bundle_of[1].is_none());
        // The one-hot block never overlaps, so it fits into one bundle.
        let (first, _) = b.bundle_of[2].expect("one-hot column bundled");
        assert!((3..10).all(|f| b.bundle_of[f].map(|(id, _)| id) == Some(first)));
        // Every bundle is exactly exclusive: a row is off the most frequent
        // bin in at most one member.
        for id in 0..b.members.len() {
            for i in 0..x.nrows() {
                let off = (0..15)
                    .filter_map(|f| match b.bundle_of[f] {
                        Some((g, k)) if g == id => {
                            Some(binned[[i, f]] != b.members[g][k].default_bin)
                        }
                        _ => None,
                    })
                    .filter(|&o| o)
                    .count();
                assert!(off <= 1);
                assert_eq!(b.columns[id][i] == 0, off == 0);
            }
        }
    }

    #[test]
    fn bundled_histograms_match_plain_ones() {
        let (x, _) = sparse_data(300, 2);
        let (binned, bins) = compute_bins(x.view(), 255, &[10], 3, BinningMethod::LightGbm, 0);
        let bundles = FeatureBundles::find(&binned).unwrap();
        let mut s = 9;
        let mut grad: Vec<f64> = (0..300).map(|_| lcg(&mut s) - 0.5).collect();
        let mut hess: Vec<f64> = (0..300).map(|_| 0.5 + lcg(&mut s)).collect();
        round_to_f32(&mut grad);
        round_to_f32(&mut hess);
        let rows: Vec<usize> = (0..300).filter(|i| i % 3 != 0).collect();
        let feats: Vec<usize> = (0..15).collect();
        let totals = node_totals(&grad, &hess, &rows);
        let plain = build_histograms(
            &binned,
            &bins,
            None,
            true,
            &grad,
            &hess,
            &rows,
            totals,
            &feats,
            &[],
        );
        let bundled = build_histograms(
            &binned,
            &bins,
            Some(&bundles),
            true,
            &grad,
            &hess,
            &rows,
            totals,
            &feats,
            &[],
        );
        for (a, b) in plain.iter().zip(&bundled) {
            assert_eq!(a.count, b.count);
            assert_eq!(a.missing_count, b.missing_count);
            for k in 0..a.grad_sum.len() {
                assert!((a.grad_sum[k] - b.grad_sum[k]).abs() < 1e-12);
                assert!((a.hess_sum[k] - b.hess_sum[k]).abs() < 1e-12);
            }
            assert!((a.missing_grad - b.missing_grad).abs() < 1e-12);
            assert!((a.missing_hess - b.missing_hess).abs() < 1e-12);
        }
    }

    #[test]
    fn bundling_leaves_the_model_unchanged() {
        let (x, y) = sparse_data(400, 3);
        let reg = LgbmRegressor::new()
            .with_n_estimators(30)
            .with_num_leaves(12)
            .with_min_child_samples(5)
            .with_categorical_features(vec![10]);
        let a = reg.clone().fit(&x, &y).unwrap().predict(&x).unwrap();
        let b = reg
            .with_enable_bundle(false)
            .fit(&x, &y)
            .unwrap()
            .predict(&x)
            .unwrap();
        let worst = a
            .iter()
            .zip(&b)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0, f64::max);
        assert!(worst < 1e-10, "{worst}");

        let labels = y.mapv(|v| f64::from(v > 0.5) + f64::from(v > 1.5));
        let clf = LgbmClassifier::new()
            .with_n_estimators(20)
            .with_min_child_samples(5);
        let a = clf
            .clone()
            .fit(&x, &labels)
            .unwrap()
            .predict_proba(&x)
            .unwrap();
        let b = clf
            .with_enable_bundle(false)
            .fit(&x, &labels)
            .unwrap()
            .predict_proba(&x)
            .unwrap();
        let worst = a
            .iter()
            .zip(&b)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0, f64::max);
        assert!(worst < 1e-10, "{worst}");
    }

    #[test]
    fn warm_start_equals_a_single_fit() {
        let (x, y) = sparse_data(500, 11);
        let opts = LgbmFitOptions::default();
        for objective in [LgbmObjective::Regression, LgbmObjective::RegressionL1] {
            let reg = |n| {
                LgbmRegressor::new()
                    .with_n_estimators(n)
                    .with_min_child_samples(5)
                    .with_subsample(0.7)
                    .with_subsample_freq(1)
                    .with_colsample_bytree(0.8)
                    .with_objective(objective)
                    .with_seed(4)
            };
            let part = reg(5).fit(&x, &y).unwrap();
            let warm = reg(12).fit_warm_start(&part, &x, &y, &opts).unwrap();
            let once = reg(12).fit(&x, &y).unwrap();
            assert_eq!(warm.n_estimators(), 12);
            assert_eq!(warm.predict(&x).unwrap(), once.predict(&x).unwrap());
            assert_eq!(warm.feature_importances(), once.feature_importances());
            // As many trees as fitted: unchanged; fewer: an error.
            let same = reg(5).fit_warm_start(&part, &x, &y, &opts).unwrap();
            assert_eq!(same.predict(&x).unwrap(), part.predict(&x).unwrap());
            assert!(reg(4).fit_warm_start(&part, &x, &y, &opts).is_err());
            assert!(reg(9)
                .with_learning_rate(0.3)
                .fit_warm_start(&part, &x, &y, &opts)
                .is_err());
            assert!(reg(9)
                .with_boosting_type(BoostingType::Goss)
                .fit_warm_start(&part, &x, &y, &opts)
                .is_err());
        }
        for n_classes in [2.0, 3.0] {
            let labels = y.mapv(|v| (v.abs() * 2.0).floor().min(n_classes - 1.0));
            let clf = |n| {
                LgbmClassifier::new()
                    .with_n_estimators(n)
                    .with_min_child_samples(5)
                    .with_subsample(0.7)
                    .with_subsample_freq(1)
                    .with_colsample_bytree(0.8)
                    .with_seed(6)
            };
            let part = clf(4).fit(&x, &labels).unwrap();
            let warm = clf(9).fit_warm_start(&part, &x, &labels, &opts).unwrap();
            let once = clf(9).fit(&x, &labels).unwrap();
            assert_eq!(warm.n_estimators(), 9);
            assert_eq!(
                warm.predict_proba(&x).unwrap(),
                once.predict_proba(&x).unwrap()
            );
            let other = labels.mapv(|v| v + 1.0);
            assert!(clf(9).fit_warm_start(&part, &x, &other, &opts).is_err());
        }
    }

    #[test]
    fn thread_count_leaves_the_model_unchanged() {
        // Large enough for the parallel histograms, split search, row
        // partitions and score updates.
        let mut s = 7;
        let x = Array2::from_shape_fn((20_000, 20), |_| lcg(&mut s) * 2.0 - 1.0);
        let y = Array1::from_iter(x.rows().into_iter().map(|r| r[0] + r[1] * r[2]));
        let labels = y.mapv(|v| f64::from(v > 0.0) + f64::from(v > 0.6));
        let fit = |jobs| {
            let reg = LgbmRegressor::new().with_n_estimators(15).with_n_jobs(jobs);
            let clf = LgbmClassifier::new()
                .with_n_estimators(10)
                .with_n_jobs(jobs);
            (
                reg.fit(&x, &y).unwrap().predict(&x).unwrap(),
                clf.fit(&x, &labels).unwrap().predict_proba(&x).unwrap(),
            )
        };
        let one = fit(Some(1));
        assert_eq!(one, fit(Some(5)));
        assert_eq!(one, fit(None));
    }

    #[test]
    fn dense_or_single_columns_are_not_bundled() {
        let mut s = 4;
        let x = Array2::from_shape_fn((200, 4), |_| lcg(&mut s));
        let (binned, _) = compute_bins(x.view(), 255, &[], 3, BinningMethod::LightGbm, 0);
        assert!(FeatureBundles::find(&binned).is_none());
        let (binned, _) = compute_bins(
            x.column(0).insert_axis(ndarray::Axis(1)),
            255,
            &[],
            3,
            BinningMethod::LightGbm,
            0,
        );
        assert!(FeatureBundles::find(&binned).is_none());
    }
}
