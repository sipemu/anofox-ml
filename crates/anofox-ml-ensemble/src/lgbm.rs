//! LightGBM-like gradient boosting booster.
//!
//! A pure-Rust implementation of the key LightGBM algorithms:
//!
//! - **Leaf-wise tree growth** with `num_leaves` (best-first expansion via priority queue)
//! - **NaN handling**: learns the best missing direction per split
//! - **GOSS** (Gradient-based One-Side Sampling)
//! - **L1 + L2 regularization**
//! - **Row and column subsampling** (`subsample`, `colsample_bytree`)
//! - **Early stopping** with an evaluation set
//! - **Categorical features** (Fisher method: sort by grad/hess, then split)
//! - **Custom objectives** (user-provided gradient/hessian)
//! - **Feature importance** (split count and total gain)
//!
//! Not included: EFB, DART, distributed training.

use anofox_ml_core::{Fit, FitWeighted, Predict, Result, RustMlError};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, AsArray, Ix1, Ix2, ShapeBuilder};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

const DEFAULT_MAX_BINS: usize = 255;
/// Special bin index reserved for NaN / missing values.
const MISSING_BIN: u8 = 255;

// ============================================================
// Binning
// ============================================================

/// How numeric features are binned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum BinningMethod {
    /// LightGBM's greedy binning with `min_data_in_bin`.
    #[default]
    LightGbm,
    /// sklearn's `_BinMapper`: midpoints between consecutive distinct values
    /// when there are at most `max_bins` of them, otherwise the
    /// `method="midpoint"` percentiles at `max_bins` equal steps (from a
    /// subsample of 200 000 rows on larger data).
    Sklearn,
}

/// sklearn's `_find_binning_thresholds` for one column (non-NaN values,
/// sorted).
fn sklearn_thresholds(sorted: &[f64], max_bins: usize) -> Vec<f64> {
    let mut distinct = sorted.to_vec();
    distinct.dedup();
    if distinct.len() <= max_bins {
        return distinct.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect();
    }
    // np.percentile(col, linspace(0, 100, max_bins + 1)[1:-1], method="midpoint")
    let n = sorted.len();
    (1..max_bins)
        .map(|k| {
            let q = k as f64 / max_bins as f64;
            let idx = q * (n - 1) as f64;
            let below = idx.floor();
            let lo = sorted[below as usize];
            let v = if idx == below {
                lo
            } else {
                0.5 * (lo + sorted[(below as usize + 1).min(n - 1)])
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
    // Column-major: histograms read one feature at a time.
    let mut binned = Array2::zeros((n, p).f());
    let mut all_bins = Vec::with_capacity(p);

    let cat_set: std::collections::HashSet<usize> = categorical_features.iter().copied().collect();

    for j in 0..p {
        let is_categorical = cat_set.contains(&j);

        let mut edges = Vec::new();

        if is_categorical {
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
            // Greedy binning matching LightGBM: collect distinct values with
            // sample counts, then greedily fill bins so each has at least
            // `min_data_in_bin` samples.  Bin boundaries are placed at the
            // midpoint between the last value of one group and the first of
            // the next.
            let mut raw: Vec<f64> = (0..n).map(|i| x[[i, j]]).filter(|v| !v.is_nan()).collect();
            raw.sort_by(f64::total_cmp);

            // Distinct values with counts.
            let mut distinct: Vec<(f64, usize)> = Vec::new();
            if !raw.is_empty() {
                let mut cur = raw[0];
                let mut cnt = 1usize;
                for &v in &raw[1..] {
                    if v == cur {
                        cnt += 1;
                    } else {
                        distinct.push((cur, cnt));
                        cur = v;
                        cnt = 1;
                    }
                }
                distinct.push((cur, cnt));
            }

            if distinct.len() > 1 {
                let max_n_bins = max_bins - 1; // reserve one for missing
                let min_per = min_data_in_bin.max(1);

                // group_ends[k] = exclusive end-index into `distinct` for group k.
                let mut group_ends: Vec<usize> = Vec::new();
                let mut cur_count = 0usize;
                for (idx, &(_, count)) in distinct.iter().enumerate() {
                    cur_count += count;
                    if cur_count >= min_per {
                        group_ends.push(idx + 1);
                        cur_count = 0;
                        if group_ends.len() >= max_n_bins {
                            break;
                        }
                    }
                }
                // Remaining values go into the last group.
                if cur_count > 0 || group_ends.is_empty() {
                    group_ends.push(distinct.len());
                } else if let Some(last) = group_ends.last_mut() {
                    // Make sure the last group extends to the end.
                    *last = distinct.len();
                }

                // Edges are midpoints between consecutive groups.
                for k in 0..group_ends.len() - 1 {
                    let last_in_k = distinct[group_ends[k] - 1].0;
                    let first_in_next = distinct[group_ends[k]].0;
                    edges.push(0.5 * (last_in_k + first_in_next));
                }
            }
            // If 0 or 1 distinct value → no edges, everything goes to bin 0.
        }

        for i in 0..n {
            let v = x[[i, j]];
            if v.is_nan() {
                binned[[i, j]] = MISSING_BIN;
            } else if is_categorical {
                // Categories without a bin share the "other" (missing) bin.
                binned[[i, j]] = categorical_bin(v, &edges).map_or(MISSING_BIN, |b| b as u8);
            } else {
                binned[[i, j]] = numeric_bin(v, &edges);
            }
        }

        all_bins.push(FeatureBins {
            edges,
            is_categorical,
        });
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
/// (prediction walks one row at a time). Works for any layout of `x`.
fn bin_matrix(x: ArrayView2<f64>, all_bins: &[FeatureBins]) -> Array2<u8> {
    let mut out = Array2::<u8>::zeros((x.nrows(), x.ncols()));
    for (j, bins) in all_bins.iter().enumerate() {
        let col = x.column(j);
        for (o, &v) in out.column_mut(j).iter_mut().zip(col.iter()) {
            *o = bin_value(v, bins);
        }
    }
    out
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

    /// `self = parent - self` (histogram subtraction: the larger child's
    /// histogram from the parent's and the smaller child's).
    fn subtract_from(&mut self, parent: &Histogram) {
        for b in 0..self.grad_sum.len() {
            self.grad_sum[b] = parent.grad_sum[b] - self.grad_sum[b];
            self.hess_sum[b] = parent.hess_sum[b] - self.hess_sum[b];
            self.count[b] = parent.count[b] - self.count[b];
        }
        self.missing_grad = parent.missing_grad - self.missing_grad;
        self.missing_hess = parent.missing_hess - self.missing_hess;
        self.missing_count = parent.missing_count - self.missing_count;
    }

    fn accumulate(&mut self, bin: u8, grad: f64, hess: f64) {
        if bin == MISSING_BIN {
            self.missing_grad += grad;
            self.missing_hess += hess;
            self.missing_count += 1;
        } else {
            let b = bin as usize;
            self.grad_sum[b] += grad;
            self.hess_sum[b] += hess;
            self.count[b] += 1;
        }
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
}

/// Compute the leaf value with L1 + L2 regularization.
/// Uses the proximal operator for the L1 term: soft-threshold then divide by (H + L2).
#[inline]
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

/// Below this many (sample, feature) pairs histograms are built serially.
const PARALLEL_HISTOGRAM_WORK: usize = 1 << 16;

/// Per-feature gradient/hessian histograms of the rows `indices`, one per
/// entry of `feature_indices` (built in parallel for large nodes).
fn build_histograms(
    binned_x: &Array2<u8>,
    gradients: &[f64],
    hessians: &[f64],
    indices: &[usize],
    feature_indices: &[usize],
) -> Vec<Histogram> {
    let one = |feat: usize| {
        let mut hist = Histogram::new(DEFAULT_MAX_BINS);
        let col = binned_x.column(feat);
        for &i in indices {
            hist.accumulate(col[i], gradients[i], hessians[i]);
        }
        hist
    };
    if indices.len() * feature_indices.len() >= PARALLEL_HISTOGRAM_WORK {
        use rayon::prelude::*;
        feature_indices.par_iter().map(|&f| one(f)).collect()
    } else {
        feature_indices.iter().map(|&f| one(f)).collect()
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
        },
    )
}

/// Best split of a node from its histograms (`hists[k]` belongs to
/// `feature_indices[k]`).
///
/// With monotone constraints, child outputs are clamped to the node's
/// `bounds`, the gain is computed for the clamped outputs, and splits whose
/// outputs violate the constraint direction are rejected.
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
    let parent_gain = if constrained {
        leaf_gain_given_output(total_grad, total_hess, l1, l2, parent_value)
    } else {
        leaf_gain(total_grad, total_hess, l1, l2)
    };

    for (&feat, hist) in feature_indices.iter().zip(hists) {
        if all_bins[feat].is_categorical {
            let candidate = best_categorical_split(
                hist,
                totals,
                total_count,
                feat,
                all_bins[feat].edges.len(),
                params,
                constrained.then_some(bounds),
                parent_gain,
            );
            if let Some(split) = candidate {
                if split.gain > best.as_ref().map_or(0.0, |b| b.gain) {
                    best = Some(split);
                }
            }
            continue;
        }
        // Bins above the feature's last value bin are empty: they would
        // repeat the last candidate with an equal (never better) gain. The
        // last *global* bin is never a threshold (nothing to its right but
        // missing values).
        let scan_order: Vec<u8> =
            (0..n_value_bins(&all_bins[feat]).min(DEFAULT_MAX_BINS - 1) as u8).collect();
        let n_candidates = scan_order.len();

        // Try both missing directions.
        for &missing_left in &[true, false] {
            let mut left_grad = if missing_left { hist.missing_grad } else { 0.0 };
            let mut left_hess = if missing_left { hist.missing_hess } else { 0.0 };
            let mut left_count: usize = if missing_left {
                hist.missing_count as usize
            } else {
                0
            };

            for &bin in &scan_order[..n_candidates] {
                let b = bin as usize;
                left_grad += hist.grad_sum[b];
                left_hess += hist.hess_sum[b];
                left_count += hist.count[b] as usize;

                if left_count < min_child_samples || left_hess < min_child_weight {
                    continue;
                }
                let right_count = total_count - left_count;
                // total_grad/hess include all samples (including missing).
                // When missing_left=true, left already includes missing →
                //   right = total - left (missing is NOT in right).
                // When missing_left=false, left does NOT include missing →
                //   right = total - left (missing IS in right, via total).
                // In both cases the formula is the same.
                let right_grad = total_grad - left_grad;
                let right_hess = total_hess - left_hess;

                if right_count < min_child_samples || right_hess < min_child_weight {
                    continue;
                }

                let mut left_val = leaf_value(left_grad, left_hess, l1, l2);
                let mut right_val = leaf_value(right_grad, right_hess, l1, l2);
                let gain = if constrained {
                    left_val = left_val.clamp(lo, hi);
                    right_val = right_val.clamp(lo, hi);
                    // Monotone constraint check: feature with monotone=+1
                    // requires left_value <= right_value (predictions
                    // increase as bin increases), -1 the opposite.
                    match monotone_of(feat, all_bins, monotone_constraints) {
                        1 if left_val > right_val => continue,
                        -1 if left_val < right_val => continue,
                        _ => {}
                    }
                    let left_g = leaf_gain_given_output(left_grad, left_hess, l1, l2, left_val);
                    let right_g = leaf_gain_given_output(right_grad, right_hess, l1, l2, right_val);
                    0.5 * (left_g + right_g - parent_gain)
                } else {
                    let left_g = leaf_gain(left_grad, left_hess, l1, l2);
                    let right_g = leaf_gain(right_grad, right_hess, l1, l2);
                    0.5 * (left_g + right_g - parent_gain)
                };

                if gain > min_split_gain && gain > best.as_ref().map_or(0.0, |b| b.gain) {
                    best = Some(BestSplit {
                        feature: feat,
                        bin_threshold: bin,
                        gain,
                        left_value: left_val,
                        right_value: right_val,
                        left_count,
                        right_count,
                        missing_left,
                        categorical_order: None,
                    });
                }
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
    for &b in &order[..=thresh_pos.min(order.len().saturating_sub(1))] {
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
/// updated with per-feature gain and split counts respectively.
#[allow(clippy::too_many_arguments)]
fn build_lgbm_tree(
    binned_x: &Array2<u8>,
    gradients: &[f64],
    hessians: &[f64],
    root_indices: Vec<usize>,
    feature_indices: &[usize],
    all_bins: &[FeatureBins],
    params: &LgbmParams,
    feature_gain_out: &mut [f64],
    feature_split_count_out: &mut [u32],
) -> TreeNode {
    let mut nodes: Vec<BuildNode> = Vec::new();

    // Root is initially a leaf with its value.
    let root_totals = node_totals(gradients, hessians, &root_indices);
    let root_value = leaf_value(
        root_totals.0,
        root_totals.1,
        params.reg_alpha,
        params.reg_lambda,
    );
    nodes.push(BuildNode::Leaf { value: root_value });

    let best_for = |hists: &[Histogram], totals, count, bounds, value| {
        find_best_split(
            hists,
            totals,
            count,
            feature_indices,
            all_bins,
            params,
            bounds,
            value,
        )
    };

    // Try to split the root.
    let root_hists = build_histograms(
        binned_x,
        gradients,
        hessians,
        &root_indices,
        feature_indices,
    );
    let root_split = best_for(
        &root_hists,
        root_totals,
        root_indices.len(),
        UNBOUNDED,
        root_value,
    );

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
        });
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
        let (left_idx, right_idx): (Vec<usize>, Vec<usize>) = indices.iter().partition(|&&i| {
            goes_left(
                col[i],
                split.bin_threshold,
                split.missing_left,
                mask.as_deref(),
            )
        });

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

        let depth = candidate.depth + 1;
        let splittable = |n: usize| {
            n >= 2 * params.min_child_samples && params.max_depth.is_none_or(|d| depth < d)
        };
        let (left_ok, right_ok) = (splittable(left_idx.len()), splittable(right_idx.len()));
        if !left_ok && !right_ok {
            continue;
        }

        // Histogram subtraction: build the smaller child's histograms and
        // derive the larger child's from the parent's.
        let left_smaller = left_idx.len() <= right_idx.len();
        let small_idx = if left_smaller { &left_idx } else { &right_idx };
        let small = build_histograms(binned_x, gradients, hessians, small_idx, feature_indices);
        let mut large = small.clone();
        for (l, parent) in large.iter_mut().zip(&candidate.hists) {
            l.subtract_from(parent);
        }
        drop(candidate.hists);
        let (left_hists, right_hists) = if left_smaller {
            (small, large)
        } else {
            (large, small)
        };

        // Try to split the new children
        for (ok, child_id, child_indices, hists, bounds, value) in [
            (
                left_ok,
                left_id,
                left_idx,
                left_hists,
                left_bounds,
                split.left_value,
            ),
            (
                right_ok,
                right_id,
                right_idx,
                right_hists,
                right_bounds,
                split.right_value,
            ),
        ] {
            if !ok {
                continue;
            }
            let totals = node_totals(gradients, hessians, &child_indices);
            if let Some(s) = best_for(&hists, totals, child_indices.len(), bounds, value) {
                heap.push(LeafCandidate {
                    node_id: child_id,
                    gain: s.gain,
                    split: s,
                    indices: child_indices,
                    hists,
                    depth,
                    bounds,
                });
            }
        }
    }

    // Convert flat BuildNode tree to TreeNode tree
    flatten_tree(&nodes, 0)
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

/// Gradient-based One-Side Sampling (GOSS).
///
/// Keeps the top `top_rate * n` samples by `|gradient|` and randomly samples
/// `other_rate * n` from the rest. To maintain an unbiased estimate of the
/// full gradient sum, the "other" samples' gradients and hessians are
/// amplified by `(1 - top_rate) / other_rate`.
///
/// Returns the selected indices and a parallel vector of amplification weights
/// (1.0 for top rows, `(1-top_rate)/other_rate` for random rows).
fn goss_sample(
    gradients: &[f64],
    indices: &[usize],
    top_rate: f64,
    other_rate: f64,
    rng: &mut StdRng,
) -> (Vec<usize>, Vec<f64>) {
    let n = indices.len();
    let top_n = (n as f64 * top_rate) as usize;
    let other_n = (n as f64 * other_rate) as usize;

    let mut sorted: Vec<usize> = indices.to_vec();
    sorted.sort_by(|&a, &b| {
        gradients[b]
            .abs()
            .partial_cmp(&gradients[a].abs())
            .unwrap_or(Ordering::Equal)
    });

    let mut selected: Vec<usize> = sorted[..top_n.min(n)].to_vec();
    let mut weights: Vec<f64> = vec![1.0; selected.len()];

    if sorted.len() > top_n && other_n > 0 {
        let rest = &sorted[top_n..];
        let mut shuffled: Vec<usize> = rest.to_vec();
        shuffled.shuffle(rng);
        let take = other_n.min(shuffled.len());
        // Amplification factor to unbias the sampled gradients:
        // E[grad_other] * (1 - top_rate) should equal the true sum of "other" gradients,
        // so each sampled row gets multiplied by (1 - top_rate) / other_rate.
        let amp = if other_rate > 0.0 {
            (1.0 - top_rate) / other_rate
        } else {
            0.0
        };
        selected.extend_from_slice(&shuffled[..take]);
        weights.extend(std::iter::repeat(amp).take(take));
    }

    (selected, weights)
}

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
    pub subsample: f64,
    pub colsample_bytree: f64,
    pub max_bins: usize,
    /// Minimum number of training samples per histogram bin.  Bins with fewer
    /// samples are merged with their neighbours, reducing the effective split
    /// resolution.  Matches LightGBM's `min_data_in_bin` (default: 3).
    pub min_data_in_bin: usize,
    /// Numeric binning: LightGBM's (default) or sklearn's.
    #[serde(default)]
    pub binning: BinningMethod,
    pub boosting_type: BoostingType,
    pub goss_top_rate: f64,
    pub goss_other_rate: f64,
    pub objective: LgbmObjective,
    /// Huber transition point (LightGBM's `alpha`, default 0.9).
    pub huber_delta: f64,
    pub early_stopping_rounds: Option<usize>,
    pub categorical_features: Vec<usize>,
    /// How categorical features are split.
    #[serde(default)]
    pub categorical_split: CategoricalSplitParams,
    /// Per-feature monotone constraint: -1 (decreasing), 0 (none), +1 (increasing).
    /// Length must match `n_features` (or be empty for no constraints).
    pub monotone_constraints: Vec<i8>,
    pub seed: u64,
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
            colsample_bytree: 1.0,
            max_bins: DEFAULT_MAX_BINS,
            min_data_in_bin: 3,
            binning: BinningMethod::LightGbm,
            boosting_type: BoostingType::Gbdt,
            goss_top_rate: 0.2,
            goss_other_rate: 0.1,
            objective: LgbmObjective::Regression,
            huber_delta: 0.9,
            early_stopping_rounds: None,
            categorical_features: Vec::new(),
            categorical_split: CategoricalSplitParams::default(),
            monotone_constraints: Vec::new(),
            seed: 0,
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
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    fn params(&self) -> LgbmParams {
        LgbmParams {
            num_leaves: self.num_leaves,
            max_depth: self.max_depth,
            min_child_samples: self.min_child_samples,
            min_child_weight: self.min_child_weight,
            min_split_gain: self.min_split_gain,
            reg_alpha: self.reg_alpha,
            reg_lambda: self.reg_lambda,
            monotone_constraints: self.monotone_constraints.clone(),
            categorical: self.categorical_split,
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

fn compute_regression_gradients(
    y: &[f64],
    preds: &[f64],
    objective: LgbmObjective,
    huber_delta: f64,
) -> (Vec<f64>, Vec<f64>) {
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
                grad[i] = if r == 0.0 { 0.0 } else { r.signum() };
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
            // LightGBM: 1 − alpha above the target, −alpha below.
            for i in 0..n {
                grad[i] = if preds[i] - y[i] >= 0.0 {
                    1.0 - alpha
                } else {
                    -alpha
                };
                hess[i] = 1.0;
            }
        }
        _ => unreachable!("regressor only supports regression objectives"),
    }

    (grad, hess)
}

fn validate_regression_objective(objective: LgbmObjective) -> Result<()> {
    match objective {
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
    values.sort_by(f64::total_cmp);
    let pos = (n - 1) as f64 * alpha;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    values[lo] + (values[hi] - values[lo]) * (pos - lo as f64)
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

/// `alpha`-percentile of `values[rows]`, weighted when `weights` is given.
fn rows_percentile(
    values: &[f64],
    rows: &[usize],
    weights: Option<ArrayView1<f64>>,
    alpha: f64,
) -> f64 {
    let mut v: Vec<f64> = rows.iter().map(|&i| values[i]).collect();
    match weights {
        Some(w) => {
            let wv: Vec<f64> = rows.iter().map(|&i| w[i]).collect();
            weighted_percentile(&v, &wv, alpha)
        }
        None => percentile(&mut v, alpha),
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
) {
    match node {
        TreeNode::Leaf { value } => {
            if !rows.is_empty() {
                *value = rows_percentile(residuals, &rows, weights, alpha);
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
            renew_leaf_outputs(left, binned_x, l, residuals, weights, alpha);
            renew_leaf_outputs(right, binned_x, r, residuals, weights, alpha);
        }
    }
}

/// LightGBM's Python package drops sample weights that are all 1, which
/// selects the unweighted percentile formulas; this keeps that behaviour.
fn effective_sample_weight(w: Option<ArrayView1<f64>>) -> Option<ArrayView1<f64>> {
    w.filter(|w| w.iter().any(|&v| v != 1.0))
}

/// Initial score of a regression objective: the weighted mean (squared
/// error, Huber) or the weighted median / `alpha`-percentile (L1,
/// quantile), as LightGBM's `BoostFromScore`.
fn regression_baseline(
    y: &[f64],
    weights: Option<ArrayView1<f64>>,
    objective: LgbmObjective,
) -> f64 {
    if let Some(alpha) = refit_percentile(objective) {
        let rows: Vec<usize> = (0..y.len()).collect();
        return rows_percentile(y, &rows, weights, alpha);
    }
    match weights {
        Some(w) => y.iter().zip(w).map(|(&v, &wi)| v * wi).sum::<f64>() / w.sum(),
        None => y.iter().sum::<f64>() / y.len() as f64,
    }
}

/// Select random feature subset for this tree.
fn sample_features(n_features: usize, frac: f64, rng: &mut StdRng) -> Vec<usize> {
    if frac >= 1.0 {
        return (0..n_features).collect();
    }
    let k = ((n_features as f64 * frac).ceil() as usize).max(1);
    let mut indices: Vec<usize> = (0..n_features).collect();
    indices.shuffle(rng);
    indices.truncate(k);
    indices.sort_unstable();
    indices
}

/// Select random row subset.
fn sample_rows(n: usize, frac: f64, rng: &mut StdRng) -> Vec<usize> {
    if frac >= 1.0 {
        return (0..n).collect();
    }
    let k = ((n as f64 * frac).ceil() as usize).max(1);
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(rng);
    indices.truncate(k);
    indices.sort_unstable();
    indices
}

impl LgbmRegressor {
    /// Fit with additional options (sample weights, init score, eval set, early stopping).
    pub fn fit_with_eval<'a, 'b>(
        &self,
        x: impl AsArray<'a, f64, Ix2>,
        y: impl AsArray<'b, f64, Ix1>,
        opts: &LgbmFitOptions<'_>,
    ) -> Result<FittedLgbmRegressor> {
        let x: ArrayView2<f64> = x.into();
        let y: ArrayView1<f64> = y.into();
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
        validate_regression_objective(self.objective)?;

        let n = x.nrows();
        let p = x.ncols();
        let y_vec: Vec<f64> = y.to_vec();
        let (binned_x, bins) = compute_bins(
            x,
            self.max_bins,
            &self.categorical_features,
            self.min_data_in_bin,
            self.binning,
            self.seed,
        );
        // Row-major copy for the per-row score updates.
        let binned_rows = binned_x.as_standard_layout().into_owned();

        let sample_weight = effective_sample_weight(opts.sample_weight);

        // Baseline / init_score
        let (baseline, mut preds): (f64, Vec<f64>) = if let Some(init) = opts.init_score {
            // init_score replaces the baseline entirely; we still store a
            // scalar "baseline" = mean(init) so predict() has a consistent
            // starting value for new rows.
            let m = init.iter().sum::<f64>() / n as f64;
            (m, init.to_vec())
        } else {
            let b = regression_baseline(&y_vec, sample_weight, self.objective);
            (b, vec![b; n])
        };

        let mut trees: Vec<TreeNode> = Vec::with_capacity(self.n_estimators);
        let mut feature_gain = vec![0.0f64; p];
        let mut feature_split_count = vec![0u32; p];

        let mut rng = StdRng::seed_from_u64(self.seed);
        let params = self.params();

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

        let mut best_eval_loss = f64::INFINITY;
        let mut best_iteration = 0usize;
        let mut best_feature_gain = feature_gain.clone();
        let mut best_feature_split_count = feature_split_count.clone();
        let mut stale_rounds = 0usize;

        for iter in 0..self.n_estimators {
            let (mut gradients, mut hessians) =
                compute_regression_gradients(&y_vec, &preds, self.objective, self.huber_delta);

            // Apply sample weights (multiply into gradients and hessians)
            if let Some(sw) = sample_weight {
                for i in 0..n {
                    gradients[i] *= sw[i];
                    hessians[i] *= sw[i];
                }
            }

            // Row selection: GBDT = bagging, GOSS = top-k + sampled rest w/ amplification
            let row_indices: Vec<usize> = match self.boosting_type {
                BoostingType::Gbdt => sample_rows(n, self.subsample, &mut rng),
                BoostingType::Goss => {
                    let all: Vec<usize> = (0..n).collect();
                    let (selected, amp_weights) = goss_sample(
                        &gradients,
                        &all,
                        self.goss_top_rate,
                        self.goss_other_rate,
                        &mut rng,
                    );
                    // Amplify gradients/hessians for the "other" sampled rows
                    for (k, &i) in selected.iter().enumerate() {
                        gradients[i] *= amp_weights[k];
                        hessians[i] *= amp_weights[k];
                    }
                    selected
                }
            };

            let feature_indices = sample_features(p, self.colsample_bytree, &mut rng);
            let refit = refit_percentile(self.objective).map(|a| (a, row_indices.clone()));

            let mut tree = build_lgbm_tree(
                &binned_x,
                &gradients,
                &hessians,
                row_indices,
                &feature_indices,
                &bins,
                &params,
                &mut feature_gain,
                &mut feature_split_count,
            );
            if let Some((alpha, rows)) = refit {
                let residuals: Vec<f64> = y_vec.iter().zip(&preds).map(|(y, p)| y - p).collect();
                renew_leaf_outputs(&mut tree, &binned_x, rows, &residuals, sample_weight, alpha);
            }

            // Update predictions (on ALL rows, not just sampled)
            for i in 0..n {
                preds[i] += self.learning_rate * tree.predict_bins(row_of(&binned_rows, i));
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

        Ok(FittedLgbmRegressor {
            trees,
            bins,
            baseline,
            learning_rate: self.learning_rate,
            n_features: p,
            feature_gain,
            feature_split_count,
            best_iteration,
        })
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
        for (pred, bins) in preds.iter_mut().zip(flat.chunks_exact(p.max(1))) {
            for tree in &self.trees {
                *pred += self.learning_rate * tree.predict_bins(bins);
            }
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
    pub subsample: f64,
    pub colsample_bytree: f64,
    pub max_bins: usize,
    pub min_data_in_bin: usize,
    /// Numeric binning: LightGBM's (default) or sklearn's.
    #[serde(default)]
    pub binning: BinningMethod,
    pub boosting_type: BoostingType,
    pub goss_top_rate: f64,
    pub goss_other_rate: f64,
    pub categorical_features: Vec<usize>,
    /// How categorical features are split.
    #[serde(default)]
    pub categorical_split: CategoricalSplitParams,
    pub monotone_constraints: Vec<i8>,
    pub class_weight: Option<LgbmClassWeight>,
    pub early_stopping_rounds: Option<usize>,
    pub seed: u64,
    /// Scale the softmax hessian by K / (K - 1) as LightGBM does; the
    /// HistGradientBoosting facade turns it off (sklearn's hessian).
    #[serde(skip, default = "default_true")]
    pub(crate) softmax_hessian_factor: bool,
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
            colsample_bytree: 1.0,
            max_bins: DEFAULT_MAX_BINS,
            min_data_in_bin: 3,
            binning: BinningMethod::LightGbm,
            boosting_type: BoostingType::Gbdt,
            goss_top_rate: 0.2,
            goss_other_rate: 0.1,
            categorical_features: Vec::new(),
            categorical_split: CategoricalSplitParams::default(),
            monotone_constraints: Vec::new(),
            class_weight: None,
            early_stopping_rounds: None,
            seed: 0,
            softmax_hessian_factor: true,
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
    pub fn with_class_weight(mut self, w: Option<LgbmClassWeight>) -> Self {
        self.class_weight = w;
        self
    }
    pub fn with_early_stopping(mut self, r: Option<usize>) -> Self {
        self.early_stopping_rounds = r;
        self
    }
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    fn params(&self) -> LgbmParams {
        LgbmParams {
            num_leaves: self.num_leaves,
            max_depth: self.max_depth,
            min_child_samples: self.min_child_samples,
            min_child_weight: self.min_child_weight,
            min_split_gain: self.min_split_gain,
            reg_alpha: self.reg_alpha,
            reg_lambda: self.reg_lambda,
            monotone_constraints: self.monotone_constraints.clone(),
            categorical: self.categorical_split,
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

        let n = x.nrows();
        let n_classes = self.classes.len();

        if n_classes == 2 {
            let mut proba = Array2::zeros((n, 2));
            let binned = bin_matrix(x, &self.bins);
            for i in 0..n {
                let mut score = self.baselines[0];
                for tree in &self.tree_sets[0] {
                    score += self.learning_rate * tree.predict_bins(row_of(&binned, i));
                }
                let p1 = 1.0 / (1.0 + (-score).exp());
                proba[[i, 0]] = 1.0 - p1;
                proba[[i, 1]] = p1;
            }
            Ok(proba)
        } else {
            let mut proba = Array2::zeros((n, n_classes));
            let binned = bin_matrix(x, &self.bins);
            let mut scores = vec![0.0; n_classes];
            for i in 0..n {
                for (c, ts) in self.tree_sets.iter().enumerate() {
                    scores[c] = self.baselines[c];
                    for tree in ts {
                        scores[c] += self.learning_rate * tree.predict_bins(row_of(&binned, i));
                    }
                }
                let max_s = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exp_sum: f64 = scores.iter().map(|&s| (s - max_s).exp()).sum();
                for c in 0..n_classes {
                    proba[[i, c]] = (scores[c] - max_s).exp() / exp_sum;
                }
            }
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
        let x: ArrayView2<f64> = x.into();
        let y: ArrayView1<f64> = y.into();
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

        let n = x.nrows();
        let p = x.ncols();
        let (binned_x, bins) = compute_bins(
            x,
            self.max_bins,
            &self.categorical_features,
            self.min_data_in_bin,
            self.binning,
            self.seed,
        );
        // Row-major copy for the per-row score updates.
        let binned_rows = binned_x.as_standard_layout().into_owned();

        let mut classes: Vec<f64> = y.iter().copied().collect();
        classes.sort_by(f64::total_cmp);
        classes.dedup();
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(RustMlError::InvalidParameter(
                "need at least 2 classes".into(),
            ));
        }

        // Combine class_weight + sample_weight into a single effective weight per row.
        let class_weights = self.compute_class_weights(y)?;
        let effective_weights: Vec<f64> = (0..n)
            .map(|i| class_weights[i] * opts.sample_weight.map_or(1.0, |sw| sw[i]))
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

        let mut rng = StdRng::seed_from_u64(self.seed);
        let params = self.params();
        let mut feature_gain = vec![0.0f64; p];
        let mut feature_split_count = vec![0u32; p];

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
            let baseline = (pos_frac / (1.0 - pos_frac + 1e-15)).ln();
            let mut raw = if let Some(init) = opts.init_score {
                init.to_vec()
            } else {
                vec![baseline; n]
            };
            let mut trees = Vec::with_capacity(self.n_estimators);
            let mut eval_raw: Option<Vec<f64>> =
                eval_binned.as_ref().map(|eb| vec![baseline; eb.nrows()]);

            for iter in 0..self.n_estimators {
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

                let row_indices: Vec<usize> = match self.boosting_type {
                    BoostingType::Gbdt => sample_rows(n, self.subsample, &mut rng),
                    BoostingType::Goss => {
                        let all: Vec<usize> = (0..n).collect();
                        let (selected, amp) = goss_sample(
                            &gradients,
                            &all,
                            self.goss_top_rate,
                            self.goss_other_rate,
                            &mut rng,
                        );
                        for (k, &i) in selected.iter().enumerate() {
                            gradients[i] *= amp[k];
                            hessians[i] *= amp[k];
                        }
                        selected
                    }
                };
                let feature_indices = sample_features(p, self.colsample_bytree, &mut rng);

                let tree = build_lgbm_tree(
                    &binned_x,
                    &gradients,
                    &hessians,
                    row_indices,
                    &feature_indices,
                    &bins,
                    &params,
                    &mut feature_gain,
                    &mut feature_split_count,
                );

                for i in 0..n {
                    raw[i] += self.learning_rate * tree.predict_bins(row_of(&binned_rows, i));
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

            Ok(FittedLgbmClassifier {
                tree_sets: vec![trees],
                bins,
                baselines: vec![baseline],
                classes,
                learning_rate: self.learning_rate,
                n_features: p,
                feature_gain,
                feature_split_count,
                best_iteration,
            })
        } else {
            // Multiclass: softmax with one tree per class per round.
            let mut tree_sets: Vec<Vec<TreeNode>> =
                vec![Vec::with_capacity(self.n_estimators); n_classes];
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
                let bl = prior.ln();
                baselines.push(bl);
                raw_scores[c] = vec![bl; n];
            }
            let mut eval_raw: Option<Vec<Vec<f64>>> = eval_binned
                .as_ref()
                .map(|eb| baselines.iter().map(|&bl| vec![bl; eb.nrows()]).collect());

            for iter in 0..self.n_estimators {
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
                for (c, &cls) in classes.iter().enumerate() {
                    let mut gradients: Vec<f64> = (0..n)
                        .map(|i| {
                            let lbl = if y[i] == cls { 1.0 } else { 0.0 };
                            (probas[i][c] - lbl) * effective_weights[i]
                        })
                        .collect();
                    let mut hessians: Vec<f64> = (0..n)
                        .map(|i| {
                            hess_factor
                                * (probas[i][c] * (1.0 - probas[i][c])).max(1e-12)
                                * effective_weights[i]
                        })
                        .collect();

                    let row_indices: Vec<usize> = match self.boosting_type {
                        BoostingType::Gbdt => sample_rows(n, self.subsample, &mut rng),
                        BoostingType::Goss => {
                            let all: Vec<usize> = (0..n).collect();
                            let (selected, amp) = goss_sample(
                                &gradients,
                                &all,
                                self.goss_top_rate,
                                self.goss_other_rate,
                                &mut rng,
                            );
                            for (k, &i) in selected.iter().enumerate() {
                                gradients[i] *= amp[k];
                                hessians[i] *= amp[k];
                            }
                            selected
                        }
                    };
                    let feature_indices = sample_features(p, self.colsample_bytree, &mut rng);

                    let tree = build_lgbm_tree(
                        &binned_x,
                        &gradients,
                        &hessians,
                        row_indices,
                        &feature_indices,
                        &bins,
                        &params,
                        &mut feature_gain,
                        &mut feature_split_count,
                    );

                    for i in 0..n {
                        raw_scores[c][i] +=
                            self.learning_rate * tree.predict_bins(row_of(&binned_rows, i));
                    }
                    if let (Some(eb), Some(er)) = (eval_binned.as_ref(), eval_raw.as_mut()) {
                        for i in 0..eb.nrows() {
                            er[c][i] += self.learning_rate * tree.predict_bins(row_of(eb, i));
                        }
                    }

                    tree_sets[c].push(tree);
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

            Ok(FittedLgbmClassifier {
                tree_sets,
                bins,
                baselines,
                classes,
                learning_rate: self.learning_rate,
                n_features: p,
                feature_gain,
                feature_split_count,
                best_iteration,
            })
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
        let proba = self.predict_proba(x)?;
        let n = x.nrows();
        let mut preds = Array1::zeros(n);
        for i in 0..n {
            let mut best_c = 0;
            let mut best_p = proba[[i, 0]];
            for c in 1..self.classes.len() {
                if proba[[i, c]] > best_p {
                    best_p = proba[[i, c]];
                    best_c = c;
                }
            }
            preds[i] = self.classes[best_c];
        }
        Ok(preds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

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

        let fitted = LgbmRegressor::new()
            .with_n_estimators(20)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .with_boosting_type(BoostingType::Goss)
            .with_goss_rates(0.6, 0.6)
            .fit(&x, &y)
            .unwrap();

        let preds = fitted.predict(&x).unwrap();
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
            .with_seed(1);
        let m2 = LgbmRegressor::new()
            .with_n_estimators(20)
            .with_num_leaves(4)
            .with_min_child_samples(1)
            .with_subsample(0.5)
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
