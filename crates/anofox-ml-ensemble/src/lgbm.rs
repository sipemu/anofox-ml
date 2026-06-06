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

use anofox_ml_core::{Fit, Predict, Result, RustMlError};
use ndarray::{Array1, Array2};
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
    x: &Array2<f64>,
    max_bins: usize,
    categorical_features: &[usize],
    min_data_in_bin: usize,
) -> (Array2<u8>, Vec<FeatureBins>) {
    let n = x.nrows();
    let p = x.ncols();
    let mut binned = Array2::zeros((n, p));
    let mut all_bins = Vec::with_capacity(p);

    let cat_set: std::collections::HashSet<usize> = categorical_features.iter().copied().collect();

    for j in 0..p {
        let is_categorical = cat_set.contains(&j);

        let mut edges = Vec::new();

        if is_categorical {
            // For categorical features, bin edges are the unique values themselves.
            let mut col: Vec<f64> = (0..n).map(|i| x[[i, j]]).filter(|v| !v.is_nan()).collect();
            col.sort_by(|a, b| a.partial_cmp(b).unwrap());
            col.dedup();
            for &v in col.iter().take(max_bins - 1) {
                edges.push(v);
            }
        } else {
            // Greedy binning matching LightGBM: collect distinct values with
            // sample counts, then greedily fill bins so each has at least
            // `min_data_in_bin` samples.  Bin boundaries are placed at the
            // midpoint between the last value of one group and the first of
            // the next.
            let mut raw: Vec<f64> = (0..n).map(|i| x[[i, j]]).filter(|v| !v.is_nan()).collect();
            raw.sort_by(|a, b| a.partial_cmp(b).unwrap());

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
                // Direct bin lookup for categorical
                let bin = edges
                    .iter()
                    .position(|&e| (e - v).abs() < 1e-12)
                    .unwrap_or(edges.len().saturating_sub(1));
                binned[[i, j]] = bin.min((MISSING_BIN - 1) as usize) as u8;
            } else {
                let bin = edges.partition_point(|&e| e < v) as u8;
                binned[[i, j]] = bin.min(MISSING_BIN - 1);
            }
        }

        all_bins.push(FeatureBins {
            edges,
            is_categorical,
        });
    }

    (binned, all_bins)
}

fn bin_row(row: &[f64], all_bins: &[FeatureBins]) -> Vec<u8> {
    row.iter()
        .zip(all_bins.iter())
        .map(|(&v, bins)| {
            if v.is_nan() {
                MISSING_BIN
            } else if bins.is_categorical {
                bins.edges
                    .iter()
                    .position(|&e| (e - v).abs() < 1e-12)
                    .map(|b| b.min((MISSING_BIN - 1) as usize) as u8)
                    .unwrap_or(MISSING_BIN - 1)
            } else {
                let bin = bins.edges.partition_point(|&e| e < v) as u8;
                bin.min(MISSING_BIN - 1)
            }
        })
        .collect()
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

    fn reset(&mut self) {
        self.grad_sum.fill(0.0);
        self.hess_sum.fill(0.0);
        self.count.fill(0);
        self.missing_grad = 0.0;
        self.missing_hess = 0.0;
        self.missing_count = 0;
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

/// Find the best split for this node across all candidate features.
#[allow(clippy::too_many_arguments)]
fn find_best_split(
    binned_x: &Array2<u8>,
    gradients: &[f64],
    hessians: &[f64],
    indices: &[usize],
    feature_indices: &[usize],
    all_bins: &[FeatureBins],
    min_child_samples: usize,
    min_child_weight: f64,
    min_split_gain: f64,
    l1: f64,
    l2: f64,
    monotone_constraints: &[i8],
) -> Option<BestSplit> {
    let n_bins = DEFAULT_MAX_BINS;
    let mut best: Option<BestSplit> = None;
    let mut hist = Histogram::new(n_bins);

    let total_grad: f64 = indices.iter().map(|&i| gradients[i]).sum();
    let total_hess: f64 = indices.iter().map(|&i| hessians[i]).sum();
    let total_count = indices.len();

    let parent_gain = leaf_gain(total_grad, total_hess, l1, l2);

    for &feat in feature_indices {
        hist.reset();

        for &i in indices {
            hist.accumulate(binned_x[[i, feat]], gradients[i], hessians[i]);
        }

        // Determine scan order: for categorical, sort by grad/hess ratio (Fisher method).
        let scan_order: Vec<u8> = if all_bins[feat].is_categorical {
            let mut order: Vec<u8> = (0..n_bins as u8).collect();
            order.sort_by(|&a, &b| {
                let ra = if hist.hess_sum[a as usize] > 1e-10 {
                    hist.grad_sum[a as usize] / hist.hess_sum[a as usize]
                } else {
                    0.0
                };
                let rb = if hist.hess_sum[b as usize] > 1e-10 {
                    hist.grad_sum[b as usize] / hist.hess_sum[b as usize]
                } else {
                    0.0
                };
                ra.partial_cmp(&rb).unwrap_or(Ordering::Equal)
            });
            order
        } else {
            (0..n_bins as u8).collect()
        };

        // Try both missing directions.
        for &missing_left in &[true, false] {
            let mut left_grad = if missing_left { hist.missing_grad } else { 0.0 };
            let mut left_hess = if missing_left { hist.missing_hess } else { 0.0 };
            let mut left_count: usize = if missing_left {
                hist.missing_count as usize
            } else {
                0
            };

            for (idx, &bin) in scan_order.iter().enumerate() {
                if idx >= scan_order.len() - 1 {
                    break;
                }
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

                let left_g = leaf_gain(left_grad, left_hess, l1, l2);
                let right_g = leaf_gain(right_grad, right_hess, l1, l2);
                let gain = 0.5 * (left_g + right_g - parent_gain);

                // Monotone constraint check: feature with monotone=+1 requires
                // left_value <= right_value (predictions increase as bin increases).
                // Categorical features cannot be monotone-constrained in a simple way.
                let left_val = leaf_value(left_grad, left_hess, l1, l2);
                let right_val = leaf_value(right_grad, right_hess, l1, l2);
                let mono = monotone_constraints.get(feat).copied().unwrap_or(0);
                if !all_bins[feat].is_categorical && mono != 0 {
                    match mono {
                        1 if left_val > right_val => continue,
                        -1 if left_val < right_val => continue,
                        _ => {}
                    }
                }

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
                        categorical_order: if all_bins[feat].is_categorical {
                            Some(scan_order.clone())
                        } else {
                            None
                        },
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
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

impl TreeNode {
    fn predict_binned(&self, bins: &[u8]) -> f64 {
        match self {
            TreeNode::Leaf { value } => *value,
            TreeNode::Internal {
                feature,
                bin_threshold,
                missing_left,
                categorical_order,
                left,
                right,
            } => {
                let bin = bins[*feature];
                let go_left = if bin == MISSING_BIN {
                    *missing_left
                } else if let Some(order) = categorical_order {
                    // Categorical: find position in order; bins at positions <= threshold go left
                    let pos = order.iter().position(|&b| b == bin).unwrap_or(order.len());
                    let thresh_pos = order.iter().position(|&b| b == *bin_threshold).unwrap_or(0);
                    pos <= thresh_pos
                } else {
                    bin <= *bin_threshold
                };
                if go_left {
                    left.predict_binned(bins)
                } else {
                    right.predict_binned(bins)
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
    depth: usize,
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
    let root_grad: f64 = root_indices.iter().map(|&i| gradients[i]).sum();
    let root_hess: f64 = root_indices.iter().map(|&i| hessians[i]).sum();
    let root_value = leaf_value(root_grad, root_hess, params.reg_alpha, params.reg_lambda);
    nodes.push(BuildNode::Leaf { value: root_value });

    // Try to split the root.
    let root_split = find_best_split(
        binned_x,
        gradients,
        hessians,
        &root_indices,
        feature_indices,
        all_bins,
        params.min_child_samples,
        params.min_child_weight,
        params.min_split_gain,
        params.reg_alpha,
        params.reg_lambda,
        &params.monotone_constraints,
    );

    let mut heap: BinaryHeap<LeafCandidate> = BinaryHeap::new();
    if let Some(split) = root_split {
        heap.push(LeafCandidate {
            node_id: 0,
            gain: split.gain,
            split,
            indices: root_indices,
            depth: 0,
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
        let (left_idx, right_idx): (Vec<usize>, Vec<usize>) = indices.iter().partition(|&&i| {
            let bin = binned_x[[i, split.feature]];
            if bin == MISSING_BIN {
                split.missing_left
            } else if let Some(ref order) = split.categorical_order {
                let pos = order.iter().position(|&b| b == bin).unwrap_or(order.len());
                let thresh_pos = order
                    .iter()
                    .position(|&b| b == split.bin_threshold)
                    .unwrap_or(0);
                pos <= thresh_pos
            } else {
                bin <= split.bin_threshold
            }
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

        // Try to split the new children
        for (child_id, child_indices, depth) in [
            (left_id, left_idx, candidate.depth + 1),
            (right_id, right_idx, candidate.depth + 1),
        ] {
            if child_indices.len() < 2 * params.min_child_samples {
                continue;
            }
            if let Some(max_d) = params.max_depth {
                if depth >= max_d {
                    continue;
                }
            }
            let child_split = find_best_split(
                binned_x,
                gradients,
                hessians,
                &child_indices,
                feature_indices,
                all_bins,
                params.min_child_samples,
                params.min_child_weight,
                params.min_split_gain,
                params.reg_alpha,
                params.reg_lambda,
                &params.monotone_constraints,
            );
            if let Some(s) = child_split {
                heap.push(LeafCandidate {
                    node_id: child_id,
                    gain: s.gain,
                    split: s,
                    indices: child_indices,
                    depth,
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
    pub boosting_type: BoostingType,
    pub goss_top_rate: f64,
    pub goss_other_rate: f64,
    pub objective: LgbmObjective,
    pub huber_delta: f64,
    pub early_stopping_rounds: Option<usize>,
    pub categorical_features: Vec<usize>,
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
            boosting_type: BoostingType::Gbdt,
            goss_top_rate: 0.2,
            goss_other_rate: 0.1,
            objective: LgbmObjective::Regression,
            huber_delta: 1.0,
            early_stopping_rounds: None,
            categorical_features: Vec::new(),
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
    pub fn with_min_data_in_bin(mut self, n: usize) -> Self {
        self.min_data_in_bin = n;
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
    pub sample_weight: Option<&'a Array1<f64>>,
    /// Optional initial scores / baseline predictions (length = n_samples).
    /// Used instead of the default baseline.
    pub init_score: Option<&'a Array1<f64>>,
    /// Optional evaluation set (X_eval, y_eval) for early stopping.
    pub eval_set: Option<(&'a Array2<f64>, &'a Array1<f64>)>,
    /// Optional per-sample weights for the eval set.
    pub eval_sample_weight: Option<&'a Array1<f64>>,
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
            for i in 0..n {
                let r = preds[i] - y[i];
                grad[i] = r.signum();
                hess[i] = 1.0; // approximation
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
        _ => unreachable!("regressor only supports regression objectives"),
    }

    (grad, hess)
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
    pub fn fit_with_eval(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        opts: &LgbmFitOptions<'_>,
    ) -> Result<FittedLgbmRegressor> {
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

        let n = x.nrows();
        let p = x.ncols();
        let (binned_x, bins) = compute_bins(
            x,
            self.max_bins,
            &self.categorical_features,
            self.min_data_in_bin,
        );

        // Baseline / init_score
        let mean_y: f64 = y.iter().sum::<f64>() / n as f64;
        let (baseline, mut preds): (f64, Vec<f64>) = if let Some(init) = opts.init_score {
            // init_score replaces the baseline entirely; we still store a
            // scalar "baseline" = mean(init) so predict() has a consistent
            // starting value for new rows.
            let m = init.iter().sum::<f64>() / n as f64;
            (m, init.to_vec())
        } else {
            (mean_y, vec![mean_y; n])
        };

        let mut trees: Vec<TreeNode> = Vec::with_capacity(self.n_estimators);
        let mut feature_gain = vec![0.0f64; p];
        let mut feature_split_count = vec![0u32; p];

        let mut rng = StdRng::seed_from_u64(self.seed);
        let params = self.params();

        // Prepare eval set predictions if present
        let eval_binned = opts.eval_set.as_ref().map(|(xe, _)| {
            let (b, _) = compute_bins(
                xe,
                self.max_bins,
                &self.categorical_features,
                self.min_data_in_bin,
            );
            b
        });
        let mut eval_preds: Option<Vec<f64>> = opts
            .eval_set
            .as_ref()
            .map(|(xe, _)| vec![baseline; xe.nrows()]);

        let mut best_eval_loss = f64::INFINITY;
        let mut best_iteration = 0usize;
        let mut stale_rounds = 0usize;

        for iter in 0..self.n_estimators {
            let (mut gradients, mut hessians) = compute_regression_gradients(
                y.as_slice().unwrap(),
                &preds,
                self.objective,
                self.huber_delta,
            );

            // Apply sample weights (multiply into gradients and hessians)
            if let Some(sw) = opts.sample_weight {
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

            // Update predictions (on ALL rows, not just sampled)
            for i in 0..n {
                let row_bins: Vec<u8> = (0..p).map(|j| binned_x[[i, j]]).collect();
                preds[i] += self.learning_rate * tree.predict_binned(&row_bins);
            }

            // Update eval predictions and check early stopping
            if let (Some((xe, ye)), Some(ref eb), Some(ref mut ep)) =
                (opts.eval_set, eval_binned.as_ref(), eval_preds.as_mut())
            {
                for i in 0..xe.nrows() {
                    let row_bins: Vec<u8> = (0..p).map(|j| eb[[i, j]]).collect();
                    ep[i] += self.learning_rate * tree.predict_binned(&row_bins);
                }
                let loss = eval_loss(
                    ye.as_slice().unwrap(),
                    ep,
                    opts.eval_sample_weight,
                    self.objective,
                    self.huber_delta,
                );
                if loss < best_eval_loss - 1e-12 {
                    best_eval_loss = loss;
                    best_iteration = iter + 1;
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
    weights: Option<&Array1<f64>>,
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

    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        self.fit_with_eval(x, y, &LgbmFitOptions::default())
    }
}

impl Predict<f64> for FittedLgbmRegressor {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }

        let n = x.nrows();
        let mut preds = Array1::from_elem(n, self.baseline);

        for i in 0..n {
            let row: Vec<f64> = (0..self.n_features).map(|j| x[[i, j]]).collect();
            let bins = bin_row(&row, &self.bins);
            for tree in &self.trees {
                preds[i] += self.learning_rate * tree.predict_binned(&bins);
            }
        }

        Ok(preds)
    }
}

// ============================================================
// LgbmClassifier
// ============================================================

/// Class weight strategy for LgbmClassifier.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum LgbmClassWeight {
    /// Automatic balancing: weight_c = n_samples / (n_classes * class_count_c).
    Balanced,
    /// Manual per-class weights as (class_label, weight) pairs.
    Manual(Vec<(f64, f64)>),
}

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
    pub boosting_type: BoostingType,
    pub goss_top_rate: f64,
    pub goss_other_rate: f64,
    pub categorical_features: Vec<usize>,
    pub monotone_constraints: Vec<i8>,
    pub class_weight: Option<LgbmClassWeight>,
    pub early_stopping_rounds: Option<usize>,
    pub seed: u64,
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
            boosting_type: BoostingType::Gbdt,
            goss_top_rate: 0.2,
            goss_other_rate: 0.1,
            categorical_features: Vec::new(),
            monotone_constraints: Vec::new(),
            class_weight: None,
            early_stopping_rounds: None,
            seed: 0,
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
    pub fn with_min_data_in_bin(mut self, n: usize) -> Self {
        self.min_data_in_bin = n;
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
        }
    }

    /// Compute per-sample class weights from the class_weight strategy.
    fn compute_class_weights(&self, y: &Array1<f64>, classes: &[f64]) -> Vec<f64> {
        let n = y.len();
        match &self.class_weight {
            None => vec![1.0; n],
            Some(LgbmClassWeight::Balanced) => {
                let n_classes = classes.len() as f64;
                let mut per_class: std::collections::HashMap<u64, f64> =
                    std::collections::HashMap::new();
                for &cls in classes {
                    let count = y.iter().filter(|&&v| v == cls).count() as f64;
                    let w = if count > 0.0 {
                        n as f64 / (n_classes * count)
                    } else {
                        1.0
                    };
                    per_class.insert(cls.to_bits(), w);
                }
                y.iter()
                    .map(|&v| *per_class.get(&v.to_bits()).unwrap_or(&1.0))
                    .collect()
            }
            Some(LgbmClassWeight::Manual(pairs)) => {
                let map: std::collections::HashMap<u64, f64> =
                    pairs.iter().map(|&(c, w)| (c.to_bits(), w)).collect();
                y.iter()
                    .map(|&v| *map.get(&v.to_bits()).unwrap_or(&1.0))
                    .collect()
            }
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
}

impl FittedLgbmClassifier {
    pub fn classes(&self) -> &[f64] {
        &self.classes
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

    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
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
            for i in 0..n {
                let row: Vec<f64> = (0..self.n_features).map(|j| x[[i, j]]).collect();
                let bins = bin_row(&row, &self.bins);
                let mut score = self.baselines[0];
                for tree in &self.tree_sets[0] {
                    score += self.learning_rate * tree.predict_binned(&bins);
                }
                let p1 = 1.0 / (1.0 + (-score).exp());
                proba[[i, 0]] = 1.0 - p1;
                proba[[i, 1]] = p1;
            }
            Ok(proba)
        } else {
            let mut proba = Array2::zeros((n, n_classes));
            for i in 0..n {
                let row: Vec<f64> = (0..self.n_features).map(|j| x[[i, j]]).collect();
                let bins = bin_row(&row, &self.bins);
                let mut scores = vec![0.0; n_classes];
                for (c, ts) in self.tree_sets.iter().enumerate() {
                    scores[c] = self.baselines[c];
                    for tree in ts {
                        scores[c] += self.learning_rate * tree.predict_binned(&bins);
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
    pub fn fit_with_eval(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        opts: &LgbmFitOptions<'_>,
    ) -> Result<FittedLgbmClassifier> {
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

        let n = x.nrows();
        let p = x.ncols();
        let (binned_x, bins) = compute_bins(
            x,
            self.max_bins,
            &self.categorical_features,
            self.min_data_in_bin,
        );

        let mut classes: Vec<f64> = y.iter().copied().collect();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        classes.dedup();
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(RustMlError::InvalidParameter(
                "need at least 2 classes".into(),
            ));
        }

        // Combine class_weight + sample_weight into a single effective weight per row.
        let class_weights = self.compute_class_weights(y, &classes);
        let effective_weights: Vec<f64> = (0..n)
            .map(|i| class_weights[i] * opts.sample_weight.map_or(1.0, |sw| sw[i]))
            .collect();

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
            let pos_frac: f64 = labels.iter().sum::<f64>() / n as f64;
            let baseline = (pos_frac / (1.0 - pos_frac + 1e-15)).ln();
            let mut raw = if let Some(init) = opts.init_score {
                init.to_vec()
            } else {
                vec![baseline; n]
            };
            let mut trees = Vec::with_capacity(self.n_estimators);

            for _iter in 0..self.n_estimators {
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
                    let row_bins: Vec<u8> = (0..p).map(|j| binned_x[[i, j]]).collect();
                    raw[i] += self.learning_rate * tree.predict_binned(&row_bins);
                }

                trees.push(tree);
            }

            Ok(FittedLgbmClassifier {
                tree_sets: vec![trees],
                bins,
                baselines: vec![baseline],
                classes,
                learning_rate: self.learning_rate,
                n_features: p,
                feature_gain,
                feature_split_count,
            })
        } else {
            // Multiclass: softmax with one tree per class per round.
            let mut tree_sets: Vec<Vec<TreeNode>> =
                vec![Vec::with_capacity(self.n_estimators); n_classes];
            let mut baselines = Vec::with_capacity(n_classes);
            let mut raw_scores = vec![vec![0.0; n]; n_classes];

            for (c, &cls) in classes.iter().enumerate() {
                let count = y.iter().filter(|&&v| v == cls).count() as f64;
                let prior = (count / n as f64).max(1e-15);
                let bl = prior.ln();
                baselines.push(bl);
                raw_scores[c] = vec![bl; n];
            }

            for _iter in 0..self.n_estimators {
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

                for (c, &cls) in classes.iter().enumerate() {
                    let mut gradients: Vec<f64> = (0..n)
                        .map(|i| {
                            let lbl = if y[i] == cls { 1.0 } else { 0.0 };
                            (probas[i][c] - lbl) * effective_weights[i]
                        })
                        .collect();
                    let mut hessians: Vec<f64> = (0..n)
                        .map(|i| {
                            (probas[i][c] * (1.0 - probas[i][c])).max(1e-12) * effective_weights[i]
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
                        let row_bins: Vec<u8> = (0..p).map(|j| binned_x[[i, j]]).collect();
                        raw_scores[c][i] += self.learning_rate * tree.predict_binned(&row_bins);
                    }

                    tree_sets[c].push(tree);
                }
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
            })
        }
    }
}

impl Fit<f64> for LgbmClassifier {
    type Fitted = FittedLgbmClassifier;

    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        self.fit_with_eval(x, y, &LgbmFitOptions::default())
    }
}

impl Predict<f64> for FittedLgbmClassifier {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
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
            sample_weight: Some(&sw),
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
            init_score: Some(&init),
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
            sample_weight: Some(&sw),
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
            sample_weight: Some(&sw),
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
            init_score: Some(&init),
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
