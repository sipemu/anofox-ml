//! Histogram-based gradient boosting (classifier and regressor).
//!
//! Much faster than classical gradient boosting for medium-to-large datasets.
//! Features are binned into at most `max_bins <= 255` discrete bins (plus one
//! dedicated bin for missing values), enabling O(n) split finding via
//! histogram accumulation instead of O(n log n) sorting.
//!
//! **Missing values** (NaN) are supported natively, like sklearn's
//! `HistGradientBoosting*`: they get their own bin, and at every split on a
//! feature that had missing values during training both directions are
//! evaluated and the better one is stored with the node. If a feature had no
//! missing values during training, missing values at prediction time follow
//! the child that received more training samples.

use anofox_ml_core::{Fit, Predict, Result, RustMlError};
use ndarray::{Array1, Array2};

/// Default (and maximum) number of bins for non-missing values.
const MAX_BINS: usize = 255;
/// Bin index reserved for missing values (NaN).
const MISSING_BIN: u8 = 255;
/// Histogram length: 255 value bins + the missing-value bin.
const HIST_LEN: usize = 256;

// ============================================================
// Binning
// ============================================================

/// Bin edges for one feature.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct FeatureBins {
    /// Sorted thresholds. Value v maps to bin i if edges[i-1] < v <= edges[i].
    edges: Vec<f64>,
}

impl FeatureBins {
    /// Number of bins used by non-missing values.
    fn n_bins_non_missing(&self) -> usize {
        self.edges.len() + 1
    }
}

/// Bin index of `v` given sorted `edges`; NaN maps to [`MISSING_BIN`].
#[inline]
fn value_to_bin(v: f64, edges: &[f64]) -> u8 {
    if v.is_nan() {
        MISSING_BIN
    } else {
        // Clamp before narrowing: value bins never collide with MISSING_BIN.
        edges
            .partition_point(|&e| e < v)
            .min(MISSING_BIN as usize - 1) as u8
    }
}

/// Validate `max_bins` (sklearn: 2 <= max_bins <= 255; one extra bin is
/// reserved for missing values).
fn validate_max_bins(max_bins: usize) -> Result<()> {
    if !(2..=MAX_BINS).contains(&max_bins) {
        return Err(RustMlError::InvalidParameter(format!(
            "max_bins must be in [2, {MAX_BINS}], got {max_bins}"
        )));
    }
    Ok(())
}

/// Bin all features into u8 indices.
///
/// Returns the binned matrix, the per-feature bin edges and whether each
/// feature contained missing values.
fn compute_bins(x: &Array2<f64>, max_bins: usize) -> (Array2<u8>, Vec<FeatureBins>, Vec<bool>) {
    let n = x.nrows();
    let p = x.ncols();
    let mut binned = Array2::zeros((n, p));
    let mut all_bins = Vec::with_capacity(p);
    let mut has_missing = Vec::with_capacity(p);

    for j in 0..p {
        let column = x.column(j);
        has_missing.push(column.iter().any(|v| v.is_nan()));

        // Edges are computed from the non-NaN values only.
        let mut col: Vec<f64> = column.iter().copied().filter(|v| !v.is_nan()).collect();
        col.sort_by(f64::total_cmp);
        col.dedup();

        // Compute quantile-based bin edges
        let n_edges = (col.len()).min(max_bins - 1);
        let mut edges = Vec::with_capacity(n_edges);
        for k in 1..=n_edges {
            let idx = (k * col.len() / (n_edges + 1)).min(col.len() - 1);
            let edge = col[idx];
            if edges.last().map_or(true, |&last: &f64| edge > last) {
                edges.push(edge);
            }
        }

        // Map values to bins (NaN -> MISSING_BIN).
        for i in 0..n {
            binned[[i, j]] = value_to_bin(x[[i, j]], &edges);
        }

        all_bins.push(FeatureBins { edges });
    }

    (binned, all_bins, has_missing)
}

/// Map a new data point's features to bin indices.
fn bin_row(row: &[f64], all_bins: &[FeatureBins]) -> Vec<u8> {
    row.iter()
        .zip(all_bins.iter())
        .map(|(&v, bins)| value_to_bin(v, &bins.edges))
        .collect()
}

// ============================================================
// Histogram tree node
// ============================================================

/// A histogram accumulating gradient/hessian sums per bin.
#[derive(Clone)]
struct Histogram {
    /// Per-bin sum of gradients. Length = n_bins.
    grad_sum: Vec<f64>,
    /// Per-bin sum of hessians. Length = n_bins.
    hess_sum: Vec<f64>,
    /// Per-bin sample count.
    count: Vec<u32>,
}

impl Histogram {
    fn new(n_bins: usize) -> Self {
        Self {
            grad_sum: vec![0.0; n_bins],
            hess_sum: vec![0.0; n_bins],
            count: vec![0; n_bins],
        }
    }

    fn reset(&mut self) {
        self.grad_sum.fill(0.0);
        self.hess_sum.fill(0.0);
        self.count.fill(0);
    }
}

/// Result of finding the best split for a node.
#[allow(dead_code)]
struct HistSplit {
    feature: usize,
    bin_threshold: u8,
    gain: f64,
    left_value: f64,
    right_value: f64,
    left_count: usize,
    right_count: usize,
    /// Whether samples in the missing-value bin go to the left child.
    missing_go_to_left: bool,
}

/// Running sums of one side of a candidate split.
#[derive(Clone, Copy, Default)]
struct SideSums {
    grad: f64,
    hess: f64,
    count: usize,
}

/// Gradient statistics of a whole node.
struct NodeSums {
    grad: f64,
    hess: f64,
    count: usize,
}

/// Static split-finding settings.
struct SplitSettings<'a> {
    bins: &'a [FeatureBins],
    has_missing: &'a [bool],
    min_samples_leaf: usize,
    l2_regularization: f64,
}

/// Evaluate one candidate (left statistics given; right = node - left) and
/// keep it in `best` if it improves the gain.
#[inline]
fn consider_split(
    best: &mut Option<HistSplit>,
    feature: usize,
    bin: usize,
    left: SideSums,
    node: &NodeSums,
    missing_go_to_left: bool,
    l2: f64,
) {
    let right = SideSums {
        grad: node.grad - left.grad,
        hess: node.hess - left.hess,
        count: node.count - left.count,
    };
    // Gain = left_term + right_term - parent_term, term = G^2 / (H + lambda)
    let parent_term = node.grad * node.grad / (node.hess + l2);
    let left_term = left.grad * left.grad / (left.hess + l2);
    let right_term = right.grad * right.grad / (right.hess + l2);
    let gain = 0.5 * (left_term + right_term - parent_term);

    if gain > best.as_ref().map_or(0.0, |b| b.gain) {
        *best = Some(HistSplit {
            feature,
            bin_threshold: bin as u8,
            gain,
            left_value: -left.grad / (left.hess + l2),
            right_value: -right.grad / (right.hess + l2),
            left_count: left.count,
            right_count: right.count,
            missing_go_to_left,
        });
    }
}

/// Find the best split across all features using histograms.
///
/// Mirrors sklearn's splitter: bins are scanned left-to-right with missing
/// values assigned to the right child (including the split "all non-missing
/// left / all missing right"), and — for features that had missing values in
/// training — right-to-left with missing values assigned to the left child.
fn find_best_hist_split(
    binned_x: &Array2<u8>,
    gradients: &[f64],
    hessians: &[f64],
    indices: &[usize],
    settings: &SplitSettings<'_>,
) -> Option<HistSplit> {
    let mut best: Option<HistSplit> = None;
    let mut hist = Histogram::new(HIST_LEN);
    let min_leaf = settings.min_samples_leaf;
    let l2 = settings.l2_regularization;

    let node = NodeSums {
        grad: indices.iter().map(|&i| gradients[i]).sum(),
        hess: indices.iter().map(|&i| hessians[i]).sum(),
        count: indices.len(),
    };

    for feat in 0..binned_x.ncols() {
        hist.reset();
        for &i in indices {
            let bin = binned_x[[i, feat]] as usize;
            hist.grad_sum[bin] += gradients[i];
            hist.hess_sum[bin] += hessians[i];
            hist.count[bin] += 1;
        }

        let n_bins = settings.bins[feat]
            .n_bins_non_missing()
            .min(MISSING_BIN as usize);
        let has_missing = settings.has_missing[feat];

        // Left to right: missing values go right. With missing values the
        // last non-missing bin may go left too (missing-only right child).
        let end = n_bins - 1 + usize::from(has_missing);
        let mut left = SideSums::default();
        for bin in 0..end {
            left.grad += hist.grad_sum[bin];
            left.hess += hist.hess_sum[bin];
            left.count += hist.count[bin] as usize;
            if left.count < min_leaf {
                continue;
            }
            if node.count - left.count < min_leaf {
                break;
            }
            consider_split(&mut best, feat, bin, left, &node, false, l2);
        }

        // Right to left: missing values go left.
        if has_missing && n_bins >= 2 {
            let mut right = SideSums::default();
            for bin in (0..n_bins - 1).rev() {
                right.grad += hist.grad_sum[bin + 1];
                right.hess += hist.hess_sum[bin + 1];
                right.count += hist.count[bin + 1] as usize;
                if right.count < min_leaf {
                    continue;
                }
                let left = SideSums {
                    grad: node.grad - right.grad,
                    hess: node.hess - right.hess,
                    count: node.count - right.count,
                };
                if left.count < min_leaf {
                    break;
                }
                consider_split(&mut best, feat, bin, left, &node, true, l2);
            }
        }
    }

    best
}

// ============================================================
// Hist tree structure
// ============================================================

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
enum HistNode {
    Leaf {
        value: f64,
    },
    Internal {
        feature: usize,
        bin_threshold: u8,
        /// Direction of samples in the missing-value bin (older models
        /// without this field sent NaN right).
        #[serde(default)]
        missing_go_to_left: bool,
        left: Box<HistNode>,
        right: Box<HistNode>,
    },
}

impl HistNode {
    fn predict_binned(&self, bins: &[u8]) -> f64 {
        let mut node = self;
        loop {
            match node {
                HistNode::Leaf { value } => return *value,
                HistNode::Internal {
                    feature,
                    bin_threshold,
                    missing_go_to_left,
                    left,
                    right,
                } => {
                    node = if goes_left(bins[*feature], *bin_threshold, *missing_go_to_left) {
                        left
                    } else {
                        right
                    };
                }
            }
        }
    }
}

/// Routing rule shared by training and prediction.
#[inline]
fn goes_left(bin: u8, bin_threshold: u8, missing_go_to_left: bool) -> bool {
    if bin == MISSING_BIN {
        missing_go_to_left
    } else {
        bin <= bin_threshold
    }
}

fn build_hist_tree(
    binned_x: &Array2<u8>,
    gradients: &[f64],
    hessians: &[f64],
    indices: &[usize],
    max_depth: usize,
    settings: &SplitSettings<'_>,
    depth: usize,
) -> HistNode {
    let leaf = |indices: &[usize]| {
        let g: f64 = indices.iter().map(|&i| gradients[i]).sum();
        let h: f64 = indices.iter().map(|&i| hessians[i]).sum();
        HistNode::Leaf {
            value: -g / (h + settings.l2_regularization),
        }
    };

    // Leaf conditions
    if depth >= max_depth || indices.len() < 2 * settings.min_samples_leaf {
        return leaf(indices);
    }

    match find_best_hist_split(binned_x, gradients, hessians, indices, settings) {
        None => leaf(indices),
        Some(s) => {
            let (left_idx, right_idx): (Vec<usize>, Vec<usize>) = indices.iter().partition(|&&i| {
                goes_left(
                    binned_x[[i, s.feature]],
                    s.bin_threshold,
                    s.missing_go_to_left,
                )
            });

            // No missing values seen in training for this feature: send
            // missing values at predict time to the larger child (sklearn).
            let missing_go_to_left = if settings.has_missing[s.feature] {
                s.missing_go_to_left
            } else {
                left_idx.len() > right_idx.len()
            };

            let left = build_hist_tree(
                binned_x,
                gradients,
                hessians,
                &left_idx,
                max_depth,
                settings,
                depth + 1,
            );
            let right = build_hist_tree(
                binned_x,
                gradients,
                hessians,
                &right_idx,
                max_depth,
                settings,
                depth + 1,
            );

            HistNode::Internal {
                feature: s.feature,
                bin_threshold: s.bin_threshold,
                missing_go_to_left,
                left: Box::new(left),
                right: Box::new(right),
            }
        }
    }
}

// ============================================================
// HistGradientBoostingRegressor
// ============================================================

/// Histogram-based gradient boosting regressor.
///
/// Much faster than `GradientBoostingRegressor` for datasets with >1000 samples.
/// Features are discretized into at most `max_bins` (default and maximum 255)
/// bins for O(n) split finding; NaN is handled natively (see module docs).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HistGradientBoostingRegressor {
    pub n_estimators: usize,
    pub learning_rate: f64,
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub l2_regularization: f64,
    pub max_bins: usize,
}

impl HistGradientBoostingRegressor {
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: 6,
            min_samples_leaf: 20,
            l2_regularization: 0.0,
            max_bins: MAX_BINS,
        }
    }

    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }
    pub fn with_max_depth(mut self, d: usize) -> Self {
        self.max_depth = d;
        self
    }
    pub fn with_min_samples_leaf(mut self, m: usize) -> Self {
        self.min_samples_leaf = m;
        self
    }
    pub fn with_l2_regularization(mut self, l2: f64) -> Self {
        self.l2_regularization = l2;
        self
    }
    pub fn with_max_bins(mut self, b: usize) -> Self {
        self.max_bins = b;
        self
    }
}

impl Default for HistGradientBoostingRegressor {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted histogram-based gradient boosting regressor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedHistGradientBoostingRegressor {
    trees: Vec<HistNode>,
    bins: Vec<FeatureBins>,
    baseline: f64,
    learning_rate: f64,
    n_features: usize,
}

impl FittedHistGradientBoostingRegressor {
    pub fn n_estimators(&self) -> usize {
        self.trees.len()
    }
}

impl Fit<f64> for HistGradientBoostingRegressor {
    type Fitted = FittedHistGradientBoostingRegressor;

    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
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

        validate_max_bins(self.max_bins)?;

        let n = x.nrows();
        let (binned_x, bins, has_missing) = compute_bins(x, self.max_bins);
        let settings = SplitSettings {
            bins: &bins,
            has_missing: &has_missing,
            min_samples_leaf: self.min_samples_leaf,
            l2_regularization: self.l2_regularization,
        };

        // Initial prediction = mean(y)
        let baseline: f64 = y.iter().sum::<f64>() / n as f64;
        let mut predictions = vec![baseline; n];
        let mut trees = Vec::with_capacity(self.n_estimators);

        let indices: Vec<usize> = (0..n).collect();

        for _ in 0..self.n_estimators {
            // Squared error: gradient = prediction - y, hessian = 1
            let gradients: Vec<f64> = (0..n).map(|i| predictions[i] - y[i]).collect();
            let hessians = vec![1.0; n];

            let tree = build_hist_tree(
                &binned_x,
                &gradients,
                &hessians,
                &indices,
                self.max_depth,
                &settings,
                0,
            );

            // Update predictions
            for (i, row_bins) in binned_x.rows().into_iter().enumerate() {
                let row_bins = row_bins
                    .as_slice()
                    .expect("binned matrix is standard layout");
                predictions[i] += self.learning_rate * tree.predict_binned(row_bins);
            }

            trees.push(tree);
        }

        Ok(FittedHistGradientBoostingRegressor {
            trees,
            bins,
            baseline,
            learning_rate: self.learning_rate,
            n_features: x.ncols(),
        })
    }
}

impl Predict<f64> for FittedHistGradientBoostingRegressor {
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
// HistGradientBoostingClassifier
// ============================================================

/// Histogram-based gradient boosting classifier.
///
/// Binary classification using log-loss (logistic). Multi-class uses OvR.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HistGradientBoostingClassifier {
    pub n_estimators: usize,
    pub learning_rate: f64,
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub l2_regularization: f64,
    pub max_bins: usize,
}

impl HistGradientBoostingClassifier {
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: 6,
            min_samples_leaf: 20,
            l2_regularization: 0.0,
            max_bins: MAX_BINS,
        }
    }

    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }
    pub fn with_max_depth(mut self, d: usize) -> Self {
        self.max_depth = d;
        self
    }
    pub fn with_min_samples_leaf(mut self, m: usize) -> Self {
        self.min_samples_leaf = m;
        self
    }
    pub fn with_l2_regularization(mut self, l2: f64) -> Self {
        self.l2_regularization = l2;
        self
    }
    pub fn with_max_bins(mut self, b: usize) -> Self {
        self.max_bins = b;
        self
    }
}

impl Default for HistGradientBoostingClassifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted histogram-based gradient boosting classifier.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedHistGradientBoostingClassifier {
    /// For binary: single set of trees. For multi-class: one set per class (OvR).
    tree_sets: Vec<Vec<HistNode>>,
    bins: Vec<FeatureBins>,
    baselines: Vec<f64>,
    classes: Vec<f64>,
    learning_rate: f64,
    n_features: usize,
}

impl FittedHistGradientBoostingClassifier {
    pub fn classes(&self) -> &[f64] {
        &self.classes
    }
    pub fn n_estimators(&self) -> usize {
        self.tree_sets.first().map_or(0, |t| t.len())
    }

    /// Predict class probabilities.
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
            // Binary: sigmoid of raw scores
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
            // Multi-class: softmax of raw scores
            let mut proba = Array2::zeros((n, n_classes));
            for i in 0..n {
                let row: Vec<f64> = (0..self.n_features).map(|j| x[[i, j]]).collect();
                let bins = bin_row(&row, &self.bins);
                let mut scores = vec![0.0; n_classes];
                for (c, tree_set) in self.tree_sets.iter().enumerate() {
                    scores[c] = self.baselines[c];
                    for tree in tree_set {
                        scores[c] += self.learning_rate * tree.predict_binned(&bins);
                    }
                }
                // Softmax
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

impl Fit<f64> for HistGradientBoostingClassifier {
    type Fitted = FittedHistGradientBoostingClassifier;

    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
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

        validate_max_bins(self.max_bins)?;

        let n = x.nrows();
        let (binned_x, bins, has_missing) = compute_bins(x, self.max_bins);
        let settings = SplitSettings {
            bins: &bins,
            has_missing: &has_missing,
            min_samples_leaf: self.min_samples_leaf,
            l2_regularization: self.l2_regularization,
        };

        // Collect classes
        let mut classes: Vec<f64> = y.iter().copied().collect();
        classes.sort_by(f64::total_cmp);
        classes.dedup();
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(RustMlError::InvalidParameter(
                "need at least 2 classes".into(),
            ));
        }

        let indices: Vec<usize> = (0..n).collect();

        if n_classes == 2 {
            // Binary: log-loss with single set of trees
            let pos_class = classes[1];
            let labels: Vec<f64> = y
                .iter()
                .map(|&v| if v == pos_class { 1.0 } else { 0.0 })
                .collect();
            let pos_frac: f64 = labels.iter().sum::<f64>() / n as f64;
            let baseline = (pos_frac / (1.0 - pos_frac + 1e-15)).ln();

            let mut raw_scores = vec![baseline; n];
            let mut trees = Vec::with_capacity(self.n_estimators);

            for _ in 0..self.n_estimators {
                // Log-loss gradients: p - y, hessians: p * (1 - p)
                let gradients: Vec<f64> = (0..n)
                    .map(|i| {
                        let p = 1.0 / (1.0 + (-raw_scores[i]).exp());
                        p - labels[i]
                    })
                    .collect();
                let hessians: Vec<f64> = (0..n)
                    .map(|i| {
                        let p = 1.0 / (1.0 + (-raw_scores[i]).exp());
                        (p * (1.0 - p)).max(1e-12)
                    })
                    .collect();

                let tree = build_hist_tree(
                    &binned_x,
                    &gradients,
                    &hessians,
                    &indices,
                    self.max_depth,
                    &settings,
                    0,
                );

                for (i, row_bins) in binned_x.rows().into_iter().enumerate() {
                    let row_bins = row_bins
                        .as_slice()
                        .expect("binned matrix is standard layout");
                    raw_scores[i] += self.learning_rate * tree.predict_binned(row_bins);
                }
                trees.push(tree);
            }

            Ok(FittedHistGradientBoostingClassifier {
                tree_sets: vec![trees],
                bins,
                baselines: vec![baseline],
                classes,
                learning_rate: self.learning_rate,
                n_features: x.ncols(),
            })
        } else {
            // Multi-class: one-vs-all with softmax
            let mut tree_sets = Vec::with_capacity(n_classes);
            let mut baselines = Vec::with_capacity(n_classes);
            let mut all_raw_scores = vec![vec![0.0; n]; n_classes];

            // Initial baselines: log(class_prior)
            for (c, &cls) in classes.iter().enumerate() {
                let count = y.iter().filter(|&&v| v == cls).count() as f64;
                let prior = count / n as f64;
                let bl = prior.ln().max(-10.0);
                baselines.push(bl);
                all_raw_scores[c] = vec![bl; n];
            }

            // Train trees for each class
            for _ in 0..self.n_estimators {
                // Compute softmax probabilities
                let mut probas = vec![vec![0.0; n_classes]; n];
                for i in 0..n {
                    let max_s = all_raw_scores
                        .iter()
                        .map(|s| s[i])
                        .fold(f64::NEG_INFINITY, f64::max);
                    let exp_sum: f64 = all_raw_scores.iter().map(|s| (s[i] - max_s).exp()).sum();
                    for c in 0..n_classes {
                        probas[i][c] = (all_raw_scores[c][i] - max_s).exp() / exp_sum;
                    }
                }

                let mut round_trees = Vec::with_capacity(n_classes);
                for (c, &cls) in classes.iter().enumerate() {
                    let gradients: Vec<f64> = (0..n)
                        .map(|i| {
                            let label = if y[i] == cls { 1.0 } else { 0.0 };
                            probas[i][c] - label
                        })
                        .collect();
                    let hessians: Vec<f64> = (0..n)
                        .map(|i| (probas[i][c] * (1.0 - probas[i][c])).max(1e-12))
                        .collect();

                    let tree = build_hist_tree(
                        &binned_x,
                        &gradients,
                        &hessians,
                        &indices,
                        self.max_depth,
                        &settings,
                        0,
                    );

                    for (i, row_bins) in binned_x.rows().into_iter().enumerate() {
                        let row_bins = row_bins
                            .as_slice()
                            .expect("binned matrix is standard layout");
                        all_raw_scores[c][i] += self.learning_rate * tree.predict_binned(row_bins);
                    }
                    round_trees.push(tree);
                }

                // Distribute trees to per-class sets
                if tree_sets.is_empty() {
                    for tree in round_trees {
                        tree_sets.push(vec![tree]);
                    }
                } else {
                    for (c, tree) in round_trees.into_iter().enumerate() {
                        tree_sets[c].push(tree);
                    }
                }
            }

            Ok(FittedHistGradientBoostingClassifier {
                tree_sets,
                bins,
                baselines,
                classes,
                learning_rate: self.learning_rate,
                n_features: x.ncols(),
            })
        }
    }
}

impl Predict<f64> for FittedHistGradientBoostingClassifier {
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
    fn test_hist_gb_regressor_basic() {
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

        let model = HistGradientBoostingRegressor::new()
            .with_n_estimators(50)
            .with_max_depth(3)
            .with_min_samples_leaf(1);

        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        for (p, t) in preds.iter().zip(y.iter()) {
            assert_abs_diff_eq!(*p, *t, epsilon = 2.0);
        }
    }

    #[test]
    fn test_hist_gb_regressor_n_estimators() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let fitted = HistGradientBoostingRegressor::new()
            .with_n_estimators(10)
            .with_min_samples_leaf(1)
            .fit(&x, &y)
            .unwrap();

        assert_eq!(fitted.n_estimators(), 10);
    }

    #[test]
    fn test_hist_gb_classifier_binary() {
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

        let model = HistGradientBoostingClassifier::new()
            .with_n_estimators(20)
            .with_max_depth(3)
            .with_min_samples_leaf(1);

        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        let correct: usize = preds.iter().zip(y.iter()).filter(|(&p, &t)| p == t).count();
        assert!(
            correct >= 8,
            "should classify most correctly, got {}/10",
            correct
        );
    }

    #[test]
    fn test_hist_gb_classifier_predict_proba() {
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [10.0],
            [11.0],
            [12.0],
            [13.0],
            [14.0]
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let fitted = HistGradientBoostingClassifier::new()
            .with_n_estimators(20)
            .with_min_samples_leaf(1)
            .fit(&x, &y)
            .unwrap();

        let proba = fitted.predict_proba(&x).unwrap();
        assert_eq!(proba.ncols(), 2);

        for i in 0..x.nrows() {
            let row_sum: f64 = (0..proba.ncols()).map(|c| proba[[i, c]]).sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_hist_gb_classifier_multiclass() {
        let x = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [5.0, 5.0],
            [6.0, 5.0],
            [7.0, 5.0],
            [0.0, 10.0],
            [1.0, 10.0],
            [2.0, 10.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];

        let fitted = HistGradientBoostingClassifier::new()
            .with_n_estimators(30)
            .with_max_depth(3)
            .with_min_samples_leaf(1)
            .fit(&x, &y)
            .unwrap();

        assert_eq!(fitted.classes(), &[0.0, 1.0, 2.0]);

        let proba = fitted.predict_proba(&x).unwrap();
        assert_eq!(proba.ncols(), 3);
    }

    #[test]
    fn test_hist_gb_regressor_shape_mismatch() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![1.0, 2.0];
        assert!(HistGradientBoostingRegressor::new().fit(&x, &y).is_err());
    }

    #[test]
    fn test_hist_gb_regressor_empty() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<f64>::zeros(0);
        assert!(HistGradientBoostingRegressor::new().fit(&x, &y).is_err());
    }

    #[test]
    fn test_hist_gb_classifier_single_class() {
        let x = array![[1.0], [2.0], [3.0]];
        let y = array![0.0, 0.0, 0.0];
        assert!(HistGradientBoostingClassifier::new().fit(&x, &y).is_err());
    }

    #[test]
    fn test_hist_gb_predict_shape_mismatch() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let fitted = HistGradientBoostingClassifier::new()
            .with_n_estimators(5)
            .with_min_samples_leaf(1)
            .fit(&x, &y)
            .unwrap();

        let x_bad = array![[1.0]];
        assert!(fitted.predict(&x_bad).is_err());
    }
}

#[cfg(test)]
mod nan_regression_tests {
    use super::*;
    use ndarray::array;

    /// #42/#47: binning used `partial_cmp().unwrap()` and panicked on NaN.
    /// NaN now gets the dedicated missing-value bin.
    #[test]
    fn binning_with_nan_uses_missing_bin() {
        let x = array![
            [1.0, f64::NAN],
            [2.0, 0.5],
            [f64::NAN, 0.7],
            [4.0, f64::NAN]
        ];
        let (binned, bins, has_missing) = compute_bins(&x, MAX_BINS);
        assert_eq!(has_missing, vec![true, true]);
        assert!(bins[0].edges.iter().all(|e| e.is_finite()));
        assert_eq!(binned[[2, 0]], MISSING_BIN);
        assert_eq!(binned[[0, 1]], MISSING_BIN);
        assert!(binned[[0, 0]] < MISSING_BIN);
        assert_eq!(bin_row(&[f64::NAN, 0.5], &bins)[0], MISSING_BIN);

        let all_nan = array![[f64::NAN], [f64::NAN]];
        let (b, bins, has_missing) = compute_bins(&all_nan, MAX_BINS);
        assert!(bins[0].edges.is_empty());
        assert!(has_missing[0]);
        assert_eq!(b[[0, 0]], MISSING_BIN);
    }

    /// #47: `partition_point(..) as u8` wrapped for more than 256 bins.
    #[test]
    fn max_bins_validated_and_bins_never_wrap() {
        let x = Array2::from_shape_fn((600, 1), |(i, _)| i as f64);
        let y = Array1::from_shape_fn(600, |i| i as f64);
        for bad in [0usize, 1, 256, 1000] {
            let err = HistGradientBoostingRegressor::new()
                .with_max_bins(bad)
                .fit(&x, &y);
            assert!(
                matches!(err, Err(RustMlError::InvalidParameter(_))),
                "{bad}"
            );
            let err = HistGradientBoostingClassifier::new()
                .with_max_bins(bad)
                .fit(&x, &y.mapv(|v| (v > 300.0) as u8 as f64));
            assert!(
                matches!(err, Err(RustMlError::InvalidParameter(_))),
                "{bad}"
            );
        }
        let (binned, bins, _) = compute_bins(&x, MAX_BINS);
        assert_eq!(bins[0].n_bins_non_missing(), MAX_BINS);
        // Monotone, within range, never the missing bin.
        let col: Vec<u8> = binned.column(0).to_vec();
        assert!(col.windows(2).all(|w| w[0] <= w[1]));
        assert_eq!(*col.last().unwrap(), (MAX_BINS - 1) as u8);
        // Values far above the last edge are clamped below MISSING_BIN.
        assert_eq!(value_to_bin(1e9, &bins[0].edges), (MAX_BINS - 1) as u8);
        let edges: Vec<f64> = (0..400).map(|v| v as f64).collect();
        assert_eq!(value_to_bin(1e9, &edges), MISSING_BIN - 1);
    }

    /// Missing values carry signal: y is high exactly when x0 is NaN.
    /// The tree must learn to route NaN to its own (right or left) child.
    #[test]
    fn learns_missing_direction() {
        let n = 60;
        let mut x = Array2::<f64>::zeros((n, 1));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            if i % 3 == 0 {
                x[[i, 0]] = f64::NAN;
                y[i] = 10.0;
            } else {
                x[[i, 0]] = i as f64;
                y[i] = 0.0;
            }
        }
        let fitted = HistGradientBoostingRegressor::new()
            .with_n_estimators(50)
            .with_learning_rate(0.3)
            .with_max_depth(1)
            .with_min_samples_leaf(1)
            .fit(&x, &y)
            .unwrap();
        let preds = fitted.predict(&array![[f64::NAN], [1.0], [59.0]]).unwrap();
        assert!((preds[0] - 10.0).abs() < 0.1, "{preds}");
        assert!(preds[1].abs() < 0.1, "{preds}");
        assert!(preds[2].abs() < 0.1, "{preds}");

        // Missing values that resemble *small* x0 must learn "missing left".
        let mut x2 = x.clone();
        let mut y2 = y.clone();
        for i in 0..n {
            if !x2[[i, 0]].is_nan() {
                y2[i] = if x2[[i, 0]] < 30.0 { 10.0 } else { 0.0 };
            }
        }
        x2[[1, 0]] = 1.0;
        let fitted = HistGradientBoostingRegressor::new()
            .with_n_estimators(50)
            .with_learning_rate(0.3)
            .with_max_depth(1)
            .with_min_samples_leaf(1)
            .fit(&x2, &y2)
            .unwrap();
        let tree = &fitted.trees[0];
        match tree {
            HistNode::Internal {
                missing_go_to_left, ..
            } => assert!(*missing_go_to_left),
            HistNode::Leaf { .. } => panic!("expected a split"),
        }
        let p = fitted.predict(&array![[f64::NAN], [5.0], [50.0]]).unwrap();
        assert!(
            (p[0] - 10.0).abs() < 0.1 && (p[1] - 10.0).abs() < 0.1,
            "{p}"
        );
        assert!(p[2].abs() < 0.1, "{p}");
    }

    /// No NaN during training: NaN at predict time follows the child with
    /// more training samples (sklearn).
    #[test]
    fn unseen_missing_goes_to_larger_child() {
        let x = Array2::from_shape_fn((20, 1), |(i, _)| i as f64);
        let y = Array1::from_shape_fn(20, |i| if i < 15 { 1.0 } else { 5.0 });
        let fitted = HistGradientBoostingRegressor::new()
            .with_n_estimators(1)
            .with_learning_rate(1.0)
            .with_max_depth(1)
            .with_min_samples_leaf(1)
            .fit(&x, &y)
            .unwrap();
        let p = fitted.predict(&array![[f64::NAN], [0.0]]).unwrap();
        assert_eq!(p[0], p[1]);
    }

    #[test]
    fn fit_predict_with_nan_does_not_panic() {
        let x = array![
            [1.0],
            [2.0],
            [f64::NAN],
            [4.0],
            [5.0],
            [6.0],
            [f64::NAN],
            [8.0]
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let fitted = HistGradientBoostingRegressor::new().fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert!(preds.iter().all(|p| p.is_finite()));
    }

    /// Same LCG as the Python script used to produce the sklearn numbers.
    struct Lcg(u64);
    impl Lcg {
        fn u(&mut self) -> f64 {
            self.0 = (1_664_525 * self.0 + 1_013_904_223) % (1 << 32);
            self.0 as f64 / (1u64 << 32) as f64
        }
    }

    /// y = x0^2 + 2 x1 + noise, x0 missing (25%) with an "x0 = 2.5" effect.
    fn make_nan_data(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
        let mut r = Lcg(seed);
        let mut x = Array2::<f64>::zeros((n, 3));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let x0 = 4.0 * r.u() - 2.0;
            let x1 = 4.0 * r.u() - 2.0;
            let x2 = 4.0 * r.u() - 2.0;
            let miss = r.u() < 0.25;
            let noise = r.u() - 0.5;
            let x0e = if miss { 2.5 } else { x0 };
            y[i] = x0e * x0e + 2.0 * x1 + 0.5 * noise;
            x[[i, 0]] = if miss { f64::NAN } else { x0 };
            x[[i, 1]] = x1;
            x[[i, 2]] = x2;
        }
        (x, y)
    }

    /// Compared with sklearn 1.8.0 `HistGradientBoosting{Regressor,Classifier}(
    /// max_iter=100, learning_rate=0.1, max_depth=3, max_leaf_nodes=None,
    /// min_samples_leaf=20, l2_regularization=0, early_stopping=False)`:
    /// regressor test MSE 0.0920 (0.323 when NaN is imputed by 0, i.e.
    /// without learning a missing direction); classifier test log-loss 0.1155.
    #[test]
    fn nan_data_loss_close_to_sklearn() {
        let (xtr, ytr) = make_nan_data(400, 1);
        let (xte, yte) = make_nan_data(200, 2);
        let fitted = HistGradientBoostingRegressor::new()
            .with_n_estimators(100)
            .with_learning_rate(0.1)
            .with_max_depth(3)
            .with_min_samples_leaf(20)
            .fit(&xtr, &ytr)
            .unwrap();
        let pred = fitted.predict(&xte).unwrap();
        let mse = (&pred - &yte).mapv(|v| v * v).mean().unwrap();
        let sklearn_mse = 0.09196874903149442;
        assert!(
            mse < sklearn_mse * 1.25,
            "mse {mse} vs sklearn {sklearn_mse}"
        );

        let ctr = ytr.mapv(|v| (v > 1.5) as u8 as f64);
        let cte = yte.mapv(|v| (v > 1.5) as u8 as f64);
        let fitted = HistGradientBoostingClassifier::new()
            .with_n_estimators(100)
            .with_learning_rate(0.1)
            .with_max_depth(3)
            .with_min_samples_leaf(20)
            .fit(&xtr, &ctr)
            .unwrap();
        let proba = fitted.predict_proba(&xte).unwrap();
        let logloss = cte
            .iter()
            .enumerate()
            .map(|(i, &t)| -(proba[[i, t as usize]].clamp(1e-15, 1.0)).ln())
            .sum::<f64>()
            / cte.len() as f64;
        let sklearn_logloss = 0.11553833161713371;
        assert!(
            logloss < sklearn_logloss * 1.25,
            "log-loss {logloss} vs sklearn {sklearn_logloss}"
        );
    }
}
