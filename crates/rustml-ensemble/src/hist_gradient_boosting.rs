//! Histogram-based gradient boosting (classifier and regressor).
//!
//! Much faster than classical gradient boosting for medium-to-large datasets.
//! Features are binned into 256 discrete bins, enabling O(n) split finding
//! via histogram accumulation instead of O(n log n) sorting.

use ndarray::{Array1, Array2};
use rustml_core::{Fit, Predict, Result, RustMlError};

const MAX_BINS: usize = 256;

// ============================================================
// Binning
// ============================================================

/// Bin edges for one feature.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct FeatureBins {
    /// Sorted thresholds. Value v maps to bin i if edges[i-1] < v <= edges[i].
    edges: Vec<f64>,
}

/// Bin all features into u8 indices.
fn compute_bins(x: &Array2<f64>, max_bins: usize) -> (Array2<u8>, Vec<FeatureBins>) {
    let n = x.nrows();
    let p = x.ncols();
    let mut binned = Array2::zeros((n, p));
    let mut all_bins = Vec::with_capacity(p);

    for j in 0..p {
        let mut col: Vec<f64> = (0..n).map(|i| x[[i, j]]).collect();
        col.sort_by(|a, b| a.partial_cmp(b).unwrap());
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

        // Map values to bins
        for i in 0..n {
            let v = x[[i, j]];
            let bin = edges.partition_point(|&e| e < v) as u8;
            binned[[i, j]] = bin;
        }

        all_bins.push(FeatureBins { edges });
    }

    (binned, all_bins)
}

/// Map a new data point's features to bin indices.
fn bin_row(row: &[f64], all_bins: &[FeatureBins]) -> Vec<u8> {
    row.iter()
        .zip(all_bins.iter())
        .map(|(&v, bins)| bins.edges.partition_point(|&e| e < v) as u8)
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
}

/// Find the best split across all features using histograms.
fn find_best_hist_split(
    binned_x: &Array2<u8>,
    gradients: &[f64],
    hessians: &[f64],
    indices: &[usize],
    n_features: usize,
    min_samples_leaf: usize,
    l2_regularization: f64,
) -> Option<HistSplit> {
    let n_bins = MAX_BINS;
    let mut best: Option<HistSplit> = None;
    let mut hist = Histogram::new(n_bins);

    // Total gradient/hessian for this node
    let total_grad: f64 = indices.iter().map(|&i| gradients[i]).sum();
    let total_hess: f64 = indices.iter().map(|&i| hessians[i]).sum();
    let total_count = indices.len();

    for feat in 0..n_features {
        hist.reset();

        // Build histogram for this feature
        for &i in indices {
            let bin = binned_x[[i, feat]] as usize;
            hist.grad_sum[bin] += gradients[i];
            hist.hess_sum[bin] += hessians[i];
            hist.count[bin] += 1;
        }

        // Scan bins to find best split
        let mut left_grad = 0.0;
        let mut left_hess = 0.0;
        let mut left_count: usize = 0;

        for bin in 0..(n_bins - 1) {
            left_grad += hist.grad_sum[bin];
            left_hess += hist.hess_sum[bin];
            left_count += hist.count[bin] as usize;

            if left_count < min_samples_leaf {
                continue;
            }
            let right_count = total_count - left_count;
            if right_count < min_samples_leaf {
                break;
            }

            let right_grad = total_grad - left_grad;
            let right_hess = total_hess - left_hess;

            // Gain = left_term + right_term - parent_term
            // where term = G^2 / (H + lambda)
            let reg = l2_regularization;
            let parent_term = total_grad * total_grad / (total_hess + reg);
            let left_term = left_grad * left_grad / (left_hess + reg);
            let right_term = right_grad * right_grad / (right_hess + reg);
            let gain = 0.5 * (left_term + right_term - parent_term);

            if gain > best.as_ref().map_or(0.0, |b| b.gain) {
                best = Some(HistSplit {
                    feature: feat,
                    bin_threshold: bin as u8,
                    gain,
                    left_value: -left_grad / (left_hess + reg),
                    right_value: -right_grad / (right_hess + reg),
                    left_count,
                    right_count,
                });
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
        left: Box<HistNode>,
        right: Box<HistNode>,
    },
}

impl HistNode {
    fn predict_binned(&self, bins: &[u8]) -> f64 {
        match self {
            HistNode::Leaf { value } => *value,
            HistNode::Internal {
                feature,
                bin_threshold,
                left,
                right,
            } => {
                if bins[*feature] <= *bin_threshold {
                    left.predict_binned(bins)
                } else {
                    right.predict_binned(bins)
                }
            }
        }
    }
}

fn build_hist_tree(
    binned_x: &Array2<u8>,
    gradients: &[f64],
    hessians: &[f64],
    indices: &[usize],
    max_depth: usize,
    min_samples_leaf: usize,
    l2_regularization: f64,
    depth: usize,
) -> HistNode {
    // Leaf conditions
    if depth >= max_depth || indices.len() < 2 * min_samples_leaf {
        let g: f64 = indices.iter().map(|&i| gradients[i]).sum();
        let h: f64 = indices.iter().map(|&i| hessians[i]).sum();
        return HistNode::Leaf {
            value: -g / (h + l2_regularization),
        };
    }

    let n_features = binned_x.ncols();
    let split = find_best_hist_split(
        binned_x,
        gradients,
        hessians,
        indices,
        n_features,
        min_samples_leaf,
        l2_regularization,
    );

    match split {
        None => {
            let g: f64 = indices.iter().map(|&i| gradients[i]).sum();
            let h: f64 = indices.iter().map(|&i| hessians[i]).sum();
            HistNode::Leaf {
                value: -g / (h + l2_regularization),
            }
        }
        Some(s) => {
            let (left_idx, right_idx): (Vec<usize>, Vec<usize>) = indices
                .iter()
                .partition(|&&i| binned_x[[i, s.feature]] <= s.bin_threshold);

            let left = build_hist_tree(
                binned_x,
                gradients,
                hessians,
                &left_idx,
                max_depth,
                min_samples_leaf,
                l2_regularization,
                depth + 1,
            );
            let right = build_hist_tree(
                binned_x,
                gradients,
                hessians,
                &right_idx,
                max_depth,
                min_samples_leaf,
                l2_regularization,
                depth + 1,
            );

            HistNode::Internal {
                feature: s.feature,
                bin_threshold: s.bin_threshold,
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
/// Features are discretized into 256 bins for O(n) split finding.
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

        let n = x.nrows();
        let (binned_x, bins) = compute_bins(x, self.max_bins);

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
                self.min_samples_leaf,
                self.l2_regularization,
                0,
            );

            // Update predictions
            for i in 0..n {
                let row_bins: Vec<u8> = (0..x.ncols()).map(|j| binned_x[[i, j]]).collect();
                predictions[i] += self.learning_rate * tree.predict_binned(&row_bins);
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

        let n = x.nrows();
        let (binned_x, bins) = compute_bins(x, self.max_bins);

        // Collect classes
        let mut classes: Vec<f64> = y.iter().copied().collect();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
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
                    self.min_samples_leaf,
                    self.l2_regularization,
                    0,
                );

                for i in 0..n {
                    let row_bins: Vec<u8> = (0..x.ncols()).map(|j| binned_x[[i, j]]).collect();
                    raw_scores[i] += self.learning_rate * tree.predict_binned(&row_bins);
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
                        self.min_samples_leaf,
                        self.l2_regularization,
                        0,
                    );

                    for i in 0..n {
                        let row_bins: Vec<u8> = (0..x.ncols()).map(|j| binned_x[[i, j]]).collect();
                        all_raw_scores[c][i] += self.learning_rate * tree.predict_binned(&row_bins);
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
