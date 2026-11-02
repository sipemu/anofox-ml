//! Isolation Forest for outlier detection.
//!
//! Mirrors `sklearn.ensemble.IsolationForest`. Each tree is built on a
//! sub-sample of the data by recursively splitting on a random feature at a
//! random threshold until each point is isolated (or `max_depth` is hit).
//! The anomaly score for a point is the average path length across the
//! forest, normalised by the expected length on a sub-sample of that size.

use anofox_ml_core::{FitUnsupervised, Predict, Result, RustMlError};
use ndarray::{Array1, ArrayView2, AsArray, Ix2};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct INode {
    feature: usize,
    threshold: f64,
    left: Option<usize>,
    right: Option<usize>,
    size: usize, // number of training samples at this leaf
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ITree {
    nodes: Vec<INode>, // nodes[0] is the root
    max_depth: usize,
}

/// Average path length of an unsuccessful BST search over `n` points
/// (sklearn `_average_path_length`): 0 for n <= 1, 1 for n = 2, else
/// `2 (ln(n - 1) + γ) - 2 (n - 1) / n`.
fn c_factor(n: usize) -> f64 {
    match n {
        0 | 1 => 0.0,
        2 => 1.0,
        _ => {
            let n = n as f64;
            2.0 * ((n - 1.0).ln() + 0.577_215_664_901_532_9) - 2.0 * (n - 1.0) / n
        }
    }
}

fn build_tree(
    x: ArrayView2<f64>,
    indices: &[usize],
    depth: usize,
    max_depth: usize,
    rng: &mut StdRng,
    nodes: &mut Vec<INode>,
) -> usize {
    let me = nodes.len();
    nodes.push(INode {
        feature: 0,
        threshold: 0.0,
        left: None,
        right: None,
        size: indices.len(),
    });

    if depth >= max_depth || indices.len() <= 1 {
        return me;
    }
    let d = x.ncols();
    // Try features in random order; first one with non-zero range wins.
    let mut features: Vec<usize> = (0..d).collect();
    features.shuffle(rng);
    let mut chosen_feature = None;
    let mut lo = 0.0;
    let mut hi = 0.0;
    for &f in &features {
        let mut mn = f64::INFINITY;
        let mut mx = f64::NEG_INFINITY;
        for &i in indices {
            let v = x[[i, f]];
            if v < mn {
                mn = v;
            }
            if v > mx {
                mx = v;
            }
        }
        if mx > mn {
            chosen_feature = Some(f);
            lo = mn;
            hi = mx;
            break;
        }
    }
    let f = match chosen_feature {
        Some(f) => f,
        None => return me,
    };
    let t = lo + rng.gen::<f64>() * (hi - lo);

    let (li, ri): (Vec<usize>, Vec<usize>) = indices.iter().partition(|&&i| x[[i, f]] < t);
    nodes[me].feature = f;
    nodes[me].threshold = t;

    if li.is_empty() || ri.is_empty() {
        // Degenerate split — leaf.
        return me;
    }
    let left = build_tree(x, &li, depth + 1, max_depth, rng, nodes);
    let right = build_tree(x, &ri, depth + 1, max_depth, rng, nodes);
    nodes[me].left = Some(left);
    nodes[me].right = Some(right);
    me
}

fn path_length(tree: &ITree, x: &Array1<f64>) -> f64 {
    let mut node = 0usize;
    let mut depth = 0.0;
    loop {
        let n = &tree.nodes[node];
        match (n.left, n.right) {
            (Some(l), Some(r)) => {
                node = if x[n.feature] < n.threshold { l } else { r };
                depth += 1.0;
            }
            _ => {
                // Reached leaf. Add c(size) for unisolated points.
                if n.size > 1 {
                    depth += c_factor(n.size);
                }
                return depth;
            }
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IsolationForest {
    pub n_estimators: usize,
    pub max_samples: usize,
    /// Expected outlier fraction in `(0, 0.5]`; `None` is sklearn's `'auto'`
    /// (offset -0.5).
    #[serde(default)]
    pub contamination: Option<f64>,
    pub seed: u64,
}

impl IsolationForest {
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            max_samples: 256,
            contamination: None,
            seed: 0,
        }
    }
    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }
    pub fn with_max_samples(mut self, n: usize) -> Self {
        self.max_samples = n;
        self
    }
    pub fn with_contamination(mut self, c: f64) -> Self {
        self.contamination = Some(c);
        self
    }
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }
}

impl Default for IsolationForest {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedIsolationForest {
    trees: Vec<ITree>,
    subsample_size: usize,
    /// sklearn's `offset_`: points with `score_samples < threshold` are
    /// outliers (`decision_function = score_samples - threshold`).
    pub threshold: f64,
    n_features: usize,
}

impl FittedIsolationForest {
    /// Higher score → more normal. sklearn returns `-anomaly_score`, where
    /// `anomaly_score = 2^(-E[h(x)] / c(n))` (Liu et al. 2008). We follow
    /// sklearn's sign convention.
    pub fn score_samples<'a>(&self, x: impl AsArray<'a, f64, Ix2>) -> Array1<f64> {
        let x: ArrayView2<f64> = x.into();
        let n = x.nrows();
        let c = c_factor(self.subsample_size);
        let n_trees = self.trees.len() as f64;
        // Scoring is per-row independent; par_iter over rows.
        let scores: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|i| {
                let row = x.row(i).to_owned();
                let total: f64 = self.trees.iter().map(|t| path_length(t, &row)).sum();
                let avg_h = total / n_trees;
                -(-avg_h / c).exp2()
            })
            .collect();
        Array1::from_vec(scores)
    }
}

impl FittedIsolationForest {
    /// `score_samples - offset` (sklearn `decision_function`): negative for
    /// outliers.
    pub fn decision_function<'a>(&self, x: impl AsArray<'a, f64, Ix2>) -> Array1<f64> {
        let x: ArrayView2<f64> = x.into();
        self.score_samples(x.view()) - self.threshold
    }
}

impl FitUnsupervised<f64> for IsolationForest {
    type Fitted = FittedIsolationForest;

    fn fit_view(&self, x: ArrayView2<f64>) -> Result<Self::Fitted> {
        if x.nrows() == 0 {
            return Err(RustMlError::EmptyInput("empty input".into()));
        }
        let n = x.nrows();
        let subsample = self.max_samples.min(n);
        let max_depth = ((subsample as f64).log2().ceil() as usize).max(1);

        // Trees are independent; build in parallel. Each gets a deterministic
        // per-tree seed derived from `self.seed` so results are reproducible.
        let trees: Vec<ITree> = (0..self.n_estimators)
            .into_par_iter()
            .map(|t| {
                let seed = self
                    .seed
                    .wrapping_add((t as u64).wrapping_mul(0x9E3779B97F4A7C15));
                let mut rng = StdRng::seed_from_u64(seed);
                let mut idx: Vec<usize> = (0..n).collect();
                idx.shuffle(&mut rng);
                idx.truncate(subsample);
                let mut nodes = Vec::new();
                build_tree(x, &idx, 0, max_depth, &mut rng, &mut nodes);
                ITree { nodes, max_depth }
            })
            .collect();

        let mut fitted = FittedIsolationForest {
            trees,
            subsample_size: subsample,
            threshold: 0.0,
            n_features: x.ncols(),
        };
        // sklearn's offset_: -0.5 for contamination='auto', else the
        // 100 · contamination percentile (linear interpolation) of the
        // training scores.
        fitted.threshold = match self.contamination {
            None => -0.5,
            Some(c) => {
                if !(c > 0.0 && c <= 0.5) {
                    return Err(RustMlError::InvalidParameter(format!(
                        "contamination must be in (0, 0.5], got {c}"
                    )));
                }
                let mut sorted: Vec<f64> = fitted.score_samples(x).to_vec();
                sorted.sort_by(f64::total_cmp);
                let pos = c * (n as f64 - 1.0);
                let lo = pos.floor() as usize;
                let hi = (lo + 1).min(n - 1);
                sorted[lo] + (pos - lo as f64) * (sorted[hi] - sorted[lo])
            }
        };

        Ok(fitted)
    }
}

impl Predict<f64> for FittedIsolationForest {
    /// Return 1.0 for inliers and -1.0 for outliers (sklearn convention).
    fn predict_view(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        let scores = self.score_samples(x);
        Ok(scores.mapv(|s| if s >= self.threshold { 1.0 } else { -1.0 }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn test_isolation_forest_flags_outlier() {
        // 50 inliers near origin, 2 wild outliers far away.
        let mut data = Vec::new();
        for i in 0..50 {
            data.push((i as f64) * 0.01 - 0.25);
            data.push((i as f64) * -0.02 + 0.5);
        }
        // Two wild outliers
        data.extend([100.0, 100.0, -100.0, -100.0]);
        let x = Array2::from_shape_vec((52, 2), data).unwrap();

        let fitted = IsolationForest::new()
            .with_n_estimators(50)
            .with_max_samples(32)
            .with_contamination(0.04)
            .with_seed(1)
            .fit(&x)
            .unwrap();
        let preds = fitted.predict(&x).unwrap();

        // The last two points should be outliers.
        assert_eq!(preds[50], -1.0);
        assert_eq!(preds[51], -1.0);
        let _ = array![1.0_f64];
    }

    #[test]
    fn test_average_path_length_matches_sklearn() {
        // sklearn _average_path_length([0, 1, 2, 3, 256])
        assert_eq!(c_factor(0), 0.0);
        assert_eq!(c_factor(1), 0.0);
        assert_eq!(c_factor(2), 1.0);
        assert!((c_factor(3) - 1.207_392_357_589_623).abs() < 1e-12);
        assert!((c_factor(256) - 10.244_770_920_119_917).abs() < 1e-12);
    }
}
