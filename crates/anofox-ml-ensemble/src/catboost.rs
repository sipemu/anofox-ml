//! CatBoost-style gradient boosting on oblivious (symmetric) trees (#127).
//!
//! Every tree splits all nodes of a level on the same feature and border,
//! so a tree of depth `d` is `d` (feature, border) pairs and `2^d` leaf
//! values, and prediction is a bit lookup. The numerics follow CatBoost
//! 1.2: its GreedyLogSum quantization (float32 features, `nan_mode=Min`),
//! the Cosine or L2 split score, Newton or Gradient leaves with CatBoost's
//! scaled L2 regularization, the removal of redundant splits, and its
//! plain boosting scheme. With CatBoost's randomness switched off
//! (`random_strength=0`, no bootstrap, `has_time=True`) the models agree
//! with CatBoost's.

use anofox_ml_core::{Fit, FitWeighted, Predict, Result, RustMlError};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, AsArray, Ix2};

// ============================================================
// Quantization (CatBoost's GreedyLogSum binarizer)
// ============================================================

/// One bin of CatBoost's greedy binarizer (`TFeatureBin` with the
/// `MaxSumLog` penalty) over the sorted values `[start, end)`.
#[derive(Clone, Copy)]
struct GreedyBin {
    start: usize,
    end: usize,
    best_split: usize,
    best_score: f64,
}

impl GreedyBin {
    fn new(start: usize, end: usize, values: &[f32]) -> Self {
        let mut bin = Self {
            start,
            end,
            best_split: start,
            best_score: 0.0,
        };
        bin.update(values);
        bin
    }

    /// `log(left) + log(right) - log(all)` (CatBoost's `CalcSplitScore`).
    fn split_score(&self, pos: usize) -> f64 {
        if pos == self.start || pos == self.end {
            return f64::NEG_INFINITY;
        }
        let ln = |w: usize| (w as f64 + 1e-8).ln();
        ln(pos - self.start) + ln(self.end - pos) - ln(self.end - self.start)
    }

    /// The best split around the middle value (`UpdateBestSplitProperties`).
    fn update(&mut self, values: &[f32]) {
        let mid = self.start + (self.end - self.start) / 2;
        let mid_value = values[mid];
        let lb = self.start + values[self.start..mid].partition_point(|&v| v < mid_value);
        let ub = mid + values[mid..self.end].partition_point(|&v| v <= mid_value);
        let (left, right) = (self.split_score(lb), self.split_score(ub));
        self.best_split = if left >= right { lb } else { ub };
        self.best_score = if self.best_split == lb { left } else { right };
    }

    fn can_split(&self) -> bool {
        self.start != self.best_split && self.end != self.best_split
    }

    /// Split off the left part; `self` keeps the right one.
    fn split(&mut self, values: &[f32]) -> Self {
        let left = Self::new(self.start, self.best_split, values);
        self.start = self.best_split;
        self.update(values);
        left
    }

    /// The border below the bin: the midpoint (in float32) of its first
    /// value and the one before it.
    fn left_border(&self, values: &[f32]) -> f32 {
        if self.start == 0 {
            return values[0];
        }
        let mut b = 0.5f32 * values[self.start - 1];
        b += 0.5f32 * values[self.start];
        b
    }
}

/// `std::priority_queue` (a max-heap by score) as libc++ implements it
/// (CatBoost is built with clang and libc++): its sift order decides ties
/// as CatBoost's binarizer does.
struct StdPriorityQueue(Vec<GreedyBin>);

impl StdPriorityQueue {
    fn less(a: &GreedyBin, b: &GreedyBin) -> bool {
        a.best_score < b.best_score
    }

    /// libc++'s `__sift_up` of the last of the first `len` elements.
    fn sift_up(v: &mut [GreedyBin], len: usize) {
        if len <= 1 {
            return;
        }
        let mut parent = (len - 2) / 2;
        let mut last = len - 1;
        if Self::less(&v[parent], &v[last]) {
            let t = v[last];
            loop {
                v[last] = v[parent];
                last = parent;
                if parent == 0 {
                    break;
                }
                parent = (parent - 1) / 2;
                if !Self::less(&v[parent], &t) {
                    break;
                }
            }
            v[last] = t;
        }
    }

    fn push(&mut self, value: GreedyBin) {
        self.0.push(value);
        let len = self.0.len();
        Self::sift_up(&mut self.0, len);
    }

    fn top(&self) -> &GreedyBin {
        &self.0[0]
    }

    /// libc++'s `__pop_heap` (Floyd's sift-down, then a sift-up) and
    /// `pop_back`.
    fn pop(&mut self) -> GreedyBin {
        let len = self.0.len();
        let top = self.0[0];
        if len > 1 {
            let v = &mut self.0;
            // __floyd_sift_down
            let mut hole = 0usize;
            let mut child = 0usize;
            loop {
                let mut child_i = hole + child + 1;
                child = 2 * child + 1;
                if child + 1 < len && Self::less(&v[child_i], &v[child_i + 1]) {
                    child_i += 1;
                    child += 1;
                }
                v[hole] = v[child_i];
                hole = child_i;
                if child > (len - 2) / 2 {
                    break;
                }
            }
            let last = len - 1;
            if hole == last {
                v[hole] = top;
            } else {
                v[hole] = v[last];
                v[last] = top;
                Self::sift_up(v, hole + 1);
            }
        }
        self.0.pop();
        top
    }
}

/// CatBoost's GreedyLogSum borders of the sorted values (duplicates kept):
/// at most `max_borders`, ascending.
fn greedy_log_sum_borders(values: &[f32], max_borders: usize) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut heap = StdPriorityQueue(Vec::new());
    heap.push(GreedyBin::new(0, values.len(), values));
    while heap.0.len() <= max_borders && heap.top().can_split() {
        let mut top = heap.pop();
        let left = top.split(values);
        heap.push(left);
        heap.push(top);
    }
    let mut borders: Vec<f32> = heap
        .0
        .iter()
        .filter(|b| b.start != 0)
        .map(|b| b.left_border(values))
        .map(|b| if b == 0.0 { 0.0 } else { b }) // -0 -> 0
        .collect();
    borders.sort_by(f32::total_cmp);
    borders.dedup();
    borders
}

/// Borders of one feature (`border_count` at most). With missing values
/// (`nan_mode=Min`) one border is `-f32::MAX`, below every value, so that
/// a split can separate them.
fn feature_borders(col: ArrayView1<f64>, border_count: usize) -> Vec<f32> {
    let mut values: Vec<f32> = col
        .iter()
        .filter(|v| !v.is_nan())
        .map(|&v| v as f32)
        .collect();
    let has_nan = values.len() < col.len();
    values.sort_by(f32::total_cmp);
    let budget = border_count.saturating_sub(usize::from(has_nan));
    let mut borders = greedy_log_sum_borders(&values, budget);
    if has_nan {
        borders.insert(0, -f32::MAX);
    }
    borders
}

/// Bucket of a value: the number of borders below it (NaN: 0).
#[inline]
fn bucket(v: f64, borders: &[f32]) -> u8 {
    if v.is_nan() {
        return 0;
    }
    let v = v as f32;
    borders.partition_point(|&b| v > b) as u8
}

// ============================================================
// Oblivious trees
// ============================================================

/// A symmetric tree: level `d` sends a row to the right (bit `d` of the
/// leaf index) when `x[feature] > border`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ObliviousTree {
    splits: Vec<(usize, f32)>,
    leaf_values: Vec<f64>,
}

impl ObliviousTree {
    fn leaf(&self, row: ArrayView1<f64>) -> usize {
        self.splits
            .iter()
            .enumerate()
            .map(|(d, &(f, border))| usize::from((row[f] as f32) > border) << d)
            .sum()
    }
}

/// How a tree's splits are scored (CatBoost's `score_function`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum CatBoostScoreFunction {
    /// `Σ v·G / √(Σ v²·W)` over the leaves (CatBoost's default).
    #[default]
    Cosine,
    /// `Σ v·G` over the leaves.
    L2,
}

/// How leaf values are estimated (CatBoost's `leaf_estimation_method`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum CatBoostLeafEstimation {
    /// `ΣDer / (−ΣDer2 + λ)` (CatBoost's default for RMSE and Logloss).
    #[default]
    Newton,
    /// `ΣDer / (ΣW + λ)`.
    Gradient,
}

/// Losses, with CatBoost's derivative signs (`der1 = −∂L/∂a`).
#[derive(Clone, Copy)]
enum Loss {
    Rmse,
    Logloss,
}

impl Loss {
    fn ders(self, target: f64, approx: f64) -> (f64, f64) {
        match self {
            Loss::Rmse => (target - approx, -1.0),
            Loss::Logloss => {
                let e = approx.exp();
                let p = e / (1.0 + e);
                let p = if p.is_nan() { 1.0 } else { p };
                (target - p, -p * (1.0 - p))
            }
        }
    }
}

/// Shared training parameters.
#[derive(Clone, Copy)]
struct TrainParams {
    iterations: usize,
    learning_rate: f64,
    depth: usize,
    l2_leaf_reg: f64,
    score_function: CatBoostScoreFunction,
    leaf_estimation: CatBoostLeafEstimation,
}

/// `sum · 1/(count + λ)` for a non-empty side, else 0 (CatBoost's
/// `CalcAverage`).
#[inline]
fn average(sum: f64, count: f64, l2: f64) -> f64 {
    let inv = if count > 0.0 { 1.0 / (count + l2) } else { 0.0 };
    sum * inv
}

/// The quantized training data.
struct Quantized {
    borders: Vec<Vec<f32>>,
    /// Per feature: the bucket of every row.
    buckets: Vec<Vec<u8>>,
}

fn quantize(x: ArrayView2<f64>, border_count: usize) -> Quantized {
    use rayon::prelude::*;
    let (borders, buckets) = (0..x.ncols())
        .into_par_iter()
        .map(|f| {
            let col = x.column(f);
            let borders = feature_borders(col, border_count);
            let buckets = col.iter().map(|&v| bucket(v, &borders)).collect();
            (borders, buckets)
        })
        .unzip();
    Quantized { borders, buckets }
}

/// Grow one oblivious tree on the weighted first derivatives `wder`
/// (`w · der1`) and weights `w`; returns the splits as (feature, border
/// index) and every row's leaf.
fn grow_tree(
    q: &Quantized,
    wder: &[f64],
    w: &[f64],
    depth: usize,
    scaled_l2: f64,
    score_function: CatBoostScoreFunction,
) -> (Vec<(usize, usize)>, Vec<u32>) {
    use rayon::prelude::*;
    let n = wder.len();
    let mut splits: Vec<(usize, usize)> = Vec::new();
    let mut leaf_of = vec![0u32; n];
    for d in 0..depth {
        let leaves = 1usize << d;
        // Best (score, border) per feature, the first best border winning.
        let per_feature: Vec<Option<(f64, usize)>> = q
            .buckets
            .par_iter()
            .zip(&q.borders)
            .map(|(col, borders)| {
                let nb = borders.len() + 1;
                if nb < 2 {
                    return None;
                }
                let mut hist = vec![(0.0f64, 0.0f64); leaves * nb];
                for i in 0..n {
                    let h = &mut hist[leaf_of[i] as usize * nb + col[i] as usize];
                    h.0 += wder[i];
                    h.1 += w[i];
                }
                let splits = nb - 1;
                let mut num = vec![0.0f64; splits];
                let mut den = vec![1e-100f64; splits];
                for leaf in 0..leaves {
                    let buckets = &hist[leaf * nb..(leaf + 1) * nb];
                    let mut all = (0.0, 0.0);
                    for b in buckets {
                        all.0 += b.0;
                        all.1 += b.1;
                    }
                    let (mut t, mut f) = (all, (0.0f64, 0.0f64));
                    for (s, b) in buckets[..splits].iter().enumerate() {
                        f.0 += b.0;
                        f.1 += b.1;
                        t.0 -= b.0;
                        t.1 -= b.1;
                        let (at, af) = (average(t.0, t.1, scaled_l2), average(f.0, f.1, scaled_l2));
                        match score_function {
                            // CatBoost's SSE kernel: `prev + (false + true)`.
                            CatBoostScoreFunction::Cosine => {
                                num[s] += af * f.0 + at * t.0;
                                den[s] += af * af * f.1 + at * at * t.1;
                            }
                            // `AddLeaf(true)` then `AddLeaf(false)`.
                            CatBoostScoreFunction::L2 => {
                                num[s] += at * t.0;
                                num[s] += af * f.0;
                            }
                        }
                    }
                }
                let score = |s: usize| match score_function {
                    CatBoostScoreFunction::Cosine => num[s] / den[s].sqrt(),
                    CatBoostScoreFunction::L2 => num[s],
                };
                let mut best: Option<(f64, usize)> = None;
                for s in 0..splits {
                    let sc = score(s);
                    if best.is_none_or(|(b, _)| sc > b) {
                        best = Some((sc, s));
                    }
                }
                best
            })
            .collect();
        let mut best: Option<(f64, usize, usize)> = None;
        for (f, cand) in per_feature.into_iter().enumerate() {
            if let Some((sc, s)) = cand {
                if best.is_none_or(|(b, _, _)| sc > b) {
                    best = Some((sc, f, s));
                }
            }
        }
        let Some((_, feature, border)) = best else {
            break;
        };
        let col = &q.buckets[feature];
        for i in 0..n {
            leaf_of[i] |= u32::from(col[i] as usize > border) << d;
        }
        splits.push((feature, border));

        // A split with no pair of non-empty sibling leaves is dropped and
        // growth stops (CatBoost's `GetRedundantSplitIdx`).
        let n_leaves = 1usize << (d + 1);
        let mut occupied = vec![false; n_leaves];
        for &l in &leaf_of {
            occupied[l as usize] = true;
        }
        let redundant = (0..=d).find(|&k| {
            (0..n_leaves)
                .filter(|idx| idx & (1 << k) == 0)
                .all(|idx| !(occupied[idx] && occupied[idx ^ (1 << k)]))
        });
        if let Some(k) = redundant {
            splits.remove(k);
            for (i, l) in leaf_of.iter_mut().enumerate() {
                *l = splits
                    .iter()
                    .enumerate()
                    .map(|(d, &(f, b))| u32::from(q.buckets[f][i] as usize > b) << d)
                    .sum();
            }
            break;
        }
    }
    (splits, leaf_of)
}

/// Plain boosting of oblivious trees from the constant `bias`.
fn boost(
    q: &Quantized,
    target: &[f64],
    weights: &[f64],
    bias: f64,
    loss: Loss,
    p: TrainParams,
) -> Vec<ObliviousTree> {
    let n = target.len();
    let sum_w: f64 = weights.iter().sum();
    let scaled_l2 = p.l2_leaf_reg * (sum_w / n as f64);
    let mut approx = vec![bias; n];
    let mut trees = Vec::with_capacity(p.iterations);
    for _ in 0..p.iterations {
        let ders: Vec<(f64, f64)> = target
            .iter()
            .zip(&approx)
            .map(|(&t, &a)| loss.ders(t, a))
            .collect();
        let wder: Vec<f64> = ders.iter().zip(weights).map(|(d, &w)| w * d.0).collect();
        let (splits, leaf_of) = grow_tree(q, &wder, weights, p.depth, scaled_l2, p.score_function);
        let n_leaves = 1usize << splits.len();
        let mut sums = vec![(0.0f64, 0.0f64, 0.0f64); n_leaves];
        for i in 0..n {
            let s = &mut sums[leaf_of[i] as usize];
            s.0 += wder[i];
            s.1 += weights[i] * ders[i].1;
            s.2 += weights[i];
        }
        let leaf_values: Vec<f64> = sums
            .iter()
            .map(|&(der, der2, w)| {
                let delta = match p.leaf_estimation {
                    CatBoostLeafEstimation::Newton => der / (-der2 + scaled_l2),
                    CatBoostLeafEstimation::Gradient => average(der, w, scaled_l2),
                };
                p.learning_rate * delta
            })
            .collect();
        for i in 0..n {
            approx[i] += leaf_values[leaf_of[i] as usize];
        }
        trees.push(ObliviousTree {
            splits: splits.iter().map(|&(f, b)| (f, q.borders[f][b])).collect(),
            leaf_values,
        });
    }
    trees
}

/// Raw scores `bias + Σ leaf values` of the rows of `x`.
fn raw_scores(trees: &[ObliviousTree], bias: f64, x: ArrayView2<f64>) -> Array1<f64> {
    use rayon::prelude::*;
    let mut out = vec![bias; x.nrows()];
    out.par_iter_mut()
        .with_min_len(1024)
        .enumerate()
        .for_each(|(i, s)| {
            let row = x.row(i);
            for t in trees {
                *s += t.leaf_values[t.leaf(row)];
            }
        });
    Array1::from(out)
}

fn validate(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    w: Option<ArrayView1<f64>>,
    p: &TrainParams,
    border_count: usize,
) -> Result<()> {
    if x.nrows() == 0 {
        return Err(RustMlError::EmptyInput("training data is empty".into()));
    }
    if x.nrows() != y.len() {
        return Err(RustMlError::ShapeMismatch(format!(
            "X has {} rows but y has {}",
            x.nrows(),
            y.len()
        )));
    }
    if let Some(w) = w {
        if w.len() != y.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "sample_weight has {} entries but y has {}",
                w.len(),
                y.len()
            )));
        }
        if w.iter().any(|&v| !(v >= 0.0) || !v.is_finite()) {
            return Err(RustMlError::InvalidParameter(
                "sample weights must be finite and non-negative".into(),
            ));
        }
    }
    if p.depth == 0 || p.depth > 16 {
        return Err(RustMlError::InvalidParameter(format!(
            "depth must be in [1, 16], got {}",
            p.depth
        )));
    }
    if !(1..=255).contains(&border_count) {
        return Err(RustMlError::InvalidParameter(format!(
            "border_count must be in [1, 255], got {border_count}"
        )));
    }
    if !(p.learning_rate > 0.0) || !(p.l2_leaf_reg >= 0.0) {
        return Err(RustMlError::InvalidParameter(
            "learning_rate must be > 0 and l2_leaf_reg >= 0".into(),
        ));
    }
    Ok(())
}

fn check_features(x: ArrayView2<f64>, n_features: usize) -> Result<()> {
    if x.ncols() != n_features {
        return Err(RustMlError::ShapeMismatch(format!(
            "expected {n_features} features, got {}",
            x.ncols()
        )));
    }
    Ok(())
}

/// CatBoost's `CalculateWeightedTargetAverage`: float32 products summed in
/// double, the mean returned as float32.
fn weighted_target_average(target: &[f64], weights: &[f64]) -> f32 {
    let sum_w: f64 = weights.iter().sum();
    let sum: f64 = target
        .iter()
        .zip(weights)
        .map(|(&t, &w)| f64::from(t as f32 * w as f32))
        .sum();
    (sum / sum_w) as f32
}

/// CatBoost stores targets and weights as float32.
fn f32_values(v: ArrayView1<f64>) -> Vec<f64> {
    v.iter().map(|&t| f64::from(t as f32)).collect()
}

macro_rules! catboost_config {
    ($(#[$doc:meta])* $name:ident, boost_from_average: $bfa:expr) => {
        $(#[$doc])*
        #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
        #[serde(default)]
        pub struct $name {
            /// Number of trees (CatBoost's `iterations`, default 1000).
            pub iterations: usize,
            /// Shrinkage of every tree (default 0.03; CatBoost picks one
            /// from the data when it is not given).
            pub learning_rate: f64,
            /// Tree depth (levels of the symmetric tree, default 6).
            pub depth: usize,
            /// L2 regularization of the leaf values (default 3), scaled by
            /// the mean sample weight as in CatBoost.
            pub l2_leaf_reg: f64,
            /// Borders per numeric feature (default 254).
            pub border_count: usize,
            pub score_function: CatBoostScoreFunction,
            pub leaf_estimation_method: CatBoostLeafEstimation,
            /// Start from the loss-optimal constant instead of 0.
            pub boost_from_average: bool,
        }

        impl Default for $name {
            fn default() -> Self {
                Self {
                    iterations: 1000,
                    learning_rate: 0.03,
                    depth: 6,
                    l2_leaf_reg: 3.0,
                    border_count: 254,
                    score_function: CatBoostScoreFunction::Cosine,
                    leaf_estimation_method: CatBoostLeafEstimation::Newton,
                    boost_from_average: $bfa,
                }
            }
        }

        impl $name {
            pub fn new() -> Self {
                Self::default()
            }
            pub fn with_iterations(mut self, n: usize) -> Self {
                self.iterations = n;
                self
            }
            pub fn with_learning_rate(mut self, lr: f64) -> Self {
                self.learning_rate = lr;
                self
            }
            pub fn with_depth(mut self, d: usize) -> Self {
                self.depth = d;
                self
            }
            pub fn with_l2_leaf_reg(mut self, l2: f64) -> Self {
                self.l2_leaf_reg = l2;
                self
            }
            pub fn with_border_count(mut self, b: usize) -> Self {
                self.border_count = b;
                self
            }
            pub fn with_score_function(mut self, s: CatBoostScoreFunction) -> Self {
                self.score_function = s;
                self
            }
            pub fn with_leaf_estimation_method(mut self, m: CatBoostLeafEstimation) -> Self {
                self.leaf_estimation_method = m;
                self
            }
            pub fn with_boost_from_average(mut self, b: bool) -> Self {
                self.boost_from_average = b;
                self
            }

            /// CatBoost keeps the learning rate and the L2 regularization
            /// as float32.
            fn train_params(&self) -> TrainParams {
                TrainParams {
                    iterations: self.iterations,
                    learning_rate: f64::from(self.learning_rate as f32),
                    depth: self.depth,
                    l2_leaf_reg: f64::from(self.l2_leaf_reg as f32),
                    score_function: self.score_function,
                    leaf_estimation: self.leaf_estimation_method,
                }
            }
        }
    };
}

catboost_config!(
    /// CatBoost-style regressor (RMSE) on oblivious trees.
    CatBoostRegressor,
    boost_from_average: true
);
catboost_config!(
    /// CatBoost-style binary classifier (Logloss) on oblivious trees.
    CatBoostClassifier,
    boost_from_average: false
);

/// Fitted [`CatBoostRegressor`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedCatBoostRegressor {
    trees: Vec<ObliviousTree>,
    bias: f64,
    borders: Vec<Vec<f32>>,
    n_features: usize,
}

impl FittedCatBoostRegressor {
    pub fn n_estimators(&self) -> usize {
        self.trees.len()
    }
    /// Quantization borders per feature.
    pub fn borders(&self) -> &[Vec<f32>] {
        &self.borders
    }
}

impl FitWeighted<f64> for CatBoostRegressor {
    type Fitted = FittedCatBoostRegressor;

    fn fit_weighted_view(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        sample_weight: Option<ArrayView1<f64>>,
    ) -> Result<Self::Fitted> {
        let p = self.train_params();
        validate(x, y, sample_weight, &p, self.border_count)?;
        let target = f32_values(y);
        let weights = sample_weight.map_or_else(|| vec![1.0; y.len()], f32_values);
        let bias = if self.boost_from_average {
            f64::from(weighted_target_average(&target, &weights))
        } else {
            0.0
        };
        let q = quantize(x, self.border_count);
        let trees = boost(&q, &target, &weights, bias, Loss::Rmse, p);
        Ok(FittedCatBoostRegressor {
            trees,
            bias,
            borders: q.borders,
            n_features: x.ncols(),
        })
    }
}

impl Fit<f64> for CatBoostRegressor {
    type Fitted = FittedCatBoostRegressor;

    fn fit_view(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<Self::Fitted> {
        self.fit_weighted_view(x, y, None)
    }
}

impl Predict<f64> for FittedCatBoostRegressor {
    fn predict_view(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        check_features(x, self.n_features)?;
        Ok(raw_scores(&self.trees, self.bias, x))
    }
}

/// Fitted [`CatBoostClassifier`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedCatBoostClassifier {
    trees: Vec<ObliviousTree>,
    bias: f64,
    borders: Vec<Vec<f32>>,
    classes: Vec<f64>,
    n_features: usize,
}

impl FittedCatBoostClassifier {
    pub fn classes(&self) -> &[f64] {
        &self.classes
    }
    pub fn n_estimators(&self) -> usize {
        self.trees.len()
    }

    /// Log-odds of the second class.
    pub fn decision_function<'a>(&self, x: impl AsArray<'a, f64, Ix2>) -> Result<Array1<f64>> {
        let x: ArrayView2<f64> = x.into();
        check_features(x, self.n_features)?;
        Ok(raw_scores(&self.trees, self.bias, x))
    }

    /// Probabilities of the two classes (columns in `classes()` order).
    pub fn predict_proba<'a>(&self, x: impl AsArray<'a, f64, Ix2>) -> Result<Array2<f64>> {
        let raw = self.decision_function(x)?;
        let mut proba = Array2::zeros((raw.len(), 2));
        for (i, &r) in raw.iter().enumerate() {
            let p1 = 1.0 / (1.0 + (-r).exp());
            proba[[i, 0]] = 1.0 - p1;
            proba[[i, 1]] = p1;
        }
        Ok(proba)
    }
}

impl FitWeighted<f64> for CatBoostClassifier {
    type Fitted = FittedCatBoostClassifier;

    fn fit_weighted_view(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        sample_weight: Option<ArrayView1<f64>>,
    ) -> Result<Self::Fitted> {
        let p = self.train_params();
        validate(x, y, sample_weight, &p, self.border_count)?;
        let mut classes: Vec<f64> = y.to_vec();
        classes.sort_by(f64::total_cmp);
        classes.dedup();
        if classes.len() != 2 {
            return Err(RustMlError::InvalidParameter(format!(
                "CatBoostClassifier supports two classes, got {}",
                classes.len()
            )));
        }
        let target: Vec<f64> = y.iter().map(|&v| f64::from(v == classes[1])).collect();
        let weights = sample_weight.map_or_else(|| vec![1.0; y.len()], f32_values);
        let bias = if self.boost_from_average {
            let p1 = f64::from(weighted_target_average(&target, &weights));
            (p1 / (1.0 - p1)).ln()
        } else {
            0.0
        };
        let q = quantize(x, self.border_count);
        let trees = boost(&q, &target, &weights, bias, Loss::Logloss, p);
        Ok(FittedCatBoostClassifier {
            trees,
            bias,
            borders: q.borders,
            classes,
            n_features: x.ncols(),
        })
    }
}

impl Fit<f64> for CatBoostClassifier {
    type Fitted = FittedCatBoostClassifier;

    fn fit_view(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<Self::Fitted> {
        self.fit_weighted_view(x, y, None)
    }
}

impl Predict<f64> for FittedCatBoostClassifier {
    fn predict_view(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        let raw = self.decision_function(x)?;
        Ok(raw.mapv(|r| {
            if r > 0.0 {
                self.classes[1]
            } else {
                self.classes[0]
            }
        }))
    }
}

/// Quantization borders (for the parity tests).
#[doc(hidden)]
pub fn catboost_borders(x: ArrayView2<f64>, border_count: usize) -> Vec<Vec<f32>> {
    quantize(x, border_count).borders
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(n: usize) -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_fn((n, 3), |(i, j)| ((i * (j + 3) * 31) % 97) as f64 / 9.0);
        let y = x.column(0).mapv(|v| (v * 0.7).sin()) + x.column(1).mapv(|v| v * 0.3);
        (x, y)
    }

    #[test]
    fn greedy_borders_take_every_gap_of_few_values() {
        let values: Vec<f32> = [1.0, 1.0, 2.0, 4.0, 4.0, 4.0, 7.0].to_vec();
        assert_eq!(greedy_log_sum_borders(&values, 254), vec![1.5, 3.0, 5.5]);
        assert_eq!(greedy_log_sum_borders(&values, 1).len(), 1);
        assert!(greedy_log_sum_borders(&[3.0; 5], 254).is_empty());
    }

    #[test]
    fn missing_values_get_the_lowest_bucket() {
        let x = Array2::from_shape_vec((4, 1), vec![f64::NAN, 1.0, 2.0, 3.0]).unwrap();
        let q = quantize(x.view(), 254);
        assert_eq!(q.borders[0][0], -f32::MAX);
        assert_eq!(q.buckets[0], vec![0, 1, 2, 3]);
    }

    #[test]
    fn trees_are_symmetric_and_fit_the_data() {
        let (x, y) = data(200);
        let f = CatBoostRegressor::new()
            .with_iterations(60)
            .with_learning_rate(0.2)
            .with_depth(3)
            .fit(&x, &y)
            .unwrap();
        assert_eq!(f.n_estimators(), 60);
        assert!(f
            .trees
            .iter()
            .all(|t| t.leaf_values.len() == 1 << t.splits.len()));
        let pred = f.predict(&x).unwrap();
        let mse = (&pred - &y).mapv(|e| e * e).mean().unwrap();
        assert!(mse < 0.1 * y.var(0.0), "{mse}");
        assert!(f.predict(Array2::<f64>::zeros((2, 2)).view()).is_err());
    }

    #[test]
    fn classifier_labels_and_probabilities() {
        let (x, y) = data(200);
        let labels = y.mapv(|v| if v > 0.5 { 7.0 } else { -1.0 });
        let f = CatBoostClassifier::new()
            .with_iterations(40)
            .with_learning_rate(0.3)
            .fit(&x, &labels)
            .unwrap();
        assert_eq!(f.classes(), &[-1.0, 7.0]);
        let proba = f.predict_proba(&x).unwrap();
        assert!(proba
            .rows()
            .into_iter()
            .all(|r| (r.sum() - 1.0).abs() < 1e-12));
        let acc = f
            .predict(&x)
            .unwrap()
            .iter()
            .zip(&labels)
            .filter(|(a, b)| a == b)
            .count() as f64
            / 200.0;
        assert!(acc > 0.9, "{acc}");
        let three = labels.mapv(|v| v.min(0.0)) + Array1::from_shape_fn(200, |i| (i % 3) as f64);
        assert!(CatBoostClassifier::new().fit(&x, &three).is_err());
    }

    #[test]
    fn invalid_parameters_are_rejected() {
        let (x, y) = data(20);
        assert!(CatBoostRegressor::new().with_depth(0).fit(&x, &y).is_err());
        assert!(CatBoostRegressor::new()
            .with_border_count(0)
            .fit(&x, &y)
            .is_err());
        assert!(CatBoostRegressor::new()
            .with_learning_rate(0.0)
            .fit(&x, &y)
            .is_err());
        let w = Array1::from_elem(19, 1.0);
        assert!(CatBoostRegressor::new()
            .fit_weighted(&x, &y, Some(&w))
            .is_err());
    }

    #[test]
    fn models_round_trip_through_serde() {
        let (x, y) = data(80);
        let f = CatBoostRegressor::new()
            .with_iterations(5)
            .fit(&x, &y)
            .unwrap();
        let bytes = anofox_ml_core::persistence::to_binary_bytes(&f).unwrap();
        let back: FittedCatBoostRegressor =
            anofox_ml_core::persistence::from_binary_bytes(&bytes).unwrap();
        assert_eq!(back.predict(&x).unwrap(), f.predict(&x).unwrap());
    }
}
