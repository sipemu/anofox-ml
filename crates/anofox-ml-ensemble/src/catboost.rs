//! CatBoost-style gradient boosting on oblivious (symmetric) trees (#127).
//!
//! Every tree splits all nodes of a level on the same feature and border,
//! so a tree of depth `d` is `d` (feature, border) pairs and `2^d` leaf
//! values, and prediction is a bit lookup. The numerics follow CatBoost
//! 1.2: its GreedyLogSum quantization (float32 features, `nan_mode=Min`),
//! the Cosine or L2 split score, Newton or Gradient leaves with CatBoost's
//! scaled L2 regularization, the removal of redundant splits, and its
//! plain and ordered boosting schemes. The losses are RMSE, Logloss and,
//! for more than two classes, MultiClass (a softmax over one approximation
//! per class, with CatBoost's full-Hessian Newton step). Categorical
//! features are split one-hot or through ordered target statistics (CTRs)
//! of single features and of feature combinations. With CatBoost's
//! defaults the models agree with CatBoost's (with `thread_count=1`): its
//! generator (`random_seed`) is reproduced draw for draw, from the initial
//! shuffle of the rows and the folds' permutations to the fold picked per
//! tree, the bootstrap (MVS, Bayesian, Bernoulli) and the noise of every
//! split's score (`random_strength`).
//!
//! Leaves take CatBoost's default number of Newton (or Gradient) steps,
//! with its backtracking: a step is halved until the objective improves,
//! the objective computed as CatBoost computes it (for Logloss its float32
//! fast-log metric on `exp(a)`). That metric is summed in blocks per
//! thread, so the multi-step Logloss models equal those of CatBoost with
//! `thread_count=1`. CatBoost keeps Logloss approximations as `exp(a)`
//! (updated with fmath's fast `exp`); ours agree with those to rounding, which
//! on large data can rarely decide a near-tie between splits differently.

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
// CatBoost's random numbers
// ============================================================

/// The 64-bit LCG of the PCG family with CatBoost's (Yandex `util`) output
/// mixing of the *new* state.
#[derive(Clone)]
struct Pcg32 {
    x: u64,
    c: u64,
}

impl Pcg32 {
    const A: u64 = 6364136223846793005;

    fn next(&mut self) -> u32 {
        self.x = self.x.wrapping_mul(Self::A).wrapping_add(self.c);
        let xorshifted = (((self.x >> 18) ^ self.x) >> 27) as u32;
        xorshifted.rotate_right((self.x >> 59) as u32)
    }

    /// `GenRand64` of a 32-bit generator: low word first.
    fn next64(&mut self) -> u64 {
        let lo = u64::from(self.next());
        lo | (u64::from(self.next()) << 32)
    }
}

/// CatBoost's `TRestorableFastRng64` (`TFastRng64`): two PCG streams
/// seeded from `TReallyFastRng32(seed)`.
#[derive(Clone)]
struct CatRng {
    r1: Pcg32,
    r2: Pcg32,
}

impl CatRng {
    /// `TFastRng64(seed1, seq1, seed2, seq2)`.
    fn from_parts(seed1: u64, seq1: u32, seed2: u64, seq2: u32) -> Self {
        // `FixSeq`: the two streams must differ.
        let mask = u32::MAX >> 1;
        let seq2 = if seq1 & mask == seq2 & mask {
            !seq2
        } else {
            seq2
        };
        Self {
            r1: Pcg32 {
                x: seed1,
                c: (u64::from(seq1) << 1) | 1,
            },
            r2: Pcg32 {
                x: seed2,
                c: (u64::from(seq2) << 1) | 1,
            },
        }
    }

    fn new(seed: u64) -> Self {
        let mut seeder = Pcg32 { x: seed, c: 1 };
        let seed1 = seeder.next64();
        let seq1 = seeder.next();
        let seed2 = seeder.next64();
        let seq2 = seeder.next();
        Self::from_parts(seed1, seq1, seed2, seq2)
    }

    /// Skip `n` draws (`Advance`).
    fn advance(&mut self, n: usize) {
        for _ in 0..n {
            self.gen();
        }
    }

    fn gen(&mut self) -> u64 {
        let x = u64::from(self.r1.next());
        let y = u64::from(self.r2.next());
        (x << 32) | y
    }

    /// Uniform in `[0, max)` by rejection (`GenUniform`).
    fn uniform(&mut self, max: u64) -> u64 {
        let limit = u64::MAX - u64::MAX % max;
        loop {
            let r = self.gen();
            if r < limit {
                return r % max;
            }
        }
    }

    /// Uniform in `[0, 1]` (`GenRandReal1`).
    fn real1(&mut self) -> f64 {
        (self.gen() >> 11) as f64 * (1.0 / 9007199254740991.0)
    }

    /// A standard normal by the polar Box-Muller method
    /// (`StdNormalDistribution`).
    fn std_normal(&mut self) -> f64 {
        loop {
            let x = self.real1() * 2.0 - 1.0;
            let y = self.real1() * 2.0 - 1.0;
            let r = x * x + y * y;
            if r <= 1.0 && r > 0.0 {
                return x * (-2.0 * r.ln() / r).sqrt();
            }
        }
    }

    /// CatBoost's fold permutation (`Shuffle` of the objects grouping):
    /// blocks of `block` rows in Fisher-Yates order (`CreateShuffledIndices`).
    fn permutation(&mut self, n: usize, block: usize) -> Vec<usize> {
        let blocks = n.div_ceil(block);
        let mut order: Vec<usize> = (0..blocks).collect();
        for i in 1..blocks {
            let j = self.uniform(i as u64 + 1) as usize;
            order.swap(i, j);
        }
        order
            .into_iter()
            .flat_map(|b| b * block..((b + 1) * block).min(n))
            .collect()
    }
}

// ============================================================
// Oblivious trees
// ============================================================

/// One level of a symmetric tree: rows going right set bit `d` of their
/// leaf index.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
enum TreeSplit {
    /// `x[feature] > border` (in float32).
    Float { feature: usize, border: f32 },
    /// The categorical `x[feature]` is the category `value` (one-hot).
    OneHot { feature: usize, value: u64 },
    /// The quantized CTR number `ctr` of the row's category exceeds
    /// `bucket`.
    Ctr { ctr: usize, bucket: u8 },
}

/// A symmetric tree: `2^d` leaf values for `d` splits.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ObliviousTree {
    splits: Vec<TreeSplit>,
    leaf_values: Vec<f64>,
}

impl ObliviousTree {
    fn leaf(&self, row: ArrayView1<f64>, ctrs: &CtrModel) -> usize {
        self.splits
            .iter()
            .enumerate()
            .map(|(d, split)| {
                let right = match *split {
                    TreeSplit::Float { feature, border } => (row[feature] as f32) > border,
                    TreeSplit::OneHot { feature, value } => category(row[feature]) == value,
                    TreeSplit::Ctr { ctr, bucket } => ctrs.bucket(ctr, row) > bucket,
                };
                usize::from(right) << d
            })
            .sum()
    }
}

// ============================================================
// Categorical features: ordered target statistics (CTRs)
// ============================================================

/// Borders of the quantized CTR values (CatBoost's `CtrBorderCount`).
const CTR_BORDER_COUNT: i32 = 15;

/// A category's key (the value's bits; categories are numeric codes).
#[inline]
fn category(v: f64) -> u64 {
    if v.is_nan() {
        f64::NAN.to_bits()
    } else {
        (v + 0.0).to_bits() // -0 and 0 are one category
    }
}

/// CatBoost's default CTRs of a categorical feature: `Borders` (the rate of
/// the target classes above a target border) with the priors 0, 0.5 and 1,
/// and `Counter` (the category's frequency) with the prior 0.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
enum CtrKind {
    Borders {
        prior: f32,
        /// The target border: the classes above `border` count.
        #[serde(default)]
        border: u8,
    },
    Counter {
        prior: f32,
    },
}

/// The default CTRs for `n_classes` target classes in CatBoost's candidate
/// order: `Borders` per target border (`n_classes - 1`) and prior, then
/// `Counter`.
fn default_ctrs(n_classes: usize) -> Vec<CtrKind> {
    let mut kinds: Vec<CtrKind> = (0..n_classes.saturating_sub(1) as u8)
        .flat_map(|border| [0.0, 0.5, 1.0].map(|prior| CtrKind::Borders { prior, border }))
        .collect();
    kinds.push(CtrKind::Counter { prior: 0.0 });
    kinds
}

/// CatBoost's `CalcCTR` (in float32): `(count + prior) / (total + 1)`,
/// shifted and scaled into `[0, 1]` (`CalcNormalization`) and truncated to
/// one of `CTR_BORDER_COUNT + 1` buckets.
#[inline]
fn calc_ctr(count_in_class: u32, total: u32, prior: f32) -> u8 {
    let (shift, norm) = (-(0.0f32.min(prior)), 1.0f32.max(prior) - 0.0f32.min(prior));
    let ctr = (count_in_class as f32 + prior) / ((total as i64 + 1) as f32);
    ((ctr + shift) / norm * CTR_BORDER_COUNT as f32) as u8
}

/// The rows of the classes above the target border `border`.
#[inline]
fn above_border(classes: &[u32], border: u8) -> u32 {
    classes.iter().skip(border as usize + 1).sum()
}

/// What a CTR counts over (CatBoost's `TProjection`): a combination of
/// categorical features and of the float / one-hot conditions of tree
/// splits. A simple CTR's projection is one categorical feature.
#[derive(
    Debug,
    Clone,
    Default,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    serde::Serialize,
    serde::Deserialize,
)]
struct Projection {
    /// Categorical features (sorted).
    cats: Vec<usize>,
    /// `x[feature] > border` conditions as (feature, border bits), sorted.
    floats: Vec<(usize, u32)>,
    /// `x[feature] == category` conditions, sorted.
    one_hots: Vec<(usize, u64)>,
}

impl Projection {
    fn single(feature: usize) -> Self {
        Self {
            cats: vec![feature],
            ..Self::default()
        }
    }

    fn is_empty(&self) -> bool {
        self.cats.is_empty() && self.floats.is_empty() && self.one_hots.is_empty()
    }

    /// CatBoost's `GetFullProjectionLength`: the conditions count as one.
    fn complexity(&self) -> usize {
        self.cats.len() + usize::from(!self.floats.is_empty() || !self.one_hots.is_empty())
    }

    /// The combination a row belongs to: its categories, then its
    /// conditions (0 / 1).
    fn key(&self, row: ArrayView1<f64>) -> Vec<u64> {
        let cats = self.cats.iter().map(|&f| category(row[f]));
        let floats = self
            .floats
            .iter()
            .map(|&(f, b)| u64::from((row[f] as f32) > f32::from_bits(b)));
        let one_hots = self
            .one_hots
            .iter()
            .map(|&(f, v)| u64::from(category(row[f]) == v));
        cats.chain(floats).chain(one_hots).collect()
    }
}

/// Final class counts of a projection's combinations over the training rows
/// (what a prediction's CTRs are computed from).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
struct CtrTable {
    projection: Projection,
    /// Per combination (sorted by key): rows of every target class.
    classes: Vec<(Vec<u64>, Vec<u32>)>,
    /// The largest combination count (the `Counter` denominator).
    max_count: u32,
}

impl CtrTable {
    /// The class counts of a combination (none for an unseen one).
    fn counts(&self, key: &[u64]) -> &[u32] {
        self.classes
            .binary_search_by(|(k, _)| k.as_slice().cmp(key))
            .map_or(&[][..], |i| &self.classes[i].1)
    }
}

/// A CTR of a fitted model: its kind over the table `table`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct Ctr {
    table: usize,
    kind: CtrKind,
}

/// The CTRs that a fitted model's splits use, and their tables.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
struct CtrModel {
    ctrs: Vec<Ctr>,
    tables: Vec<CtrTable>,
}

impl CtrModel {
    /// The quantized CTR `ctr` of a row (from the training counts; an
    /// unseen combination has none).
    fn bucket(&self, ctr: usize, row: ArrayView1<f64>) -> u8 {
        let Ctr { table, kind } = self.ctrs[ctr];
        let table = &self.tables[table];
        let classes = table.counts(&table.projection.key(row));
        let total = classes.iter().sum();
        match kind {
            CtrKind::Borders { prior, border } => {
                calc_ctr(above_border(classes, border), total, prior)
            }
            CtrKind::Counter { prior } => calc_ctr(total, table.max_count, prior),
        }
    }

    /// The index of the CTR `kind` over `table`, added on first use.
    fn register(&mut self, table: &CtrTable, kind: CtrKind) -> usize {
        let t = match self
            .tables
            .iter()
            .position(|t| t.projection == table.projection)
        {
            Some(t) => t,
            None => {
                self.tables.push(table.clone());
                self.tables.len() - 1
            }
        };
        match self
            .ctrs
            .iter()
            .position(|c| c.table == t && c.kind == kind)
        {
            Some(c) => c,
            None => {
                self.ctrs.push(Ctr { table: t, kind });
                self.ctrs.len() - 1
            }
        }
    }
}

// ============================================================
// Training
// ============================================================

/// How a tree's splits are scored (CatBoost's `score_function`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum CatBoostScoreFunction {
    /// `Σ v·G / √(Σ v²·W)` over the leaves (CatBoost's default).
    #[default]
    Cosine,
    /// `Σ v·G` over the leaves.
    L2,
}

/// CatBoost's `boosting_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum CatBoostBoostingType {
    /// The trees are scored and fitted on all rows.
    #[default]
    Plain,
    /// Ordered boosting: the splits are scored with approximations that
    /// did not see the scored rows (against CatBoost's prediction shift).
    Ordered,
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

/// CatBoost's `bootstrap_type`: how the rows a tree is scored on are drawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CatBoostBootstrap {
    /// All rows, weight 1.
    No,
    /// Weights `(-ln u)^bagging_temperature`.
    Bayesian,
    /// A `subsample` share of the rows.
    Bernoulli,
    /// Minimal variance sampling: rows kept by the size of their gradient,
    /// reweighted (CatBoost's default for one approximation).
    #[serde(rename = "MVS")]
    Mvs,
}

/// Whether a leaf estimation step is halved until the loss improves
/// (CatBoost's `leaf_estimation_backtracking`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum CatBoostBacktracking {
    /// Halve a step until the objective improves at all (CatBoost's
    /// default).
    #[default]
    AnyImprovement,
    /// Take every step whole.
    No,
}

/// Losses, with CatBoost's derivative signs (`der1 = −∂L/∂a`).
#[derive(Clone, Copy)]
enum Loss {
    Rmse,
    Logloss,
    /// Softmax over `k` approximations (CatBoost's `MultiClass`); the target
    /// is the class index.
    MultiClass(usize),
}

impl Loss {
    /// Approximations per row.
    fn dims(self) -> usize {
        match self {
            Loss::MultiClass(k) => k,
            _ => 1,
        }
    }

    /// Second derivatives per row: the upper triangle of the Hessian, row
    /// by row (CatBoost's packed `THessianInfo`).
    fn hessian_len(self) -> usize {
        let k = self.dims();
        k * (k + 1) / 2
    }

    /// The weighted derivatives of one row into `der` (`dims`) and `der2`
    /// (`hessian_len`).
    fn ders(self, target: f64, approx: &[f64], w: f64, der: &mut [f64], der2: &mut [f64]) {
        match self {
            Loss::Rmse => {
                der[0] = w * (target - approx[0]);
                der2[0] = -w;
            }
            Loss::Logloss => {
                let e = approx[0].exp();
                let p = e / (1.0 + e);
                let p = if p.is_nan() { 1.0 } else { p };
                der[0] = w * (target - p);
                der2[0] = w * (-p * (1.0 - p));
            }
            // CatBoost's `TMultiClassError::CalcDersMulti`.
            Loss::MultiClass(k) => {
                let max = approx.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut sum = 0.0;
                for (d, &a) in der.iter_mut().zip(approx) {
                    *d = (a - max).exp();
                    sum += *d;
                }
                for d in der.iter_mut() {
                    *d /= sum;
                }
                let mut idx = 0;
                for y in 0..k {
                    der2[idx] = der[y] * (der[y] - 1.0);
                    idx += 1;
                    for x in y + 1..k {
                        der2[idx] = der[y] * der[x];
                        idx += 1;
                    }
                }
                for d in der.iter_mut() {
                    *d = -*d;
                }
                der[target as usize] += 1.0;
                if w != 1.0 {
                    der.iter_mut().for_each(|d| *d *= w);
                    der2.iter_mut().for_each(|d| *d *= w);
                }
            }
        }
    }

    /// CatBoost's objective metric of one row (`RMSE`, `Logloss`,
    /// `MultiClass`), unweighted: what backtracking compares.
    fn metric(self, target: f64, approx: &[f64]) -> f64 {
        match self {
            Loss::Rmse => (target - approx[0]) * (target - approx[0]),
            Loss::Logloss => {
                let a = approx[0];
                let e = a.exp();
                if e.is_finite() {
                    (1.0 + e).ln() - target * a
                } else {
                    (1.0 - target) * a
                }
            }
            Loss::MultiClass(_) => {
                let max = approx.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let sum: f64 = approx.iter().map(|&a| (a - max).exp()).sum();
                -((approx[target as usize] - max) - sum.ln())
            }
        }
    }

    /// Whether CatBoost stores this loss's approximations as `exp(a)`
    /// (`IsStoreExpApprox`), which changes how its metric and learning
    /// folds are computed.
    fn exp_approx(self) -> bool {
        matches!(self, Loss::Logloss)
    }

    /// How a learning fold's approximation moves by the leaf value `v`
    /// (CatBoost's `ApplyLearningRate`): `v · lr`, or for stored `exp(a)`
    /// `fast_exp(FastLogf(exp(v)) · lr)`, i.e. `FastLogf(exp(v)) · lr`.
    fn fold_step(self, v: f64, lr: f64) -> f64 {
        if self.exp_approx() {
            f64::from(fast_logf(v.exp() as f32)) * lr
        } else {
            v * lr
        }
    }

    /// The weighted metric of the rows `0..rows` at `approx` moved by the
    /// leaf values `leaves` (CatBoost's `EvalErrorsWithLeaves` and
    /// `GetFinalError`).
    #[allow(clippy::too_many_arguments)]
    fn metric_with_leaves(
        self,
        target: &[f64],
        approx: &[f64],
        leaves: &[f64],
        leaf_of: &[u32],
        weights: &[f64],
        rows: usize,
    ) -> f64 {
        if self.exp_approx() {
            return logloss_metric_exp(target, approx, leaves, leaf_of, weights, rows);
        }
        let k = self.dims();
        let (mut sum, mut sum_w) = (0.0, 0.0);
        let mut point = vec![0.0; k];
        for i in 0..rows {
            let leaf = leaf_of[i] as usize;
            for d in 0..k {
                point[d] = approx[i * k + d] + leaves[leaf * k + d];
            }
            sum += weights[i] * self.metric(target[i], &point);
            sum_w += weights[i];
        }
        match self {
            Loss::Rmse => (sum / (sum_w + 1e-38)).sqrt(),
            _ => sum / sum_w,
        }
    }

    /// The weighted derivatives of the rows `0..rows`.
    fn all_ders(self, target: &[f64], approx: &[f64], weights: &[f64], rows: usize) -> Ders {
        let (k, h) = (self.dims(), self.hessian_len());
        let mut out = Ders {
            k,
            h,
            der: vec![0.0; rows * k],
            der2: vec![0.0; rows * h],
        };
        for i in 0..rows {
            self.ders(
                target[i],
                &approx[i * k..(i + 1) * k],
                weights[i],
                &mut out.der[i * k..(i + 1) * k],
                &mut out.der2[i * h..(i + 1) * h],
            );
        }
        out
    }
}

/// Paul Mineiro's `FastLogf` (CatBoost's `library/cpp/fast_log`, the same
/// arithmetic as its SSE `VectorFastLogf`): a float logarithm accurate to
/// about 1e-5.
#[inline]
#[allow(clippy::excessive_precision, clippy::approx_constant)]
fn fast_logf(x: f32) -> f32 {
    let bits = x.to_bits();
    let m = f32::from_bits((bits & 0x007f_ffff) | 0x3f00_0000);
    let y = bits as i32 as f32 * 1.1920928955078125e-7f32;
    let log2 = y - 124.22551499f32 - m * 1.498030302f32 - 1.72587999f32 / (m + 0.3520887068f32);
    log2 * 0.69314718f32
}

/// CatBoost's Logloss metric on stored `exp(a)` approximations (what its
/// backtracking compares): `EvalCrossEntropyVectorized`, i.e. float32 lanes
/// of four rows with `VectorFastLogf`, horizontally added per block of 4096
/// rows, and the rows past the last full group in double with `FastLogf`.
/// The leaf deltas enter as `exp(δ)`.
fn logloss_metric_exp(
    target: &[f64],
    approx: &[f64],
    leaves: &[f64],
    leaf_of: &[u32],
    weights: &[f64],
    rows: usize,
) -> f64 {
    let exp_leaves: Vec<f64> = leaves.iter().map(|v| v.exp()).collect();
    let (mut s0, mut s1) = (0.0f64, 0.0f64);
    for start in (0..rows).step_by(4096) {
        let end = (start + 4096).min(rows);
        let (mut lanes0, mut lanes1) = ([0.0f32; 4], [0.0f32; 4]);
        let mut i = start;
        while i + 4 <= end {
            for lane in 0..4 {
                let r = i + lane;
                let prob = if target[r] > 0.5 { 1.0f32 } else { 0.0 };
                let (e, d) = (approx[r].exp(), exp_leaves[leaf_of[r] as usize]);
                let non_exp = fast_logf(e as f32) + fast_logf(d as f32);
                let e = e * d;
                let value = if e > f64::MAX {
                    (1.0 - prob) * non_exp
                } else {
                    fast_logf((e + 1.0) as f32) - prob * non_exp
                };
                let w = weights[r] as f32;
                lanes0[lane] += w * value;
                lanes1[lane] += w;
            }
            i += 4;
        }
        s0 += f64::from((lanes0[0] + lanes0[1]) + (lanes0[2] + lanes0[3]));
        s1 += f64::from((lanes1[0] + lanes1[1]) + (lanes1[2] + lanes1[3]));
        for r in i..end {
            let w = f64::from(weights[r] as f32);
            let prob = if target[r] > 0.5 { 1.0 } else { 0.0 };
            let (e, d) = (approx[r].exp(), exp_leaves[leaf_of[r] as usize]);
            let mut non_exp = f64::from(fast_logf(e as f32));
            let e = e * d;
            non_exp += f64::from(fast_logf(d as f32));
            s0 += w * if e.is_finite() {
                f64::from(fast_logf((1.0 + e) as f32)) - prob * non_exp
            } else {
                (1.0 - prob) * non_exp
            };
            s1 += w;
        }
    }
    s0 / s1
}

/// Weighted first and second derivatives, `k` and `h` per row.
struct Ders {
    k: usize,
    h: usize,
    der: Vec<f64>,
    der2: Vec<f64>,
}

/// LAPACK's `dppsv` with `uplo = 'L'` (the reference `dpptrf` Cholesky
/// factorization and `dpptrs` solve, which CatBoost calls): solves `A x = b`
/// in place for the symmetric positive definite `A`, packed column by column
/// of its lower triangle. `false` when `A` is not positive definite.
fn solve_packed_cholesky(ap: &mut [f64], b: &mut [f64]) -> bool {
    let n = b.len();
    // dpptrf: A = L Lᵀ.
    let mut jj = 0;
    for j in 0..n {
        let ajj = ap[jj];
        if !(ajj > 0.0) {
            return false;
        }
        let ajj = ajj.sqrt();
        ap[jj] = ajj;
        if j + 1 < n {
            let m = n - j - 1;
            let r = 1.0 / ajj;
            for t in 1..=m {
                ap[jj + t] *= r;
            }
            // dspr: the trailing matrix minus x xᵀ.
            let mut kk = jj + m + 1;
            for c in 0..m {
                let xc = ap[jj + 1 + c];
                if xc != 0.0 {
                    let temp = -xc;
                    for (k, i) in (c..m).enumerate() {
                        ap[kk + k] += ap[jj + 1 + i] * temp;
                    }
                }
                kk += m - c;
            }
            jj += m + 1;
        }
    }
    // dtpsv: L y = b, then Lᵀ x = y.
    let mut kk = 0;
    for j in 0..n {
        if b[j] != 0.0 {
            b[j] /= ap[kk];
            let temp = b[j];
            for (k, i) in (j + 1..n).enumerate() {
                b[i] -= temp * ap[kk + 1 + k];
            }
        }
        kk += n - j;
    }
    let mut kk = n * (n + 1) / 2 - 1;
    for j in (0..n).rev() {
        let mut temp = b[j];
        let mut k = kk;
        for i in (j + 1..n).rev() {
            temp -= ap[k] * b[i];
            k -= 1;
        }
        temp /= ap[kk + j + 1 - n];
        b[j] = temp;
        if j > 0 {
            kk -= n - j;
        }
    }
    true
}

/// CatBoost's `CalcDeltaNewtonMulti`: the Newton step `(−H + λI)⁻¹ g` from
/// the summed derivatives `g` and the packed Hessian `H`, with the L2
/// regularization scaled in float32 (`SolveNewtonEquation`). A system that
/// is not positive definite (an empty leaf without regularization) gives 0.
fn newton_multi(der: &[f64], der2: &[f64], l2: f64, sum_w: f64, count: f64, out: &mut [f64]) {
    let k = der.len();
    let l2 = (l2 * (sum_w / count)) as f32;
    let mut max_trace = l2;
    let mut idx = 0;
    for row in 0..k {
        max_trace = max_trace.max(-der2[idx] as f32);
        idx += k - row;
    }
    let adjusted = f64::from(l2.max(max_trace * f32::EPSILON));
    let mut hessian = der2.to_vec();
    let mut idx = 0;
    for row in 0..k {
        hessian[idx] -= adjusted;
        idx += k - row;
    }
    hessian.iter_mut().for_each(|v| *v = -*v);
    for (o, &d) in out.iter_mut().zip(der) {
        *o = -d;
    }
    if !solve_packed_cholesky(&mut hessian, out) {
        out.iter_mut().for_each(|v| *v = 0.0);
        return;
    }
    out.iter_mut().for_each(|v| *v = -*v);
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
    model_size_reg: f64,
    ordered: bool,
    fold_len_multiplier: f64,
    /// Newton / Gradient steps per leaf estimation.
    leaf_iterations: usize,
    /// Halve a step until the loss improves (CatBoost's
    /// `leaf_estimation_backtracking=AnyImprovement`).
    backtracking: bool,
    /// Keep the rows' order (no permutations).
    has_time: bool,
    /// The scale of the split scores' noise (float32, as in CatBoost).
    random_strength: f64,
    bootstrap: CatBoostBootstrap,
    subsample: f64,
    bagging_temperature: f64,
    mvs_reg: Option<f64>,
    permutation_count: usize,
}

/// `sum · 1/(count + λ)` for a non-empty side, else 0 (CatBoost's
/// `CalcAverage`).
#[inline]
fn average(sum: f64, count: f64, l2: f64) -> f64 {
    let inv = if count > 0.0 { 1.0 / (count + l2) } else { 0.0 };
    sum * inv
}

/// What a split candidate column splits on.
enum ColumnSplit {
    Float {
        feature: usize,
        borders: Vec<f32>,
    },
    OneHot {
        feature: usize,
        values: Vec<u64>,
    },
    Ctr {
        table: std::sync::Arc<CtrTable>,
        kind: CtrKind,
    },
}

/// One split candidate: the bucket of every row.
struct Column {
    split: ColumnSplit,
    buckets: Vec<u8>,
    n_buckets: usize,
    /// CTRs: the (projection, is `Counter`) key of their type and the
    /// projection's combination count, for CatBoost's model size penalty.
    ctr_key: Option<((Projection, bool), usize)>,
}

impl Column {
    fn one_hot(&self) -> bool {
        matches!(self.split, ColumnSplit::OneHot { .. })
    }

    /// Whether a row of bucket `b` goes right at the split `s`.
    #[inline]
    fn goes_right(&self, b: u8, s: usize) -> bool {
        if self.one_hot() {
            b as usize == s
        } else {
            b as usize > s
        }
    }

    /// The model's split; a CTR enters `ctrs` on first use.
    fn tree_split(&self, s: usize, ctrs: &mut CtrModel) -> TreeSplit {
        match &self.split {
            ColumnSplit::Float { feature, borders } => TreeSplit::Float {
                feature: *feature,
                border: borders[s],
            },
            ColumnSplit::OneHot { feature, values } => TreeSplit::OneHot {
                feature: *feature,
                value: values[s],
            },
            ColumnSplit::Ctr { table, kind } => TreeSplit::Ctr {
                ctr: ctrs.register(table, *kind),
                bucket: s as u8,
            },
        }
    }
}

/// The quantized training data: the split candidates in CatBoost's order
/// (numeric features, then one-hot categorical features, then the CTRs of
/// the other categorical features).
struct Quantized {
    columns: Vec<Column>,
    /// Per numeric feature: its borders (empty for categorical ones).
    borders: Vec<Vec<f32>>,
    /// The categorical features with CTRs, for their combinations.
    ctr: Option<CtrSource>,
    /// CatBoost's random draws per tree level for the float and one-hot
    /// features: (`rsm` draws, score instances).
    draws: (usize, usize),
    /// The categorical features (sorted), for per-type feature indices.
    cat_features: Vec<usize>,
}

/// A categorical feature with CTRs: every row's category id (in the order
/// of first appearance, CatBoost's perfect hash) and the categories.
struct CatIds {
    feature: usize,
    ids: Vec<u32>,
    values: Vec<u64>,
}

/// What the CTR columns are built from.
struct CtrSource {
    features: Vec<CatIds>,
    /// The target class of every row and the number of classes.
    classes: Vec<u8>,
    n_classes: usize,
    /// The largest projection complexity (CatBoost's `max_ctr_complexity`).
    max_complexity: usize,
}

/// The CTR columns of a projection, from every row's combination id
/// (`ids`, dense in the order of first appearance) and each combination's
/// key: `Borders` per target border and prior on the online (ordered) class
/// counts, then `Counter` on the final counts.
fn ctr_columns(
    projection: Projection,
    ids: &[u32],
    keys: Vec<Vec<u64>>,
    classes: &[u8],
    n_classes: usize,
) -> Vec<Column> {
    let k = keys.len();
    // Online (ordered) class counts: each row sees the rows before it.
    let mut running = vec![vec![0u32; n_classes]; k];
    let mut online: Vec<u32> = Vec::with_capacity(ids.len() * n_classes);
    for (&id, &c) in ids.iter().zip(classes) {
        let counts = &mut running[id as usize];
        online.extend_from_slice(counts);
        counts[c as usize] += 1;
    }
    let totals: Vec<u32> = running.iter().map(|c| c.iter().sum()).collect();
    let max_count = totals.iter().copied().max().unwrap_or(0);
    let mut table_classes: Vec<(Vec<u64>, Vec<u32>)> = keys.into_iter().zip(running).collect();
    table_classes.sort_by(|a, b| a.0.cmp(&b.0));
    let table = std::sync::Arc::new(CtrTable {
        projection: projection.clone(),
        classes: table_classes,
        max_count,
    });
    default_ctrs(n_classes)
        .into_iter()
        .map(|kind| {
            let buckets = match kind {
                CtrKind::Borders { prior, border } => online
                    .chunks(n_classes)
                    .map(|c| calc_ctr(above_border(c, border), c.iter().sum(), prior))
                    .collect(),
                CtrKind::Counter { prior } => ids
                    .iter()
                    .map(|&id| calc_ctr(totals[id as usize], max_count, prior))
                    .collect(),
            };
            let counter = matches!(kind, CtrKind::Counter { .. });
            Column {
                split: ColumnSplit::Ctr {
                    table: table.clone(),
                    kind,
                },
                buckets,
                n_buckets: CTR_BORDER_COUNT as usize + 1,
                ctr_key: Some(((projection.clone(), counter), k)),
            }
        })
        .collect()
}

impl Quantized {
    /// Every training row's combination of `projection` (dense ids in the
    /// order of first appearance) and each combination's key.
    fn combinations(&self, projection: &Projection) -> (Vec<u32>, Vec<Vec<u64>>) {
        let src = self.ctr.as_ref().expect("CTR features");
        let n = src.classes.len();
        // The components of every row: category ids, then conditions.
        let mut components: Vec<(Vec<u32>, Vec<u64>)> = Vec::new();
        for &f in &projection.cats {
            let cat = src
                .features
                .iter()
                .find(|c| c.feature == f)
                .expect("CTR feature");
            components.push((cat.ids.clone(), cat.values.clone()));
        }
        let condition = |pred: &dyn Fn(&Column) -> Option<usize>| -> Vec<u32> {
            let (column, s) = self
                .columns
                .iter()
                .find_map(|c| pred(c).map(|s| (c, s)))
                .expect("a tree condition's column");
            column
                .buckets
                .iter()
                .map(|&b| u32::from(column.goes_right(b, s)))
                .collect()
        };
        for &(f, bits) in &projection.floats {
            let rows = condition(&|c| match &c.split {
                ColumnSplit::Float { feature, borders } if *feature == f => {
                    borders.iter().position(|b| b.to_bits() == bits)
                }
                _ => None,
            });
            components.push((rows, vec![0, 1]));
        }
        for &(f, v) in &projection.one_hots {
            let rows = condition(&|c| match &c.split {
                ColumnSplit::OneHot { feature, values } if *feature == f => {
                    values.iter().position(|&x| x == v)
                }
                _ => None,
            });
            components.push((rows, vec![0, 1]));
        }
        // Combine the components pairwise; the last pass numbers the
        // combinations in the order of their first row.
        let mut ids = components[0].0.clone();
        for (rows, _) in &components[1..] {
            let mut index = std::collections::HashMap::new();
            for (id, &r) in ids.iter_mut().zip(rows) {
                let next = index.len() as u32;
                *id = *index.entry((*id, r)).or_insert(next);
            }
        }
        let n_ids = ids.iter().map(|&i| i as usize + 1).max().unwrap_or(0);
        let mut keys = vec![Vec::new(); n_ids];
        for i in 0..n {
            let key = &mut keys[ids[i] as usize];
            if key.is_empty() {
                *key = components.iter().map(|(r, v)| v[r[i] as usize]).collect();
            }
        }
        (ids, keys)
    }

    /// The CTR columns of `projection`.
    fn projection_columns(&self, projection: Projection) -> Vec<Column> {
        let src = self.ctr.as_ref().expect("CTR features");
        let (ids, keys) = self.combinations(&projection);
        ctr_columns(projection, &ids, keys, &src.classes, src.n_classes)
    }
}

/// How categorical features are used.
#[derive(Clone, Copy)]
struct CategoricalParams<'a> {
    features: &'a [usize],
    one_hot_max_size: usize,
    /// The target class (`0..n_classes`) of every row and the number of
    /// classes, for the `Borders` CTRs.
    classes: Option<(&'a [u8], usize)>,
    /// CatBoost's `max_ctr_complexity`: 1 for no combinations.
    max_ctr_complexity: usize,
}

fn quantize(
    x: ArrayView2<f64>,
    border_count: usize,
    cat: CategoricalParams<'_>,
) -> Result<Quantized> {
    use rayon::prelude::*;
    let p = x.ncols();
    if let Some(&f) = cat.features.iter().find(|&&f| f >= p) {
        return Err(RustMlError::InvalidParameter(format!(
            "cat_features: feature {f} out of range for {p} features"
        )));
    }
    let is_cat = |f: usize| cat.features.contains(&f);
    let float: Vec<(Vec<f32>, Vec<u8>)> = (0..p)
        .into_par_iter()
        .map(|f| {
            if is_cat(f) {
                return (Vec::new(), Vec::new());
            }
            let col = x.column(f);
            let borders = feature_borders(col, border_count);
            let buckets = col.iter().map(|&v| bucket(v, &borders)).collect();
            (borders, buckets)
        })
        .collect();
    let mut columns = Vec::new();
    let mut borders = Vec::with_capacity(p);
    for (f, (b, buckets)) in float.into_iter().enumerate() {
        if !is_cat(f) && !b.is_empty() {
            columns.push(Column {
                split: ColumnSplit::Float {
                    feature: f,
                    borders: b.clone(),
                },
                buckets,
                n_buckets: b.len() + 1,
                ctr_key: None,
            });
        }
        borders.push(b);
    }

    let mut cat_features: Vec<usize> = cat.features.to_vec();
    cat_features.sort_unstable();
    cat_features.dedup();
    // Categories in the order of their first appearance (CatBoost's
    // perfect hash).
    let mut ctr_features = Vec::new();
    // Categorical features with two values (CatBoost packs them as binary
    // features), and whether they are one-hot candidates.
    let mut two_valued: Vec<bool> = Vec::new();
    for &f in &cat_features {
        let mut index = std::collections::HashMap::new();
        let mut values = Vec::new();
        let ids: Vec<u32> = x
            .column(f)
            .iter()
            .map(|&v| {
                let key = category(v);
                *index.entry(key).or_insert_with(|| {
                    values.push(key);
                    values.len() as u32 - 1
                })
            })
            .collect();
        if values.len() == 2 {
            two_valued.push(values.len() <= cat.one_hot_max_size);
        }
        if values.len() <= cat.one_hot_max_size {
            if values.len() >= 2 {
                columns.push(Column {
                    n_buckets: values.len(),
                    split: ColumnSplit::OneHot { feature: f, values },
                    buckets: ids.iter().map(|&i| i as u8).collect(),
                    ctr_key: None,
                });
            }
        } else {
            ctr_features.push((f, ids, values));
        }
    }

    let draws = feature_draws(&columns, &two_valued);
    let mut q = Quantized {
        columns,
        borders,
        ctr: None,
        draws,
        cat_features: cat_features.clone(),
    };
    if !ctr_features.is_empty() {
        let (classes, n_classes) = cat.classes.ok_or_else(|| {
            RustMlError::InvalidParameter("categorical features with CTRs need a target".into())
        })?;
        q.ctr = Some(CtrSource {
            features: ctr_features
                .into_iter()
                .map(|(feature, ids, values)| CatIds {
                    feature,
                    ids,
                    values,
                })
                .collect(),
            classes: classes.to_vec(),
            n_classes,
            max_complexity: cat.max_ctr_complexity,
        });
        let simple: Vec<usize> = q
            .ctr
            .as_ref()
            .unwrap()
            .features
            .iter()
            .map(|c| c.feature)
            .collect();
        for f in simple {
            let columns = q.projection_columns(Projection::single(f));
            q.columns.extend(columns);
        }
    }
    Ok(q)
}

/// CatBoost's random draws per tree level for the float and one-hot
/// features (`SelectCandidatesAndCleanupStatsFromPrevTree` and
/// `SelectBestCandidate`): one `rsm` draw per feature, and one score
/// instance per feature, except that the binary ones (one border, two
/// categories) are packed eight to a candidate (`TPackedBinaryFeaturesData`:
/// floats, then categoricals; `two_valued` says which categoricals with two
/// values are one-hot candidates).
fn feature_draws(columns: &[Column], two_valued: &[bool]) -> (usize, usize) {
    let mut rsm = 0;
    let mut single = 0;
    let mut packed: Vec<bool> = Vec::new();
    for c in columns {
        match &c.split {
            ColumnSplit::Float { borders, .. } => {
                rsm += 1;
                if borders.len() == 1 {
                    packed.push(true);
                } else {
                    single += 1;
                }
            }
            ColumnSplit::OneHot { values, .. } => {
                rsm += 1;
                if values.len() != 2 {
                    single += 1;
                }
            }
            ColumnSplit::Ctr { .. } => {}
        }
    }
    packed.extend_from_slice(two_valued);
    let packs = packed.chunks(8).filter(|p| p.iter().any(|&c| c)).count();
    (rsm, single + packs)
}

/// The derivatives a tree's splits are scored on. Plain boosting: all rows
/// (`body = 0`). Ordered boosting: one part per body / tail of the fold,
/// whose leaf estimates come from the body rows and are scored on the tail
/// rows (CatBoost's `CalcStatsKernel` and `UpdateScoreBinKernelOrdered`).
struct ScorePart<'a> {
    /// `w · der1` of the rows `0..tail`, `k` per row.
    wder: &'a [f64],
    /// The same times the bootstrap weights (the scored rows' derivatives).
    swder: &'a [f64],
    body: usize,
    tail: usize,
    scaled_l2: f64,
}

/// Bucket statistics: weighted derivative and weight sums of the scored
/// rows, and (ordered) derivative and weight sums of the body rows.
#[derive(Clone, Copy, Default)]
struct BucketStats {
    swd: f64,
    sw: f64,
    sd: f64,
    count: f64,
}

impl BucketStats {
    fn add(&mut self, o: &Self) {
        self.swd += o.swd;
        self.sw += o.sw;
        self.sd += o.sd;
        self.count += o.count;
    }
    fn remove(&mut self, o: &Self) {
        self.swd -= o.swd;
        self.sw -= o.sw;
        self.sd -= o.sd;
        self.count -= o.count;
    }
}

/// A split candidate: a column of the quantized data, or one of the CTR
/// columns of a feature combination.
#[derive(Clone)]
enum ColumnRef {
    Base(usize),
    Combination(std::sync::Arc<Vec<Column>>, usize),
}

impl ColumnRef {
    fn get<'a>(&'a self, q: &'a Quantized) -> &'a Column {
        match self {
            ColumnRef::Base(c) => &q.columns[*c],
            ColumnRef::Combination(columns, c) => &columns[*c],
        }
    }
}

/// The CTR columns of the feature combinations for the next level of a tree
/// (CatBoost's `AddTreeCtrs`): the projection of the tree's float and
/// one-hot splits and those of its CTR splits, each extended by every
/// categorical feature with CTRs, up to `max_ctr_complexity`. The columns
/// of a projection are cached across trees.
type CombinationCache = std::collections::HashMap<Projection, std::sync::Arc<Vec<Column>>>;

/// Also returns CatBoost's `rsm` draws for them: one per base projection
/// and categorical feature with CTRs, whatever the complexity limit.
fn combination_columns(
    q: &Quantized,
    splits: &[(ColumnRef, usize)],
    cache: &mut CombinationCache,
) -> (Vec<std::sync::Arc<Vec<Column>>>, usize) {
    let Some(src) = &q.ctr else {
        return (Vec::new(), 0);
    };
    let mut conditions = Projection::default();
    let mut seen = Vec::new();
    for (c, s) in splits {
        match &c.get(q).split {
            ColumnSplit::Float { feature, borders } => {
                conditions.floats.push((*feature, borders[*s].to_bits()));
            }
            ColumnSplit::OneHot { feature, values } => {
                conditions.one_hots.push((*feature, values[*s]));
            }
            ColumnSplit::Ctr { table, .. } => seen.push(table.projection.clone()),
        }
    }
    // The conditions keep the tree's order (`GetBinFeatures`), and the
    // bases are visited in the order of CatBoost's `THashSet`.
    seen.insert(0, conditions);
    let mut set = YHashSet::default();
    for base in seen {
        let hash = q.projection_hash(&base);
        set.insert(base, hash);
    }
    let bases: Vec<&Projection> = set.iter().filter(|p| !p.is_empty()).collect();
    let draws = bases.len() * src.features.len();
    if src.max_complexity < 2 {
        return (Vec::new(), draws);
    }
    let mut added: Vec<Projection> = Vec::new();
    for base in bases {
        for cat in &src.features {
            if base.cats.contains(&cat.feature) {
                continue;
            }
            let mut projection = base.clone();
            projection.cats.push(cat.feature);
            projection.cats.sort_unstable();
            if projection.complexity() > src.max_complexity || added.contains(&projection) {
                continue;
            }
            added.push(projection);
        }
    }
    let columns = added
        .into_iter()
        .map(|projection| {
            cache
                .entry(projection.clone())
                .or_insert_with(|| std::sync::Arc::new(q.projection_columns(projection)))
                .clone()
        })
        .collect();
    (columns, draws)
}

/// `IntHash` of Yandex `util` (Thomas Wang's 64-bit mix).
fn int_hash64(mut key: u64) -> u64 {
    key = key.wrapping_add(!(key << 32));
    key ^= key >> 22;
    key = key.wrapping_add(!(key << 13));
    key ^= key >> 8;
    key = key.wrapping_add(key << 3);
    key ^= key >> 15;
    key = key.wrapping_add(!(key << 27));
    key ^= key >> 31;
    key
}

/// CatBoost's `TVecHash`: `984121 · h + item` in 32 bits from 1988712,
/// returned as an `int` (hashed as `size_t`, so sign-extended).
fn vec_hash(items: impl Iterator<Item = u64>) -> u64 {
    let mut res: u32 = 1988712;
    for h in items {
        res = u64::from(984121u32.wrapping_mul(res)).wrapping_add(h) as u32;
    }
    res as i32 as i64 as u64
}

impl Quantized {
    /// CatBoost's `TProjection::GetHash`: `MultiHash` of the `TVecHash`es
    /// of its categorical features, its `TBinFeature`s (float index, border
    /// index) and its `TOneHotSplit`s (categorical index, category index),
    /// all by per-type feature index.
    fn projection_hash(&self, p: &Projection) -> u64 {
        let float_index =
            |f: usize| (0..f).filter(|g| !self.cat_features.contains(g)).count() as u64;
        let cat_index = |f: usize| self.cat_features.iter().filter(|&&g| g < f).count() as u64;
        let cats = vec_hash(p.cats.iter().map(|&f| cat_index(f)));
        let bins = vec_hash(p.floats.iter().map(|&(f, bits)| {
            let s = self.borders[f]
                .iter()
                .position(|b| b.to_bits() == bits)
                .unwrap_or(0);
            int_hash64(s as u64) ^ float_index(f)
        }));
        if p.one_hots.is_empty() {
            return int_hash64(bins) ^ cats;
        }
        let one_hots = vec_hash(p.one_hots.iter().map(|&(f, v)| {
            let s = self
                .columns
                .iter()
                .find_map(|c| match &c.split {
                    ColumnSplit::OneHot { feature, values } if *feature == f => {
                        values.iter().position(|&x| x == v)
                    }
                    _ => None,
                })
                .unwrap_or(0);
            int_hash64(s as u64) ^ cat_index(f)
        }));
        int_hash64(int_hash64(one_hots) ^ bins) ^ cats
    }
}

/// The iteration order of Yandex `util`'s `THashSet` (`THashTable`): prime
/// bucket counts, a node inserted at the head of its bucket, a rehash
/// before an insertion would exceed the bucket count, buckets visited in
/// order.
#[derive(Default)]
struct YHashSet {
    items: Vec<Projection>,
    /// Per bucket: item indices, head first (one static bucket at first).
    buckets: Vec<Vec<(usize, u64)>>,
}

impl YHashSet {
    const PRIMES: [usize; 12] = [7, 17, 29, 53, 97, 193, 389, 769, 1543, 3079, 6151, 12289];

    fn insert(&mut self, item: Projection, hash: u64) {
        if self.buckets.is_empty() {
            self.buckets.push(Vec::new());
        }
        // `reserve(num_elements + 1)`.
        let (old, hint) = (self.buckets.len(), self.items.len() + 1);
        if hint + 1 > old && !(old != 1 && hint <= old) {
            let size = *Self::PRIMES
                .iter()
                .find(|&&p| p >= hint + 1)
                .unwrap_or(&12289);
            if size > old {
                let mut buckets = vec![Vec::new(); size];
                for chain in std::mem::take(&mut self.buckets) {
                    for node in chain {
                        buckets[(node.1 % size as u64) as usize].insert(0, node);
                    }
                }
                self.buckets = buckets;
            }
        }
        let n = self.buckets.len() as u64;
        let bucket = if n == 1 { 0 } else { (hash % n) as usize };
        if self.buckets[bucket]
            .iter()
            .any(|&(i, _)| self.items[i] == item)
        {
            return;
        }
        self.items.push(item);
        self.buckets[bucket].insert(0, (self.items.len() - 1, hash));
    }

    fn iter(&self) -> impl Iterator<Item = &Projection> {
        self.buckets.iter().flatten().map(|&(i, _)| &self.items[i])
    }
}

/// CatBoost's candidate list of a level (after `CompressCandidates`): one
/// entry per non-binary float or one-hot feature, per pack of eight binary
/// features (`TPackedBinaryFeaturesData`: one-border floats, then
/// two-valued categoricals, packed whether candidates or not), per simple
/// CTR feature and per tree CTR projection. An entry holds its candidates,
/// each the (column, split) bins it chooses its best from.
fn candidate_entries(q: &Quantized, candidates: &[ColumnRef]) -> Vec<Vec<Vec<(usize, usize)>>> {
    let mut single = Vec::new();
    let mut packed: Vec<Option<(usize, usize)>> = Vec::new();
    let mut packed_cats: Vec<(usize, Option<(usize, usize)>)> = Vec::new();
    let mut ctr_groups: Vec<(Projection, Vec<(usize, usize)>)> = Vec::new();
    for (c, cand) in candidates.iter().enumerate() {
        let column = cand.get(q);
        match &column.split {
            ColumnSplit::Float { borders, .. } if borders.len() == 1 => packed.push(Some((c, 0))),
            ColumnSplit::Float { borders, .. } => {
                single.push(vec![(0..borders.len()).map(|s| (c, s)).collect()]);
            }
            // A two-valued categorical's bit is its second category.
            ColumnSplit::OneHot { feature, values } if values.len() == 2 => {
                packed_cats.push((*feature, Some((c, 1))));
            }
            ColumnSplit::OneHot { values, .. } => {
                single.push(vec![(0..values.len()).map(|s| (c, s)).collect()]);
            }
            ColumnSplit::Ctr { table, .. } => {
                match ctr_groups.iter_mut().find(|(p, _)| *p == table.projection) {
                    Some((_, group)) => group.push((c, 0)),
                    None => ctr_groups.push((table.projection.clone(), vec![(c, 0)])),
                }
            }
        }
    }
    // Two-valued categoricals with CTRs take a bit too, as no candidate.
    if let Some(src) = &q.ctr {
        for cat in src.features.iter().filter(|c| c.values.len() == 2) {
            packed_cats.push((cat.feature, None));
        }
    }
    packed_cats.sort_by_key(|&(f, _)| f);
    packed.extend(packed_cats.into_iter().map(|(_, c)| c));
    let mut entries = single;
    for pack in packed.chunks(8) {
        let bins: Vec<(usize, usize)> = pack.iter().flatten().copied().collect();
        if !bins.is_empty() {
            entries.push(vec![bins]);
        }
    }
    for (_, group) in ctr_groups {
        entries.push(
            group
                .into_iter()
                .map(|(c, _)| {
                    let n_splits = candidates[c].get(q).n_buckets - 1;
                    (0..n_splits).map(|s| (c, s)).collect()
                })
                .collect(),
        );
    }
    entries
}

/// Grow one oblivious tree on the weighted first derivatives of the
/// `parts` (`k` approximations per row) and the weights `w`; returns the
/// splits as (column, split index) and every row's leaf. The scores of the
/// `k` dimensions add up (CatBoost's `CalcScores`).
#[allow(clippy::too_many_arguments)]
fn grow_tree(
    q: &Quantized,
    parts: &[ScorePart<'_>],
    k: usize,
    w: &[f64],
    ordered: bool,
    depth: usize,
    score_function: CatBoostScoreFunction,
    model_size_reg: f64,
    used_ctrs: &mut std::collections::HashSet<(Projection, bool)>,
    cache: &mut CombinationCache,
    rng: &mut CatRng,
    score_std: f64,
    sample: &Sample,
) -> (Vec<(ColumnRef, usize)>, Vec<u32>) {
    use rayon::prelude::*;
    let n = w.len();
    let mut score_before_split = 0.0;
    let mut splits: Vec<(ColumnRef, usize)> = Vec::new();
    let mut leaf_of = vec![0u32; n];
    for d in 0..depth {
        let leaves = 1usize << d;
        let (combinations, tree_ctr_draws) = combination_columns(q, &splits, cache);
        let candidates: Vec<ColumnRef> = (0..q.columns.len())
            .map(ColumnRef::Base)
            .chain(
                combinations
                    .iter()
                    .flat_map(|c| (0..c.len()).map(|i| ColumnRef::Combination(c.clone(), i))),
            )
            .collect();
        // CatBoost's draws of this level, in its order: `rsm` per feature,
        // per simple CTR feature and per tree CTR, then the seed of the
        // candidates' score noise.
        let n_ctr_features = q.ctr.as_ref().map_or(0, |c| c.features.len());
        for _ in 0..q.draws.0 + n_ctr_features + tree_ctr_draws {
            rng.real1();
        }
        let noise_seed = rng.gen();
        // CatBoost's `GetCatFeatureWeight`: a CTR type not yet in the model
        // scores `(1 + combinations / max combinations)^-model_size_reg`
        // times less, the maximum over this level's CTR candidates.
        let max_categories = candidates
            .iter()
            .filter_map(|c| c.get(q).ctr_key.as_ref().map(|(_, k)| *k))
            .max()
            .unwrap_or(1)
            .max(1);
        // The score of every split of every column.
        let per_column: Vec<Option<Vec<f64>>> = candidates
            .par_iter()
            .map(|c| {
                let column = c.get(q);
                let nb = column.n_buckets;
                let one_hot = column.one_hot();
                let n_splits = if one_hot { nb } else { nb - 1 };
                if nb < 2 {
                    return None;
                }
                // Per part and dimension: leaves × buckets.
                let mut hist = vec![BucketStats::default(); parts.len() * k * leaves * nb];
                for (pi, part) in parts.iter().enumerate() {
                    for i in 0..part.tail {
                        if !sample.taken[i] {
                            continue;
                        }
                        let cell = leaf_of[i] as usize * nb + column.buckets[i] as usize;
                        for d in 0..k {
                            let h = &mut hist[(pi * k + d) * leaves * nb + cell];
                            if i < part.body {
                                h.sd += part.wder[i * k + d];
                                h.count += w[i];
                            } else {
                                h.swd += part.swder[i * k + d];
                                h.sw += sample.weights[i];
                            }
                        }
                    }
                }
                let mut num = vec![0.0f64; n_splits];
                let mut den = vec![1e-100f64; n_splits];
                for (pi, part) in parts.iter().enumerate() {
                    let l2 = part.scaled_l2;
                    let mut add = |s: usize, t: &BucketStats, f: &BucketStats| {
                        // Plain: averages of the scored rows; ordered: of
                        // the body rows.
                        let (at, af) = if ordered {
                            (average(t.sd, t.count, l2), average(f.sd, f.count, l2))
                        } else {
                            (average(t.swd, t.sw, l2), average(f.swd, f.sw, l2))
                        };
                        match score_function {
                            // CatBoost's SSE kernel: `prev + (false + true)`.
                            CatBoostScoreFunction::Cosine => {
                                num[s] += af * f.swd + at * t.swd;
                                den[s] += af * af * f.sw + at * at * t.sw;
                            }
                            // `AddLeaf(true)` then `AddLeaf(false)`.
                            CatBoostScoreFunction::L2 => {
                                num[s] += at * t.swd;
                                num[s] += af * f.swd;
                            }
                        }
                    };
                    // CatBoost's `CalcScoresForLeaf`.
                    for leaf in (0..k).flat_map(|d| (0..leaves).map(move |l| (d, l))) {
                        let start = ((pi * k + leaf.0) * leaves + leaf.1) * nb;
                        let buckets = &hist[start..start + nb];
                        let mut all = BucketStats::default();
                        for b in buckets {
                            all.add(b);
                        }
                        if one_hot {
                            let mut f = all;
                            for (s, b) in buckets.iter().enumerate() {
                                f.remove(b);
                                add(s, b, &f);
                                f.add(b);
                            }
                        } else {
                            let (mut t, mut f) = (all, BucketStats::default());
                            for (s, b) in buckets[..n_splits].iter().enumerate() {
                                f.add(b);
                                t.remove(b);
                                add(s, &t, &f);
                            }
                        }
                    }
                }
                Some(
                    (0..n_splits)
                        .map(|s| match score_function {
                            CatBoostScoreFunction::Cosine => num[s] / den[s].sqrt(),
                            CatBoostScoreFunction::L2 => num[s],
                        })
                        .collect(),
                )
            })
            .collect();
        // CatBoost's candidate list: the non-binary float and one-hot
        // features, the packs of binary ones, the simple CTRs per feature,
        // then the tree CTRs per projection. Each entry draws the noise of
        // its splits from its own generator (`SetBestScore`), each
        // candidate a noisy instance of its best score from the main one
        // (`SelectBestCandidate`).
        let entries = candidate_entries(q, &candidates);
        let mut best: Option<(f64, usize, usize)> = None;
        let mut best_gain = f64::NEG_INFINITY;
        let mut best_score = f64::MIN;
        for (t, entry) in entries.iter().enumerate() {
            let mut local = CatRng::new(noise_seed.wrapping_add(t as u64));
            local.advance(10);
            for bins in entry {
                let mut val = f64::MIN;
                let mut chosen: Option<(usize, usize)> = None;
                let mut best_instance = f64::MIN;
                for &(c, s) in bins {
                    let Some(scores) = &per_column[c] else {
                        continue;
                    };
                    let instance = scores[s] + local.std_normal() * score_std;
                    if instance > best_instance {
                        best_instance = instance;
                        val = scores[s];
                        chosen = Some((c, s));
                    }
                }
                let mut score = val + rng.std_normal() * score_std;
                let Some((c, s)) = chosen else {
                    continue;
                };
                if let Some((key, k)) = &candidates[c].get(q).ctr_key {
                    if !used_ctrs.contains(key) {
                        score *= (1.0 + *k as f64 / max_categories as f64).powf(-model_size_reg);
                    }
                }
                let gain = score - score_before_split;
                if gain > best_gain {
                    best_gain = gain;
                    best_score = score;
                    best = Some((score, c, s));
                }
            }
        }
        let Some((_, c, s)) = best else {
            break;
        };
        score_before_split = best_score;
        let chosen = candidates[c].clone();
        let column = chosen.get(q);
        for i in 0..n {
            leaf_of[i] |= u32::from(column.goes_right(column.buckets[i], s)) << d;
        }
        if let Some((key, _)) = &column.ctr_key {
            used_ctrs.insert(key.clone());
        }
        splits.push((chosen, s));

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
                    .map(|(d, (c, s))| {
                        let column = c.get(q);
                        u32::from(column.goes_right(column.buckets[i], *s)) << d
                    })
                    .sum();
            }
            break;
        }
    }
    (splits, leaf_of)
}

/// CatBoost's body / tail prefixes of a fold for ordered boosting: the
/// first body is 1 row (`min(100, n / 50)` above 500 rows), every tail
/// ends at `ceil(body · fold_len_multiplier)`.
fn body_tails(n: usize, multiplier: f64, weights: &[f64]) -> Vec<(usize, usize, f64)> {
    let mut left = if n > 500 { 100.min(n / 50) } else { 1 }.min(n);
    let mut out = Vec::new();
    while out.is_empty() || left < n {
        let body = left;
        let tail = ((left as f64 * multiplier).ceil() as usize).min(n);
        let body_weight: f64 = weights[..body].iter().sum();
        out.push((body, tail, body_weight));
        left = tail;
    }
    out
}

/// Per-leaf sums of the weighted derivatives and of the weights.
struct LeafSums {
    der: Vec<f64>,
    der2: Vec<f64>,
    w: Vec<f64>,
}

impl LeafSums {
    fn of(leaf_of: &[u32], n_leaves: usize, ders: &Ders, weights: &[f64], rows: usize) -> Self {
        let (k, h) = (ders.k, ders.h);
        let mut s = Self {
            der: vec![0.0; n_leaves * k],
            der2: vec![0.0; n_leaves * h],
            w: vec![0.0; n_leaves],
        };
        for i in 0..rows {
            s.add(leaf_of[i] as usize, ders, i, weights[i]);
        }
        s
    }

    fn add(&mut self, leaf: usize, ders: &Ders, i: usize, w: f64) {
        let (k, h) = (ders.k, ders.h);
        for d in 0..k {
            self.der[leaf * k + d] += ders.der[i * k + d];
        }
        for d in 0..h {
            self.der2[leaf * h + d] += ders.der2[i * h + d];
        }
        self.w[leaf] += w;
    }

    /// The (unscaled) delta of `leaf` into `out`, with the L2
    /// regularization `l2` scaled by the mean weight `sum_w / count`.
    fn delta(
        &self,
        leaf: usize,
        l2: f64,
        sum_w: f64,
        count: f64,
        p: &TrainParams,
        out: &mut [f64],
    ) {
        let k = out.len();
        let h = k * (k + 1) / 2;
        let der = &self.der[leaf * k..(leaf + 1) * k];
        let scaled_l2 = l2 * (sum_w / count);
        match p.leaf_estimation {
            CatBoostLeafEstimation::Newton if k == 1 => {
                out[0] = der[0] / (-self.der2[leaf] + scaled_l2);
            }
            CatBoostLeafEstimation::Newton => {
                let der2 = &self.der2[leaf * h..(leaf + 1) * h];
                newton_multi(der, der2, l2, sum_w, count, out);
            }
            CatBoostLeafEstimation::Gradient => {
                for (o, &d) in out.iter_mut().zip(der) {
                    *o = average(d, self.w[leaf], scaled_l2);
                }
            }
        }
    }
}

/// The rows a leaf estimation runs on: the approximations `approx` of the
/// rows `0..rows` (`k` each), their leaves and the L2 scaling statistics
/// (weight sum, row count).
struct LeafProblem<'a> {
    loss: Loss,
    target: &'a [f64],
    approx: &'a [f64],
    weights: &'a [f64],
    leaf_of: &'a [u32],
    n_leaves: usize,
    rows: usize,
    stats: (f64, f64),
    /// The model's leaves (CatBoost's `CalcLeafValues`): the metric is taken
    /// at the moved approximations plus the step. A learning fold
    /// (`CalcApproxDelta`): at `approx` plus the summed steps.
    moving: bool,
}

/// CatBoost's leaf estimation (`FastGradientWalker`): `leaf_iterations`
/// Newton or Gradient steps, each from the derivatives at the moved
/// approximations. With backtracking a step is halved until the metric
/// improves, every try using up an iteration. `accepted` sees each taken
/// step and that iteration's leaf sums. Returns the summed leaf values
/// (`k` per leaf, before the learning rate) and the number of steps
/// computed.
fn estimate_leaves(
    lp: &LeafProblem<'_>,
    p: &TrainParams,
    mut accepted: impl FnMut(&[f64], &mut LeafSums),
) -> (Vec<f64>, usize) {
    let k = lp.loss.dims();
    let (n_leaves, rows) = (lp.n_leaves, lp.rows);
    let mut steps = 0;
    let backtracking = p.backtracking && p.leaf_iterations > 1;
    let metric = |approx: &[f64], leaves: &[f64]| {
        lp.loss
            .metric_with_leaves(lp.target, approx, leaves, lp.leaf_of, lp.weights, rows)
    };
    let mut total = vec![0.0; n_leaves * k];
    let mut point = lp.approx[..rows * k].to_vec();
    let mut value = if backtracking {
        metric(&point, &total)
    } else {
        0.0
    };
    let mut it = 0;
    while it < p.leaf_iterations {
        steps += 1;
        let ders = lp.loss.all_ders(lp.target, &point, lp.weights, rows);
        let mut sums = LeafSums::of(lp.leaf_of, n_leaves, &ders, lp.weights, rows);
        let mut step = vec![0.0; n_leaves * k];
        for (leaf, o) in step.chunks_mut(k).enumerate() {
            sums.delta(leaf, p.l2_leaf_reg, lp.stats.0, lp.stats.1, p, o);
        }
        let take = |scaled: &[f64], total: &mut [f64], point: &mut [f64]| {
            for (t, s) in total.iter_mut().zip(scaled) {
                *t += s;
            }
            for i in 0..rows {
                let leaf = lp.leaf_of[i] as usize;
                for d in 0..k {
                    point[i * k + d] += scaled[leaf * k + d];
                }
            }
        };
        if !backtracking {
            take(&step, &mut total, &mut point);
            accepted(&step, &mut sums);
            it += 1;
            continue;
        }
        let mut scale = 1.0;
        loop {
            let scaled: Vec<f64> = step.iter().map(|&v| scale * v).collect();
            let after = if lp.moving {
                metric(&point, &scaled)
            } else {
                let candidate: Vec<f64> = total.iter().zip(&scaled).map(|(t, s)| t + s).collect();
                metric(lp.approx, &candidate)
            };
            if after < value {
                value = after;
                take(&scaled, &mut total, &mut point);
                accepted(&scaled, &mut sums);
                break;
            }
            scale /= 2.0;
            it += 1;
            if it >= p.leaf_iterations {
                break;
            }
        }
        it += 1;
    }
    (total, steps)
}

/// A learning fold (CatBoost's `TFold`): the training rows in the fold's
/// permutation, with the CTRs computed online in that order, its body /
/// tail prefixes (plain boosting: one, of all rows) and their
/// approximations.
struct LearnFold {
    q: Quantized,
    target: Vec<f64>,
    weights: Vec<f64>,
    tails: Vec<(usize, usize, f64)>,
    approx: Vec<Vec<f64>>,
    cache: CombinationCache,
}

/// `v` in the order `perm` (`out[i] = v[perm[i] * k ..]`, `k` per row).
fn permute_rows(v: &[f64], perm: &[usize], k: usize) -> Vec<f64> {
    perm.iter()
        .flat_map(|&r| v[r * k..(r + 1) * k].iter().copied())
        .collect()
}

impl Quantized {
    /// The data in the order `perm` (row `i` is the original row
    /// `perm[i]`): the same columns, the CTRs recomputed online in that
    /// order.
    fn permuted(&self, perm: &[usize]) -> Quantized {
        let rows = |b: &[u8]| perm.iter().map(|&r| b[r]).collect::<Vec<u8>>();
        let mut columns: Vec<Column> = self
            .columns
            .iter()
            .filter(|c| !matches!(c.split, ColumnSplit::Ctr { .. }))
            .map(|c| Column {
                split: match &c.split {
                    ColumnSplit::Float { feature, borders } => ColumnSplit::Float {
                        feature: *feature,
                        borders: borders.clone(),
                    },
                    ColumnSplit::OneHot { feature, values } => ColumnSplit::OneHot {
                        feature: *feature,
                        values: values.clone(),
                    },
                    ColumnSplit::Ctr { .. } => unreachable!(),
                },
                buckets: rows(&c.buckets),
                n_buckets: c.n_buckets,
                ctr_key: None,
            })
            .collect();
        let ctr = self.ctr.as_ref().map(|src| CtrSource {
            features: src
                .features
                .iter()
                .map(|c| CatIds {
                    feature: c.feature,
                    ids: perm.iter().map(|&r| c.ids[r]).collect(),
                    values: c.values.clone(),
                })
                .collect(),
            classes: perm.iter().map(|&r| src.classes[r]).collect(),
            n_classes: src.n_classes,
            max_complexity: src.max_complexity,
        });
        let mut q = Quantized {
            columns: Vec::new(),
            borders: self.borders.clone(),
            ctr,
            draws: self.draws,
            cat_features: self.cat_features.clone(),
        };
        std::mem::swap(&mut q.columns, &mut columns);
        if let Some(src) = &q.ctr {
            let features: Vec<usize> = src.features.iter().map(|c| c.feature).collect();
            for f in features {
                let c = q.projection_columns(Projection::single(f));
                q.columns.extend(c);
            }
        }
        q
    }

    /// The split `split` (of any fold) as a split of this data.
    fn resolve(&self, split: &ColumnRef, cache: &mut CombinationCache) -> ColumnRef {
        match split {
            ColumnRef::Base(c) => ColumnRef::Base(*c),
            ColumnRef::Combination(columns, j) => {
                let ColumnSplit::Ctr { table, .. } = &columns[*j].split else {
                    unreachable!("combinations are CTRs")
                };
                let own = cache
                    .entry(table.projection.clone())
                    .or_insert_with(|| {
                        std::sync::Arc::new(self.projection_columns(table.projection.clone()))
                    })
                    .clone();
                ColumnRef::Combination(own, *j)
            }
        }
    }

    /// Every row's leaf under `splits` (resolved in this data).
    fn leaves(&self, splits: &[(ColumnRef, usize)], cache: &mut CombinationCache) -> Vec<u32> {
        let resolved: Vec<(ColumnRef, usize)> = splits
            .iter()
            .map(|(c, s)| (self.resolve(c, cache), *s))
            .collect();
        let n = self.columns.first().map_or(0, |c| c.buckets.len());
        (0..n)
            .map(|i| {
                resolved
                    .iter()
                    .enumerate()
                    .map(|(d, (c, s))| {
                        let column = c.get(self);
                        u32::from(column.goes_right(column.buckets[i], *s)) << d
                    })
                    .sum()
            })
            .collect()
    }
}

/// The rows a tree is scored on (CatBoost's bootstrap, per tree): every
/// row's bootstrap weight, its weight for the scores (times its sample
/// weight, in float32) and whether it is taken.
struct Sample {
    boot: Vec<f64>,
    weights: Vec<f64>,
    taken: Vec<bool>,
}

/// libc++'s `std::partition` (the bidirectional algorithm): the elements
/// with `pred` first; returns their count.
fn libcxx_partition(v: &mut [f64], pred: impl Fn(f64) -> bool) -> usize {
    let (mut first, mut last) = (0, v.len());
    loop {
        loop {
            if first == last {
                return first;
            }
            if !pred(v[first]) {
                break;
            }
            first += 1;
        }
        loop {
            last -= 1;
            if first == last {
                return first;
            }
            if pred(v[last]) {
                break;
            }
        }
        v.swap(first, last);
        first += 1;
    }
}

/// CatBoost's `TMvsSampler::CalculateThreshold`.
fn mvs_threshold(v: &mut [f64], mut small: f64, mut large: usize, size: f64) -> f64 {
    let threshold = v[0];
    let mid_begin = libcxx_partition(v, |c| c < threshold);
    let mid_end = mid_begin + libcxx_partition(&mut v[mid_begin..], |c| c <= threshold);
    let small_update: f64 = v[..mid_begin].iter().fold(0.0, |a, &b| a + b);
    let large_update = v.len() - mid_end;
    let middle = mid_end - mid_begin;
    let sum_middle = middle as f64 * threshold;
    let estimated = (small + small_update) / threshold + (large + large_update + middle) as f64;
    if estimated > size {
        if mid_end != v.len() {
            small += sum_middle + small_update;
            mvs_threshold(&mut v[mid_end..], small, large, size)
        } else {
            (small + small_update + sum_middle) / (size - large as f64)
        }
    } else if mid_begin != 0 {
        large += large_update + middle;
        mvs_threshold(&mut v[..mid_begin], small, large, size)
    } else {
        small / (size - large as f64 - middle as f64 - large_update as f64)
    }
}

/// CatBoost's `Bootstrap` of a tree on a fold (`sampling_frequency=PerTree`):
/// the bootstrap weights, the scores' weights (`CalcWeightedData`) and the
/// taken rows (`TCalcScoreFold::Sample`).
#[allow(clippy::too_many_arguments)]
fn bootstrap(
    rng: &mut CatRng,
    ders: &[Ders],
    tails: &[(usize, usize, f64)],
    learn_weights: &[f64],
    previous: Option<&[f64]>,
    k: usize,
    p: &TrainParams,
) -> Sample {
    let n = learn_weights.len();
    let mut boot = vec![1.0f64; n];
    let mut taken = vec![true; n];
    match p.bootstrap {
        CatBoostBootstrap::No => {}
        CatBoostBootstrap::Bayesian => {
            let t = p.bagging_temperature as f32;
            if t != 0.0 {
                let seed = rng.gen();
                for (block, rows) in boot.chunks_mut(1000).enumerate() {
                    let mut local = CatRng::new(seed.wrapping_add(block as u64));
                    local.advance(10);
                    for b in rows {
                        let w = -fast_logf((local.real1() + 1e-100) as f32);
                        *b = f64::from(w.powf(t));
                    }
                }
            }
        }
        CatBoostBootstrap::Bernoulli => {}
        CatBoostBootstrap::Mvs => {
            let rate = p.subsample as f32;
            if rate != 1.0 {
                // The derivatives: of the fold (plain), or each row's from
                // the prefix it is a tail row of (ordered).
                let mut der = vec![0.0; n * k];
                for (t, (d, &(body, tail, _))) in ders.iter().zip(tails).enumerate() {
                    let from = if t == 0 { 0 } else { body };
                    der[from * k..tail * k].copy_from_slice(&d.der[from * k..tail * k]);
                }
                let row = |i: usize| -> f64 {
                    (0..k)
                        .map(|dim| der[i * k + dim] * der[i * k + dim])
                        .fold(0.0, |a, b| a + b)
                };
                let lambda = match (p.mvs_reg, previous) {
                    (Some(l), _) => f64::from(l as f32),
                    (None, Some(leaves)) => {
                        let n_leaves = leaves.len() / k;
                        let sum: f64 = (0..n_leaves)
                            .map(|l| {
                                (0..k)
                                    .map(|d| leaves[l * k + d] * leaves[l * k + d])
                                    .fold(0.0, |a, b| a + b)
                                    .sqrt()
                            })
                            .fold(0.0, |a, b| a + b);
                        let mean = sum / n_leaves as f64;
                        mean * mean
                    }
                    (None, None) => {
                        // `CalculateMeanGradValue`: 128 blocks.
                        let block = n.div_ceil(128).max(1);
                        let sums: Vec<f64> = (0..n)
                            .collect::<Vec<_>>()
                            .chunks(block)
                            .map(|rows| rows.iter().fold(0.0, |a, &i| a + row(i).sqrt()))
                            .collect();
                        let mean = sums.iter().fold(0.0, |a, b| a + b) / n as f64;
                        mean * mean
                    }
                };
                let seed = rng.gen();
                for (block, start) in (0..n).step_by(8192).enumerate() {
                    let end = (start + 8192).min(n);
                    let mut local = CatRng::new(seed.wrapping_add(block as u64));
                    local.advance(10);
                    let mut candidates: Vec<f64> = (start..end)
                        .map(|i| {
                            let mut c = lambda;
                            for dim in 0..k {
                                c += der[i * k + dim] * der[i * k + dim];
                            }
                            c.sqrt()
                        })
                        .collect();
                    let size = f64::from(rate) * (end - start) as f64;
                    let threshold = mvs_threshold(&mut candidates, 0.0, 0, size);
                    for i in start..end {
                        let g = (row(i) + lambda).sqrt();
                        let prob = if g > threshold { 1.0 } else { g / threshold };
                        boot[i] = if prob > f64::EPSILON {
                            let r = local.real1();
                            f64::from(((1.0 / prob) * f64::from(u8::from(r < prob))) as f32)
                        } else {
                            0.0
                        };
                        taken[i] = (boot[i] as f32) > f32::EPSILON;
                    }
                }
            }
        }
    }
    // `CalcWeightedData`: the scores' weights in float32.
    let weights: Vec<f64> = boot
        .iter()
        .zip(learn_weights)
        .map(|(&b, &w)| f64::from(b as f32 * w as f32))
        .collect();
    // `SetSampledControl`: Bernoulli draws a row at a time.
    if p.bootstrap == CatBoostBootstrap::Bernoulli && p.subsample as f32 != 1.0 {
        let rate = f64::from(p.subsample as f32);
        for t in taken.iter_mut() {
            *t = rng.real1() < rate;
        }
    }
    Sample {
        boot,
        weights,
        taken,
    }
}

/// The scale of the score noise of a tree (CatBoost's `CalcScoreStDev`
/// with `NormalWithModelSizeDecrease`): `random_strength` times the root
/// mean square of the weighted derivatives (ordered boosting: of the tail
/// rows of every prefix) times `m / (1 + m)` with `m = exp(ln n - model
/// length)`, the model length being `iteration · learning_rate`.
fn score_std(
    ders: &[Ders],
    tails: &[(usize, usize, f64)],
    ordered: bool,
    n: usize,
    iteration: usize,
    p: &TrainParams,
) -> f64 {
    if p.random_strength == 0.0 {
        return 0.0;
    }
    let (mut sum2, mut count) = (0.0, 0usize);
    for (d, &(body, tail, _)) in ders.iter().zip(tails) {
        let from = if ordered { body } else { 0 };
        let k = d.k;
        for dim in 0..k {
            // `L2NormSquared`: blocks of 10000 rows.
            let rows: Vec<f64> = (from..tail).map(|i| d.der[i * k + dim]).collect();
            for block in rows.chunks(10000) {
                sum2 += block.iter().map(|v| v * v).sum::<f64>();
            }
        }
        count += tail - from;
    }
    let std = (sum2 / count as f64).sqrt();
    let model_length = iteration as f64 * p.learning_rate;
    let left = ((n as f64).ln() - model_length).exp();
    p.random_strength * std * (left / (1.0 + left))
}

/// Move a learning fold by the tree with the leaves `leaf_of` (CatBoost's
/// `UpdateLearningFold`): per body / tail prefix, the leaf values of the
/// body rows (`CalcApproxDeltaSimple` / `Multi`).
fn update_fold(
    fold: &mut LearnFold,
    leaf_of: &[u32],
    n_leaves: usize,
    loss: Loss,
    p: &TrainParams,
) {
    let k = loss.dims();
    let (target, weights) = (&fold.target, &fold.weights);
    for (&(body, tail, bw), a) in fold.tails.iter().zip(&mut fold.approx) {
        let lp = LeafProblem {
            loss,
            target,
            approx: &a[..],
            weights,
            leaf_of,
            n_leaves,
            rows: body,
            stats: (bw, body as f64),
            moving: false,
        };
        if k == 1 {
            // Each prefix's approximation moves by the leaf values of its
            // body rows.
            let (leaves, _) = estimate_leaves(&lp, p, |_, _| {});
            for i in 0..tail {
                a[i] += loss.fold_step(leaves[leaf_of[i] as usize], p.learning_rate);
            }
        } else {
            // Multiclass (CatBoost's `CalcApproxDeltaMulti`): the body rows
            // by the leaf values of the body; after every step, each tail
            // row by a Newton step of the body and the tail rows up to it,
            // at its moved approximation.
            let mut tail_delta = vec![0.0; (tail - body) * k];
            let (mut der, mut der2) = (vec![0.0; k], vec![0.0; loss.hessian_len()]);
            let (leaves, _) = estimate_leaves(&lp, p, |_, sums| {
                for i in body..tail {
                    let leaf = leaf_of[i] as usize;
                    let o = &mut tail_delta[(i - body) * k..(i - body + 1) * k];
                    let point: Vec<f64> = (0..k).map(|dim| a[i * k + dim] + o[dim]).collect();
                    loss.ders(target[i], &point, weights[i], &mut der, &mut der2);
                    let row = Ders {
                        k,
                        h: der2.len(),
                        der: der.clone(),
                        der2: der2.clone(),
                    };
                    sums.add(leaf, &row, 0, weights[i]);
                    let mut delta = vec![0.0; k];
                    sums.delta(leaf, p.l2_leaf_reg, bw, body as f64, p, &mut delta);
                    for dim in 0..k {
                        o[dim] += delta[dim];
                    }
                }
            });
            for i in 0..tail {
                let delta = if i < body {
                    let leaf = leaf_of[i] as usize;
                    &leaves[leaf * k..(leaf + 1) * k]
                } else {
                    &tail_delta[(i - body) * k..(i - body + 1) * k]
                };
                for dim in 0..k {
                    a[i * k + dim] += delta[dim] * p.learning_rate;
                }
            }
        }
    }
}

/// Boosting of oblivious trees from the constant `bias` (one per
/// approximation), as CatBoost's `TrainOneIteration` does it.
///
/// The trees are searched on a learning fold picked at random each
/// iteration. Plain boosting scores all of its rows; ordered boosting scores
/// its body / tail prefixes, whose approximations only ever saw the rows
/// before their tail. The model's leaf values come from the averaging fold.
/// Without `has_time`, CTRs (and ordered boosting) need permutations: fold 0
/// keeps the given order, the other `permutation_count - 2` folds and (with
/// CTRs) the averaging fold are shuffled with CatBoost's generator, whose
/// draws follow CatBoost's.
fn boost(
    q: &Quantized,
    target: &[f64],
    weights: &[f64],
    bias: &[f64],
    loss: Loss,
    p: TrainParams,
    mut rng: CatRng,
) -> (Vec<ObliviousTree>, CtrModel) {
    let n = target.len();
    let k = loss.dims();
    let has_ctrs = q.ctr.is_some();
    let learn_permuted = !p.has_time && (has_ctrs || p.ordered);
    let average_permuted = !p.has_time && has_ctrs;
    let n_folds = if learn_permuted {
        p.permutation_count.saturating_sub(1).max(1)
    } else {
        1
    };
    let block = if learn_permuted {
        (n / 1000 + 1).min(256)
    } else {
        n
    };
    let start = |rows: usize| -> Vec<f64> { bias.iter().copied().cycle().take(rows * k).collect() };
    let identity: Vec<usize> = (0..n).collect();
    let make_fold = |perm: &[usize]| -> LearnFold {
        let target = permute_rows(target, perm, 1);
        let weights = permute_rows(weights, perm, 1);
        let sum_w: f64 = weights.iter().sum();
        let tails = if p.ordered {
            body_tails(n, p.fold_len_multiplier, &weights)
        } else {
            vec![(n, n, sum_w)]
        };
        LearnFold {
            q: q.permuted(perm),
            approx: tails.iter().map(|&(_, t, _)| start(t)).collect(),
            target,
            weights,
            tails,
            cache: CombinationCache::new(),
        }
    };
    let mut folds: Vec<LearnFold> = (0..n_folds)
        .map(|f| {
            let perm = if f > 0 && learn_permuted {
                rng.permutation(n, block)
            } else {
                identity.clone()
            };
            make_fold(&perm)
        })
        .collect();
    let average_perm = if average_permuted {
        rng.permutation(n, block)
    } else {
        identity.clone()
    };
    let mut average = make_fold(&average_perm);
    let avg_sum_w: f64 = average.weights.iter().sum();
    let mut approx = start(n);
    // Plain boosting with an unpermuted averaging fold shares its
    // approximations with fold 0 (`UseAveragingFoldAsFoldZero`).
    let shared = !p.ordered && !average_permuted;
    let mut trees = Vec::with_capacity(p.iterations);
    let mut used_ctrs = std::collections::HashSet::new();
    let mut ctrs = CtrModel::default();
    for iteration in 0..p.iterations {
        let taken = (rng.gen() % n_folds as u64) as usize;
        rng.gen(); // the derivatives' seeds
        let fold = &mut folds[taken];
        let ders: Vec<Ders> = fold
            .tails
            .iter()
            .zip(&fold.approx)
            .map(|(&(_, t, _), a)| loss.all_ders(&fold.target, a, &fold.weights, t))
            .collect();
        let previous = trees
            .last()
            .map(|t: &ObliviousTree| t.leaf_values.as_slice());
        let sample = bootstrap(&mut rng, &ders, &fold.tails, &fold.weights, previous, k, &p);
        let sampled: Vec<Vec<f64>> = ders
            .iter()
            .map(|d| {
                d.der
                    .iter()
                    .enumerate()
                    .map(|(j, v)| v * sample.boot[j / k])
                    .collect()
            })
            .collect();
        let parts: Vec<ScorePart<'_>> = fold
            .tails
            .iter()
            .zip(&ders)
            .zip(&sampled)
            .map(|((&(body, tail, bw), d), sd)| ScorePart {
                wder: &d.der,
                swder: sd,
                body: if p.ordered { body } else { 0 },
                tail,
                scaled_l2: p.l2_leaf_reg * (bw / body as f64),
            })
            .collect();
        let (splits, _) = grow_tree(
            &fold.q,
            &parts,
            k,
            &fold.weights,
            p.ordered,
            p.depth,
            p.score_function,
            p.model_size_reg,
            &mut used_ctrs,
            &mut fold.cache,
            &mut rng,
            score_std(&ders, &fold.tails, p.ordered, n, iteration, &p),
            &sample,
        );
        rng.gen(); // the folds' seeds
        let n_leaves = 1usize << splits.len();

        // The model's leaves, from the averaging fold.
        let leaf_of = average.q.leaves(&splits, &mut average.cache);
        let lp = LeafProblem {
            loss,
            target: &average.target,
            approx: &approx,
            weights: &average.weights,
            leaf_of: &leaf_of,
            n_leaves,
            rows: n,
            stats: (avg_sum_w, n as f64),
            moving: true,
        };
        let (leaves, steps) = estimate_leaves(&lp, &p, |_, _| {});
        if k == 1 {
            // `CalcLeafValuesSimple` draws a seed per step.
            for _ in 0..steps {
                rng.gen();
            }
        }
        let leaf_values: Vec<f64> = leaves.into_iter().map(|v| v * p.learning_rate).collect();
        for i in 0..n {
            let leaf = leaf_of[i] as usize;
            for d in 0..k {
                approx[i * k + d] += leaf_values[leaf * k + d];
            }
        }
        for (f, fold) in folds.iter_mut().enumerate() {
            if f == 0 && shared {
                fold.approx[0].clone_from(&approx);
                continue;
            }
            let leaf_of = fold.q.leaves(&splits, &mut fold.cache);
            update_fold(fold, &leaf_of, n_leaves, loss, &p);
        }
        let source = &folds[taken];
        let tree_splits = splits
            .iter()
            .map(|(c, s)| c.get(&source.q).tree_split(*s, &mut ctrs))
            .collect();
        trees.push(ObliviousTree {
            splits: tree_splits,
            leaf_values,
        });
    }
    (trees, ctrs)
}

/// Raw scores `bias + Σ leaf values` of the rows of `x`, one column per
/// approximation (`bias.len()`).
fn raw_scores(
    trees: &[ObliviousTree],
    ctrs: &CtrModel,
    bias: &[f64],
    x: ArrayView2<f64>,
) -> Array2<f64> {
    use rayon::prelude::*;
    let k = bias.len();
    let mut out: Vec<f64> = bias.iter().copied().cycle().take(x.nrows() * k).collect();
    out.par_chunks_mut(k)
        .with_min_len(1024)
        .enumerate()
        .for_each(|(i, s)| {
            let row = x.row(i);
            for t in trees {
                let leaf = t.leaf(row, ctrs);
                for (d, v) in s.iter_mut().enumerate() {
                    *v += t.leaf_values[leaf * k + d];
                }
            }
        });
    Array2::from_shape_vec((x.nrows(), k), out).expect("rows × approximations")
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

/// The class (0 / 1) of every regression target for the `Borders` CTRs:
/// CatBoost's single MinEntropy target border (`SelectBorders` with one
/// border): the split of the sorted distinct targets minimizing
/// `Σ w·ln(w + 1e-8)` over the two sides, at the float32 midpoint.
fn target_classes(target: &[f64]) -> Vec<u8> {
    let mut values: Vec<f32> = target.iter().map(|&t| t as f32).collect();
    values.sort_by(f32::total_cmp);
    let mut distinct: Vec<(f32, f64)> = Vec::new();
    for v in values {
        match distinct.last_mut() {
            Some((last, c)) if *last == v => *c += 1.0,
            _ => distinct.push((v, 1.0)),
        }
    }
    let total: f64 = distinct.iter().map(|d| d.1).sum();
    let penalty = |w: f64| w * (w + 1e-8).ln();
    let mut best: Option<(f64, usize)> = None;
    let mut left = 0.0;
    for k in 1..distinct.len() {
        left += distinct[k - 1].1;
        let cost = penalty(left) + penalty(total - left);
        if best.is_none_or(|(c, _)| cost < c) {
            best = Some((cost, k));
        }
    }
    let border = match best {
        Some((_, k)) => (distinct[k - 1].0 + distinct[k].0) * 0.5f32,
        None => distinct.first().map_or(0.0, |d| d.0),
    };
    target
        .iter()
        .map(|&t| u8::from((t as f32) > border))
        .collect()
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
            /// Categorical features (their values are category codes).
            /// Those with at most `one_hot_max_size` categories are split
            /// one-hot, the others through CatBoost's ordered target
            /// statistics (CTRs), of single features and of combinations
            /// (`max_ctr_complexity`).
            pub cat_features: Vec<usize>,
            /// Default 2.
            pub one_hot_max_size: usize,
            /// The most features a CTR combines (CatBoost's
            /// `max_ctr_complexity`; the conditions of a tree's float and
            /// one-hot splits count as one). Default (`None`): 4, or 1 for
            /// fewer than 200 iterations, as in CatBoost.
            pub max_ctr_complexity: Option<usize>,
            /// Penalty on CTRs new to the model (CatBoost's
            /// `model_size_reg`, default 0.5).
            pub model_size_reg: f64,
            /// Plain (default) or ordered boosting.
            pub boosting_type: CatBoostBoostingType,
            /// Growth of the body / tail prefixes of ordered boosting
            /// (default 2).
            pub fold_len_multiplier: f64,
            /// Newton / Gradient steps per leaf estimation (CatBoost's
            /// `leaf_estimation_iterations`). Default (`None`): CatBoost's,
            /// 10 Newton or 40 Gradient steps for Logloss, 1 Newton or 10
            /// Gradient steps for MultiClass, 1 for RMSE, and 1 for any
            /// loss below 200 iterations with fewer than 20 features.
            pub leaf_estimation_iterations: Option<usize>,
            pub leaf_estimation_backtracking: CatBoostBacktracking,
            /// Keep the rows' order: the CTRs and ordered boosting see the
            /// rows as given, with no permutations (CatBoost's `has_time`,
            /// default false).
            pub has_time: bool,
            /// CatBoost's `permutation_count` (default 4): the learning
            /// folds are `permutation_count - 1` orders of the rows.
            pub permutation_count: usize,
            /// The seed of CatBoost's generator (`random_seed`, default 0).
            pub random_seed: u64,
            /// The noise of the split scores (CatBoost's `random_strength`,
            /// default 1; `NormalWithModelSizeDecrease`): normal with the
            /// root mean square of the derivatives times this, fading with
            /// the model's length.
            pub random_strength: f64,
            /// CatBoost's `bootstrap_type`. Default (`None`): MVS, or
            /// Bayesian for more than two classes.
            pub bootstrap_type: Option<CatBoostBootstrap>,
            /// The share of rows (`subsample`). Default (`None`): 0.8 for
            /// MVS, 0.66 for Bernoulli, 1 below 100 rows.
            pub subsample: Option<f64>,
            /// Bayesian bootstrap's temperature (default 1).
            pub bagging_temperature: f64,
            /// MVS's regularization; default (`None`): from the last tree's
            /// leaves (the first tree: from the gradients).
            pub mvs_reg: Option<f64>,
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
                    cat_features: Vec::new(),
                    one_hot_max_size: 2,
                    max_ctr_complexity: None,
                    model_size_reg: 0.5,
                    boosting_type: CatBoostBoostingType::Plain,
                    fold_len_multiplier: 2.0,
                    leaf_estimation_iterations: None,
                    leaf_estimation_backtracking: CatBoostBacktracking::AnyImprovement,
                    has_time: false,
                    permutation_count: 4,
                    random_seed: 0,
                    random_strength: 1.0,
                    bootstrap_type: None,
                    subsample: None,
                    bagging_temperature: 1.0,
                    mvs_reg: None,
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
            pub fn with_cat_features(mut self, features: Vec<usize>) -> Self {
                self.cat_features = features;
                self
            }
            pub fn with_one_hot_max_size(mut self, n: usize) -> Self {
                self.one_hot_max_size = n;
                self
            }
            pub fn with_leaf_estimation_iterations(mut self, n: usize) -> Self {
                self.leaf_estimation_iterations = Some(n);
                self
            }
            pub fn with_leaf_estimation_backtracking(mut self, b: CatBoostBacktracking) -> Self {
                self.leaf_estimation_backtracking = b;
                self
            }

            /// CatBoost's default leaf estimation iterations for `loss`
            /// (`GetEstimationMethodDefaults`, then
            /// `UpdateLeavesEstimationIterations`).
            fn leaf_iterations(&self, loss: Loss, n_features: usize) -> usize {
                if let Some(n) = self.leaf_estimation_iterations {
                    return n;
                }
                if self.iterations < 200 && n_features < 20 {
                    return 1;
                }
                let newton = self.leaf_estimation_method == CatBoostLeafEstimation::Newton;
                match (loss, newton) {
                    (Loss::Rmse, _) => 1,
                    (Loss::Logloss, true) => 10,
                    (Loss::Logloss, false) => 40,
                    (Loss::MultiClass(_), true) => 1,
                    (Loss::MultiClass(_), false) => 10,
                }
            }

            pub fn with_has_time(mut self, h: bool) -> Self {
                self.has_time = h;
                self
            }
            pub fn with_permutation_count(mut self, n: usize) -> Self {
                self.permutation_count = n;
                self
            }
            pub fn with_bootstrap_type(mut self, b: CatBoostBootstrap) -> Self {
                self.bootstrap_type = Some(b);
                self
            }
            pub fn with_subsample(mut self, s: f64) -> Self {
                self.subsample = Some(s);
                self
            }
            pub fn with_bagging_temperature(mut self, t: f64) -> Self {
                self.bagging_temperature = t;
                self
            }
            pub fn with_mvs_reg(mut self, r: f64) -> Self {
                self.mvs_reg = Some(r);
                self
            }

            /// CatBoost's bootstrap defaults for `loss` and `n` rows.
            fn sampling(&self, loss: Loss, n: usize) -> (CatBoostBootstrap, f64) {
                let kind = self.bootstrap_type.unwrap_or(match loss {
                    Loss::MultiClass(_) => CatBoostBootstrap::Bayesian,
                    _ => CatBoostBootstrap::Mvs,
                });
                let rate = self.subsample.unwrap_or(if n < 100 {
                    1.0
                } else if kind == CatBoostBootstrap::Mvs {
                    0.8
                } else {
                    0.66
                });
                (kind, rate)
            }

            pub fn with_random_strength(mut self, r: f64) -> Self {
                self.random_strength = r;
                self
            }
            pub fn with_random_seed(mut self, seed: u64) -> Self {
                self.random_seed = seed;
                self
            }

            pub fn with_max_ctr_complexity(mut self, n: usize) -> Self {
                self.max_ctr_complexity = Some(n);
                self
            }

            /// CatBoost's `ShuffleLearnDataIfNeeded`: without `has_time`,
            /// with categorical features or ordered boosting, the rows are
            /// shuffled first (the generator then goes on training).
            fn learn_order(&self, n: usize, rng: &mut CatRng) -> Option<Vec<usize>> {
                let need = !self.has_time
                    && (!self.cat_features.is_empty()
                        || self.boosting_type == CatBoostBoostingType::Ordered);
                need.then(|| rng.permutation(n, 1))
            }

            /// CatBoost's default: 1 below 200 iterations, else 4.
            fn max_ctr_complexity(&self) -> usize {
                self.max_ctr_complexity
                    .unwrap_or(if self.iterations < 200 { 1 } else { 4 })
            }
            pub fn with_model_size_reg(mut self, r: f64) -> Self {
                self.model_size_reg = r;
                self
            }
            pub fn with_boosting_type(mut self, b: CatBoostBoostingType) -> Self {
                self.boosting_type = b;
                self
            }
            pub fn with_fold_len_multiplier(mut self, m: f64) -> Self {
                self.fold_len_multiplier = m;
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
                    model_size_reg: self.model_size_reg,
                    ordered: self.boosting_type == CatBoostBoostingType::Ordered,
                    fold_len_multiplier: self.fold_len_multiplier,
                    leaf_iterations: 1,
                    backtracking: self.leaf_estimation_backtracking
                        == CatBoostBacktracking::AnyImprovement,
                    has_time: self.has_time,
                    random_strength: f64::from(self.random_strength as f32),
                    bootstrap: CatBoostBootstrap::No,
                    subsample: 1.0,
                    bagging_temperature: self.bagging_temperature,
                    mvs_reg: self.mvs_reg,
                    permutation_count: self.permutation_count,
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
    /// CatBoost-style classifier on oblivious trees: Logloss for two
    /// classes, MultiClass (softmax) for more, as CatBoost picks them.
    CatBoostClassifier,
    boost_from_average: false
);

/// Fitted [`CatBoostRegressor`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedCatBoostRegressor {
    trees: Vec<ObliviousTree>,
    #[serde(default)]
    ctrs: CtrModel,
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
        if self.leaf_estimation_iterations == Some(0) {
            return Err(RustMlError::InvalidParameter(
                "leaf_estimation_iterations must be at least 1".into(),
            ));
        }
        if self.max_ctr_complexity == Some(0) {
            return Err(RustMlError::InvalidParameter(
                "max_ctr_complexity must be at least 1".into(),
            ));
        }
        let mut rng = CatRng::new(self.random_seed);
        let shuffled = self.learn_order(x.nrows(), &mut rng).map(|order| {
            (
                x.select(ndarray::Axis(0), &order),
                y.select(ndarray::Axis(0), &order),
                sample_weight.map(|w| w.select(ndarray::Axis(0), &order)),
            )
        });
        let (x, y, sample_weight) = match &shuffled {
            Some((x, y, w)) => (x.view(), y.view(), w.as_ref().map(|w| w.view())),
            None => (x.view(), y.view(), sample_weight.as_ref().map(|w| w.view())),
        };
        let target = f32_values(y);
        let weights = sample_weight.map_or_else(|| vec![1.0; y.len()], f32_values);
        let bias = if self.boost_from_average {
            f64::from(weighted_target_average(&target, &weights))
        } else {
            0.0
        };
        let classes_of = target_classes(&target);
        let cat = CategoricalParams {
            features: &self.cat_features,
            one_hot_max_size: self.one_hot_max_size,
            classes: Some((&classes_of, 2)),
            max_ctr_complexity: self.max_ctr_complexity(),
        };
        let q = quantize(x, self.border_count, cat)?;
        let (bootstrap, subsample) = self.sampling(Loss::Rmse, x.nrows());
        let p = TrainParams {
            leaf_iterations: self.leaf_iterations(Loss::Rmse, x.ncols()),
            bootstrap,
            subsample,
            ..p
        };
        let (trees, ctrs) = boost(&q, &target, &weights, &[bias], Loss::Rmse, p, rng);
        Ok(FittedCatBoostRegressor {
            trees,
            ctrs,
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
        let raw = raw_scores(&self.trees, &self.ctrs, &[self.bias], x);
        Ok(raw.column(0).to_owned())
    }
}

/// Fitted [`CatBoostClassifier`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedCatBoostClassifier {
    /// Leaf values: one per leaf (two classes) or one per leaf and class.
    trees: Vec<ObliviousTree>,
    #[serde(default)]
    ctrs: CtrModel,
    /// The starting approximations (one, or one per class).
    bias: Vec<f64>,
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

    /// The model's raw scores (CatBoost's `RawFormulaVal`): one column of
    /// log-odds of the second class for two classes, else one column per
    /// class (softmax logits, in `classes()` order).
    pub fn raw_predictions<'a>(&self, x: impl AsArray<'a, f64, Ix2>) -> Result<Array2<f64>> {
        let x: ArrayView2<f64> = x.into();
        check_features(x, self.n_features)?;
        Ok(raw_scores(&self.trees, &self.ctrs, &self.bias, x))
    }

    /// Log-odds of the second class (two classes; see
    /// [`raw_predictions`](Self::raw_predictions) for more).
    pub fn decision_function<'a>(&self, x: impl AsArray<'a, f64, Ix2>) -> Result<Array1<f64>> {
        if self.classes.len() != 2 {
            return Err(RustMlError::InvalidParameter(format!(
                "decision_function of a {}-class model: use raw_predictions",
                self.classes.len()
            )));
        }
        Ok(self.raw_predictions(x)?.column(0).to_owned())
    }

    /// Class probabilities (columns in `classes()` order): the sigmoid of
    /// the log-odds, or the softmax of the class scores.
    pub fn predict_proba<'a>(&self, x: impl AsArray<'a, f64, Ix2>) -> Result<Array2<f64>> {
        let raw = self.raw_predictions(x)?;
        if self.classes.len() == 2 {
            let mut proba = Array2::zeros((raw.nrows(), 2));
            for (i, &r) in raw.column(0).iter().enumerate() {
                let p1 = 1.0 / (1.0 + (-r).exp());
                proba[[i, 0]] = 1.0 - p1;
                proba[[i, 1]] = p1;
            }
            return Ok(proba);
        }
        let mut proba = raw;
        for mut row in proba.rows_mut() {
            let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            row.mapv_inplace(|v| (v - max).exp());
            let sum = row.sum();
            row.mapv_inplace(|v| v / sum);
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
        if self.leaf_estimation_iterations == Some(0) {
            return Err(RustMlError::InvalidParameter(
                "leaf_estimation_iterations must be at least 1".into(),
            ));
        }
        if self.max_ctr_complexity == Some(0) {
            return Err(RustMlError::InvalidParameter(
                "max_ctr_complexity must be at least 1".into(),
            ));
        }
        let mut rng = CatRng::new(self.random_seed);
        let shuffled = self.learn_order(x.nrows(), &mut rng).map(|order| {
            (
                x.select(ndarray::Axis(0), &order),
                y.select(ndarray::Axis(0), &order),
                sample_weight.map(|w| w.select(ndarray::Axis(0), &order)),
            )
        });
        let (x, y, sample_weight) = match &shuffled {
            Some((x, y, w)) => (x.view(), y.view(), w.as_ref().map(|w| w.view())),
            None => (x.view(), y.view(), sample_weight.as_ref().map(|w| w.view())),
        };
        let mut classes: Vec<f64> = y.to_vec();
        classes.sort_by(f64::total_cmp);
        classes.dedup();
        if !(2..=255).contains(&classes.len()) {
            return Err(RustMlError::InvalidParameter(format!(
                "CatBoostClassifier needs 2 to 255 classes, got {}",
                classes.len()
            )));
        }
        let multiclass = classes.len() > 2;
        if multiclass && self.boost_from_average {
            return Err(RustMlError::InvalidParameter(
                "boost_from_average is not supported for more than two classes".into(),
            ));
        }
        // The class index of every row (Logloss: 0 / 1, MultiClass: 0..k).
        let target: Vec<f64> = y
            .iter()
            .map(|v| classes.partition_point(|c| c < v) as f64)
            .collect();
        let weights = sample_weight.map_or_else(|| vec![1.0; y.len()], f32_values);
        let (loss, bias) = if multiclass {
            (Loss::MultiClass(classes.len()), vec![0.0; classes.len()])
        } else if self.boost_from_average {
            let p1 = f64::from(weighted_target_average(&target, &weights));
            (Loss::Logloss, vec![(p1 / (1.0 - p1)).ln()])
        } else {
            (Loss::Logloss, vec![0.0])
        };
        let classes_of: Vec<u8> = target.iter().map(|&t| t as u8).collect();
        let cat = CategoricalParams {
            features: &self.cat_features,
            one_hot_max_size: self.one_hot_max_size,
            classes: Some((&classes_of, classes.len())),
            max_ctr_complexity: self.max_ctr_complexity(),
        };
        let q = quantize(x, self.border_count, cat)?;
        let (bootstrap, subsample) = self.sampling(loss, x.nrows());
        let p = TrainParams {
            leaf_iterations: self.leaf_iterations(loss, x.ncols()),
            bootstrap,
            subsample,
            ..p
        };
        let (trees, ctrs) = boost(&q, &target, &weights, &bias, loss, p, rng);
        Ok(FittedCatBoostClassifier {
            trees,
            ctrs,
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
        let raw = self.raw_predictions(x)?;
        Ok(raw
            .rows()
            .into_iter()
            .map(|r| {
                if r.len() == 1 {
                    self.classes[usize::from(r[0] > 0.0)]
                } else {
                    // The first most likely class.
                    let best = (1..r.len()).fold(0, |b, c| if r[c] > r[b] { c } else { b });
                    self.classes[best]
                }
            })
            .collect())
    }
}

/// Quantization borders (for the parity tests).
#[doc(hidden)]
pub fn catboost_borders(x: ArrayView2<f64>, border_count: usize) -> Vec<Vec<f32>> {
    let cat = CategoricalParams {
        features: &[],
        one_hot_max_size: 0,
        classes: None,
        max_ctr_complexity: 1,
    };
    quantize(x, border_count, cat).unwrap().borders
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
        let cat = CategoricalParams {
            features: &[],
            one_hot_max_size: 0,
            classes: None,
            max_ctr_complexity: 1,
        };
        let q = quantize(x.view(), 254, cat).unwrap();
        assert_eq!(q.borders[0][0], -f32::MAX);
        assert_eq!(q.columns[0].buckets, vec![0, 1, 2, 3]);
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
        assert!(CatBoostClassifier::new()
            .fit(&x, &labels.mapv(|_| 1.0))
            .is_err());
    }

    #[test]
    fn rng_matches_yandex_util() {
        // util/random/fast_ut.cpp
        assert_eq!(CatRng::new(17).gen(), 14895365814383052362);
        let mut rng = CatRng::from_parts(0, 1, 2, 3);
        let draws: Vec<u64> = (0..15).map(|_| rng.uniform(100)).collect();
        assert_eq!(
            draws,
            [37, 43, 76, 17, 12, 87, 60, 4, 83, 47, 57, 81, 28, 45, 66]
        );
    }

    #[test]
    fn packed_cholesky_solves_spd_systems() {
        // [[4, 2, 0], [2, 5, 1], [0, 1, 3]] x = [2, 3, 4], lower triangle
        // packed column by column.
        let mut ap = vec![4.0, 2.0, 0.0, 5.0, 1.0, 3.0];
        let mut b = vec![2.0, 3.0, 4.0];
        assert!(solve_packed_cholesky(&mut ap, &mut b));
        let a = [[4.0, 2.0, 0.0], [2.0, 5.0, 1.0], [0.0, 1.0, 3.0]];
        for (row, rhs) in a.iter().zip([2.0, 3.0, 4.0]) {
            let lhs: f64 = row.iter().zip(&b).map(|(a, x)| a * x).sum();
            assert!((lhs - rhs).abs() < 1e-14);
        }
        assert!(!solve_packed_cholesky(
            &mut [0.0, 0.0, 0.0],
            &mut [1.0, 1.0]
        ));
    }

    #[test]
    fn multiclass_probabilities_and_scores() {
        let (x, y) = data(240);
        let labels = y.mapv(|v| {
            if v < 0.3 {
                5.0
            } else if v < 1.2 {
                6.0
            } else {
                9.0
            }
        });
        let f = CatBoostClassifier::new()
            .with_iterations(30)
            .with_learning_rate(0.3)
            .with_depth(3)
            .fit(&x, &labels)
            .unwrap();
        assert_eq!(f.classes(), &[5.0, 6.0, 9.0]);
        assert!(f
            .trees
            .iter()
            .all(|t| t.leaf_values.len() == 3 << t.splits.len()));
        let proba = f.predict_proba(&x).unwrap();
        assert_eq!(proba.ncols(), 3);
        assert!(proba
            .rows()
            .into_iter()
            .all(|r| (r.sum() - 1.0).abs() < 1e-12));
        assert_eq!(f.raw_predictions(&x).unwrap().ncols(), 3);
        assert!(f.decision_function(&x).is_err());
        let acc = f
            .predict(&x)
            .unwrap()
            .iter()
            .zip(&labels)
            .filter(|(a, b)| a == b)
            .count();
        assert!(acc as f64 / 240.0 > 0.85, "{acc}");
        assert!(CatBoostClassifier::new()
            .with_boost_from_average(true)
            .fit(&x, &labels)
            .is_err());
    }

    #[test]
    fn ctrs_are_ordered_and_quantized_as_catboost() {
        // (good + prior) / (total + 1) in 15ths, truncated.
        assert_eq!(calc_ctr(0, 0, 0.5), 7);
        assert_eq!(calc_ctr(3, 4, 0.0), 9);
        assert_eq!(calc_ctr(4, 4, 1.0), 15);
        // Online counts: each row sees only the rows before it.
        let x = Array2::from_shape_vec((5, 1), vec![7.0, 7.0, 7.0, 3.0, 3.0]).unwrap();
        let classes = [1u8, 1, 0, 0, 1];
        let cat = CategoricalParams {
            features: &[0],
            one_hot_max_size: 1,
            classes: Some((&classes, 2)),
            max_ctr_complexity: 1,
        };
        let q = quantize(x.view(), 254, cat).unwrap();
        // Borders with prior 0: 0/1, 1/2, 2/3, 0/1, 0/2.
        assert_eq!(q.columns[0].buckets, vec![0, 7, 10, 0, 0]);
        // Counter: counts 3 and 2 over (max 3) + 1.
        assert_eq!(q.columns[3].buckets, vec![11, 11, 11, 7, 7]);
        let ColumnSplit::Ctr { table, .. } = &q.columns[0].split else {
            panic!("a CTR column");
        };
        assert_eq!(table.counts(&[category(7.0)]), [1, 2]);
        // Few categories: one-hot, no CTRs.
        let cat = CategoricalParams {
            features: &[0],
            one_hot_max_size: 2,
            classes: Some((&classes, 2)),
            max_ctr_complexity: 1,
        };
        let q = quantize(x.view(), 254, cat).unwrap();
        assert!(q.columns[0].one_hot() && q.ctr.is_none());
    }

    #[test]
    fn ctr_combinations_follow_the_tree() {
        // A target that only the combination of two categorical features
        // explains (their parity).
        let n = 400;
        let x = Array2::from_shape_fn((n, 3), |(i, j)| match j {
            0 => ((i * 7) % 5) as f64,
            1 => ((i * 11) % 6) as f64,
            _ => (i % 13) as f64 / 13.0,
        });
        let y = Array1::from_shape_fn(n, |i| ((x[[i, 0]] + x[[i, 1]]) as usize % 2) as f64);
        let m = |c| {
            CatBoostClassifier::new()
                .with_iterations(40)
                .with_learning_rate(0.3)
                .with_depth(3)
                .with_cat_features(vec![0, 1])
                .with_max_ctr_complexity(c)
        };
        let single = m(1).fit(&x, &y).unwrap();
        let combined = m(2).fit(&x, &y).unwrap();
        assert!(single
            .ctrs
            .tables
            .iter()
            .all(|t| t.projection.complexity() == 1));
        let pair = Projection {
            cats: vec![0, 1],
            ..Projection::default()
        };
        assert!(combined.ctrs.tables.iter().any(|t| t.projection == pair));
        let acc = |f: &FittedCatBoostClassifier| {
            let p = f.predict(&x).unwrap();
            p.iter().zip(&y).filter(|(a, b)| a == b).count() as f64 / n as f64
        };
        assert!(
            acc(&combined) > acc(&single),
            "{} {}",
            acc(&combined),
            acc(&single)
        );
        assert_eq!(acc(&combined), 1.0);
        assert!(m(0).fit(&x, &y).is_err());
    }

    #[test]
    fn ordered_boosting_prefixes_double() {
        let w = vec![1.0; 10];
        let tails: Vec<(usize, usize)> =
            body_tails(10, 2.0, &w).iter().map(|t| (t.0, t.1)).collect();
        assert_eq!(tails, vec![(1, 2), (2, 4), (4, 8), (8, 10)]);
        let w = vec![1.0; 1200];
        assert_eq!(body_tails(1200, 2.0, &w)[0].0, 24);
        let (x, y) = data(150);
        let m = |b| {
            CatBoostRegressor::new()
                .with_iterations(20)
                .with_learning_rate(0.2)
                .with_boosting_type(b)
        };
        let plain = m(CatBoostBoostingType::Plain).fit(&x, &y).unwrap();
        let ordered = m(CatBoostBoostingType::Ordered).fit(&x, &y).unwrap();
        assert_ne!(plain.predict(&x).unwrap(), ordered.predict(&x).unwrap());
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
