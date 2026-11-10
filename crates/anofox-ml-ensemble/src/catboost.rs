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
/// What a [`StdPriorityQueue`] orders by.
trait HeapKey {
    fn key(&self) -> f64;
}

impl HeapKey for GreedyBin {
    fn key(&self) -> f64 {
        self.best_score
    }
}

struct StdPriorityQueue<T>(Vec<T>);

impl<T: Copy + HeapKey> StdPriorityQueue<T> {
    fn less(a: &T, b: &T) -> bool {
        a.key() < b.key()
    }

    /// libc++'s `__sift_up` of the last of the first `len` elements.
    fn sift_up(v: &mut [T], len: usize) {
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

    fn push(&mut self, value: T) {
        self.0.push(value);
        let len = self.0.len();
        Self::sift_up(&mut self.0, len);
    }

    fn top(&self) -> &T {
        &self.0[0]
    }

    /// libc++'s `__pop_heap` (Floyd's sift-down, then a sift-up) and
    /// `pop_back`.
    fn pop(&mut self) -> T {
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

    /// The permutation of grouped rows (`Shuffle` of a non-trivial
    /// grouping): the queries `queries` (row bounds) in shuffled order, the
    /// rows of each shuffled in turn; also the new query bounds.
    fn group_permutation(&mut self, queries: &[usize]) -> (Vec<usize>, Vec<usize>) {
        let order = self.permutation(queries.len() - 1, 1);
        let mut perm = Vec::with_capacity(queries[queries.len() - 1]);
        let mut bounds = vec![0];
        for q in order {
            let start = perm.len();
            perm.extend(queries[q]..queries[q + 1]);
            let rows = &mut perm[start..];
            for i in 1..rows.len() {
                let j = self.uniform(i as u64 + 1) as usize;
                rows.swap(i, j);
            }
            bounds.push(perm.len());
        }
        (perm, bounds)
    }

    /// A fold permutation of `n` rows: in blocks, or by query when
    /// `queries` is non-empty (with its query bounds).
    fn fold_permutation(
        &mut self,
        n: usize,
        block: usize,
        queries: &[usize],
    ) -> (Vec<usize>, Vec<usize>) {
        if queries.is_empty() {
            (self.permutation(n, block), Vec::new())
        } else {
            self.group_permutation(queries)
        }
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
    /// Non-symmetric trees (Depthwise): the children of the node of every
    /// split, a child `c < 0` being leaf `!c`; empty for oblivious trees.
    #[serde(default)]
    children: NodeChildren,
}

impl ObliviousTree {
    fn leaf(&self, row: ArrayView1<f64>, ctrs: &CtrModel) -> usize {
        let right = |split: &TreeSplit| match *split {
            TreeSplit::Float { feature, border } => (row[feature] as f32) > border,
            TreeSplit::OneHot { feature, value } => category(row[feature]) == value,
            TreeSplit::Ctr { ctr, bucket } => ctrs.bucket(ctr, row) > bucket,
        };
        if !self.children.is_empty() {
            let mut node = 0i32;
            loop {
                let (l, r) = self.children[node as usize];
                node = if right(&self.splits[node as usize]) {
                    r
                } else {
                    l
                };
                if node < 0 {
                    return !node as usize;
                }
            }
        }
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
    /// The loss-optimal constant of the leaf's residuals (CatBoost's
    /// default for MAE and Quantile: their weighted quantile).
    Exact,
}

/// The loss of [`CatBoostRegressor`] (CatBoost's `loss_function`).
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub enum CatBoostRegressionLoss {
    #[default]
    #[serde(rename = "RMSE")]
    Rmse,
    #[serde(rename = "MAE")]
    Mae,
    /// The `alpha` quantile.
    Quantile { alpha: f64 },
    /// The `alpha` expectile.
    Expectile { alpha: f64 },
    /// Poisson log-likelihood on the log scale; predictions are `exp(raw)`.
    Poisson,
    /// Tweedie with `1 < variance_power < 2` on the log scale; predictions
    /// are `exp(raw)`.
    Tweedie { variance_power: f64 },
    /// The `alpha` quantile of `exp(raw)` (predictions stay raw, as in
    /// CatBoost).
    LogLinQuantile { alpha: f64 },
    /// Huber loss with threshold `delta` (CatBoost's `Huber:delta`).
    Huber { delta: f64 },
    /// `log(cosh(raw - target))` (CatBoost's `LogCosh`; Exact leaves).
    LogCosh,
    /// `|raw - target|^q`, `q >= 1` (CatBoost's `Lq:q`).
    Lq { q: f64 },
    /// A mean and a log standard deviation per row (CatBoost's
    /// `RMSEWithUncertainty`, NGBoost's natural gradient): `predict` gives
    /// the mean, `predict_uncertainty` the mean and the variance.
    #[serde(rename = "RMSEWithUncertainty")]
    RmseWithUncertainty,
    /// The RMSE summed over k targets per row (CatBoost's `MultiRMSE`):
    /// fit with [`CatBoostRegressor::fit_multi`], predict with
    /// [`FittedCatBoostRegressor::predict_multi`].
    #[serde(rename = "MultiRMSE")]
    MultiRmse,
    /// The RMSE after removing each query's mean residual (CatBoost's
    /// `QueryRMSE`): fit with [`CatBoostRegressor::fit_with_groups`].
    #[serde(rename = "QueryRMSE")]
    QueryRmse,
    /// The logistic loss of every pair of a query's rows with different
    /// targets (CatBoost's `PairLogit`, pairs generated from the targets):
    /// fit with [`CatBoostRegressor::fit_with_groups`].
    #[serde(rename = "PairLogit")]
    PairLogit,
    /// PairLogit on pairs drawn anew every iteration from `permutations`
    /// noisy rankings by the current scores, neighbours weighted by
    /// `decay` (CatBoost's `YetiRank`, Classic mode, Gumbel noise; CatBoost
    /// defaults 10 and 0.85, and `l2_leaf_reg` 0).
    #[serde(rename = "YetiRank")]
    YetiRank { permutations: usize, decay: f64 },
    /// PairLogit whose splits and leaves are scored on the pairs (CatBoost's
    /// `PairLogitPairwise`; plain symmetric trees, numeric features).
    #[serde(rename = "PairLogitPairwise")]
    PairLogitPairwise,
    /// YetiRank whose splits and leaves are scored on the pairs (CatBoost's
    /// `YetiRankPairwise`; plain symmetric trees, numeric features).
    #[serde(rename = "YetiRankPairwise")]
    YetiRankPairwise { permutations: usize, decay: f64 },
}

/// CatBoost's `grow_policy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum CatBoostGrowPolicy {
    /// Oblivious trees: one split per level.
    #[default]
    SymmetricTree,
    /// Level by level, each leaf with its own split.
    Depthwise,
    /// The leaf of the largest gain next, up to `max_leaves`.
    Lossguide,
}

/// CatBoost's overfitting detector (`od_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CatBoostOdType {
    None,
    /// Stop when the expected improvement is unlikely (`od_pval`).
    IncToDec,
    /// Stop `od_wait` iterations after the best one.
    Iter,
}

/// CatBoost's `model_shrink_mode`: the factor the model is shrunk by
/// before iteration `i > 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CatBoostShrinkMode {
    /// `1 - rate · learning_rate`.
    Constant,
    /// `1 - rate / i`.
    Decreasing,
}

/// CatBoost's `BuildMonotonicLinearOrdersOnLeafs`: for the constraints of
/// a tree's levels, the leaves of every subtree of the unconstrained
/// levels in the order the constrained ones require to be increasing.
fn monotone_orders(constraints: &[i8]) -> Vec<Vec<usize>> {
    let free = constraints.iter().filter(|&&c| c == 0).count();
    (0..1usize << free)
        .map(|mut subtree| {
            let (mut bit, mut least) = (1usize, 0usize);
            let mut masks = Vec::new();
            for &c in constraints {
                if c == 0 {
                    if subtree & 1 == 1 {
                        least |= bit;
                    }
                    subtree >>= 1;
                } else {
                    masks.push(bit);
                    if c == -1 {
                        least |= bit;
                    }
                }
                bit <<= 1;
            }
            let m = masks.len();
            (0..1usize << m)
                .map(|rank| {
                    let mut leaf = least;
                    for (depth, &mask) in masks.iter().enumerate() {
                        if (rank >> (m - 1 - depth)) & 1 == 1 {
                            leaf ^= mask;
                        }
                    }
                    leaf
                })
                .collect()
        })
        .collect()
}

/// CatBoost's `CalcOneDimensionalIsotonicRegression` (pool adjacent
/// violators) of `values` along `order`, in place.
fn isotonic(values: &mut [f64], weights: &[f64], order: &[usize]) {
    // (begin, end, weight, weighted sum) of each level set.
    let mut sets: Vec<(usize, usize, f64, f64)> = Vec::new();
    for (rank, &i) in order.iter().enumerate() {
        let mut set = (rank, rank + 1, weights[i], weights[i] * values[i]);
        while let Some(last) = sets.last() {
            if last.3 / last.2 >= set.3 / set.2 {
                set = (last.0, set.1, last.2 + set.2, last.3 + set.3);
                sets.pop();
            } else {
                break;
            }
        }
        sets.push(set);
    }
    for (begin, end, w, sum) in sets {
        let v = sum / w;
        for &i in &order[begin..end] {
            values[i] = v;
        }
    }
}

/// The classifier's loss (CatBoost's `loss_function`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum CatBoostClassifierLoss {
    /// Logloss for two classes, MultiClass for more.
    #[default]
    Auto,
    /// Logloss on probabilities in [0, 1] as targets (CatBoost's
    /// `CrossEntropy`); classes 0 and 1, no categorical features or class
    /// weights.
    CrossEntropy,
    /// One binary Logloss per class (CatBoost's `MultiClassOneVsAll`):
    /// `predict_proba` gives each class's sigmoid, which CatBoost does not
    /// normalize.
    MultiClassOneVsAll,
}

/// CatBoost's `auto_class_weights`, from the classes' weight sums `S_c`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CatBoostAutoClassWeights {
    /// `max S / S_c`.
    Balanced,
    /// `sqrt(max S / S_c)`.
    SqrtBalanced,
}

/// CatBoost's `CalculateClassWeights` (one thread): the classes' weight
/// sums in float64, stored as float32 and passed to the weight function
/// as integers (its `std::function<float(ui64, ui64)>` truncates them).
fn auto_class_weights(
    classes_of: &[u8],
    n_classes: usize,
    weights: &[f64],
    auto: CatBoostAutoClassWeights,
) -> Vec<f32> {
    let mut sums = vec![0.0f64; n_classes];
    for (&c, &w) in classes_of.iter().zip(weights) {
        sums[c as usize] += w;
    }
    let sums: Vec<f32> = sums.iter().map(|&s| s as f32).collect();
    let max = sums.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let max_int = max as u64 as f32;
    sums.iter()
        .map(|&s| {
            let s = s as u64 as f32;
            if s > 1e-8 {
                match auto {
                    CatBoostAutoClassWeights::Balanced => max_int / s,
                    CatBoostAutoClassWeights::SqrtBalanced => (max_int / s).sqrt(),
                }
            } else {
                1.0
            }
        })
        .collect()
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
#[derive(Clone, Copy, PartialEq)]
enum Loss {
    Rmse,
    /// Quantile (`mae`: CatBoost's MAE, whose metric is doubled).
    Quantile {
        alpha: f64,
        delta: f64,
        mae: bool,
    },
    Expectile {
        alpha: f64,
    },
    Poisson,
    Tweedie {
        power: f64,
    },
    LogLinQuantile {
        alpha: f64,
    },
    Huber {
        delta: f64,
    },
    LogCosh,
    Lq {
        q: f64,
    },
    Logloss,
    /// Softmax over `k` approximations (CatBoost's `MultiClass`); the target
    /// is the class index.
    MultiClass(usize),
    /// A sigmoid per class (CatBoost's `MultiClassOneVsAll`), with a
    /// diagonal Hessian.
    MultiClassOneVsAll(usize),
    /// Mean and log standard deviation (CatBoost's `RMSEWithUncertainty`),
    /// with a diagonal Hessian.
    RmseWithUncertainty,
    /// The RMSE over `k` targets per row (CatBoost's `MultiRMSE`), with a
    /// diagonal Hessian.
    MultiRmse(usize),
    /// The RMSE of the residuals around each query's mean residual
    /// (CatBoost's `QueryRMSE`, a querywise error).
    QueryRmse,
    /// The logistic loss of a query's pairs (CatBoost's `PairLogit`, a
    /// pairwise error on stored `exp(a)`).
    PairLogit,
    /// PairLogit on pairs regenerated every iteration (CatBoost's
    /// `YetiRank`).
    YetiRank {
        permutations: usize,
        decay: f64,
    },
    /// PairLogit with pairwise scoring (CatBoost's `PairLogitPairwise`).
    PairLogitPairwise,
    /// YetiRank with pairwise scoring (CatBoost's `YetiRankPairwise`).
    YetiRankPairwise {
        permutations: usize,
        decay: f64,
    },
}

impl Loss {
    /// Approximations per row.
    fn dims(self) -> usize {
        match self {
            Loss::MultiClass(k) | Loss::MultiClassOneVsAll(k) | Loss::MultiRmse(k) => k,
            Loss::RmseWithUncertainty => 2,
            _ => 1,
        }
    }

    /// A querywise loss: derivatives and metric need the query bounds.
    fn querywise(self) -> bool {
        self == Loss::QueryRmse || self.pairwise()
    }

    /// A pairwise loss: leaves and scores weigh rows by their pairs.
    fn pairwise(self) -> bool {
        matches!(
            self,
            Loss::PairLogit
                | Loss::YetiRank { .. }
                | Loss::PairLogitPairwise
                | Loss::YetiRankPairwise { .. }
        )
    }

    /// Splits and leaves scored on pairs (CatBoost's `IsPairwiseScoring`).
    fn pairwise_scoring(self) -> bool {
        matches!(
            self,
            Loss::PairLogitPairwise | Loss::YetiRankPairwise { .. }
        )
    }

    /// The pairs' redraw parameters (YetiRank, YetiRankPairwise).
    fn yeti_rank(self) -> Option<(usize, f64)> {
        match self {
            Loss::YetiRank {
                permutations,
                decay,
            }
            | Loss::YetiRankPairwise {
                permutations,
                decay,
            } => Some((permutations, decay)),
            _ => None,
        }
    }

    /// Target values per row (the engine's targets are row-major).
    fn target_dims(self) -> usize {
        match self {
            Loss::MultiRmse(k) => k,
            _ => 1,
        }
    }

    /// Second derivatives per row: the upper triangle of the Hessian, row
    /// by row (CatBoost's packed `THessianInfo`).
    fn hessian_len(self) -> usize {
        let k = self.dims();
        match self {
            Loss::MultiClassOneVsAll(_) | Loss::RmseWithUncertainty | Loss::MultiRmse(_) => k,
            _ => k * (k + 1) / 2,
        }
    }

    /// The weighted derivatives of one row into `der` (`dims`) and `der2`
    /// (`hessian_len`); `targets` holds the row's `target_dims` values.
    fn ders(self, targets: &[f64], approx: &[f64], w: f64, der: &mut [f64], der2: &mut [f64]) {
        let target = targets[0];
        match self {
            Loss::Rmse => {
                der[0] = w * (target - approx[0]);
                der2[0] = -w;
            }
            // CatBoost's `TQuantileError`, ... (`CalcDer · w`); the error
            // functions parse `alpha` as float32.
            Loss::Quantile { alpha, delta, .. } => {
                let alpha = f64::from(alpha as f32);
                let val = target - approx[0];
                let d = if val.abs() < delta {
                    0.0
                } else if val > 0.0 {
                    alpha
                } else {
                    -(1.0 - alpha)
                };
                der[0] = w * d;
                der2[0] = 0.0;
            }
            Loss::Expectile { alpha } => {
                let e = target - approx[0];
                let (d, d2) = if e > 0.0 {
                    (2.0 * alpha * e, -2.0 * alpha)
                } else {
                    (2.0 * (1.0 - alpha) * e, -2.0 * (1.0 - alpha))
                };
                der[0] = w * d;
                der2[0] = w * d2;
            }
            Loss::Poisson => {
                let e = approx[0].exp();
                der[0] = w * (target - e);
                der2[0] = w * -e;
            }
            Loss::Tweedie { power } => {
                let (a, b) = (
                    ((1.0 - power) * approx[0]).exp(),
                    ((2.0 - power) * approx[0]).exp(),
                );
                der[0] = w * (target * a - b);
                der2[0] = w * (target * a * (1.0 - power) - b * (2.0 - power));
            }
            Loss::LogLinQuantile { alpha } => {
                let e = approx[0].exp();
                der[0] = w * if target - e > 0.0 {
                    alpha * e
                } else {
                    -(1.0 - alpha) * e
                };
                der2[0] = 0.0;
            }
            // CatBoost's `THuberError`, `TLogCoshError`, `TLqError`.
            Loss::Huber { delta } => {
                let diff = target - approx[0];
                let (d, d2) = if diff.abs() < delta {
                    (diff, -1.0)
                } else if diff > 0.0 {
                    (delta, 0.0)
                } else {
                    (-delta, 0.0)
                };
                der[0] = w * d;
                der2[0] = w * d2;
            }
            Loss::LogCosh => {
                let c = (approx[0] - target).cosh();
                der[0] = w * -(approx[0] - target).tanh();
                der2[0] = w * (-1.0 / (c * c));
            }
            Loss::Lq { q } => {
                let abs = (approx[0] - target).abs();
                let sign = if target - approx[0] > 0.0 { 1.0 } else { -1.0 };
                der[0] = w * (q * sign * abs.powf(q - 1.0));
                der2[0] = w * (-q * (q - 1.0) * abs.powf(q - 2.0));
            }
            Loss::Logloss => {
                let e = approx[0].exp();
                let p = e / (1.0 + e);
                let p = if p.is_nan() { 1.0 } else { p };
                der[0] = w * (target - p);
                der2[0] = w * (-p * (1.0 - p));
            }
            // CatBoost's `TRMSEWithUncertaintyError::CalcDers`.
            Loss::RmseWithUncertainty => {
                let diff = target - approx[0];
                let mut prec = [-2.0 * approx[1]];
                fast_exp_with_inf(&mut prec);
                let sq = diff * diff;
                der[0] = w * diff;
                der[1] = w * (sq * prec[0] - 1.0);
                der2[0] = -w;
                der2[1] = -2.0 * w * sq * prec[0];
            }
            Loss::QueryRmse
            | Loss::PairLogit
            | Loss::YetiRank { .. }
            | Loss::PairLogitPairwise
            | Loss::YetiRankPairwise { .. } => {
                unreachable!("querywise derivatives need the queries")
            }
            // CatBoost's `TMultiRMSEError::CalcDers`.
            Loss::MultiRmse(_) => {
                for (d, (&t, &a)) in targets.iter().zip(approx).enumerate() {
                    der[d] = w * (t - a);
                    der2[d] = -w;
                }
            }
            // CatBoost's `TMultiClassOneVsAllError::CalcDersMulti`.
            Loss::MultiClassOneVsAll(_) => {
                der.copy_from_slice(approx);
                fast_exp_with_inf(der);
                for (d, h) in der.iter_mut().zip(der2.iter_mut()) {
                    let p = *d / (1.0 + *d);
                    *d = -p;
                    *h = -p * (1.0 - p);
                }
                der[target as usize] += 1.0;
                if w != 1.0 {
                    der.iter_mut().for_each(|d| *d *= w);
                    der2.iter_mut().for_each(|d| *d *= w);
                }
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
    fn metric(self, targets: &[f64], approx: &[f64]) -> f64 {
        let target = targets[0];
        match self {
            Loss::Rmse => (target - approx[0]) * (target - approx[0]),
            Loss::Quantile { .. }
            | Loss::Expectile { .. }
            | Loss::Poisson
            | Loss::Tweedie { .. }
            | Loss::LogLinQuantile { .. } => self.metric_moved(target, approx[0], 0.0),
            // CatBoost's `THuberLossMetric`, `TLqMetric`.
            Loss::Huber { delta } => {
                let m = (approx[0] - target).abs();
                if m < delta {
                    0.5 * (m * m)
                } else {
                    delta * (m - 0.5 * delta)
                }
            }
            Loss::LogCosh => self.metric_moved(target, approx[0], 0.0),
            Loss::Lq { q } => (approx[0] - target).abs().powf(q),
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
            // `TRMSEWithUncertaintyMetric`: the Gaussian negative
            // log-likelihood.
            Loss::RmseWithUncertainty => {
                let mut e = [-2.0 * approx[1]];
                fast_exp_with_inf(&mut e);
                let d = approx[0] - target;
                0.9189385332046 + approx[1] + 0.5 * e[0] * (d * d)
            }
            Loss::QueryRmse
            | Loss::PairLogit
            | Loss::YetiRank { .. }
            | Loss::PairLogitPairwise
            | Loss::YetiRankPairwise { .. } => {
                unreachable!("the querywise metric needs the queries")
            }
            // `TMultiRMSEMetric`'s squared errors.
            Loss::MultiRmse(_) => targets
                .iter()
                .zip(approx)
                .map(|(&t, &a)| (a - t) * (a - t))
                .sum(),
            // `TMultiClassOneVsAllMetric`.
            Loss::MultiClassOneVsAll(k) => {
                let mut sum = 0.0;
                for &a in approx {
                    let e = a.exp();
                    sum += if e.is_finite() { -(1.0 + e).ln() } else { -a };
                }
                sum += approx[target as usize];
                -(sum / k as f64)
            }
        }
    }

    /// CatBoost's metric of one row at `approx` moved by the leaf value
    /// `delta`, in its order of operations (`TQuantileMetric`,
    /// `TExpectileMetric`, `TPoissonMetric` on stored `exp(a)`,
    /// `TTweedieMetric`, `TLogLinQuantileMetric`).
    fn metric_moved(self, target: f64, approx: f64, delta: f64) -> f64 {
        match self {
            Loss::Quantile {
                alpha,
                delta: d,
                mae,
            } => {
                let mut val = target - approx - delta;
                let m = if val.abs() < d {
                    0.0
                } else if val > 0.0 {
                    alpha
                } else {
                    -(1.0 - alpha)
                };
                if val < -d {
                    val += d;
                } else if val > d {
                    val -= d;
                }
                if mae {
                    2.0 * (m * val)
                } else {
                    m * val
                }
            }
            Loss::Expectile { alpha } => {
                let val = target - approx - delta;
                let m = if val > 0.0 { alpha } else { 1.0 - alpha };
                m * (val * val)
            }
            Loss::Poisson => {
                let e = approx.exp() * delta.exp();
                e - target * f64::from(fast_logf(e as f32))
            }
            Loss::Tweedie { power } => {
                let a = approx + delta;
                let margin = -target * ((1.0 - power) * a).exp() / (1.0 - power);
                margin + ((2.0 - power) * a).exp() / (2.0 - power)
            }
            Loss::LogLinQuantile { alpha } => {
                let val = target - approx.exp() * delta.exp();
                (if val > 0.0 { alpha } else { -(1.0 - alpha) }) * val
            }
            // `TLogCoshMetric`: FastLogf of cosh, linear far out.
            Loss::LogCosh => {
                let m = approx - target + delta;
                if m.abs() >= 12.0 {
                    m.abs() - f64::from(fast_logf(2.0))
                } else {
                    f64::from(fast_logf(m.cosh() as f32))
                }
            }
            Loss::Lq { q } => (approx - target + delta).abs().powf(q),
            _ => self.metric(&[target], &[approx + delta]),
        }
    }

    /// CatBoost's objective metric on an eval set, whose approximations it
    /// keeps on the raw scale (`TestApprox`).
    fn eval_metric(self, target: &[f64], approx: &[f64], weights: &[f64], groups: &Groups) -> f64 {
        if self == Loss::QueryRmse {
            return query_rmse(target, approx, &[], &[], weights, &groups.queries);
        }
        if self == Loss::PairLogit {
            return pair_logit_metric(approx, &[], &[], groups, approx.len(), true);
        }
        if let Loss::MultiRmse(k) = self {
            return multi_rmse(target, approx, &[], &[], weights, k);
        }
        let k = self.dims();
        let (mut sum, mut sum_w) = (0.0, 0.0);
        for (i, (&t, &w)) in target.iter().zip(weights).enumerate() {
            let a = &approx[i * k..(i + 1) * k];
            let m = match self {
                Loss::Poisson => a[0].exp() - t * a[0],
                Loss::LogLinQuantile { alpha } => {
                    let v = t - a[0].exp();
                    (if v > 0.0 { alpha } else { -(1.0 - alpha) }) * v
                }
                _ => self.metric(&[t], a),
            };
            sum += m * w;
            sum_w += w;
        }
        match self {
            Loss::Rmse => (sum / (sum_w + 1e-38)).sqrt(),
            Loss::LogCosh => sum / (sum_w + 1e-38),
            _ if sum_w == 0.0 => 0.0,
            _ => sum / sum_w,
        }
    }

    /// Whether CatBoost stores this loss's approximations as `exp(a)`
    /// (`IsStoreExpApprox`), which changes how its metric and learning
    /// folds are computed.
    fn exp_approx(self) -> bool {
        matches!(
            self,
            Loss::Logloss
                | Loss::Poisson
                | Loss::LogLinQuantile { .. }
                | Loss::PairLogit
                | Loss::YetiRank { .. }
                | Loss::PairLogitPairwise
                | Loss::YetiRankPairwise { .. }
        )
    }

    /// CatBoost's default `boost_from_average`: on for RMSE, MAE and
    /// Quantile.
    fn default_boost_from_average(self) -> bool {
        matches!(
            self,
            Loss::Rmse | Loss::Quantile { .. } | Loss::MultiRmse(_)
        )
    }

    /// Predictions are `exp(raw)` (CatBoost's `Exponent` prediction type).
    fn exp_prediction(self) -> bool {
        matches!(self, Loss::Poisson | Loss::Tweedie { .. })
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
        groups: &Groups,
    ) -> f64 {
        if self.pairwise() {
            return pair_logit_metric(approx, leaves, leaf_of, groups, rows, false);
        }
        if self == Loss::Logloss {
            return logloss_metric_exp(target, approx, leaves, leaf_of, weights, rows);
        }
        if self == Loss::QueryRmse {
            return query_rmse(
                target,
                approx,
                leaves,
                leaf_of,
                weights,
                query_prefix(&groups.queries, rows),
            );
        }
        if let Loss::MultiRmse(k) = self {
            return multi_rmse(
                &target[..rows * k],
                &approx[..rows * k],
                leaves,
                leaf_of,
                &weights[..rows],
                k,
            );
        }
        if self.dims() == 1 && self != Loss::Rmse {
            let (mut sum, mut sum_w) = (0.0, 0.0);
            for i in 0..rows {
                let w = weights[i];
                sum += self.metric_moved(target[i], approx[i], leaves[leaf_of[i] as usize]) * w;
                sum_w += w;
            }
            return if self == Loss::LogCosh {
                sum / (sum_w + 1e-38)
            } else {
                sum / sum_w
            };
        }
        let k = self.dims();
        let (mut sum, mut sum_w) = (0.0, 0.0);
        let mut point = vec![0.0; k];
        for i in 0..rows {
            let leaf = leaf_of[i] as usize;
            for d in 0..k {
                point[d] = approx[i * k + d] + leaves[leaf * k + d];
            }
            sum += weights[i] * self.metric(&target[i..=i], &point);
            sum_w += weights[i];
        }
        match self {
            Loss::Rmse => (sum / (sum_w + 1e-38)).sqrt(),
            _ => sum / sum_w,
        }
    }

    /// The weighted derivatives of the rows `0..rows`.
    fn all_ders(
        self,
        target: &[f64],
        approx: &[f64],
        weights: &[f64],
        rows: usize,
        groups: &Groups,
    ) -> Ders {
        let (k, h, t) = (self.dims(), self.hessian_len(), self.target_dims());
        let mut out = Ders {
            k,
            h,
            der: vec![0.0; rows * k],
            der2: vec![0.0; rows * h],
        };
        if self == Loss::QueryRmse {
            // CatBoost's `TQueryRmseError::CalcDersForQueries`.
            for q in query_prefix(&groups.queries, rows).windows(2) {
                let avg = query_average(target, approx, &[], &[], weights, q[0], q[1]);
                for i in q[0]..q[1] {
                    out.der[i] = (target[i] - approx[i] - avg) * weights[i];
                    out.der2[i] = -weights[i];
                }
            }
            return out;
        }
        if self.pairwise() {
            // CatBoost's `TPairLogitError::CalcDersForQueries` on `exp(a)`;
            // the competitors of the rows `0..rows` are among them.
            let e: Vec<f64> = approx[..rows].iter().map(|a| a.exp()).collect();
            for w in 0..rows {
                let (mut der, mut der2) = (0.0, 0.0);
                for &(l, cw) in groups.competitors(w) {
                    let cw = f64::from(cw);
                    let p = e[l] / (e[l] + e[w]);
                    der += cw * p;
                    out.der[l] -= cw * p;
                    der2 += cw * p * (p - 1.0);
                    out.der2[l] += cw * p * (p - 1.0);
                }
                out.der[w] += der;
                out.der2[w] += der2;
            }
            return out;
        }
        for i in 0..rows {
            self.ders(
                &target[i * t..(i + 1) * t],
                &approx[i * k..(i + 1) * k],
                weights[i],
                &mut out.der[i * k..(i + 1) * k],
                &mut out.der2[i * h..(i + 1) * h],
            );
        }
        out
    }
}

/// Query groups of the rows (querywise losses): the query bounds, and for
/// pairwise losses each row's competitors (the rows it beats, with the
/// pair weight; CatBoost's `TQueryInfo::Competitors`).
#[derive(Debug, Clone, Default)]
struct Groups {
    /// Row offsets of the queries, from 0 to the row count (empty: none).
    queries: Vec<usize>,
    /// Row `i`'s competitors are `comp[comp_start[i]..comp_start[i + 1]]`.
    comp_start: Vec<usize>,
    comp: Vec<(usize, f32)>,
    /// Pairwise scoring: each pair's bootstrapped weight
    /// (`TCompetitor::SampleWeight`; empty: the pair weights).
    sample: Vec<f32>,
}

impl Groups {
    fn competitors(&self, row: usize) -> &[(usize, f32)] {
        &self.comp[self.comp_start[row]..self.comp_start[row + 1]]
    }

    /// CatBoost's `UnpackPairsFromQueries` of the rows `0..rows`: (winner,
    /// loser, sample weight) in winner order.
    fn flat_pairs(&self, rows: usize) -> Vec<(usize, usize, f32)> {
        let mut out = Vec::with_capacity(self.comp_start[rows]);
        for w in 0..rows {
            for idx in self.comp_start[w]..self.comp_start[w + 1] {
                let (l, cw) = self.comp[idx];
                let sw = if self.sample.is_empty() {
                    cw
                } else {
                    self.sample[idx]
                };
                out.push((w, l, sw));
            }
        }
        out
    }

    /// CatBoost's bootstrap of pairwise scoring (`GenerateBernoulliWeights
    /// ForPairs` / `GenerateBayesianWeightsForPairs`, per-object sampling):
    /// blocks of 1000 queries, each with its own generator; a draw per
    /// query, then one per pair.
    fn bootstrap_pairs(
        &mut self,
        rng: &mut CatRng,
        kind: CatBoostBootstrap,
        rate: f64,
        temperature: f64,
    ) {
        match kind {
            CatBoostBootstrap::Bernoulli if rate != 1.0 => {}
            CatBoostBootstrap::Bayesian if temperature != 0.0 => {}
            _ => return,
        }
        let seed = rng.gen();
        let n_queries = self.queries.len().saturating_sub(1);
        if self.sample.len() != self.comp.len() {
            self.sample = self.comp.iter().map(|&(_, w)| w).collect();
        }
        let rate = f64::from(rate as f32);
        let t = temperature as f32;
        for b in 0..n_queries.div_ceil(1000) {
            let mut local = CatRng::new(seed.wrapping_add(b as u64));
            local.advance(10);
            for q in b * 1000..((b + 1) * 1000).min(n_queries) {
                local.real1(); // the query's draw (per-object sampling ignores it)
                for idx in self.comp_start[self.queries[q]]..self.comp_start[self.queries[q + 1]] {
                    let weight = self.comp[idx].1;
                    self.sample[idx] = if kind == CatBoostBootstrap::Bernoulli {
                        if local.real1() < rate {
                            weight
                        } else {
                            0.0
                        }
                    } else {
                        let w = (-fast_logf((local.real1() + 1e-100) as f32)).powf(t);
                        weight * w
                    };
                }
            }
        }
    }

    /// CatBoost's PairLogit pairs (`GeneratePairLogitPairs`, all of them):
    /// in each query, every row beats the rows of a smaller target, in row
    /// order, with weight 1.
    fn with_pairs(queries: Vec<usize>, target: &[f64]) -> Self {
        let mut comp_start = vec![0];
        let mut comp = Vec::new();
        for q in queries.windows(2) {
            for w in q[0]..q[1] {
                let t = target[w] as f32;
                comp.extend(
                    (q[0]..q[1])
                        .filter(|&l| (target[l] as f32) < t)
                        .map(|l| (l, 1.0)),
                );
                comp_start.push(comp.len());
            }
        }
        Self {
            queries,
            comp_start,
            comp,
            sample: Vec::new(),
        }
    }

    /// The groups of the rows in the order `perm` (row `i` is the row
    /// `perm[i]`) with the query bounds `queries` (`GetGroupInfosSubset`:
    /// each row keeps its competitors in their order).
    fn permuted(&self, perm: &[usize], queries: Vec<usize>) -> Self {
        if self.comp_start.is_empty() {
            return Self {
                queries,
                ..Self::default()
            };
        }
        let mut inverse = vec![0; perm.len()];
        for (i, &r) in perm.iter().enumerate() {
            inverse[r] = i;
        }
        let mut comp_start = vec![0];
        let mut comp = Vec::with_capacity(self.comp.len());
        for &r in perm {
            comp.extend(self.competitors(r).iter().map(|&(l, w)| (inverse[l], w)));
            comp_start.push(comp.len());
        }
        Self {
            queries,
            comp_start,
            comp,
            sample: Vec::new(),
        }
    }

    /// CatBoost's `CalcPairwiseWeights`: each row's summed pair weights
    /// (float32).
    fn pair_weights(&self, n: usize) -> Vec<f64> {
        let mut w = vec![0f32; n];
        for row in 0..n {
            for &(l, cw) in self.competitors(row) {
                w[row] += cw;
                w[l] += cw;
            }
        }
        w.into_iter().map(f64::from).collect()
    }
}

/// library/cpp/dot_product's `DotProduct` of doubles as CatBoost runs it
/// on AVX2 + FMA: SSE pairs below 16 values, else four lanes with the
/// leftover first and fused multiply-adds.
fn catboost_dot(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 16 {
        let (mut s1, mut s2) = ([0.0f64; 2], [0.0f64; 2]);
        let mut i = 0;
        while n - i >= 4 {
            s1[0] += a[i] * b[i];
            s1[1] += a[i + 1] * b[i + 1];
            s2[0] += a[i + 2] * b[i + 2];
            s2[1] += a[i + 3] * b[i + 3];
            i += 4;
        }
        if n - i >= 2 {
            s1[0] += a[i] * b[i];
            s1[1] += a[i + 1] * b[i + 1];
            i += 2;
        }
        s1[0] += s2[0];
        s1[1] += s2[1];
        if n - i > 0 {
            s1[0] += 0.0;
            s1[1] += a[i] * b[i];
        }
        return s1[0] + s1[1];
    }
    let (mut s1, mut s2) = ([0.0f64; 4], [0.0f64; 4]);
    let mut i = n % 4;
    for l in 0..i {
        s1[l] = a[l] * b[l];
    }
    while n - i >= 8 {
        for l in 0..4 {
            s1[l] = a[i + l].mul_add(b[i + l], s1[l]);
            s2[l] = a[i + 4 + l].mul_add(b[i + 4 + l], s2[l]);
        }
        i += 8;
    }
    if n - i > 0 {
        for l in 0..4 {
            s1[l] = a[i + l].mul_add(b[i + l], s1[l]);
        }
    }
    let s: Vec<f64> = (0..4).map(|l| s1[l] + s2[l]).collect();
    (s[0] + s[2]) + (s[1] + s[3])
}

/// CatBoost's `PFound` (decay 0.85, all positions) or `NDCG` (Base,
/// LogPosition, all positions) of raw approximations over the queries:
/// rows ranked by score, ties pessimistically (`CompareDocs`).
fn ranking_metric(target: &[f64], approx: &[f64], queries: &[usize], pfound: bool) -> f64 {
    let (mut sum, mut count) = (0.0, 0.0);
    let mut decay: Vec<f64> = Vec::new();
    for q in queries.windows(2) {
        let (begin, end) = (q[0], q[1]);
        let rel = &target[begin..end];
        let score = &approx[begin..end];
        let mut order: Vec<usize> = (0..end - begin).collect();
        order.sort_by(|&l, &r| {
            use std::cmp::Ordering;
            if score[l] != score[r] {
                score[r].partial_cmp(&score[l]).unwrap_or(Ordering::Equal)
            } else {
                rel[l].partial_cmp(&rel[r]).unwrap_or(Ordering::Equal)
            }
        });
        if pfound {
            let (mut look, mut found) = (1.0f64, 0.0f64);
            for &d in &order {
                found += rel[d] * look;
                look *= (1.0 - rel[d]) * 0.85;
            }
            sum += found;
        } else {
            while decay.len() < order.len() {
                let i = decay.len();
                decay.push(if i == 0 {
                    1.0
                } else {
                    1.0 / ((i + 2) as f64).log2()
                });
            }
            let ranked: Vec<f64> = order.iter().map(|&d| rel[d]).collect();
            let dcg = catboost_dot(&ranked, &decay[..ranked.len()]);
            let mut ideal = rel.to_vec();
            ideal.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let idcg = catboost_dot(&ideal, &decay[..ideal.len()]);
            sum += if idcg > 0.0 { dcg / idcg } else { 1.0 };
        }
        count += 1.0;
    }
    if count != 0.0 {
        sum / count
    } else {
        0.0
    }
}

/// CatBoost's `GenRandUI64Vector`: `n` draws of a generator seeded `seed`.
fn seed_vector(n: usize, seed: u64) -> Vec<u64> {
    let mut rng = CatRng::new(seed);
    (0..n).map(|_| rng.gen()).collect()
}

/// CatBoost's `UpdatePairsForYetiRank` (Classic mode, Gumbel noise) on the
/// queries of the rows `0..rows` at the raw approximations `approx`: per
/// query `permutations` rankings by `exp(a)` times Gumbel noise, each
/// neighbouring pair of different relevance weighted `0.15 · decay^pos ·
/// |Δ relevance|`, averaged.
fn yeti_rank_groups(
    all_queries: &[usize],
    target: &[f64],
    approx: &[f64],
    rows: usize,
    seed: u64,
    permutations: usize,
    decay: f64,
) -> Groups {
    let queries = query_prefix(all_queries, rows);
    let n_queries = queries.len() - 1;
    // `TExecRangeParams::SetBlockCount(CB_THREAD_LIMIT)`.
    let block_size = n_queries.div_ceil(128).max(1);
    let block_seeds = seed_vector(n_queries.div_ceil(block_size), seed);
    let mut comp_start = vec![0];
    let mut comp = Vec::new();
    let mut weights = Vec::new();
    for (b, &block_seed) in block_seeds.iter().enumerate() {
        let mut block_rng = CatRng::new(block_seed);
        for q in b * block_size..((b + 1) * block_size).min(n_queries) {
            let (begin, end) = (queries[q], queries[q + 1]);
            let size = end - begin;
            let rel: Vec<f32> = target[begin..end].iter().map(|&t| t as f32).collect();
            let mut rng = CatRng::new(block_rng.gen());
            weights.clear();
            weights.resize(size * size, 0f32);
            let mut order: Vec<usize> = Vec::with_capacity(size);
            for _ in 0..permutations {
                let noisy: Vec<f64> = approx[begin..end]
                    .iter()
                    .map(|a| {
                        let u = rng.real1() as f32;
                        a.exp() * f64::from(u / (1.000001f32 - u))
                    })
                    .collect();
                order.clear();
                order.extend(0..size);
                // `StableSort` by the noisy score, descending.
                order.sort_by(|&i, &j| {
                    noisy[j]
                        .partial_cmp(&noisy[i])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut coef = 1.0f64;
                for d in 1..size {
                    let (f, s) = (order[d - 1], order[d]);
                    let w = (0.15 * coef * f64::from((rel[f] - rel[s]).abs())) as f32;
                    if rel[f] > rel[s] {
                        weights[f * size + s] += w;
                    } else if rel[f] < rel[s] {
                        weights[s * size + f] += w;
                    }
                    coef *= decay;
                }
            }
            for w in 0..size {
                for l in 0..size {
                    let cw = 1.0f32 * weights[w * size + l] / permutations as f32;
                    if cw != 0.0 {
                        comp.push((begin + l, cw));
                    }
                }
                comp_start.push(comp.len());
            }
        }
    }
    // Rows past the queries have no competitors.
    let n = target.len();
    while comp_start.len() <= n {
        comp_start.push(comp.len());
    }
    Groups {
        queries: all_queries.to_vec(),
        comp_start,
        comp,
        sample: Vec::new(),
    }
}

/// CatBoost's `TPairLogitMetric` over the pairs of the rows `0..rows`, at
/// `approx` moved by `leaves` when given. An eval set (`raw`) keeps raw
/// approximations: shifted by the query maximum, through fast exp, plus
/// 1e-38.
fn pair_logit_metric(
    approx: &[f64],
    leaves: &[f64],
    leaf_of: &[u32],
    groups: &Groups,
    rows: usize,
    raw: bool,
) -> f64 {
    let (mut sum, mut sum_w) = (0.0, 0.0);
    let mut e = Vec::new();
    for q in query_prefix(&groups.queries, rows).windows(2) {
        e.clear();
        e.extend((q[0]..q[1]).map(|i| {
            let delta = if leaves.is_empty() {
                0.0
            } else {
                leaves[leaf_of[i] as usize]
            };
            approx[i] + delta
        }));
        if raw {
            let max = e.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            e.iter_mut().for_each(|v| *v -= max);
            fast_exp_with_inf(&mut e);
            e.iter_mut().for_each(|v| *v += 1e-38);
        } else {
            e.iter_mut().for_each(|v| *v = v.exp());
        }
        for w in q[0]..q[1] {
            for &(l, cw) in groups.competitors(w) {
                let (ew, el) = (e[w - q[0]], e[l - q[0]]);
                let cw = f64::from(cw);
                sum += -cw * (ew / (ew + el)).ln();
                sum_w += cw;
            }
        }
    }
    sum / sum_w
}

/// The query bounds `queries` (row offsets, first 0) of the rows `0..rows`,
/// which end on a query bound.
fn query_prefix(queries: &[usize], rows: usize) -> &[usize] {
    let end = queries.partition_point(|&b| b <= rows);
    debug_assert_eq!(queries[end - 1], rows);
    &queries[..end]
}

/// The weighted mean residual of the rows `begin..end` (CatBoost's
/// `CalcQueryAvrg`), at `approx` moved by `leaves` when given.
fn query_average(
    target: &[f64],
    approx: &[f64],
    leaves: &[f64],
    leaf_of: &[u32],
    weights: &[f64],
    begin: usize,
    end: usize,
) -> f64 {
    let (mut sum, mut count) = (0.0, 0.0);
    for i in begin..end {
        let delta = if leaves.is_empty() {
            0.0
        } else {
            leaves[leaf_of[i] as usize]
        };
        sum += (target[i] - approx[i] - delta) * weights[i];
        count += weights[i];
    }
    if count > 0.0 {
        sum / count
    } else {
        0.0
    }
}

/// CatBoost's `TQueryRMSEMetric` over the queries `queries`.
fn query_rmse(
    target: &[f64],
    approx: &[f64],
    leaves: &[f64],
    leaf_of: &[u32],
    weights: &[f64],
    queries: &[usize],
) -> f64 {
    let (mut sum, mut sum_w) = (0.0, 0.0);
    for q in queries.windows(2) {
        let avg = query_average(target, approx, leaves, leaf_of, weights, q[0], q[1]);
        for i in q[0]..q[1] {
            let delta = if leaves.is_empty() {
                0.0
            } else {
                leaves[leaf_of[i] as usize]
            };
            let e = target[i] - approx[i] - delta - avg;
            sum += (e * e) * weights[i];
            sum_w += weights[i];
        }
    }
    (sum / (sum_w + 1e-38)).sqrt()
}

/// CatBoost's `TMultiRMSEMetric` of row-major targets and approximations
/// (moved by `leaves` when given): squared errors summed dimension by
/// dimension, then over the weights.
fn multi_rmse(
    target: &[f64],
    approx: &[f64],
    leaves: &[f64],
    leaf_of: &[u32],
    weights: &[f64],
    k: usize,
) -> f64 {
    let mut sum = 0.0;
    for d in 0..k {
        for (i, &w) in weights.iter().enumerate() {
            let mut a = approx[i * k + d];
            if !leaves.is_empty() {
                a += leaves[leaf_of[i] as usize * k + d];
            }
            let e = a - target[i * k + d];
            sum += w * (e * e);
        }
    }
    let sum_w: f64 = weights.iter().sum();
    if sum_w == 0.0 {
        0.0
    } else {
        (sum / sum_w).sqrt()
    }
}

/// Rows per parallel chunk of the per-row loops of the leaf estimation.
const PARALLEL_ROWS: usize = 1024;

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
    let point = exp_point(&approx[..rows]);
    logloss_metric_exp_at(target, &point, leaves, leaf_of, weights)
}

/// Every row's `exp(a)` and `FastLogf(exp(a))` (what the Logloss metric
/// needs of the approximations; cached while they do not move).
fn exp_point(approx: &[f64]) -> Vec<(f64, f32)> {
    use rayon::prelude::*;
    approx
        .par_iter()
        .with_min_len(PARALLEL_ROWS)
        .map(|a| {
            let e = a.exp();
            (e, fast_logf(e as f32))
        })
        .collect()
}

/// [`Loss::all_ders`] of Logloss at approximations given by
/// [`exp_point`] (the same arithmetic on the cached `exp(a)`).
fn logloss_ders_at(target: &[f64], point: &[(f64, f32)], weights: &[f64]) -> Ders {
    use rayon::prelude::*;
    let (der, der2) = point
        .par_iter()
        .with_min_len(PARALLEL_ROWS)
        .zip(target)
        .zip(weights)
        .map(|((&(e, _), &t), &w)| {
            let p = e / (1.0 + e);
            let p = if p.is_nan() { 1.0 } else { p };
            (w * (t - p), w * (-p * (1.0 - p)))
        })
        .unzip();
    Ders {
        k: 1,
        h: 1,
        der,
        der2,
    }
}

/// [`logloss_metric_exp`] of approximations given by [`exp_point`].
fn logloss_metric_exp_at(
    target: &[f64],
    point: &[(f64, f32)],
    leaves: &[f64],
    leaf_of: &[u32],
    weights: &[f64],
) -> f64 {
    let rows = point.len();
    let exp_leaves: Vec<f64> = leaves.iter().map(|v| v.exp()).collect();
    let log_leaves: Vec<f32> = exp_leaves.iter().map(|&d| fast_logf(d as f32)).collect();
    // The blocks in parallel, added in order.
    use rayon::prelude::*;
    let block = |start: usize| {
        let end = (start + 4096).min(rows);
        let (mut lanes0, mut lanes1) = ([0.0f32; 4], [0.0f32; 4]);
        let mut i = start;
        while i + 4 <= end {
            for lane in 0..4 {
                let r = i + lane;
                let prob = if target[r] > 0.5 { 1.0f32 } else { 0.0 };
                let leaf = leaf_of[r] as usize;
                let ((e, le), d) = (point[r], exp_leaves[leaf]);
                let non_exp = le + log_leaves[leaf];
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
        let lanes = (
            f64::from((lanes0[0] + lanes0[1]) + (lanes0[2] + lanes0[3])),
            f64::from((lanes1[0] + lanes1[1]) + (lanes1[2] + lanes1[3])),
        );
        // The rows past the last group of four.
        let rest: Vec<(f64, f64)> = (i..end)
            .map(|r| {
                let w = f64::from(weights[r] as f32);
                let prob = if target[r] > 0.5 { 1.0 } else { 0.0 };
                let leaf = leaf_of[r] as usize;
                let ((e, le), d) = (point[r], exp_leaves[leaf]);
                let mut non_exp = f64::from(le);
                let e = e * d;
                non_exp += f64::from(log_leaves[leaf]);
                let value = if e.is_finite() {
                    f64::from(fast_logf((1.0 + e) as f32)) - prob * non_exp
                } else {
                    (1.0 - prob) * non_exp
                };
                (w * value, w)
            })
            .collect();
        (lanes, rest)
    };
    let blocks: Vec<_> = (0..rows).into_par_iter().step_by(4096).map(block).collect();
    let (mut s0, mut s1) = (0.0f64, 0.0f64);
    for ((l0, l1), rest) in blocks {
        s0 += l0;
        s1 += l1;
        for (v, w) in rest {
            s0 += v;
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
#[derive(Clone)]
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
    /// Pairwise scoring's prior (`bayesian_matrix_reg`, float32).
    pairwise_prior: f64,
    bootstrap: CatBoostBootstrap,
    subsample: f64,
    bagging_temperature: f64,
    mvs_reg: Option<f64>,
    permutation_count: usize,
    /// Per numeric feature: -1, 0 or +1 (empty: no constraints).
    monotone: Vec<i8>,
    /// CatBoost's `model_shrink_rate` (0 = none) and `model_shrink_mode`.
    shrink_rate: f64,
    shrink_mode: CatBoostShrinkMode,
    /// The overfitting detector's p-value threshold (0: off; `Iter`: 1)
    /// and wait.
    od_threshold: f64,
    od_wait: usize,
    /// Shrink the model to its best iteration on the eval set.
    use_best_model: bool,
    grow_policy: CatBoostGrowPolicy,
    min_data_in_leaf: usize,
    max_leaves: usize,
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

/// CatBoost's `CalcScoresForLeaf`: every split of a leaf's buckets, as
/// `visit(split, true side, false side)`.
fn each_split(
    one_hot: bool,
    buckets: &[BucketStats],
    n_splits: usize,
    mut visit: impl FnMut(usize, &BucketStats, &BucketStats),
) {
    let mut all = BucketStats::default();
    for b in buckets {
        all.add(b);
    }
    if one_hot {
        let mut f = all;
        for (s, b) in buckets.iter().enumerate() {
            f.remove(b);
            visit(s, b, &f);
            f.add(b);
        }
    } else {
        let (mut t, mut f) = (all, BucketStats::default());
        for (s, b) in buckets[..n_splits].iter().enumerate() {
            f.add(b);
            t.remove(b);
            visit(s, &t, &f);
        }
    }
}

impl Column {
    /// The monotone constraint of the column (numeric features only).
    fn monotone(&self, constraints: &[i8]) -> i8 {
        match self.split {
            ColumnSplit::Float { feature, .. } => constraints.get(feature).copied().unwrap_or(0),
            _ => 0,
        }
    }

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
    monotone: &[i8],
    pairwise: Option<PairwiseScoring>,
) -> (Vec<(ColumnRef, usize)>, Vec<u32>) {
    use rayon::prelude::*;
    let n = w.len();
    let mut score_before_split = 0.0;
    let mut splits: Vec<(ColumnRef, usize)> = Vec::new();
    let mut leaf_of = vec![0u32; n];
    // Plain boosting with one approximation: the taken rows and their
    // (weighted derivative, weight), gathered once per tree.
    let plain_taken = (!ordered && k == 1 && parts.len() == 1).then(|| {
        let part = &parts[0];
        let rows: Vec<u32> = (0..part.tail as u32)
            .filter(|&i| sample.taken[i as usize])
            .collect();
        let contrib: Vec<(f64, f64)> = rows
            .iter()
            .map(|&i| (part.swder[i as usize], sample.weights[i as usize]))
            .collect();
        (rows, contrib)
    });
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
        // Plain boosting with one approximation: the taken rows' leaves.
        let plain_rows = plain_taken.as_ref().map(|(rows, contrib)| {
            let leaves: Vec<u32> = rows.iter().map(|&i| leaf_of[i as usize]).collect();
            (rows.as_slice(), leaves, contrib.as_slice())
        });
        // Pairwise scoring: each pack of binary features is scored as one
        // candidate (its bits' scores, from the pack values).
        let mut pack_scores = std::collections::HashMap::new();
        if let Some(pw) = &pairwise {
            let wder = parts[0].wder;
            for entry in candidate_entries(q, &candidates) {
                let bins = &entry[0];
                let binary = bins.iter().all(|&(c, _)| {
                    matches!(&candidates[c].get(q).split, ColumnSplit::Float { borders, .. } if borders.len() == 1)
                });
                if !binary {
                    continue;
                }
                let mut values = vec![0u32; n];
                for (bit, &(c, _)) in bins.iter().enumerate() {
                    let column = candidates[c].get(q);
                    for (v, &b) in values.iter_mut().zip(&column.buckets) {
                        *v |= u32::from(b) << bit;
                    }
                }
                let scores = pairwise_split_scores(
                    &values,
                    1 << bins.len(),
                    bins.len(),
                    &leaf_of,
                    leaves,
                    wder,
                    &pw.pairs,
                    pw.l2,
                    pw.prior,
                );
                for (&(c, _), score) in bins.iter().zip(scores) {
                    pack_scores.insert(c, score);
                }
            }
        }
        // The score of every split of every column.
        let per_column: Vec<Option<Vec<f64>>> = candidates
            .par_iter()
            .enumerate()
            .map(|(ci, c)| {
                let column = c.get(q);
                let nb = column.n_buckets;
                if let Some(pw) = &pairwise {
                    if nb < 2 {
                        return None;
                    }
                    if let Some(&score) = pack_scores.get(&ci) {
                        return Some(vec![score]);
                    }
                    let buckets: Vec<u32> = column.buckets.iter().map(|&b| u32::from(b)).collect();
                    return Some(pairwise_split_scores(
                        &buckets,
                        nb,
                        0,
                        &leaf_of,
                        leaves,
                        parts[0].wder,
                        &pw.pairs,
                        pw.l2,
                        pw.prior,
                    ));
                }
                let one_hot = column.one_hot();
                let n_splits = if one_hot { nb } else { nb - 1 };
                if nb < 2 {
                    return None;
                }
                // Per part and dimension: leaves × buckets.
                let mut hist = vec![BucketStats::default(); parts.len() * k * leaves * nb];
                if let Some((rows, row_leaves, contrib)) = &plain_rows {
                    // Plain boosting, one approximation: the taken rows in
                    // order, as the general loop sums them.
                    for ((&i, &leaf), &(swd, sw)) in rows.iter().zip(row_leaves).zip(*contrib) {
                        let h = &mut hist[leaf as usize * nb + column.buckets[i as usize] as usize];
                        h.swd += swd;
                        h.sw += sw;
                    }
                }
                for (pi, part) in parts.iter().enumerate().filter(|_| plain_rows.is_none()) {
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
                // Monotone constraints (CatBoost's `UpdateScores`): the
                // leaf values of every split, made monotone by isotonic
                // regression along the tree's leaf orders, then scored leaf
                // by leaf (`AddLeaf`: the false sides, then the true ones).
                if !monotone.is_empty() {
                    let mut constraints: Vec<i8> = splits
                        .iter()
                        .map(|(c, _)| c.get(q).monotone(monotone))
                        .collect();
                    constraints.push(column.monotone(monotone));
                    let orders = monotone_orders(&constraints);
                    let nl = 2 * leaves;
                    for (pi, part) in parts.iter().enumerate() {
                        let l2 = part.scaled_l2;
                        for d in 0..k {
                            let mut delta = vec![0.0f64; n_splits * nl];
                            let mut body_w = vec![0.0f64; n_splits * nl];
                            let mut tail = vec![(0.0f64, 0.0f64); n_splits * nl];
                            for leaf in 0..leaves {
                                let start = ((pi * k + d) * leaves + leaf) * nb;
                                each_split(
                                    one_hot,
                                    &hist[start..start + nb],
                                    n_splits,
                                    |s, t, f| {
                                        for (id, st) in [(leaf, f), (leaf + leaves, t)] {
                                            let (v, w) = if ordered {
                                                (average(st.sd, st.count, l2), st.count)
                                            } else {
                                                (average(st.swd, st.sw, l2), st.sw)
                                            };
                                            delta[s * nl + id] = v;
                                            body_w[s * nl + id] = w + l2;
                                            tail[s * nl + id] = (st.swd, st.sw);
                                        }
                                    },
                                );
                            }
                            for s in 0..n_splits {
                                let values = &mut delta[s * nl..(s + 1) * nl];
                                let weights = &body_w[s * nl..(s + 1) * nl];
                                for order in &orders {
                                    isotonic(values, weights, order);
                                }
                                for (v, &(swd, sw)) in values.iter().zip(&tail[s * nl..]) {
                                    num[s] += v * swd;
                                    if score_function == CatBoostScoreFunction::Cosine {
                                        den[s] += v * v * sw;
                                    }
                                }
                            }
                        }
                    }
                }
                for (pi, part) in parts.iter().enumerate().filter(|_| monotone.is_empty()) {
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
                    for leaf in (0..k).flat_map(|d| (0..leaves).map(move |l| (d, l))) {
                        let start = ((pi * k + leaf.0) * leaves + leaf.1) * nb;
                        each_split(one_hot, &hist[start..start + nb], n_splits, &mut add);
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
        // Each entry's best noisy instance per candidate (from the entry's
        // own generator, in parallel), then the candidates' noise from the
        // main one, in order.
        let instances: Vec<Vec<_>> = entries
            .par_iter()
            .enumerate()
            .map(|(t, entry)| {
                let mut local = CatRng::new(noise_seed.wrapping_add(t as u64));
                local.advance(10);
                entry
                    .iter()
                    .map(|bins| {
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
                        (val, chosen)
                    })
                    .collect()
            })
            .collect();
        for &(val, chosen) in instances.iter().flatten() {
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

/// The split scores of one leaf's bucket statistics (`k` × buckets per
/// column), CatBoost's `CalcScoresForSubCandidate` with `AddLeafPlain`.
fn leaf_scores(
    q: &Quantized,
    candidates: &[ColumnRef],
    stats: &[Vec<BucketStats>],
    k: usize,
    l2: f64,
    score_function: CatBoostScoreFunction,
) -> Vec<Option<Vec<f64>>> {
    candidates
        .iter()
        .zip(stats)
        .map(|(c, st)| {
            let column = c.get(q);
            let nb = column.n_buckets;
            if nb < 2 {
                return None;
            }
            let one_hot = column.one_hot();
            let n_splits = if one_hot { nb } else { nb - 1 };
            let mut num = vec![0.0f64; n_splits];
            let mut den = vec![1e-100f64; n_splits];
            for d in 0..k {
                each_split(one_hot, &st[d * nb..(d + 1) * nb], n_splits, |s, t, f| {
                    let (at, af) = (average(t.swd, t.sw, l2), average(f.swd, f.sw, l2));
                    match score_function {
                        CatBoostScoreFunction::Cosine => {
                            num[s] += af * f.swd + at * t.swd;
                            den[s] += af * af * f.sw + at * at * t.sw;
                        }
                        CatBoostScoreFunction::L2 => {
                            num[s] += at * t.swd;
                            num[s] += af * f.swd;
                        }
                    }
                });
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
        .collect()
}

/// CatBoost's `SelectBestCandidate`: every candidate entry's noisy best
/// split (its own generator from `noise_seed`), each noised again from the
/// main one; the largest gain over `score_before`.
#[allow(clippy::too_many_arguments)]
fn select_best(
    entries: &[Vec<Vec<(usize, usize)>>],
    scores: &[Option<Vec<f64>>],
    noise_seed: u64,
    rng: &mut CatRng,
    score_std: f64,
    score_before: f64,
    penalty: &dyn Fn(usize) -> f64,
) -> Option<(f64, usize, usize)> {
    let mut best: Option<(f64, usize, usize)> = None;
    let mut best_gain = f64::NEG_INFINITY;
    for (t, entry) in entries.iter().enumerate() {
        let mut local = CatRng::new(noise_seed.wrapping_add(t as u64));
        local.advance(10);
        for bins in entry {
            let mut val = f64::MIN;
            let mut chosen: Option<(usize, usize)> = None;
            let mut best_instance = f64::MIN;
            for &(c, s) in bins {
                let Some(sc) = &scores[c] else {
                    continue;
                };
                let instance = sc[s] + local.std_normal() * score_std;
                if instance > best_instance {
                    best_instance = instance;
                    val = sc[s];
                    chosen = Some((c, s));
                }
            }
            let mut score = val + rng.std_normal() * score_std;
            let Some((c, s)) = chosen else {
                continue;
            };
            score *= penalty(c);
            let gain = score - score_before;
            if gain > best_gain {
                best_gain = gain;
                best = Some((score, c, s));
            }
        }
    }
    best
}

/// CatBoost's `GetCatFeatureWeight` for the candidates: a CTR type not yet
/// in the model scores `(1 + k / max k)^-model_size_reg` times less.
fn ctr_penalties(
    q: &Quantized,
    candidates: &[ColumnRef],
    used_ctrs: &std::collections::HashSet<(Projection, bool)>,
    model_size_reg: f64,
) -> Vec<f64> {
    let max = candidates
        .iter()
        .filter_map(|c| c.get(q).ctr_key.as_ref().map(|(_, k)| *k))
        .max()
        .unwrap_or(1)
        .max(1);
    candidates
        .iter()
        .map(|c| match &c.get(q).ctr_key {
            Some((key, k)) if !used_ctrs.contains(key) => {
                (1.0 + *k as f64 / max as f64).powf(-model_size_reg)
            }
            _ => 1.0,
        })
        .collect()
}

/// A non-symmetric tree's nodes (CatBoost's `TNonSymmetricTreeStructure`):
/// per node its split and children, a child `c < 0` being leaf `!c`.
type NodeChildren = Vec<(i32, i32)>;

/// CatBoost's `GreedyTensorSearchDepthwise` (plain boosting, no CTRs):
/// level by level, every leaf with at least `min_data` sampled rows is
/// split by its best candidate if that gains at least 1e-9. Of two
/// siblings the smaller one's bucket statistics are computed, the larger
/// one's are the parent's minus them (one approximation, from CatBoost's
/// FIFO of the parents' statistics). Splitting leaf `l` keeps `l` for the
/// false side and numbers the true side next.
#[allow(clippy::too_many_arguments)]
fn grow_depthwise(
    q: &Quantized,
    part: &ScorePart<'_>,
    k: usize,
    depth: usize,
    score_function: CatBoostScoreFunction,
    rng: &mut CatRng,
    score_std: f64,
    sample: &Sample,
    min_data: usize,
    used_ctrs: &mut std::collections::HashSet<(Projection, bool)>,
    model_size_reg: f64,
    object_weights: bool,
) -> (Vec<(ColumnRef, usize)>, NodeChildren) {
    let candidates: Vec<ColumnRef> = (0..q.columns.len()).map(ColumnRef::Base).collect();
    let level_draws = q.draws.0 + q.ctr.as_ref().map_or(0, |c| c.features.len());
    let entries = candidate_entries(q, &candidates);
    let n = sample.taken.len();
    let mut rows_of: Vec<Vec<usize>> = vec![(0..n).filter(|&i| sample.taken[i]).collect()];
    let mut splits: Vec<(ColumnRef, usize)> = Vec::new();
    let mut children: NodeChildren = Vec::new();
    let mut leaf_parent: Vec<i32> = vec![-1];
    let l2 = part.scaled_l2;
    let trick = k == 1;
    let stats_of = |rows: &[usize], w: &[f64]| -> Vec<Vec<BucketStats>> {
        candidates
            .iter()
            .map(|c| {
                let column = c.get(q);
                let nb = column.n_buckets;
                let mut st = vec![BucketStats::default(); k * nb];
                for &i in rows {
                    let b = column.buckets[i] as usize;
                    for d in 0..k {
                        let h = &mut st[d * nb + b];
                        h.swd += part.swder[i * k + d];
                        h.sw += w[i];
                    }
                }
                st
            })
            .collect()
    };
    // `CalcBestScoreAndCandidate`: (gain, best split) of a leaf's stats.
    let score = |stats: &[Vec<BucketStats>],
                 rows: &[usize],
                 w: &[f64],
                 rng: &mut CatRng,
                 used: &std::collections::HashSet<(Projection, bool)>| {
        let penalties = ctr_penalties(q, &candidates, used, model_size_reg);
        let noise_seed = rng.gen();
        let scores = leaf_scores(q, &candidates, stats, k, l2, score_function);
        // `CalcScoreWithoutSplit`: the leaf's sums (all dimensions).
        let (mut swd, mut sw) = (0.0, 0.0);
        for d in 0..k {
            for &i in rows {
                swd += part.swder[i * k + d];
            }
        }
        for &i in rows {
            sw += w[i];
        }
        let a = average(swd, sw, l2);
        let before = match score_function {
            CatBoostScoreFunction::Cosine => (a * swd) / (1e-100 + a * a * sw).sqrt(),
            CatBoostScoreFunction::L2 => a * swd,
        };
        let best = select_best(
            &entries,
            &scores,
            noise_seed,
            rng,
            score_std,
            before,
            &|c| penalties[c],
        );
        best.map_or((0.0, None), |(s, c, sp)| (s - before, Some((c, sp))))
    };
    let subtract = |parent: &[Vec<BucketStats>], small: &[Vec<BucketStats>]| {
        parent
            .iter()
            .zip(small)
            .map(|(p, s)| {
                p.iter()
                    .zip(s)
                    .map(|(p, s)| BucketStats {
                        swd: p.swd - s.swd,
                        sw: p.sw - s.sw,
                        sd: 0.0,
                        count: 0.0,
                    })
                    .collect()
            })
            .collect::<Vec<Vec<BucketStats>>>()
    };
    let mut parents: std::collections::VecDeque<Vec<Vec<BucketStats>>> = Default::default();
    let mut current: Vec<usize> = vec![0];
    let wts = sample.leafwise_weights(object_weights);
    for _ in 0..depth {
        for _ in 0..level_draws {
            rng.real1();
        }
        let mut next_level = Vec::new();
        let mut pending: Option<(f64, Option<(usize, usize)>)> = None;
        for id in 0..current.len() {
            let size = rows_of[current[id]].len();
            let next_size = current.get(id + 1).map_or(0, |&l| rows_of[l].len());
            let next_considered = next_size >= min_data;
            if size < min_data {
                continue;
            }
            let even = id % 2 == 0;
            let push = |parents: &mut std::collections::VecDeque<_>,
                        g: f64,
                        b: &Option<(usize, usize)>,
                        st| {
                if !(g < 1e-9) && b.is_some() {
                    parents.push_back(st);
                }
            };
            let (gain, best) = if let (false, Some(p)) = (even, pending.take()) {
                p
            } else if even && next_considered && trick && size <= next_size {
                let small = stats_of(&rows_of[current[id]], &wts);
                let (g, b) = score(&small, &rows_of[current[id]], &wts, rng, used_ctrs);
                let large = subtract(&parents[0], &small);
                let (g2, b2) = score(&large, &rows_of[current[id + 1]], &wts, rng, used_ctrs);
                parents.pop_front();
                push(&mut parents, g, &b, small);
                push(&mut parents, g2, &b2, large);
                pending = Some((g2, b2));
                (g, b)
            } else if even && next_considered && trick {
                let small = stats_of(&rows_of[current[id + 1]], &wts);
                let (g2, b2) = score(&small, &rows_of[current[id + 1]], &wts, rng, used_ctrs);
                let large = subtract(&parents[0], &small);
                let (g, b) = score(&large, &rows_of[current[id]], &wts, rng, used_ctrs);
                parents.pop_front();
                push(&mut parents, g, &b, large);
                push(&mut parents, g2, &b2, small);
                pending = Some((g2, b2));
                (g, b)
            } else {
                let st = stats_of(&rows_of[current[id]], &wts);
                let (g, b) = score(&st, &rows_of[current[id]], &wts, rng, used_ctrs);
                parents.pop_front();
                if trick {
                    push(&mut parents, g, &b, st);
                }
                pending = None;
                (g, b)
            };
            let Some((c, s)) = best else {
                continue;
            };
            if gain < 1e-9 {
                continue;
            }
            // `AddSplit`: the leaf keeps its id on the false side.
            let leaf = current[id];
            let new_leaf = children.len() + 1;
            let node = children.len() as i32;
            let parent = leaf_parent[leaf];
            if parent >= 0 {
                let ch = &mut children[parent as usize];
                if ch.0 == !(leaf as i32) {
                    ch.0 = node;
                } else {
                    ch.1 = node;
                }
            }
            children.push((!(leaf as i32), !(new_leaf as i32)));
            leaf_parent[leaf] = node;
            leaf_parent.push(node);
            splits.push((candidates[c].clone(), s));
            let column = candidates[c].get(q);
            if let Some((key, _)) = &column.ctr_key {
                used_ctrs.insert(key.clone());
            }
            let (right, left): (Vec<usize>, Vec<usize>) = rows_of[leaf]
                .iter()
                .partition(|&&i| column.goes_right(column.buckets[i], s));
            rows_of[leaf] = left;
            rows_of.push(right);
            next_level.push(leaf);
            next_level.push(new_leaf);
        }
        current = next_level;
    }
    (splits, children)
}

/// A leaf waiting to be split (CatBoost's `TSplitLeafCandidate`): its
/// gain, best split and statistics (an index into the caller's store).
#[derive(Clone, Copy)]
struct SplitCandidate {
    leaf: usize,
    gain: f64,
    column: usize,
    split: usize,
    stats: usize,
}

impl HeapKey for SplitCandidate {
    fn key(&self) -> f64 {
        self.gain
    }
}

/// CatBoost's `GreedyTensorSearchLossguide` (plain boosting, no CTRs): the
/// leaf of the largest gain is split next (libc++'s `priority_queue`)
/// until `max_leaves`; its children are scored if within `max_depth` with
/// at least `min_data` sampled rows (the smaller one directly, the other as
/// the parent's statistics minus it, for one approximation).
#[allow(clippy::too_many_arguments)]
fn grow_lossguide(
    q: &Quantized,
    part: &ScorePart<'_>,
    k: usize,
    max_depth: usize,
    max_leaves: usize,
    score_function: CatBoostScoreFunction,
    rng: &mut CatRng,
    score_std: f64,
    sample: &Sample,
    min_data: usize,
    used_ctrs: &mut std::collections::HashSet<(Projection, bool)>,
    model_size_reg: f64,
    object_weights: bool,
) -> (Vec<(ColumnRef, usize)>, NodeChildren) {
    let wts = sample.leafwise_weights(object_weights);
    let candidates: Vec<ColumnRef> = (0..q.columns.len()).map(ColumnRef::Base).collect();
    let level_draws = q.draws.0 + q.ctr.as_ref().map_or(0, |c| c.features.len());
    let entries = candidate_entries(q, &candidates);
    let n = sample.taken.len();
    let mut rows_of: Vec<Vec<usize>> = vec![(0..n).filter(|&i| sample.taken[i]).collect()];
    let mut depth_of = vec![0usize];
    let mut splits: Vec<(ColumnRef, usize)> = Vec::new();
    let mut children: NodeChildren = Vec::new();
    let mut leaf_parent: Vec<i32> = vec![-1];
    let l2 = part.scaled_l2;
    let trick = k == 1;
    let stats_of = |rows: &[usize]| -> Vec<Vec<BucketStats>> {
        candidates
            .iter()
            .map(|c| {
                let column = c.get(q);
                let nb = column.n_buckets;
                let mut st = vec![BucketStats::default(); k * nb];
                for &i in rows {
                    let b = column.buckets[i] as usize;
                    for d in 0..k {
                        let h = &mut st[d * nb + b];
                        h.swd += part.swder[i * k + d];
                        h.sw += wts[i];
                    }
                }
                st
            })
            .collect()
    };
    let score = |stats: &[Vec<BucketStats>],
                 rows: &[usize],
                 rng: &mut CatRng,
                 used: &std::collections::HashSet<(Projection, bool)>| {
        let penalties = ctr_penalties(q, &candidates, used, model_size_reg);
        let noise_seed = rng.gen();
        let scores = leaf_scores(q, &candidates, stats, k, l2, score_function);
        let (mut swd, mut sw) = (0.0, 0.0);
        for d in 0..k {
            for &i in rows {
                swd += part.swder[i * k + d];
            }
        }
        for &i in rows {
            sw += wts[i];
        }
        let a = average(swd, sw, l2);
        let before = match score_function {
            CatBoostScoreFunction::Cosine => (a * swd) / (1e-100 + a * a * sw).sqrt(),
            CatBoostScoreFunction::L2 => a * swd,
        };
        select_best(
            &entries,
            &scores,
            noise_seed,
            rng,
            score_std,
            before,
            &|c| penalties[c],
        )
        .map(|(s, c, sp)| (s - before, c, sp))
    };
    let subtract = |parent: &[Vec<BucketStats>], small: &[Vec<BucketStats>]| {
        parent
            .iter()
            .zip(small)
            .map(|(p, s)| {
                p.iter()
                    .zip(s)
                    .map(|(p, s)| BucketStats {
                        swd: p.swd - s.swd,
                        sw: p.sw - s.sw,
                        sd: 0.0,
                        count: 0.0,
                    })
                    .collect()
            })
            .collect::<Vec<Vec<BucketStats>>>()
    };
    let mut store: Vec<Vec<Vec<BucketStats>>> = Vec::new();
    let mut queue: StdPriorityQueue<SplitCandidate> = StdPriorityQueue(Vec::new());
    let offer = |queue: &mut StdPriorityQueue<SplitCandidate>,
                 store: &mut Vec<Vec<Vec<BucketStats>>>,
                 leaf: usize,
                 best: Option<(f64, usize, usize)>,
                 stats: Vec<Vec<BucketStats>>| {
        if let Some((gain, column, split)) = best {
            if gain >= 1e-9 {
                store.push(stats);
                queue.push(SplitCandidate {
                    leaf,
                    gain,
                    column,
                    split,
                    stats: store.len() - 1,
                });
            }
        }
    };
    if max_depth > 0 && rows_of[0].len() >= min_data {
        for _ in 0..level_draws {
            rng.real1();
        }
        let st = stats_of(&rows_of[0]);
        let best = score(&st, &rows_of[0], rng, used_ctrs);
        offer(&mut queue, &mut store, 0, best, st);
    }
    while !queue.0.is_empty() && children.len() + 1 < max_leaves {
        let cand = *queue.top();
        queue.pop();
        let leaf = cand.leaf;
        let new_leaf = children.len() + 1;
        let node = children.len() as i32;
        let parent = leaf_parent[leaf];
        if parent >= 0 {
            let ch = &mut children[parent as usize];
            if ch.0 == !(leaf as i32) {
                ch.0 = node;
            } else {
                ch.1 = node;
            }
        }
        children.push((!(leaf as i32), !(new_leaf as i32)));
        leaf_parent[leaf] = node;
        leaf_parent.push(node);
        splits.push((candidates[cand.column].clone(), cand.split));
        let column = candidates[cand.column].get(q);
        if let Some((key, _)) = &column.ctr_key {
            used_ctrs.insert(key.clone());
        }
        let (right, left): (Vec<usize>, Vec<usize>) = rows_of[leaf]
            .iter()
            .partition(|&&i| column.goes_right(column.buckets[i], cand.split));
        rows_of[leaf] = left;
        rows_of.push(right);
        let d = depth_of[leaf] + 1;
        depth_of[leaf] = d;
        depth_of.push(d);
        // `FindBestCandidate` of the two children.
        let (l, r) = (leaf, new_leaf);
        let (nl, nr) = (rows_of[l].len(), rows_of[r].len());
        let need_l = d < max_depth && nl >= min_data;
        let need_r = d < max_depth && nr >= min_data;
        if !need_l && !need_r {
            continue;
        }
        for _ in 0..level_draws {
            rng.real1();
        }
        let direct = !trick || need_l != need_r;
        let (l_direct, r_direct) = (direct || nl <= nr, direct || nl > nr);
        let (mut ls, mut rs) = (Vec::new(), Vec::new());
        let (mut lb, mut rb) = (None, None);
        if need_l && l_direct {
            ls = stats_of(&rows_of[l]);
            lb = score(&ls, &rows_of[l], rng, used_ctrs);
        }
        if need_r && r_direct {
            rs = stats_of(&rows_of[r]);
            rb = score(&rs, &rows_of[r], rng, used_ctrs);
        }
        if need_l && !l_direct {
            ls = subtract(&store[cand.stats], &rs);
            lb = score(&ls, &rows_of[l], rng, used_ctrs);
        }
        if need_r && !r_direct {
            rs = subtract(&store[cand.stats], &ls);
            rb = score(&rs, &rows_of[r], rng, used_ctrs);
        }
        if need_l {
            offer(&mut queue, &mut store, l, lb, ls);
        }
        if need_r {
            offer(&mut queue, &mut store, r, rb, rs);
        }
    }
    (splits, children)
}

/// CatBoost's body / tail prefixes of a fold for ordered boosting: the
/// first body is 1 row (`min(100, n / 50)` above 500 rows), every tail
/// ends at `ceil(body · fold_len_multiplier)`.
fn body_tails(
    n: usize,
    multiplier: f64,
    weights: &[f64],
    queries: &[usize],
) -> Vec<(usize, usize, f64)> {
    // CatBoost's `UpdateSize`: sizes extend to the end of their query.
    let snap = |size: usize| {
        let size = size.min(n);
        if queries.is_empty() {
            size
        } else {
            queries[queries.partition_point(|&b| b < size)]
        }
    };
    let mut left = snap(if n > 500 { 100.min(n / 50) } else { 1 });
    let mut out = Vec::new();
    while out.is_empty() || left < n {
        let body = left;
        let tail = snap((left as f64 * multiplier).ceil() as usize);
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
        if k == 1 && h == 1 {
            // One approximation: the same sums without the inner loops.
            for (((&leaf, &d), &d2), &w) in leaf_of[..rows]
                .iter()
                .zip(&ders.der)
                .zip(&ders.der2)
                .zip(&weights[..rows])
            {
                let leaf = leaf as usize;
                s.der[leaf] += d;
                s.der2[leaf] += d2;
                s.w[leaf] += w;
            }
            return s;
        }
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
        let h = self.der2.len() / self.w.len().max(1);
        let der = &self.der[leaf * k..(leaf + 1) * k];
        let scaled_l2 = l2 * (sum_w / count);
        match p.leaf_estimation {
            CatBoostLeafEstimation::Newton if k == 1 => {
                out[0] = der[0] / (-self.der2[leaf] + scaled_l2);
            }
            // A diagonal Hessian (`TDiagonalHessian::SolveNewtonEquation`).
            CatBoostLeafEstimation::Newton if h == k => {
                let der2 = &self.der2[leaf * h..(leaf + 1) * h];
                let l2 = f64::from((l2 * (sum_w / count)) as f32);
                for ((o, &d), &d2) in out.iter_mut().zip(der).zip(der2) {
                    *o = -d / (d2 - l2);
                }
            }
            CatBoostLeafEstimation::Newton => {
                let der2 = &self.der2[leaf * h..(leaf + 1) * h];
                newton_multi(der, der2, l2, sum_w, count, out);
            }
            CatBoostLeafEstimation::Gradient | CatBoostLeafEstimation::Exact => {
                for (o, &d) in out.iter_mut().zip(der) {
                    *o = average(d, self.w[leaf], scaled_l2);
                }
            }
        }
    }
}

/// CatBoost's `CalcSampleQuantile`: below 100 values the first value of
/// the stable sort whose cumulative weight reaches `alpha` of the total;
/// else 100 bisection steps (with libc++'s `std::partition` of the rows
/// still undecided), returning the upper end.
fn sample_quantile(values: &[f32], weights: &[f32], alpha: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    if alpha <= 0.0 {
        return f64::from(values.iter().copied().fold(f32::INFINITY, f32::min));
    }
    let total: f64 = weights.iter().fold(0.0, |s, &w| s + f64::from(w));
    let need = total * alpha;
    let mut elements: Vec<(f32, f32)> = values
        .iter()
        .copied()
        .zip(weights.iter().copied())
        .collect();
    if values.len() < 100 {
        elements.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut sum = 0.0;
        for &(v, w) in &elements {
            sum += f64::from(w);
            if sum >= need - f64::EPSILON {
                return f64::from(v);
            }
        }
        return f64::from(elements.last().unwrap().0);
    }
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let (mut lq, mut rq) = (f64::from(min) - f64::EPSILON, f64::from(max));
    let (mut l, mut r) = (0usize, elements.len());
    let mut collected = 0.0;
    for _ in 0..100 {
        let q = (lq + rq) / 2.0;
        let point = l + libcxx_partition(&mut elements[l..r], |e: (f32, f32)| f64::from(e.0) <= q);
        let left: f64 = elements[l..point]
            .iter()
            .fold(0.0, |s, e| s + f64::from(e.1));
        if collected + left < need - f64::EPSILON {
            l = point;
            lq = q;
            collected += left;
        } else {
            r = point;
            rq = q;
        }
    }
    rq
}

/// CatBoost's `CalculateWeightedTargetQuantile`: the sample quantile moved
/// by `delta` towards the side that holds less than `alpha` of the weight,
/// as float32.
fn weighted_quantile(values: &[f32], weights: &[f32], alpha: f64, delta: f64) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut q = sample_quantile(values, weights, alpha);
    if delta > 0.0 {
        let total: f64 = weights.iter().fold(0.0, |s, &w| s + f64::from(w));
        let need = total * alpha;
        let (mut less, mut equal) = (0.0f64, 0.0f64);
        for (&v, &w) in values.iter().zip(weights) {
            let v = f64::from(v);
            if v < q {
                less += f64::from(w);
            } else if v == q {
                equal += f64::from(w);
            }
        }
        if less + equal * alpha >= need - f64::EPSILON {
            q -= delta;
        } else {
            q += delta;
        }
    }
    q as f32
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
    /// The tree's monotone leaf orders (empty: unconstrained).
    orders: &'a [Vec<usize>],
    /// The weights of Exact leaves (CatBoost's `SampleWeights`).
    exact_weights: &'a [f64],
    /// Query groups (querywise losses; else empty).
    groups: &'a Groups,
}

/// CatBoost's `CalculateWeightedTargetVariance` (float32 squares).
fn weighted_target_variance(target: &[f64], weights: &[f64], mean: f32) -> f32 {
    let total: f64 = weights.iter().sum();
    let sum: f64 = target
        .iter()
        .zip(weights)
        .map(|(&t, &w)| {
            let d = t as f32 - mean;
            f64::from(d * d * w as f32)
        })
        .sum();
    (sum / total) as f32
}

/// fmath's table of `2^(i / 2048)` mantissas (`ExpdVar`).
fn expd_table() -> &'static [u64; 2048] {
    static TABLE: std::sync::OnceLock<[u64; 2048]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = [0u64; 2048];
        for (i, v) in t.iter_mut().enumerate() {
            *v = 2f64.powf(i as f64 * (1.0 / 2048.0)).to_bits() & ((1u64 << 52) - 1);
        }
        t
    })
}

/// fmath's `expd` arithmetic on a clamped `x`.
#[inline]
fn expd_core(x: f64) -> f64 {
    const B: f64 = (3u64 << 51) as f64;
    let a = 2048.0 / 2f64.ln();
    let ra = 1.0 / a;
    let adj: u64 = (1 << (11 + 10)) - (1 << 11);
    let d = x * a + B;
    let bits = d.to_bits();
    let iax = expd_table()[(bits & 2047) as usize];
    let t = (d - B) * ra - x;
    let u = ((bits.wrapping_add(adj) >> 11) << 52) | iax;
    let y = (3.0000000027955394 - t) * (t * t) * 0.16666666685227835064 - t + 1.0;
    y * f64::from_bits(u)
}

/// CatBoost's `FastExpWithInfInplace` with AVX2: fmath's `expd_v`, which
/// clamps blocks of four values and computes the rest with `expd` (0 and
/// inf outside its range).
fn fast_exp_with_inf(x: &mut [f64]) {
    let blocks = x.len() / 4 * 4;
    let (head, tail) = x.split_at_mut(blocks);
    for v in head {
        *v = expd_core(v.clamp(-708.39641853226408, 709.78272569338397));
    }
    for v in tail {
        *v = if *v <= -708.39641853226408 {
            0.0
        } else if *v >= 709.78271289338397 {
            f64::INFINITY
        } else {
            expd_core(*v)
        };
    }
}

/// CatBoost's `CalculateOptimalConstApproxForLogCosh`: the root of
/// `Σ w tanh(a - r)` by bisection between the smallest and largest
/// residual (100 halvings or a width of 1e-9), as float32.
fn logcosh_optimum(residuals: &[f32], weights: &[f32]) -> f32 {
    let Some(first) = residuals.first() else {
        return 0.0;
    };
    let (mut left, mut right) = residuals
        .iter()
        .fold((f64::from(*first), f64::from(*first)), |(lo, hi), &r| {
            (lo.min(f64::from(r)), hi.max(f64::from(r)))
        });
    for _ in 0..100 {
        if right - left <= 1e-9 {
            break;
        }
        let m = (left + right) / 2.0;
        let value: f64 = residuals
            .iter()
            .zip(weights)
            .map(|(&r, &w)| (m - f64::from(r)).tanh() * f64::from(w))
            .sum();
        if value > 0.0 {
            right = m;
        } else {
            left = m;
        }
    }
    left as f32
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
    if p.leaf_estimation == CatBoostLeafEstimation::Exact {
        // CatBoost's `CalcExactLeafDeltas`: the loss-optimal constant of
        // each leaf's float32 residuals (one step, no generator draw).
        let mut residuals = vec![Vec::new(); n_leaves];
        let mut weights = vec![Vec::new(); n_leaves];
        for i in 0..rows {
            let leaf = lp.leaf_of[i] as usize;
            residuals[leaf].push((lp.target[i] - lp.approx[i]) as f32);
            weights[leaf].push(lp.exact_weights[i] as f32);
        }
        let step: Vec<f64> = residuals
            .iter()
            .zip(&weights)
            .map(|(r, w)| match lp.loss {
                Loss::Quantile { alpha, delta, .. } => {
                    f64::from(weighted_quantile(r, w, alpha, delta))
                }
                Loss::LogCosh => f64::from(logcosh_optimum(r, w)),
                _ => unreachable!("Exact leaves are validated to Quantile, MAE and LogCosh"),
            })
            .collect();
        return (step, 0);
    }
    let mut steps = 0;
    let backtracking = p.backtracking && p.leaf_iterations > 1;
    // Logloss: the approximations' exponentials and logarithms, cached for
    // the base and the current point (the metric is evaluated many times).
    let logloss = lp.loss == Loss::Logloss;
    let base_exp = if logloss && !lp.moving {
        exp_point(&lp.approx[..rows])
    } else {
        Vec::new()
    };
    let mut point_exp: Vec<(f64, f32)> = Vec::new();
    let metric = |approx: &[f64], leaves: &[f64], cached: &[(f64, f32)]| {
        if logloss {
            logloss_metric_exp_at(lp.target, cached, leaves, lp.leaf_of, lp.weights)
        } else {
            lp.loss.metric_with_leaves(
                lp.target, approx, leaves, lp.leaf_of, lp.weights, rows, lp.groups,
            )
        }
    };
    let mut total = vec![0.0; n_leaves * k];
    let mut point = lp.approx[..rows * k].to_vec();
    if logloss && backtracking && lp.moving {
        point_exp = exp_point(&point);
    }
    let mut value = if backtracking {
        let cached = if lp.moving { &point_exp } else { &base_exp };
        metric(&point, &total, cached)
    } else {
        0.0
    };
    let mut it = 0;
    while it < p.leaf_iterations {
        steps += 1;
        let ders = if logloss && !point_exp.is_empty() {
            logloss_ders_at(lp.target, &point_exp, lp.weights)
        } else {
            lp.loss
                .all_ders(lp.target, &point, lp.weights, rows, lp.groups)
        };
        let mut sums = LeafSums::of(lp.leaf_of, n_leaves, &ders, lp.weights, rows);
        let mut step = vec![0.0; n_leaves * k];
        if lp.loss.pairwise_scoring() {
            // CatBoost's `CalculatePairwiseLeafValues` of the leaves' pair
            // weights and derivative sums.
            let weights = pairwise_weight_sums(lp.groups, lp.leaf_of, n_leaves, rows);
            step = pairwise_leaf_values(&weights, &sums.der, p.l2_leaf_reg, p.pairwise_prior);
        } else if lp.orders.is_empty() {
            for (leaf, o) in step.chunks_mut(k).enumerate() {
                sums.delta(leaf, p.l2_leaf_reg, lp.stats.0, lp.stats.1, p, o);
            }
        } else {
            // CatBoost's `CalcMonotonicLeafDeltasSimple`: the leaf values
            // after the step (the model's: from the summed steps; a fold's:
            // from 0) made monotone along the orders, weighted by the
            // steps' denominators.
            let scaled_l2 = p.l2_leaf_reg * (lp.stats.0 / lp.stats.1);
            let weights: Vec<f64> = (0..n_leaves)
                .map(|leaf| match p.leaf_estimation {
                    CatBoostLeafEstimation::Gradient | CatBoostLeafEstimation::Exact => {
                        sums.w[leaf] + scaled_l2
                    }
                    CatBoostLeafEstimation::Newton => -sums.der2[leaf] + scaled_l2,
                })
                .collect();
            let current = if lp.moving {
                total.clone()
            } else {
                vec![0.0; n_leaves]
            };
            let mut values: Vec<f64> = (0..n_leaves)
                .map(|leaf| current[leaf] + sums.der[leaf] / weights[leaf])
                .collect();
            for order in lp.orders {
                isotonic(&mut values, &weights, order);
            }
            for leaf in 0..n_leaves {
                step[leaf] = values[leaf] - current[leaf];
            }
        }
        let take = |scaled: &[f64], total: &mut [f64], point: &mut [f64]| {
            use rayon::prelude::*;
            for (t, s) in total.iter_mut().zip(scaled) {
                *t += s;
            }
            point[..rows * k]
                .par_chunks_mut(k)
                .with_min_len(PARALLEL_ROWS)
                .zip(&lp.leaf_of[..rows])
                .for_each(|(row, &leaf)| {
                    let leaf = leaf as usize;
                    for (v, s) in row.iter_mut().zip(&scaled[leaf * k..(leaf + 1) * k]) {
                        *v += s;
                    }
                });
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
                metric(&point, &scaled, &point_exp)
            } else {
                let candidate: Vec<f64> = total.iter().zip(&scaled).map(|(t, s)| t + s).collect();
                metric(lp.approx, &candidate, &base_exp)
            };
            if after < value {
                value = after;
                take(&scaled, &mut total, &mut point);
                if logloss && lp.moving {
                    point_exp = exp_point(&point);
                }
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
    /// Query groups in the fold's order (querywise losses; else empty).
    groups: Groups,
    target: Vec<f64>,
    weights: Vec<f64>,
    tails: Vec<(usize, usize, f64)>,
    approx: Vec<Vec<f64>>,
    cache: CombinationCache,
    /// CatBoost's `TFold::SampleWeights`: the bootstrap times the sample
    /// weights of the fold's last turn as the taken fold (1 before), what
    /// its Exact leaves are weighted with.
    sample_weights: Vec<f64>,
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

    /// Every row's leaf: oblivious `splits`, or the nodes `children`.
    fn tree_leaves(
        &self,
        splits: &[(ColumnRef, usize)],
        children: &[(i32, i32)],
        cache: &mut CombinationCache,
    ) -> Vec<u32> {
        if children.is_empty() {
            return self.leaves(splits, cache);
        }
        let n = self.columns.first().map_or(0, |c| c.buckets.len());
        (0..n)
            .map(|i| {
                let mut node = 0i32;
                loop {
                    let (c, s) = &splits[node as usize];
                    let column = c.get(self);
                    let (l, r) = children[node as usize];
                    node = if column.goes_right(column.buckets[i], *s) {
                        r
                    } else {
                        l
                    };
                    if node < 0 {
                        return !node as u32;
                    }
                }
            })
            .collect()
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

impl Sample {
    /// The weights a non-symmetric tree is scored with. With
    /// `object_weights` (a pairwise loss, whose `weights` are pairwise),
    /// CatBoost's leafwise scoring takes the rows' sample weights instead:
    /// the bootstrap weight (no sample weights).
    fn leafwise_weights(&self, object_weights: bool) -> Vec<f64> {
        if !object_weights {
            return self.weights.clone();
        }
        self.boot
            .iter()
            .zip(&self.taken)
            .map(|(&b, &t)| if t { f64::from(b as f32) } else { 0.0 })
            .collect()
    }
}

/// libc++'s `std::partition` (the bidirectional algorithm): the elements
/// with `pred` first; returns their count.
fn libcxx_partition<T: Copy>(v: &mut [T], pred: impl Fn(T) -> bool) -> usize {
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
                // The blocks draw from generators of their own, in
                // parallel.
                use rayon::prelude::*;
                let seed = rng.gen();
                boot.par_chunks_mut(8192)
                    .zip(taken.par_chunks_mut(8192))
                    .enumerate()
                    .for_each(|(block, (boot, taken))| {
                        let start = block * 8192;
                        let end = start + boot.len();
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
                        for (j, i) in (start..end).enumerate() {
                            let g = (row(i) + lambda).sqrt();
                            let prob = if g > threshold { 1.0 } else { g / threshold };
                            boot[j] = if prob > f64::EPSILON {
                                let r = local.real1();
                                f64::from(((1.0 / prob) * f64::from(u8::from(r < prob))) as f32)
                            } else {
                                0.0
                            };
                            taken[j] = (boot[j] as f32) > f32::EPSILON;
                        }
                    });
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
    orders: &[Vec<usize>],
    seed: u64,
) {
    let k = loss.dims();
    if let Some((permutations, decay)) = loss.yeti_rank() {
        // `CalcApproxDeltaSimple`: pairs drawn at the fold's scores.
        let seed = seed_vector(fold.tails.len(), seed)[0];
        let n = fold.weights.len();
        fold.groups = yeti_rank_groups(
            &fold.groups.queries,
            &fold.target,
            &fold.approx[0],
            n,
            seed,
            permutations,
            decay,
        );
        fold.weights = fold.groups.pair_weights(n);
    }
    let (target, weights) = (&fold.target, &fold.weights);
    let exact_weights = &fold.sample_weights;
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
            orders,
            exact_weights,
            groups: &fold.groups,
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
                    let t = loss.target_dims();
                    loss.ders(
                        &target[i * t..(i + 1) * t],
                        &point,
                        weights[i],
                        &mut der,
                        &mut der2,
                    );
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
/// A validation set (CatBoost's `eval_set`): raw features, targets (class
/// indices for classifiers) and weights, as float32 values.
struct EvalSet<'a> {
    x: ArrayView2<'a, f64>,
    target: Vec<f64>,
    weights: Vec<f64>,
    /// Query groups (querywise losses; else empty).
    groups: Groups,
    /// YetiRank's eval metric is PFound (targets in [0, 1]), else NDCG.
    pfound: bool,
}

/// An eval set for `fit_with_eval` (CatBoost's `eval_set`): features,
/// targets and optional weights.
#[derive(Debug, Clone, Copy)]
pub struct CatBoostEvalSet<'a> {
    pub x: ArrayView2<'a, f64>,
    pub y: ArrayView1<'a, f64>,
    pub sample_weight: Option<ArrayView1<'a, f64>>,
}

/// An eval set for [`CatBoostRegressor::fit_multi_with_eval`]: features,
/// one target column per MultiRMSE target and optional weights.
#[derive(Debug, Clone, Copy)]
pub struct CatBoostMultiEvalSet<'a> {
    pub x: ArrayView2<'a, f64>,
    pub y: ArrayView2<'a, f64>,
    pub sample_weight: Option<ArrayView1<'a, f64>>,
}

/// What [`boost`] returns.
struct Boosted {
    trees: Vec<ObliviousTree>,
    ctrs: CtrModel,
    bias: Vec<f64>,
    /// The eval set's metric after every iteration.
    eval_metrics: Vec<f64>,
    /// The iteration of the best eval metric (0-based).
    best_iteration: Option<usize>,
}

/// CatBoost's `TErrorTracker` with its `IncToDec` overfitting detector
/// (`Iter` is `IncToDec` with threshold 1): the first strictly smallest
/// error, and a stop once the p-value falls below the threshold.
struct ErrorTracker {
    threshold: f64,
    wait: usize,
    best: f64,
    best_iteration: Option<usize>,
    empty: bool,
    local_max: f64,
    expected_inc: f64,
    from_max: usize,
    errors: std::collections::VecDeque<f64>,
}

impl ErrorTracker {
    fn new(threshold: f64, wait: usize) -> Self {
        Self {
            threshold,
            wait,
            best: f64::INFINITY,
            best_iteration: None,
            empty: true,
            local_max: 0.0,
            expected_inc: 0.0,
            from_max: 0,
            errors: std::collections::VecDeque::new(),
        }
    }

    /// Record the error of `iteration`; whether to stop.
    fn add(&mut self, error: f64, iteration: usize) -> bool {
        if error < self.best {
            self.best = error;
            self.best_iteration = Some(iteration);
        }
        if self.threshold <= 0.0 {
            return false;
        }
        // The detector maximizes.
        let e = -error;
        if self.empty || e > self.local_max {
            if self.empty {
                self.empty = false;
                self.expected_inc = 0.0;
            }
            self.local_max = e;
            self.from_max = 0;
        } else {
            self.from_max += 1;
        }
        self.errors.push_front(e);
        if self.errors.len() > 2000 {
            self.errors.pop_back();
        }
        self.expected_inc *= 0.99;
        let mut mult = 1.0;
        for &x in &self.errors {
            self.expected_inc = self.expected_inc.max(mult * (e - x));
            mult *= 0.99;
        }
        let p_value = if self.from_max >= self.wait {
            let ratio = self.expected_inc / (self.local_max - e).max(1e-10);
            (-0.5 / ratio.max(1e-10)).exp()
        } else {
            1.0
        };
        !self.empty && p_value < self.threshold
    }
}

#[allow(clippy::too_many_arguments)]
fn boost(
    q: &Quantized,
    target: &[f64],
    weights: &[f64],
    bias: &[f64],
    loss: Loss,
    p: TrainParams,
    mut rng: CatRng,
    eval: Option<&EvalSet<'_>>,
    groups: &Groups,
) -> Boosted {
    let n = weights.len();
    let k = loss.dims();
    let has_ctrs = q.ctr.is_some();
    let learn_permuted = !p.has_time && (has_ctrs || p.ordered);
    let average_permuted = !p.has_time && has_ctrs;
    let n_folds = if learn_permuted {
        p.permutation_count.saturating_sub(1).max(1)
    } else {
        1
    };
    let block = if !groups.queries.is_empty() {
        1
    } else if learn_permuted {
        (n / 1000 + 1).min(256)
    } else {
        n
    };
    let start = |rows: usize| -> Vec<f64> { bias.iter().copied().cycle().take(rows * k).collect() };
    let identity: Vec<usize> = (0..n).collect();
    let make_fold = |perm: &[usize], groups: Groups| -> LearnFold {
        let target = permute_rows(target, perm, loss.target_dims());
        let mut weights = permute_rows(weights, perm, 1);
        let sum_w: f64 = weights.iter().sum();
        let tails = if p.ordered {
            body_tails(n, p.fold_len_multiplier, &weights, &groups.queries)
        } else {
            vec![(n, n, sum_w)]
        };
        // Pairwise losses weigh leaves and scores by the rows' pairs
        // (`PairwiseWeights`); the L2 scaling keeps the sample weights.
        if loss.pairwise() {
            weights = groups.pair_weights(n);
        }
        LearnFold {
            q: q.permuted(perm),
            groups,
            approx: tails.iter().map(|&(_, t, _)| start(t)).collect(),
            target,
            weights,
            tails,
            cache: CombinationCache::new(),
            sample_weights: vec![1.0; n],
        }
    };
    let mut folds: Vec<LearnFold> = (0..n_folds)
        .map(|f| {
            let (perm, bounds) = if f > 0 && learn_permuted {
                rng.fold_permutation(n, block, &groups.queries)
            } else {
                (identity.clone(), groups.queries.clone())
            };
            make_fold(&perm, groups.permuted(&perm, bounds))
        })
        .collect();
    let (average_perm, average_queries) = if average_permuted {
        rng.fold_permutation(n, block, &groups.queries)
    } else {
        (identity.clone(), groups.queries.clone())
    };
    let mut average = make_fold(
        &average_perm,
        groups.permuted(&average_perm, average_queries),
    );
    let avg_sum_w: f64 = weights.iter().sum();
    let mut approx = start(n);
    // Plain boosting with an unpermuted averaging fold shares its
    // approximations with fold 0 (`UseAveragingFoldAsFoldZero`).
    // (`UseAveragingFoldAsFoldZero`: not for pairwise losses.)
    let shared = !p.ordered && !average_permuted && !loss.pairwise();
    let mut trees = Vec::with_capacity(p.iterations);
    let mut used_ctrs = std::collections::HashSet::new();
    let mut ctrs = CtrModel::default();
    // CatBoost's model shrinkage (float32 rate and learning rate): every
    // learning approximation is scaled before each iteration after the
    // first (`ScaleAllApproxes`); the bias then too, the trees at the end.
    let shrink_rate = f64::from(p.shrink_rate as f32);
    let mut shrink_history: Vec<f64> = Vec::new();
    let mut bias = bias.to_vec();
    let unit_weights = vec![1.0; n];
    // CatBoost's `TestApprox`: the bias plus the trees so far.
    let mut eval_approx: Vec<f64> = eval.map_or_else(Vec::new, |e| {
        bias.iter().copied().cycle().take(e.x.nrows() * k).collect()
    });
    let mut eval_metrics = Vec::new();
    let mut tracker = ErrorTracker::new(p.od_threshold, p.od_wait);
    for iteration in 0..p.iterations {
        if shrink_rate > 0.0 {
            let factor = if iteration == 0 {
                1.0
            } else {
                match p.shrink_mode {
                    CatBoostShrinkMode::Constant => 1.0 - shrink_rate * p.learning_rate,
                    CatBoostShrinkMode::Decreasing => 1.0 - shrink_rate / iteration as f64,
                }
            };
            if iteration > 0 {
                for a in folds
                    .iter_mut()
                    .flat_map(|f| f.approx.iter_mut())
                    .chain(std::iter::once(&mut average.approx[0]))
                    .chain(std::iter::once(&mut approx))
                {
                    // Stored `exp(a)` (Logloss) moves as by a leaf value.
                    for v in a.iter_mut() {
                        *v = loss.fold_step(*v, factor);
                    }
                }
                for b in &mut bias {
                    *b *= factor;
                }
                for v in &mut eval_approx {
                    *v *= factor;
                }
            }
            shrink_history.push(factor);
        }
        let taken = (rng.gen() % n_folds as u64) as usize;
        let der_seed = rng.gen(); // the derivatives' seeds
        let fold = &mut folds[taken];
        if let Some((permutations, decay)) = loss.yeti_rank() {
            // `CalcWeightedDerivatives`: pairs drawn at the fold's scores
            // (one body / tail: no ordered boosting).
            let seed = seed_vector(fold.tails.len(), der_seed)[0];
            fold.groups = yeti_rank_groups(
                &fold.groups.queries,
                &fold.target,
                &fold.approx[0],
                n,
                seed,
                permutations,
                decay,
            );
            fold.weights = fold.groups.pair_weights(n);
        }
        let ders: Vec<Ders> = fold
            .tails
            .iter()
            .zip(&fold.approx)
            .map(|(&(_, t, _), a)| loss.all_ders(&fold.target, a, &fold.weights, t, &fold.groups))
            .collect();
        let previous = trees
            .last()
            .map(|t: &ObliviousTree| t.leaf_values.as_slice());
        // Pairwise scoring samples the pairs (all rows taken).
        let sample = if loss.pairwise_scoring() {
            fold.groups
                .bootstrap_pairs(&mut rng, p.bootstrap, p.subsample, p.bagging_temperature);
            Sample {
                boot: vec![1.0; n],
                weights: vec![1.0; n],
                taken: vec![true; n],
            }
        } else {
            bootstrap(&mut rng, &ders, &fold.tails, &fold.weights, previous, k, &p)
        };
        fold.sample_weights.clone_from(&sample.weights);
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
        let std = score_std(&ders, &fold.tails, p.ordered, n, iteration, &p);
        let (splits, children) = if p.grow_policy == CatBoostGrowPolicy::Lossguide {
            grow_lossguide(
                &fold.q,
                &parts[0],
                k,
                p.depth,
                p.max_leaves,
                p.score_function,
                &mut rng,
                std,
                &sample,
                p.min_data_in_leaf,
                &mut used_ctrs,
                p.model_size_reg,
                loss.pairwise(),
            )
        } else if p.grow_policy == CatBoostGrowPolicy::Depthwise {
            grow_depthwise(
                &fold.q,
                &parts[0],
                k,
                p.depth,
                p.score_function,
                &mut rng,
                std,
                &sample,
                p.min_data_in_leaf,
                &mut used_ctrs,
                p.model_size_reg,
                loss.pairwise(),
            )
        } else {
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
                std,
                &sample,
                &p.monotone,
                loss.pairwise_scoring().then(|| PairwiseScoring {
                    pairs: fold.groups.flat_pairs(n),
                    l2: p.l2_leaf_reg,
                    prior: p.pairwise_prior,
                }),
            );
            (splits, Vec::new())
        };
        let fold_seeds = seed_vector(n_folds, rng.gen()); // the folds' seeds
        let n_leaves = if p.grow_policy == CatBoostGrowPolicy::SymmetricTree {
            1usize << splits.len()
        } else {
            children.len() + 1
        };
        let tree_monotone: Vec<i8> = splits
            .iter()
            .map(|(c, _)| c.get(&fold.q).monotone(&p.monotone))
            .collect();
        let orders = if tree_monotone.iter().any(|&m| m != 0) {
            monotone_orders(&tree_monotone)
        } else {
            Vec::new()
        };

        // The model's leaves, from the averaging fold.
        let leaf_of = average
            .q
            .tree_leaves(&splits, &children, &mut average.cache);
        let mut seeded_steps = 0;
        if let Some((permutations, decay)) = loss.yeti_rank() {
            // `CalcLeafValuesSimple`'s seed of its one step.
            seeded_steps = 1;
            average.groups = yeti_rank_groups(
                &average.groups.queries,
                &average.target,
                &approx,
                n,
                rng.gen(),
                permutations,
                decay,
            );
            average.weights = average.groups.pair_weights(n);
        }
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
            orders: &orders,
            // The averaging fold is never bootstrapped.
            exact_weights: &unit_weights,
            groups: &average.groups,
        };
        let (leaves, steps) = estimate_leaves(&lp, &p, |_, _| {});
        if k == 1 {
            // `CalcLeafValuesSimple` draws a seed per step.
            for _ in seeded_steps..steps {
                rng.gen();
            }
        }
        let mut leaves = leaves;
        if loss.pairwise() {
            // `NormalizeLeafValues`: a pairwise model's leaves have weighted
            // mean 0 (the sample weights of the averaging fold's rows).
            let mut leaf_w = vec![0.0; n_leaves];
            for (i, &leaf) in leaf_of.iter().enumerate() {
                leaf_w[leaf as usize] += weights[average_perm[i]];
            }
            let mean = leaves.iter().zip(&leaf_w).map(|(v, w)| v * w).sum::<f64>()
                / leaf_w.iter().sum::<f64>();
            for (v, &w) in leaves.iter_mut().zip(&leaf_w) {
                *v = if w.abs() > 1e-9 { *v - mean } else { 0.0 };
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
            let leaf_of = fold.q.tree_leaves(&splits, &children, &mut fold.cache);
            update_fold(fold, &leaf_of, n_leaves, loss, &p, &orders, fold_seeds[f]);
        }
        let source = &folds[taken];
        let tree_splits = splits
            .iter()
            .map(|(c, s)| c.get(&source.q).tree_split(*s, &mut ctrs))
            .collect();
        let tree = ObliviousTree {
            splits: tree_splits,
            leaf_values,
            children,
        };
        if let Some(e) = eval {
            for (i, row) in e.x.rows().into_iter().enumerate() {
                let leaf = tree.leaf(row, &ctrs);
                for d in 0..k {
                    eval_approx[i * k + d] += tree.leaf_values[leaf * k + d];
                }
            }
            // YetiRank's PFound / NDCG are maximized.
            let (metric, tracked) = if matches!(loss, Loss::YetiRank { .. }) {
                let m = ranking_metric(&e.target, &eval_approx, &e.groups.queries, e.pfound);
                (m, -m)
            } else {
                let m = loss.eval_metric(&e.target, &eval_approx, &e.weights, &e.groups);
                (m, m)
            };
            eval_metrics.push(metric);
            trees.push(tree);
            if tracker.add(tracked, iteration) {
                break;
            }
            continue;
        }
        trees.push(tree);
    }
    // `use_best_model`: the model up to the best iteration (the bias keeps
    // the shrinkage of every iteration, as CatBoost's does).
    let best_iteration = tracker.best_iteration;
    if p.use_best_model {
        if let Some(b) = best_iteration {
            trees.truncate(b + 1);
            shrink_history.truncate(b + 1);
        }
    }
    let mut accumulated = 1.0;
    for (tree, &factor) in trees.iter_mut().zip(&shrink_history).rev() {
        for v in &mut tree.leaf_values {
            *v *= accumulated;
        }
        accumulated *= factor;
    }
    Boosted {
        trees,
        ctrs,
        bias,
        eval_metrics,
        best_iteration,
    }
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

/// LAPACK's `dposv` with `uplo = 'U'` (reference `dpotf2` Cholesky, below
/// its block size of 64, and `dpotrs` through `dtrsm`), which CatBoost's
/// `SolveLinearSystemCholesky` calls: solves `A x = b` in place for the
/// `n × n` column-major `A` (upper triangle). A matrix that is not
/// positive definite leaves `b` unchanged, as `dposv` does.
fn dposv_upper(a: &mut [f64], b: &mut [f64]) {
    let n = b.len();
    if n == 1 {
        b[0] /= a[0];
        return;
    }
    // Reference `ddot`: unrolled by 5 after the leftover terms.
    let ddot = |x: &[f64], y: &[f64]| -> f64 {
        let m = x.len() % 5;
        let mut t = 0.0;
        for i in 0..m {
            t += x[i] * y[i];
        }
        let mut i = m;
        while i < x.len() {
            t = t
                + x[i] * y[i]
                + x[i + 1] * y[i + 1]
                + x[i + 2] * y[i + 2]
                + x[i + 3] * y[i + 3]
                + x[i + 4] * y[i + 4];
            i += 5;
        }
        t
    };
    let at = |r: usize, c: usize| c * n + r;
    for j in 0..n {
        let col: Vec<f64> = (0..j).map(|r| a[at(r, j)]).collect();
        let ajj = a[at(j, j)] - ddot(&col, &col);
        if ajj <= 0.0 || ajj.is_nan() {
            return;
        }
        let ajj = ajj.sqrt();
        a[at(j, j)] = ajj;
        if j + 1 < n {
            // dgemv('T'): a(j, c) -= a(0..j, c) · a(0..j, j).
            for c in j + 1..n {
                let mut temp = 0.0;
                for r in 0..j {
                    temp += a[at(r, c)] * col[r];
                }
                a[at(j, c)] += -temp;
            }
            let r = 1.0 / ajj;
            for c in j + 1..n {
                a[at(j, c)] *= r;
            }
        }
    }
    // dtrsm: Uᵀ y = b, then U x = y.
    for i in 0..n {
        let mut temp = b[i];
        for k in 0..i {
            temp -= a[at(k, i)] * b[k];
        }
        b[i] = temp / a[at(i, i)];
    }
    for k in (0..n).rev() {
        if b[k] != 0.0 {
            b[k] /= a[at(k, k)];
            for i in 0..k {
                b[i] -= b[k] * a[at(i, k)];
            }
        }
    }
}

/// CatBoost's `CalculatePairwiseLeafValues`: the leaf values of the
/// pairwise system `weights` (`L × L`, row-major) and `der`, with the
/// prior spread over the cells, the last leaf fixed at 0, then centred.
fn pairwise_leaf_values(weights: &[f64], der: &[f64], l2: f64, prior: f64) -> Vec<f64> {
    let size = der.len();
    let cell_prior = 1.0 / size as f64;
    let non_diag = -prior * cell_prior;
    let diag = prior * (1.0 - cell_prior) + l2;
    let mut res = if size == 2 {
        vec![der[0] / (weights[0] + diag), 0.0]
    } else {
        let m = size - 1;
        let mut matrix = vec![0.0; m * m];
        for y in 0..m {
            for x in 0..y {
                matrix[y * m + x] = weights[y * size + x] + non_diag;
            }
            matrix[y * m + y] = weights[y * size + y] + diag;
        }
        let mut res = der[..m].to_vec();
        dposv_upper(&mut matrix, &mut res);
        res.push(0.0);
        res
    };
    let mean = res.iter().sum::<f64>() / size as f64;
    res.iter_mut().for_each(|v| *v -= mean);
    res
}

/// CatBoost's `ComputePairwiseWeightSums` of the queries of the rows
/// `0..rows` (pair weights, blocks of `⌈queries / 128⌉` queries merged in
/// order): `L × L`, row-major.
fn pairwise_weight_sums(
    groups: &Groups,
    leaf_of: &[u32],
    n_leaves: usize,
    rows: usize,
) -> Vec<f64> {
    let queries = query_prefix(&groups.queries, rows);
    let n_queries = queries.len() - 1;
    let block = n_queries.div_ceil(128).max(1);
    let mut merged = vec![0.0f64; n_leaves * n_leaves];
    for b in 0..n_queries.div_ceil(block) {
        let mut sum = vec![0.0f64; n_leaves * n_leaves];
        for q in b * block..((b + 1) * block).min(n_queries) {
            for w in queries[q]..queries[q + 1] {
                for &(l, cw) in groups.competitors(w) {
                    let (lw, ll) = (leaf_of[w] as usize, leaf_of[l] as usize);
                    if lw == ll {
                        continue;
                    }
                    let cw = f64::from(cw);
                    sum[lw * n_leaves + ll] -= cw;
                    sum[ll * n_leaves + lw] -= cw;
                    sum[lw * n_leaves + lw] += cw;
                    sum[ll * n_leaves + ll] += cw;
                }
            }
        }
        if b == 0 {
            merged = sum;
        } else {
            merged.iter_mut().zip(&sum).for_each(|(m, s)| *m += s);
        }
    }
    merged
}

/// What pairwise split scoring needs: the taken fold's pairs with their
/// bootstrapped weights, the L2 regularization and the prior (float32).
struct PairwiseScoring {
    pairs: Vec<(usize, usize, f32)>,
    l2: f64,
    prior: f64,
}

/// One pair cell's statistics (CatBoost's `TBucketPairWeightStatistics`).
#[derive(Clone, Copy, Default)]
struct PairCell {
    smaller: f64,
    greater: f64,
}

/// CatBoost's `TPairwiseScoreCalcer::CalculateScore` (SSE lanes, products
/// then adds).
fn pairwise_score(avrg: &[f64], der: &[f64], weights: &[f64]) -> f64 {
    let size = der.len();
    let mut score = 0.0;
    for x in 0..size {
        let row = &weights[x * size..(x + 1) * size];
        let (mut s0, mut s2) = ([0.0f64; 2], [0.0f64; 2]);
        let mut y = 0;
        while y + 4 <= size {
            s0[0] += avrg[y] * row[y];
            s0[1] += avrg[y + 1] * row[y + 1];
            s2[0] += avrg[y + 2] * row[y + 2];
            s2[1] += avrg[y + 3] * row[y + 3];
            y += 4;
        }
        let mut sub = (s0[0] + s0[1]) + (s2[0] + s2[1]);
        for y in size - size % 4..size {
            sub += avrg[y] * row[y];
        }
        score += avrg[x] * (der[x] - 0.5 * sub);
    }
    score
}

/// `UpdateWeightSumFromTotal`.
fn pairwise_add_total(w: &mut [f64], size: usize, y: usize, x: usize, total: f64) {
    w[(2 * y + 1) * size + 2 * x + 1] += total;
    w[(2 * x + 1) * size + 2 * y + 1] += total;
    w[(2 * x + 1) * size + 2 * x + 1] -= total;
    w[(2 * y + 1) * size + 2 * y + 1] -= total;
}

/// `UpdateWeightSumFromNonDiagStats`.
fn pairwise_add_non_diag(
    w: &mut [f64],
    size: usize,
    y: usize,
    x: usize,
    xy: PairCell,
    yx: PairCell,
) {
    let w00 = xy.greater + yx.greater;
    let w01 = xy.smaller - xy.greater;
    let w10 = yx.smaller - yx.greater;
    let w11 = -(xy.smaller + yx.smaller);
    let at = |r: usize, c: usize| r * size + c;
    w[at(2 * x, 2 * y)] += w00;
    w[at(2 * y, 2 * x)] += w00;
    w[at(2 * x, 2 * y + 1)] += w01;
    w[at(2 * y + 1, 2 * x)] += w01;
    w[at(2 * x + 1, 2 * y)] += w10;
    w[at(2 * y, 2 * x + 1)] += w10;
    w[at(2 * x + 1, 2 * y + 1)] += w11;
    w[at(2 * y + 1, 2 * x + 1)] += w11;
    w[at(2 * y, 2 * y)] -= w00 + w10;
    w[at(2 * x, 2 * x)] -= w00 + w01;
    w[at(2 * x + 1, 2 * x + 1)] -= w10 + w11;
    w[at(2 * y + 1, 2 * y + 1)] -= w01 + w11;
}

/// The diagonal update of a leaf's own pairs.
fn pairwise_add_diag(w: &mut [f64], size: usize, y: usize, delta: f64) {
    w[2 * y * size + 2 * y + 1] += delta;
    w[(2 * y + 1) * size + 2 * y] += delta;
    w[2 * y * size + 2 * y] -= delta;
    w[(2 * y + 1) * size + 2 * y + 1] -= delta;
}

/// CatBoost's pairwise split scores (`ComputePairwiseStats` and
/// `CalculatePairwiseScore`) of a float feature (`bits == 0`: every border
/// of `buckets`, `n_buckets` values) or of a pack of binary features
/// (`bits > 0`: `buckets` holds the pack value, one score per bit).
#[allow(clippy::too_many_arguments)]
fn pairwise_split_scores(
    buckets: &[u32],
    n_buckets: usize,
    bits: usize,
    leaf_of: &[u32],
    leaves: usize,
    wder: &[f64],
    pairs: &[(usize, usize, f32)],
    l2: f64,
    prior: f64,
) -> Vec<f64> {
    let n = leaf_of.len();
    let mut der_sums = vec![0.0f64; leaves * n_buckets];
    for i in 0..n {
        der_sums[leaf_of[i] as usize * n_buckets + buckets[i] as usize] += wder[i];
    }
    let cells = if bits > 0 { 2 * bits } else { n_buckets };
    let mut stats = vec![PairCell::default(); leaves * leaves * cells];
    let cell = |a: usize, b: usize, c: usize| (a * leaves + b) * cells + c;
    for &(w, l, sw) in pairs {
        if w == l {
            continue;
        }
        let sw = f64::from(sw);
        let (wl, ll) = (leaf_of[w] as usize, leaf_of[l] as usize);
        let (wb, lb) = (buckets[w] as usize, buckets[l] as usize);
        if bits == 0 {
            if wb > lb {
                stats[cell(ll, wl, lb)].smaller -= sw;
                stats[cell(ll, wl, wb)].greater -= sw;
            } else {
                stats[cell(wl, ll, wb)].smaller -= sw;
                stats[cell(wl, ll, lb)].greater -= sw;
            }
        } else {
            for bit in 0..bits {
                let (wbit, lbit) = ((wb >> bit) & 1, (lb >> bit) & 1);
                if wbit > lbit {
                    stats[cell(ll, wl, 2 * bit)].smaller -= sw;
                    stats[cell(ll, wl, 2 * bit + 1)].greater -= sw;
                } else {
                    stats[cell(wl, ll, 2 * bit + wbit)].smaller -= sw;
                    stats[cell(wl, ll, 2 * bit + lbit)].greater -= sw;
                }
            }
        }
    }
    let size = 2 * leaves;
    let mut out = Vec::new();
    if bits == 0 {
        let mut der = vec![0.0f64; size];
        let mut weights = vec![0.0f64; size * size];
        for leaf in 0..leaves {
            for b in 0..n_buckets {
                der[2 * leaf + 1] += der_sums[leaf * n_buckets + b];
            }
        }
        for y in 0..leaves {
            for x in y + 1..leaves {
                // SSE pairs of buckets, then the rest.
                let (mut xy0, mut xy2, mut yx0, mut yx2) =
                    ([0.0f64; 2], [0.0f64; 2], [0.0f64; 2], [0.0f64; 2]);
                let mut b = 0;
                while b + 4 <= n_buckets {
                    for lane in 0..2 {
                        xy0[lane] += stats[cell(x, y, b + lane)].smaller;
                        xy2[lane] += stats[cell(x, y, b + 2 + lane)].smaller;
                        yx0[lane] += stats[cell(y, x, b + lane)].smaller;
                        yx2[lane] += stats[cell(y, x, b + 2 + lane)].smaller;
                    }
                    b += 4;
                }
                let mut total =
                    (xy0[0] + xy0[1]) + (xy2[0] + xy2[1]) + (yx0[0] + yx0[1]) + (yx2[0] + yx2[1]);
                for b in n_buckets - n_buckets % 4..n_buckets {
                    total += stats[cell(x, y, b)].smaller + stats[cell(y, x, b)].smaller;
                }
                pairwise_add_total(&mut weights, size, y, x, total);
            }
        }
        for split in 0..n_buckets - 1 {
            for y in 0..leaves {
                let d = der_sums[y * n_buckets + split];
                der[2 * y] += d;
                der[2 * y + 1] -= d;
                let own = stats[cell(y, y, split)];
                pairwise_add_diag(&mut weights, size, y, own.smaller - own.greater);
                for x in y + 1..leaves {
                    pairwise_add_non_diag(
                        &mut weights,
                        size,
                        y,
                        x,
                        stats[cell(x, y, split)],
                        stats[cell(y, x, split)],
                    );
                }
            }
            let values = pairwise_leaf_values(&weights, &der, l2, prior);
            out.push(pairwise_score(&values, &der, &weights));
        }
    } else {
        for bit in 0..bits {
            let mut der = vec![0.0f64; size];
            for y in 0..leaves {
                for b in 0..n_buckets {
                    der[y * 2 + ((b >> bit) & 1)] += der_sums[y * n_buckets + b];
                }
            }
            let mut weights = vec![0.0f64; size * size];
            for y in 0..leaves {
                let own = stats[cell(y, y, 2 * bit)];
                pairwise_add_diag(&mut weights, size, y, own.smaller - own.greater);
                for x in y + 1..leaves {
                    let total = stats[cell(x, y, 2 * bit)].smaller
                        + stats[cell(y, x, 2 * bit)].smaller
                        + stats[cell(x, y, 2 * bit + 1)].smaller
                        + stats[cell(y, x, 2 * bit + 1)].smaller;
                    pairwise_add_total(&mut weights, size, y, x, total);
                    pairwise_add_non_diag(
                        &mut weights,
                        size,
                        y,
                        x,
                        stats[cell(x, y, 2 * bit)],
                        stats[cell(y, x, 2 * bit)],
                    );
                }
            }
            let values = pairwise_leaf_values(&weights, &der, l2, prior);
            out.push(pairwise_score(&values, &der, &weights));
        }
    }
    out
}

/// The query groups of `loss` (with pairs for a pairwise one).
fn groups_of(loss: Loss, queries: Vec<usize>, target: &[f64]) -> Groups {
    if loss.pairwise() {
        Groups::with_pairs(queries, target)
    } else {
        Groups {
            queries,
            ..Groups::default()
        }
    }
}

/// Query bounds (row offsets from 0 to `n`) of the query sizes `group`.
fn query_bounds(group: &[usize], n: usize) -> Result<Vec<usize>> {
    if group.contains(&0) || group.iter().sum::<usize>() != n {
        return Err(RustMlError::InvalidParameter(format!(
            "the query sizes must be positive and sum to the {n} rows"
        )));
    }
    let mut bounds = vec![0];
    for &g in group {
        bounds.push(bounds[bounds.len() - 1] + g);
    }
    Ok(bounds)
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
    (
        $(#[$doc:meta])* $name:ident
        $(, extra: { $($(#[$fdoc:meta])* $field:ident: $fty:ty = $fdef:expr),* $(,)? })?
    ) => {
        $(#[$doc])*
        #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
        #[serde(default)]
        pub struct $name {
            /// Number of trees (CatBoost's `iterations`, default 1000).
            pub iterations: usize,
            /// Shrinkage of every tree. Default (`None`): CatBoost's
            /// automatic learning rate from the row and iteration counts
            /// (when `l2_leaf_reg`, `leaf_estimation_method` and
            /// `leaf_estimation_iterations` keep their defaults; else 0.03).
            pub learning_rate: Option<f64>,
            /// Tree depth (levels of the symmetric tree, default 6).
            pub depth: usize,
            /// L2 regularization of the leaf values, scaled by the mean
            /// sample weight as in CatBoost. Default (`None`): CatBoost's, 3,
            /// or 0 for YetiRank.
            pub l2_leaf_reg: Option<f64>,
            /// Borders per numeric feature. Default (`None`): CatBoost's,
            /// 254, or 32 for pairwise scoring.
            pub border_count: Option<usize>,
            pub score_function: CatBoostScoreFunction,
            /// Default (`None`): CatBoost's for the loss (Exact for MAE and
            /// Quantile, Gradient for LogLinQuantile, else Newton).
            pub leaf_estimation_method: Option<CatBoostLeafEstimation>,
            /// Start from the loss-optimal constant instead of 0.
            /// Default (`None`): CatBoost's, on for RMSE, MAE and Quantile.
            pub boost_from_average: Option<bool>,
            /// Categorical features (their values are category codes).
            /// Those with at most `one_hot_max_size` categories are split
            /// one-hot, the others through CatBoost's ordered target
            /// statistics (CTRs), of single features and of combinations
            /// (`max_ctr_complexity`).
            pub cat_features: Vec<usize>,
            /// Default (`None`): CatBoost's, 2, or 10 for a groupwise loss
            /// (QueryRMSE).
            pub one_hot_max_size: Option<usize>,
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
            /// The noise of the split scores (CatBoost's `random_strength`;
            /// `NormalWithModelSizeDecrease`): normal with the root mean
            /// square of the derivatives times this, fading with the model's
            /// length. Default (`None`): CatBoost's, 1, or 0 for pairwise
            /// scoring.
            pub random_strength: Option<f64>,
            /// The prior of the pairwise leaf systems (CatBoost's
            /// `bayesian_matrix_reg`, default 0.1).
            pub bayesian_matrix_reg: f64,
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
            /// Per feature: -1 (decreasing), 0 or +1 (increasing); numeric
            /// features only (CatBoost's `monotone_constraints`; empty =
            /// none). Not with more than two classes.
            pub monotone_constraints: Vec<i8>,
            /// CatBoost's `model_shrink_rate`: the model is shrunk before
            /// every iteration. Default (`None`): 0, or with monotone
            /// constraints 0.01 (`Constant`) / 0.2 (`Decreasing`).
            pub model_shrink_rate: Option<f64>,
            /// CatBoost's `model_shrink_mode` (default `Constant`).
            pub model_shrink_mode: CatBoostShrinkMode,
            /// Keep the trees up to the best iteration on the eval set
            /// (`fit_with_eval`). Default (`None`): on with an eval set.
            pub use_best_model: Option<bool>,
            /// CatBoost's overfitting detector on the eval set. Default
            /// (`None`): `IncToDec` with `od_pval`, else `Iter` with
            /// `od_wait`, else none.
            pub od_type: Option<CatBoostOdType>,
            /// Iterations to wait after the best one (default 20).
            pub od_wait: Option<usize>,
            /// `IncToDec`'s p-value threshold (default 0: off).
            pub od_pval: Option<f64>,
            /// CatBoost's `grow_policy`: oblivious trees (default), Depthwise
            /// (plain boosting, numeric and one-hot features) or Lossguide
            /// (also CTR features; classifiers with CTRs can differ from
            /// CatBoost by picking another CTR in a leaf, #154).
            pub grow_policy: CatBoostGrowPolicy,
            /// Depthwise: leaves with fewer sampled rows are not split
            /// (CatBoost's `min_data_in_leaf`, default 1).
            pub min_data_in_leaf: usize,
            /// Lossguide: the most leaves of a tree (default 31).
            pub max_leaves: usize,
            /// Threads of a fit (CatBoost's `thread_count`): `None` = the
            /// physical cores (at most the current rayon pool's), which on
            /// SMT / hybrid CPUs beats all logical ones. The model does not
            /// depend on it.
            pub thread_count: Option<usize>,
            $($($(#[$fdoc])* pub $field: $fty,)*)?
        }

        impl Default for $name {
            fn default() -> Self {
                Self {
                    iterations: 1000,
                    learning_rate: None,
                    depth: 6,
                    l2_leaf_reg: None,
                    border_count: None,
                    score_function: CatBoostScoreFunction::Cosine,
                    leaf_estimation_method: None,
                    boost_from_average: None,
                    cat_features: Vec::new(),
                    one_hot_max_size: None,
                    max_ctr_complexity: None,
                    model_size_reg: 0.5,
                    boosting_type: CatBoostBoostingType::Plain,
                    fold_len_multiplier: 2.0,
                    leaf_estimation_iterations: None,
                    leaf_estimation_backtracking: CatBoostBacktracking::AnyImprovement,
                    has_time: false,
                    permutation_count: 4,
                    random_seed: 0,
                    random_strength: None,
                    bayesian_matrix_reg: 0.1,
                    bootstrap_type: None,
                    subsample: None,
                    bagging_temperature: 1.0,
                    mvs_reg: None,
                    monotone_constraints: Vec::new(),
                    model_shrink_rate: None,
                    model_shrink_mode: CatBoostShrinkMode::Constant,
                    use_best_model: None,
                    od_type: None,
                    od_wait: None,
                    od_pval: None,
                    grow_policy: CatBoostGrowPolicy::SymmetricTree,
                    min_data_in_leaf: 1,
                    max_leaves: 31,
                    thread_count: None,
                    $($($field: $fdef,)*)?
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
                self.learning_rate = Some(lr);
                self
            }
            pub fn with_depth(mut self, d: usize) -> Self {
                self.depth = d;
                self
            }
            pub fn with_l2_leaf_reg(mut self, l2: f64) -> Self {
                self.l2_leaf_reg = Some(l2);
                self
            }
            pub fn with_border_count(mut self, b: usize) -> Self {
                self.border_count = Some(b);
                self
            }
            pub fn with_bayesian_matrix_reg(mut self, r: f64) -> Self {
                self.bayesian_matrix_reg = r;
                self
            }
            /// Borders per feature for `loss` (CatBoost's default: 32 for
            /// pairwise scoring).
            fn border_count_for(&self, loss: Loss) -> usize {
                self.border_count
                    .unwrap_or(if loss.pairwise_scoring() { 32 } else { 254 })
            }
            pub fn with_score_function(mut self, s: CatBoostScoreFunction) -> Self {
                self.score_function = s;
                self
            }
            pub fn with_leaf_estimation_method(mut self, m: CatBoostLeafEstimation) -> Self {
                self.leaf_estimation_method = Some(m);
                self
            }
            pub fn with_boost_from_average(mut self, b: bool) -> Self {
                self.boost_from_average = Some(b);
                self
            }
            pub fn with_cat_features(mut self, features: Vec<usize>) -> Self {
                self.cat_features = features;
                self
            }
            /// CatBoost's `one_hot_max_size`: 2, or 10 for a groupwise
            /// loss (`SetOneHotMaxSizeAndPrintNotice`).
            fn one_hot_max_size_for(&self, loss: Loss) -> usize {
                self.one_hot_max_size
                    .unwrap_or(if loss.querywise() { 10 } else { 2 })
            }
            pub fn with_one_hot_max_size(mut self, n: usize) -> Self {
                self.one_hot_max_size = Some(n);
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
            /// CatBoost's leaf estimation method for `loss` (Exact needs no
            /// monotone constraints).
            fn leaf_method(&self, loss: Loss) -> CatBoostLeafEstimation {
                let monotone = self.monotone_constraints.iter().any(|&m| m != 0);
                self.leaf_estimation_method.unwrap_or(match loss {
                    Loss::Quantile { .. } if !monotone => CatBoostLeafEstimation::Exact,
                    Loss::Quantile { .. } | Loss::LogLinQuantile { .. } => {
                        CatBoostLeafEstimation::Gradient
                    }
                    Loss::LogCosh if !monotone => CatBoostLeafEstimation::Exact,
                    Loss::Lq { q } if q < 2.0 => CatBoostLeafEstimation::Gradient,
                    // CPU pairwise scoring has no Newton.
                    l if l.pairwise_scoring() => CatBoostLeafEstimation::Gradient,
                    _ => CatBoostLeafEstimation::Newton,
                })
            }

            /// CatBoost's `boost_from_average` for `loss`.
            fn boost_from_average_for(&self, loss: Loss) -> bool {
                self.boost_from_average.unwrap_or(loss.default_boost_from_average())
            }

            fn leaf_iterations(&self, loss: Loss, n_features: usize) -> usize {
                if let Some(n) = self.leaf_estimation_iterations {
                    return n;
                }
                let method = self.leaf_method(loss);
                if self.iterations < 200 && n_features < 20 {
                    return 1;
                }
                // CatBoost's `GetEstimationMethodDefaults` (CPU): Newton,
                // Gradient steps.
                let (newton, gradient) = match loss {
                    Loss::Rmse | Loss::Quantile { .. } | Loss::Tweedie { .. } => (1, 1),
                    Loss::Huber { .. } | Loss::LogCosh | Loss::Lq { .. } => (1, 1),
                    Loss::RmseWithUncertainty | Loss::MultiRmse(_) => (1, 1),
                    Loss::QueryRmse | Loss::PairLogit | Loss::YetiRank { .. } => (1, 1),
                    Loss::YetiRankPairwise { .. } => (1, 1),
                    Loss::PairLogitPairwise => (1, 50),
                    Loss::LogLinQuantile { .. } => (1, 1),
                    Loss::Expectile { .. } => (5, 10),
                    Loss::Poisson => (10, 1),
                    Loss::Logloss => (10, 40),
                    Loss::MultiClass(_) | Loss::MultiClassOneVsAll(_) => (1, 10),
                };
                match method {
                    CatBoostLeafEstimation::Newton => newton,
                    CatBoostLeafEstimation::Gradient => gradient,
                    CatBoostLeafEstimation::Exact => 1,
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
            pub fn with_monotone_constraints(mut self, m: Vec<i8>) -> Self {
                self.monotone_constraints = m;
                self
            }
            pub fn with_model_shrink_rate(mut self, r: f64) -> Self {
                self.model_shrink_rate = Some(r);
                self
            }
            pub fn with_model_shrink_mode(mut self, m: CatBoostShrinkMode) -> Self {
                self.model_shrink_mode = m;
                self
            }
            pub fn with_use_best_model(mut self, b: bool) -> Self {
                self.use_best_model = Some(b);
                self
            }
            /// Stop `n` iterations after the best one on the eval set
            /// (CatBoost's `early_stopping_rounds`: `Iter` with `od_wait`).
            pub fn with_early_stopping_rounds(mut self, n: usize) -> Self {
                self.od_type = Some(CatBoostOdType::Iter);
                self.od_wait = Some(n);
                self
            }
            pub fn with_od_type(mut self, t: CatBoostOdType) -> Self {
                self.od_type = Some(t);
                self
            }
            pub fn with_od_wait(mut self, n: usize) -> Self {
                self.od_wait = Some(n);
                self
            }
            pub fn with_grow_policy(mut self, g: CatBoostGrowPolicy) -> Self {
                self.grow_policy = g;
                self
            }
            pub fn with_thread_count(mut self, n: Option<usize>) -> Self {
                self.thread_count = n;
                self
            }
            pub fn with_max_leaves(mut self, n: usize) -> Self {
                self.max_leaves = n;
                self
            }
            pub fn with_min_data_in_leaf(mut self, n: usize) -> Self {
                self.min_data_in_leaf = n;
                self
            }
            pub fn with_od_pval(mut self, p: f64) -> Self {
                self.od_pval = Some(p);
                self
            }
            pub fn with_mvs_reg(mut self, r: f64) -> Self {
                self.mvs_reg = Some(r);
                self
            }

            /// CatBoost's learning rate for `loss` on `n` rows: the given
            /// one, else `TAutoLRParamsGuesser` (CPU, no best model),
            /// `exp(A ln n + B) · exp(C ln iterations) / exp(C ln 1000)`
            /// capped at 0.5 and rounded to 6 digits.
            fn learning_rate_for(&self, loss: Loss, n: usize, use_best: bool) -> f64 {
                if let Some(lr) = self.learning_rate {
                    return lr;
                }
                let defaults = self.l2_leaf_reg.unwrap_or(3.0) == 3.0
                    && self.leaf_estimation_method.is_none()
                    && self.leaf_estimation_iterations.is_none();
                // (A, B, C, D): dataset size and iteration coefficients.
                let bfa = if matches!(loss, Loss::Logloss | Loss::MultiClass(_)) {
                    self.boost_from_average.unwrap_or(false)
                } else {
                    self.boost_from_average_for(loss)
                };
                let coeffs = match (loss, bfa, use_best) {
                    (Loss::Logloss, true, true) => (0.246, -5.127, -0.451, 0.978),
                    (Loss::Logloss, true, false) => (0.408, -7.299, -0.928, 2.701),
                    (Loss::Logloss, false, true) => (0.247, -5.158, -0.435, 0.934),
                    (Loss::Logloss, false, false) => (0.427, -7.525, -0.917, 2.63),
                    (Loss::MultiClass(_), false, true) => (0.02, -2.364, -0.382, 0.924),
                    (Loss::MultiClass(_), false, false) => (0.051, -2.889, -0.845, 2.928),
                    (Loss::Rmse, true, true) => (0.157, -4.062, -0.61, 1.557),
                    (Loss::Rmse, true, false) => (0.158, -4.287, -0.813, 2.571),
                    (Loss::Rmse, false, true) => (0.189, -4.383, -0.623, 1.439),
                    (Loss::Rmse, false, false) => (0.178, -4.473, -0.76, 2.133),
                    _ => return 0.03,
                };
                if !defaults {
                    return 0.03;
                }
                let (a, b, c, d): (f64, f64, f64, f64) = coeffs;
                let custom = (c * (self.iterations as f64).ln() + d).exp();
                let default = (c * 1000f64.ln() + d).exp();
                let base = (a * (n as f64).ln() + b).exp();
                let lr = (base * custom / default).min(0.5);
                let m = 10f64.powi(6);
                (lr * m).round() / m
            }

            /// CatBoost's bootstrap defaults for `loss` and `n` rows.
            fn sampling(&self, loss: Loss, n: usize) -> (CatBoostBootstrap, f64) {
                let kind = self.bootstrap_type.unwrap_or(match loss {
                    Loss::MultiClass(_) | Loss::MultiClassOneVsAll(_) => {
                        CatBoostBootstrap::Bayesian
                    }
                    // Pairwise scoring: Bernoulli pairs, half of them.
                    l if l.pairwise_scoring() => CatBoostBootstrap::Bernoulli,
                    _ => CatBoostBootstrap::Mvs,
                });
                let rate = self.subsample.unwrap_or(if n < 100 {
                    1.0
                } else if kind == CatBoostBootstrap::Mvs {
                    0.8
                } else if loss.pairwise_scoring() {
                    0.5
                } else {
                    0.66
                });
                (kind, rate)
            }

            pub fn with_random_strength(mut self, r: f64) -> Self {
                self.random_strength = Some(r);
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
            /// `use_best_model` for a fit with or without an eval set (whose
            /// shape is checked).
            fn use_best_model_with(
                &self,
                eval: Option<&CatBoostEvalSet<'_>>,
                n_features: usize,
            ) -> Result<bool> {
                let invalid = |m: String| Err(RustMlError::InvalidParameter(m));
                let Some(e) = eval else {
                    if self.use_best_model == Some(true) {
                        return invalid("use_best_model needs an eval set".into());
                    }
                    return Ok(false);
                };
                if e.x.nrows() == 0 || e.x.nrows() != e.y.len() {
                    return invalid("the eval set needs as many rows as targets, at least one".into());
                }
                if e.x.ncols() != n_features {
                    return invalid(format!(
                        "the eval set has {} features, X {n_features}",
                        e.x.ncols()
                    ));
                }
                if e.sample_weight.is_some_and(|w| w.len() != e.y.len()) {
                    return invalid("the eval set's sample_weight does not match its rows".into());
                }
                Ok(self.use_best_model.unwrap_or(true))
            }

            /// What Depthwise trees support here.
            fn check_grow_policy(&self, has_ctrs: bool) -> Result<()> {
                if self.grow_policy == CatBoostGrowPolicy::SymmetricTree {
                    return Ok(());
                }
                let invalid = |m: &str| Err(RustMlError::InvalidParameter(m.into()));
                if self.boosting_type == CatBoostBoostingType::Ordered {
                    return invalid("Depthwise / Lossguide trees need plain boosting");
                }
                // Depthwise with CTRs does not match CatBoost yet (#154).
                if has_ctrs && self.grow_policy == CatBoostGrowPolicy::Depthwise {
                    return invalid(
                        "Depthwise trees support numeric and one-hot features only (Lossguide also CTRs)",
                    );
                }
                if self.monotone_constraints.iter().any(|&m| m != 0) {
                    return invalid("Depthwise / Lossguide trees do not support monotone constraints");
                }
                if self.min_data_in_leaf == 0 {
                    return invalid("min_data_in_leaf must be at least 1");
                }
                if self.max_leaves == 0 {
                    return invalid("max_leaves must be at least 1");
                }
                Ok(())
            }

            /// CatBoost's checks of `monotone_constraints` and the shrinkage.
            fn check_monotone(&self, n_features: usize) -> Result<()> {
                let invalid = |m: String| Err(RustMlError::InvalidParameter(m));
                let m = &self.monotone_constraints;
                if !m.is_empty() && m.len() != n_features {
                    return invalid(format!(
                        "monotone_constraints has {} entries but X has {n_features} features",
                        m.len()
                    ));
                }
                if m.iter().any(|v| !(-1..=1).contains(v)) {
                    return invalid("monotone_constraints must be -1, 0 or 1".into());
                }
                if self.cat_features.iter().any(|&f| m.get(f).is_some_and(|&v| v != 0)) {
                    return invalid("monotone constraints apply to numeric features only".into());
                }
                if self.model_shrink_rate.is_some_and(|r| !(0.0..1.0).contains(&r)) {
                    return invalid("model_shrink_rate must be in [0, 1)".into());
                }
                Ok(())
            }

            fn train_params(&self) -> TrainParams {
                TrainParams {
                    iterations: self.iterations,
                    learning_rate: f64::from(self.learning_rate.unwrap_or(0.03) as f32),
                    depth: self.depth,
                    // CatBoost makes an L2 of 0 1e-20 (`SetLeavesEstimationDefault`).
                    l2_leaf_reg: f64::from(match self.l2_leaf_reg.unwrap_or(3.0) as f32 {
                        0.0 => 1e-20,
                        l2 => l2,
                    }),
                    score_function: self.score_function,
                    leaf_estimation: self.leaf_estimation_method.unwrap_or_default(),
                    model_size_reg: self.model_size_reg,
                    ordered: self.boosting_type == CatBoostBoostingType::Ordered,
                    fold_len_multiplier: self.fold_len_multiplier,
                    leaf_iterations: 1,
                    backtracking: self.leaf_estimation_backtracking
                        == CatBoostBacktracking::AnyImprovement,
                    has_time: self.has_time,
                    random_strength: f64::from(self.random_strength.unwrap_or(1.0) as f32),
                    pairwise_prior: f64::from(self.bayesian_matrix_reg as f32),
                    bootstrap: CatBoostBootstrap::No,
                    subsample: 1.0,
                    bagging_temperature: self.bagging_temperature,
                    mvs_reg: self.mvs_reg,
                    permutation_count: self.permutation_count,
                    monotone: if self.monotone_constraints.iter().any(|&m| m != 0) {
                        self.monotone_constraints.clone()
                    } else {
                        Vec::new()
                    },
                    // CatBoost's default with monotone constraints: 0.01
                    // (Constant mode) or 0.2 (Decreasing).
                    shrink_rate: self.model_shrink_rate.unwrap_or(
                        if self.monotone_constraints.iter().any(|&m| m != 0) {
                            match self.model_shrink_mode {
                                CatBoostShrinkMode::Constant => 0.01,
                                CatBoostShrinkMode::Decreasing => 0.2,
                            }
                        } else {
                            0.0
                        },
                    ),
                    shrink_mode: self.model_shrink_mode,
                    od_threshold: match self.od_type.unwrap_or(if self.od_pval.is_some() {
                        CatBoostOdType::IncToDec
                    } else if self.od_wait.is_some() {
                        CatBoostOdType::Iter
                    } else {
                        CatBoostOdType::None
                    }) {
                        CatBoostOdType::None => 0.0,
                        CatBoostOdType::Iter => 1.0,
                        CatBoostOdType::IncToDec => self.od_pval.unwrap_or(0.0),
                    },
                    od_wait: self.od_wait.unwrap_or(20),
                    use_best_model: false,
                    grow_policy: self.grow_policy,
                    min_data_in_leaf: self.min_data_in_leaf,
                    max_leaves: self.max_leaves,
                }
            }
        }
    };
}

catboost_config!(
    /// CatBoost-style regressor (RMSE) on oblivious trees.
    CatBoostRegressor,
    extra: {
        /// CatBoost's `loss_function` (default RMSE).
        loss_function: CatBoostRegressionLoss = CatBoostRegressionLoss::Rmse,
    }
);

impl CatBoostRegressor {
    /// CatBoost's checks of the loss, its parameters and the target.
    fn check_loss(&self, loss: Loss, target: &[f64]) -> Result<()> {
        if let Loss::MultiRmse(k) = loss {
            if k < 2 {
                return Err(RustMlError::InvalidParameter(
                    "MultiRMSE needs at least two target columns (RMSE for one)".into(),
                ));
            }
            // CatBoost builds a CTR target border per target column; not
            // ported.
            if !self.cat_features.is_empty() {
                return Err(RustMlError::InvalidParameter(
                    "MultiRMSE does not support categorical features".into(),
                ));
            }
        }
        let invalid = |m: &str| Err(RustMlError::InvalidParameter(m.into()));
        let unit = |a: f64| (0.0..=1.0).contains(&a);
        match loss {
            Loss::Quantile { alpha, .. }
            | Loss::Expectile { alpha }
            | Loss::LogLinQuantile { alpha }
                if !unit(alpha) =>
            {
                return invalid("alpha must be in [0, 1]");
            }
            Loss::Tweedie { power } if !(power > 1.0 && power < 2.0) => {
                return invalid("variance_power must be in (1, 2)");
            }
            Loss::Poisson | Loss::Tweedie { .. } if target.iter().any(|&t| t < 0.0) => {
                return invalid("Poisson and Tweedie need non-negative targets");
            }
            Loss::Huber { delta } if !(delta > 0.0) => return invalid("delta must be positive"),
            Loss::Lq { q } if !(q >= 1.0) => return invalid("q must be at least 1"),
            _ => {}
        }
        if self.boost_from_average_for(loss)
            && !matches!(
                loss,
                Loss::Rmse | Loss::Quantile { .. } | Loss::MultiRmse(_)
            )
        {
            return invalid(
                "boost_from_average is supported for RMSE, MAE, Quantile and MultiRMSE",
            );
        }
        if self.leaf_method(loss) == CatBoostLeafEstimation::Exact {
            if !matches!(loss, Loss::Quantile { .. } | Loss::LogCosh) {
                return invalid(
                    "Exact leaf estimation is only available for MAE, Quantile and LogCosh",
                );
            }
            if self.monotone_constraints.iter().any(|&m| m != 0) {
                return invalid("monotone constraints are unsupported for Exact leaves");
            }
        }
        Ok(())
    }

    pub fn with_loss_function(mut self, l: CatBoostRegressionLoss) -> Self {
        self.loss_function = l;
        self
    }

    /// The engine's loss for `targets` target columns.
    fn loss(&self, targets: usize) -> Loss {
        match self.loss_function {
            CatBoostRegressionLoss::Rmse => Loss::Rmse,
            CatBoostRegressionLoss::Mae => Loss::Quantile {
                alpha: 0.5,
                delta: 1e-6,
                mae: true,
            },
            CatBoostRegressionLoss::Quantile { alpha } => Loss::Quantile {
                alpha,
                delta: 1e-6,
                mae: false,
            },
            // Error function and metric parse `alpha` as float32.
            CatBoostRegressionLoss::Expectile { alpha } => Loss::Expectile {
                alpha: f64::from(alpha as f32),
            },
            CatBoostRegressionLoss::Poisson => Loss::Poisson,
            CatBoostRegressionLoss::Tweedie { variance_power } => Loss::Tweedie {
                power: variance_power,
            },
            CatBoostRegressionLoss::LogLinQuantile { alpha } => Loss::LogLinQuantile {
                alpha: f64::from(alpha as f32),
            },
            CatBoostRegressionLoss::Huber { delta } => Loss::Huber { delta },
            CatBoostRegressionLoss::LogCosh => Loss::LogCosh,
            CatBoostRegressionLoss::Lq { q } => Loss::Lq { q },
            CatBoostRegressionLoss::RmseWithUncertainty => Loss::RmseWithUncertainty,
            CatBoostRegressionLoss::MultiRmse => Loss::MultiRmse(targets),
            CatBoostRegressionLoss::QueryRmse => Loss::QueryRmse,
            CatBoostRegressionLoss::PairLogit => Loss::PairLogit,
            CatBoostRegressionLoss::YetiRank {
                permutations,
                decay,
            } => Loss::YetiRank {
                permutations,
                decay,
            },
            CatBoostRegressionLoss::PairLogitPairwise => Loss::PairLogitPairwise,
            CatBoostRegressionLoss::YetiRankPairwise {
                permutations,
                decay,
            } => Loss::YetiRankPairwise {
                permutations,
                decay,
            },
        }
    }
}
catboost_config!(
    /// CatBoost-style classifier on oblivious trees: Logloss for two
    /// classes, MultiClass (softmax) for more, as CatBoost picks them.
    CatBoostClassifier,
    extra: {
        /// Per-class weights (CatBoost's `class_weights`, by class in
        /// ascending order of the labels), multiplied into the sample
        /// weights.
        class_weights: Option<Vec<f64>> = None,
        /// CatBoost's `auto_class_weights` (used without `class_weights`).
        auto_class_weights: Option<CatBoostAutoClassWeights> = None,
        /// The loss: Logloss / MultiClass by the classes, or CrossEntropy.
        loss_function: CatBoostClassifierLoss = CatBoostClassifierLoss::Auto,
    }
);

impl CatBoostClassifier {
    pub fn with_loss_function(mut self, l: CatBoostClassifierLoss) -> Self {
        self.loss_function = l;
        self
    }
    pub fn with_class_weights(mut self, w: Vec<f64>) -> Self {
        self.class_weights = Some(w);
        self
    }
    pub fn with_auto_class_weights(mut self, a: CatBoostAutoClassWeights) -> Self {
        self.auto_class_weights = Some(a);
        self
    }

    /// The sample weights with the class weights (CatBoost's
    /// `MakeClassificationWeights`: float32 products).
    /// The class weights: given, or CatBoost's automatic ones from the
    /// training rows (which also weight an eval set).
    fn class_weight_vector(
        &self,
        classes_of: &[u8],
        n_classes: usize,
        weights: &[f64],
    ) -> Result<Option<Vec<f32>>> {
        Ok(Some(match (&self.class_weights, self.auto_class_weights) {
            (Some(w), _) => {
                if w.len() != n_classes {
                    return Err(RustMlError::InvalidParameter(format!(
                        "class_weights has {} entries for {n_classes} classes",
                        w.len()
                    )));
                }
                if w.iter().any(|&v| !(v >= 0.0) || !v.is_finite()) {
                    return Err(RustMlError::InvalidParameter(
                        "class_weights must be finite and non-negative".into(),
                    ));
                }
                w.iter().map(|&v| v as f32).collect()
            }
            (None, Some(auto)) => auto_class_weights(classes_of, n_classes, weights, auto),
            (None, None) => return Ok(None),
        }))
    }
}

/// Sample weights times the class weights (CatBoost's
/// `MakeClassificationWeights`: float32 products).
fn apply_class_weights(
    weights: Vec<f64>,
    classes_of: &[u8],
    class_weights: Option<&[f32]>,
) -> Vec<f64> {
    match class_weights {
        None => weights,
        Some(cw) => weights
            .iter()
            .zip(classes_of)
            .map(|(&w, &c)| f64::from(w as f32 * cw[c as usize]))
            .collect(),
    }
}

/// Fitted [`CatBoostRegressor`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FittedCatBoostRegressor {
    trees: Vec<ObliviousTree>,
    #[serde(default)]
    ctrs: CtrModel,
    bias: f64,
    /// Predictions are `exp(raw)` (Poisson, Tweedie).
    #[serde(default)]
    exp_prediction: bool,
    #[serde(default)]
    eval_metrics: Vec<f64>,
    #[serde(default)]
    best_iteration: Option<usize>,
    borders: Vec<Vec<f32>>,
    n_features: usize,
    /// RMSEWithUncertainty: the starting log standard deviation (the model
    /// then has two approximations per row).
    #[serde(default)]
    log_sigma_bias: Option<f64>,
    /// MultiRMSE: the starting approximations of the targets after the
    /// first (the model then has one approximation per target).
    #[serde(default)]
    multi_bias: Vec<f64>,
}

impl FittedCatBoostRegressor {
    fn biases(&self) -> Vec<f64> {
        std::iter::once(self.bias)
            .chain(self.log_sigma_bias)
            .chain(self.multi_bias.iter().copied())
            .collect()
    }

    /// MultiRMSE: one predicted column per target.
    pub fn predict_multi<'a>(&self, x: impl AsArray<'a, f64, Ix2>) -> Result<Array2<f64>> {
        let x: ArrayView2<f64> = x.into();
        check_features(x, self.n_features)?;
        if self.multi_bias.is_empty() {
            return Err(RustMlError::InvalidParameter(
                "predict_multi needs MultiRMSE".into(),
            ));
        }
        Ok(raw_scores(&self.trees, &self.ctrs, &self.biases(), x))
    }

    /// Target columns the model predicts (MultiRMSE: more than one).
    pub fn n_targets(&self) -> usize {
        1 + self.multi_bias.len()
    }

    /// RMSEWithUncertainty: the mean and the variance `exp(2 raw₁)` of
    /// every row (CatBoost's default prediction for the loss).
    pub fn predict_uncertainty<'a>(&self, x: impl AsArray<'a, f64, Ix2>) -> Result<Array2<f64>> {
        let x: ArrayView2<f64> = x.into();
        check_features(x, self.n_features)?;
        if self.log_sigma_bias.is_none() {
            return Err(RustMlError::InvalidParameter(
                "predict_uncertainty needs RMSEWithUncertainty".into(),
            ));
        }
        let mut raw = raw_scores(&self.trees, &self.ctrs, &self.biases(), x);
        raw.column_mut(1).mapv_inplace(|v| (2.0 * v).exp());
        Ok(raw)
    }
    pub fn n_estimators(&self) -> usize {
        self.trees.len()
    }
    /// The eval set's metric after every iteration (`fit_with_eval`).
    pub fn eval_metrics(&self) -> &[f64] {
        &self.eval_metrics
    }
    /// The iteration with the best eval metric (0-based).
    pub fn best_iteration(&self) -> Option<usize> {
        self.best_iteration
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
        self.fit_inner(x, y, sample_weight, None)
    }
}

impl CatBoostRegressor {
    /// Fit with an eval set (CatBoost's `eval_set`): its metric after every
    /// iteration, the overfitting detector and `use_best_model` (default
    /// on).
    pub fn fit_with_eval(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        sample_weight: Option<ArrayView1<f64>>,
        eval: CatBoostEvalSet<'_>,
    ) -> Result<FittedCatBoostRegressor> {
        self.fit_inner(x, y, sample_weight, Some(eval))
    }

    fn fit_inner(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        sample_weight: Option<ArrayView1<f64>>,
        eval: Option<CatBoostEvalSet<'_>>,
    ) -> Result<FittedCatBoostRegressor> {
        if self.loss_function == CatBoostRegressionLoss::MultiRmse {
            return Err(RustMlError::InvalidParameter(
                "MultiRMSE has a target matrix: use fit_multi".into(),
            ));
        }
        if self.loss_function == CatBoostRegressionLoss::QueryRmse {
            return Err(RustMlError::InvalidParameter(
                "QueryRMSE needs query groups: use fit_with_groups".into(),
            ));
        }
        self.fit_grouped(x, y, sample_weight, None, eval.map(|e| (e, None)))
    }

    /// Fit a querywise loss (QueryRMSE) on query groups: `group` holds the
    /// sizes of the queries, whose rows are consecutive (CatBoost's
    /// `group_id`).
    pub fn fit_with_groups(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        sample_weight: Option<ArrayView1<f64>>,
        group: &[usize],
    ) -> Result<FittedCatBoostRegressor> {
        self.fit_grouped(x, y, sample_weight, Some(group), None)
    }

    /// [`fit_with_groups`](Self::fit_with_groups) with an eval set and its
    /// query sizes `eval_group`.
    pub fn fit_with_groups_eval(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        sample_weight: Option<ArrayView1<f64>>,
        group: &[usize],
        eval: CatBoostEvalSet<'_>,
        eval_group: &[usize],
    ) -> Result<FittedCatBoostRegressor> {
        self.fit_grouped(
            x,
            y,
            sample_weight,
            Some(group),
            Some((eval, Some(eval_group))),
        )
    }

    fn fit_grouped(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        sample_weight: Option<ArrayView1<f64>>,
        group: Option<&[usize]>,
        eval: Option<(CatBoostEvalSet<'_>, Option<&[usize]>)>,
    ) -> Result<FittedCatBoostRegressor> {
        let loss = self.loss(1);
        if group.is_some() != loss.querywise() {
            return Err(RustMlError::InvalidParameter(
                "query groups go with a querywise loss (QueryRMSE, PairLogit)".into(),
            ));
        }
        if loss.pairwise() && sample_weight.is_some() {
            return Err(RustMlError::InvalidParameter(
                "PairLogit and YetiRank do not take sample weights".into(),
            ));
        }
        if loss.pairwise_scoring() {
            let invalid = |m: &str| Err(RustMlError::InvalidParameter(m.into()));
            // CatBoost's: plain symmetric trees, no Newton, no monotone
            // constraints. Not ported: categorical features (one-hot and CTR
            // pairwise statistics).
            if self.boosting_type == CatBoostBoostingType::Ordered
                || self.grow_policy != CatBoostGrowPolicy::SymmetricTree
            {
                return invalid("pairwise scoring needs plain boosting and symmetric trees");
            }
            if self.leaf_estimation_method == Some(CatBoostLeafEstimation::Newton) {
                return invalid("pairwise scoring has no Newton leaf estimation");
            }
            if self.monotone_constraints.iter().any(|&m| m != 0) || !self.cat_features.is_empty() {
                return invalid("pairwise scoring supports neither monotone constraints nor categorical features");
            }
            if self.depth > 6 {
                return invalid("pairwise scoring supports depth up to 6");
            }
            if let Some((permutations, decay)) = loss.yeti_rank() {
                if permutations == 0 || !(decay > 0.0 && decay <= 1.0) {
                    return invalid("YetiRankPairwise needs permutations >= 1 and decay in (0, 1]");
                }
                if self.leaf_estimation_iterations.is_some_and(|i| i != 1) {
                    return invalid("YetiRankPairwise takes one leaf step");
                }
            }
        }
        if let Loss::YetiRank {
            permutations,
            decay,
        } = loss
        {
            let invalid = |m: &str| Err(RustMlError::InvalidParameter(m.into()));
            if permutations == 0 || !(decay > 0.0 && decay <= 1.0) {
                return invalid("YetiRank needs permutations >= 1 and decay in (0, 1]");
            }
            // Not ported: pairs per body / tail prefix, the PFound / NDCG
            // eval metrics, pairs per leaf iteration.
            if self.boosting_type == CatBoostBoostingType::Ordered {
                return invalid("YetiRank supports plain boosting only");
            }
            if self.leaf_estimation_iterations.is_some_and(|i| i != 1)
                || self
                    .leaf_estimation_method
                    .is_some_and(|m| m != CatBoostLeafEstimation::Newton)
            {
                return invalid("YetiRank takes one Newton leaf step");
            }
        }
        let use_best = self.use_best_model_with(eval.as_ref().map(|e| &e.0), x.ncols())?;
        let queries = group.map(|g| query_bounds(g, x.nrows())).transpose()?;
        let in_unit = |mut t: ndarray::iter::Iter<'_, f64, ndarray::Ix1>| {
            t.all(|&v| (0.0..=1.0).contains(&(v as f32)))
        };
        let eval = eval
            .map(|(e, g)| -> Result<EvalSet<'_>> {
                Ok(EvalSet {
                    x: e.x,
                    target: f32_values(e.y),
                    weights: e
                        .sample_weight
                        .map_or_else(|| vec![1.0; e.y.len()], f32_values),
                    groups: match g {
                        Some(g) => groups_of(loss, query_bounds(g, e.y.len())?, &f32_values(e.y)),
                        None => Groups::default(),
                    },
                    // `UpdateYetiRankEvalMetric`: PFound when the learn and
                    // eval targets are in [0, 1], else NDCG.
                    pfound: in_unit(y.iter()) && in_unit(e.y.iter()),
                })
            })
            .transpose()?;
        self.fit_targets(
            x,
            y.insert_axis(ndarray::Axis(1)),
            sample_weight,
            eval,
            use_best,
            queries,
        )
    }

    /// MultiRMSE (CatBoost's multi-regression): `y` holds one column per
    /// target.
    pub fn fit_multi(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView2<f64>,
        sample_weight: Option<ArrayView1<f64>>,
    ) -> Result<FittedCatBoostRegressor> {
        self.fit_multi_inner(x, y, sample_weight, None)
    }

    /// [`fit_multi`](Self::fit_multi) with an eval set, as
    /// [`fit_with_eval`](Self::fit_with_eval).
    pub fn fit_multi_with_eval(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView2<f64>,
        sample_weight: Option<ArrayView1<f64>>,
        eval: CatBoostMultiEvalSet<'_>,
    ) -> Result<FittedCatBoostRegressor> {
        self.fit_multi_inner(x, y, sample_weight, Some(eval))
    }

    fn fit_multi_inner(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView2<f64>,
        sample_weight: Option<ArrayView1<f64>>,
        eval: Option<CatBoostMultiEvalSet<'_>>,
    ) -> Result<FittedCatBoostRegressor> {
        if self.loss_function != CatBoostRegressionLoss::MultiRmse {
            return Err(RustMlError::InvalidParameter(
                "fit_multi needs loss_function MultiRMSE".into(),
            ));
        }
        if y.ncols() == 0 {
            return Err(RustMlError::InvalidParameter(
                "MultiRMSE needs at least two target columns (RMSE for one)".into(),
            ));
        }
        if let Some(e) = &eval {
            if e.y.ncols() != y.ncols() {
                return Err(RustMlError::ShapeMismatch(format!(
                    "the eval set has {} targets, y {}",
                    e.y.ncols(),
                    y.ncols()
                )));
            }
        }
        let checked = eval.map(|e| CatBoostEvalSet {
            x: e.x,
            y: e.y.index_axis_move(ndarray::Axis(1), 0),
            sample_weight: e.sample_weight,
        });
        let use_best = self.use_best_model_with(checked.as_ref(), x.ncols())?;
        let eval = eval.map(|e| EvalSet {
            x: e.x,
            target: e.y.iter().map(|&t| f64::from(t as f32)).collect(),
            weights: e
                .sample_weight
                .map_or_else(|| vec![1.0; e.y.nrows()], f32_values),
            groups: Groups::default(),
            pfound: false,
        });
        self.fit_targets(x, y, sample_weight, eval, use_best, None)
    }

    /// Fit on the target columns `y` (one, or MultiRMSE's k).
    fn fit_targets(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView2<f64>,
        sample_weight: Option<ArrayView1<f64>>,
        eval_set: Option<EvalSet<'_>>,
        use_best: bool,
        queries: Option<Vec<usize>>,
    ) -> Result<FittedCatBoostRegressor> {
        let p = self.train_params();
        validate(
            x,
            y.column(0),
            sample_weight,
            &p,
            self.border_count.unwrap_or(254),
        )?;
        self.check_monotone(x.ncols())?;
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
        let loss = self.loss(y.ncols());
        let mut rng = CatRng::new(self.random_seed);
        // Pairs come from the given order (then follow the shuffle).
        let mut groups = match queries {
            Some(q) => groups_of(loss, q, &f32_values(y.column(0))),
            None => Groups::default(),
        };
        // Grouped rows shuffle by query (`ShuffleLearnDataIfNeeded`).
        let order = if groups.queries.is_empty() {
            self.learn_order(x.nrows(), &mut rng)
        } else {
            self.learn_order(0, &mut rng).map(|_| {
                let (order, bounds) = rng.group_permutation(&groups.queries);
                groups = groups.permuted(&order, bounds);
                order
            })
        };
        let shuffled = order.map(|order| {
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
        let weights = sample_weight.map_or_else(|| vec![1.0; y.nrows()], f32_values);
        let columns: Vec<Vec<f64>> = y.columns().into_iter().map(f32_values).collect();
        // Row-major targets.
        let target: Vec<f64> = y.iter().map(|&t| f64::from(t as f32)).collect();
        let target0 = &columns[0];
        self.check_loss(loss, &target)?;
        // RMSEWithUncertainty always starts from the target's mean and log
        // standard deviation (CatBoost's `CalcOptimumConstApprox`).
        let start_sigma = (loss == Loss::RmseWithUncertainty).then(|| {
            let mean = weighted_target_average(target0, &weights);
            0.5 * f64::from(weighted_target_variance(target0, &weights, mean)).ln()
        });
        let bias = if loss == Loss::RmseWithUncertainty {
            f64::from(weighted_target_average(target0, &weights))
        } else if !self.boost_from_average_for(loss) {
            0.0
        } else if let Loss::Quantile { alpha, delta, .. } = loss {
            let t: Vec<f32> = target0.iter().map(|&v| v as f32).collect();
            let w: Vec<f32> = weights.iter().map(|&v| v as f32).collect();
            f64::from(weighted_quantile(&t, &w, alpha, delta))
        } else {
            f64::from(weighted_target_average(target0, &weights))
        };
        // MultiRMSE starts each target from its RMSE optimum.
        let multi_bias: Vec<f64> = if matches!(loss, Loss::MultiRmse(_)) {
            columns[1..]
                .iter()
                .map(|c| {
                    if self.boost_from_average_for(loss) {
                        f64::from(weighted_target_average(c, &weights))
                    } else {
                        0.0
                    }
                })
                .collect()
        } else {
            Vec::new()
        };
        let classes_of = target_classes(target0);
        let cat = CategoricalParams {
            features: &self.cat_features,
            one_hot_max_size: self.one_hot_max_size_for(loss),
            classes: Some((&classes_of, 2)),
            max_ctr_complexity: self.max_ctr_complexity(),
        };
        let q = quantize(x, self.border_count_for(loss), cat)?;
        self.check_grow_policy(q.ctr.is_some())?;
        let (bootstrap, subsample) = self.sampling(loss, x.nrows());
        // CatBoost's YetiRank default `l2_leaf_reg` is 0 (made 1e-20).
        // PairLogitPairwise's is 5.
        let l2 = match (self.l2_leaf_reg, loss) {
            (None, Loss::YetiRank { .. } | Loss::YetiRankPairwise { .. }) => f64::from(1e-20f32),
            (None, Loss::PairLogitPairwise) => 5.0,
            _ => p.l2_leaf_reg,
        };
        // Pairwise scoring has no score noise by default.
        let random_strength = match self.random_strength {
            None if loss.pairwise_scoring() => 0.0,
            _ => p.random_strength,
        };
        let p = TrainParams {
            l2_leaf_reg: l2,
            random_strength,
            pairwise_prior: f64::from(self.bayesian_matrix_reg as f32),
            learning_rate: f64::from(self.learning_rate_for(loss, x.nrows(), use_best) as f32),
            leaf_iterations: self.leaf_iterations(loss, x.ncols()),
            leaf_estimation: self.leaf_method(loss),
            bootstrap,
            subsample,
            use_best_model: use_best,
            ..p
        };
        let start: Vec<f64> = std::iter::once(bias)
            .chain(start_sigma)
            .chain(multi_bias.iter().copied())
            .collect();
        let b = crate::lgbm::with_fit_threads(self.thread_count, || {
            boost(
                &q,
                &target,
                &weights,
                &start,
                loss,
                p,
                rng,
                eval_set.as_ref(),
                &groups,
            )
        });
        Ok(FittedCatBoostRegressor {
            trees: b.trees,
            ctrs: b.ctrs,
            bias: b.bias[0],
            log_sigma_bias: start_sigma.map(|_| b.bias[1]),
            multi_bias: if multi_bias.is_empty() {
                Vec::new()
            } else {
                b.bias[1..].to_vec()
            },
            eval_metrics: b.eval_metrics,
            best_iteration: b.best_iteration,
            exp_prediction: loss.exp_prediction(),
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
        if !self.multi_bias.is_empty() {
            return Err(RustMlError::InvalidParameter(
                "a MultiRMSE model predicts one column per target: use predict_multi".into(),
            ));
        }
        let raw = raw_scores(&self.trees, &self.ctrs, &self.biases(), x);
        let raw = raw.column(0).to_owned();
        Ok(if self.exp_prediction {
            raw.mapv(f64::exp)
        } else {
            raw
        })
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
    #[serde(default)]
    eval_metrics: Vec<f64>,
    #[serde(default)]
    best_iteration: Option<usize>,
    /// MultiClassOneVsAll: a sigmoid per class.
    #[serde(default)]
    one_vs_all: bool,
}

impl FittedCatBoostClassifier {
    /// The eval set's metric after every iteration (`fit_with_eval`).
    pub fn eval_metrics(&self) -> &[f64] {
        &self.eval_metrics
    }
    /// The iteration with the best eval metric (0-based).
    pub fn best_iteration(&self) -> Option<usize> {
        self.best_iteration
    }
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
        if self.one_vs_all {
            return Ok(raw.mapv(|v| 1.0 / (1.0 + (-v).exp())));
        }
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
        self.fit_inner(x, y, sample_weight, None)
    }
}

impl CatBoostClassifier {
    /// Fit with an eval set (CatBoost's `eval_set`): see
    /// [`CatBoostRegressor::fit_with_eval`]. Its labels must be classes of
    /// `y`; the class weights weight it too.
    pub fn fit_with_eval(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        sample_weight: Option<ArrayView1<f64>>,
        eval: CatBoostEvalSet<'_>,
    ) -> Result<FittedCatBoostClassifier> {
        self.fit_inner(x, y, sample_weight, Some(eval))
    }

    fn fit_inner(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        sample_weight: Option<ArrayView1<f64>>,
        eval: Option<CatBoostEvalSet<'_>>,
    ) -> Result<FittedCatBoostClassifier> {
        let p = self.train_params();
        let use_best = self.use_best_model_with(eval.as_ref(), x.ncols())?;
        validate(x, y, sample_weight, &p, self.border_count.unwrap_or(254))?;
        self.check_monotone(x.ncols())?;
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
        let cross_entropy = self.loss_function == CatBoostClassifierLoss::CrossEntropy;
        if cross_entropy {
            if y.iter().any(|v| !(0.0..=1.0).contains(v)) {
                return Err(RustMlError::InvalidParameter(
                    "CrossEntropy needs targets in [0, 1]".into(),
                ));
            }
            if !self.cat_features.is_empty()
                || self.class_weights.is_some()
                || self.auto_class_weights.is_some()
            {
                return Err(RustMlError::InvalidParameter(
                    "CrossEntropy supports neither categorical features nor class weights".into(),
                ));
            }
        }
        let mut classes: Vec<f64> = if cross_entropy {
            vec![0.0, 1.0]
        } else {
            y.to_vec()
        };
        classes.sort_by(f64::total_cmp);
        classes.dedup();
        if !(2..=255).contains(&classes.len()) {
            return Err(RustMlError::InvalidParameter(format!(
                "CatBoostClassifier needs 2 to 255 classes, got {}",
                classes.len()
            )));
        }
        let one_vs_all = self.loss_function == CatBoostClassifierLoss::MultiClassOneVsAll;
        let multiclass = classes.len() > 2 || one_vs_all;
        if multiclass && !p.monotone.is_empty() {
            return Err(RustMlError::InvalidParameter(
                "monotone constraints are not supported for more than two classes".into(),
            ));
        }
        if self.leaf_estimation_method == Some(CatBoostLeafEstimation::Exact) {
            return Err(RustMlError::InvalidParameter(
                "Exact leaf estimation is only available for MAE and Quantile".into(),
            ));
        }
        let boost_from_average = self.boost_from_average.unwrap_or(false);
        if multiclass && boost_from_average {
            return Err(RustMlError::InvalidParameter(
                "boost_from_average is not supported for more than two classes".into(),
            ));
        }
        // The class index of every row (Logloss: 0 / 1, MultiClass: 0..k),
        // or the float32 probability (CrossEntropy).
        let target: Vec<f64> = if cross_entropy {
            y.iter().map(|&v| f64::from(v as f32)).collect()
        } else {
            y.iter()
                .map(|v| classes.partition_point(|c| c < v) as f64)
                .collect()
        };
        let classes_of: Vec<u8> = target.iter().map(|&t| t as u8).collect();
        let weights = sample_weight.map_or_else(|| vec![1.0; y.len()], f32_values);
        let class_weights = self.class_weight_vector(&classes_of, classes.len(), &weights)?;
        let weights = apply_class_weights(weights, &classes_of, class_weights.as_deref());
        let eval_set = match eval {
            None => None,
            Some(e) => {
                let mut target = Vec::with_capacity(e.y.len());
                for &v in e.y.iter() {
                    if cross_entropy {
                        if !(0.0..=1.0).contains(&v) {
                            return Err(RustMlError::InvalidParameter(
                                "CrossEntropy needs eval targets in [0, 1]".into(),
                            ));
                        }
                        target.push(f64::from(v as f32));
                        continue;
                    }
                    match classes.binary_search_by(|c| c.total_cmp(&v)) {
                        Ok(c) => target.push(c as f64),
                        Err(_) => {
                            return Err(RustMlError::InvalidParameter(format!(
                                "the eval set has the label {v}, not in y"
                            )))
                        }
                    }
                }
                let of: Vec<u8> = target.iter().map(|&t| t as u8).collect();
                let w = e
                    .sample_weight
                    .map_or_else(|| vec![1.0; e.y.len()], f32_values);
                Some(EvalSet {
                    x: e.x,
                    weights: apply_class_weights(w, &of, class_weights.as_deref()),
                    target,
                    groups: Groups::default(),
                    pfound: false,
                })
            }
        };
        let (loss, bias) = if one_vs_all {
            (
                Loss::MultiClassOneVsAll(classes.len()),
                vec![0.0; classes.len()],
            )
        } else if multiclass {
            (Loss::MultiClass(classes.len()), vec![0.0; classes.len()])
        } else if boost_from_average {
            let p1 = f64::from(weighted_target_average(&target, &weights));
            (Loss::Logloss, vec![(p1 / (1.0 - p1)).ln()])
        } else {
            (Loss::Logloss, vec![0.0])
        };
        let cat = CategoricalParams {
            features: &self.cat_features,
            one_hot_max_size: self.one_hot_max_size_for(loss),
            classes: Some((&classes_of, classes.len())),
            max_ctr_complexity: self.max_ctr_complexity(),
        };
        let q = quantize(x, self.border_count_for(loss), cat)?;
        self.check_grow_policy(q.ctr.is_some())?;
        let (bootstrap, subsample) = self.sampling(loss, x.nrows());
        // CatBoost has no automatic learning rate for CrossEntropy.
        let learning_rate = match (cross_entropy, self.learning_rate) {
            (true, None) => 0.03,
            _ => self.learning_rate_for(loss, x.nrows(), use_best),
        };
        let p = TrainParams {
            learning_rate: f64::from(learning_rate as f32),
            leaf_iterations: self.leaf_iterations(loss, x.ncols()),
            bootstrap,
            subsample,
            ..p
        };
        let p = TrainParams {
            use_best_model: use_best,
            ..p
        };
        let b = crate::lgbm::with_fit_threads(self.thread_count, || {
            boost(
                &q,
                &target,
                &weights,
                &bias,
                loss,
                p,
                rng,
                eval_set.as_ref(),
                &Groups::default(),
            )
        });
        Ok(FittedCatBoostClassifier {
            trees: b.trees,
            ctrs: b.ctrs,
            bias: b.bias,
            eval_metrics: b.eval_metrics,
            best_iteration: b.best_iteration,
            borders: q.borders,
            classes,
            n_features: x.ncols(),
            one_vs_all,
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
        let tails: Vec<(usize, usize)> = body_tails(10, 2.0, &w, &[])
            .iter()
            .map(|t| (t.0, t.1))
            .collect();
        assert_eq!(tails, vec![(1, 2), (2, 4), (4, 8), (8, 10)]);
        let w = vec![1.0; 1200];
        assert_eq!(body_tails(1200, 2.0, &w, &[])[0].0, 24);
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
        // Losses: their parameters, targets and leaf methods.
        let fit = |l: CatBoostRegressionLoss, y: &Array1<f64>| {
            CatBoostRegressor::new()
                .with_iterations(2)
                .with_loss_function(l)
                .fit(&x, y)
        };
        assert!(fit(CatBoostRegressionLoss::Poisson, &y.mapv(|v| v - 10.0)).is_err());
        assert!(fit(
            CatBoostRegressionLoss::Tweedie {
                variance_power: 2.5
            },
            &y
        )
        .is_err());
        assert!(fit(CatBoostRegressionLoss::Quantile { alpha: 1.5 }, &y).is_err());
        assert!(CatBoostRegressor::new()
            .with_leaf_estimation_method(CatBoostLeafEstimation::Exact)
            .fit(&x, &y)
            .is_err());
        assert!(CatBoostRegressor::new()
            .with_loss_function(CatBoostRegressionLoss::Poisson)
            .with_boost_from_average(true)
            .fit(&x, &y)
            .is_err());
        // Depthwise / Lossguide: plain boosting, numeric and one-hot features.
        for g in [CatBoostGrowPolicy::Depthwise, CatBoostGrowPolicy::Lossguide] {
            assert!(CatBoostRegressor::new()
                .with_iterations(2)
                .with_grow_policy(g)
                .with_boosting_type(CatBoostBoostingType::Ordered)
                .fit(&x, &y)
                .is_err());
            let m = CatBoostRegressor::new()
                .with_iterations(3)
                .with_grow_policy(g)
                .fit(&x, &y)
                .unwrap();
            let bytes = anofox_ml_core::persistence::to_binary_bytes(&m).unwrap();
            let back: FittedCatBoostRegressor =
                anofox_ml_core::persistence::from_binary_bytes(&bytes).unwrap();
            assert_eq!(back.predict(&x).unwrap(), m.predict(&x).unwrap());
        }
        let counts = y.mapv(|v| (v * 3.0).abs().round());
        let m = fit(CatBoostRegressionLoss::Poisson, &counts).unwrap();
        assert!(m.predict(&x).unwrap().iter().all(|&p| p > 0.0));
    }

    #[test]
    fn monotone_constraints_hold_and_are_validated() {
        let (x, y) = data(300);
        let m = CatBoostRegressor::new()
            .with_iterations(40)
            .with_learning_rate(0.3)
            .with_depth(4)
            .with_monotone_constraints(vec![-1, 1, 0])
            .fit(&x, &y)
            .unwrap();
        // Along a constrained feature, with the others fixed.
        for row in [0usize, 7, 42] {
            let preds: Vec<f64> = (0..60)
                .map(|t| {
                    let mut r = x.row(row).to_owned().insert_axis(ndarray::Axis(0));
                    r[[0, 0]] = t as f64 / 5.0;
                    r[[0, 1]] = 12.0 - t as f64 / 5.0;
                    m.predict(&r).unwrap()[0]
                })
                .collect();
            assert!(preds.windows(2).all(|w| w[1] <= w[0] + 1e-12), "{preds:?}");
        }
        let fit = |c: Vec<i8>| {
            CatBoostRegressor::new()
                .with_iterations(3)
                .with_monotone_constraints(c)
                .fit(&x, &y)
        };
        assert!(fit(vec![1, 0]).is_err());
        assert!(fit(vec![2, 0, 0]).is_err());
        let labels = y.mapv(|v| (v * 3.0).rem_euclid(3.0).floor());
        assert!(CatBoostClassifier::new()
            .with_iterations(3)
            .with_monotone_constraints(vec![1, 0, 0])
            .fit(&x, &labels)
            .is_err());
        assert!(CatBoostClassifier::new()
            .with_iterations(3)
            .with_class_weights(vec![1.0])
            .fit(&x, &labels)
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

    #[test]
    fn multi_rmse_fits_a_target_matrix_and_checks_its_use() {
        let (x, y) = data(120);
        let ys = ndarray::stack(ndarray::Axis(1), &[y.view(), (&y * -2.0 + 1.0).view()]).unwrap();
        let multi = CatBoostRegressor::new()
            .with_iterations(40)
            .with_learning_rate(0.3)
            .with_loss_function(CatBoostRegressionLoss::MultiRmse);
        let f = multi.fit_multi(x.view(), ys.view(), None).unwrap();
        assert_eq!(f.n_targets(), 2);
        let pred = f.predict_multi(&x).unwrap();
        assert_eq!(pred.dim(), (120, 2));
        // Each column follows its own target.
        for c in 0..2 {
            let e = (&pred.column(c) - &ys.column(c))
                .mapv(|e| e * e)
                .mean()
                .unwrap();
            assert!(e < 0.5 * ys.column(c).var(0.0), "{c}: {e}");
        }
        assert!(f.predict(&x).is_err());
        let bytes = anofox_ml_core::persistence::to_binary_bytes(&f).unwrap();
        let back: FittedCatBoostRegressor =
            anofox_ml_core::persistence::from_binary_bytes(&bytes).unwrap();
        assert_eq!(back.predict_multi(&x).unwrap(), pred);

        // MultiRMSE needs fit_multi, two columns, no categorical features;
        // fit_multi needs MultiRMSE.
        assert!(multi.fit(&x, &y).is_err());
        assert!(multi
            .fit_multi(x.view(), ys.slice(ndarray::s![.., ..1]), None)
            .is_err());
        assert!(multi
            .clone()
            .with_cat_features(vec![2])
            .fit_multi(x.view(), ys.view(), None)
            .is_err());
        let rmse = CatBoostRegressor::new().with_iterations(5);
        assert!(rmse.fit_multi(x.view(), ys.view(), None).is_err());
        assert!(rmse.fit(&x, &y).unwrap().predict_multi(&x).is_err());
    }

    #[test]
    fn query_rmse_needs_and_checks_query_groups() {
        let (x, y) = data(60);
        let groups = [10, 20, 5, 25];
        let q = CatBoostRegressor::new()
            .with_iterations(10)
            .with_loss_function(CatBoostRegressionLoss::QueryRmse);
        let f = q
            .fit_with_groups(x.view(), y.view(), None, &groups)
            .unwrap();
        assert_eq!(f.n_estimators(), 10);
        // A query-wise shift of the targets leaves the model unchanged.
        let shifted: Array1<f64> = y
            .iter()
            .enumerate()
            .map(|(i, &v)| v + if i < 30 { 3.0 } else { -1.0 })
            .collect();
        let g = q
            .fit_with_groups(x.view(), shifted.view(), None, &groups)
            .unwrap();
        let (a, b) = (f.predict(&x).unwrap(), g.predict(&x).unwrap());
        assert!(a.iter().zip(&b).all(|(a, b)| (a - b).abs() < 1e-5));
        assert!(q.fit(&x, &y).is_err());
        assert!(q
            .fit_with_groups(x.view(), y.view(), None, &[30, 29])
            .is_err());
        assert!(q
            .fit_with_groups(x.view(), y.view(), None, &[60, 0])
            .is_err());
        let rmse = CatBoostRegressor::new().with_iterations(3);
        assert!(rmse
            .fit_with_groups(x.view(), y.view(), None, &groups)
            .is_err());
        // PairLogit: shift-invariant leaves, no sample weights.
        let pl = CatBoostRegressor::new()
            .with_iterations(5)
            .with_loss_function(CatBoostRegressionLoss::PairLogit);
        let labels = y.mapv(|v| (v * 2.0).round());
        assert!(pl
            .fit_with_groups(x.view(), labels.view(), None, &groups)
            .is_ok());
        assert!(pl
            .fit_with_groups(x.view(), labels.view(), Some(labels.view()), &groups)
            .is_err());
        // YetiRank: plain boosting, one Newton step; NDCG on an eval set.
        let yr = CatBoostRegressor::new()
            .with_iterations(5)
            .with_loss_function(CatBoostRegressionLoss::YetiRank {
                permutations: 3,
                decay: 0.85,
            });
        assert!(yr
            .fit_with_groups(x.view(), labels.view(), None, &groups)
            .is_ok());
        assert!(yr
            .clone()
            .with_boosting_type(CatBoostBoostingType::Ordered)
            .fit_with_groups(x.view(), labels.view(), None, &groups)
            .is_err());
        let eval = CatBoostEvalSet {
            x: x.view(),
            y: labels.view(),
            sample_weight: None,
        };
        let f = yr
            .fit_with_groups_eval(x.view(), labels.view(), None, &groups, eval, &groups)
            .unwrap();
        assert_eq!(f.eval_metrics().len(), 5);
        // Pairwise scoring: plain symmetric trees on numeric features.
        let pw = CatBoostRegressor::new()
            .with_iterations(5)
            .with_loss_function(CatBoostRegressionLoss::PairLogitPairwise);
        assert!(pw
            .fit_with_groups(x.view(), labels.view(), None, &groups)
            .is_ok());
        for bad in [
            pw.clone().with_grow_policy(CatBoostGrowPolicy::Depthwise),
            pw.clone().with_cat_features(vec![2]),
            pw.clone()
                .with_leaf_estimation_method(CatBoostLeafEstimation::Newton),
            pw.clone().with_depth(7),
        ] {
            assert!(bad
                .fit_with_groups(x.view(), labels.view(), None, &groups)
                .is_err());
        }
    }
}
