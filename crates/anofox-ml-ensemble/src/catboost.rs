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
//! randomness switched off (`random_strength=0`, no bootstrap,
//! `has_time=True`) the models agree with CatBoost's.
//!
//! Leaves take one Newton (or Gradient) step. CatBoost's Logloss defaults
//! to 10 steps from 200 iterations or 20 features on; pass
//! `leaf_estimation_iterations=1` to CatBoost to compare.

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

    let mut q = Quantized {
        columns,
        borders,
        ctr: None,
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

/// The derivatives a tree's splits are scored on. Plain boosting: all rows
/// (`body = 0`). Ordered boosting: one part per body / tail of the fold,
/// whose leaf estimates come from the body rows and are scored on the tail
/// rows (CatBoost's `CalcStatsKernel` and `UpdateScoreBinKernelOrdered`).
struct ScorePart<'a> {
    /// `w · der1` of the rows `0..tail`, `k` per row.
    wder: &'a [f64],
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

fn combination_columns(
    q: &Quantized,
    splits: &[(ColumnRef, usize)],
    cache: &mut CombinationCache,
) -> Vec<std::sync::Arc<Vec<Column>>> {
    let Some(src) = &q.ctr else {
        return Vec::new();
    };
    if src.max_complexity < 2 || splits.is_empty() {
        return Vec::new();
    }
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
    conditions.floats.sort_unstable();
    conditions.one_hots.sort_unstable();
    seen.insert(0, conditions);
    let mut added: Vec<Projection> = Vec::new();
    for base in seen.iter().filter(|p| !p.is_empty()) {
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
    added
        .into_iter()
        .map(|projection| {
            cache
                .entry(projection.clone())
                .or_insert_with(|| std::sync::Arc::new(q.projection_columns(projection)))
                .clone()
        })
        .collect()
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
) -> (Vec<(ColumnRef, usize)>, Vec<u32>) {
    use rayon::prelude::*;
    let n = w.len();
    let mut splits: Vec<(ColumnRef, usize)> = Vec::new();
    let mut leaf_of = vec![0u32; n];
    for d in 0..depth {
        let leaves = 1usize << d;
        let combinations = combination_columns(q, &splits, cache);
        let candidates: Vec<ColumnRef> = (0..q.columns.len())
            .map(ColumnRef::Base)
            .chain(
                combinations
                    .iter()
                    .flat_map(|c| (0..c.len()).map(|i| ColumnRef::Combination(c.clone(), i))),
            )
            .collect();
        // CatBoost's `GetCatFeatureWeight`: a CTR type not yet in the model
        // scores `(1 + combinations / max combinations)^-model_size_reg`
        // times less, the maximum over this level's CTR candidates.
        let max_categories = candidates
            .iter()
            .filter_map(|c| c.get(q).ctr_key.as_ref().map(|(_, k)| *k))
            .max()
            .unwrap_or(1)
            .max(1);
        // Best (score, split) per column, the first best split winning.
        let per_column: Vec<Option<(f64, usize)>> = candidates
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
                        let cell = leaf_of[i] as usize * nb + column.buckets[i] as usize;
                        for d in 0..k {
                            let h = &mut hist[(pi * k + d) * leaves * nb + cell];
                            if i < part.body {
                                h.sd += part.wder[i * k + d];
                                h.count += w[i];
                            } else {
                                h.swd += part.wder[i * k + d];
                                h.sw += w[i];
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
                let score = |s: usize| match score_function {
                    CatBoostScoreFunction::Cosine => num[s] / den[s].sqrt(),
                    CatBoostScoreFunction::L2 => num[s],
                };
                let mut best: Option<(f64, usize)> = None;
                for s in 0..n_splits {
                    let sc = score(s);
                    if best.is_none_or(|(b, _)| sc > b) {
                        best = Some((sc, s));
                    }
                }
                best
            })
            .collect();
        let mut best: Option<(f64, usize, usize)> = None;
        for (c, cand) in per_column.into_iter().enumerate() {
            if let Some((mut sc, s)) = cand {
                if let Some((key, k)) = &candidates[c].get(q).ctr_key {
                    if !used_ctrs.contains(key) {
                        sc *= (1.0 + *k as f64 / max_categories as f64).powf(-model_size_reg);
                    }
                }
                if best.is_none_or(|(b, _, _)| sc > b) {
                    best = Some((sc, c, s));
                }
            }
        }
        let Some((_, c, s)) = best else {
            break;
        };
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

/// Leaf deltas (Newton or Gradient, `k` per leaf) from the rows `0..rows`,
/// times the learning rate.
fn leaf_deltas(
    leaf_of: &[u32],
    n_leaves: usize,
    ders: &Ders,
    weights: &[f64],
    rows: usize,
    (sum_w, count): (f64, f64),
    p: &TrainParams,
) -> Vec<f64> {
    let k = ders.k;
    let sums = LeafSums::of(leaf_of, n_leaves, ders, weights, rows);
    let mut out = vec![0.0; n_leaves * k];
    for (leaf, o) in out.chunks_mut(k).enumerate() {
        sums.delta(leaf, p.l2_leaf_reg, sum_w, count, p, o);
        o.iter_mut().for_each(|v| *v *= p.learning_rate);
    }
    out
}

/// Boosting of oblivious trees from the constant `bias` (one per
/// approximation). Plain: the trees are scored and fitted on all rows.
/// Ordered (CatBoost's ordered boosting): the splits are scored on the body /
/// tail prefixes of the rows, whose approximations only ever saw the rows
/// before their tail, and the leaf values of the model come from all rows.
fn boost(
    q: &Quantized,
    target: &[f64],
    weights: &[f64],
    bias: &[f64],
    loss: Loss,
    p: TrainParams,
) -> (Vec<ObliviousTree>, CtrModel) {
    let n = target.len();
    let k = loss.dims();
    let sum_w: f64 = weights.iter().sum();
    let scaled_l2 = p.l2_leaf_reg * (sum_w / n as f64);
    let tails = if p.ordered {
        body_tails(n, p.fold_len_multiplier, weights)
    } else {
        Vec::new()
    };
    let start = |rows: usize| -> Vec<f64> { bias.iter().copied().cycle().take(rows * k).collect() };
    let mut fold_approx: Vec<Vec<f64>> = tails.iter().map(|&(_, t, _)| start(t)).collect();
    let mut approx = start(n);
    let mut trees = Vec::with_capacity(p.iterations);
    let mut used_ctrs = std::collections::HashSet::new();
    let mut cache = CombinationCache::new();
    let mut ctrs = CtrModel::default();
    for _ in 0..p.iterations {
        let ders = loss.all_ders(target, &approx, weights, n);
        let (splits, leaf_of) = if p.ordered {
            let fold_ders: Vec<Ders> = tails
                .iter()
                .zip(&fold_approx)
                .map(|(&(_, t, _), a)| loss.all_ders(target, a, weights, t))
                .collect();
            let parts: Vec<ScorePart<'_>> = tails
                .iter()
                .zip(&fold_ders)
                .map(|(&(body, tail, bw), d)| ScorePart {
                    wder: &d.der,
                    body,
                    tail,
                    scaled_l2: p.l2_leaf_reg * (bw / body as f64),
                })
                .collect();
            let grown = grow_tree(
                q,
                &parts,
                k,
                weights,
                true,
                p.depth,
                p.score_function,
                p.model_size_reg,
                &mut used_ctrs,
                &mut cache,
            );
            let n_leaves = 1usize << grown.0.len();
            let leaf_of = &grown.1;
            for ((&(body, tail, bw), a), d) in tails.iter().zip(&mut fold_approx).zip(&fold_ders) {
                let body_stats = (bw, body as f64);
                if k == 1 {
                    // Each prefix's approximation moves by the leaf values
                    // of its body rows.
                    let deltas = leaf_deltas(leaf_of, n_leaves, d, weights, body, body_stats, &p);
                    for i in 0..tail {
                        a[i] += deltas[leaf_of[i] as usize];
                    }
                } else {
                    // Multiclass (CatBoost's `CalcApproxDeltaMulti`): the
                    // body rows by the leaf values of the body, every tail
                    // row by those of the body and the tail rows up to it.
                    let mut sums = LeafSums::of(leaf_of, n_leaves, d, weights, body);
                    let mut deltas = vec![0.0; n_leaves * k];
                    for (leaf, o) in deltas.chunks_mut(k).enumerate() {
                        sums.delta(leaf, p.l2_leaf_reg, bw, body as f64, &p, o);
                    }
                    let mut row_delta = vec![0.0; (tail - body) * k];
                    for i in body..tail {
                        let leaf = leaf_of[i] as usize;
                        sums.add(leaf, d, i, weights[i]);
                        let o = &mut row_delta[(i - body) * k..(i - body + 1) * k];
                        sums.delta(leaf, p.l2_leaf_reg, bw, body as f64, &p, o);
                    }
                    for i in 0..tail {
                        let delta = if i < body {
                            let leaf = leaf_of[i] as usize;
                            &deltas[leaf * k..(leaf + 1) * k]
                        } else {
                            &row_delta[(i - body) * k..(i - body + 1) * k]
                        };
                        for dim in 0..k {
                            a[i * k + dim] += delta[dim] * p.learning_rate;
                        }
                    }
                }
            }
            grown
        } else {
            let parts = [ScorePart {
                wder: &ders.der,
                body: 0,
                tail: n,
                scaled_l2,
            }];
            grow_tree(
                q,
                &parts,
                k,
                weights,
                false,
                p.depth,
                p.score_function,
                p.model_size_reg,
                &mut used_ctrs,
                &mut cache,
            )
        };
        let n_leaves = 1usize << splits.len();
        let leaf_values = leaf_deltas(&leaf_of, n_leaves, &ders, weights, n, (sum_w, n as f64), &p);
        for i in 0..n {
            let leaf = leaf_of[i] as usize;
            for d in 0..k {
                approx[i * k + d] += leaf_values[leaf * k + d];
            }
        }
        trees.push(ObliviousTree {
            splits: splits
                .iter()
                .map(|(c, s)| c.get(q).tree_split(*s, &mut ctrs))
                .collect(),
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
            pub fn with_max_ctr_complexity(mut self, n: usize) -> Self {
                self.max_ctr_complexity = Some(n);
                self
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
        if self.max_ctr_complexity == Some(0) {
            return Err(RustMlError::InvalidParameter(
                "max_ctr_complexity must be at least 1".into(),
            ));
        }
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
        let (trees, ctrs) = boost(&q, &target, &weights, &[bias], Loss::Rmse, p);
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
        if self.max_ctr_complexity == Some(0) {
            return Err(RustMlError::InvalidParameter(
                "max_ctr_complexity must be at least 1".into(),
            ));
        }
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
        let (trees, ctrs) = boost(&q, &target, &weights, &bias, loss, p);
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
