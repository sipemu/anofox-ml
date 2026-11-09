//! scikit-learn's random-splitter tree, reproduced draw for draw: the
//! xorshift `our_rand_r` seeded from `RandomState.randint(0, RAND_R_MAX)`,
//! `RandomSplitter.node_split_random` (Fisher–Yates feature draws with the
//! constant-feature bookkeeping shared across nodes, uniform thresholds in
//! `[min, max)` of the float32 feature values, in-place sample
//! partitioning), `DepthFirstTreeBuilder` and the squared-error, Gini and
//! entropy criteria with sklearn's left/right accumulation. Backs the
//! single `ExtraTree*` estimators and `RandomTreesEmbedding`.

use anofox_ml_core::numpy_random::RandomState;
use serde::{Deserialize, Serialize};

const RAND_R_MAX: u32 = 0x7FFF_FFFF;
const FEATURE_THRESHOLD: f64 = 1e-7;
const EPSILON: f64 = f64::EPSILON;

/// sklearn's `our_rand_r` / `rand_int` / `rand_uniform`.
struct RandR(u32);

impl RandR {
    fn next(&mut self) -> u32 {
        if self.0 == 0 {
            self.0 = 1;
        }
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0 % (RAND_R_MAX + 1)
    }

    fn int(&mut self, low: usize, high: usize) -> usize {
        low + (self.next() as usize) % (high - low)
    }

    fn uniform(&mut self, low: f64, high: f64) -> f64 {
        ((high - low) * f64::from(self.next()) / f64::from(RAND_R_MAX)) + low
    }
}

/// The split quality measure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SkCriterion {
    SquaredError,
    Gini,
    Entropy,
}

/// Growth limits (sklearn's resolved `max_depth`, `min_samples_split`, …).
#[derive(Debug, Clone, Copy)]
pub struct SkLimits {
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub min_weight_leaf: f64,
    pub max_features: usize,
    pub min_impurity_decrease: f64,
}

/// One node; leaves have `left == usize::MAX`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkNode {
    pub left: usize,
    pub right: usize,
    pub feature: usize,
    pub threshold: f64,
    /// Mean target (regression) or class fractions (classification).
    pub value: Vec<f64>,
}

/// A fitted tree, nodes in sklearn's depth-first order (node ids equal
/// sklearn's `tree_` ids, so `apply` gives the same leaf indices).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkTree {
    pub nodes: Vec<SkNode>,
    pub n_features: usize,
}

impl SkTree {
    pub const LEAF: usize = usize::MAX;

    /// Leaf id of a row (sklearn compares the float32 value with the
    /// float64 threshold).
    pub fn apply_row(&self, row: &[f64]) -> usize {
        let mut n = 0;
        while self.nodes[n].left != Self::LEAF {
            let node = &self.nodes[n];
            let v = f64::from(row[node.feature] as f32);
            n = if v <= node.threshold {
                node.left
            } else {
                node.right
            };
        }
        n
    }

    pub fn value_of(&self, row: &[f64]) -> &[f64] {
        &self.nodes[self.apply_row(row)].value
    }
}

/// Criterion state over `samples[start..end]`.
struct Crit<'a> {
    kind: SkCriterion,
    y: &'a [f64],
    w: Option<&'a [f64]>,
    n_classes: usize,
    weighted_n_samples: f64,
    start: usize,
    end: usize,
    pos: usize,
    wn: f64,
    wl: f64,
    wr: f64,
    sum_total: Vec<f64>,
    sum_left: Vec<f64>,
    sum_right: Vec<f64>,
    sq_sum_total: f64,
}

impl<'a> Crit<'a> {
    fn weight(&self, i: usize) -> f64 {
        self.w.map_or(1.0, |w| w[i])
    }

    fn width(&self) -> usize {
        if self.kind == SkCriterion::SquaredError {
            1
        } else {
            self.n_classes
        }
    }

    fn init(&mut self, samples: &[usize], start: usize, end: usize) {
        self.start = start;
        self.end = end;
        self.sum_total = vec![0.0; self.width()];
        self.sq_sum_total = 0.0;
        self.wn = 0.0;
        for &i in &samples[start..end] {
            let w = self.weight(i);
            if self.kind == SkCriterion::SquaredError {
                let wy = w * self.y[i];
                self.sum_total[0] += wy;
                self.sq_sum_total += wy * self.y[i];
            } else {
                self.sum_total[self.y[i] as usize] += w;
            }
            self.wn += w;
        }
        self.reset();
    }

    fn reset(&mut self) {
        self.pos = self.start;
        self.wl = 0.0;
        self.wr = self.wn;
        self.sum_left = vec![0.0; self.width()];
        self.sum_right.clone_from(&self.sum_total);
    }

    fn reverse_reset(&mut self) {
        self.pos = self.end;
        self.wl = self.wn;
        self.wr = 0.0;
        self.sum_left.clone_from(&self.sum_total);
        self.sum_right = vec![0.0; self.width()];
    }

    fn add(&mut self, i: usize, sign: f64) {
        let w = self.weight(i);
        if self.kind == SkCriterion::SquaredError {
            self.sum_left[0] += sign * w * self.y[i];
        } else {
            self.sum_left[self.y[i] as usize] += sign * w;
        }
        self.wl += sign * w;
    }

    fn update(&mut self, samples: &[usize], new_pos: usize) {
        if new_pos - self.pos <= self.end - new_pos {
            for p in self.pos..new_pos {
                self.add(samples[p], 1.0);
            }
        } else {
            self.reverse_reset();
            for p in (new_pos..self.end).rev() {
                self.add(samples[p], -1.0);
            }
        }
        self.wr = self.wn - self.wl;
        for c in 0..self.sum_total.len() {
            self.sum_right[c] = self.sum_total[c] - self.sum_left[c];
        }
        self.pos = new_pos;
    }

    fn impurity_of(&self, sum: &[f64], w: f64, sq: f64) -> f64 {
        match self.kind {
            SkCriterion::SquaredError => sq / w - (sum[0] / w).powi(2),
            SkCriterion::Gini => {
                let s: f64 = sum.iter().map(|c| c * c).sum();
                1.0 - s / (w * w)
            }
            SkCriterion::Entropy => {
                let mut e = 0.0;
                for &c in sum {
                    if c > 0.0 {
                        let p = c / w;
                        e -= p * (p.ln() / std::f64::consts::LN_2);
                    }
                }
                e
            }
        }
    }

    fn node_impurity(&self) -> f64 {
        self.impurity_of(&self.sum_total, self.wn, self.sq_sum_total)
    }

    fn children_impurity(&self, samples: &[usize]) -> (f64, f64) {
        let mut sq_left = 0.0;
        if self.kind == SkCriterion::SquaredError {
            for &i in &samples[self.start..self.pos] {
                let w = self.weight(i);
                sq_left += w * self.y[i] * self.y[i];
            }
        }
        let sq_right = self.sq_sum_total - sq_left;
        (
            self.impurity_of(&self.sum_left, self.wl, sq_left),
            self.impurity_of(&self.sum_right, self.wr, sq_right),
        )
    }

    fn proxy_improvement(&self, samples: &[usize]) -> f64 {
        if self.kind == SkCriterion::SquaredError {
            self.sum_left[0] * self.sum_left[0] / self.wl
                + self.sum_right[0] * self.sum_right[0] / self.wr
        } else {
            let (il, ir) = self.children_impurity(samples);
            -self.wr * ir - self.wl * il
        }
    }

    fn improvement(&self, parent: f64, il: f64, ir: f64) -> f64 {
        (self.wn / self.weighted_n_samples)
            * (parent - (self.wr / self.wn * ir) - (self.wl / self.wn * il))
    }

    fn node_value(&self) -> Vec<f64> {
        self.sum_total.iter().map(|s| s / self.wn).collect()
    }
}

struct Split {
    pos: usize,
    feature: usize,
    threshold: f64,
    impurity_left: f64,
    impurity_right: f64,
    improvement: f64,
}

/// Fit sklearn's `ExtraTree*` tree on `x` (row-major, cast to float32 as
/// sklearn does), `y` (targets, or class codes `0..n_classes`).
#[allow(clippy::too_many_arguments)]
pub fn fit_random_tree(
    x: &[f32],
    n_samples: usize,
    n_features: usize,
    y: &[f64],
    sample_weight: Option<&[f64]>,
    criterion: SkCriterion,
    n_classes: usize,
    limits: SkLimits,
    rng: &mut RandomState,
) -> SkTree {
    let mut rand = RandR(rng.randint(u64::from(RAND_R_MAX)) as u32);
    let xv = |i: usize, f: usize| x[i * n_features + f];
    let mut samples: Vec<usize> = (0..n_samples)
        .filter(|&i| sample_weight.is_none_or(|w| w[i] != 0.0))
        .collect();
    let weighted_n_samples = sample_weight.map_or(n_samples as f64, |w| w.iter().sum());
    let mut features: Vec<usize> = (0..n_features).collect();
    let mut constant_features: Vec<usize> = vec![0; n_features];
    let mut fvals: Vec<f32> = vec![0.0; n_samples];
    let mut crit = Crit {
        kind: criterion,
        y,
        w: sample_weight,
        n_classes,
        weighted_n_samples,
        start: 0,
        end: 0,
        pos: 0,
        wn: 0.0,
        wl: 0.0,
        wr: 0.0,
        sum_total: Vec::new(),
        sum_left: Vec::new(),
        sum_right: Vec::new(),
        sq_sum_total: 0.0,
    };
    let mut nodes: Vec<SkNode> = Vec::new();
    // (start, end, depth, parent, is_left, impurity, n_constant_features)
    let mut stack: Vec<(usize, usize, usize, usize, bool, f64, usize)> =
        vec![(0, samples.len(), 0, usize::MAX, false, f64::INFINITY, 0)];
    let mut first = true;
    while let Some((start, end, depth, parent, is_left, mut impurity, n_known_constants)) =
        stack.pop()
    {
        let n_node = end - start;
        crit.init(&samples, start, end);
        let mut is_leaf = depth >= limits.max_depth
            || n_node < limits.min_samples_split
            || n_node < 2 * limits.min_samples_leaf
            || crit.wn < 2.0 * limits.min_weight_leaf;
        if first {
            impurity = crit.node_impurity();
            first = false;
        }
        is_leaf = is_leaf || impurity <= EPSILON;
        let mut n_constants_out = n_known_constants;
        let mut split: Option<Split> = None;
        if !is_leaf {
            // ---- node_split_random ----
            let mut best: Option<(f64, usize, usize, f64)> = None; // (proxy, pos, feature, threshold)
            let mut f_i = n_features;
            let (mut n_found, mut n_drawn, mut n_total) = (0, 0, n_known_constants);
            let mut n_visited = 0;
            let mut last_feature: Option<usize> = None;
            while f_i > n_total
                && (n_visited < limits.max_features || n_visited <= n_found + n_drawn)
            {
                n_visited += 1;
                let mut f_j = rand.int(n_drawn, f_i - n_found);
                if f_j < n_known_constants {
                    features.swap(n_drawn, f_j);
                    n_drawn += 1;
                    continue;
                }
                f_j += n_found;
                let feature = features[f_j];
                last_feature = Some(feature);
                // find_min_max (float32 values)
                let mut min_v = xv(samples[start], feature);
                let mut max_v = min_v;
                fvals[start] = min_v;
                for p in start + 1..end {
                    let v = xv(samples[p], feature);
                    fvals[p] = v;
                    if v < min_v {
                        min_v = v;
                    } else if v > max_v {
                        max_v = v;
                    }
                }
                if f64::from(max_v) <= f64::from(min_v) + FEATURE_THRESHOLD {
                    features[f_j] = features[n_total];
                    features[n_total] = feature;
                    n_found += 1;
                    n_total += 1;
                    continue;
                }
                f_i -= 1;
                features.swap(f_i, f_j);
                let mut threshold = rand.uniform(f64::from(min_v), f64::from(max_v));
                if threshold == f64::from(max_v) {
                    threshold = f64::from(min_v);
                }
                // partition_samples
                let mut p = start;
                let mut part_end = end;
                while p < part_end {
                    if f64::from(fvals[p]) <= threshold {
                        p += 1;
                    } else {
                        part_end -= 1;
                        fvals.swap(p, part_end);
                        samples.swap(p, part_end);
                    }
                }
                let pos = part_end;
                if pos - start < limits.min_samples_leaf || end - pos < limits.min_samples_leaf {
                    continue;
                }
                crit.reset();
                crit.update(&samples, pos);
                if crit.wl < limits.min_weight_leaf || crit.wr < limits.min_weight_leaf {
                    continue;
                }
                let proxy = crit.proxy_improvement(&samples);
                if best.is_none_or(|b| proxy > b.0) {
                    best = Some((proxy, pos, feature, threshold));
                }
            }
            if let Some((_, pos, feature, threshold)) = best {
                if last_feature != Some(feature) {
                    // partition_samples_final (no missing values)
                    let mut p = start;
                    let mut part_end = end - 1;
                    while p < part_end {
                        if f64::from(xv(samples[p], feature)) <= threshold {
                            p += 1;
                        } else {
                            samples.swap(p, part_end);
                            part_end -= 1;
                        }
                    }
                }
                crit.reset();
                crit.update(&samples, pos);
                let (il, ir) = crit.children_impurity(&samples);
                split = Some(Split {
                    pos,
                    feature,
                    threshold,
                    impurity_left: il,
                    impurity_right: ir,
                    improvement: crit.improvement(impurity, il, ir),
                });
            }
            // memcpy(features[0..n_known], constant_features[0..n_known]);
            // memcpy(constant_features[n_known..n_known+n_found], features[n_known..]).
            features[..n_known_constants].copy_from_slice(&constant_features[..n_known_constants]);
            let found: Vec<usize> =
                features[n_known_constants..n_known_constants + n_found].to_vec();
            constant_features[n_known_constants..n_known_constants + n_found]
                .copy_from_slice(&found);
            n_constants_out = n_total;
            match &split {
                Some(s)
                    if s.pos < end && s.improvement + EPSILON >= limits.min_impurity_decrease => {}
                _ => is_leaf = true,
            }
        }
        let node_id = nodes.len();
        if parent != usize::MAX {
            if is_left {
                nodes[parent].left = node_id;
            } else {
                nodes[parent].right = node_id;
            }
        }
        // sum_total is the node's (reset / update leave it unchanged).
        let value = crit.node_value();
        match (is_leaf, split) {
            (false, Some(s)) => {
                nodes.push(SkNode {
                    left: 0,
                    right: 0,
                    feature: s.feature,
                    threshold: s.threshold,
                    value,
                });
                stack.push((
                    s.pos,
                    end,
                    depth + 1,
                    node_id,
                    false,
                    s.impurity_right,
                    n_constants_out,
                ));
                stack.push((
                    start,
                    s.pos,
                    depth + 1,
                    node_id,
                    true,
                    s.impurity_left,
                    n_constants_out,
                ));
            }
            _ => nodes.push(SkNode {
                left: SkTree::LEAF,
                right: SkTree::LEAF,
                feature: 0,
                threshold: -2.0,
                value,
            }),
        }
    }
    SkTree { nodes, n_features }
}
