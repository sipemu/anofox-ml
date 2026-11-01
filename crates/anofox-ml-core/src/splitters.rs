//! Additional cross-validation splitters.
//!
//! Every splitter returns `Vec<CvSplit>` (`(train_indices, test_indices)`
//! pairs with ascending indices), so the result can be passed to the
//! `*_with_splits` helpers ([`crate::cross_validate_with_splits`],
//! [`crate::grid_search_cv_with_splits`], ...).
//!
//! Group splitters mirror scikit-learn's `GroupKFold`,
//! `StratifiedGroupKFold`, `LeaveOneGroupOut`, `LeavePGroupsOut` and
//! `GroupShuffleSplit`: samples sharing a group label never end up on both
//! sides of a split.
//!
//! [`purged_k_fold`] and [`combinatorial_purged_cv`] implement López de
//! Prado's designs for labels that span a time interval (*Advances in
//! Financial Machine Learning*, ch. 7 and 12): training samples whose label
//! interval overlaps a test block are purged, and an embargo drops the
//! samples right after each test block. Use anofox-forecast's walk-forward
//! cross-validation for forecasting models; these splitters are for ML
//! models fitted on time-indexed features.
//!
//! [`time_series_split_with`] and [`group_time_series_split`] are
//! scikit-learn's `TimeSeriesSplit` with `test_size` / `gap` /
//! `max_train_size` (plus a `min_train_size` check, as anofox-regression's
//! `TimeSeriesSplitOptions`); [`stratified_k_fold_continuous`] stratifies a
//! continuous target on its quantile bins.
//!
//! [`spatial_block_k_fold`], [`spatial_buffer`] and
//! [`buffered_leave_one_out`] are for spatially autocorrelated data, where
//! random k-fold overstates skill (Roberts et al. 2017): whole spatial blocks
//! are held out and/or training points within a buffer distance of the test
//! points are dropped.

use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::error::{Result, RustMlError};
use crate::float::Float;
use crate::utils::CvSplit;

/// Sorted unique group labels and, per sample, the index of its group.
fn encode_groups(groups: &Array1<usize>) -> (Vec<usize>, Vec<usize>) {
    let mut unique: Vec<usize> = groups.iter().copied().collect();
    unique.sort_unstable();
    unique.dedup();
    let idx = groups
        .iter()
        .map(|g| unique.binary_search(g).expect("label present"))
        .collect();
    (unique, idx)
}

/// `(train, test)` split with the samples of the groups in `test_groups`
/// (group indices, as returned by [`encode_groups`]) on the test side.
fn split_by_groups(group_idx: &[usize], in_test: &[bool]) -> CvSplit {
    let mut train = Vec::new();
    let mut test = Vec::new();
    for (i, &g) in group_idx.iter().enumerate() {
        if in_test[g] {
            test.push(i);
        } else {
            train.push(i);
        }
    }
    (train, test)
}

/// One split per fold from a group → fold assignment.
fn splits_from_group_folds(group_idx: &[usize], group_fold: &[usize], k: usize) -> Vec<CvSplit> {
    (0..k)
        .map(|f| {
            let in_test: Vec<bool> = group_fold.iter().map(|&gf| gf == f).collect();
            split_by_groups(group_idx, &in_test)
        })
        .collect()
}

fn check_n_splits_groups(k: usize, n_groups: usize) -> Result<()> {
    if k < 2 {
        return Err(RustMlError::InvalidParameter(
            "n_splits must be >= 2".into(),
        ));
    }
    if k > n_groups {
        return Err(RustMlError::InvalidParameter(format!(
            "n_splits={k} exceeds the number of groups ({n_groups})"
        )));
    }
    Ok(())
}

/// Group k-fold with size-balanced folds (scikit-learn `GroupKFold`).
///
/// Groups are assigned largest first to the fold with the fewest samples so
/// far, which keeps the fold sizes as equal as the group sizes allow. Ties
/// between equally large groups are broken by the smaller group label, and
/// ties between equally light folds by the lower fold index.
pub fn group_k_fold(groups: &Array1<usize>, k: usize) -> Result<Vec<CvSplit>> {
    let (unique, group_idx) = encode_groups(groups);
    check_n_splits_groups(k, unique.len())?;
    let mut sizes = vec![0usize; unique.len()];
    for &g in &group_idx {
        sizes[g] += 1;
    }
    let mut order: Vec<usize> = (0..unique.len()).collect();
    order.sort_by(|&a, &b| sizes[b].cmp(&sizes[a]).then(a.cmp(&b)));
    let mut fold_size = vec![0usize; k];
    let mut group_fold = vec![0usize; unique.len()];
    for g in order {
        let lightest = (0..k).min_by_key(|&f| (fold_size[f], f)).expect("k >= 2");
        fold_size[lightest] += sizes[g];
        group_fold[g] = lightest;
    }
    Ok(splits_from_group_folds(&group_idx, &group_fold, k))
}

/// Group k-fold over randomly permuted groups (scikit-learn
/// `GroupKFold(shuffle=True)`): the shuffled group list is cut into `k`
/// contiguous chunks of `n_groups / k` or `n_groups / k + 1` groups, so fold
/// sizes are balanced in groups, not in samples.
pub fn group_k_fold_shuffled(groups: &Array1<usize>, k: usize, seed: u64) -> Result<Vec<CvSplit>> {
    let (unique, group_idx) = encode_groups(groups);
    let n_groups = unique.len();
    check_n_splits_groups(k, n_groups)?;
    let mut perm: Vec<usize> = (0..n_groups).collect();
    perm.shuffle(&mut rand::rngs::StdRng::seed_from_u64(seed));
    let mut group_fold = vec![0usize; n_groups];
    // numpy.array_split: the first n_groups % k chunks get one extra group.
    let (base, extra) = (n_groups / k, n_groups % k);
    let mut pos = 0;
    for f in 0..k {
        let len = base + usize::from(f < extra);
        for &g in &perm[pos..pos + len] {
            group_fold[g] = f;
        }
        pos += len;
    }
    Ok(splits_from_group_folds(&group_idx, &group_fold, k))
}

/// Stratified group k-fold (scikit-learn `StratifiedGroupKFold`).
///
/// Keeps groups intact while making the class distribution of every fold as
/// close to the overall one as possible: groups are visited in decreasing
/// order of the standard deviation of their class counts and each is put in
/// the fold that minimises the mean (over classes) standard deviation of the
/// per-fold class fractions; ties go to the fold with fewer samples. With
/// `seed = Some(s)` the group order is shuffled first (groups with equal
/// spread keep the shuffled order), as `shuffle=True` does.
pub fn stratified_group_k_fold<F: Float>(
    y: &Array1<F>,
    groups: &Array1<usize>,
    k: usize,
    seed: Option<u64>,
) -> Result<Vec<CvSplit>> {
    if y.len() != groups.len() {
        return Err(RustMlError::ShapeMismatch(format!(
            "y has {} elements but groups has {}",
            y.len(),
            groups.len()
        )));
    }
    let (unique, group_idx) = encode_groups(groups);
    check_n_splits_groups(k, unique.len())?;
    let classes = crate::utils::group_by_class(y);
    let n_classes = classes.len();
    let mut class_of = vec![0usize; y.len()];
    for (c, members) in classes.iter().enumerate() {
        for &i in members {
            class_of[i] = c;
        }
    }
    let class_count: Vec<f64> = classes.iter().map(|m| m.len() as f64).collect();
    if class_count.iter().all(|&c| (k as f64) > c) {
        return Err(RustMlError::InvalidParameter(format!(
            "n_splits={k} cannot be greater than the number of members in each class"
        )));
    }
    let n_groups = unique.len();
    let mut counts = vec![vec![0.0f64; n_classes]; n_groups];
    for (i, &g) in group_idx.iter().enumerate() {
        counts[g][class_of[i]] += 1.0;
    }

    let std = |v: &[f64]| -> f64 {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64).sqrt()
    };
    let mut order: Vec<usize> = (0..n_groups).collect();
    if let Some(s) = seed {
        order.shuffle(&mut rand::rngs::StdRng::seed_from_u64(s));
    }
    let spread: Vec<f64> = counts.iter().map(|c| std(c)).collect();
    // Stable sort by decreasing spread (keeps the (shuffled) order on ties).
    order.sort_by(|&a, &b| spread[b].total_cmp(&spread[a]));

    let mut fold_counts = vec![vec![0.0f64; n_classes]; k];
    let mut group_fold = vec![0usize; n_groups];
    for g in order {
        let mut best = 0;
        let mut best_eval = f64::INFINITY;
        let mut best_size = f64::INFINITY;
        for f in 0..k {
            for c in 0..n_classes {
                fold_counts[f][c] += counts[g][c];
            }
            let eval = (0..n_classes)
                .map(|c| {
                    let fractions: Vec<f64> = (0..k)
                        .map(|ff| fold_counts[ff][c] / class_count[c])
                        .collect();
                    std(&fractions)
                })
                .sum::<f64>()
                / n_classes as f64;
            for c in 0..n_classes {
                fold_counts[f][c] -= counts[g][c];
            }
            let size: f64 = fold_counts[f].iter().sum();
            // numpy.isclose defaults: rtol 1e-5, atol 1e-8.
            let close = (eval - best_eval).abs() <= 1e-8 + 1e-5 * best_eval.abs();
            if eval < best_eval && !close || (close && size < best_size) {
                best = f;
                best_eval = eval;
                best_size = size;
            }
        }
        for c in 0..n_classes {
            fold_counts[best][c] += counts[g][c];
        }
        group_fold[g] = best;
    }
    Ok(splits_from_group_folds(&group_idx, &group_fold, k))
}

/// Leave-one-group-out (scikit-learn `LeaveOneGroupOut`): one split per
/// group (in ascending label order), testing on that group.
pub fn leave_one_group_out(groups: &Array1<usize>) -> Result<Vec<CvSplit>> {
    leave_p_groups_out(groups, 1)
}

/// Upper bound on the number of splits [`leave_p_groups_out`] generates.
pub const MAX_LEAVE_P_GROUPS_OUT_SPLITS: usize = 100_000;

/// Leave-p-groups-out (scikit-learn `LeavePGroupsOut`): one split for every
/// combination of `p` groups (lexicographic in ascending label order),
/// testing on those groups. Needs at least `p + 1` groups; returns an error
/// when the number of combinations exceeds
/// [`MAX_LEAVE_P_GROUPS_OUT_SPLITS`].
pub fn leave_p_groups_out(groups: &Array1<usize>, p: usize) -> Result<Vec<CvSplit>> {
    let (unique, group_idx) = encode_groups(groups);
    let n_groups = unique.len();
    if p == 0 {
        return Err(RustMlError::InvalidParameter("p must be >= 1".into()));
    }
    if p >= n_groups {
        return Err(RustMlError::InvalidParameter(format!(
            "leave-{p}-groups-out needs at least {} groups, got {n_groups}",
            p + 1
        )));
    }
    let mut n_combinations: u128 = 1;
    for i in 0..p {
        n_combinations = n_combinations * (n_groups - i) as u128 / (i + 1) as u128;
        if n_combinations > MAX_LEAVE_P_GROUPS_OUT_SPLITS as u128 {
            return Err(RustMlError::InvalidParameter(format!(
                "leave-{p}-groups-out over {n_groups} groups exceeds \
                 {MAX_LEAVE_P_GROUPS_OUT_SPLITS} splits"
            )));
        }
    }
    let mut out = Vec::with_capacity(n_combinations as usize);
    let mut combo: Vec<usize> = (0..p).collect();
    loop {
        let mut in_test = vec![false; n_groups];
        for &g in &combo {
            in_test[g] = true;
        }
        out.push(split_by_groups(&group_idx, &in_test));
        // Next combination in lexicographic order.
        let mut i = p;
        while i > 0 && combo[i - 1] == n_groups - p + i - 1 {
            i -= 1;
        }
        if i == 0 {
            break;
        }
        combo[i - 1] += 1;
        for j in i..p {
            combo[j] = combo[j - 1] + 1;
        }
    }
    Ok(out)
}

/// Group shuffle split (scikit-learn `GroupShuffleSplit`): `n_splits`
/// random splits in which `test_size` is a fraction of the **groups**
/// (`ceil(test_size · n_groups)` groups on the test side, the rest — or
/// `floor(train_size · n_groups)` when `train_size` is given — on the
/// training side). Each split is seeded from `seed + split_index`.
pub fn group_shuffle_split(
    groups: &Array1<usize>,
    n_splits: usize,
    test_size: f64,
    train_size: Option<f64>,
    seed: u64,
) -> Result<Vec<CvSplit>> {
    let (unique, group_idx) = encode_groups(groups);
    let n_groups = unique.len();
    if n_splits < 1 {
        return Err(RustMlError::InvalidParameter(
            "n_splits must be >= 1".into(),
        ));
    }
    if !(test_size > 0.0 && test_size < 1.0) {
        return Err(RustMlError::InvalidParameter(
            "test_size must be in (0, 1)".into(),
        ));
    }
    let n_test = (test_size * n_groups as f64).ceil() as usize;
    let n_train = match train_size {
        Some(t) if t > 0.0 && t < 1.0 => (t * n_groups as f64).floor() as usize,
        Some(_) => {
            return Err(RustMlError::InvalidParameter(
                "train_size must be in (0, 1)".into(),
            ))
        }
        None => n_groups.saturating_sub(n_test),
    };
    if n_train == 0 || n_test == 0 || n_train + n_test > n_groups {
        return Err(RustMlError::InvalidParameter(format!(
            "with {n_groups} groups, test_size={test_size} and train_size={train_size:?} \
             leave an empty or overlapping side"
        )));
    }
    let mut out = Vec::with_capacity(n_splits);
    for s in 0..n_splits {
        let mut perm: Vec<usize> = (0..n_groups).collect();
        perm.shuffle(&mut rand::rngs::StdRng::seed_from_u64(
            seed.wrapping_add(s as u64),
        ));
        let mut side = vec![0u8; n_groups]; // 0 unused, 1 test, 2 train
        for &g in &perm[..n_test] {
            side[g] = 1;
        }
        for &g in &perm[n_test..n_test + n_train] {
            side[g] = 2;
        }
        let mut train = Vec::new();
        let mut test = Vec::new();
        for (i, &g) in group_idx.iter().enumerate() {
            match side[g] {
                1 => test.push(i),
                2 => train.push(i),
                _ => {}
            }
        }
        out.push((train, test));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Purged k-fold / combinatorial purged CV
// ---------------------------------------------------------------------------

/// Validate label intervals: equal lengths, finite, `t_end >= t_start`, and
/// `t_start` non-decreasing (samples in time order).
fn check_intervals(t_start: &[f64], t_end: &[f64]) -> Result<()> {
    if t_start.len() != t_end.len() {
        return Err(RustMlError::ShapeMismatch(format!(
            "t_start has {} elements but t_end has {}",
            t_start.len(),
            t_end.len()
        )));
    }
    for i in 0..t_start.len() {
        if !t_start[i].is_finite() || !t_end[i].is_finite() || t_end[i] < t_start[i] {
            return Err(RustMlError::InvalidParameter(format!(
                "label interval {i} must be finite with t_end >= t_start"
            )));
        }
        if i > 0 && t_start[i] < t_start[i - 1] {
            return Err(RustMlError::InvalidParameter(
                "samples must be sorted by t_start".into(),
            ));
        }
    }
    Ok(())
}

/// Contiguous index ranges of `n` samples in `k` blocks (numpy
/// `array_split`: the first `n % k` blocks get one extra sample).
fn contiguous_blocks(n: usize, k: usize) -> Vec<(usize, usize)> {
    let (base, extra) = (n / k, n % k);
    let mut start = 0;
    (0..k)
        .map(|b| {
            let end = start + base + usize::from(b < extra);
            let r = (start, end);
            start = end;
            r
        })
        .collect()
}

/// Training mask for the test blocks `blocks` (index ranges, ascending and
/// non-adjacent): a sample stays in training iff it is not a test sample,
/// its label interval does not overlap any test block's span
/// `[t_start[first], max t_end]`, and it is not among the `embargo` samples
/// that follow a test block's span.
fn purged_train_mask(
    t_start: &[f64],
    t_end: &[f64],
    blocks: &[(usize, usize)],
    embargo: usize,
) -> Vec<bool> {
    let n = t_start.len();
    let mut train = vec![true; n];
    for &(a, b) in blocks {
        let t0 = t_start[a];
        let t1 = t_end[a..b]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        for i in 0..n {
            if (a..b).contains(&i) || (t_start[i] <= t1 && t_end[i] >= t0) {
                train[i] = false;
            }
        }
        // Embargo: the first `embargo` samples starting after the span.
        let first_after = t_start.partition_point(|&t| t <= t1);
        for flag in train.iter_mut().skip(first_after).take(embargo) {
            *flag = false;
        }
    }
    train
}

/// Merge the selected (sorted) groups into maximal runs of adjacent groups.
fn merge_adjacent(groups: &[usize], ranges: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    for &g in groups {
        let (a, b) = ranges[g];
        match out.last_mut() {
            Some(last) if last.1 == a => last.1 = b,
            _ => out.push((a, b)),
        }
    }
    out
}

fn split_from_masks(train: &[bool], blocks: &[(usize, usize)]) -> CvSplit {
    let test: Vec<usize> = blocks.iter().flat_map(|&(a, b)| a..b).collect();
    let train_idx = (0..train.len()).filter(|&i| train[i]).collect();
    (train_idx, test)
}

/// Purged k-fold (López de Prado, AFML §7.4).
///
/// Samples must be in time order; sample `i`'s label covers
/// `[t_start[i], t_end[i]]` (e.g. the observation time and the time its
/// label is resolved). The samples are cut into `n_splits` contiguous test
/// blocks; for each block, training samples whose label interval overlaps
/// the block's span are **purged**, and the `embargo` samples that start
/// right after the span are dropped as well. With point labels
/// (`t_end == t_start`) and `embargo = 0` this is plain unshuffled k-fold.
pub fn purged_k_fold(
    t_start: &[f64],
    t_end: &[f64],
    n_splits: usize,
    embargo: usize,
) -> Result<Vec<CvSplit>> {
    check_intervals(t_start, t_end)?;
    let n = t_start.len();
    if n_splits < 2 || n_splits > n {
        return Err(RustMlError::InvalidParameter(format!(
            "n_splits must be in [2, n_samples={n}], got {n_splits}"
        )));
    }
    let blocks = contiguous_blocks(n, n_splits);
    let mut out = Vec::with_capacity(n_splits);
    for &blk in &blocks {
        let train = purged_train_mask(t_start, t_end, &[blk], embargo);
        let split = split_from_masks(&train, &[blk]);
        if split.0.is_empty() {
            return Err(RustMlError::InvalidParameter(
                "purging and embargo left an empty training set".into(),
            ));
        }
        out.push(split);
    }
    Ok(out)
}

/// Splits of a combinatorial purged cross-validation and their assembly into
/// backtest paths.
#[derive(Debug, Clone, PartialEq)]
pub struct CombinatorialPurgedCv {
    /// One split per combination of `n_test_groups` test groups
    /// (lexicographic), purged and embargoed.
    pub splits: Vec<CvSplit>,
    /// Test groups of every split.
    pub test_groups: Vec<Vec<usize>>,
    /// Index range `[start, end)` of every group.
    pub group_ranges: Vec<(usize, usize)>,
    /// Backtest paths: `paths[j][g]` is the split whose prediction of group
    /// `g` belongs to path `j`. There are `C(n_groups - 1, n_test_groups - 1)`
    /// paths, each covering every group exactly once.
    pub paths: Vec<Vec<usize>>,
}

/// Combinatorial purged cross-validation (López de Prado, AFML §12.4).
///
/// The (time-ordered) samples are cut into `n_groups` contiguous groups and
/// every combination of `n_test_groups` of them is a test set; each
/// contiguous run of test groups is purged and embargoed as in
/// [`purged_k_fold`]. The `C(N, k)` out-of-sample predictions recombine into
/// `C(N-1, k-1)` complete backtest paths ([`CombinatorialPurgedCv::paths`]),
/// which gives a distribution of backtest performance instead of a single
/// walk-forward path.
pub fn combinatorial_purged_cv(
    t_start: &[f64],
    t_end: &[f64],
    n_groups: usize,
    n_test_groups: usize,
    embargo: usize,
) -> Result<CombinatorialPurgedCv> {
    check_intervals(t_start, t_end)?;
    let n = t_start.len();
    if n_groups < 2 || n_groups > n {
        return Err(RustMlError::InvalidParameter(format!(
            "n_groups must be in [2, n_samples={n}], got {n_groups}"
        )));
    }
    if n_test_groups == 0 || n_test_groups >= n_groups {
        return Err(RustMlError::InvalidParameter(format!(
            "n_test_groups must be in [1, n_groups - 1], got {n_test_groups}"
        )));
    }
    let ranges = contiguous_blocks(n, n_groups);
    let mut splits = Vec::new();
    let mut test_groups = Vec::new();
    let mut combo: Vec<usize> = (0..n_test_groups).collect();
    loop {
        let blocks = merge_adjacent(&combo, &ranges);
        let train = purged_train_mask(t_start, t_end, &blocks, embargo);
        let split = split_from_masks(&train, &blocks);
        if split.0.is_empty() {
            return Err(RustMlError::InvalidParameter(
                "purging and embargo left an empty training set".into(),
            ));
        }
        splits.push(split);
        test_groups.push(combo.clone());
        let k = n_test_groups;
        let mut i = k;
        while i > 0 && combo[i - 1] == n_groups - k + i - 1 {
            i -= 1;
        }
        if i == 0 {
            break;
        }
        combo[i - 1] += 1;
        for j in i..k {
            combo[j] = combo[j - 1] + 1;
        }
    }
    // Path j takes, for every group, the j-th split (in split order) that
    // tests that group.
    let mut per_group: Vec<Vec<usize>> = vec![Vec::new(); n_groups];
    for (s, tg) in test_groups.iter().enumerate() {
        for &g in tg {
            per_group[g].push(s);
        }
    }
    let n_paths = per_group[0].len();
    let paths = (0..n_paths)
        .map(|j| (0..n_groups).map(|g| per_group[g][j]).collect())
        .collect();
    Ok(CombinatorialPurgedCv {
        splits,
        test_groups,
        group_ranges: ranges,
        paths,
    })
}

// ---------------------------------------------------------------------------
// Time-series splits with options, group time-series splits
// ---------------------------------------------------------------------------

/// Options of [`time_series_split_with`] (scikit-learn `TimeSeriesSplit`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeSeriesSplitOptions {
    /// Number of splits (test blocks), at least 1. Default 5.
    pub n_splits: usize,
    /// Samples per test block; default `n / (n_splits + 1)`.
    pub test_size: Option<usize>,
    /// Samples left out between the end of the training window and each
    /// test block. Default 0.
    pub gap: usize,
    /// Keep at most this many samples right before the gap (sliding
    /// window); default `None` = expanding window.
    pub max_train_size: Option<usize>,
    /// Smallest admissible training window: an error is returned when a
    /// split has fewer training samples. Default `None`.
    pub min_train_size: Option<usize>,
}

impl Default for TimeSeriesSplitOptions {
    fn default() -> Self {
        Self {
            n_splits: 5,
            test_size: None,
            gap: 0,
            max_train_size: None,
            min_train_size: None,
        }
    }
}

impl TimeSeriesSplitOptions {
    /// Default options with `n_splits` splits.
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            ..Self::default()
        }
    }
}

/// Forward-chaining splits of `n` time-ordered samples with scikit-learn's
/// `TimeSeriesSplit(n_splits, test_size, gap, max_train_size)` semantics:
/// the last `n_splits · test_size` samples form the test blocks, each
/// training window ends `gap` samples before its block and is truncated to
/// the most recent `max_train_size` samples when given.
pub fn time_series_split_with(n: usize, opts: &TimeSeriesSplitOptions) -> Result<Vec<CvSplit>> {
    let k = opts.n_splits;
    if k < 1 {
        return Err(RustMlError::InvalidParameter(
            "n_splits must be >= 1".into(),
        ));
    }
    let test_size = opts.test_size.unwrap_or(n / (k + 1));
    if test_size == 0 {
        return Err(RustMlError::InvalidParameter(format!(
            "n_samples={n} is too small for {k} splits"
        )));
    }
    if n <= opts.gap + test_size * k {
        return Err(RustMlError::InvalidParameter(format!(
            "too many splits={k} with test_size={test_size} and gap={} for n_samples={n}",
            opts.gap
        )));
    }
    let first_test = n - k * test_size;
    let mut out = Vec::with_capacity(k);
    for b in 0..k {
        let test_start = first_test + b * test_size;
        let train_end = test_start - opts.gap;
        let train_start = match opts.max_train_size {
            Some(m) if m < train_end => train_end - m,
            _ => 0,
        };
        if let Some(min) = opts.min_train_size {
            if train_end - train_start < min {
                return Err(RustMlError::InvalidParameter(format!(
                    "split {b} has {} training samples, fewer than min_train_size={min}",
                    train_end - train_start
                )));
            }
        }
        out.push((
            (train_start..train_end).collect(),
            (test_start..test_start + test_size).collect(),
        ));
    }
    Ok(out)
}

/// Time-series splits over **time keys** (e.g. dates, with several samples
/// per date): the distinct keys, sorted, are split by
/// [`time_series_split_with`] and every sample follows its key, so samples
/// sharing a time point never straddle a split (anofox-regression's
/// `group_time_series_splits`).
pub fn group_time_series_split<T: Ord + Clone>(
    time_keys: &[T],
    opts: &TimeSeriesSplitOptions,
) -> Result<Vec<CvSplit>> {
    let mut keys: Vec<T> = time_keys.to_vec();
    keys.sort();
    keys.dedup();
    let code: Vec<usize> = time_keys
        .iter()
        .map(|t| keys.binary_search(t).expect("key present"))
        .collect();
    let splits = time_series_split_with(keys.len(), opts)?;
    Ok(splits
        .into_iter()
        .map(|(tr, te)| {
            let mut side = vec![0u8; keys.len()];
            tr.iter().for_each(|&t| side[t] = 1);
            te.iter().for_each(|&t| side[t] = 2);
            (
                (0..code.len()).filter(|&i| side[code[i]] == 1).collect(),
                (0..code.len()).filter(|&i| side[code[i]] == 2).collect(),
            )
        })
        .collect())
}

/// Quantile strata of a continuous target: the sorted values are cut into
/// `n_bins` equally populated bins; tied values share a bin
/// (anofox-regression's `quantile_strata`).
pub fn quantile_strata<F: Float>(y: &Array1<F>, n_bins: usize) -> Vec<usize> {
    let n = y.len();
    let n_bins = n_bins.max(1);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| y[a].partial_cmp(&y[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = vec![0usize; n];
    let mut prev_bin = 0;
    for (r, &i) in order.iter().enumerate() {
        let tied = r > 0 && y[order[r - 1]] == y[i];
        let bin = if tied { prev_bin } else { r * n_bins / n };
        out[i] = bin;
        prev_bin = bin;
    }
    out
}

/// Stratified k-fold for a **continuous** target: stratify on
/// [`quantile_strata`] with `n_bins` bins, so every fold covers the whole
/// range of `y`.
pub fn stratified_k_fold_continuous<F: Float>(
    y: &Array1<F>,
    k: usize,
    n_bins: usize,
    seed: u64,
) -> Result<Vec<CvSplit>> {
    if y.iter().any(|v| v.is_nan()) {
        return Err(RustMlError::InvalidParameter("y contains NaN".into()));
    }
    let strata = quantile_strata(y, n_bins);
    let labels: Array1<f64> = strata.iter().map(|&s| s as f64).collect();
    let placeholder = Array2::<f64>::zeros((labels.len(), 1));
    crate::utils::stratified_k_fold(&placeholder, &labels, k, seed)
}

// ---------------------------------------------------------------------------
// Spatial cross-validation
// ---------------------------------------------------------------------------

/// Distance between two coordinate rows for spatial cross-validation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpatialMetric {
    /// Euclidean distance in the units of the coordinates.
    Euclidean,
    /// Great-circle distance in kilometres; coordinates are
    /// `(latitude, longitude)` in degrees (mean Earth radius 6371.0088 km).
    Haversine,
}

impl SpatialMetric {
    fn distance(self, a: ndarray::ArrayView1<f64>, b: ndarray::ArrayView1<f64>) -> f64 {
        match self {
            SpatialMetric::Euclidean => a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f64>()
                .sqrt(),
            SpatialMetric::Haversine => {
                let (la1, lo1) = (a[0].to_radians(), a[1].to_radians());
                let (la2, lo2) = (b[0].to_radians(), b[1].to_radians());
                let h = ((la2 - la1) / 2.0).sin().powi(2)
                    + la1.cos() * la2.cos() * ((lo2 - lo1) / 2.0).sin().powi(2);
                2.0 * 6371.0088 * h.sqrt().min(1.0).asin()
            }
        }
    }
}

fn check_coords(coords: &Array2<f64>, metric: SpatialMetric) -> Result<()> {
    if coords.nrows() == 0 || coords.ncols() == 0 {
        return Err(RustMlError::EmptyInput("coordinates are empty".into()));
    }
    if coords.iter().any(|v| !v.is_finite()) {
        return Err(RustMlError::InvalidParameter(
            "coordinates must be finite".into(),
        ));
    }
    if metric == SpatialMetric::Haversine && coords.ncols() != 2 {
        return Err(RustMlError::InvalidParameter(
            "haversine needs (latitude, longitude) coordinates".into(),
        ));
    }
    Ok(())
}

/// Spatial block k-fold (as R's `blockCV::cv_spatial` with square blocks).
///
/// Points are binned into a regular grid of `block_size`-wide cells (one
/// cell index per coordinate column, in the coordinate units); whole cells
/// are then assigned to `k` folds in random order (`seed`), each to the fold
/// with the fewest points so far. Points of one cell always share a fold.
/// For irregular blocks (e.g. k-means clusters of the coordinates), pass the
/// block labels as groups to [`group_k_fold`] instead.
pub fn spatial_block_k_fold(
    coords: &Array2<f64>,
    block_size: f64,
    k: usize,
    seed: u64,
) -> Result<Vec<CvSplit>> {
    check_coords(coords, SpatialMetric::Euclidean)?;
    if !(block_size > 0.0 && block_size.is_finite()) {
        return Err(RustMlError::InvalidParameter(
            "block_size must be positive and finite".into(),
        ));
    }
    let cells: Vec<Vec<i64>> = coords
        .rows()
        .into_iter()
        .map(|r| r.iter().map(|v| (v / block_size).floor() as i64).collect())
        .collect();
    let mut unique = cells.clone();
    unique.sort();
    unique.dedup();
    let cell_of: Vec<usize> = cells
        .iter()
        .map(|c| unique.binary_search(c).expect("cell present"))
        .collect();
    check_n_splits_groups(k, unique.len())?;
    let mut sizes = vec![0usize; unique.len()];
    for &c in &cell_of {
        sizes[c] += 1;
    }
    let mut order: Vec<usize> = (0..unique.len()).collect();
    order.shuffle(&mut rand::rngs::StdRng::seed_from_u64(seed));
    let mut fold_size = vec![0usize; k];
    let mut cell_fold = vec![0usize; unique.len()];
    for c in order {
        let lightest = (0..k).min_by_key(|&f| (fold_size[f], f)).expect("k >= 2");
        fold_size[lightest] += sizes[c];
        cell_fold[c] = lightest;
    }
    Ok(splits_from_group_folds(&cell_of, &cell_fold, k))
}

/// Apply a spatial buffer to existing splits: every training point within
/// `radius` of any test point of the same split is removed (spatialsample's
/// `buffer`). Works with any splitter, e.g. [`spatial_block_k_fold`] or a
/// plain k-fold.
pub fn spatial_buffer(
    splits: &[CvSplit],
    coords: &Array2<f64>,
    radius: f64,
    metric: SpatialMetric,
) -> Result<Vec<CvSplit>> {
    check_coords(coords, metric)?;
    if !(radius >= 0.0 && radius.is_finite()) {
        return Err(RustMlError::InvalidParameter(
            "radius must be finite and >= 0".into(),
        ));
    }
    let n = coords.nrows();
    let mut out = Vec::with_capacity(splits.len());
    for (s, (train, test)) in splits.iter().enumerate() {
        if let Some(&bad) = train.iter().chain(test.iter()).find(|&&i| i >= n) {
            return Err(RustMlError::InvalidParameter(format!(
                "split {s} contains index {bad} >= n_samples ({n})"
            )));
        }
        let kept: Vec<usize> = train
            .iter()
            .copied()
            .filter(|&i| {
                test.iter()
                    .all(|&j| metric.distance(coords.row(i), coords.row(j)) > radius)
            })
            .collect();
        if kept.is_empty() {
            return Err(RustMlError::InvalidParameter(format!(
                "the buffer removed every training point of split {s}"
            )));
        }
        out.push((kept, test.clone()));
    }
    Ok(out)
}

/// Buffered leave-one-out (Le Rest et al. 2014): one split per point, whose
/// training set excludes every point within `radius` of it.
pub fn buffered_leave_one_out(
    coords: &Array2<f64>,
    radius: f64,
    metric: SpatialMetric,
) -> Result<Vec<CvSplit>> {
    let n = coords.nrows();
    let loo: Vec<CvSplit> = (0..n)
        .map(|i| ((0..n).filter(|&j| j != i).collect(), vec![i]))
        .collect();
    spatial_buffer(&loo, coords, radius, metric)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn tests_of(splits: &[CvSplit]) -> Vec<Vec<usize>> {
        splits.iter().map(|(_, te)| te.clone()).collect()
    }

    fn assert_groups_disjoint(groups: &Array1<usize>, splits: &[CvSplit]) {
        for (train, test) in splits {
            for &i in test {
                assert!(train.iter().all(|&j| groups[j] != groups[i]));
            }
        }
    }

    #[test]
    fn group_k_fold_matches_sklearn() {
        // sklearn 1.8 GroupKFold(3) (distinct group sizes, so numpy's
        // unstable argsort cannot reorder ties).
        let g = array![2, 2, 2, 2, 2, 0, 0, 0, 0, 4, 4, 4, 1, 1, 5, 3, 3, 3, 3, 3, 3];
        let s = group_k_fold(&g, 3).unwrap();
        assert_eq!(
            tests_of(&s),
            vec![
                vec![14, 15, 16, 17, 18, 19, 20],
                vec![0, 1, 2, 3, 4, 12, 13],
                vec![5, 6, 7, 8, 9, 10, 11],
            ]
        );
        assert_groups_disjoint(&g, &s);
        for (train, test) in &s {
            assert_eq!(train.len() + test.len(), g.len());
        }
    }

    #[test]
    fn group_k_fold_shuffled_partitions_groups() {
        let g = array![0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 6, 6];
        let a = group_k_fold_shuffled(&g, 3, 7).unwrap();
        let b = group_k_fold_shuffled(&g, 3, 7).unwrap();
        assert_eq!(a, b);
        assert_groups_disjoint(&g, &a);
        let mut seen: Vec<usize> = a.iter().flat_map(|(_, te)| te.clone()).collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..g.len()).collect::<Vec<_>>());
        // 7 groups over 3 folds: 3, 2, 2 groups.
        let n_groups_per_fold: Vec<usize> = a
            .iter()
            .map(|(_, te)| {
                let mut gs: Vec<usize> = te.iter().map(|&i| g[i]).collect();
                gs.dedup();
                gs.len()
            })
            .collect();
        assert_eq!(n_groups_per_fold, vec![3, 2, 2]);
    }

    #[test]
    fn stratified_group_k_fold_matches_sklearn() {
        let g = array![3, 3, 3, 1, 1, 0, 0, 0, 0, 2, 5, 5, 4, 4, 6, 6, 6];
        let y = array![0., 1., 0., 1., 1., 0., 0., 1., 0., 1., 1., 0., 0., 0., 1., 1., 1.];
        let s = stratified_group_k_fold(&y, &g, 3, None).unwrap();
        // sklearn 1.8 StratifiedGroupKFold(3)
        assert_eq!(
            tests_of(&s),
            vec![
                vec![0, 1, 2, 14, 15, 16],
                vec![5, 6, 7, 8, 9],
                vec![3, 4, 10, 11, 12, 13],
            ]
        );
        let shuffled = stratified_group_k_fold(&y, &g, 3, Some(1)).unwrap();
        assert_groups_disjoint(&g, &shuffled);
    }

    #[test]
    fn leave_groups_out_match_sklearn() {
        let g = array![3, 3, 3, 1, 1, 0, 0, 0, 0, 2, 5, 5, 4, 4, 6, 6, 6];
        let logo = leave_one_group_out(&g).unwrap();
        assert_eq!(logo.len(), 7);
        assert_eq!(
            tests_of(&logo)[..3].to_vec(),
            vec![vec![5, 6, 7, 8], vec![3, 4], vec![9]]
        );
        let lpgo = leave_p_groups_out(&g, 2).unwrap();
        assert_eq!(lpgo.len(), 21);
        assert_eq!(
            tests_of(&lpgo)[..4].to_vec(),
            vec![
                vec![3, 4, 5, 6, 7, 8],
                vec![5, 6, 7, 8, 9],
                vec![0, 1, 2, 5, 6, 7, 8],
                vec![5, 6, 7, 8, 12, 13],
            ]
        );
        assert!(leave_p_groups_out(&g, 7).is_err());
    }

    #[test]
    fn group_shuffle_split_counts_groups() {
        let g = array![0, 0, 0, 1, 2, 2, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9];
        let s = group_shuffle_split(&g, 5, 0.3, None, 0).unwrap();
        assert_eq!(s.len(), 5);
        assert_groups_disjoint(&g, &s);
        for (train, test) in &s {
            let mut tg: Vec<usize> = test.iter().map(|&i| g[i]).collect();
            tg.dedup();
            assert_eq!(tg.len(), 3); // ceil(0.3 * 10)
            assert_eq!(train.len() + test.len(), g.len());
        }
        let partial = group_shuffle_split(&g, 1, 0.2, Some(0.5), 3).unwrap();
        assert!(partial[0].0.len() + partial[0].1.len() < g.len());
        assert!(group_shuffle_split(&g, 1, 0.9, Some(0.5), 3).is_err());
    }

    fn intervals(n: usize, horizon: f64) -> (Vec<f64>, Vec<f64>) {
        let t0: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let t1: Vec<f64> = t0.iter().map(|t| t + horizon).collect();
        (t0, t1)
    }

    fn assert_purged(t0: &[f64], t1: &[f64], split: &CvSplit) {
        let (train, test) = split;
        for &i in train {
            for &j in test {
                let overlap = t0[i] <= t1[j] && t1[i] >= t0[j];
                assert!(!overlap, "train {i} overlaps test {j}");
            }
        }
    }

    #[test]
    fn purged_k_fold_point_labels_is_k_fold() {
        let (t0, t1) = intervals(11, 0.0);
        let s = purged_k_fold(&t0, &t1, 3, 0).unwrap();
        assert_eq!(s, crate::utils::k_fold(11, 3).unwrap());
    }

    #[test]
    fn purged_k_fold_purges_and_embargoes() {
        // Labels resolve 2 steps later: [i, i + 2].
        let (t0, t1) = intervals(20, 2.0);
        let s = purged_k_fold(&t0, &t1, 4, 1).unwrap();
        for split in &s {
            assert_purged(&t0, &t1, split);
        }
        // Middle block 5..10 spans [5, 11]: samples 3, 4 (overlap before),
        // 10, 11 (overlap after) are purged and 12 is embargoed.
        let (train, test) = &s[1];
        assert_eq!(test, &(5..10).collect::<Vec<_>>());
        let expected: Vec<usize> = (0..3).chain(13..20).collect();
        assert_eq!(train, &expected);
    }

    #[test]
    fn combinatorial_purged_cv_counts_and_paths() {
        let (t0, t1) = intervals(60, 1.0);
        let cv = combinatorial_purged_cv(&t0, &t1, 6, 2, 2).unwrap();
        assert_eq!(cv.splits.len(), 15); // C(6, 2)
        assert_eq!(cv.paths.len(), 5); // C(5, 1) = 2/6 · 15
        for split in &cv.splits {
            assert_purged(&t0, &t1, split);
        }
        // Every path covers each group exactly once with a split that
        // actually tests that group.
        for path in &cv.paths {
            assert_eq!(path.len(), 6);
            for (g, &s) in path.iter().enumerate() {
                assert!(cv.test_groups[s].contains(&g));
            }
        }
        // Every (split, group) prediction is used by exactly one path.
        let mut used = std::collections::HashSet::new();
        for path in &cv.paths {
            for (g, &s) in path.iter().enumerate() {
                assert!(used.insert((s, g)));
            }
        }
        assert_eq!(used.len(), 15 * 2);
        // Adjacent test groups (0, 1) form one block: no train sample inside.
        let (train, test) = &cv.splits[0];
        assert_eq!(test, &(0..20).collect::<Vec<_>>());
        assert_eq!(train[0], 23); // purged 20, embargo 21..23
    }

    #[test]
    fn purged_splitters_validate_input() {
        let (t0, t1) = intervals(10, 1.0);
        assert!(purged_k_fold(&t0, &t1[..9], 2, 0).is_err());
        let mut unsorted = t0.clone();
        unsorted.swap(2, 3);
        assert!(purged_k_fold(&unsorted, &t1, 2, 0).is_err());
        assert!(combinatorial_purged_cv(&t0, &t1, 4, 4, 0).is_err());
        // Labels spanning everything leave no training data.
        let wide: Vec<f64> = vec![100.0; 10];
        assert!(purged_k_fold(&t0, &wide, 2, 0).is_err());
    }

    fn grid_coords() -> Array2<f64> {
        // 6 x 4 points on a unit grid.
        Array2::from_shape_fn((24, 2), |(i, j)| {
            if j == 0 {
                (i % 6) as f64
            } else {
                (i / 6) as f64
            }
        })
    }

    #[test]
    fn spatial_block_k_fold_keeps_cells_together() {
        let c = grid_coords();
        let s = spatial_block_k_fold(&c, 2.0, 3, 5).unwrap();
        assert_eq!(s.len(), 3);
        let cell = |i: usize| {
            (
                (c[[i, 0]] / 2.0).floor() as i64,
                (c[[i, 1]] / 2.0).floor() as i64,
            )
        };
        let mut seen = vec![0; 24];
        for (train, test) in &s {
            for &i in test {
                seen[i] += 1;
                assert!(train.iter().all(|&j| cell(j) != cell(i)));
            }
        }
        assert!(seen.iter().all(|&v| v == 1));
        // 6 cells of 4 points over 3 folds: 8 points each.
        assert!(s.iter().all(|(_, te)| te.len() == 8));
        assert_eq!(s, spatial_block_k_fold(&c, 2.0, 3, 5).unwrap());
    }

    #[test]
    fn spatial_buffer_drops_close_training_points() {
        let c = grid_coords();
        let folds = crate::utils::k_fold(24, 4).unwrap();
        let b = spatial_buffer(&folds, &c, 1.5, SpatialMetric::Euclidean).unwrap();
        for (train, test) in &b {
            for &i in train {
                for &j in test {
                    let d =
                        ((c[[i, 0]] - c[[j, 0]]).powi(2) + (c[[i, 1]] - c[[j, 1]]).powi(2)).sqrt();
                    assert!(d > 1.5);
                }
            }
        }
        let loo = buffered_leave_one_out(&c, 1.0, SpatialMetric::Euclidean).unwrap();
        // Corner point 0 at (0, 0): neighbours (1, 0) and (0, 1) removed.
        assert_eq!(loo[0].0.len(), 24 - 1 - 2);
        assert!(spatial_buffer(&folds, &c, 100.0, SpatialMetric::Euclidean).is_err());
    }

    #[test]
    fn haversine_distance() {
        // Berlin (52.52, 13.405) to Paris (48.8566, 2.3522): ~878 km.
        let c = ndarray::array![[52.52, 13.405], [48.8566, 2.3522]];
        let d = SpatialMetric::Haversine.distance(c.row(0), c.row(1));
        assert!((d - 878.0).abs() < 5.0, "{d}");
        let loo = buffered_leave_one_out(&c, 500.0, SpatialMetric::Haversine).unwrap();
        assert_eq!(loo[0].0, vec![1]);
        assert!(buffered_leave_one_out(&c, 1000.0, SpatialMetric::Haversine).is_err());
    }

    fn bounds(s: &[CvSplit]) -> Vec<(usize, usize, usize, usize)> {
        s.iter()
            .map(|(tr, te)| (tr[0], tr[tr.len() - 1] + 1, te[0], te[te.len() - 1] + 1))
            .collect()
    }

    #[test]
    fn time_series_split_with_matches_sklearn() {
        // sklearn 1.8 TimeSeriesSplit on 20 samples.
        let o = |k, test_size, gap, max_train_size| TimeSeriesSplitOptions {
            n_splits: k,
            test_size,
            gap,
            max_train_size,
            min_train_size: None,
        };
        assert_eq!(
            bounds(&time_series_split_with(20, &o(3, None, 0, None)).unwrap()),
            vec![(0, 5, 5, 10), (0, 10, 10, 15), (0, 15, 15, 20)]
        );
        assert_eq!(
            time_series_split_with(20, &o(3, None, 0, None)).unwrap(),
            crate::utils::time_series_split(20, 3).unwrap()
        );
        assert_eq!(
            bounds(&time_series_split_with(20, &o(3, None, 2, None)).unwrap()),
            vec![(0, 3, 5, 10), (0, 8, 10, 15), (0, 13, 15, 20)]
        );
        assert_eq!(
            bounds(&time_series_split_with(20, &o(4, Some(3), 1, None)).unwrap()),
            vec![
                (0, 7, 8, 11),
                (0, 10, 11, 14),
                (0, 13, 14, 17),
                (0, 16, 17, 20)
            ]
        );
        assert_eq!(
            bounds(&time_series_split_with(20, &o(3, None, 1, Some(5))).unwrap()),
            vec![(0, 4, 5, 10), (4, 9, 10, 15), (9, 14, 15, 20)]
        );
        assert!(time_series_split_with(10, &o(3, Some(3), 1, None)).is_err());
        let min = TimeSeriesSplitOptions {
            min_train_size: Some(6),
            ..TimeSeriesSplitOptions::new(3)
        };
        assert!(time_series_split_with(20, &min).is_err());
    }

    #[test]
    fn group_time_series_split_keeps_time_points_together() {
        // Two samples per day, days unsorted in the input.
        let days = [3, 1, 2, 1, 3, 2, 4, 4, 5, 5, 6, 6];
        let s = group_time_series_split(&days, &TimeSeriesSplitOptions::new(2)).unwrap();
        assert_eq!(s.len(), 2);
        for (train, test) in &s {
            let max_train = train.iter().map(|&i| days[i]).max().unwrap();
            let min_test = test.iter().map(|&i| days[i]).min().unwrap();
            assert!(max_train < min_test);
            assert_eq!(test.len(), 4); // 2 days x 2 samples
        }
    }

    #[test]
    fn quantile_stratification() {
        let y = array![5.0, 1.0, 3.0, 3.0, 9.0, 0.0, 7.0, 2.0, 8.0, 6.0];
        // Same bins as anofox-regression's quantile_strata test.
        assert_eq!(quantile_strata(&y, 5), vec![2, 0, 1, 1, 4, 0, 3, 1, 4, 3]);
        let yc = Array1::from_shape_fn(50, |i| ((i * 37) % 50) as f64 * 0.1);
        let s = stratified_k_fold_continuous(&yc, 5, 5, 0).unwrap();
        // Every fold gets two samples of every quantile bin.
        let strata = quantile_strata(&yc, 5);
        for (_, test) in &s {
            for b in 0..5 {
                assert_eq!(test.iter().filter(|&&i| strata[i] == b).count(), 2);
            }
        }
    }
}
