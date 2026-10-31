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

use ndarray::Array1;
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
}
