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
}
