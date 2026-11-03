//! Successive halving for hyperparameter search.
//!
//! Mirrors `sklearn.model_selection.HalvingGridSearchCV` and
//! `HalvingRandomSearchCV`. At each iteration all surviving candidates are
//! scored by K-fold cross-validation on a subsample of `n_resources` rows; the
//! top `ceil(n_candidates / factor)` survive and the resources grow by
//! `factor`.
//!
//! The schedule follows sklearn exactly: `n_iterations = min(1 +
//! floor(log_factor(n_candidates)), 1 + floor(log_factor(n_samples /
//! min_resources)))` and iteration `i` uses `min_resources * factor^i` rows.
//! The best candidate is the best one of the **last** iteration.
//!
//! The subsample of iteration `i` is a prefix of a seeded permutation of the
//! rows (so subsamples are nested). When `y` looks like class labels (see
//! [`looks_like_class_labels`]) the permutation interleaves the classes, so
//! every prefix keeps the class proportions, and the per-iteration CV is
//! stratified ([`default_cv_splits`]).
//!
//! Scores are ranked with a NaN-safe total order: NaN scores (e.g. a diverged
//! fit) rank last and are eliminated first; the search never panics on them.

use std::cmp::Ordering;

use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::error::{Result, RustMlError};
use crate::float::Float;
use crate::utils::{default_cv_splits, group_by_class, looks_like_class_labels, CvSplit};

/// Default number of CV folds per iteration (sklearn's `cv=5`), capped by the
/// number of rows in the iteration's subsample.
pub const DEFAULT_HALVING_CV: usize = 5;

/// Result of a halving search (the useful parts of sklearn's `cv_results_`).
#[derive(Debug, Clone)]
pub struct HalvingResult<F: Float> {
    /// Index of the best candidate of the last iteration.
    pub best_params_index: usize,
    /// Its mean CV score in the last iteration (NaN only if every candidate of
    /// the last iteration scored NaN).
    pub best_score: F,
    /// Mean CV score per candidate in the last iteration; NaN for candidates
    /// eliminated earlier.
    pub final_scores: Vec<F>,
    /// Number of iterations run (sklearn `n_iterations_`).
    pub rounds: usize,
    /// Rows used per iteration (sklearn `n_resources_`).
    pub n_resources: Vec<usize>,
    /// Candidates evaluated per iteration (sklearn `n_candidates_`).
    pub n_candidates: Vec<usize>,
    /// `iteration_scores[i][c]`: mean CV score of candidate `c` in iteration
    /// `i`, NaN if `c` was not evaluated in that iteration.
    pub iteration_scores: Vec<Vec<F>>,
}

/// `floor(log_base(x))` for integers `x >= 1`, `base >= 2` (exact).
fn ilog_floor(x: usize, base: usize) -> usize {
    let mut p = 0;
    let mut v = base;
    while v <= x {
        p += 1;
        match v.checked_mul(base) {
            Some(nv) => v = nv,
            None => break,
        }
    }
    p
}

/// Descending order with NaN last (never panics).
fn cmp_desc_nan_last<F: Float>(a: F, b: F) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => b.to_f64().unwrap().total_cmp(&a.to_f64().unwrap()),
    }
}

/// Seeded row order whose prefixes are (approximately) stratified when `y`
/// holds class labels, and a plain shuffle otherwise.
fn subsample_order<F: Float>(y: &Array1<F>, seed: u64) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(seed);
    let n = y.len();
    if !looks_like_class_labels(y) {
        let mut idx: Vec<usize> = (0..n).collect();
        idx.shuffle(&mut rng);
        return idx;
    }
    // Member j of a class of size m gets key (j + 0.5) / m: sorting by key
    // interleaves the classes proportionally; ties are broken randomly.
    let mut keyed: Vec<(f64, f64, usize)> = Vec::with_capacity(n);
    for mut members in group_by_class(y.view()) {
        members.shuffle(&mut rng);
        let m = members.len() as f64;
        for (j, i) in members.into_iter().enumerate() {
            keyed.push(((j as f64 + 0.5) / m, rng.gen::<f64>(), i));
        }
    }
    keyed.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    keyed.into_iter().map(|(_, _, i)| i).collect()
}

/// Successive halving over `param_configs`, scoring each candidate by
/// `min(5, n_resources)`-fold CV per iteration ([`default_cv_splits`]:
/// stratified for class-like `y`, shuffled K-fold otherwise).
///
/// `factor` is the rate at which candidates are eliminated and resources
/// grown. `min_resources` is the subsample size of the first iteration.
/// `param_configs[i]` is a closure `(x_train, y_train, x_eval) -> predictions`
/// that fits + predicts for candidate `i`; `scorer` is higher-is-better. An
/// error from a candidate or the scorer aborts the search; a NaN score just
/// ranks the candidate last.
pub fn halving_grid_search_cv<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    factor: usize,
    min_resources: usize,
    seed: u64,
    param_configs: &[impl Fn(&Array2<F>, &Array1<F>, &Array2<F>) -> Result<Array1<F>>],
    scorer: impl Fn(&Array1<F>, &Array1<F>) -> Result<F>,
) -> Result<HalvingResult<F>> {
    halving_grid_search_cv_with_splitter(
        x,
        y,
        factor,
        min_resources,
        seed,
        param_configs,
        scorer,
        |xs: &Array2<F>, ys: &Array1<F>, s: u64| {
            default_cv_splits(xs, ys, DEFAULT_HALVING_CV.min(ys.len()), s)
        },
    )
}

/// [`halving_grid_search_cv`] with a pluggable per-iteration CV splitter.
///
/// `splitter(x_sub, y_sub, seed)` receives the iteration's subsample and must
/// return `(train, test)` index pairs into it (e.g. [`crate::utils::shuffled_k_fold`],
/// [`crate::utils::stratified_k_fold`], [`crate::splitters::group_k_fold`]). Each
/// candidate's iteration score is the mean of its per-split scores.
#[allow(clippy::too_many_arguments)]
pub fn halving_grid_search_cv_with_splitter<F, P, S, Sp>(
    x: &Array2<F>,
    y: &Array1<F>,
    factor: usize,
    min_resources: usize,
    seed: u64,
    param_configs: &[P],
    scorer: S,
    splitter: Sp,
) -> Result<HalvingResult<F>>
where
    F: Float,
    P: Fn(&Array2<F>, &Array1<F>, &Array2<F>) -> Result<Array1<F>>,
    S: Fn(&Array1<F>, &Array1<F>) -> Result<F>,
    Sp: Fn(&Array2<F>, &Array1<F>, u64) -> Result<Vec<CvSplit>>,
{
    if param_configs.is_empty() {
        return Err(RustMlError::InvalidParameter("no candidates".into()));
    }
    if factor < 2 {
        return Err(RustMlError::InvalidParameter("factor must be ≥ 2".into()));
    }
    let n = x.nrows();
    if y.len() != n {
        return Err(RustMlError::ShapeMismatch(format!(
            "X has {} rows but y has {} elements",
            n,
            y.len()
        )));
    }
    if min_resources < 2 || min_resources > n {
        return Err(RustMlError::InvalidParameter(format!(
            "min_resources must be in 2..={}",
            n
        )));
    }

    let n_total = param_configs.len();
    let n_required = 1 + ilog_floor(n_total, factor);
    let n_possible = 1 + ilog_floor(n / min_resources, factor);
    let n_iterations = n_required.min(n_possible);

    let order = subsample_order(y, seed);

    let mut surviving: Vec<usize> = (0..n_total).collect();
    let mut n_resources = Vec::with_capacity(n_iterations);
    let mut n_candidates = Vec::with_capacity(n_iterations);
    let mut iteration_scores: Vec<Vec<F>> = Vec::with_capacity(n_iterations);
    let mut resources = min_resources;

    for iter in 0..n_iterations {
        if iter > 0 {
            resources *= factor; // <= n by construction of n_possible
        }
        let used = resources.min(n);
        let subset = &order[..used];
        let xs = select_rows(x, subset);
        let ys = select_elements(y, subset);
        let splits = splitter(&xs, &ys, seed)?;
        if splits.is_empty() {
            return Err(RustMlError::InvalidParameter(
                "splitter returned no (train, test) splits".into(),
            ));
        }
        let folds: Vec<_> = splits
            .iter()
            .map(|(tr, te)| {
                if tr.is_empty() || te.is_empty() || tr.iter().chain(te).any(|&i| i >= used) {
                    return Err(RustMlError::InvalidParameter(
                        "splitter returned an empty or out-of-range split".into(),
                    ));
                }
                Ok((
                    select_rows(&xs, tr),
                    select_elements(&ys, tr),
                    select_rows(&xs, te),
                    select_elements(&ys, te),
                ))
            })
            .collect::<Result<_>>()?;

        let mut scores = vec![F::nan(); n_total];
        for &cand in &surviving {
            let mut sum = F::zero();
            for (x_tr, y_tr, x_te, y_te) in &folds {
                let pred = param_configs[cand](x_tr, y_tr, x_te)?;
                sum += scorer(y_te, &pred)?;
            }
            scores[cand] = sum / F::from_usize(folds.len()).unwrap();
        }
        n_resources.push(used);
        n_candidates.push(surviving.len());

        // Rank (stable: ties keep candidate order), NaN last.
        surviving.sort_by(|&a, &b| cmp_desc_nan_last(scores[a], scores[b]));
        iteration_scores.push(scores);
        if iter + 1 < n_iterations {
            let keep = surviving.len().div_ceil(factor).max(1);
            surviving.truncate(keep);
            surviving.sort_unstable();
        }
    }

    let last = iteration_scores.last().unwrap().clone();
    let best_params_index = surviving[0];
    Ok(HalvingResult {
        best_params_index,
        best_score: last[best_params_index],
        final_scores: last,
        rounds: n_iterations,
        n_resources,
        n_candidates,
        iteration_scores,
    })
}

/// HalvingRandomSearchCV: same successive-halving loop but the candidate set
/// is generated by sampling a user-supplied factory `n_candidates` times.
///
/// `factory(rng_seed)` should return a per-call closure that fits + predicts
/// — the seed is what the caller uses to draw a random hyperparameter point.
#[allow(clippy::too_many_arguments)]
pub fn halving_random_search_cv<F, MkFit, FitFn>(
    x: &Array2<F>,
    y: &Array1<F>,
    n_candidates: usize,
    factor: usize,
    min_resources: usize,
    seed: u64,
    factory: MkFit,
    scorer: impl Fn(&Array1<F>, &Array1<F>) -> Result<F>,
) -> Result<HalvingResult<F>>
where
    F: Float,
    MkFit: Fn(u64) -> FitFn,
    FitFn: Fn(&Array2<F>, &Array1<F>, &Array2<F>) -> Result<Array1<F>>,
{
    if n_candidates == 0 {
        return Err(RustMlError::InvalidParameter(
            "n_candidates must be >= 1".into(),
        ));
    }
    // Draw `n_candidates` candidate closures from the factory.
    let candidates: Vec<_> = (0..n_candidates)
        .map(|i| factory(seed.wrapping_add(i as u64).wrapping_add(1)))
        .collect();
    halving_grid_search_cv(x, y, factor, min_resources, seed, &candidates, scorer)
}

fn select_rows<F: Float>(x: &Array2<F>, indices: &[usize]) -> Array2<F> {
    x.select(ndarray::Axis(0), indices)
}

fn select_elements<F: Float>(y: &Array1<F>, indices: &[usize]) -> Array1<F> {
    Array1::from_vec(indices.iter().map(|&i| y[i]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_halving_random_search_with_random_factory() {
        // 10 random candidates: factory draws a slope from a per-seed PRNG.
        // The "true" slope is 2.0; we want halving to converge on a slope
        // close to that.
        let x = Array2::from_shape_vec((40, 1), (0..40).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_vec((0..40).map(|i| 2.0 * i as f64).collect());

        let factory = |s: u64| {
            // Deterministic per-seed candidate: pretend we drew a slope from
            // some distribution that depends on `s`. Slopes range 0.5..3.5.
            let slope = 0.5 + (s as f64 % 30.0) * 0.1;
            move |_xt: &Array2<f64>, _yt: &Array1<f64>, xv: &Array2<f64>| {
                Ok(xv.column(0).mapv(|v| slope * v))
            }
        };

        let scorer = |yt: &Array1<f64>, yp: &Array1<f64>| -> Result<f64> {
            let mse: f64 = yt
                .iter()
                .zip(yp.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                / yt.len() as f64;
            Ok(-mse)
        };
        let result = halving_random_search_cv(&x, &y, 10, 2, 8, 42, factory, scorer).unwrap();
        // The winning candidate's slope should be close to 2.0.
        let winning_seed = 42_u64
            .wrapping_add(result.best_params_index as u64)
            .wrapping_add(1);
        let winning_slope = 0.5 + (winning_seed as f64 % 30.0) * 0.1;
        assert!(
            (winning_slope - 2.0).abs() < 0.5,
            "winning slope={} too far from true=2.0",
            winning_slope
        );
    }

    #[test]
    fn test_halving_picks_better_candidate() {
        let x = Array2::from_shape_vec((40, 1), (0..40).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_vec((0..40).map(|i| 2.0 * i as f64 + 3.0).collect());

        // Two candidates: one predicts true line, the other predicts zeros.
        let good: Cand = Box::new(|_xt, _yt, xv| Ok(xv.column(0).mapv(|v| 2.0 * v + 3.0)));
        let bad: Cand = Box::new(|_xt, _yt, xv| Ok(Array1::<f64>::zeros(xv.nrows())));

        let candidates = vec![good, bad];
        let result = halving_grid_search_cv(&x, &y, 2, 8, 0, &candidates, |yt, yp| {
            let mse: f64 = yt
                .iter()
                .zip(yp.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                / yt.len() as f64;
            Ok(-mse)
        })
        .unwrap();
        assert_eq!(result.best_params_index, 0);
        let _ = array![1.0_f64];
    }

    type Cand = Box<dyn Fn(&Array2<f64>, &Array1<f64>, &Array2<f64>) -> Result<Array1<f64>>>;

    fn neg_mse(yt: &Array1<f64>, yp: &Array1<f64>) -> Result<f64> {
        Ok(-yt
            .iter()
            .zip(yp.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / yt.len() as f64)
    }

    /// Candidate `i` predicts `slope_i * x` with slopes spread around 2.
    fn slope_candidates(k: usize) -> Vec<Cand> {
        (0..k)
            .map(|i| {
                let slope = 0.5 + 0.35 * i as f64;
                Box::new(
                    move |_xt: &Array2<f64>, _yt: &Array1<f64>, xv: &Array2<f64>| {
                        Ok(xv.column(0).mapv(|v| slope * v))
                    },
                ) as Cand
            })
            .collect()
    }

    fn line(n: usize) -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec((n, 1), (0..n).map(|i| i as f64).collect()).unwrap();
        let y = x.column(0).mapv(|v| 2.0 * v);
        (x, y)
    }

    #[test]
    fn test_halving_schedule_matches_sklearn() {
        // sklearn HalvingGridSearchCV n_resources_ / n_candidates_ for
        // (n_candidates, factor, min_resources, n_samples).
        type Case = (
            usize,
            usize,
            usize,
            usize,
            &'static [usize],
            &'static [usize],
        );
        let cases: [Case; 5] = [
            (10, 3, 8, 100, &[8, 24, 72], &[10, 4, 2]),
            (10, 2, 8, 40, &[8, 16, 32], &[10, 5, 3]),
            (2, 2, 8, 40, &[8, 16], &[2, 1]),
            (7, 2, 5, 100, &[5, 10, 20], &[7, 4, 2]),
            (1, 2, 8, 40, &[8], &[1]),
        ];
        for (k, factor, min_r, n, res, cands) in cases {
            let (x, y) = line(n);
            let r = halving_grid_search_cv(&x, &y, factor, min_r, 0, &slope_candidates(k), neg_mse)
                .unwrap();
            assert_eq!(r.n_resources, res, "case {:?}", (k, factor, min_r, n));
            assert_eq!(r.n_candidates, cands, "case {:?}", (k, factor, min_r, n));
            assert_eq!(r.rounds, res.len());
            assert_eq!(r.iteration_scores.len(), res.len());
            // final_scores: NaN exactly for candidates not in the last iteration.
            let evaluated = r.final_scores.iter().filter(|s| !s.is_nan()).count();
            assert_eq!(evaluated, *cands.last().unwrap());
            assert!(!r.final_scores[r.best_params_index].is_nan());
            assert_eq!(r.best_score, r.final_scores[r.best_params_index]);
        }
    }

    #[test]
    fn test_halving_nan_candidate_is_eliminated_without_panic() {
        let (x, y) = line(60);
        let mut cands = slope_candidates(4);
        // Candidate 4 "diverges": NaN predictions -> NaN score.
        cands.push(Box::new(|_xt, _yt, xv| {
            Ok(Array1::from_elem(xv.nrows(), f64::NAN))
        }));
        let r = halving_grid_search_cv(&x, &y, 2, 10, 3, &cands, neg_mse).unwrap();
        // First iteration evaluates it (NaN), then it is dropped.
        assert!(r.iteration_scores[0][4].is_nan());
        assert_eq!(r.n_candidates[0], 5);
        assert_eq!(r.n_candidates[1], 3);
        assert!(r.iteration_scores[1][4].is_nan());
        assert!(r.final_scores[4].is_nan());
        assert_ne!(r.best_params_index, 4);
        // Real slopes 0.5, 0.85, 1.2, 1.55 -> closest to 2 is index 3.
        assert_eq!(r.best_params_index, 3);
        assert!(r.best_score.is_finite());
    }

    #[test]
    fn test_halving_all_nan_does_not_panic() {
        let (x, y) = line(30);
        let cands: Vec<Cand> = (0..3)
            .map(|_| {
                Box::new(|_xt: &Array2<f64>, _yt: &Array1<f64>, xv: &Array2<f64>| {
                    Ok(Array1::from_elem(xv.nrows(), f64::NAN))
                }) as Cand
            })
            .collect();
        let r = halving_grid_search_cv(&x, &y, 2, 6, 0, &cands, neg_mse).unwrap();
        assert!(r.best_score.is_nan());
    }

    #[test]
    fn test_halving_deterministic_per_seed() {
        // Candidates that depend on the training data so the CV splits matter.
        let n = 48;
        let x =
            Array2::from_shape_vec((n, 1), (0..n).map(|i| (i * 7 % 13) as f64).collect()).unwrap();
        let y = Array1::from_vec((0..n).map(|i| ((i * 5) % 11) as f64 * 0.3).collect());
        let cands: Vec<Cand> = (0..6)
            .map(|k| {
                let shrink = k as f64 * 0.2;
                Box::new(
                    move |_xt: &Array2<f64>, yt: &Array1<f64>, xv: &Array2<f64>| {
                        let m = yt.mean().unwrap();
                        Ok(xv.column(0).mapv(|v| (1.0 - shrink) * m + shrink * v * 0.1))
                    },
                ) as Cand
            })
            .collect();
        let a = halving_grid_search_cv(&x, &y, 2, 6, 11, &cands, neg_mse).unwrap();
        let b = halving_grid_search_cv(&x, &y, 2, 6, 11, &cands, neg_mse).unwrap();
        assert_eq!(a.best_params_index, b.best_params_index);
        assert_eq!(a.n_resources, b.n_resources);
        for (ra, rb) in a.iteration_scores.iter().zip(b.iteration_scores.iter()) {
            for (sa, sb) in ra.iter().zip(rb.iter()) {
                assert!(sa.to_bits() == sb.to_bits());
            }
        }
        // A different seed changes the subsample / folds (scores differ).
        let c = halving_grid_search_cv(&x, &y, 2, 6, 12, &cands, neg_mse).unwrap();
        assert_ne!(
            a.iteration_scores[0]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            c.iteration_scores[0]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_halving_stratified_subsample_and_folds_for_classes() {
        // 80 samples, 20% minority class; min_resources = 10. The stratified
        // prefix holds exactly 2 minority samples and stratified 5-fold puts
        // them in different folds, so every training fold sees both classes
        // (a random prefix / unstratified split regularly would not).
        let n = 80;
        let y = Array1::from_vec((0..n).map(|i| if i % 5 == 0 { 1.0 } else { 0.0 }).collect());
        let x = Array2::from_shape_vec((n, 1), (0..n).map(|i| i as f64).collect()).unwrap();
        let cand: Cand = Box::new(|_xt, yt, xv| {
            let has_both = yt.iter().any(|&v| v == 1.0) && yt.iter().any(|&v| v == 0.0);
            if !has_both {
                return Err(RustMlError::InvalidParameter("single-class fold".into()));
            }
            Ok(Array1::zeros(xv.nrows()))
        });
        let cands = vec![cand];
        let acc = |yt: &Array1<f64>, yp: &Array1<f64>| -> Result<f64> {
            Ok(yt.iter().zip(yp).filter(|(a, b)| a == b).count() as f64 / yt.len() as f64)
        };
        for seed in 0..20 {
            let r = halving_grid_search_cv(&x, &y, 2, 10, seed, &cands, acc).unwrap();
            assert_eq!(r.n_resources, vec![10]);
        }
        // Subsample order keeps class proportions in every prefix.
        let order = subsample_order(&y, 5);
        let minority_in_first_20 = order[..20].iter().filter(|&&i| y[i] == 1.0).count();
        assert_eq!(minority_in_first_20, 4);
    }

    #[test]
    fn test_halving_custom_splitter() {
        let (x, y) = line(40);
        let calls = std::cell::Cell::new(0usize);
        let r = halving_grid_search_cv_with_splitter(
            &x,
            &y,
            2,
            10,
            0,
            &slope_candidates(4),
            neg_mse,
            |xs: &Array2<f64>, _ys: &Array1<f64>, s: u64| {
                calls.set(calls.get() + 1);
                crate::utils::shuffled_k_fold(xs.nrows(), 3, s)
            },
        )
        .unwrap();
        assert_eq!(calls.get(), r.rounds);
        // Splitter errors propagate.
        let err = halving_grid_search_cv_with_splitter(
            &x,
            &y,
            2,
            10,
            0,
            &slope_candidates(2),
            neg_mse,
            |_xs: &Array2<f64>, _ys: &Array1<f64>, _s: u64| Ok(vec![]),
        );
        assert!(err.is_err());
    }
}
