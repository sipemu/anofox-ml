//! Resampling-based model validation beyond plain cross-validation scores.
//!
//! - [`permutation_test_score`]: significance of a cross-validated score
//!   against label-permuted refits (scikit-learn `permutation_test_score`,
//!   Ojala & Garriga 2010).
//! - [`bootstrap_splits`] / [`bootstrap_validate`]: bootstrap estimates of
//!   the prediction error — out-of-bag, Efron's .632, the .632+ estimator
//!   (Efron & Tibshirani 1997) and the optimism-corrected (Harrell) error.

use ndarray::{Array1, Array2, Axis};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;

use crate::error::{Result, RustMlError};
use crate::float::Float;
use crate::utils::{cross_val_score_with_splits, CvSplit};

/// Result of [`permutation_test_score`].
#[derive(Debug, Clone, PartialEq)]
pub struct PermutationTestResult<F> {
    /// Mean cross-validated score on the true labels.
    pub score: F,
    /// Mean cross-validated score for every label permutation.
    pub permutation_scores: Vec<F>,
    /// `(C + 1) / (n_permutations + 1)`, where `C` counts permutations
    /// scoring at least as well as `score`.
    pub p_value: f64,
}

/// `y` with its entries permuted (within each group when `groups` is given).
fn permute_labels<F: Float>(y: &Array1<F>, groups: Option<&Array1<usize>>, seed: u64) -> Array1<F> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let n = y.len();
    let mut order: Vec<usize> = (0..n).collect();
    match groups {
        None => order.shuffle(&mut rng),
        Some(g) => {
            let mut labels: Vec<usize> = g.iter().copied().collect();
            labels.sort_unstable();
            labels.dedup();
            for label in labels {
                let members: Vec<usize> = (0..n).filter(|&i| g[i] == label).collect();
                let mut shuffled = members.clone();
                shuffled.shuffle(&mut rng);
                for (&i, &j) in members.iter().zip(shuffled.iter()) {
                    order[i] = j;
                }
            }
        }
    }
    y.select(Axis(0), &order)
}

/// Permutation test of a cross-validated score (scikit-learn
/// `permutation_test_score`).
///
/// Computes the mean score over `splits` on the true labels and on
/// `n_permutations` copies of `y` with permuted labels (permuted within
/// groups when `groups` is given, so group structure is preserved). The
/// p-value tests the null hypothesis that features and labels are
/// independent; the scorer must be *greater is better*. Permutations run in
/// parallel and are seeded from `seed + permutation_index`.
#[allow(clippy::too_many_arguments)]
pub fn permutation_test_score<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    splits: &[CvSplit],
    groups: Option<&Array1<usize>>,
    n_permutations: usize,
    seed: u64,
    fit_predict: impl Fn(&Array2<F>, &Array1<F>, &Array2<F>) -> Result<Array1<F>> + Sync,
    scorer: impl Fn(&Array1<F>, &Array1<F>) -> Result<F> + Sync,
) -> Result<PermutationTestResult<F>> {
    if n_permutations == 0 {
        return Err(RustMlError::InvalidParameter(
            "n_permutations must be >= 1".into(),
        ));
    }
    if let Some(g) = groups {
        if g.len() != y.len() {
            return Err(RustMlError::ShapeMismatch(format!(
                "groups has {} elements but y has {}",
                g.len(),
                y.len()
            )));
        }
    }
    let mean = |s: Vec<F>| s.iter().fold(F::zero(), |a, &b| a + b) / F::from(s.len()).unwrap();
    let score = mean(cross_val_score_with_splits(
        x,
        y,
        splits,
        &fit_predict,
        &scorer,
    )?);
    let permutation_scores = (0..n_permutations)
        .into_par_iter()
        .map(|p| {
            let yp = permute_labels(y, groups, seed.wrapping_add(p as u64));
            cross_val_score_with_splits(x, &yp, splits, &fit_predict, &scorer).map(mean)
        })
        .collect::<Result<Vec<F>>>()?;
    let at_least = permutation_scores.iter().filter(|&&s| s >= score).count();
    Ok(PermutationTestResult {
        score,
        permutation_scores,
        p_value: (at_least + 1) as f64 / (n_permutations + 1) as f64,
    })
}

/// `n_bootstraps` bootstrap splits: the training side is a sample of `n`
/// indices drawn **with replacement** (so it contains duplicates), the test
/// side the out-of-bag indices (ascending). Seeded from `seed`.
pub fn bootstrap_splits(n: usize, n_bootstraps: usize, seed: u64) -> Result<Vec<CvSplit>> {
    check_bootstrap(n, n_bootstraps)?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    Ok((0..n_bootstraps)
        .map(|_| {
            let train: Vec<usize> = (0..n)
                .map(|_| rand::Rng::gen_range(&mut rng, 0..n))
                .collect();
            (train.clone(), out_of_bag(n, &train))
        })
        .collect())
}

/// Stratified bootstrap: every class is resampled with replacement on its
/// own, so each bootstrap sample keeps the class counts of `y`.
pub fn bootstrap_splits_stratified<F: Float>(
    y: &Array1<F>,
    n_bootstraps: usize,
    seed: u64,
) -> Result<Vec<CvSplit>> {
    let n = y.len();
    check_bootstrap(n, n_bootstraps)?;
    let classes = crate::utils::group_by_class(y);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    Ok((0..n_bootstraps)
        .map(|_| {
            let mut train = Vec::with_capacity(n);
            for members in &classes {
                for _ in 0..members.len() {
                    train.push(members[rand::Rng::gen_range(&mut rng, 0..members.len())]);
                }
            }
            train.sort_unstable();
            (train.clone(), out_of_bag(n, &train))
        })
        .collect())
}

fn check_bootstrap(n: usize, n_bootstraps: usize) -> Result<()> {
    if n < 2 {
        return Err(RustMlError::InvalidParameter(
            "bootstrap needs at least 2 samples".into(),
        ));
    }
    if n_bootstraps == 0 {
        return Err(RustMlError::InvalidParameter(
            "n_bootstraps must be >= 1".into(),
        ));
    }
    Ok(())
}

fn out_of_bag(n: usize, train: &[usize]) -> Vec<usize> {
    let mut in_bag = vec![false; n];
    for &i in train {
        in_bag[i] = true;
    }
    (0..n).filter(|&i| !in_bag[i]).collect()
}

/// Bootstrap estimates of the prediction error (lower is better).
#[derive(Debug, Clone, PartialEq)]
pub struct BootstrapValidation<F> {
    /// Apparent (resubstitution) error of the model fitted on all data.
    pub apparent: F,
    /// Mean out-of-bag error over the bootstrap samples that leave at least
    /// one sample out (Efron's `Err^(1)`, averaged per bootstrap).
    pub out_of_bag: F,
    /// Efron's .632 estimate `0.368 · apparent + 0.632 · out_of_bag`.
    pub point632: F,
    /// Efron & Tibshirani's .632+ estimate, which moves towards the
    /// out-of-bag error when the model overfits.
    pub point632_plus: F,
    /// Optimism-corrected error `apparent + mean(optimism)` (Harrell), with
    /// `optimism_b = loss(fit_b on all data) − loss(fit_b on its bootstrap
    /// sample)`.
    pub optimism_corrected: F,
    /// No-information error rate γ: the loss of the full-data predictions
    /// against every permutation pairing of targets and predictions.
    pub no_information: F,
    /// Relative overfitting rate `R ∈ [0, 1]` used by .632+.
    pub relative_overfitting: F,
    /// Bootstrap samples with an empty out-of-bag set (skipped for
    /// `out_of_bag`).
    pub n_skipped: usize,
}

/// Maximum number of `(yᵢ, ŷⱼ)` pairs used for the no-information rate;
/// larger problems use a seeded random subset of pairs.
pub const MAX_NO_INFORMATION_PAIRS: usize = 1_000_000;

/// Bootstrap validation of a modelling procedure (`fit_predict(x_train,
/// y_train, x_test) -> predictions`) under a loss (`loss(y_true, y_pred)`,
/// lower is better), over `splits` from [`bootstrap_splits`] /
/// [`bootstrap_splits_stratified`].
///
/// Returns the apparent, out-of-bag, .632, .632+ and optimism-corrected
/// errors at once ([`BootstrapValidation`]). The no-information rate uses
/// all `n²` target/prediction pairs up to [`MAX_NO_INFORMATION_PAIRS`] and a
/// random subset (seeded by `seed`) beyond.
pub fn bootstrap_validate<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    splits: &[CvSplit],
    seed: u64,
    fit_predict: impl Fn(&Array2<F>, &Array1<F>, &Array2<F>) -> Result<Array1<F>> + Sync,
    loss: impl Fn(&Array1<F>, &Array1<F>) -> Result<F> + Sync,
) -> Result<BootstrapValidation<F>> {
    let n = y.len();
    if x.nrows() != n {
        return Err(RustMlError::ShapeMismatch(format!(
            "X has {} rows but y has {n} elements",
            x.nrows()
        )));
    }
    if splits.is_empty() {
        return Err(RustMlError::InvalidParameter(
            "need at least one bootstrap split".into(),
        ));
    }
    if let Some(&bad) = splits
        .iter()
        .flat_map(|(a, b)| a.iter().chain(b))
        .find(|&&i| i >= n)
    {
        return Err(RustMlError::InvalidParameter(format!(
            "split index {bad} >= n_samples ({n})"
        )));
    }
    let full_pred = fit_predict(x, y, x)?;
    if full_pred.len() != n {
        return Err(RustMlError::ShapeMismatch(
            "fit_predict must return one prediction per test row".into(),
        ));
    }
    let apparent = loss(y, &full_pred)?;

    // Per bootstrap: (optimism, Some(oob loss)) — one fit, predicting the
    // bootstrap sample and the full data in one call.
    let per_boot = splits
        .par_iter()
        .map(|(train, test)| -> Result<(F, Option<F>)> {
            let xb = x.select(Axis(0), train);
            let yb = y.select(Axis(0), train);
            let stacked = ndarray::concatenate(Axis(0), &[xb.view(), x.view()])
                .map_err(|e| RustMlError::ShapeMismatch(e.to_string()))?;
            let pred = fit_predict(&xb, &yb, &stacked)?;
            if pred.len() != train.len() + n {
                return Err(RustMlError::ShapeMismatch(
                    "fit_predict must return one prediction per test row".into(),
                ));
            }
            let pred_boot = pred.slice(ndarray::s![..train.len()]).to_owned();
            let pred_full = pred.slice(ndarray::s![train.len()..]).to_owned();
            let optimism = loss(y, &pred_full)? - loss(&yb, &pred_boot)?;
            let oob = if test.is_empty() {
                None
            } else {
                Some(loss(
                    &y.select(Axis(0), test),
                    &pred_full.select(Axis(0), test),
                )?)
            };
            Ok((optimism, oob))
        })
        .collect::<Result<Vec<_>>>()?;
    let nf = |v: usize| F::from(v).unwrap();
    let optimism = per_boot.iter().fold(F::zero(), |a, (o, _)| a + *o) / nf(per_boot.len());
    let oobs: Vec<F> = per_boot.iter().filter_map(|(_, o)| *o).collect();
    if oobs.is_empty() {
        return Err(RustMlError::InvalidParameter(
            "every bootstrap sample has an empty out-of-bag set".into(),
        ));
    }
    let out_of_bag = oobs.iter().fold(F::zero(), |a, &b| a + b) / nf(oobs.len());

    // No-information rate γ over (y_i, ŷ_j) pairs.
    let (yi, pj): (Vec<F>, Vec<F>) = if n * n <= MAX_NO_INFORMATION_PAIRS {
        (0..n * n).map(|t| (y[t / n], full_pred[t % n])).unzip()
    } else {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..MAX_NO_INFORMATION_PAIRS)
            .map(|_| {
                let (i, j) = (
                    rand::Rng::gen_range(&mut rng, 0..n),
                    rand::Rng::gen_range(&mut rng, 0..n),
                );
                (y[i], full_pred[j])
            })
            .unzip()
    };
    let gamma = loss(&Array1::from_vec(yi), &Array1::from_vec(pj))?;

    let c368 = F::from(0.368).unwrap();
    let c632 = F::from(0.632).unwrap();
    let point632 = c368 * apparent + c632 * out_of_bag;
    let oob_capped = if out_of_bag < gamma {
        out_of_bag
    } else {
        gamma
    };
    let r = if oob_capped > apparent && gamma > apparent {
        let v = (oob_capped - apparent) / (gamma - apparent);
        if v > F::one() {
            F::one()
        } else {
            v
        }
    } else {
        F::zero()
    };
    let w = c632 / (F::one() - c368 * r);
    let point632_plus = (F::one() - w) * apparent + w * oob_capped;
    Ok(BootstrapValidation {
        apparent,
        out_of_bag,
        point632,
        point632_plus,
        optimism_corrected: apparent + optimism,
        no_information: gamma,
        relative_overfitting: r,
        n_skipped: per_boot.len() - oobs.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Nearest-class-mean classifier on one feature.
    fn centroid_fit_predict(
        xt: &Array2<f64>,
        yt: &Array1<f64>,
        xs: &Array2<f64>,
    ) -> Result<Array1<f64>> {
        let mean = |c: f64| {
            let v: Vec<f64> = (0..yt.len())
                .filter(|&i| yt[i] == c)
                .map(|i| xt[[i, 0]])
                .collect();
            if v.is_empty() {
                f64::NAN
            } else {
                v.iter().sum::<f64>() / v.len() as f64
            }
        };
        let (m0, m1) = (mean(0.0), mean(1.0));
        Ok(xs.column(0).mapv(|v| {
            if m1.is_nan() || (!m0.is_nan() && (v - m0).abs() <= (v - m1).abs()) {
                0.0
            } else {
                1.0
            }
        }))
    }

    fn accuracy(t: &Array1<f64>, p: &Array1<f64>) -> Result<f64> {
        Ok(t.iter().zip(p).filter(|(a, b)| a == b).count() as f64 / t.len() as f64)
    }

    #[test]
    fn separable_problem_has_minimal_p_value() {
        let n = 40;
        let x = Array2::from_shape_fn((n, 1), |(i, _)| {
            if i % 2 == 0 {
                i as f64 * 0.01
            } else {
                5.0 + i as f64 * 0.01
            }
        });
        let y = Array1::from_shape_fn(n, |i| (i % 2) as f64);
        let splits = crate::utils::shuffled_k_fold(n, 5, 0).unwrap();
        let r =
            permutation_test_score(&x, &y, &splits, None, 99, 1, centroid_fit_predict, accuracy)
                .unwrap();
        assert_eq!(r.score, 1.0);
        assert_eq!(r.permutation_scores.len(), 99);
        assert!((r.p_value - 0.01).abs() < 1e-12);
        // Deterministic per seed.
        let r2 =
            permutation_test_score(&x, &y, &splits, None, 99, 1, centroid_fit_predict, accuracy)
                .unwrap();
        assert_eq!(r, r2);
    }

    #[test]
    fn uninformative_feature_is_not_significant() {
        let n = 40;
        let x = Array2::from_shape_fn((n, 1), |(i, _)| ((i * 7) % 11) as f64);
        let y = Array1::from_shape_fn(n, |i| ((i * 3 + i / 5) % 2) as f64);
        let splits = crate::utils::shuffled_k_fold(n, 4, 2).unwrap();
        let r = permutation_test_score(
            &x,
            &y,
            &splits,
            None,
            199,
            3,
            centroid_fit_predict,
            accuracy,
        )
        .unwrap();
        assert!(r.p_value > 0.05, "p = {}", r.p_value);
    }

    #[test]
    fn group_permutation_stays_within_groups() {
        let y = array![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let g = array![0, 0, 0, 1, 1, 1];
        for s in 0..20 {
            let p = permute_labels(&y, Some(&g), s);
            let mut a: Vec<f64> = p.slice(ndarray::s![..3]).to_vec();
            a.sort_by(f64::total_cmp);
            assert_eq!(a, vec![1.0, 2.0, 3.0]);
        }
        let x = Array2::<f64>::zeros((6, 1));
        let splits = crate::utils::k_fold(6, 2).unwrap();
        assert!(permutation_test_score(
            &x,
            &y,
            &splits,
            Some(&array![0, 1]),
            5,
            0,
            centroid_fit_predict,
            accuracy
        )
        .is_err());
    }

    fn mse(t: &Array1<f64>, p: &Array1<f64>) -> Result<f64> {
        Ok(t.iter().zip(p).map(|(a, b)| (a - b) * (a - b)).sum::<f64>() / t.len() as f64)
    }

    /// 1-nearest-neighbour regressor: memorises the training data.
    fn one_nn(xt: &Array2<f64>, yt: &Array1<f64>, xs: &Array2<f64>) -> Result<Array1<f64>> {
        Ok(xs.column(0).mapv(|v| {
            let mut best = 0;
            for i in 1..yt.len() {
                if (xt[[i, 0]] - v).abs() < (xt[[best, 0]] - v).abs() {
                    best = i;
                }
            }
            yt[best]
        }))
    }

    /// Mean predictor: cannot overfit.
    fn mean_model(_: &Array2<f64>, yt: &Array1<f64>, xs: &Array2<f64>) -> Result<Array1<f64>> {
        Ok(Array1::from_elem(xs.nrows(), yt.mean().unwrap()))
    }

    #[test]
    fn bootstrap_splits_are_out_of_bag() {
        let s = bootstrap_splits(30, 20, 4).unwrap();
        assert_eq!(s.len(), 20);
        for (train, test) in &s {
            assert_eq!(train.len(), 30);
            assert!(test.iter().all(|i| !train.contains(i)));
            let mut all: Vec<usize> = train.iter().chain(test).copied().collect();
            all.sort_unstable();
            all.dedup();
            assert_eq!(all.len(), 30);
        }
        assert_eq!(s, bootstrap_splits(30, 20, 4).unwrap());
        let y = Array1::from_shape_fn(30, |i| (i % 3) as f64);
        for (train, _) in bootstrap_splits_stratified(&y, 5, 1).unwrap() {
            for c in 0..3 {
                assert_eq!(train.iter().filter(|&&i| y[i] == c as f64).count(), 10);
            }
        }
    }

    #[test]
    fn point632_plus_corrects_an_overfitting_model() {
        // Pure-noise target: 1-NN memorises (apparent error 0) but the
        // out-of-bag error is about twice the variance.
        let n = 60;
        let x = Array2::from_shape_fn((n, 1), |(i, _)| i as f64);
        let y = Array1::from_shape_fn(n, |i| ((i * 37 + 11) % 17) as f64);
        let splits = bootstrap_splits(n, 100, 0).unwrap();
        let r = bootstrap_validate(&x, &y, &splits, 0, one_nn, mse).unwrap();
        assert_eq!(r.apparent, 0.0);
        assert!(r.out_of_bag > 0.0);
        assert!((r.point632 - 0.632 * r.out_of_bag).abs() < 1e-12);
        assert!(
            r.relative_overfitting > 0.9,
            "R = {}",
            r.relative_overfitting
        );
        assert!(r.point632_plus > r.point632);
        assert!(r.point632_plus <= r.out_of_bag.min(r.no_information) + 1e-12);
        assert!(r.optimism_corrected > r.apparent);
    }

    #[test]
    fn point632_plus_without_overfitting_caps_at_no_information() {
        let n = 40;
        let x = Array2::from_shape_fn((n, 1), |(i, _)| i as f64);
        let y = Array1::from_shape_fn(n, |i| ((i * 13) % 7) as f64);
        let splits = bootstrap_splits(n, 50, 9).unwrap();
        let r = bootstrap_validate(&x, &y, &splits, 0, mean_model, mse).unwrap();
        // The mean model's apparent error equals its no-information rate,
        // so R = 0 and .632+ = 0.368·apparent + 0.632·min(OOB, γ)
        // (Efron & Tibshirani cap the out-of-bag error at γ).
        assert!((r.no_information - r.apparent).abs() < 1e-9);
        assert_eq!(r.relative_overfitting, 0.0);
        let expected = 0.368 * r.apparent + 0.632 * r.out_of_bag.min(r.no_information);
        assert!((r.point632_plus - expected).abs() < 1e-12);
    }
}
