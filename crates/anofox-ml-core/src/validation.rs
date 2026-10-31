//! Resampling-based model validation beyond plain cross-validation scores.
//!
//! - [`permutation_test_score`]: significance of a cross-validated score
//!   against label-permuted refits (scikit-learn `permutation_test_score`,
//!   Ojala & Garriga 2010).

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
}
