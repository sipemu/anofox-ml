//! Model inspection utilities: permutation importance.
//!
//! Mirrors `sklearn.inspection.permutation_importance`. For each feature, the
//! column is permuted `n_repeats` times and the drop in score is recorded;
//! the result is the (mean, std) drop per feature across repetitions.

use ndarray::{Array1, Array2, Axis};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::error::Result;
use crate::traits::Predict;

/// Result of a permutation-importance run.
#[derive(Debug, Clone)]
pub struct PermutationImportance {
    /// Mean drop in score per feature, length = n_features.
    pub importances_mean: Array1<f64>,
    /// Std-dev of the drop per feature across repeats.
    pub importances_std: Array1<f64>,
    /// Per-feature, per-repeat drop matrix (n_features × n_repeats).
    pub importances: Array2<f64>,
}

/// Compute permutation feature importance for a fitted model.
///
/// `score_fn(y_true, y_pred) -> f64` should return higher-is-better (e.g. R²
/// or accuracy). Importance for feature `j` is `baseline_score - score_after_permuting_j`,
/// averaged across `n_repeats` shuffles.
pub fn permutation_importance<M, S>(
    model: &M,
    x: &Array2<f64>,
    y: &Array1<f64>,
    n_repeats: usize,
    seed: u64,
    score_fn: S,
) -> Result<PermutationImportance>
where
    M: Predict<f64>,
    S: Fn(&Array1<f64>, &Array1<f64>) -> f64,
{
    let n_features = x.ncols();
    let n_samples = x.nrows();

    let baseline_pred = model.predict(x)?;
    let baseline_score = score_fn(y, &baseline_pred);

    let mut importances = Array2::<f64>::zeros((n_features, n_repeats));
    let mut rng = StdRng::seed_from_u64(seed);

    for j in 0..n_features {
        for r in 0..n_repeats {
            let mut x_perm = x.clone();
            let mut idx: Vec<usize> = (0..n_samples).collect();
            idx.shuffle(&mut rng);

            // Save original column, write permuted column in-place.
            let original_col: Vec<f64> = x.column(j).iter().copied().collect();
            for (i, &p) in idx.iter().enumerate() {
                x_perm[[i, j]] = original_col[p];
            }

            let perm_pred = model.predict(&x_perm)?;
            let perm_score = score_fn(y, &perm_pred);
            importances[[j, r]] = baseline_score - perm_score;
        }
    }

    let mut mean = Array1::<f64>::zeros(n_features);
    let mut std = Array1::<f64>::zeros(n_features);
    for j in 0..n_features {
        let row = importances.index_axis(Axis(0), j);
        let m = row.sum() / n_repeats as f64;
        let v = row.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n_repeats as f64;
        mean[j] = m;
        std[j] = v.sqrt();
    }

    Ok(PermutationImportance {
        importances_mean: mean,
        importances_std: std,
        importances,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::Predict;
    use ndarray::array;

    // A trivial linear model wired up directly to exercise the importance loop.
    struct Linear {
        coef: Array1<f64>,
        bias: f64,
    }
    impl Predict<f64> for Linear {
        fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
            let mut out = Array1::<f64>::from_elem(x.nrows(), self.bias);
            for i in 0..x.nrows() {
                for j in 0..x.ncols() {
                    out[i] += x[[i, j]] * self.coef[j];
                }
            }
            Ok(out)
        }
    }

    fn neg_mse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        let mse: f64 = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&t, &p)| (t - p).powi(2))
            .sum::<f64>()
            / y_true.len() as f64;
        -mse
    }

    #[test]
    fn test_importance_ranks_by_coefficient_magnitude() {
        // y = 10*x0 + 0*x1 + 0.1*x2; permuting x0 should hurt the most.
        let n = 100;
        let mut x = Array2::<f64>::zeros((n, 3));
        for i in 0..n {
            x[[i, 0]] = (i as f64) - 50.0;
            x[[i, 1]] = ((i * 7 % 13) as f64) - 6.0;
            x[[i, 2]] = ((i * 3 % 11) as f64) - 5.0;
        }
        let y = x.column(0).mapv(|v| 10.0 * v)
            + x.column(2).mapv(|v| 0.1 * v);

        let model = Linear {
            coef: array![10.0, 0.0, 0.1],
            bias: 0.0,
        };
        let r = permutation_importance(&model, &x, &y, 20, 42, neg_mse).unwrap();

        assert!(r.importances_mean[0] > r.importances_mean[1]);
        assert!(r.importances_mean[0] > r.importances_mean[2]);
        // x1 has zero coefficient — permuting it should not hurt.
        assert!(r.importances_mean[1].abs() < 0.1);
        assert_eq!(r.importances.shape(), &[3, 20]);
    }

    #[test]
    fn test_zero_repeats_gives_zero_size_matrix() {
        let x = array![[1.0], [2.0], [3.0]];
        let y = array![1.0, 2.0, 3.0];
        let model = Linear { coef: array![1.0], bias: 0.0 };
        let r = permutation_importance(&model, &x, &y, 0, 0, neg_mse).unwrap();
        assert_eq!(r.importances.shape(), &[1, 0]);
    }
}
