//! Successive halving for hyperparameter search.
//!
//! Mirrors `sklearn.model_selection.HalvingGridSearchCV` and
//! `HalvingRandomSearchCV`. At each round, evaluate all surviving candidates
//! on a small "resource" subset of the data; keep the top `1 / factor`,
//! multiply resources by `factor`, repeat until 1 candidate remains.

use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::error::{Result, RustMlError};
use crate::float::Float;

/// Result of a halving search.
#[derive(Debug, Clone)]
pub struct HalvingResult<F: Float> {
    pub best_params_index: usize,
    pub best_score: F,
    /// Score per candidate at the final round (NaN for those eliminated earlier).
    pub final_scores: Vec<F>,
    pub rounds: usize,
}

/// `factor` is the rate at which both candidates are eliminated and resources
/// grown. `min_resources` is the starting subset size (n_samples).
/// `param_configs[i]` is a closure that fits + predicts for candidate `i`.
pub fn halving_grid_search_cv<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    factor: usize,
    min_resources: usize,
    seed: u64,
    param_configs: &[impl Fn(&Array2<F>, &Array1<F>, &Array2<F>) -> Result<Array1<F>>],
    scorer: impl Fn(&Array1<F>, &Array1<F>) -> Result<F>,
) -> Result<HalvingResult<F>> {
    if param_configs.is_empty() {
        return Err(RustMlError::InvalidParameter("no candidates".into()));
    }
    if factor < 2 {
        return Err(RustMlError::InvalidParameter("factor must be ≥ 2".into()));
    }
    let n = x.nrows();
    if min_resources < 2 || min_resources > n {
        return Err(RustMlError::InvalidParameter(format!(
            "min_resources must be in 2..={}", n
        )));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut shuffle_idx: Vec<usize> = (0..n).collect();
    shuffle_idx.shuffle(&mut rng);

    let mut surviving: Vec<usize> = (0..param_configs.len()).collect();
    let mut resources = min_resources;
    let mut last_scores: Vec<F> = vec![F::neg_infinity(); param_configs.len()];
    let mut rounds = 0usize;

    while surviving.len() > 1 {
        rounds += 1;
        let used = resources.min(n);
        let subset = &shuffle_idx[..used];

        // 80/20 split inside the subset for scoring.
        let cut = (used * 4 / 5).max(2).min(used - 1);
        let train_idx = &subset[..cut];
        let test_idx = &subset[cut..];

        let x_train = select_rows(x, train_idx);
        let y_train = select_elements(y, train_idx);
        let x_test = select_rows(x, test_idx);
        let y_test = select_elements(y, test_idx);

        for &cand in &surviving {
            let pred = param_configs[cand](&x_train, &y_train, &x_test)?;
            let s = scorer(&y_test, &pred)?;
            last_scores[cand] = s;
        }

        // Keep top ceil(len / factor) candidates.
        let keep = ((surviving.len() + factor - 1) / factor).max(1);
        let mut ranked: Vec<usize> = surviving.clone();
        ranked.sort_by(|&a, &b| last_scores[b].partial_cmp(&last_scores[a]).unwrap());
        surviving = ranked[..keep].to_vec();

        if resources >= n {
            break;
        }
        resources = (resources * factor).min(n);
    }

    let (best_params_index, &best_score) = surviving
        .iter()
        .map(|&i| (i, &last_scores[i]))
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    Ok(HalvingResult {
        best_params_index,
        best_score,
        final_scores: last_scores,
        rounds,
    })
}

fn select_rows<F: Float>(x: &Array2<F>, indices: &[usize]) -> Array2<F> {
    let ncols = x.ncols();
    let mut data = Vec::with_capacity(indices.len() * ncols);
    for &i in indices {
        for j in 0..ncols {
            data.push(x[[i, j]]);
        }
    }
    Array2::from_shape_vec((indices.len(), ncols), data).unwrap()
}

fn select_elements<F: Float>(y: &Array1<F>, indices: &[usize]) -> Array1<F> {
    Array1::from_vec(indices.iter().map(|&i| y[i]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_halving_picks_better_candidate() {
        let x = Array2::from_shape_vec((40, 1), (0..40).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_vec((0..40).map(|i| 2.0 * i as f64 + 3.0).collect());

        // Two candidates: one predicts true line, the other predicts zeros.
        let good: Box<dyn Fn(&Array2<f64>, &Array1<f64>, &Array2<f64>) -> Result<Array1<f64>>> =
            Box::new(|_xt, _yt, xv| {
                Ok(xv.column(0).mapv(|v| 2.0 * v + 3.0))
            });
        let bad: Box<dyn Fn(&Array2<f64>, &Array1<f64>, &Array2<f64>) -> Result<Array1<f64>>> =
            Box::new(|_xt, _yt, xv| Ok(Array1::<f64>::zeros(xv.nrows())));

        let candidates = vec![good, bad];
        let result = halving_grid_search_cv(
            &x, &y, 2, 8, 0, &candidates,
            |yt, yp| {
                let mse: f64 = yt.iter().zip(yp.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>()
                    / yt.len() as f64;
                Ok(-mse)
            },
        ).unwrap();
        assert_eq!(result.best_params_index, 0);
        let _ = array![1.0_f64];
    }
}
