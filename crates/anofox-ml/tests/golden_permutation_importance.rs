//! Golden test for permutation_importance against sklearn 1.8.0.
//!
//! Permutation importance is stochastic; sklearn's RNG order differs from ours.
//! We assert (a) the rank order of mean importances matches sklearn exactly,
//! and (b) the top-feature mean importance is within 10% of sklearn's.

mod common;

use anofox_ml::core::{permutation_importance, Predict, Result};
use anofox_ml::prelude::*;
use anofox_ml_metrics::r2_score;
use common::{json_to_array1, json_to_array2, load_golden_data};
use ndarray::Array1;

fn argsort_desc(v: &Array1<f64>) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap());
    idx
}

#[test]
fn test_golden_permutation_importance_ridge_5feat() {
    let cases = load_golden_data("permutation_importance.json");
    let case = &cases[0];

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let alpha = case["alpha"].as_f64().unwrap();
    let sklearn_mean = json_to_array1(&case["sklearn_importances_mean"]);
    let sklearn_rank: Vec<usize> = case["sklearn_rank_desc"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();

    // Fit a comparable Ridge.
    let fitted = RidgeRegressor::new()
        .with_lambda(alpha)
        .fit(&x, &y)
        .unwrap();

    // Sanity: R² should match sklearn's (Ridge is closed-form).
    let preds = fitted.predict(&x).unwrap();
    let r2 = r2_score(&y, &preds).unwrap();
    let baseline_r2 = case["baseline_r2"].as_f64().unwrap();
    assert!(
        (r2 - baseline_r2).abs() < 1e-6,
        "R² mismatch: anofox-ml={}, sklearn={}",
        r2,
        baseline_r2
    );

    // Wrap as a Predict-only model to feed permutation_importance.
    struct Wrap(anofox_ml::regression::FittedRidgeRegressor);
    impl Predict<f64> for Wrap {
        fn predict(&self, x: &ndarray::Array2<f64>) -> Result<Array1<f64>> {
            self.0.predict(x)
        }
    }
    let wrap = Wrap(fitted);

    let r2_fn =
        |y_true: &Array1<f64>, y_pred: &Array1<f64>| r2_score(y_true, y_pred).unwrap_or(0.0);

    let imp = permutation_importance(&wrap, &x, &y, 50, 0, r2_fn).unwrap();

    let rank = argsort_desc(&imp.importances_mean);

    assert_eq!(
        rank, sklearn_rank,
        "rank order disagrees: anofox-ml={:?}, sklearn={:?}, imps={:?}",
        rank, sklearn_rank, imp.importances_mean
    );

    // The top feature's importance should be within ~10% of sklearn's.
    let top = sklearn_rank[0];
    let ours = imp.importances_mean[top];
    let theirs = sklearn_mean[top];
    let rel = (ours - theirs).abs() / theirs.abs().max(1e-9);
    assert!(
        rel < 0.10,
        "top-feature importance differs by {:.1}%: anofox-ml={}, sklearn={}",
        rel * 100.0,
        ours,
        theirs
    );
}
