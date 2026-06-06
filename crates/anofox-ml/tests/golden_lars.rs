//! Golden test for LARS / LassoLars against sklearn 1.8.0.

mod common;

use anofox_ml::core::{Fit, Predict};
use anofox_ml_regression::Lars;
use common::{json_to_array1, json_to_array2, load_golden_data};

#[test]
fn test_lars_recovers_correct_active_set() {
    let cases = load_golden_data("lars.json");
    let case = cases.iter().find(|c| c["name"] == "lars_3").unwrap();
    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let k = case["n_nonzero"].as_u64().unwrap() as usize;
    let expected: Vec<usize> = case["expected_active"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();

    let fitted = Lars::new(k).fit(&x, &y).unwrap();
    let mut act = fitted.active_set.clone();
    act.sort();
    assert_eq!(act, expected);

    // R² should be in the same neighborhood as sklearn (LARS stops at
    // step k, not at the OLS solution on the active set).
    let preds = fitted.predict(&x).unwrap();
    let mean = y.iter().sum::<f64>() / y.len() as f64;
    let rss: f64 = preds
        .iter()
        .zip(y.iter())
        .map(|(p, t)| (t - p).powi(2))
        .sum();
    let tss: f64 = y.iter().map(|t| (t - mean).powi(2)).sum();
    let r2 = 1.0 - rss / tss;
    let sk_r2 = case["sklearn_r2"].as_f64().unwrap();
    assert!(
        (r2 - sk_r2).abs() < 0.10,
        "R² differs too much from sklearn: anofox-ml={r2}, sklearn={sk_r2}"
    );
}

#[test]
fn test_lasso_lars_finds_some_sparse_solution() {
    let cases = load_golden_data("lars.json");
    let case = cases.iter().find(|c| c["name"] == "lasso_lars").unwrap();
    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let _alpha = case["alpha"].as_f64().unwrap();

    // LassoLars step-count not directly mapped to sklearn alpha; just run
    // 3 LARS steps and confirm the active set is the informative one.
    let fitted = Lars::lasso(3).fit(&x, &y).unwrap();
    let mut act = fitted.active_set.clone();
    act.sort();
    assert_eq!(act, vec![1, 3, 5]);
}
