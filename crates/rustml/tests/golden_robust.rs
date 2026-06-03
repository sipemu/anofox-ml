//! Behavioral parity test for RANSACRegressor / TheilSenRegressor.

mod common;

use common::{json_to_array1, json_to_array2, load_golden_data};
use rustml::prelude::*;
use rustml_regression::{RansacRegressor, TheilSenRegressor};

#[test]
fn test_ransac_recovers_inlier_line() {
    let cases = load_golden_data("robust.json");
    let case = cases.iter().find(|c| c["name"] == "ransac_line").unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let sk_slope = case["sklearn_slope"].as_f64().unwrap();
    let sk_intercept = case["sklearn_intercept"].as_f64().unwrap();

    let fitted = RansacRegressor::new()
        .with_min_samples(2)
        .with_residual_threshold(0.5)
        .with_max_trials(200)
        .with_seed(0)
        .fit(&x, &y)
        .unwrap();

    // Both should recover slope ≈ 2, intercept ≈ 1.
    assert!((fitted.coef[0] - 2.0).abs() < 0.1, "rustml slope: {}", fitted.coef[0]);
    assert!((sk_slope - 2.0).abs() < 0.1);
    assert!((fitted.intercept - 1.0).abs() < 0.3);
    assert!((sk_intercept - 1.0).abs() < 0.3);
}

#[test]
fn test_theil_sen_recovers_inlier_line() {
    let cases = load_golden_data("robust.json");
    let case = cases.iter().find(|c| c["name"] == "theil_sen_line").unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let sk_slope = case["sklearn_slope"].as_f64().unwrap();

    let fitted = TheilSenRegressor::new()
        .with_max_subpopulation(2000)
        .with_seed(0)
        .fit(&x, &y)
        .unwrap();

    // TheilSen with ~17% outliers should still recover slope near 2.
    assert!(
        (fitted.coef[0] - 2.0).abs() < 0.5,
        "rustml slope: {}, sklearn slope: {}",
        fitted.coef[0],
        sk_slope
    );
}
