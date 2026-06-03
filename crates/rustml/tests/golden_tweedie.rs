//! Golden test for TweedieRegressor / GammaRegressor against sklearn 1.8.0.
//!
//! sklearn's `alpha` is per-sample-normalized (loss is (1/2n) * deviance +
//! (alpha/2) * ||β||²) while anofox-regression's `lambda` is on the un-
//! normalized loss. We scale anofox's lambda by `n` to match sklearn.
//! IRLS convergence criteria still differ slightly, so we use a 1% relative
//! tolerance on predictions.

mod common;

use common::{json_to_array1, json_to_array2, load_golden_data};
use rustml::prelude::*;
use rustml_regression::TweedieRegressor;

fn assert_rel_close(actual: &ndarray::Array1<f64>, expected: &ndarray::Array1<f64>, rel_tol: f64, context: &str) {
    assert_eq!(actual.len(), expected.len(), "{} length mismatch", context);
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let rel = (a - e).abs() / e.abs().max(1e-9);
        assert!(rel < rel_tol, "{}[{}]: rustml={}, sklearn={}, rel={}", context, i, a, e, rel);
    }
}

#[test]
fn test_golden_tweedie_power_1p5() {
    let cases = load_golden_data("tweedie.json");
    let case = cases.iter().find(|c| c["name"] == "tweedie_p1p5").unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let power = case["power"].as_f64().unwrap();
    let anofox_lambda = case["anofox_lambda"].as_f64().unwrap();
    let expected = json_to_array1(&case["predictions"]);

    let fitted = TweedieRegressor::new(power)
        .with_alpha(anofox_lambda)
        .with_max_iter(500)
        .fit(&x, &y)
        .unwrap();
    let preds = fitted.predict(&x).unwrap();
    assert_rel_close(&preds, &expected, 0.01, "tweedie 1.5 predictions");
}

#[test]
fn test_golden_gamma() {
    let cases = load_golden_data("tweedie.json");
    let case = cases.iter().find(|c| c["name"] == "gamma").unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let anofox_lambda = case["anofox_lambda"].as_f64().unwrap();
    let expected = json_to_array1(&case["predictions"]);

    let fitted = rustml_regression::GammaRegressor::new()
        .with_alpha(anofox_lambda)
        .fit(&x, &y)
        .unwrap();
    let preds = fitted.predict(&x).unwrap();
    assert_rel_close(&preds, &expected, 0.01, "gamma predictions");
}
