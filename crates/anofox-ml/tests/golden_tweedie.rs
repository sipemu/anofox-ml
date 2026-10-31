//! Golden test for TweedieRegressor / GammaRegressor against sklearn 1.8.0.
//!
//! `with_alpha` uses sklearn's semantics (loss is (1/2n) * deviance +
//! (alpha/2) * ||β||²), so the sklearn alpha is passed through unchanged
//! (the golden file's `anofox_lambda = n * alpha` is the old raw-λ value).

mod common;

use anofox_ml::prelude::*;
use anofox_ml_regression::TweedieRegressor;
use common::{json_to_array1, json_to_array2, load_golden_data};

fn assert_rel_close(
    actual: &ndarray::Array1<f64>,
    expected: &ndarray::Array1<f64>,
    rel_tol: f64,
    context: &str,
) {
    assert_eq!(actual.len(), expected.len(), "{} length mismatch", context);
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let rel = (a - e).abs() / e.abs().max(1e-9);
        assert!(
            rel < rel_tol,
            "{}[{}]: anofox-ml={}, sklearn={}, rel={}",
            context,
            i,
            a,
            e,
            rel
        );
    }
}

#[test]
fn test_golden_tweedie_power_1p5() {
    let cases = load_golden_data("tweedie.json");
    let case = cases.iter().find(|c| c["name"] == "tweedie_p1p5").unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let power = case["power"].as_f64().unwrap();
    let alpha = case["sklearn_alpha"].as_f64().unwrap();
    let expected = json_to_array1(&case["predictions"]);

    let fitted = TweedieRegressor::new(power)
        .with_alpha(alpha)
        .with_max_iter(500)
        .fit(&x, &y)
        .unwrap();
    let preds = fitted.predict(&x).unwrap();
    assert_rel_close(&preds, &expected, 1e-3, "tweedie 1.5 predictions");
}

#[test]
fn test_golden_gamma() {
    let cases = load_golden_data("tweedie.json");
    let case = cases.iter().find(|c| c["name"] == "gamma").unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let alpha = case["sklearn_alpha"].as_f64().unwrap();
    let expected = json_to_array1(&case["predictions"]);

    let fitted = anofox_ml_regression::GammaRegressor::new()
        .with_alpha(alpha)
        .fit(&x, &y)
        .unwrap();
    let preds = fitted.predict(&x).unwrap();
    assert_rel_close(&preds, &expected, 1e-3, "gamma predictions");
}
