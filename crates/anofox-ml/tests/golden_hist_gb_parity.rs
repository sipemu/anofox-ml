//! HistGradientBoosting losses and categorical features against sklearn
//! 1.8.0 (#127).
//! Generator: `test_harness/generators/gen_hist_gb_parity.py`. The facade
//! reproduces sklearn's float32 gradients, split order and gain arithmetic,
//! so predictions agree to rounding.

mod common;

use anofox_ml::prelude::*;
use anofox_ml_ensemble::HistGradientBoostingLoss;
use common::{json_to_array1, load_golden_data};
use ndarray::{Array1, Array2};
use serde_json::Value;

/// 2-D array with JSON `null` read as `NaN`.
fn array2_with_nan(val: &Value) -> Array2<f64> {
    let rows = val.as_array().unwrap();
    let p = rows[0].as_array().unwrap().len();
    let flat: Vec<f64> = rows
        .iter()
        .flat_map(|r| r.as_array().unwrap().iter())
        .map(|v| v.as_f64().unwrap_or(f64::NAN))
        .collect();
    Array2::from_shape_vec((rows.len(), p), flat).unwrap()
}

fn model(params: &Value) -> HistGradientBoostingRegressor {
    let loss = match params["loss"].as_str().unwrap() {
        "squared_error" => HistGradientBoostingLoss::SquaredError,
        "absolute_error" => HistGradientBoostingLoss::AbsoluteError,
        "quantile" => HistGradientBoostingLoss::Quantile,
        "poisson" => HistGradientBoostingLoss::Poisson,
        "gamma" => HistGradientBoostingLoss::Gamma,
        other => panic!("unknown loss {other}"),
    };
    let mut m = HistGradientBoostingRegressor::new()
        .with_loss(loss)
        .with_n_estimators(params["max_iter"].as_u64().unwrap() as usize)
        .with_learning_rate(params["learning_rate"].as_f64().unwrap())
        .with_max_leaf_nodes(Some(params["max_leaf_nodes"].as_u64().unwrap() as usize))
        .with_min_samples_leaf(params["min_samples_leaf"].as_u64().unwrap() as usize)
        .with_l2_regularization(params["l2_regularization"].as_f64().unwrap());
    if let Some(q) = params["quantile"].as_f64() {
        m = m.with_quantile(q);
    }
    m.with_categorical_features(categorical(params))
}

fn categorical(params: &Value) -> Vec<usize> {
    params["categorical_features"]
        .as_array()
        .map(|a| a.iter().map(|v| v.as_u64().unwrap() as usize).collect())
        .unwrap_or_default()
}

/// Largest error relative to `max(|expected|, 1)`.
fn worst(pred: impl IntoIterator<Item = f64>, expected: impl IntoIterator<Item = f64>) -> f64 {
    pred.into_iter()
        .zip(expected)
        .map(|(a, e)| (a - e).abs() / e.abs().max(1.0))
        .fold(0.0, f64::max)
}

#[test]
fn regressor_losses_match_sklearn() {
    let mut failures = Vec::new();
    for case in load_golden_data("hist_gb_parity.json") {
        if case["kind"] == "classifier" {
            continue;
        }
        let name = case["name"].as_str().unwrap();
        let x = array2_with_nan(&case["X"]);
        let y = json_to_array1(&case["y"]);
        let sw: Option<Array1<f64>> =
            (!case["sample_weight"].is_null()).then(|| json_to_array1(&case["sample_weight"]));
        let expected = json_to_array1(&case["predictions"]);
        let fitted = model(&case["params"])
            .fit_weighted(&x, &y, sw.as_ref())
            .unwrap();
        let pred = fitted.predict(&array2_with_nan(&case["X_test"])).unwrap();
        let worst = worst(pred, expected);
        if worst > 1e-12 {
            failures.push(format!("{name}: max rel. error {worst:e}"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn classifier_categorical_matches_sklearn() {
    let mut failures = Vec::new();
    for case in load_golden_data("hist_gb_parity.json") {
        if case["kind"] != "classifier" {
            continue;
        }
        let name = case["name"].as_str().unwrap();
        let params = &case["params"];
        let fitted = HistGradientBoostingClassifier::new()
            .with_n_estimators(params["max_iter"].as_u64().unwrap() as usize)
            .with_learning_rate(params["learning_rate"].as_f64().unwrap())
            .with_max_leaf_nodes(Some(params["max_leaf_nodes"].as_u64().unwrap() as usize))
            .with_min_samples_leaf(params["min_samples_leaf"].as_u64().unwrap() as usize)
            .with_l2_regularization(params["l2_regularization"].as_f64().unwrap())
            .with_categorical_features(categorical(params))
            .fit(&array2_with_nan(&case["X"]), &json_to_array1(&case["y"]))
            .unwrap();
        let proba = fitted
            .predict_proba(&array2_with_nan(&case["X_test"]))
            .unwrap();
        let expected = array2_with_nan(&case["predictions"]);
        let worst = worst(proba.iter().copied(), expected.iter().copied());
        if worst > 1e-12 {
            failures.push(format!("{name}: max error {worst:e}"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
