//! HistGradientBoosting losses, categorical features, constraints and early
//! stopping against sklearn 1.8.0 (#127).
//! Generator: `test_harness/generators/gen_hist_gb_parity.py`. The facade
//! reproduces sklearn's float32 gradients, split order and gain arithmetic,
//! so predictions agree to rounding.

mod common;

use anofox_ml::prelude::*;
use anofox_ml_ensemble::{
    EarlyStopping, HistGradientBoostingLoss, InteractionConstraints, TestSize,
};
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

/// Apply sklearn's early-stopping parameters (absent = sklearn's defaults).
macro_rules! early_stopping {
    ($m:expr, $params:expr) => {{
        let p = $params;
        let mut m = $m.with_early_stopping(match &p["early_stopping"] {
            Value::Bool(true) => EarlyStopping::Enabled,
            Value::Bool(false) => EarlyStopping::Disabled,
            _ => EarlyStopping::Auto,
        });
        match &p["validation_fraction"] {
            Value::Null if p.get("validation_fraction").is_some() => {
                m = m.with_validation_fraction(None)
            }
            Value::Number(v) if v.is_u64() => {
                m = m.with_validation_fraction(Some(TestSize::Count(v.as_u64().unwrap() as usize)))
            }
            Value::Number(v) => {
                m = m.with_validation_fraction(Some(TestSize::Fraction(v.as_f64().unwrap())))
            }
            _ => {}
        }
        if let Some(n) = p["n_iter_no_change"].as_u64() {
            m = m.with_n_iter_no_change(n as usize);
        }
        if let Some(t) = p["tol"].as_f64() {
            m = m.with_tol(t);
        }
        if let Some(r) = p["random_state"].as_u64() {
            m = m.with_random_state(r as u32);
        }
        m
    }};
}

fn model(params: &Value) -> HistGradientBoostingRegressor {
    let loss = match params["loss"].as_str().unwrap_or("squared_error") {
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
        .with_l2_regularization(params["l2_regularization"].as_f64().unwrap_or(0.0));
    if let Some(q) = params["quantile"].as_f64() {
        m = m.with_quantile(q);
    }
    let (mono, inter) = constraints(params);
    m = m
        .with_categorical_features(categorical(params))
        .with_monotonic_cst(mono);
    m = early_stopping!(m, params);
    match inter {
        Some(c) => m.with_interaction_cst(c),
        None => m,
    }
}

/// Compare `n_iter_` and the score curves of an early-stopping case.
fn check_scores(case: &Value, n_iter: usize, train: &[f64], validation: &[f64]) -> Option<String> {
    let expected = case["n_iter"].as_u64()?;
    let curve = |key: &str| json_to_array1(&case[key]).to_vec();
    let (et, ev) = (curve("train_score"), curve("validation_score"));
    if n_iter != expected as usize || et.len() != train.len() || ev.len() != validation.len() {
        return Some(format!(
            "n_iter {n_iter} vs {expected}, curves {}/{} vs {}/{}",
            train.len(),
            validation.len(),
            et.len(),
            ev.len()
        ));
    }
    let w = worst(train.iter().copied(), et).max(worst(validation.iter().copied(), ev));
    (w > 1e-12).then(|| format!("scores differ by {w:e}"))
}

fn constraints(params: &Value) -> (Vec<i8>, Option<InteractionConstraints>) {
    let mono = params["monotonic_cst"]
        .as_array()
        .map(|a| a.iter().map(|v| v.as_i64().unwrap() as i8).collect())
        .unwrap_or_default();
    let inter = match &params["interaction_cst"] {
        Value::Null => None,
        Value::String(s) if s == "pairwise" => Some(InteractionConstraints::Pairwise),
        Value::String(s) if s == "no_interactions" => Some(InteractionConstraints::NoInteractions),
        Value::Array(groups) => Some(InteractionConstraints::Groups(
            groups
                .iter()
                .map(|g| {
                    g.as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_u64().unwrap() as usize)
                        .collect()
                })
                .collect(),
        )),
        other => panic!("unknown interaction_cst {other}"),
    };
    (mono, inter)
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
        if let Some(msg) = check_scores(
            &case,
            fitted.n_iter(),
            fitted.train_score(),
            fitted.validation_score(),
        ) {
            failures.push(format!("{name}: {msg}"));
        }
        let pred = fitted.predict(&array2_with_nan(&case["X_test"])).unwrap();
        let worst = worst(pred, expected);
        if worst > 1e-12 {
            failures.push(format!("{name}: max rel. error {worst:e}"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn classifier_matches_sklearn() {
    let mut failures = Vec::new();
    for case in load_golden_data("hist_gb_parity.json") {
        if case["kind"] != "classifier" {
            continue;
        }
        let name = case["name"].as_str().unwrap();
        let params = &case["params"];
        let (mono, inter) = constraints(params);
        let mut model = HistGradientBoostingClassifier::new()
            .with_n_estimators(params["max_iter"].as_u64().unwrap() as usize)
            .with_learning_rate(params["learning_rate"].as_f64().unwrap())
            .with_max_leaf_nodes(Some(params["max_leaf_nodes"].as_u64().unwrap() as usize))
            .with_min_samples_leaf(params["min_samples_leaf"].as_u64().unwrap() as usize)
            .with_l2_regularization(params["l2_regularization"].as_f64().unwrap_or(0.0))
            .with_categorical_features(categorical(params))
            .with_monotonic_cst(mono);
        if let Some(c) = inter {
            model = model.with_interaction_cst(c);
        }
        model = early_stopping!(model, params);
        let sw: Option<Array1<f64>> =
            (!case["sample_weight"].is_null()).then(|| json_to_array1(&case["sample_weight"]));
        let fitted = model
            .fit_weighted(
                &array2_with_nan(&case["X"]),
                &json_to_array1(&case["y"]),
                sw.as_ref(),
            )
            .unwrap();
        if let Some(msg) = check_scores(
            &case,
            fitted.n_iter(),
            fitted.train_score(),
            fitted.validation_score(),
        ) {
            failures.push(format!("{name}: {msg}"));
        }
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
