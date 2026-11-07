//! CatBoost-style estimators against catboost 1.2.10 (#127), with its
//! randomness switched off. Generator: `test_harness/generators/gen_catboost_parity.py`.

mod common;

use anofox_ml::prelude::*;
use anofox_ml_ensemble::catboost::catboost_borders;
use anofox_ml_ensemble::{
    CatBoostBacktracking, CatBoostBoostingType, CatBoostBootstrap, CatBoostClassifier,
    CatBoostLeafEstimation, CatBoostRegressor, CatBoostScoreFunction,
};
use common::{json_to_array1, load_golden_file};
use ndarray::{Array1, Array2};
use serde_json::Value;

/// The models agree with CatBoost to rounding.
const TOL: f64 = 1e-12;

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

fn worst(pred: impl IntoIterator<Item = f64>, expected: impl IntoIterator<Item = f64>) -> f64 {
    pred.into_iter()
        .zip(expected)
        .map(|(a, e)| (a - e).abs() / e.abs().max(1.0))
        .fold(0.0, f64::max)
}

#[test]
fn quantization_matches_catboost() {
    let data = load_golden_file("catboost_parity.json");
    for case in data["quantization"].as_array().unwrap() {
        let x = array2_with_nan(&case["X"]);
        let count = case["border_count"].as_u64().unwrap() as usize;
        let ours = catboost_borders(x.view(), count);
        for (f, expected) in case["borders"].as_array().unwrap().iter().enumerate() {
            let expected: Vec<f32> = expected
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            assert_eq!(ours[f], expected, "{} feature {f}", case["name"]);
        }
    }
}

macro_rules! configure {
    ($m:expr, $p:expr) => {{
        let p = $p;
        let mut m = $m
            .with_iterations(p["iterations"].as_u64().unwrap() as usize)
            .with_depth(p["depth"].as_u64().unwrap_or(6) as usize);
        if let Some(lr) = p["learning_rate"].as_f64() {
            m = m.with_learning_rate(lr);
        }
        if let Some(l2) = p["l2_leaf_reg"].as_f64() {
            m = m.with_l2_leaf_reg(l2);
        }
        if let Some(b) = p["border_count"].as_u64() {
            m = m.with_border_count(b as usize);
        }
        if p["score_function"] == "L2" {
            m = m.with_score_function(CatBoostScoreFunction::L2);
        }
        if p["leaf_estimation_method"] == "Gradient" {
            m = m.with_leaf_estimation_method(CatBoostLeafEstimation::Gradient);
        }
        if let Some(b) = p["boost_from_average"].as_bool() {
            m = m.with_boost_from_average(b);
        }
        if let Some(c) = p["cat_features"].as_array() {
            m = m.with_cat_features(c.iter().map(|v| v.as_u64().unwrap() as usize).collect());
        }
        if p["boosting_type"] == "Ordered" {
            m = m.with_boosting_type(CatBoostBoostingType::Ordered);
        }
        if let Some(f) = p["fold_len_multiplier"].as_f64() {
            m = m.with_fold_len_multiplier(f);
        }
        // The generator's common settings (not stored) include `has_time`.
        m = m.with_has_time(p["has_time"].as_bool().unwrap_or(true));
        m = m.with_random_strength(p["random_strength"].as_f64().unwrap_or(0.0));
        m = m.with_bootstrap_type(match p["bootstrap_type"].as_str().unwrap_or("No") {
            "Bayesian" => CatBoostBootstrap::Bayesian,
            "Bernoulli" => CatBoostBootstrap::Bernoulli,
            "MVS" => CatBoostBootstrap::Mvs,
            _ => CatBoostBootstrap::No,
        });
        if let Some(v) = p["subsample"].as_f64() {
            m = m.with_subsample(v);
        }
        if let Some(v) = p["bagging_temperature"].as_f64() {
            m = m.with_bagging_temperature(v);
        }
        if let Some(v) = p["mvs_reg"].as_f64() {
            m = m.with_mvs_reg(v);
        }
        if let Some(n) = p["random_seed"].as_u64() {
            m = m.with_random_seed(n);
        }
        if let Some(n) = p["permutation_count"].as_u64() {
            m = m.with_permutation_count(n as usize);
        }
        if let Some(n) = p["leaf_estimation_iterations"].as_u64() {
            m = m.with_leaf_estimation_iterations(n as usize);
        }
        if p["leaf_estimation_backtracking"] == "No" {
            m = m.with_leaf_estimation_backtracking(CatBoostBacktracking::No);
        }
        if let Some(n) = p["max_ctr_complexity"].as_u64() {
            m = m.with_max_ctr_complexity(n as usize);
        }
        if let Some(n) = p["one_hot_max_size"].as_u64() {
            m = m.with_one_hot_max_size(n as usize);
        }
        m
    }};
}

#[test]
fn catboost_models_match_catboost() {
    let data = load_golden_file("catboost_parity.json");
    let mut report = Vec::new();
    let mut failures = Vec::new();
    for case in data["models"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let x = array2_with_nan(&case["X"]);
        let y = json_to_array1(&case["y"]);
        let sw: Option<Array1<f64>> =
            (!case["sample_weight"].is_null()).then(|| json_to_array1(&case["sample_weight"]));
        let x_test = array2_with_nan(&case["X_test"]);
        let err = if case["kind"] == "regressor" {
            let f = configure!(CatBoostRegressor::new(), &case["params"])
                .fit_weighted(&x, &y, sw.as_ref())
                .unwrap();
            worst(
                f.predict(&x_test).unwrap(),
                json_to_array1(&case["predictions"]),
            )
        } else {
            let f = configure!(CatBoostClassifier::new(), &case["params"])
                .fit_weighted(&x, &y, sw.as_ref())
                .unwrap();
            worst(
                f.predict_proba(&x_test).unwrap(),
                array2_with_nan(&case["predictions"]),
            )
        };
        report.push(format!("{name}: {err:e}"));
        if err > TOL {
            failures.push(format!("{name}: {err:e}"));
        }
    }
    eprintln!("{report:#?}");
    assert!(failures.is_empty(), "{failures:#?}");
}
