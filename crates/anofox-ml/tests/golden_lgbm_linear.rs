//! LgbmRegressor with linear trees against LightGBM 4.7.0 (#208).
//! Generator: `test_harness/generators/gen_lgbm_linear.py`.

mod common;

use anofox_ml::prelude::*;
use anofox_ml_ensemble::{LgbmFitOptions, LgbmObjective};
use common::{json_to_array1, load_golden_data};
use ndarray::Array2;
use serde_json::Value;

/// LightGBM inverts each leaf's normal equations with Eigen's full-pivot
/// LU; the predictions agree to rounding.
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

#[test]
fn linear_trees_match_lightgbm() {
    let mut failures = Vec::new();
    for case in load_golden_data("lgbm_linear.json") {
        let name = case["name"].as_str().unwrap();
        let p = &case["params"];
        let mut m = LgbmRegressor::new()
            .with_n_estimators(p["n_estimators"].as_u64().unwrap() as usize)
            .with_learning_rate(p["learning_rate"].as_f64().unwrap())
            .with_num_leaves(p["num_leaves"].as_u64().unwrap() as usize)
            .with_min_child_samples(p["min_child_samples"].as_u64().unwrap() as usize)
            .with_linear_tree(true, p["linear_lambda"].as_f64().unwrap_or(0.0))
            .with_categorical_features(
                case["categorical"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as usize)
                    .collect(),
            );
        if p["objective"] == "poisson" {
            m = m.with_objective(LgbmObjective::Poisson);
        }
        if let Some(seed) = p["random_state"].as_u64() {
            m = m
                .with_seed(seed)
                .with_subsample(p["subsample"].as_f64().unwrap())
                .with_subsample_freq(p["subsample_freq"].as_u64().unwrap() as usize);
        }
        let sw = (!case["sample_weight"].is_null()).then(|| json_to_array1(&case["sample_weight"]));
        let opts = LgbmFitOptions {
            sample_weight: sw.as_ref().map(|w| w.view()),
            ..Default::default()
        };
        let x = array2_with_nan(&case["X"]);
        let y = json_to_array1(&case["y"]);
        let fitted = m.fit_with_eval(&x, &y, &opts).unwrap();
        let pred = fitted.predict(&array2_with_nan(&case["X_test"])).unwrap();
        let expected = json_to_array1(&case["predictions"]);
        let err = pred
            .iter()
            .zip(&expected)
            .map(|(a, e)| (a - e).abs() / e.abs().max(1.0))
            .fold(0.0, f64::max);
        eprintln!("{name}: {err:e}");
        if err > TOL {
            failures.push(format!("{name}: max error {err:e}"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
