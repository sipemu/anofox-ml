//! LgbmRegressor / LgbmClassifier against LightGBM 4.7.0 (#127).
//! Generator: `test_harness/generators/gen_lgbm_parity.py`.

mod common;

use anofox_ml::prelude::*;
use anofox_ml_ensemble::{DartParams, LgbmClassWeight, LgbmFitOptions, LgbmObjective};
use common::{json_to_array1, load_golden_data};
use ndarray::{Array1, Array2};
use serde_json::Value;

/// The engine reproduces LightGBM's binning, count estimates, split scans
/// and float32 labels / gradients; most cases agree to rounding.
const TOL: f64 = 1e-8;

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

/// Largest error relative to `max(|expected|, 1)`.
fn worst(pred: impl IntoIterator<Item = f64>, expected: impl IntoIterator<Item = f64>) -> f64 {
    pred.into_iter()
        .zip(expected)
        .map(|(a, e)| (a - e).abs() / e.abs().max(1.0))
        .fold(0.0, f64::max)
}

/// Apply the shared LightGBM parameters to either model.
macro_rules! configure {
    ($m:expr, $p:expr) => {{
        let p = $p;
        let mut m = $m
            .with_n_estimators(p["n_estimators"].as_u64().unwrap() as usize)
            .with_learning_rate(p["learning_rate"].as_f64().unwrap())
            .with_num_leaves(p["num_leaves"].as_u64().unwrap() as usize);
        if let Some(v) = p["min_child_samples"].as_u64() {
            m = m.with_min_child_samples(v as usize);
        }
        if let Some(v) = p["reg_lambda"].as_f64() {
            m = m.with_reg_lambda(v);
        }
        if let Some(v) = p["reg_alpha"].as_f64() {
            m = m.with_reg_alpha(v);
        }
        if p["boosting_type"] == "dart" {
            let d = DartParams::default();
            m = m.with_dart(DartParams {
                drop_rate: p["drop_rate"].as_f64().unwrap_or(d.drop_rate),
                max_drop: p["max_drop"].as_i64().unwrap_or(d.max_drop),
                skip_drop: p["skip_drop"].as_f64().unwrap_or(d.skip_drop),
                uniform_drop: p["uniform_drop"].as_bool().unwrap_or(d.uniform_drop),
                xgboost_dart_mode: p["xgboost_dart_mode"]
                    .as_bool()
                    .unwrap_or(d.xgboost_dart_mode),
                drop_seed: p["drop_seed"].as_u64().map_or(d.drop_seed, |v| v as u32),
            });
        }
        if let Some(b) = p["enable_bundle"].as_bool() {
            m = m.with_enable_bundle(b);
        }
        if let Some(c) = p["monotone_constraints"].as_array() {
            m = m.with_monotone_constraints(c.iter().map(|v| v.as_i64().unwrap() as i8).collect());
        }
        m
    }};
}

fn sample_weight(case: &Value) -> Option<Array1<f64>> {
    (!case["sample_weight"].is_null()).then(|| json_to_array1(&case["sample_weight"]))
}

#[test]
fn lgbm_matches_lightgbm() {
    let mut failures = Vec::new();
    let mut report = Vec::new();
    for case in load_golden_data("lgbm_parity.json") {
        let name = case["name"].as_str().unwrap();
        let p = &case["params"];
        let x = array2_with_nan(&case["X"]);
        let y = json_to_array1(&case["y"]);
        let sw = sample_weight(&case);
        let x_test = array2_with_nan(&case["X_test"]);
        let categorical = if name.contains("categorical") {
            vec![2]
        } else {
            vec![]
        };
        let opts = LgbmFitOptions {
            sample_weight: sw.as_ref().map(|w| w.view()),
            ..LgbmFitOptions::default()
        };
        let err = if case["kind"] == "regressor" {
            let objective = match p["objective"].as_str().unwrap_or("regression") {
                "regression" => LgbmObjective::Regression,
                "regression_l1" => LgbmObjective::RegressionL1,
                "huber" => LgbmObjective::Huber,
                "quantile" => LgbmObjective::Quantile {
                    alpha: p["alpha"].as_f64().unwrap(),
                },
                "poisson" => LgbmObjective::Poisson,
                "gamma" => LgbmObjective::Gamma,
                other => panic!("unknown objective {other}"),
            };
            let mut m = configure!(LgbmRegressor::new(), p)
                .with_objective(objective)
                .with_categorical_features(categorical);
            if let Some(g) = p["min_split_gain"].as_f64() {
                m = m.with_min_split_gain(g);
            }
            let f = m.fit_with_eval(&x, &y, &opts).unwrap();
            worst(
                f.predict(&x_test).unwrap(),
                json_to_array1(&case["predictions"]),
            )
        } else {
            let mut m = configure!(LgbmClassifier::new(), p).with_categorical_features(categorical);
            if p["class_weight"] == "balanced" {
                m = m.with_class_weight(Some(LgbmClassWeight::Balanced));
            }
            let f = m.fit_with_eval(&x, &y, &opts).unwrap();
            worst(
                f.predict_proba(&x_test).unwrap(),
                array2_with_nan(&case["predictions"]),
            )
        };
        report.push(format!("{name}: {err:e}"));
        if err > TOL {
            failures.push(format!("{name}: max error {err:e}"));
        }
    }
    eprintln!("{report:#?}");
    assert!(failures.is_empty(), "{failures:#?}");
}
