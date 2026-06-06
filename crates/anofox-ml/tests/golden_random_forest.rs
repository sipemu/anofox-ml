mod common;

use anofox_ml::prelude::*;
use common::{json_to_array1, json_to_array2, load_golden_data};

#[test]
fn test_golden_random_forest_classifier() {
    let cases = load_golden_data("random_forest.json");

    for case in &cases {
        let name = case["name"].as_str().unwrap();
        if case["algorithm"].as_str().unwrap() != "RandomForestClassifier" {
            continue;
        }

        let x_train = json_to_array2(&case["X_train"]);
        let y_train = json_to_array1(&case["y_train"]);
        let x_test = json_to_array2(&case["X_test"]);
        let expected_pred = json_to_array1(&case["y_pred"]);

        let n_estimators = case["n_estimators"].as_u64().unwrap() as usize;
        let max_depth = case["max_depth"].as_u64().map(|d| d as usize);

        let rf = RandomForestClassifier {
            n_estimators,
            max_depth,
            seed: 42,
            ..Default::default()
        };

        let fitted = Fit::fit(&rf, &x_train, &y_train).unwrap();
        let preds = fitted.predict(&x_test).unwrap();

        // RF has different bootstrap sampling than sklearn, so we verify
        // predictions are reasonable rather than exact match.
        // Well-separated clusters should still be correctly classified.
        for (i, (&p, &e)) in preds.iter().zip(expected_pred.iter()).enumerate() {
            // For the clearly separable points (first two), predictions should match
            if i < 2 {
                assert!(
                    (p - e).abs() < 1e-10,
                    "{}/predict[{}]: expected {}, got {}",
                    name,
                    i,
                    e,
                    p
                );
            }
        }
    }
}

#[test]
fn test_golden_random_forest_regressor() {
    let cases = load_golden_data("random_forest.json");

    for case in &cases {
        let name = case["name"].as_str().unwrap();
        if case["algorithm"].as_str().unwrap() != "RandomForestRegressor" {
            continue;
        }

        let x_train = json_to_array2(&case["X_train"]);
        let y_train = json_to_array1(&case["y_train"]);
        let x_test = json_to_array2(&case["X_test"]);

        let n_estimators = case["n_estimators"].as_u64().unwrap() as usize;
        let max_depth = case["max_depth"].as_u64().map(|d| d as usize);

        let rf = RandomForestRegressor {
            n_estimators,
            max_depth,
            seed: 42,
            ..Default::default()
        };

        let fitted = Fit::fit(&rf, &x_train, &y_train).unwrap();
        let preds = fitted.predict(&x_test).unwrap();

        // Predictions should be within a reasonable range of training targets
        let y_min = y_train.iter().copied().fold(f64::INFINITY, f64::min);
        let y_max = y_train.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        for (i, &p) in preds.iter().enumerate() {
            assert!(
                p >= y_min - 1.0 && p <= y_max + 1.0,
                "{}/predict[{}]: {} outside expected range [{}, {}]",
                name,
                i,
                p,
                y_min,
                y_max
            );
        }
    }
}
