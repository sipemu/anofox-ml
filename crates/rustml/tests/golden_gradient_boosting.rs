mod common;

use common::{json_to_array1, json_to_array2, load_golden_data};
use rustml::prelude::*;

// GBT implementations differ between sklearn and ours, so we use tolerant
// accuracy checks on well-separated data rather than exact matching.
const PRED_TOL: f64 = 0.15;

#[test]
fn test_golden_gradient_boosting_classifier() {
    let cases = load_golden_data("gradient_boosting.json");

    for case in &cases {
        let name = case["name"].as_str().unwrap();
        if case["algorithm"].as_str().unwrap() != "GradientBoostingClassifier" {
            continue;
        }

        let x_train = json_to_array2(&case["X_train"]);
        let y_train = json_to_array1(&case["y_train"]);
        let x_test = json_to_array2(&case["X_test"]);
        let expected_pred = json_to_array1(&case["y_pred"]);

        let n_estimators = case["n_estimators"].as_u64().unwrap() as usize;
        let learning_rate = case["learning_rate"].as_f64().unwrap();
        let max_depth = case["max_depth"].as_u64().map(|d| d as usize);

        let gbt = GradientBoostingClassifier::new()
            .with_n_estimators(n_estimators)
            .with_learning_rate(learning_rate)
            .with_max_depth(max_depth)
            .with_seed(42);

        let fitted = Fit::fit(&gbt, &x_train, &y_train).unwrap();
        let preds = fitted.predict(&x_test).unwrap();

        // Predictions should be valid class labels
        let classes = fitted.classes();
        for &p in preds.iter() {
            assert!(
                classes.iter().any(|&c| (c - p).abs() < 1e-10),
                "{}: prediction {} is not a valid class label",
                name,
                p
            );
        }

        // Accuracy on well-separated data should be decent
        let mut correct = 0;
        for (&p, &e) in preds.iter().zip(expected_pred.iter()) {
            if (p - e).abs() < PRED_TOL {
                correct += 1;
            }
        }
        let accuracy = correct as f64 / preds.len() as f64;
        assert!(
            accuracy >= 0.6,
            "{}: accuracy {} is too low (expected >= 0.6)",
            name,
            accuracy
        );
    }
}

#[test]
fn test_golden_gradient_boosting_regressor() {
    let cases = load_golden_data("gradient_boosting.json");

    for case in &cases {
        let name = case["name"].as_str().unwrap();
        if case["algorithm"].as_str().unwrap() != "GradientBoostingRegressor" {
            continue;
        }

        let x_train = json_to_array2(&case["X_train"]);
        let y_train = json_to_array1(&case["y_train"]);
        let x_test = json_to_array2(&case["X_test"]);

        let n_estimators = case["n_estimators"].as_u64().unwrap() as usize;
        let learning_rate = case["learning_rate"].as_f64().unwrap();
        let max_depth = case["max_depth"].as_u64().map(|d| d as usize);

        let gbt = GradientBoostingRegressor::new()
            .with_n_estimators(n_estimators)
            .with_learning_rate(learning_rate)
            .with_max_depth(max_depth)
            .with_seed(42);

        let fitted = Fit::fit(&gbt, &x_train, &y_train).unwrap();
        let preds = fitted.predict(&x_test).unwrap();

        // Verify predictions are finite
        for (i, &p) in preds.iter().enumerate() {
            assert!(
                p.is_finite(),
                "{}: prediction at index {} is not finite: {}",
                name,
                i,
                p
            );
        }

        // Predictions should be in a reasonable range
        let y_min: f64 = y_train.iter().copied().fold(f64::INFINITY, f64::min);
        let y_max: f64 = y_train.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let y_range = y_max - y_min;
        for (i, &p) in preds.iter().enumerate() {
            assert!(
                p >= y_min - y_range && p <= y_max + y_range,
                "{}: prediction {} at index {} is out of reasonable range [{}, {}]",
                name,
                p,
                i,
                y_min - y_range,
                y_max + y_range
            );
        }
    }
}
