mod common;

use common::{
    assert_array1_close, assert_array2_close, assert_close, json_to_array1, load_golden_data,
};
use rustml::prelude::*;

const TOL: f64 = 1e-10;

#[test]
fn test_golden_regression_metrics() {
    let cases = load_golden_data("metrics.json");

    for case in &cases {
        let name = case["name"].as_str().unwrap();
        if !name.starts_with("regression") {
            continue;
        }

        let y_true = json_to_array1(&case["y_true"]);
        let y_pred = json_to_array1(&case["y_pred"]);

        let expected_mse = case["mse"].as_f64().unwrap();
        let expected_mae = case["mae"].as_f64().unwrap();
        let expected_r2 = case["r2"].as_f64().unwrap();

        assert_close(
            mse(&y_true, &y_pred).unwrap(),
            expected_mse,
            TOL,
            &format!("{}/mse", name),
        );
        assert_close(
            mae(&y_true, &y_pred).unwrap(),
            expected_mae,
            TOL,
            &format!("{}/mae", name),
        );
        assert_close(
            r2_score(&y_true, &y_pred).unwrap(),
            expected_r2,
            TOL,
            &format!("{}/r2", name),
        );
    }
}

#[test]
fn test_golden_classification_metrics() {
    let cases = load_golden_data("metrics.json");

    for case in &cases {
        let name = case["name"].as_str().unwrap();
        if !name.contains("classification") {
            continue;
        }

        let y_true = json_to_array1(&case["y_true"]);
        let y_pred = json_to_array1(&case["y_pred"]);

        // Accuracy
        let expected_acc = case["accuracy"].as_f64().unwrap();
        assert_close(
            accuracy_score(&y_true, &y_pred).unwrap(),
            expected_acc,
            TOL,
            &format!("{}/accuracy", name),
        );

        // Confusion matrix
        let expected_cm = common::json_to_array2(&case["confusion_matrix"]);
        let actual_cm = confusion_matrix(&y_true, &y_pred).unwrap();
        assert_array2_close(
            &actual_cm,
            &expected_cm,
            TOL,
            &format!("{}/confusion_matrix", name),
        );

        // Precision
        let expected_prec = json_to_array1(&case["precision"]);
        let actual_prec = precision(&y_true, &y_pred).unwrap();
        assert_array1_close(
            &actual_prec,
            &expected_prec,
            TOL,
            &format!("{}/precision", name),
        );

        // Recall
        let expected_rec = json_to_array1(&case["recall"]);
        let actual_rec = recall(&y_true, &y_pred).unwrap();
        assert_array1_close(&actual_rec, &expected_rec, TOL, &format!("{}/recall", name));

        // F1
        let expected_f1 = json_to_array1(&case["f1"]);
        let actual_f1 = f1_score(&y_true, &y_pred).unwrap();
        assert_array1_close(&actual_f1, &expected_f1, TOL, &format!("{}/f1", name));
    }
}
