//! Edge case golden tests for LGBM, validated against real LightGBM 4.6.
//!
//! These exercise tricky inputs that could reveal implementation divergence
//! from the reference: constant targets, extreme regularization, NaN in test
//! only, duplicate rows, constant features, and many classes.

mod common;

use anofox_ml::prelude::*;
use common::{json_to_array1, json_to_array2, load_golden_data};
use ndarray::Array2;

fn json_to_array2_with_nan(val: &serde_json::Value) -> Array2<f64> {
    let rows: Vec<Vec<f64>> = val
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap_or(f64::NAN))
                .collect()
        })
        .collect();
    let nrows = rows.len();
    let ncols = rows[0].len();
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    Array2::from_shape_vec((nrows, ncols), flat).unwrap()
}

#[test]
fn test_golden_lgbm_edge_constant_y() {
    let cases = load_golden_data("lgbm_edge_cases.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "edge_constant_y")
        .unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let expected = case["expected_constant"].as_f64().unwrap();

    let fitted = LgbmRegressor::new()
        .with_n_estimators(20)
        .with_num_leaves(4)
        .with_learning_rate(0.1)
        .with_min_child_samples(1)
        .fit(&x, &y)
        .unwrap();

    let preds = fitted.predict(&x).unwrap();
    for &p in preds.iter() {
        assert!(
            (p - expected).abs() < 0.5,
            "expected ~{}, got {}",
            expected,
            p
        );
    }
}

#[test]
fn test_golden_lgbm_edge_tiny_lr() {
    let cases = load_golden_data("lgbm_edge_cases.json");
    let case = cases.iter().find(|c| c["name"] == "edge_tiny_lr").unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let mean_y = case["mean_y"].as_f64().unwrap();

    // Very small learning rate: predictions should be essentially mean(y)
    let fitted = LgbmRegressor::new()
        .with_n_estimators(10)
        .with_num_leaves(4)
        .with_learning_rate(1e-8)
        .with_min_child_samples(1)
        .fit(&x, &y)
        .unwrap();

    let preds = fitted.predict(&x).unwrap();
    for &p in preds.iter() {
        assert!(
            (p - mean_y).abs() < 0.5,
            "tiny lr should give ~mean_y={}, got {}",
            mean_y,
            p
        );
    }
}

#[test]
fn test_golden_lgbm_edge_extreme_l2() {
    let cases = load_golden_data("lgbm_edge_cases.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "edge_extreme_l2")
        .unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let mean_y = case["mean_y"].as_f64().unwrap();

    let fitted = LgbmRegressor::new()
        .with_n_estimators(10)
        .with_num_leaves(4)
        .with_learning_rate(0.1)
        .with_min_child_samples(1)
        .with_reg_lambda(1e6)
        .fit(&x, &y)
        .unwrap();

    let preds = fitted.predict(&x).unwrap();
    for &p in preds.iter() {
        assert!(
            (p - mean_y).abs() < 1.0,
            "extreme L2 should collapse to mean_y={}, got {}",
            mean_y,
            p
        );
    }
}

#[test]
fn test_golden_lgbm_edge_many_classes() {
    let cases = load_golden_data("lgbm_edge_cases.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "edge_many_classes")
        .unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let n_classes = case["n_classes"].as_u64().unwrap() as usize;
    let lgbm_acc = case["accuracy"].as_f64().unwrap();

    let fitted = LgbmClassifier::new()
        .with_n_estimators(30)
        .with_num_leaves(4)
        .with_learning_rate(0.1)
        .with_min_child_samples(1)
        .fit(&x, &y)
        .unwrap();

    assert_eq!(fitted.classes().len(), n_classes);

    let preds = fitted.predict(&x).unwrap();
    let correct: usize = preds.iter().zip(y.iter()).filter(|(&p, &t)| p == t).count();
    let our_acc = correct as f64 / x.nrows() as f64;

    assert!(
        our_acc >= lgbm_acc - 0.2,
        "accuracy {} should be within 0.2 of LightGBM's {}",
        our_acc,
        lgbm_acc
    );
}

#[test]
fn test_golden_lgbm_edge_nan_in_test_only() {
    let cases = load_golden_data("lgbm_edge_cases.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "edge_nan_in_test_only")
        .unwrap();

    let x_train = json_to_array2(&case["X_train"]);
    let y_train = json_to_array1(&case["y_train"]);
    let x_test = json_to_array2_with_nan(&case["X_test"]);

    let fitted = LgbmRegressor::new()
        .with_n_estimators(10)
        .with_num_leaves(4)
        .with_learning_rate(0.1)
        .with_min_child_samples(1)
        .fit(&x_train, &y_train)
        .unwrap();

    let preds = fitted.predict(&x_test).unwrap();
    for (i, &p) in preds.iter().enumerate() {
        assert!(p.is_finite(), "pred[{}] should be finite", i);
    }
}

#[test]
fn test_golden_lgbm_edge_duplicate_rows() {
    let cases = load_golden_data("lgbm_edge_cases.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "edge_duplicate_rows")
        .unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let lgbm_acc = case["accuracy"].as_f64().unwrap();

    let fitted = LgbmClassifier::new()
        .with_n_estimators(10)
        .with_num_leaves(4)
        .with_learning_rate(0.1)
        .with_min_child_samples(1)
        .fit(&x, &y)
        .unwrap();

    let preds = fitted.predict(&x).unwrap();
    let correct: usize = preds.iter().zip(y.iter()).filter(|(&p, &t)| p == t).count();
    let our_acc = correct as f64 / y.len() as f64;

    assert!(
        our_acc >= lgbm_acc - 0.2,
        "duplicate rows accuracy {} should be within 0.2 of LightGBM's {}",
        our_acc,
        lgbm_acc
    );
}

#[test]
fn test_golden_lgbm_edge_constant_feature_zero_importance() {
    let cases = load_golden_data("lgbm_edge_cases.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "edge_constant_feature")
        .unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let lgbm_imp: Vec<f64> = case["feature_importances"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    let fitted = LgbmRegressor::new()
        .with_n_estimators(20)
        .with_num_leaves(4)
        .with_learning_rate(0.1)
        .with_min_child_samples(1)
        .fit(&x, &y)
        .unwrap();

    let imp = fitted.feature_importances();
    assert_eq!(imp.len(), 2);

    // Constant feature should have zero importance in both implementations
    assert_eq!(imp[1], 0.0);
    assert_eq!(lgbm_imp[1], 0.0);

    // Feature 0 should have all the importance
    assert!(imp[0] > 0.0);
    assert!(lgbm_imp[0] > 0.0);
}
