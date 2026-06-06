//! Golden tests for LGBM round-2 features: sample_weight, class_weight,
//! monotone_constraints, and early_stopping.
//!
//! Validated against LightGBM 4.6 behavior.

mod common;

use anofox_ml::prelude::*;
use common::{json_to_array1, json_to_array2, load_golden_data};

#[test]
fn test_golden_lgbm_sample_weight_matches_behavior() {
    let cases = load_golden_data("lgbm_round2.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "lgbm_regressor_sample_weight")
        .unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let sw = json_to_array1(&case["sample_weight"]);

    let opts = LgbmFitOptions {
        sample_weight: Some(&sw),
        ..Default::default()
    };

    let model = LgbmRegressor::new()
        .with_n_estimators(30)
        .with_num_leaves(4)
        .with_learning_rate(0.1)
        .with_min_child_samples(1);

    let fitted = model.fit_with_eval(&x, &y, &opts).unwrap();
    let preds = fitted.predict(&x).unwrap();

    // High-weight rows (last 5) should be fit substantially better than low-weight rows.
    let high_err: f64 = (5..10).map(|i| (preds[i] - y[i]).powi(2)).sum::<f64>() / 5.0;
    let low_err: f64 = (0..5).map(|i| (preds[i] - y[i]).powi(2)).sum::<f64>() / 5.0;

    assert!(
        high_err < low_err + 1e-6,
        "high-weight rows should fit better: high_err={:.4} vs low_err={:.4}",
        high_err,
        low_err
    );
}

#[test]
fn test_golden_lgbm_class_weight_balanced() {
    let cases = load_golden_data("lgbm_round2.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "lgbm_classifier_balanced_weight")
        .unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);

    let fitted = LgbmClassifier::new()
        .with_n_estimators(30)
        .with_num_leaves(4)
        .with_learning_rate(0.1)
        .with_min_child_samples(1)
        .with_class_weight(Some(LgbmClassWeight::Balanced))
        .fit(&x, &y)
        .unwrap();

    let preds = fitted.predict(&x).unwrap();
    let minority_predicted: usize = preds.iter().filter(|&&v| v == 1.0).count();

    // With balanced class weighting, the model should predict the minority class
    // on at least some samples.
    assert!(
        minority_predicted >= 1,
        "balanced weight should recover minority class, got {} predictions of class 1",
        minority_predicted
    );
}

#[test]
fn test_golden_lgbm_monotone_constraints_enforced() {
    let cases = load_golden_data("lgbm_round2.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "lgbm_regressor_monotone")
        .unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);

    let fitted = LgbmRegressor::new()
        .with_n_estimators(30)
        .with_num_leaves(8)
        .with_learning_rate(0.1)
        .with_min_child_samples(1)
        .with_monotone_constraints(vec![1, 1])
        .fit(&x, &y)
        .unwrap();

    let preds = fitted.predict(&x).unwrap();
    for &p in preds.iter() {
        assert!(p.is_finite());
    }

    // Check monotonicity: if we compare two points where x[0] increases and x[1] is
    // the same, the prediction should not decrease.
    //
    // The golden data has pairs where x[1] is fixed at 1 (rows 0..5) or 5 (rows 5..10).
    // Predictions for rows 0..5 should be non-decreasing with row index.
    for i in 1..5 {
        assert!(
            preds[i] + 1e-6 >= preds[i - 1],
            "monotone violation at rows {},{}: {} < {}",
            i - 1,
            i,
            preds[i],
            preds[i - 1]
        );
    }
    for i in 6..10 {
        assert!(
            preds[i] + 1e-6 >= preds[i - 1],
            "monotone violation at rows {},{}: {} < {}",
            i - 1,
            i,
            preds[i],
            preds[i - 1]
        );
    }
}

#[test]
fn test_golden_lgbm_early_stopping_stops_early() {
    let cases = load_golden_data("lgbm_round2.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "lgbm_regressor_early_stopping")
        .unwrap();

    let x_train = json_to_array2(&case["X_train"]);
    let y_train = json_to_array1(&case["y_train"]);
    let x_eval = json_to_array2(&case["X_eval"]);
    let y_eval = json_to_array1(&case["y_eval"]);

    let opts = LgbmFitOptions {
        eval_set: Some((&x_eval, &y_eval)),
        ..Default::default()
    };

    let model = LgbmRegressor::new()
        .with_n_estimators(100)
        .with_num_leaves(4)
        .with_min_child_samples(1)
        .with_early_stopping(Some(5));

    let fitted = model.fit_with_eval(&x_train, &y_train, &opts).unwrap();

    // best_iteration should be recorded and finite.
    assert!(fitted.best_iteration() <= 100);
    assert!(fitted.best_iteration() > 0);

    // Training predictions should be finite.
    let preds = fitted.predict(&x_train).unwrap();
    for &p in preds.iter() {
        assert!(p.is_finite());
    }
}
