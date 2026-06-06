//! Golden tests for LgbmRegressor / LgbmClassifier.
//!
//! Validates our LightGBM-lite implementation against real LightGBM 4.6 output
//! stored in `tests/golden_data/lgbm_golden.json`. Because our implementation
//! differs from the C++ reference in binning, tie-breaking, and numerical
//! precision, we validate *behavioral* equivalence (MSE, R², accuracy) with
//! reasonable tolerances rather than exact prediction match.

mod common;

use anofox_ml::prelude::*;
use common::{json_to_array1, json_to_array2, load_golden_data};
use ndarray::{Array1, Array2};

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

fn mse(pred: &Array1<f64>, target: &Array1<f64>) -> f64 {
    pred.iter()
        .zip(target.iter())
        .map(|(&p, &t)| (p - t).powi(2))
        .sum::<f64>()
        / pred.len() as f64
}

fn r2(pred: &Array1<f64>, target: &Array1<f64>) -> f64 {
    let mean = target.iter().sum::<f64>() / target.len() as f64;
    let ss_res: f64 = pred
        .iter()
        .zip(target.iter())
        .map(|(&p, &t)| (p - t).powi(2))
        .sum();
    let ss_tot: f64 = target.iter().map(|&t| (t - mean).powi(2)).sum();
    if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    }
}

fn accuracy(pred: &Array1<f64>, target: &Array1<f64>) -> f64 {
    let correct: usize = pred
        .iter()
        .zip(target.iter())
        .filter(|(&p, &t)| (p - t).abs() < 1e-10)
        .count();
    correct as f64 / pred.len() as f64
}

#[test]
fn test_golden_lgbm_regressor_linear() {
    let cases = load_golden_data("lgbm_golden.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "lgbm_regressor_linear")
        .unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let lgbm_mse = case["mse"].as_f64().unwrap();

    let model = LgbmRegressor::new()
        .with_n_estimators(50)
        .with_num_leaves(4)
        .with_learning_rate(0.1)
        .with_min_child_samples(1);

    let fitted = Fit::fit(&model, &x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();
    let our_mse = mse(&preds, &y);

    // With min_data_in_bin=3 matching LightGBM's default, our MSE should be
    // within a few percent of theirs.
    assert!(
        (our_mse - lgbm_mse).abs() < lgbm_mse * 0.05 + 0.01,
        "Our MSE {:.6} differs from LightGBM's {:.6}",
        our_mse,
        lgbm_mse
    );
    // Per-point predictions should be close.
    let lgbm_preds = json_to_array1(&case["predictions"]);
    for (i, (&o, &s)) in preds.iter().zip(lgbm_preds.iter()).enumerate() {
        assert!(
            (o - s).abs() < 0.5,
            "pred[{}]: ours={:.4}, lgbm={:.4}",
            i,
            o,
            s
        );
    }
}

#[test]
fn test_golden_lgbm_classifier_binary_matches_lgbm_accuracy() {
    let cases = load_golden_data("lgbm_golden.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "lgbm_classifier_binary")
        .unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let lgbm_acc = case["accuracy"].as_f64().unwrap();

    let model = LgbmClassifier::new()
        .with_n_estimators(30)
        .with_num_leaves(4)
        .with_learning_rate(0.1)
        .with_min_child_samples(1);

    let fitted = Fit::fit(&model, &x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();
    let our_acc = accuracy(&preds, &y);

    // On well-separated data, both implementations should hit 100%.
    assert!(
        our_acc >= lgbm_acc - 0.1,
        "Our accuracy {:.4} is worse than LightGBM's {:.4}",
        our_acc,
        lgbm_acc
    );
}

#[test]
fn test_golden_lgbm_classifier_binary_proba_shape() {
    let cases = load_golden_data("lgbm_golden.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "lgbm_classifier_binary")
        .unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);

    let fitted = LgbmClassifier::new()
        .with_n_estimators(30)
        .with_num_leaves(4)
        .with_min_child_samples(1)
        .fit(&x, &y)
        .unwrap();

    let proba = fitted.predict_proba(&x).unwrap();
    assert_eq!(proba.ncols(), 2);

    // Each row should sum to 1
    for i in 0..x.nrows() {
        let row_sum: f64 = (0..proba.ncols()).map(|c| proba[[i, c]]).sum();
        assert!((row_sum - 1.0).abs() < 1e-10, "row {} sum = {}", i, row_sum);
        // Probabilities in [0, 1]
        for c in 0..proba.ncols() {
            let p = proba[[i, c]];
            assert!(
                (0.0..=1.0).contains(&p),
                "proba[{},{}] = {} out of range",
                i,
                c,
                p
            );
        }
    }
}

#[test]
fn test_golden_lgbm_classifier_multiclass() {
    let cases = load_golden_data("lgbm_golden.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "lgbm_classifier_multiclass")
        .unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let lgbm_acc = case["accuracy"].as_f64().unwrap();

    let fitted = LgbmClassifier::new()
        .with_n_estimators(30)
        .with_num_leaves(4)
        .with_min_child_samples(1)
        .fit(&x, &y)
        .unwrap();

    let preds = fitted.predict(&x).unwrap();
    let our_acc = accuracy(&preds, &y);

    assert!(
        our_acc >= lgbm_acc - 0.2,
        "multiclass accuracy {:.4} worse than LightGBM's {:.4}",
        our_acc,
        lgbm_acc
    );

    // Softmax should be valid
    let proba = fitted.predict_proba(&x).unwrap();
    assert_eq!(proba.ncols(), 3);
    for i in 0..x.nrows() {
        let sum: f64 = (0..3).map(|c| proba[[i, c]]).sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
}

#[test]
fn test_golden_lgbm_regressor_nan_handling() {
    let cases = load_golden_data("lgbm_golden.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "lgbm_regressor_nan")
        .unwrap();

    let x = json_to_array2_with_nan(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let lgbm_mse = case["mse"].as_f64().unwrap();

    let fitted = LgbmRegressor::new()
        .with_n_estimators(20)
        .with_num_leaves(4)
        .with_min_child_samples(1)
        .fit(&x, &y)
        .unwrap();

    let preds = fitted.predict(&x).unwrap();

    // All predictions must be finite (no NaN leakage)
    for (i, &p) in preds.iter().enumerate() {
        assert!(p.is_finite(), "pred[{}] = {}", i, p);
    }

    let our_mse = mse(&preds, &y);
    // With the fixed missing-value gradient accounting, our MSE should be
    // within a few percent of LightGBM's.
    assert!(
        (our_mse - lgbm_mse).abs() < lgbm_mse * 0.10 + 0.5,
        "NaN-handling MSE {:.6} differs from LightGBM's {:.6}",
        our_mse,
        lgbm_mse
    );
}

#[test]
fn test_golden_lgbm_regressor_l2_regularization() {
    let cases = load_golden_data("lgbm_golden.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "lgbm_regressor_l2_reg")
        .unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let lgbm_r2 = case["r2"].as_f64().unwrap();

    let fitted = LgbmRegressor::new()
        .with_n_estimators(30)
        .with_num_leaves(8)
        .with_min_child_samples(2)
        .with_reg_lambda(1.0)
        .fit(&x, &y)
        .unwrap();

    let preds = fitted.predict(&x).unwrap();
    let our_r2 = r2(&preds, &y);

    // With min_data_in_bin matching LightGBM, our R² should be close.
    // On a 50-sample dataset, minor bin boundary differences cause a small gap.
    assert!(
        (our_r2 - lgbm_r2).abs() < 0.03,
        "R² {:.6} differs from LightGBM's {:.6}",
        our_r2,
        lgbm_r2
    );
}

#[test]
fn test_golden_lgbm_feature_importances_sum_to_one() {
    let cases = load_golden_data("lgbm_golden.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "lgbm_regressor_l2_reg")
        .unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);

    let fitted = LgbmRegressor::new()
        .with_n_estimators(10)
        .with_num_leaves(4)
        .with_min_child_samples(2)
        .fit(&x, &y)
        .unwrap();

    let imp = fitted.feature_importances();
    let sum: f64 = imp.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-9 || sum == 0.0,
        "feature_importances sum = {}, expected ~1.0",
        sum
    );
}
