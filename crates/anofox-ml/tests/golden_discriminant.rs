//! Golden test for LDA / QDA against sklearn 1.8.0.

mod common;

use anofox_ml::prelude::*;
use common::{json_to_array1, json_to_array2, load_golden_data};

fn accuracy(p: &ndarray::Array1<f64>, t: &ndarray::Array1<f64>) -> f64 {
    let correct = p
        .iter()
        .zip(t.iter())
        .filter(|(a, b)| (*a - *b).abs() < 0.5)
        .count();
    correct as f64 / p.len() as f64
}

#[test]
fn test_lda_matches_sklearn_high_agreement() {
    let cases = load_golden_data("discriminant.json");
    let case = cases.iter().find(|c| c["name"] == "lda_3class").unwrap();
    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let sklearn_pred = json_to_array1(&case["sklearn_predictions"]);

    let fitted = LinearDiscriminantAnalysis::new().fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    // sklearn solver=lsqr gives identical math; expect agreement on ≥ 98%
    // of samples (a small fraction may flip near class boundaries due to
    // tiny linear-algebra differences).
    let agree = accuracy(&preds, &sklearn_pred);
    let acc_rustml = accuracy(&preds, &y);
    assert!(agree >= 0.98, "agreement with sklearn = {agree}");
    assert!(acc_rustml >= 0.85, "anofox-ml accuracy = {acc_rustml}");
}

#[test]
fn test_qda_matches_sklearn_high_agreement() {
    let cases = load_golden_data("discriminant.json");
    let case = cases.iter().find(|c| c["name"] == "qda_binary").unwrap();
    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let sklearn_pred = json_to_array1(&case["sklearn_predictions"]);

    let fitted = QuadraticDiscriminantAnalysis::new().fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    let agree = accuracy(&preds, &sklearn_pred);
    let acc_rustml = accuracy(&preds, &y);
    assert!(agree >= 0.97, "agreement with sklearn = {agree}");
    assert!(acc_rustml >= 0.85, "anofox-ml accuracy = {acc_rustml}");
}
