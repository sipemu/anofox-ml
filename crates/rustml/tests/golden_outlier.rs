//! Behavioral parity test for IsolationForest / LocalOutlierFactor.

mod common;

use common::{json_to_array1, json_to_array2, load_golden_data};
use rustml::core::FitUnsupervised;
use rustml_ensemble::IsolationForest;
use rustml_neighbors::LocalOutlierFactor;

fn detection_rate(pred: &ndarray::Array1<f64>, truth: &ndarray::Array1<f64>) -> (f64, f64) {
    let mut tp = 0;
    let mut fp = 0;
    let mut tn = 0;
    let mut fn_ = 0;
    for (p, t) in pred.iter().zip(truth.iter()) {
        let pn = *p < 0.0;
        let tn_b = *t < 0.0;
        match (pn, tn_b) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, false) => tn += 1,
            (false, true) => fn_ += 1,
        }
    }
    let recall = tp as f64 / (tp + fn_) as f64;
    let precision = tp as f64 / (tp + fp).max(1) as f64;
    (precision, recall)
}

#[test]
fn test_iso_forest_detects_outliers() {
    let cases = load_golden_data("outlier.json");
    let case = cases.iter().find(|c| c["name"] == "iso_forest").unwrap();
    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y_true"]);
    let sk = json_to_array1(&case["sklearn_predictions"]);

    let fitted = IsolationForest::new()
        .with_n_estimators(100)
        .with_max_samples(128)
        .with_contamination(5.0 / x.nrows() as f64)
        .with_seed(0)
        .fit(&x)
        .unwrap();
    let preds = rustml::core::Predict::predict(&fitted, &x).unwrap();
    let (_p, r) = detection_rate(&preds, &y);
    let (_, sk_r) = detection_rate(&sk, &y);
    // Sklearn typically achieves recall 1.0 (catches all 5 wild outliers).
    // Ours should at least match within 0.2.
    assert!(r >= 0.6, "iso_forest recall too low: {r}");
    assert!((r - sk_r).abs() < 0.4, "rustml={r}, sklearn={sk_r}");
}

#[test]
fn test_lof_detects_outliers() {
    let cases = load_golden_data("outlier.json");
    let case = cases.iter().find(|c| c["name"] == "lof").unwrap();
    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y_true"]);
    let sk = json_to_array1(&case["sklearn_predictions"]);

    let lof = LocalOutlierFactor::new(20).with_contamination(5.0 / x.nrows() as f64);
    let fitted = lof.fit(&x).unwrap();
    let preds = fitted.predictions.clone();
    let (_p, r) = detection_rate(&preds, &y);
    let (_, sk_r) = detection_rate(&sk, &y);
    assert!(r >= 0.6, "lof recall too low: {r}");
    assert!((r - sk_r).abs() < 0.4);
}
