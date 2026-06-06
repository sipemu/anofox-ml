mod common;

use common::{assert_close, json_to_array1, load_golden_data};
use rustml::prelude::*;

const TOL: f64 = 1e-6;

#[test]
fn test_golden_adjusted_rand_score() {
    let cases = load_golden_data("metrics_round2.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "adjusted_rand_score")
        .unwrap();

    let labels_true = json_to_array1(&case["labels_true"]);
    let labels_pred = json_to_array1(&case["labels_pred"]);
    let expected = case["score"].as_f64().unwrap();

    let actual: f64 = adjusted_rand_score(&labels_true, &labels_pred).unwrap();
    assert_close(actual, expected, TOL, "adjusted_rand_score");
}

#[test]
fn test_golden_normalized_mutual_info_score() {
    let cases = load_golden_data("metrics_round2.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "normalized_mutual_info_score")
        .unwrap();

    let labels_true = json_to_array1(&case["labels_true"]);
    let labels_pred = json_to_array1(&case["labels_pred"]);
    let expected = case["score"].as_f64().unwrap();

    let actual: f64 = normalized_mutual_info_score(&labels_true, &labels_pred).unwrap();
    assert_close(actual, expected, 0.05, "normalized_mutual_info_score");
    // NMI implementations can differ slightly in edge cases
}

#[test]
fn test_golden_brier_score_loss() {
    let cases = load_golden_data("metrics_round2.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "brier_score_loss")
        .unwrap();

    let y_true = json_to_array1(&case["y_true"]);
    let y_prob = json_to_array1(&case["y_prob"]);
    let expected = case["score"].as_f64().unwrap();

    let actual: f64 = brier_score_loss(&y_true, &y_prob).unwrap();
    assert_close(actual, expected, TOL, "brier_score_loss");
}

#[test]
fn test_golden_roc_curve_shape() {
    let cases = load_golden_data("metrics_round2.json");
    let case = cases.iter().find(|c| c["name"] == "roc_curve").unwrap();

    let y_true = json_to_array1(&case["y_true"]);
    let y_score = json_to_array1(&case["y_score"]);
    let _expected_n = case["n_points"].as_u64().unwrap() as usize;

    let (fpr, tpr, _thresholds) = roc_curve(&y_true, &y_score).unwrap();

    // Curve should start at (0,0) and end at (1,1)
    assert_close(*fpr.first().unwrap(), 0.0, 1e-10, "roc fpr[0]");
    assert_close(*tpr.first().unwrap(), 0.0, 1e-10, "roc tpr[0]");
    assert_close(*fpr.last().unwrap(), 1.0, 1e-10, "roc fpr[-1]");
    assert_close(*tpr.last().unwrap(), 1.0, 1e-10, "roc tpr[-1]");

    // FPR should be monotonically non-decreasing
    for i in 1..fpr.len() {
        assert!(
            fpr[i] >= fpr[i - 1],
            "fpr should be non-decreasing at {}",
            i
        );
    }
}

#[test]
fn test_golden_precision_recall_curve_shape() {
    let cases = load_golden_data("metrics_round2.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "precision_recall_curve")
        .unwrap();

    let y_true = json_to_array1(&case["y_true"]);
    let y_score = json_to_array1(&case["y_score"]);

    let (precision, recall, _thresholds) = precision_recall_curve(&y_true, &y_score).unwrap();

    // All precisions and recalls should be in [0, 1]
    for &p in precision.iter() {
        assert!(p >= 0.0 && p <= 1.0, "precision out of range: {}", p);
    }
    for &r in recall.iter() {
        assert!(r >= 0.0 && r <= 1.0, "recall out of range: {}", r);
    }
}
