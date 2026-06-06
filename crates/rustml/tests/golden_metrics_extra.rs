mod common;

use common::{assert_close, json_to_array1, json_to_array2, load_golden_data};
use rustml::prelude::*;

const TOL: f64 = 1e-6;

#[test]
fn test_golden_log_loss() {
    let cases = load_golden_data("metrics_extra.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "log_loss_binary")
        .unwrap();

    let y_true = json_to_array1(&case["y_true"]);
    let y_prob = json_to_array1(&case["y_prob"]);
    let expected = case["log_loss"].as_f64().unwrap();

    let actual: f64 = log_loss(&y_true, &y_prob).unwrap();
    assert_close(actual, expected, TOL, "log_loss");
}

#[test]
fn test_golden_balanced_accuracy() {
    let cases = load_golden_data("metrics_extra.json");

    for case in &cases {
        let name = case["name"].as_str().unwrap();
        if !name.starts_with("balanced_accuracy") {
            continue;
        }
        let y_true = json_to_array1(&case["y_true"]);
        let y_pred = json_to_array1(&case["y_pred"]);
        let expected = case["balanced_accuracy"].as_f64().unwrap();

        let actual: f64 = balanced_accuracy_score(&y_true, &y_pred).unwrap();
        assert_close(
            actual,
            expected,
            TOL,
            &format!("{}/balanced_accuracy", name),
        );
    }
}

#[test]
fn test_golden_cohen_kappa() {
    let cases = load_golden_data("metrics_extra.json");
    let case = cases.iter().find(|c| c["name"] == "cohen_kappa").unwrap();

    let y_true = json_to_array1(&case["y_true"]);
    let y_pred = json_to_array1(&case["y_pred"]);
    let expected = case["cohen_kappa"].as_f64().unwrap();

    let actual: f64 = cohen_kappa_score(&y_true, &y_pred).unwrap();
    assert_close(actual, expected, TOL, "cohen_kappa");
}

#[test]
fn test_golden_silhouette_score() {
    let cases = load_golden_data("metrics_extra.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "silhouette_well_separated")
        .unwrap();

    let x = json_to_array2(&case["X"]);
    let labels = json_to_array1(&case["labels"]);
    let expected = case["silhouette_score"].as_f64().unwrap();

    let actual: f64 = silhouette_score(&x, &labels).unwrap();
    assert_close(actual, expected, TOL, "silhouette_score");
}

#[test]
fn test_golden_median_absolute_error() {
    let cases = load_golden_data("metrics_extra.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "regression_extra")
        .unwrap();

    let y_true = json_to_array1(&case["y_true"]);
    let y_pred = json_to_array1(&case["y_pred"]);
    let expected = case["median_absolute_error"].as_f64().unwrap();

    let actual: f64 = median_absolute_error(&y_true, &y_pred).unwrap();
    assert_close(actual, expected, TOL, "median_absolute_error");
}

#[test]
fn test_golden_mean_squared_log_error() {
    let cases = load_golden_data("metrics_extra.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "regression_extra")
        .unwrap();

    let y_true = json_to_array1(&case["y_true"]);
    let y_pred = json_to_array1(&case["y_pred"]);
    let expected = case["mean_squared_log_error"].as_f64().unwrap();

    let actual: f64 = mean_squared_log_error(&y_true, &y_pred).unwrap();
    assert_close(actual, expected, TOL, "mean_squared_log_error");
}

#[test]
fn test_golden_roc_auc() {
    let cases = load_golden_data("metrics_extra.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "extended_classification")
        .unwrap();

    let y_true = json_to_array1(&case["y_true"]);
    let y_scores = json_to_array1(&case["y_scores"]);
    let expected = case["roc_auc"].as_f64().unwrap();

    let actual: f64 = roc_auc_score(&y_true, &y_scores).unwrap();
    assert_close(actual, expected, TOL, "roc_auc");
}

#[test]
fn test_golden_average_precision() {
    let cases = load_golden_data("metrics_extra.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "extended_classification")
        .unwrap();

    let y_true = json_to_array1(&case["y_true"]);
    let y_scores = json_to_array1(&case["y_scores"]);
    let expected = case["average_precision"].as_f64().unwrap();

    let actual: f64 = average_precision_score(&y_true, &y_scores).unwrap();
    assert_close(actual, expected, TOL, "average_precision");
}

#[test]
fn test_golden_matthews_corrcoef() {
    let cases = load_golden_data("metrics_extra.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "extended_classification")
        .unwrap();

    let y_true = json_to_array1(&case["y_true"]);
    let y_pred = json_to_array1(&case["y_pred"]);
    let expected = case["matthews_corrcoef"].as_f64().unwrap();

    let actual: f64 = matthews_corrcoef(&y_true, &y_pred).unwrap();
    assert_close(actual, expected, TOL, "matthews_corrcoef");
}

#[test]
fn test_golden_explained_variance() {
    let cases = load_golden_data("metrics_extra.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "regression_extended")
        .unwrap();

    let y_true = json_to_array1(&case["y_true"]);
    let y_pred = json_to_array1(&case["y_pred"]);
    let expected = case["explained_variance"].as_f64().unwrap();

    let actual: f64 = explained_variance_score(&y_true, &y_pred).unwrap();
    assert_close(actual, expected, TOL, "explained_variance");
}

#[test]
fn test_golden_max_error() {
    let cases = load_golden_data("metrics_extra.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "regression_extended")
        .unwrap();

    let y_true = json_to_array1(&case["y_true"]);
    let y_pred = json_to_array1(&case["y_pred"]);
    let expected = case["max_error"].as_f64().unwrap();

    let actual: f64 = max_error(&y_true, &y_pred).unwrap();
    assert_close(actual, expected, TOL, "max_error");
}

#[test]
fn test_golden_mape() {
    let cases = load_golden_data("metrics_extra.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "regression_extended")
        .unwrap();

    let y_true = json_to_array1(&case["y_true"]);
    let y_pred = json_to_array1(&case["y_pred"]);
    let expected = case["mape"].as_f64().unwrap();

    let actual: f64 = mean_absolute_percentage_error(&y_true, &y_pred).unwrap();
    assert_close(actual, expected, TOL, "mape");
}
