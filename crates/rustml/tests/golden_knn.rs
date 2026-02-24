mod common;

use common::{assert_array1_close, json_to_array1, json_to_array2, load_golden_data};
use rustml::prelude::*;

const TOL: f64 = 1e-10;

#[test]
fn test_golden_knn_classifier() {
    let cases = load_golden_data("knn.json");

    for case in &cases {
        let name = case["name"].as_str().unwrap();
        if case["algorithm"].as_str().unwrap() != "KnnClassifier" {
            continue;
        }

        let x_train = json_to_array2(&case["X_train"]);
        let y_train = json_to_array1(&case["y_train"]);
        let x_test = json_to_array2(&case["X_test"]);
        let expected_pred = json_to_array1(&case["y_pred"]);

        let n_neighbors = case["n_neighbors"].as_u64().unwrap() as usize;
        let weights = match case["weights"].as_str().unwrap() {
            "uniform" => WeightFunction::Uniform,
            "distance" => WeightFunction::Distance,
            w => panic!("unknown weight function: {}", w),
        };
        let metric = match case["metric"].as_str().unwrap() {
            "euclidean" => DistanceMetric::Euclidean,
            "manhattan" => DistanceMetric::Manhattan,
            m => panic!("unknown metric: {}", m),
        };

        let knn = KnnClassifier {
            n_neighbors,
            weights,
            metric,
        };

        let fitted = Fit::fit(&knn, &x_train, &y_train).unwrap();
        let preds = fitted.predict(&x_test).unwrap();

        assert_array1_close(&preds, &expected_pred, TOL, &format!("{}/predict", name));
    }
}

#[test]
fn test_golden_knn_regressor() {
    let cases = load_golden_data("knn.json");

    for case in &cases {
        let name = case["name"].as_str().unwrap();
        if case["algorithm"].as_str().unwrap() != "KnnRegressor" {
            continue;
        }

        let x_train = json_to_array2(&case["X_train"]);
        let y_train = json_to_array1(&case["y_train"]);
        let x_test = json_to_array2(&case["X_test"]);
        let expected_pred = json_to_array1(&case["y_pred"]);

        let n_neighbors = case["n_neighbors"].as_u64().unwrap() as usize;
        let weights = match case["weights"].as_str().unwrap() {
            "uniform" => WeightFunction::Uniform,
            "distance" => WeightFunction::Distance,
            w => panic!("unknown weight function: {}", w),
        };
        let metric = match case["metric"].as_str().unwrap() {
            "euclidean" => DistanceMetric::Euclidean,
            "manhattan" => DistanceMetric::Manhattan,
            m => panic!("unknown metric: {}", m),
        };

        let knn = KnnRegressor {
            n_neighbors,
            weights,
            metric,
        };

        let fitted = Fit::fit(&knn, &x_train, &y_train).unwrap();
        let preds = fitted.predict(&x_test).unwrap();

        assert_array1_close(&preds, &expected_pred, TOL, &format!("{}/predict", name));
    }
}
