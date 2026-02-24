mod common;

use common::{assert_array2_close, json_to_array2, load_golden_data};
use rustml::prelude::*;

const TOL: f64 = 1e-10;

#[test]
fn test_golden_standard_scaler() {
    let cases = load_golden_data("preprocessing.json");

    for case in &cases {
        let name = case["name"].as_str().unwrap();
        if case["algorithm"].as_str().unwrap() != "StandardScaler" {
            continue;
        }

        let x = json_to_array2(&case["X"]);
        let expected_transformed = json_to_array2(&case["X_transformed"]);
        let expected_inverse = json_to_array2(&case["X_inverse"]);

        let scaler = StandardScaler::default();
        let fitted = FitUnsupervised::<f64>::fit(&scaler, &x).unwrap();

        // Check mean
        let expected_mean = common::json_to_array1(&case["mean"]);
        common::assert_array1_close(
            fitted.mean(),
            &expected_mean,
            TOL,
            &format!("{}/mean", name),
        );

        // Check std
        let expected_std = common::json_to_array1(&case["std"]);
        common::assert_array1_close(
            fitted.std(),
            &expected_std,
            TOL,
            &format!("{}/std", name),
        );

        // Check transform
        let transformed = fitted.transform(&x).unwrap();
        assert_array2_close(
            &transformed,
            &expected_transformed,
            TOL,
            &format!("{}/transform", name),
        );

        // Check inverse_transform roundtrip
        let inversed = fitted.inverse_transform(&transformed).unwrap();
        assert_array2_close(
            &inversed,
            &expected_inverse,
            TOL,
            &format!("{}/inverse", name),
        );
    }
}

#[test]
fn test_golden_minmax_scaler() {
    let cases = load_golden_data("preprocessing.json");

    for case in &cases {
        let name = case["name"].as_str().unwrap();
        if case["algorithm"].as_str().unwrap() != "MinMaxScaler" {
            continue;
        }

        let x = json_to_array2(&case["X"]);
        let expected_transformed = json_to_array2(&case["X_transformed"]);
        let expected_inverse = json_to_array2(&case["X_inverse"]);

        let feature_min = case["feature_min"].as_f64().unwrap();
        let feature_max = case["feature_max"].as_f64().unwrap();

        let scaler = MinMaxScaler {
            feature_min,
            feature_max,
        };
        let fitted = FitUnsupervised::<f64>::fit(&scaler, &x).unwrap();

        // Check data_min
        let expected_data_min = common::json_to_array1(&case["data_min"]);
        common::assert_array1_close(
            fitted.data_min(),
            &expected_data_min,
            TOL,
            &format!("{}/data_min", name),
        );

        // Check data_max
        let expected_data_max = common::json_to_array1(&case["data_max"]);
        common::assert_array1_close(
            fitted.data_max(),
            &expected_data_max,
            TOL,
            &format!("{}/data_max", name),
        );

        // Check transform
        let transformed = fitted.transform(&x).unwrap();
        assert_array2_close(
            &transformed,
            &expected_transformed,
            TOL,
            &format!("{}/transform", name),
        );

        // Check inverse_transform roundtrip
        let inversed = fitted.inverse_transform(&transformed).unwrap();
        assert_array2_close(
            &inversed,
            &expected_inverse,
            TOL,
            &format!("{}/inverse", name),
        );
    }
}
