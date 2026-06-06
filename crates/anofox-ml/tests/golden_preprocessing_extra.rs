mod common;

use anofox_ml::prelude::*;
use common::{assert_array2_close, json_to_array2, load_golden_data};

const TOL: f64 = 1e-10;

#[test]
fn test_golden_max_abs_scaler() {
    let cases = load_golden_data("preprocessing_extra.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "max_abs_scaler")
        .unwrap();

    let x = json_to_array2(&case["X"]);
    let expected = json_to_array2(&case["X_transformed"]);

    let fitted = MaxAbsScaler::new().fit(&x).unwrap();
    let actual = fitted.transform(&x).unwrap();

    assert_array2_close(&actual, &expected, TOL, "max_abs_scaler");
}

#[test]
fn test_golden_normalizer_l2() {
    let cases = load_golden_data("preprocessing_extra.json");
    let case = cases.iter().find(|c| c["name"] == "normalizer_l2").unwrap();

    let x = json_to_array2(&case["X"]);
    let expected = json_to_array2(&case["X_transformed"]);

    let fitted = Normalizer::new().fit(&x).unwrap();
    let actual = fitted.transform(&x).unwrap();

    assert_array2_close(&actual, &expected, TOL, "normalizer_l2");
}

#[test]
fn test_golden_normalizer_l1() {
    let cases = load_golden_data("preprocessing_extra.json");
    let case = cases.iter().find(|c| c["name"] == "normalizer_l1").unwrap();

    let x = json_to_array2(&case["X"]);
    let expected = json_to_array2(&case["X_transformed"]);

    let fitted = Normalizer::new().with_norm(NormType::L1).fit(&x).unwrap();
    let actual = fitted.transform(&x).unwrap();

    assert_array2_close(&actual, &expected, TOL, "normalizer_l1");
}

#[test]
fn test_golden_binarizer() {
    let cases = load_golden_data("preprocessing_extra.json");
    let case = cases.iter().find(|c| c["name"] == "binarizer").unwrap();

    let x = json_to_array2(&case["X"]);
    let expected = json_to_array2(&case["X_transformed"]);
    let threshold = case["threshold"].as_f64().unwrap();

    let fitted = Binarizer::new(threshold).fit(&x).unwrap();
    let actual = fitted.transform(&x).unwrap();

    assert_array2_close(&actual, &expected, TOL, "binarizer");
}

#[test]
fn test_golden_robust_scaler() {
    let cases = load_golden_data("preprocessing_extra.json");
    let case = cases.iter().find(|c| c["name"] == "robust_scaler").unwrap();

    let x = json_to_array2(&case["X"]);
    let expected = json_to_array2(&case["X_transformed"]);

    let fitted = RobustScaler::new().fit(&x).unwrap();
    let actual = fitted.transform(&x).unwrap();

    assert_array2_close(&actual, &expected, TOL, "robust_scaler");
}
