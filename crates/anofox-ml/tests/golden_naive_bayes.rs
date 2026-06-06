mod common;

use anofox_ml::prelude::*;
use common::{assert_array1_close, json_to_array1, json_to_array2, load_golden_data};

const TOL: f64 = 1e-10;

#[test]
fn test_golden_gaussian_nb() {
    let cases = load_golden_data("naive_bayes.json");

    for case in &cases {
        let name = case["name"].as_str().unwrap();

        let x_train = json_to_array2(&case["X_train"]);
        let y_train = json_to_array1(&case["y_train"]);
        let x_test = json_to_array2(&case["X_test"]);
        let expected_pred = json_to_array1(&case["y_pred"]);

        let gnb = GaussianNB::default();
        let fitted = Fit::fit(&gnb, &x_train, &y_train).unwrap();
        let preds = fitted.predict(&x_test).unwrap();

        assert_array1_close(&preds, &expected_pred, TOL, &format!("{}/predict", name));
    }
}
