//! Golden test for MultiOutputRegressor against sklearn 1.8.0.

mod common;

use anofox_ml::core::MultiOutputRegressor;
use anofox_ml::prelude::*;
use common::{assert_array2_close, json_to_array2, load_golden_data};

#[test]
fn test_golden_multi_output_ridge() {
    let cases = load_golden_data("multi_output.json");
    let case = &cases[0];

    let x = json_to_array2(&case["X"]);
    let y = json_to_array2(&case["Y"]);
    let alpha = case["alpha"].as_f64().unwrap();
    let expected = json_to_array2(&case["predictions"]);

    let inner = RidgeRegressor::new().with_lambda(alpha);
    let mor = MultiOutputRegressor::<f64>::new(inner);
    let fitted = mor.fit_2d(&x, &y).unwrap();
    let preds = fitted.predict_2d(&x).unwrap();

    assert_array2_close(&preds, &expected, 1e-6, "mor ridge predictions");
}
