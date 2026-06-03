//! Golden test for TransformedTargetRegressor against sklearn 1.8.0.

mod common;

use common::{assert_array1_close, json_to_array1, json_to_array2, load_golden_data};
use rustml::prelude::*;
use rustml::regression::TransformedTargetRegressor;

#[test]
fn test_golden_transformed_target_ridge_log_exp() {
    let cases = load_golden_data("transformed_target.json");
    let case = cases.iter().find(|c| c["name"] == "ridge_log_exp").unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let alpha = case["alpha"].as_f64().unwrap();
    let expected = json_to_array1(&case["predictions"]);

    let inner = RidgeRegressor::new().with_lambda(alpha);
    let wrapped = TransformedTargetRegressor::new(inner, f64::ln, f64::exp);
    let fitted = wrapped.fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    // Ridge has a closed-form solution; the round-trip log/exp is exact in
    // sklearn (np.log then np.exp). Allow a tight tolerance.
    assert_array1_close(&preds, &expected, 1e-6, "tt_ridge_log_exp predictions");
}
