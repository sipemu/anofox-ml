//! Golden test for PLSRegression against sklearn 1.8.0.

mod common;

use anofox_ml::core::{Fit, Predict};
use anofox_ml_preprocessing::PlsRegression;
use common::{assert_array1_close, json_to_array1, json_to_array2, load_golden_data};

#[test]
fn test_pls1_matches_sklearn() {
    let cases = load_golden_data("pls.json");
    let case = &cases[0];
    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let k = case["n_components"].as_u64().unwrap() as usize;
    let expected = json_to_array1(&case["sklearn_predictions"]);

    let fitted = PlsRegression::new(k).fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();
    // NIPALS is deterministic; predictions should match to 1e-6.
    assert_array1_close(&preds, &expected, 1e-6, "pls1 predictions");
}
