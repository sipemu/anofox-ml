//! Golden test for OrthogonalMatchingPursuit against sklearn 1.8.0.

mod common;

use common::{assert_array1_close, json_to_array1, json_to_array2, load_golden_data};
use rustml::prelude::*;
use rustml_regression::OrthogonalMatchingPursuit;

#[test]
fn test_omp_matches_sklearn() {
    let cases = load_golden_data("omp.json");
    let case = &cases[0];

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let n_nonzero = case["n_nonzero"].as_u64().unwrap() as usize;
    let expected = json_to_array1(&case["sklearn_predictions"]);
    let expected_active: Vec<usize> = case["expected_active"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();

    let fitted = OrthogonalMatchingPursuit::new()
        .with_n_nonzero_coefs(n_nonzero)
        .fit(&x, &y)
        .unwrap();

    let mut active = fitted.active_set.clone();
    active.sort();
    assert_eq!(active, expected_active);

    let preds = fitted.predict(&x).unwrap();
    assert_array1_close(&preds, &expected, 1e-6, "omp predictions");
}
