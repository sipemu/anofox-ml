//! Golden test for KNeighborsTransformer (#113) against scikit-learn 1.8.0
//! (`gen_kneighbors_transformer.py`).

mod common;

use anofox_ml_core::{FitUnsupervised, Transform};
use anofox_ml_neighbors::{GraphMode, KNeighborsTransformer};
use common::{assert_array2_close, json_to_array2};
use serde_json::Value;

#[test]
fn kneighbors_transformer_matches_sklearn() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden_data/kneighbors_transformer.json");
    let cases: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    for c in cases.as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let mode = if c["mode"] == "distance" {
            GraphMode::Distance
        } else {
            GraphMode::Connectivity
        };
        let x = json_to_array2(&c["X"]);
        let f = KNeighborsTransformer {
            mode,
            n_neighbors: c["n_neighbors"].as_u64().unwrap() as usize,
        }
        .fit(&x)
        .unwrap();
        assert_array2_close(
            &f.transform(&x).unwrap(),
            &json_to_array2(&c["train"]),
            1e-12,
            name,
        );
        assert_array2_close(
            &f.transform(&json_to_array2(&c["X_test"])).unwrap(),
            &json_to_array2(&c["test"]),
            1e-12,
            name,
        );
    }
}
