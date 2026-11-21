//! Golden tests for PatchExtractor (#261) against scikit-learn 1.8.0
//! (`test_harness/generators/gen_patch_extractor.py`).

mod common;

use anofox_ml_core::{FitUnsupervised, Transform};
use anofox_ml_preprocessing::{MaxPatches, PatchExtractor};
use common::{assert_array2_close, json_to_array2};
use serde_json::Value;

#[test]
fn patch_extractor_matches_sklearn() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden_data/patch_extractor.json");
    let golden: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    for c in golden["cases"].as_array().unwrap() {
        let p = &c["params"];
        let dims: Vec<usize> = c["image_shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let ps: Vec<usize> = p["patch_size"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let max_patches = match &p["max_patches"] {
            Value::Number(v) if v.is_u64() => Some(MaxPatches::Count(v.as_u64().unwrap() as usize)),
            Value::Number(v) => Some(MaxPatches::Fraction(v.as_f64().unwrap())),
            _ => None,
        };
        let est = PatchExtractor {
            image_shape: (dims[0], dims[1], dims[2]),
            patch_size: Some((ps[0], ps[1])),
            max_patches,
            seed: p["random_state"].as_u64().unwrap_or(0) as u32,
        };
        let x = json_to_array2(&c["X"]);
        let out = est.fit(&x).unwrap().transform(&x).unwrap();
        assert_array2_close(&out, &json_to_array2(&c["patches"]), 1e-15, &format!("{p}"));
    }
}
