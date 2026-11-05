//! Golden test for FeatureAgglomeration (#113, batch 8) against
//! scikit-learn 1.8.0 (`gen_feature_agglomeration.py`).

mod common;

use anofox_ml_cluster::{FeatureAgglomeration, Linkage};
use anofox_ml_core::{FitUnsupervised, Transform};
use common::{assert_array2_close, json_to_array2};
use serde_json::Value;

#[test]
fn feature_agglomeration_matches_sklearn() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden_data/feature_agglomeration.json");
    let cases: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    for c in cases.as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let linkage = match c["linkage"].as_str().unwrap() {
            "ward" => Linkage::Ward,
            "complete" => Linkage::Complete,
            "average" => Linkage::Average,
            _ => Linkage::Single,
        };
        let x = json_to_array2(&c["X"]);
        let f = FeatureAgglomeration {
            n_clusters: c["n_clusters"].as_u64().unwrap() as usize,
            linkage,
        }
        .fit(&x)
        .unwrap();
        let labels: Vec<u64> = c["labels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        assert_eq!(
            f.labels().iter().map(|&l| l as u64).collect::<Vec<_>>(),
            labels,
            "{name}"
        );
        let t = f.transform(&x).unwrap();
        assert_array2_close(&t, &json_to_array2(&c["transform"]), 1e-12, name);
        assert_array2_close(
            &f.inverse_transform(t.view()).unwrap(),
            &json_to_array2(&c["inverse"]),
            1e-12,
            name,
        );
    }
}
