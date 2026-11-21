//! Golden tests for SpectralCoclustering (#261) against scikit-learn 1.8.0
//! (`test_harness/generators/gen_spectral_coclustering.py`): identical
//! row and column labels.

mod common;

use anofox_ml_cluster::SpectralCoclustering;
use anofox_ml_core::FitUnsupervised;
use common::json_to_array2;
use serde_json::Value;

fn labels(v: &Value) -> Vec<usize> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect()
}

#[test]
fn spectral_coclustering_matches_sklearn() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden_data/spectral_coclustering.json");
    let golden: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    for c in golden["cases"].as_array().unwrap() {
        let p = &c["params"];
        let est = SpectralCoclustering {
            n_clusters: p["n_clusters"].as_u64().unwrap() as usize,
            n_svd_vecs: p["n_svd_vecs"].as_u64().map(|v| v as usize),
            n_init: p["n_init"].as_u64().map_or(10, |v| v as usize),
            seed: p["random_state"].as_u64().unwrap() as u32,
        };
        let fit = est.fit(&json_to_array2(&c["X"])).unwrap();
        assert_eq!(fit.row_labels, labels(&c["row_labels"]), "{p}");
        assert_eq!(fit.column_labels, labels(&c["column_labels"]), "{p}");
    }
}
