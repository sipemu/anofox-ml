//! Golden tests for SpectralBiclustering (#261) against scikit-learn 1.8.0
//! (`test_harness/generators/gen_spectral_biclustering.py`): identical
//! row and column labels for the bistochastic, scale and log methods.

mod common;

use anofox_ml_cluster::{BiclusterMethod, SpectralBiclustering};
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
fn spectral_biclustering_matches_sklearn() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden_data/spectral_biclustering.json");
    let golden: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    for c in golden["cases"].as_array().unwrap() {
        let p = &c["params"];
        let (n_row_clusters, n_column_clusters) = match &p["n_clusters"] {
            Value::Array(k) => (k[0].as_u64().unwrap(), k[1].as_u64().unwrap()),
            k => (k.as_u64().unwrap(), k.as_u64().unwrap()),
        };
        let est = SpectralBiclustering {
            n_row_clusters: n_row_clusters as usize,
            n_column_clusters: n_column_clusters as usize,
            method: match p["method"].as_str().unwrap_or("bistochastic") {
                "scale" => BiclusterMethod::Scale,
                "log" => BiclusterMethod::Log,
                _ => BiclusterMethod::Bistochastic,
            },
            n_components: p["n_components"].as_u64().map_or(6, |v| v as usize),
            n_best: p["n_best"].as_u64().map_or(3, |v| v as usize),
            n_svd_vecs: p["n_svd_vecs"].as_u64().map(|v| v as usize),
            n_init: p["n_init"].as_u64().map_or(10, |v| v as usize),
            seed: p["random_state"].as_u64().unwrap() as u32,
        };
        let fit = est.fit(&json_to_array2(&c["X"])).unwrap();
        assert_eq!(fit.row_labels, labels(&c["row_labels"]), "{p}");
        assert_eq!(fit.column_labels, labels(&c["column_labels"]), "{p}");
    }
}
