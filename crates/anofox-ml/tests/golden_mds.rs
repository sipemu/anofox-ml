//! Golden tests for MDS (SMACOF, #260) against scikit-learn 1.8.0
//! (`test_harness/generators/gen_mds.py`).

mod common;

use anofox_ml_core::FitUnsupervised;
use anofox_ml_manifold::{Mds, MdsInit};
use common::{assert_array2_close, assert_close, json_to_array2};
use serde_json::Value;

#[test]
fn mds_matches_sklearn() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden_data/mds.json");
    let golden: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    for c in golden["cases"].as_array().unwrap() {
        let p = &c["params"];
        let est = Mds {
            n_components: p["n_components"].as_u64().map_or(2, |v| v as usize),
            metric_mds: p["metric_mds"].as_bool().unwrap_or(true),
            n_init: p["n_init"].as_u64().map_or(4, |v| v as usize),
            init: if p["init"] == "classical_mds" {
                MdsInit::ClassicalMds
            } else {
                MdsInit::Random
            },
            max_iter: p["max_iter"].as_u64().map_or(300, |v| v as usize),
            precomputed: p["metric"] == "precomputed",
            seed: p["random_state"].as_u64().unwrap_or(0) as u32,
            ..Default::default()
        };
        let fit = est.fit(&json_to_array2(&c["X"])).unwrap();
        let ctx = format!("{p}");
        assert_array2_close(
            &fit.embedding,
            &json_to_array2(&c["embedding"]),
            1e-10,
            &ctx,
        );
        assert_close(fit.stress, c["stress"].as_f64().unwrap(), 1e-10, &ctx);
        assert_eq!(fit.n_iter as u64, c["n_iter"].as_u64().unwrap(), "{ctx}");
    }
}
