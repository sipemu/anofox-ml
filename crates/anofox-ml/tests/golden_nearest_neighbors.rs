//! Golden tests for NearestNeighbors and RadiusNeighborsTransformer (#260)
//! against scikit-learn 1.8.0
//! (`test_harness/generators/gen_nearest_neighbors.py`).

mod common;

use anofox_ml_core::{FitUnsupervised, Transform};
use anofox_ml_neighbors::{
    DistanceMetric, GraphMode, NearestNeighbors, NeighborAlgorithm, RadiusNeighborsTransformer,
};
use common::{assert_array2_close, json_to_array2};
use serde_json::Value;

fn golden() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden_data/nearest_neighbors.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn metric(p: &Value) -> DistanceMetric {
    if p["metric"] == "manhattan" {
        DistanceMetric::Manhattan
    } else {
        DistanceMetric::Euclidean
    }
}

#[test]
fn kneighbors_matches_sklearn() {
    for c in golden()["knn"].as_array().unwrap() {
        let p = &c["params"];
        let est = NearestNeighbors {
            n_neighbors: p["n_neighbors"].as_u64().unwrap() as usize,
            metric: metric(p),
            algorithm: if p["algorithm"] == "brute" {
                NeighborAlgorithm::BruteForce
            } else {
                NeighborAlgorithm::Auto
            },
        };
        let fitted = est.fit(&json_to_array2(&c["X"])).unwrap();
        let (dist, ind) = fitted
            .kneighbors(json_to_array2(&c["X_query"]).view(), None)
            .unwrap();
        let ctx = format!("{p}");
        assert_array2_close(&dist, &json_to_array2(&c["distances"]), 1e-12, &ctx);
        assert_eq!(
            ind.mapv(|v| v as f64),
            json_to_array2(&c["indices"]),
            "{ctx}"
        );
    }
}

#[test]
fn radius_neighbors_transformer_matches_sklearn() {
    for c in golden()["radius"].as_array().unwrap() {
        let p = &c["params"];
        let est = RadiusNeighborsTransformer {
            radius: p["radius"].as_f64().unwrap(),
            mode: if p["mode"] == "connectivity" {
                GraphMode::Connectivity
            } else {
                GraphMode::Distance
            },
            metric: metric(p),
            algorithm: NeighborAlgorithm::Auto,
        };
        let out = est
            .fit(&json_to_array2(&c["X"]))
            .unwrap()
            .transform(&json_to_array2(&c["X_query"]))
            .unwrap();
        assert_array2_close(
            &out,
            &json_to_array2(&c["transform"]),
            1e-12,
            &format!("{p}"),
        );
    }
}
