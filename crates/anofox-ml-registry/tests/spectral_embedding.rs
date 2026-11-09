//! SpectralEmbedding through the registry (#113): the training embedding
//! via `embedding`, against scikit-learn 1.8.0
//! (`test_harness/generators/gen_spectral_embedding.py`).

use anofox_ml_registry::{load, make, Out};
use ndarray::{Array1, Array2, Axis};
use serde_json::{json, Value};

fn a2(v: &Value) -> Array2<f64> {
    let rows: Vec<Array1<f64>> = v
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect()
        })
        .collect();
    let views: Vec<_> = rows.iter().map(|r| r.view()).collect();
    ndarray::stack(Axis(0), &views).unwrap()
}

#[test]
fn spectral_embedding_kind_matches_scikit_learn() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../anofox-ml/tests/golden_data/spectral_embedding.json");
    let data: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let x = a2(&data["X"]);
    // The default case (k-NN affinity, n_neighbors = n / 10) and RBF.
    for (case, params) in [(0, json!({})), (2, json!({"affinity": "Rbf"}))] {
        let mut m = make("SpectralEmbedding", &params).unwrap();
        m.fit(x.view(), None, None).unwrap();
        let m = load("SpectralEmbedding", &m.to_bytes().unwrap()).unwrap();
        let Out::A2(e) = m.embedding().unwrap() else {
            panic!()
        };
        for (o, v) in e.iter().zip(&a2(&data["cases"][case]["embedding"])) {
            assert!((o - v).abs() < 1e-8, "{params}: {o} vs {v}");
        }
    }
}
