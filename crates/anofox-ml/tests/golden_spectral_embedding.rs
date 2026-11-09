//! SpectralEmbedding against scikit-learn 1.8.0
//! (`test_harness/generators/gen_spectral_embedding.py`).

mod common;

use anofox_ml::core::FitUnsupervised;
use anofox_ml_manifold::{SpectralAffinity, SpectralEmbedding};
use common::{json_to_array2, load_golden_file};

#[test]
fn spectral_embedding_matches_scikit_learn() {
    let data = load_golden_file("spectral_embedding.json");
    let x = json_to_array2(&data["X"]);
    for case in data["cases"].as_array().unwrap() {
        let p = &case["params"];
        let mut m = SpectralEmbedding::new(p["n_components"].as_u64().unwrap_or(2) as usize);
        if p["affinity"] == "rbf" {
            m.affinity = SpectralAffinity::Rbf;
            m.gamma = p["gamma"].as_f64();
        }
        m.n_neighbors = p["n_neighbors"].as_u64().map(|k| k as usize);
        let fitted = FitUnsupervised::fit(&m, &x).unwrap();
        let expected = json_to_array2(&case["embedding"]);
        assert_eq!(fitted.embedding.dim(), expected.dim(), "{p}");
        for (o, e) in fitted.embedding.iter().zip(&expected) {
            assert!((o - e).abs() < 1e-8, "{p}: {o} vs {e}");
        }
    }
}
