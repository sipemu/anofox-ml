//! SparsePCA (`method="lars"`) against scikit-learn 1.8.0
//! (`gen_sparse_pca.py`).

mod common;

use anofox_ml_core::{FitUnsupervised, Transform};
use anofox_ml_preprocessing::SparsePca;
use common::{json_to_array2, load_golden_file};

#[test]
fn sparse_pca_matches_scikit_learn() {
    let data = load_golden_file("sparse_pca.json");
    let x = json_to_array2(&data["X"]);
    for case in data["cases"].as_array().unwrap() {
        let p = &case["params"];
        let m = SparsePca {
            n_components: p["n_components"].as_u64().map(|v| v as usize),
            alpha: p["alpha"].as_f64().unwrap_or(1.0),
            ridge_alpha: p["ridge_alpha"].as_f64().unwrap_or(0.01),
            ..Default::default()
        };
        let f = m.fit(&x).unwrap();
        assert_eq!(
            f.n_iter as u64,
            case["n_iter"].as_u64().unwrap(),
            "{p}: n_iter"
        );
        for (o, e) in f.error.iter().zip(case["error"].as_array().unwrap()) {
            let e = e.as_f64().unwrap();
            assert!((o - e).abs() < 1e-8 * e, "{p}: cost {o} vs {e}");
        }
        for (o, e) in f
            .components
            .iter()
            .zip(&json_to_array2(&case["components"]))
        {
            assert!((o - e).abs() < 1e-7, "{p}: component {o} vs {e}");
        }
        let t = f
            .transform(&x.slice(ndarray::s![..5, ..]).to_owned())
            .unwrap();
        for (o, e) in t.iter().zip(&json_to_array2(&case["transform"])) {
            assert!((o - e).abs() < 1e-7, "{p}: transform {o} vs {e}");
        }
    }
}
