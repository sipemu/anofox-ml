//! NeighborhoodComponentsAnalysis against scikit-learn 1.8.0
//! (`test_harness/generators/gen_nca.py`). The L-BFGS iterates are not
//! SciPy's, and NCA's map is identified only up to `A → QA` (`Q`
//! orthogonal; LDA's scaling signs are LAPACK's choice) and grows without
//! bound once the classes separate, so the objective reached is compared.

mod common;

use anofox_ml_core::Fit;
use anofox_ml_neighbors::NeighborhoodComponentsAnalysis;
use common::{json_to_array1, json_to_array2, load_golden_file};

#[test]
fn nca_matches_scikit_learn() {
    let data = load_golden_file("nca.json");
    let (x, y) = (json_to_array2(&data["X"]), json_to_array1(&data["y"]));
    for case in data["cases"].as_array().unwrap() {
        let p = &case["params"];
        let m = NeighborhoodComponentsAnalysis {
            n_components: p["n_components"].as_u64().map(|c| c as usize),
            max_iter: p["max_iter"].as_u64().unwrap_or(50) as usize,
            tol: p["tol"].as_f64().unwrap_or(1e-5),
            ..Default::default()
        };
        let f = m.fit(&x, &y).unwrap();
        let ours = f.expected_correct(x.view(), y.view()).unwrap();
        let theirs = case["loss"].as_f64().unwrap();
        eprintln!("{p}: objective {ours:.6} vs {theirs:.6} (of {})", x.nrows());
        // Runs that stop on the plateau the objective reaches as the map
        // grows agree to well within one row; runs cut off by max_iter
        // stop at different points of their paths.
        let cut_off = case["n_iter"].as_u64().unwrap() as usize > m.max_iter;
        let tol = if cut_off { 0.5 } else { 0.05 };
        assert!(
            (ours - theirs).abs() < tol,
            "{p}: objective {ours} vs {theirs}"
        );
    }
}
