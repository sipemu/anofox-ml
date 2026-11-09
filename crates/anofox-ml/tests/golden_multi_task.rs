//! MultiTaskLasso / MultiTaskElasticNet against scikit-learn 1.8.0
//! (`test_harness/generators/gen_multi_task.py`, solved to tol=1e-12).

mod common;

use anofox_ml_regression::MultiTaskElasticNet;
use common::{json_to_array1, json_to_array2, load_golden_file};

#[test]
fn multi_task_models_match_scikit_learn() {
    let data = load_golden_file("multi_task.json");
    let (x, y) = (json_to_array2(&data["X"]), json_to_array2(&data["Y"]));
    for case in data["cases"].as_array().unwrap() {
        let p = &case["params"];
        let l1_ratio = if case["kind"] == "MultiTaskLasso" { 1.0 } else { p["l1_ratio"].as_f64().unwrap() };
        let mut m = MultiTaskElasticNet::new(p["alpha"].as_f64().unwrap(), l1_ratio);
        m.fit_intercept = p["fit_intercept"].as_bool().unwrap_or(true);
        let f = m.fit(&x, &y).unwrap();
        for (o, e) in f.coef.iter().zip(&json_to_array2(&case["coef"])) {
            assert!((o - e).abs() < 1e-8, "{p}: coef {o} vs {e}");
        }
        for (o, e) in f.intercept.iter().zip(&json_to_array1(&case["intercept"])) {
            assert!((o - e).abs() < 1e-8, "{p}: intercept {o} vs {e}");
        }
        let pred = f.predict(&x.slice(ndarray::s![..3, ..])).unwrap();
        for (o, e) in pred.iter().zip(&json_to_array2(&case["predict"])) {
            assert!((o - e).abs() < 1e-8, "{p}: predict {o} vs {e}");
        }
    }
}
