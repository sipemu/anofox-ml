//! Golden tests for SGDOneClassSVM (#261) against scikit-learn 1.8.0
//! (`test_harness/generators/gen_sgd_one_class_svm.py`): coef_, offset_,
//! n_iter_, t_ and decision_function for an int random_state, across the
//! learning-rate schedules, averaging, sample weights and no intercept.

mod common;

use anofox_ml_core::FitUnsupervisedWeighted;
use anofox_ml_linear::{OneClassLearningRate, SgdOneClassSvm};
use common::json_to_array2;
use serde_json::Value;

fn close(a: f64, b: f64, what: &str) {
    assert!(
        (a - b).abs() <= 1e-9 * (1.0 + b.abs()),
        "{what}: {a} vs sklearn {b}"
    );
}

#[test]
fn sgd_one_class_svm_matches_sklearn() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden_data/sgd_one_class_svm.json");
    let golden: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let x = json_to_array2(&golden["X"]);
    let xt = json_to_array2(&golden["X_test"]);
    let w: Vec<f64> = golden["sample_weight"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    for c in golden["cases"].as_array().unwrap() {
        let p = &c["params"];
        let est = SgdOneClassSvm {
            nu: p["nu"].as_f64().unwrap_or(0.5),
            fit_intercept: p["fit_intercept"].as_bool().unwrap_or(true),
            max_iter: p["max_iter"].as_u64().map_or(1000, |v| v as usize),
            tol: match p.get("tol") {
                Some(Value::Null) => None,
                Some(t) => t.as_f64(),
                None => Some(1e-3),
            },
            shuffle: p["shuffle"].as_bool().unwrap_or(true),
            learning_rate: match p["learning_rate"].as_str().unwrap_or("optimal") {
                "constant" => OneClassLearningRate::Constant,
                "invscaling" => OneClassLearningRate::InvScaling,
                "adaptive" => OneClassLearningRate::Adaptive,
                _ => OneClassLearningRate::Optimal,
            },
            eta0: p["eta0"].as_f64().unwrap_or(0.01),
            power_t: p["power_t"].as_f64().unwrap_or(0.5),
            average: match &p["average"] {
                Value::Bool(b) => usize::from(*b),
                v => v.as_u64().unwrap_or(0) as usize,
            },
            seed: p["random_state"].as_u64().unwrap() as u32,
        };
        let sw = c["weighted"]
            .as_bool()
            .unwrap()
            .then(|| ndarray::ArrayView1::from(&w[..]));
        let fit = est.fit_unsupervised_weighted_view(x.view(), sw).unwrap();
        assert_eq!(fit.n_iter, c["n_iter"].as_u64().unwrap() as usize, "{p}");
        close(fit.t, c["t"].as_f64().unwrap(), "t_");
        close(
            fit.offset,
            c["offset"].as_f64().unwrap(),
            &format!("offset {p}"),
        );
        for (a, b) in fit.coef.iter().zip(c["coef"].as_array().unwrap()) {
            close(*a, b.as_f64().unwrap(), &format!("coef {p}"));
        }
        let dec = fit.decision_function(xt.view()).unwrap();
        for (a, b) in dec.iter().zip(c["decision_function"].as_array().unwrap()) {
            close(*a, b.as_f64().unwrap(), &format!("decision {p}"));
        }
    }
}
