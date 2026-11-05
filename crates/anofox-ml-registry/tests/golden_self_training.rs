//! Golden test for the SelfTrainingClassifier kind (#113, batch 8) against
//! scikit-learn 1.8.0 (`gen_self_training.py`).

use anofox_ml_registry::{make, Out};
use ndarray::{Array1, Array2};
use serde_json::{json, Value};

fn a1(v: &Value) -> Array1<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn a2(v: &Value) -> Array2<f64> {
    let rows: Vec<Array1<f64>> = v.as_array().unwrap().iter().map(a1).collect();
    let views: Vec<_> = rows.iter().map(|r| r.view()).collect();
    ndarray::stack(ndarray::Axis(0), &views).unwrap()
}

#[test]
fn self_training_matches_sklearn() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../anofox-ml/tests/golden_data/self_training.json");
    let cases: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    for c in cases.as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let p = &c["params"];
        let criterion = if p["criterion"] == "k_best" {
            json!({ "KBest": p["k_best"] })
        } else {
            json!({ "Threshold": p["threshold"] })
        };
        let max_iter = if p.get("max_iter").is_some() {
            p["max_iter"].clone()
        } else {
            json!(10)
        };
        let params =
            json!({ "estimator": "GaussianNB", "criterion": criterion, "max_iter": max_iter });
        let (x, y) = (a2(&c["X"]), a1(&c["y"]));
        let mut m = make("SelfTrainingClassifier", &params).unwrap();
        m.fit(x.view(), Some(y.view()), None).unwrap();
        let state: Value = serde_json::from_str(&m.fitted_json().unwrap()).unwrap();
        assert_eq!(state["n_iter"], c["n_iter"], "{name}");
        assert_eq!(state["termination"], c["termination"], "{name}");
        let tr: Value = serde_json::to_value(a1(&state["transduction"]["data"]).to_vec()).unwrap();
        assert_eq!(
            tr,
            serde_json::to_value(a1(&c["transduction"]).to_vec()).unwrap(),
            "{name}: transduction"
        );
        assert_eq!(
            a1(&state["labeled_iter"]["data"]),
            a1(&c["labeled_iter"]),
            "{name}: labeled_iter"
        );
        match m.predict(x.view()).unwrap() {
            Out::A1(p) => assert_eq!(p, a1(&c["predict"]), "{name}"),
            Out::A2(_) => panic!("1-D predictions expected"),
        }
        if let Out::A2(p) = m.predict_proba(x.view()).unwrap() {
            let d = (&p - &a2(&c["predict_proba"]))
                .mapv(f64::abs)
                .fold(0.0f64, |a, &b| a.max(b));
            assert!(d < 1e-10, "{name}: proba diff {d}");
        }
    }
}
