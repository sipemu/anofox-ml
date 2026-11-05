//! Golden tests for the meta-estimator kinds (#113, batch 3) against
//! scikit-learn 1.8.0 (`test_harness/generators/gen_meta.py`; the data
//! lives with the other fixtures in `crates/anofox-ml/tests/golden_data`).

use anofox_ml_registry::{make, Out};
use ndarray::{Array1, Array2};
use serde_json::{json, Value};

fn golden() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../anofox-ml/tests/golden_data/meta.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

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

fn out2(o: Out) -> Array2<f64> {
    match o {
        Out::A2(a) => a,
        Out::A1(a) => a.insert_axis(ndarray::Axis(1)),
    }
}

fn out1(o: Out) -> Array1<f64> {
    match o {
        Out::A1(a) => a,
        Out::A2(_) => panic!("expected 1-D"),
    }
}

fn close(a: &Array2<f64>, b: &Array2<f64>, tol: f64, name: &str) {
    assert_eq!(a.dim(), b.dim(), "{name}");
    let d = (a - b).mapv(f64::abs).fold(0.0f64, |m, &v| m.max(v));
    assert!(d <= tol, "{name}: max diff {d}");
}

fn fit(kind: &str, params: Value, c: &Value) -> Box<dyn anofox_ml_registry::DynModel> {
    let (x, y) = (a2(&c["X"]), a1(&c["y"]));
    let mut m = make(kind, &params).unwrap();
    m.fit(x.view(), Some(y.view()), None).unwrap();
    m
}

fn base(c: &Value) -> Value {
    json!({ "estimator": c["kind"], "estimator_params": c["params"] })
}

#[test]
fn one_vs_rest_matches_sklearn() {
    for c in golden()["ovr"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let m = fit("OneVsRestClassifier", base(c), c);
        let x = a2(&c["X"]);
        assert_eq!(
            out1(m.predict(x.view()).unwrap()),
            a1(&c["predict"]),
            "{name}"
        );
        close(
            &out2(m.predict_proba(x.view()).unwrap()),
            &a2(&c["predict_proba"]),
            1e-6,
            name,
        );
    }
}

#[test]
fn one_vs_one_matches_sklearn() {
    for c in golden()["ovo"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let m = fit("OneVsOneClassifier", base(c), c);
        let x = a2(&c["X"]);
        assert_eq!(
            out1(m.predict(x.view()).unwrap()),
            a1(&c["predict"]),
            "{name}"
        );
        close(
            &out2(m.decision_function(x.view()).unwrap()),
            &a2(&c["decision_function"]),
            1e-6,
            name,
        );
    }
}

#[test]
fn fixed_threshold_matches_sklearn() {
    for c in golden()["fixed"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let mut p = base(c);
        p["threshold"] = c["threshold"].clone();
        p["pos_label"] = c["pos_label"].clone();
        let m = fit("FixedThresholdClassifier", p, c);
        let x = a2(&c["X"]);
        assert_eq!(
            out1(m.predict(x.view()).unwrap()),
            a1(&c["predict"]),
            "{name}"
        );
    }
}

#[test]
fn tuned_threshold_matches_sklearn() {
    for c in golden()["tuned"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let mut p = base(c);
        p["scoring"] = json!(match c["scoring"].as_str().unwrap() {
            "balanced_accuracy" => "BalancedAccuracy",
            "accuracy" => "Accuracy",
            "f1" => "F1",
            _ => "Recall",
        });
        let m = fit("TunedThresholdClassifierCV", p, c);
        let state: Value = serde_json::from_str(&m.fitted_json().unwrap()).unwrap();
        let (t, s) = (
            state["threshold"].as_f64().unwrap(),
            state["best_score"].as_f64().unwrap(),
        );
        let (et, es) = (
            c["best_threshold"].as_f64().unwrap(),
            c["best_score"].as_f64().unwrap(),
        );
        assert!((s - es).abs() < 1e-9, "{name}: best score {s} vs {es}");
        assert!((t - et).abs() < 1e-6, "{name}: threshold {t} vs {et}");
        let x = a2(&c["X"]);
        assert_eq!(
            out1(m.predict(x.view()).unwrap()),
            a1(&c["predict"]),
            "{name}"
        );
    }
}
