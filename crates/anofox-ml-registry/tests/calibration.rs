//! `CalibratedClassifierCV` through the registry (#125), against
//! scikit-learn 1.8.0 (`test_harness/generators/gen_calibration.py`; the
//! data lives with the other fixtures in `crates/anofox-ml/tests/golden_data`).

use anofox_ml_registry::{make, Out};
use ndarray::{Array1, Array2};
use serde_json::{json, Value};

fn golden() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../anofox-ml/tests/golden_data/calibration.json");
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

fn out1(o: Out) -> Array1<f64> {
    match o {
        Out::A1(a) => a,
        Out::A2(a) => panic!("expected a 1-D output, got {:?}", a.dim()),
    }
}

fn out2(o: Out) -> Array2<f64> {
    match o {
        Out::A2(a) => a,
        Out::A1(a) => panic!("expected a 2-D output, got {} values", a.len()),
    }
}

#[test]
fn calibrated_probabilities_match_scikit_learn() {
    // Isotonic is exact; sigmoid differs by scipy's L-BFGS-B stopping
    // (gtol 1e-6) from our Newton solution of the same problem: within 1e-7.
    let mut failures = Vec::new();
    for case in golden()["cases"].as_array().unwrap() {
        let name = format!(
            "{} / {} classes / {} / ensemble={}",
            case["estimator"], case["classes"], case["method"], case["ensemble"]
        );
        let mut m = make(
            "CalibratedClassifierCV",
            &json!({
                "estimator": case["estimator"],
                "method": case["method"],
                "ensemble": case["ensemble"],
                "cv": 5,
            }),
        )
        .unwrap();
        let (x, y) = (a2(&case["X"]), a1(&case["y"]));
        m.fit(x.view(), Some(y.view()), None).unwrap();
        let ours = out2(m.predict_proba(a2(&case["Q"]).view()).unwrap());
        let expected = a2(&case["proba"]);
        let tol = if case["method"] == "isotonic" {
            1e-12
        } else {
            1e-7
        };
        let worst = ours
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        if worst > tol {
            failures.push(format!("{name}: worst {worst:e}"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn calibration_checks_its_spec_and_round_trips() {
    let case = &golden()["cases"][4]; // GaussianNB, 3 classes
    let (x, y) = (a2(&case["X"]), a1(&case["y"]));
    let spec = |extra: Value| {
        let mut p = json!({"estimator": "GaussianNB"});
        p.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        make("CalibratedClassifierCV", &p)
    };
    let e = spec(json!({"cv": 1}))
        .unwrap()
        .fit(x.view(), Some(y.view()), None)
        .unwrap_err()
        .to_string();
    assert!(e.contains("cv must be at least 2"), "{e}");
    assert!(spec(json!({"method": "platt"})).is_err());
    // A regressor has neither decision_function nor predict_proba.
    let mut m = make(
        "CalibratedClassifierCV",
        &json!({"estimator": "OlsRegressor"}),
    )
    .unwrap();
    let e = m
        .fit(x.view(), Some(y.view()), None)
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("neither decision_function nor predict_proba"),
        "{e}"
    );

    for method in ["sigmoid", "isotonic"] {
        let mut m = spec(json!({"method": method, "ensemble": false})).unwrap();
        m.fit(x.view(), Some(y.view()), None).unwrap();
        let back =
            anofox_ml_registry::load("CalibratedClassifierCV", &m.to_bytes().unwrap()).unwrap();
        let q = a2(&case["Q"]);
        assert_eq!(
            out2(back.predict_proba(q.view()).unwrap()),
            out2(m.predict_proba(q.view()).unwrap())
        );
        assert_eq!(
            out1(back.predict(q.view()).unwrap()),
            out1(m.predict(q.view()).unwrap())
        );
    }
}
