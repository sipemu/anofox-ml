//! `DynModel::predict_log_proba` (#125): the estimators' own log-probabilities
//! through the registry.

use anofox_ml_registry::{make, Out};
use ndarray::{array, Array1, Array2};
use serde_json::json;

fn out2(o: Out) -> Array2<f64> {
    match o {
        Out::A2(a) => a,
        Out::A1(a) => panic!("expected a 2-D output, got {} values", a.len()),
    }
}

/// Two tight clusters 100 apart: far from its own cluster, a row's
/// probability of the other class underflows.
fn separated() -> (Array2<f64>, Array1<f64>) {
    let x = array![
        [0.0, 0.1],
        [0.1, 0.0],
        [-0.1, 0.05],
        [0.05, -0.1],
        [100.0, 100.1],
        [100.1, 100.0],
        [99.9, 100.05],
        [100.05, 99.9]
    ];
    let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    (x, y)
}

#[test]
fn gaussian_nb_log_proba_is_native() {
    let (x, y) = separated();
    let mut m = make("GaussianNB", &json!({})).unwrap();
    m.fit(x.view(), Some(y.view()), None).unwrap();
    let lp = out2(m.predict_log_proba(x.view()).unwrap());
    // ln(predict_proba) stops at ln(1e-300) ≈ -690.8; the joint
    // log-likelihoods do not.
    assert!(
        lp[[0, 1]] < -1000.0,
        "log P(class 1 | row 0) = {}",
        lp[[0, 1]]
    );
    assert!(lp[[0, 0]].abs() < 1e-12);
}

/// Small count data with three classes (non-negative integers, as the
/// discrete naive Bayes kinds need).
fn counts() -> (Array2<f64>, Array1<f64>) {
    let x = array![
        [3.0, 0.0, 1.0],
        [4.0, 1.0, 0.0],
        [2.0, 0.0, 2.0],
        [0.0, 3.0, 1.0],
        [1.0, 4.0, 0.0],
        [0.0, 2.0, 2.0],
        [1.0, 0.0, 4.0],
        [0.0, 1.0, 3.0],
        [2.0, 1.0, 3.0]
    ];
    let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];
    (x, y)
}

#[test]
fn log_proba_matches_proba_for_every_kind_that_has_it() {
    let (x, y) = counts();
    for (kind, params) in [
        ("GaussianNB", json!({})),
        ("MultinomialNB", json!({})),
        ("BernoulliNB", json!({})),
        ("ComplementNB", json!({})),
        ("CategoricalNB", json!({})),
        ("LinearDiscriminantAnalysis", json!({})),
        ("QuadraticDiscriminantAnalysis", json!({"reg": 0.5})),
        ("LogisticRegressor", json!({})),
        ("Svc", json!({"probability": true})),
        ("GaussianMixture", json!({"n_components": 2})),
    ] {
        let mut m = make(kind, &params).unwrap();
        let y = (kind != "GaussianMixture").then_some(y.view());
        m.fit(x.view(), y, None).unwrap();
        assert!(m.capabilities().contains(&"predict_log_proba"), "{kind}");
        let lp = out2(m.predict_log_proba(x.view()).unwrap());
        let p = out2(m.predict_proba(x.view()).unwrap());
        assert_eq!(lp.dim(), p.dim(), "{kind}");
        for (l, p) in lp.iter().zip(&p) {
            assert!((l.exp() - p).abs() < 1e-12, "{kind}: exp({l}) vs {p}");
        }
    }
}

#[test]
fn classifiers_without_it_refuse_by_name() {
    let (x, y) = counts();
    let mut m = make("RandomForestClassifier", &json!({"n_estimators": 3})).unwrap();
    m.fit(x.view(), Some(y.view()), None).unwrap();
    assert!(!m.capabilities().contains(&"predict_log_proba"));
    let err = m.predict_log_proba(x.view()).unwrap_err().to_string();
    assert!(
        err.contains("RandomForestClassifier does not support predict_log_proba"),
        "{err}"
    );
}

#[test]
fn log_proba_survives_a_round_trip() {
    let (x, y) = separated();
    let mut m = make("GaussianNB", &json!({})).unwrap();
    m.fit(x.view(), Some(y.view()), None).unwrap();
    let restored = anofox_ml_registry::load("GaussianNB", &m.to_bytes().unwrap()).unwrap();
    assert_eq!(
        out2(restored.predict_log_proba(x.view()).unwrap()),
        out2(m.predict_log_proba(x.view()).unwrap())
    );
}
