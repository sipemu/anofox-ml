//! `StackingRegressor` / `StackingClassifier` through the registry (#125),
//! against scikit-learn 1.8.0 on the same inline data (cv = 5 without
//! shuffling).

use anofox_ml_registry::{make, DynModel, Out};
use ndarray::{Array1, Array2};
use serde_json::{json, Value};

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

fn data() -> (Array2<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    let x = Array2::from_shape_fn((30, 3), |(i, j)| {
        ((i * 7 + j * 3 + i * j) % 13) as f64 / 6.0 - 1.0
    });
    let y = x
        .rows()
        .into_iter()
        .map(|r| r[0] - 2.0 * r[1] + r[2] * r[2])
        .collect();
    let y3 = x
        .rows()
        .into_iter()
        .map(|r| {
            if r[0] + r[1] < -0.5 {
                0.0
            } else if r[0] - r[2] < 0.3 {
                1.0
            } else {
                2.0
            }
        })
        .collect();
    let y2 = x
        .rows()
        .into_iter()
        .map(|r| f64::from(r[0] + 0.5 * r[2] > 0.0))
        .collect();
    (x, y, y3, y2)
}

fn grid() -> Array2<f64> {
    let mut rows = Vec::new();
    for a in [-0.9, -0.2, 0.4, 0.95] {
        for b in [-0.6, 0.3] {
            for c in [-0.4, 0.7] {
                rows.extend([a, b, c]);
            }
        }
    }
    Array2::from_shape_vec((16, 3), rows).unwrap()
}

fn close(ours: &[f64], expected: &[f64], what: &str) {
    assert_eq!(ours.len(), expected.len(), "{what}");
    for (o, e) in ours.iter().zip(expected) {
        assert!((o - e).abs() < 1e-10, "{what}: {o} vs {e}");
    }
}

fn regressor(passthrough: bool) -> Box<dyn DynModel> {
    make(
        "StackingRegressor",
        &json!({
            "estimators": [
                {"estimator": "OlsRegressor"},
                {"estimator": "KnnRegressor", "estimator_params": {"n_neighbors": 3}}
            ],
            "final_estimator": {"estimator": "OlsRegressor"},
            "passthrough": passthrough,
        }),
    )
    .unwrap()
}

fn classifier(extra: Value) -> Box<dyn DynModel> {
    let mut spec = json!({
        "estimators": [
            {"estimator": "GaussianNB"},
            {"estimator": "KnnClassifier", "estimator_params": {"n_neighbors": 3}}
        ],
        "final_estimator": {"estimator": "GaussianNB"},
    });
    spec.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    make("StackingClassifier", &spec).unwrap()
}

#[test]
fn stacking_regressor_matches_scikit_learn() {
    // StackingRegressor([LinearRegression(), KNN(3)], final_estimator=LinearRegression(),
    // passthrough=...).
    let (x, y, _, _) = data();
    let q = grid();
    let q4 = q.slice(ndarray::s![..4, ..]);
    for (pt, expected) in [
        (
            false,
            [
                0.6242276837542806,
                0.6537218287883435,
                -1.2626009994366982,
                -1.182579471164725,
            ],
        ),
        (
            true,
            [
                0.5747949592910733,
                0.7890730101239583,
                -1.1958565143580437,
                -1.0605306138856747,
            ],
        ),
    ] {
        let mut m = regressor(pt);
        m.fit(x.view(), Some(y.view()), None).unwrap();
        close(
            out1(m.predict(q4).unwrap()).as_slice().unwrap(),
            &expected,
            &format!("passthrough={pt}"),
        );
    }
}

#[test]
fn stacking_classifier_matches_scikit_learn() {
    // StackingClassifier([GaussianNB(), KNN(3)], final_estimator=GaussianNB()):
    // stack_method auto = predict_proba, the first column dropped for two classes.
    let (x, _, y3, y2) = data();
    let q = grid();
    let q3 = q.slice(ndarray::s![..3, ..]);
    let cases = [
        (
            &y2,
            vec![
                0.9999966421854511,
                3.35781454886939e-06,
                0.9999966369783544,
                3.363021645639241e-06,
                0.999996643960402,
                3.3560395979712674e-06,
            ],
            vec![
                0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.,
            ],
        ),
        (
            &y3,
            vec![
                0.8983197180544823,
                0.10168028194551759,
                1.9967330178703837e-264,
                0.8880035558762229,
                0.1119964441237771,
                1.0084083704825943e-250,
                2.2989223987576734e-06,
                0.9999977010776011,
                3.817456214112387e-26,
            ],
            vec![
                0., 0., 1., 1., 1., 0., 1., 1., 0., 1., 2., 2., 2., 0., 2., 2.,
            ],
        ),
    ];
    for (y, proba, labels) in cases {
        let mut m = classifier(json!({}));
        m.fit(x.view(), Some(y.view()), None).unwrap();
        close(
            out2(m.predict_proba(q3).unwrap()).as_slice().unwrap(),
            &proba,
            "proba",
        );
        assert_eq!(out1(m.predict(q.view()).unwrap()).to_vec(), labels);
    }
}

#[test]
fn stacking_checks_its_spec_and_round_trips() {
    let (x, y, y3, _) = data();
    let mut empty = make("StackingRegressor", &json!({})).unwrap();
    let e = empty
        .fit(x.view(), Some(y.view()), None)
        .unwrap_err()
        .to_string();
    assert!(e.contains("at least one estimator"), "{e}");

    let mut m = regressor(false);
    m.set_params(&json!({"cv": 1})).unwrap();
    let e = m
        .fit(x.view(), Some(y.view()), None)
        .unwrap_err()
        .to_string();
    assert!(e.contains("cv must be at least 2"), "{e}");

    // kNN has no decision_function.
    let mut c = classifier(json!({"stack_method": "decision_function"}));
    let e = c
        .fit(x.view(), Some(y3.view()), None)
        .unwrap_err()
        .to_string();
    assert!(e.contains("has no decision_function"), "{e}");

    let mut c = classifier(json!({"passthrough": true}));
    c.fit(x.view(), Some(y3.view()), None).unwrap();
    let back = anofox_ml_registry::load("StackingClassifier", &c.to_bytes().unwrap()).unwrap();
    let q = grid();
    assert_eq!(
        out2(back.predict_proba(q.view()).unwrap()),
        out2(c.predict_proba(q.view()).unwrap())
    );
    assert_eq!(
        out1(back.predict(q.view()).unwrap()),
        out1(c.predict(q.view()).unwrap())
    );
}
