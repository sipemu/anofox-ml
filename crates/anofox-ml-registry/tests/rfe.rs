//! `Rfe` through the registry (#125): recursive feature elimination by the
//! base model's `coef` (else its `feature_importances`), against
//! scikit-learn 1.8.0 on the same inline data.

use anofox_ml_registry::{make, Out};
use ndarray::{Array1, Array2};
use serde_json::{json, Value};

fn data() -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_fn((30, 5), |(i, j)| {
        ((i * 7 + j * 3 + i * j) % 13) as f64 / 6.0 - 1.0 + 0.02 * ((i * (j + 2)) % 5) as f64
    });
    let y = Array1::from_shape_fn(30, |i| {
        let r = x.row(i);
        2.0 * r[0] - r[2] + 0.5 * r[4] + 0.1 * r[1] + (((i * 3) % 7) as f64 - 3.0) / 20.0
    });
    (x, y)
}

#[test]
fn rfe_matches_scikit_learn() {
    // RFE(est, n_features_to_select=k, step=s): ranking_ and predict(X[:3]).
    let (x, y) = data();
    let cases = [
        (
            json!({"estimator": "OlsRegressor", "n_features_to_select": 2}),
            [1, 3, 1, 4, 2],
            [-1.8405868158015481, 1.126136620719057, -2.401311568333821],
        ),
        (
            json!({"estimator": "RidgeRegressor", "n_features_to_select": 3, "step": 2}),
            [1, 2, 1, 2, 1],
            [-1.4440222362799369, 1.2148246122579898, -2.0750648993613767],
        ),
        // Default: half the features (2 of 5).
        (
            json!({"estimator": "OlsRegressor"}),
            [1, 3, 1, 4, 2],
            [-1.8405868158015481, 1.126136620719057, -2.401311568333821],
        ),
    ];
    for (spec, ranking, pred) in cases {
        let mut m = make("Rfe", &spec).unwrap();
        m.fit(x.view(), Some(y.view()), None).unwrap();
        let fitted: Value = serde_json::from_str(&m.fitted_json().unwrap()).unwrap();
        let ours: Vec<u64> = fitted["ranking"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        assert_eq!(ours, ranking, "{spec}");
        let Out::A1(p) = m.predict(x.slice(ndarray::s![..3, ..])).unwrap() else {
            panic!()
        };
        for (o, e) in p.iter().zip(pred) {
            assert!((o - e).abs() < 1e-12, "{spec}: {o} vs {e}");
        }
        let Out::A2(t) = m.transform(x.view()).unwrap() else {
            panic!()
        };
        assert_eq!(t.ncols(), ranking.iter().filter(|&&r| r == 1).count());
    }
}

#[test]
fn coef_is_listed_for_linear_kinds_and_rfe_round_trips() {
    let (x, y) = data();
    let mut ols = make("OlsRegressor", &json!({})).unwrap();
    ols.fit(x.view(), Some(y.view()), None).unwrap();
    assert!(ols.capabilities().contains(&"coef"));
    let Out::A1(c) = ols.coef().unwrap() else {
        panic!()
    };
    assert_eq!(c.len(), 5);

    // A kind with neither coef nor feature_importances.
    let mut bad = make("Rfe", &json!({"estimator": "KnnRegressor"})).unwrap();
    let e = bad
        .fit(x.view(), Some(y.view()), None)
        .unwrap_err()
        .to_string();
    assert!(e.contains("neither coef nor feature_importances"), "{e}");

    let mut m = make(
        "Rfe",
        &json!({"estimator": "DecisionTreeRegressor", "estimator_params": {"max_depth": 3}}),
    )
    .unwrap();
    m.fit(x.view(), Some(y.view()), None).unwrap();
    let back = anofox_ml_registry::load("Rfe", &m.to_bytes().unwrap()).unwrap();
    let (Out::A1(a), Out::A1(b)) = (
        back.predict(x.view()).unwrap(),
        m.predict(x.view()).unwrap(),
    ) else {
        panic!()
    };
    assert_eq!(a, b);
}
