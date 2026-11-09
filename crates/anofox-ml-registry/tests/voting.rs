//! `VotingRegressor` / `VotingClassifier` through the registry (#125),
//! against scikit-learn 1.8.0 on the same inline data.

use anofox_ml_registry::{make, Out};
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

fn data() -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let x = Array2::from_shape_fn((30, 3), |(i, j)| {
        ((i * 7 + j * 3 + i * j) % 13) as f64 / 6.0 - 1.0
    });
    let y = x
        .rows()
        .into_iter()
        .map(|r| r[0] - 2.0 * r[1] + r[2] * r[2])
        .collect();
    let yc = x
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
    (x, y, yc)
}

/// A 4 × 2 × 2 grid of query points.
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

fn classifier(voting: &str, weights: Value) -> Box<dyn anofox_ml_registry::DynModel> {
    make(
        "VotingClassifier",
        &json!({
            "estimators": [
                {"estimator": "GaussianNB", "estimator_params": {}},
                {"estimator": "KnnClassifier", "estimator_params": {"n_neighbors": 3}}
            ],
            "voting": voting,
            "weights": weights,
        }),
    )
    .unwrap()
}

#[test]
fn voting_regressor_matches_scikit_learn() {
    // VotingRegressor([LinearRegression(), KNeighborsRegressor(3)], weights=[2, 1]).
    let (x, y, _) = data();
    let mut m = make(
        "VotingRegressor",
        &json!({
            "estimators": [
                {"estimator": "OlsRegressor", "estimator_params": {}},
                {"estimator": "KnnRegressor", "estimator_params": {"n_neighbors": 3}}
            ],
            "weights": [2.0, 1.0],
        }),
    )
    .unwrap();
    m.fit(x.view(), Some(y.view()), None).unwrap();
    let p = out1(m.predict(x.slice(ndarray::s![..4, ..])).unwrap());
    for (o, e) in p.iter().zip([
        0.23487634762190748,
        -1.155750860545445,
        -0.3689452831422986,
        2.3270942852191925,
    ]) {
        assert!((o - e).abs() < 1e-12, "{o} vs {e}");
    }
}

#[test]
fn voting_classifier_matches_scikit_learn() {
    // VotingClassifier([GaussianNB(), KNeighborsClassifier(3)], voting=..., weights=...)
    // on the grid; equal-weight hard ties go to the first class.
    let (x, _, yc) = data();
    let q = grid();
    let cases = [
        (
            "hard",
            json!([1.0, 1.0]),
            [
                0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 2., 1., 2., 1., 2., 2.,
            ],
        ),
        (
            "soft",
            json!([1.0, 1.0]),
            [
                0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 2., 1., 2., 0., 2., 2.,
            ],
        ),
        (
            "hard",
            json!([1.0, 2.0]),
            [
                0., 0., 1., 0., 1., 0., 1., 1., 0., 1., 2., 1., 2., 1., 2., 2.,
            ],
        ),
        (
            "soft",
            json!([1.0, 2.0]),
            [
                0., 0., 1., 0., 0., 0., 1., 1., 0., 1., 2., 1., 2., 1., 2., 2.,
            ],
        ),
    ];
    for (voting, weights, expected) in cases {
        let mut m = classifier(voting, weights.clone());
        m.fit(x.view(), Some(yc.view()), None).unwrap();
        assert_eq!(
            out1(m.predict(q.view()).unwrap()).to_vec(),
            expected.to_vec(),
            "{voting} {weights}"
        );
    }
    let mut soft = classifier("soft", json!([1.0, 2.0]));
    soft.fit(x.view(), Some(yc.view()), None).unwrap();
    let p = out2(soft.predict_proba(q.slice(ndarray::s![..3, ..])).unwrap());
    let expected = [
        [
            0.9892187387107431,
            0.010773043774717448,
            8.21751453958191e-06,
        ],
        [
            0.9807024986191145,
            0.019296574808678874,
            9.265722065294191e-07,
        ],
        [
            0.3019822123570583,
            0.6979419499784526,
            7.583766448898723e-05,
        ],
    ];
    for (i, row) in expected.iter().enumerate() {
        for (j, e) in row.iter().enumerate() {
            assert!(
                (p[[i, j]] - e).abs() < 1e-12,
                "proba[{i},{j}] {} vs {e}",
                p[[i, j]]
            );
        }
    }
}

#[test]
fn voting_checks_its_spec_and_round_trips() {
    let (x, y, yc) = data();
    let mut empty = make("VotingRegressor", &json!({})).unwrap();
    let e = empty
        .fit(x.view(), Some(y.view()), None)
        .unwrap_err()
        .to_string();
    assert!(e.contains("at least one estimator"), "{e}");

    let mut short = classifier("hard", json!([1.0]));
    let e = short
        .fit(x.view(), Some(yc.view()), None)
        .unwrap_err()
        .to_string();
    assert!(e.contains("1 weights for 2 estimators"), "{e}");

    let mut hard = classifier("hard", Value::Null);
    hard.fit(x.view(), Some(yc.view()), None).unwrap();
    let e = hard.predict_proba(x.view()).unwrap_err().to_string();
    assert!(e.contains("soft"), "{e}");

    let mut soft = classifier("soft", json!([1.0, 2.0]));
    soft.fit(x.view(), Some(yc.view()), None).unwrap();
    let back = anofox_ml_registry::load("VotingClassifier", &soft.to_bytes().unwrap()).unwrap();
    let q = grid();
    assert_eq!(
        out2(back.predict_proba(q.view()).unwrap()),
        out2(soft.predict_proba(q.view()).unwrap())
    );
    assert_eq!(
        out1(back.predict(q.view()).unwrap()),
        out1(soft.predict(q.view()).unwrap())
    );
}
