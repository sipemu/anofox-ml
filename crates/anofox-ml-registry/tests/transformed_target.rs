//! `TransformedTargetRegressor` through the registry (#125): a named
//! function of `y` or a registered transformer fitted on `y`, against
//! scikit-learn 1.8.0 on the same inline data.

use anofox_ml_registry::{make, DynModel, Out};
use ndarray::{Array1, Array2};
use serde_json::{json, Value};

fn out1(o: Out) -> Array1<f64> {
    match o {
        Out::A1(a) => a,
        Out::A2(a) => panic!("expected a 1-D output, got {:?}", a.dim()),
    }
}

/// Positive, log-linear targets.
fn data() -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_fn((30, 3), |(i, j)| {
        ((i * 7 + j * 3 + i * j) % 13) as f64 / 6.0 - 1.0
    });
    let y = Array1::from_shape_fn(30, |i| {
        let r = x.row(i);
        (0.8 * r[0] - 0.5 * r[1] + 0.3 * r[2] + (((i * 5) % 7) as f64 - 3.0) / 20.0).exp()
    });
    (x, y)
}

fn ttr(spec: Value) -> Box<dyn DynModel> {
    make("TransformedTargetRegressor", &spec).unwrap()
}

#[test]
fn transformed_target_matches_scikit_learn() {
    let (x, y) = data();
    let ols = json!({"estimator": "OlsRegressor"});
    let cases = [
        // func=np.log / np.exp, np.log1p / np.expm1, np.sqrt / np.square
        (
            json!({"regressor": ols, "func": "log"}),
            [
                0.5598357875469169,
                0.6011030178738522,
                0.6680979833671142,
                2.1142240612410226,
            ],
        ),
        (
            json!({"regressor": ols, "func": "log1p"}),
            [
                0.5610620113029532,
                0.6144882131523429,
                0.6985314079761249,
                2.1102847847907213,
            ],
        ),
        (
            json!({"regressor": ols, "func": "sqrt"}),
            [
                0.5606430868681828,
                0.6089084554083084,
                0.6933440696285234,
                2.130745123677894,
            ],
        ),
        // transformer=StandardScaler(), and PowerTransformer() around KNN(3)
        (
            json!({"regressor": ols, "transformer": {"estimator": "StandardScaler"}}),
            [
                0.5607010579287801,
                0.6070779588325939,
                0.7167850374422542,
                2.1684919084304037,
            ],
        ),
        (
            json!({
                "regressor": {"estimator": "KnnRegressor", "estimator_params": {"n_neighbors": 3}},
                "transformer": {"estimator": "PowerTransformer"},
            }),
            [
                0.5490970891444147,
                0.5968379529291441,
                0.6485764698561907,
                2.0777990338811145,
            ],
        ),
    ];
    for (spec, expected) in cases {
        let mut m = ttr(spec.clone());
        m.fit(x.view(), Some(y.view()), None).unwrap();
        let p = out1(m.predict(x.slice(ndarray::s![..4, ..])).unwrap());
        // PowerTransformer matches sklearn to its Brent tolerance on λ.
        let tol = if spec["transformer"]["estimator"] == "PowerTransformer" {
            1e-8
        } else {
            1e-10
        };
        for (o, e) in p.iter().zip(expected) {
            assert!((o - e).abs() < tol, "{spec}: {o} vs {e}");
        }
    }
}

#[test]
fn transformed_target_checks_its_spec_and_round_trips() {
    let (x, y) = data();
    let ols = json!({"estimator": "OlsRegressor"});
    let e = ttr(
        json!({"regressor": ols, "func": "log", "transformer": {"estimator": "StandardScaler"}}),
    )
    .fit(x.view(), Some(y.view()), None)
    .unwrap_err()
    .to_string();
    assert!(e.contains("not both"), "{e}");

    let negative = y.mapv(|v| v - 1.0);
    let e = ttr(json!({"regressor": ols, "func": "log"}))
        .fit(x.view(), Some(negative.view()), None)
        .unwrap_err()
        .to_string();
    assert!(e.contains("not finite"), "{e}");

    // VarianceThreshold has no inverse_transform.
    let e = ttr(json!({"regressor": ols, "transformer": {"estimator": "VarianceThreshold"}}))
        .fit(x.view(), Some(y.view()), None)
        .unwrap_err()
        .to_string();
    assert!(e.contains("no inverse_transform"), "{e}");

    for spec in [
        json!({"regressor": ols, "func": "sqrt"}),
        json!({"regressor": ols, "transformer": {"estimator": "PowerTransformer"}}),
    ] {
        let mut m = ttr(spec);
        m.fit(x.view(), Some(y.view()), None).unwrap();
        let back =
            anofox_ml_registry::load("TransformedTargetRegressor", &m.to_bytes().unwrap()).unwrap();
        assert_eq!(
            out1(back.predict(x.view()).unwrap()),
            out1(m.predict(x.view()).unwrap())
        );
    }
}
