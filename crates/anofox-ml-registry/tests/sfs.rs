//! `SequentialFeatureSelector` through the registry (#125): greedy forward
//! or backward selection by the mean cross-validated score (R² for
//! regressors, accuracy for classifiers), against scikit-learn 1.8.0.

use anofox_ml_registry::{make, Out};
use ndarray::{Array1, Array2};
use serde_json::{json, Value};

fn data() -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let x = Array2::from_shape_fn((30, 5), |(i, j)| {
        ((i * 7 + j * 3 + i * j) % 13) as f64 / 6.0 - 1.0 + 0.02 * ((i * (j + 2)) % 5) as f64
    });
    let y = Array1::from_shape_fn(30, |i| {
        let r = x.row(i);
        2.0 * r[0] - r[2] + 0.5 * r[4] + 0.1 * r[1] + (((i * 3) % 7) as f64 - 3.0) / 20.0
    });
    let yc = Array1::from_shape_fn(30, |i| {
        let r = x.row(i);
        f64::from(r[0] - r[3] + 0.3 * (5.0 * i as f64).sin() > 0.0)
    });
    (x, y, yc)
}

#[test]
fn sequential_selection_matches_scikit_learn() {
    // SequentialFeatureSelector(est, n_features_to_select=k, direction=d).get_support()
    let (x, y, yc) = data();
    let cases = [
        (
            "OlsRegressor",
            &y,
            json!(2),
            "forward",
            [true, false, true, false, false],
        ),
        (
            "OlsRegressor",
            &y,
            json!(3),
            "backward",
            [true, false, true, false, true],
        ),
        (
            "GaussianNB",
            &yc,
            json!(2),
            "forward",
            [true, false, false, true, false],
        ),
        (
            "GaussianNB",
            &yc,
            json!(2),
            "backward",
            [true, false, false, true, false],
        ),
        (
            "OlsRegressor",
            &y,
            Value::Null,
            "forward",
            [true, false, true, false, false],
        ),
    ];
    for (est, target, k, direction, support) in cases {
        let spec = json!({"estimator": est, "n_features_to_select": k, "direction": direction});
        let mut m = make("SequentialFeatureSelector", &spec).unwrap();
        m.fit(x.view(), Some(target.view()), None).unwrap();
        let fitted: Value = serde_json::from_str(&m.fitted_json().unwrap()).unwrap();
        let ours: Vec<bool> = fitted["support"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_bool().unwrap())
            .collect();
        assert_eq!(ours, support, "{spec}");
        let Out::A2(t) = m.transform(x.view()).unwrap() else {
            panic!()
        };
        assert_eq!(t.ncols(), support.iter().filter(|&&s| s).count());
        let back =
            anofox_ml_registry::load("SequentialFeatureSelector", &m.to_bytes().unwrap()).unwrap();
        let Out::A2(t2) = back.transform(x.view()).unwrap() else {
            panic!()
        };
        assert_eq!(t, t2);
    }
    let mut bad = make(
        "SequentialFeatureSelector",
        &json!({"n_features_to_select": 5}),
    )
    .unwrap();
    assert!(bad.fit(x.view(), Some(y.view()), None).is_err());
}
