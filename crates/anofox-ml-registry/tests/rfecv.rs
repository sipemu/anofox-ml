//! `Rfecv` through the registry (#125): RFE with the number of features
//! chosen by the cross-validated score, against scikit-learn 1.8.0.

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

fn fitted(m: &dyn anofox_ml_registry::DynModel) -> Value {
    serde_json::from_str(&m.fitted_json().unwrap()).unwrap()
}

#[test]
fn rfecv_matches_scikit_learn() {
    // RFECV(LinearRegression()) and RFECV(LinearRegression(), step=2,
    // min_features_to_select=2): ranking_, cv_results_["mean_test_score"]
    // (fewest features first) and predict(X[:2]).
    let (x, y) = data();
    let cases = [
        (
            json!({"estimator": "OlsRegressor"}),
            vec![1, 1, 1, 2, 1],
            vec![
                0.6631308613611673,
                0.9472544247617325,
                0.9912487037083778,
                0.9925152055917771,
                0.9919809066381114,
            ],
            Some([-1.602251762428871, 1.408771560542287]),
        ),
        (
            json!({"estimator": "OlsRegressor", "step": 2, "min_features_to_select": 2}),
            vec![1, 1, 1, 1, 1],
            vec![0.9472544247617325, 0.9912487037083778, 0.9919809066381114],
            None,
        ),
    ];
    for (spec, ranking, scores, pred) in cases {
        let mut m = make("Rfecv", &spec).unwrap();
        m.fit(x.view(), Some(y.view()), None).unwrap();
        let f = fitted(m.as_ref());
        let ours: Vec<u64> = f["rfe"]["ranking"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        assert_eq!(ours, ranking, "{spec}");
        let mean: Vec<f64> = f["mean_test_score"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        assert_eq!(mean.len(), scores.len(), "{spec}");
        for (o, e) in mean.iter().zip(&scores) {
            assert!((o - e).abs() < 1e-12, "{spec}: score {o} vs {e}");
        }
        if let Some(pred) = pred {
            let Out::A1(p) = m.predict(x.slice(ndarray::s![..2, ..])).unwrap() else {
                panic!()
            };
            for (o, e) in p.iter().zip(pred) {
                assert!((o - e).abs() < 1e-12, "{o} vs {e}");
            }
        }
        let back = anofox_ml_registry::load("Rfecv", &m.to_bytes().unwrap()).unwrap();
        let (Out::A2(a), Out::A2(b)) = (
            back.transform(x.view()).unwrap(),
            m.transform(x.view()).unwrap(),
        ) else {
            panic!()
        };
        assert_eq!(a, b);
    }
}
