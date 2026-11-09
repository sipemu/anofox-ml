//! Full linear-model fits in the registry (#125): R², the selected CV
//! alpha, the OLS / Ridge coefficient table and prediction intervals, and
//! models stored before these fields existed still load.

use anofox_ml_core::Fit;
use anofox_ml_registry::{load, make, Out};
use ndarray::{Array1, Array2};
use serde_json::{json, Value};

fn data() -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_fn((40, 3), |(i, j)| ((i * 7 + j * 5) % 11) as f64 / 3.0 - 1.5);
    let y = Array1::from_shape_fn(40, |i| {
        2.0 * x[[i, 0]] - x[[i, 1]] + 0.5 + ((i * 13) % 7) as f64 / 10.0
    });
    (x, y)
}

fn fitted(kind: &str, params: Value) -> (Box<dyn anofox_ml_registry::DynModel>, Value) {
    let (x, y) = data();
    let mut m = make(kind, &params).unwrap();
    m.fit(x.view(), Some(y.view()), None).unwrap();
    let m = load(kind, &m.to_bytes().unwrap()).unwrap();
    let state: Value = serde_json::from_str(&m.fitted_json().unwrap()).unwrap();
    (m, state)
}

#[test]
fn ols_and_ridge_keep_summary_and_intervals() {
    let (x, y) = data();
    let direct = anofox_ml_regression::OlsRegressor::new()
        .with_inference(true)
        .fit(&x, &y)
        .unwrap();
    let (m, state) = fitted("OlsRegressor", json!({"inference": true}));
    assert!((state["r_squared"].as_f64().unwrap() - direct.r_squared()).abs() < 1e-12);
    let se = &state["summary"]["std_errors"]["data"];
    assert_eq!(se.as_array().unwrap().len(), 3);
    let Out::A2(iv) = m.predict_interval(x.view(), 0.1).unwrap() else {
        panic!("n × 2 expected")
    };
    let want = direct
        .predict_interval(&x, anofox_ml_regression::IntervalKind::Prediction, 0.9)
        .unwrap();
    for i in 0..x.nrows() {
        assert!((iv[[i, 0]] - want.lower[i]).abs() < 1e-12);
        assert!((iv[[i, 1]] - want.upper[i]).abs() < 1e-12);
    }
    assert!(m.capabilities().contains(&"predict_interval"));
    assert!(m.predict_interval(x.view(), 1.5).is_err());

    let (m, state) = fitted("RidgeRegressor", json!({"inference": true}));
    assert!(state["summary"].is_object());
    assert!(m.predict_interval(x.view(), 0.05).is_ok());
    // Without inference there is no table and no interval.
    let (m, state) = fitted("OlsRegressor", json!({}));
    assert!(state["summary"].is_null());
    assert!(m.predict_interval(x.view(), 0.1).is_err());
}

#[test]
fn r_squared_and_cv_alpha_are_stored() {
    for kind in [
        "LassoRegressor",
        "ElasticNetRegressor",
        "HuberRegressor",
        "QuantileRegressor",
    ] {
        let (_, state) = fitted(kind, json!({}));
        assert!(state["r_squared"].as_f64().is_some(), "{kind}");
    }
    let (x, y) = data();
    let direct = anofox_ml_regression::LassoCrossValidated::new()
        .fit(&x, &y)
        .unwrap();
    let (_, state) = fitted("LassoCrossValidated", json!({}));
    assert!((state["alpha"].as_f64().unwrap() - direct.best_alpha()).abs() < 1e-15);
}

/// A model stored with only coefficients and intercept (anofox-ml <= 0.4).
#[test]
fn compact_models_from_older_versions_still_load() {
    #[derive(serde::Serialize)]
    struct LinearFit {
        coef: Array1<f64>,
        intercept: f64,
    }
    let config = anofox_ml_regression::OlsRegressor::new();
    let fitted = LinearFit {
        coef: Array1::from(vec![1.0, 2.0, 0.0]),
        intercept: 0.5,
    };
    let bytes =
        anofox_ml_core::persistence::to_binary_bytes(&anofox_ml_core::persistence::ModelBundle {
            config: &config,
            fitted: &fitted,
        })
        .unwrap();
    let m = load("OlsRegressor", &bytes).unwrap();
    let x = Array2::from_shape_vec((1, 3), vec![1.0, 1.0, 1.0]).unwrap();
    let Out::A1(p) = m.predict(x.view()).unwrap() else {
        panic!()
    };
    assert_eq!(p[0], 3.5);
    assert!(m.predict_interval(x.view(), 0.1).is_err());
}
