//! LightGBM objectives through the registry's JSON configs (#206): the
//! registry model equals the directly built one.

use anofox_ml_core::{Fit, Predict};
use anofox_ml_ensemble::{LgbmClassifier, LgbmObjective, LgbmRegressor};
use anofox_ml_registry::{make, Out};
use ndarray::{Array1, Array2};
use serde_json::json;

fn data() -> (Array2<f64>, Array1<f64>) {
    let n = 120;
    let x = Array2::from_shape_fn((n, 2), |(i, j)| ((i * (3 + 4 * j)) % 17) as f64 / 8.0 - 1.0);
    let y = Array1::from_shape_fn(n, |i| (x[[i, 0]] - 0.5 * x[[i, 1]]).exp() * (i % 3) as f64);
    (x, y)
}

fn out(o: Out) -> Vec<f64> {
    match o {
        Out::A1(a) => a.to_vec(),
        Out::A2(a) => a.iter().copied().collect(),
    }
}

#[test]
fn regressor_objectives_from_json() {
    let (x, y) = data();
    let y01 = y.mapv(|v| v / (1.0 + v));
    let cases = [
        (
            json!({"Tweedie": {"variance_power": 1.3}}),
            LgbmObjective::Tweedie {
                variance_power: 1.3,
            },
            &y,
        ),
        (json!("CrossEntropy"), LgbmObjective::CrossEntropy, &y01),
    ];
    for (objective, direct, y) in cases {
        let params = json!({"objective": objective, "n_estimators": 10, "min_child_samples": 5});
        let mut m = make("LgbmRegressor", &params).unwrap();
        m.fit(x.view(), Some(y.view()), None).unwrap();
        let expected = LgbmRegressor::new()
            .with_n_estimators(10)
            .with_min_child_samples(5)
            .with_objective(direct)
            .fit(&x, y)
            .unwrap()
            .predict(&x)
            .unwrap();
        assert_eq!(out(m.predict(x.view()).unwrap()), expected.to_vec());
    }
}

#[test]
fn classifier_one_vs_all_from_json() {
    let (x, y) = data();
    let y = y.mapv(|v| (v > 0.0) as u8 as f64 + (v > 1.5) as u8 as f64);
    let params = json!({"multiclass_ova": true, "n_estimators": 10, "min_child_samples": 5});
    let mut m = make("LgbmClassifier", &params).unwrap();
    m.fit(x.view(), Some(y.view()), None).unwrap();
    let expected = LgbmClassifier::new()
        .with_n_estimators(10)
        .with_min_child_samples(5)
        .with_multiclass_ova(true)
        .fit(&x, &y)
        .unwrap()
        .predict_proba(&x)
        .unwrap();
    assert_eq!(
        out(m.predict_proba(x.view()).unwrap()),
        expected.iter().copied().collect::<Vec<_>>()
    );
}

#[test]
fn ranker_fits_with_groups() {
    let (x, y) = data();
    let labels = y.mapv(|v| (v.min(3.0)).floor());
    let group = vec![20usize; 6];
    let mut m = make(
        "LgbmRanker",
        &json!({"n_estimators": 10, "min_child_samples": 5}),
    )
    .unwrap();
    assert!(m.fit(x.view(), Some(labels.view()), None).is_err());
    m.fit_with_group(x.view(), Some(labels.view()), None, &group)
        .unwrap();
    let expected = LgbmRegressor::ranker()
        .with_n_estimators(10)
        .with_min_child_samples(5)
        .fit_with_eval(
            &x,
            &labels,
            &anofox_ml_ensemble::LgbmFitOptions {
                group: Some(&group),
                ..Default::default()
            },
        )
        .unwrap()
        .predict(&x)
        .unwrap();
    assert_eq!(out(m.predict(x.view()).unwrap()), expected.to_vec());
    // Groups that do not cover the rows are an error.
    assert!(m
        .fit_with_group(x.view(), Some(labels.view()), None, &[10, 10])
        .is_err());
}
