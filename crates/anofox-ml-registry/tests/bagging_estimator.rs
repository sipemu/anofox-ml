//! Bagging over any registered kind (#125): `BaggingEstimatorRegressor` /
//! `BaggingEstimatorClassifier` with a nested `{estimator, estimator_params}`.

use anofox_ml_core::{Fit, Predict};
use anofox_ml_registry::{make, Out};
use ndarray::{Array1, Array2};
use serde_json::json;

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
        .map(|r| f64::from(r[0] + r[1] > 0.0))
        .collect();
    (x, y, yc)
}

#[test]
fn one_unresampled_estimator_is_the_base_kind() {
    // Nothing to resample: one estimator on every row and feature.
    let (x, y, _) = data();
    let mut bag = make(
        "BaggingEstimatorRegressor",
        &json!({
            "estimator": "KnnRegressor",
            "estimator_params": {"n_neighbors": 3},
            "n_estimators": 1,
            "bootstrap": false,
            "max_samples": 1.0,
        }),
    )
    .unwrap();
    bag.fit(x.view(), Some(y.view()), None).unwrap();
    let mut base = make("KnnRegressor", &json!({"n_neighbors": 3})).unwrap();
    base.fit(x.view(), Some(y.view()), None).unwrap();
    assert_eq!(
        out1(bag.predict(x.view()).unwrap()),
        out1(base.predict(x.view()).unwrap())
    );
}

#[test]
fn registry_bagging_follows_the_crates_draws() {
    // Same sampling settings and seed: the same rows and features per
    // estimator, so the same ensemble as the crate's generic bagging.
    let (x, y, yc) = data();
    let params = json!({"estimator_params": {"max_depth": 3}, "n_estimators": 7, "max_features": {"Fraction": 0.67}, "seed": 4});
    let mut reg = make(
        "BaggingEstimatorRegressor",
        &json!({"estimator": "DecisionTreeRegressor"}),
    )
    .unwrap();
    reg.set_params(&params).unwrap();
    reg.fit(x.view(), Some(y.view()), None).unwrap();
    let crate_reg = anofox_ml_ensemble::BaggingRegressor {
        n_estimators: 7,
        max_features: Some(anofox_ml_trees::MaxFeatures::Fraction(0.67)),
        seed: 4,
        ..Default::default()
    }
    .with_estimator(anofox_ml_trees::DecisionTreeRegressor::default().with_max_depth(Some(3)))
    .fit(&x, &y)
    .unwrap();
    assert_eq!(
        out1(reg.predict(x.view()).unwrap()),
        crate_reg.predict(&x).unwrap()
    );

    let mut clf = make("BaggingEstimatorClassifier", &json!({})).unwrap();
    clf.set_params(&params).unwrap();
    clf.fit(x.view(), Some(yc.view()), None).unwrap();
    let crate_clf = anofox_ml_ensemble::BaggingClassifier {
        n_estimators: 7,
        max_features: Some(anofox_ml_trees::MaxFeatures::Fraction(0.67)),
        seed: 4,
        ..Default::default()
    }
    .with_estimator(anofox_ml_trees::DecisionTreeClassifier::default().with_max_depth(Some(3)))
    .fit(&x, &yc)
    .unwrap();
    assert_eq!(
        out2(clf.predict_proba(x.view()).unwrap()),
        crate_clf.predict_proba(&x).unwrap()
    );
}

#[test]
fn unknown_bases_error_and_ensembles_round_trip() {
    let (x, y, yc) = data();
    let mut bad = make(
        "BaggingEstimatorRegressor",
        &json!({"estimator": "NoSuchKind"}),
    )
    .unwrap();
    let err = bad
        .fit(x.view(), Some(y.view()), None)
        .unwrap_err()
        .to_string();
    assert!(err.contains("NoSuchKind"), "{err}");

    let mut clf = make(
        "BaggingEstimatorClassifier",
        &json!({"estimator": "LogisticRegressor", "n_estimators": 4, "seed": 1}),
    )
    .unwrap();
    clf.fit(x.view(), Some(yc.view()), None).unwrap();
    let back =
        anofox_ml_registry::load("BaggingEstimatorClassifier", &clf.to_bytes().unwrap()).unwrap();
    assert_eq!(
        out2(back.predict_proba(x.view()).unwrap()),
        out2(clf.predict_proba(x.view()).unwrap())
    );
    assert_eq!(
        out1(back.predict(x.view()).unwrap()),
        out1(clf.predict(x.view()).unwrap())
    );
}
