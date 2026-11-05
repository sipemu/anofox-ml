//! Golden tests for the gap batch 1 (#113) against scikit-learn 1.8.0:
//! DummyClassifier / DummyRegressor, NearestCentroid, RidgeClassifier and
//! Perceptron (`test_harness/generators/gen_baselines.py`).

mod common;

use anofox_ml_core::{
    ClassWeight, DecisionFunction, DummyClassifier, DummyClassifierStrategy, DummyRegressor,
    DummyRegressorStrategy, Fit, FitWeighted, Predict, PredictProba,
};
use anofox_ml_linear::{Perceptron, SgdPenalty};
use anofox_ml_neighbors::{CentroidMetric, CentroidPriors, NearestCentroid};
use anofox_ml_regression::RidgeClassifier;
use common::{assert_array1_close, assert_array2_close, json_to_array1, json_to_array2};
use ndarray::Array1;
use serde_json::Value;

fn golden() -> Value {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden_data/baselines.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn weights(c: &Value) -> Option<Array1<f64>> {
    (!c["sample_weight"].is_null()).then(|| json_to_array1(&c["sample_weight"]))
}

#[test]
fn dummy_matches_sklearn() {
    for c in golden()["dummy"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let (x, y, w) = (json_to_array2(&c["X"]), json_to_array1(&c["y"]), weights(c));
        let strategy = c["strategy"].as_str().unwrap();
        if c["kind"] == "classifier" {
            let s = match strategy {
                "most_frequent" => DummyClassifierStrategy::MostFrequent,
                "prior" => DummyClassifierStrategy::Prior,
                _ => DummyClassifierStrategy::Constant(c["constant"].as_f64().unwrap()),
            };
            let m = DummyClassifier::new(s)
                .fit_weighted(&x, &y, w.as_ref())
                .unwrap();
            assert_array1_close(
                &Array1::from(m.class_prior().to_vec()),
                &json_to_array1(&c["class_prior"]),
                1e-12,
                name,
            );
            assert_eq!(
                m.predict(&x).unwrap(),
                json_to_array1(&c["predict"]),
                "{name}"
            );
            assert_array2_close(
                &m.predict_proba(&x).unwrap(),
                &json_to_array2(&c["predict_proba"]),
                1e-12,
                name,
            );
        } else {
            let s = match strategy {
                "mean" => DummyRegressorStrategy::Mean,
                "median" => DummyRegressorStrategy::Median,
                "quantile" => DummyRegressorStrategy::Quantile(c["quantile"].as_f64().unwrap()),
                _ => DummyRegressorStrategy::Constant(c["constant"].as_f64().unwrap()),
            };
            let m = DummyRegressor::new(s)
                .fit_weighted(&x, &y, w.as_ref())
                .unwrap();
            common::assert_close(m.constant(), c["constant_"].as_f64().unwrap(), 1e-12, name);
        }
    }
}

#[test]
fn nearest_centroid_matches_sklearn() {
    for c in golden()["nearest_centroid"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let p = &c["params"];
        let mut est = NearestCentroid::new();
        if p["metric"] == "manhattan" {
            est = est.with_metric(CentroidMetric::Manhattan);
        }
        if let Some(t) = p["shrink_threshold"].as_f64() {
            est = est.with_shrink_threshold(t);
        }
        match &p["priors"] {
            Value::String(s) if s == "empirical" => {
                est = est.with_priors(CentroidPriors::Empirical)
            }
            Value::Array(v) => {
                est = est.with_priors(CentroidPriors::Given(
                    v.iter().map(|x| x.as_f64().unwrap()).collect(),
                ))
            }
            _ => {}
        }
        let m = est
            .fit(&json_to_array2(&c["X"]), &json_to_array1(&c["y"]))
            .unwrap();
        let xt = json_to_array2(&c["X_test"]);
        assert_array2_close(m.centroids(), &json_to_array2(&c["centroids"]), 1e-10, name);
        assert_eq!(
            m.predict(&xt).unwrap(),
            json_to_array1(&c["predict"]),
            "{name}"
        );
        if !c["predict_proba"].is_null() {
            assert_array2_close(
                &m.predict_proba(&xt).unwrap(),
                &json_to_array2(&c["predict_proba"]),
                1e-9,
                name,
            );
            assert_array2_close(
                &m.decision_function(&xt).unwrap(),
                &json_to_array2(&c["decision_function"]),
                1e-8,
                name,
            );
        }
    }
}

#[test]
fn ridge_classifier_matches_sklearn() {
    for c in golden()["ridge_classifier"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let p = &c["params"];
        let mut est = RidgeClassifier::new().with_alpha(p["alpha"].as_f64().unwrap());
        if p["fit_intercept"] == false {
            est = est.with_fit_intercept(false);
        }
        if p["class_weight"] == "balanced" {
            est = est.with_class_weight(ClassWeight::Balanced);
        }
        let m = est
            .fit_weighted(
                &json_to_array2(&c["X"]),
                &json_to_array1(&c["y"]),
                weights(c).as_ref(),
            )
            .unwrap();
        let xt = json_to_array2(&c["X_test"]);
        assert_array2_close(m.coefficients(), &json_to_array2(&c["coef"]), 1e-8, name);
        assert_array1_close(m.intercept(), &json_to_array1(&c["intercept"]), 1e-8, name);
        assert_array2_close(
            &m.decision_function(&xt).unwrap(),
            &json_to_array2(&c["decision_function"]),
            1e-8,
            name,
        );
        assert_eq!(
            m.predict(&xt).unwrap(),
            json_to_array1(&c["predict"]),
            "{name}"
        );
    }
}

#[test]
fn perceptron_matches_sklearn() {
    for c in golden()["perceptron"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let p = &c["params"];
        let mut est = Perceptron::new().with_shuffle(false);
        if p["penalty"] == "l2" {
            est = est
                .with_penalty(SgdPenalty::L2)
                .with_alpha(p["alpha"].as_f64().unwrap());
        }
        if let Some(eta) = p["eta0"].as_f64() {
            est = est.with_eta0(eta);
        }
        if let Some(it) = p["max_iter"].as_u64() {
            est = est.with_max_iter(it as usize);
        }
        if p.get("tol").is_some_and(|t| t.is_null()) {
            est = est.with_tol(None);
        }
        let x = json_to_array2(&c["X"]);
        let m = est.fit(&x, &json_to_array1(&c["y"])).unwrap();
        let coef = json_to_array2(&c["coef"]);
        assert_eq!(m.weights().len(), coef.nrows(), "{name}");
        for (k, w) in m.weights().iter().enumerate() {
            assert_array1_close(w, &coef.row(k).to_owned(), 1e-9, name);
        }
        assert_array1_close(
            &Array1::from(m.biases().to_vec()),
            &json_to_array1(&c["intercept"]),
            1e-9,
            name,
        );
        assert_eq!(m.n_iter() as u64, c["n_iter"].as_u64().unwrap(), "{name}");
        assert_eq!(
            m.predict(&x).unwrap(),
            json_to_array1(&c["predict"]),
            "{name}"
        );
    }
}
