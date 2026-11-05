//! Golden tests for LogisticRegressionCV and OrthogonalMatchingPursuitCV
//! (#113, batch 7) against scikit-learn 1.8.0
//! (`test_harness/generators/gen_cv_linear.py`).

mod common;

use anofox_ml_core::{Fit, Predict, PredictProba};
use anofox_ml_regression::{
    CGrid, LogisticCvScoring, LogisticRegressionCV, OrthogonalMatchingPursuitCV,
};
use common::{
    assert_array1_close, assert_array2_close, assert_close, json_to_array1, json_to_array2,
};
use serde_json::Value;

fn golden() -> Value {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden_data/cv_linear.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn logistic_regression_cv_matches_sklearn() {
    for c in golden()["logistic"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let p = &c["params"];
        let mut est = LogisticRegressionCV::default();
        est.base = est.base.with_tol(1e-10).with_max_iter(100_000);
        if let Some(cs) = p["Cs"].as_array() {
            est.cs = CGrid::Values(cs.iter().map(|v| v.as_f64().unwrap()).collect());
        }
        if let Some(cv) = p["cv"].as_u64() {
            est.cv = cv as usize;
        }
        if p["scoring"] == "neg_log_loss" {
            est.scoring = LogisticCvScoring::NegLogLoss;
        }
        let (x, y) = (json_to_array2(&c["X"]), json_to_array1(&c["y"]));
        let m = est.fit(&x, &y).unwrap();
        assert_array2_close(m.scores(), &json_to_array2(&c["scores"]), 1e-6, name);
        assert_close(m.c(), c["C"].as_f64().unwrap(), 1e-12, name);
        assert_eq!(
            m.predict(&x).unwrap(),
            json_to_array1(&c["predict"]),
            "{name}"
        );
        assert_array2_close(
            &m.predict_proba(&x).unwrap(),
            &json_to_array2(&c["predict_proba"]),
            1e-6,
            name,
        );
    }
}

#[test]
fn omp_cv_matches_sklearn() {
    for c in golden()["omp"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let p = &c["params"];
        let est = OrthogonalMatchingPursuitCV {
            max_iter: p["max_iter"].as_u64().map(|v| v as usize),
            cv: p["cv"].as_u64().unwrap_or(5) as usize,
            fit_intercept: p["fit_intercept"].as_bool().unwrap_or(true),
        };
        let (x, y) = (json_to_array2(&c["X"]), json_to_array1(&c["y"]));
        let m = est.fit(&x, &y).unwrap();
        assert_eq!(
            m.n_nonzero_coefs() as u64,
            c["n_nonzero_coefs"].as_u64().unwrap(),
            "{name}"
        );
        assert_array1_close(&m.model().coef, &json_to_array1(&c["coef"]), 1e-9, name);
        assert_close(
            m.model().intercept,
            c["intercept"].as_f64().unwrap(),
            1e-9,
            name,
        );
        assert_array1_close(
            &m.predict(&x).unwrap(),
            &json_to_array1(&c["predict"]),
            1e-9,
            name,
        );
    }
}
