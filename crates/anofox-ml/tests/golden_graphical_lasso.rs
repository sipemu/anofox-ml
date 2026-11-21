//! Golden tests for GraphicalLasso and GraphicalLassoCV (#259) against
//! scikit-learn 1.8.0 (`test_harness/generators/gen_graphical_lasso.py`).

mod common;

use anofox_ml_core::FitUnsupervised;
use anofox_ml_preprocessing::{GlassoAlphas, GraphicalLasso, GraphicalLassoCV};
use common::{
    assert_array1_close, assert_array2_close, assert_close, json_to_array1, json_to_array2,
};
use serde_json::Value;

fn golden() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden_data/graphical_lasso.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn graphical_lasso_matches_sklearn() {
    for c in golden()["single"].as_array().unwrap() {
        let p = &c["params"];
        let est = GraphicalLasso {
            alpha: p["alpha"].as_f64().unwrap(),
            max_iter: p["max_iter"].as_u64().map_or(100, |v| v as usize),
            assume_centered: p["assume_centered"].as_bool().unwrap_or(false),
            ..Default::default()
        };
        let fit = est.fit(&json_to_array2(&c["X"])).unwrap();
        let ctx = format!("{p}");
        assert_array1_close(
            fit.cov.location(),
            &json_to_array1(&c["location"]),
            1e-12,
            &ctx,
        );
        assert_array2_close(
            fit.cov.covariance(),
            &json_to_array2(&c["covariance"]),
            1e-10,
            &ctx,
        );
        assert_array2_close(
            fit.cov.precision(),
            &json_to_array2(&c["precision"]),
            1e-9,
            &ctx,
        );
        assert_eq!(fit.n_iter as u64, c["n_iter"].as_u64().unwrap(), "{ctx}");
        let costs = json_to_array2(&c["costs"]);
        assert_eq!(fit.costs.len(), costs.nrows(), "{ctx}");
        for (k, &(cost, gap)) in fit.costs.iter().enumerate() {
            assert_close(cost, costs[[k, 0]], 1e-9, &ctx);
            assert_close(gap, costs[[k, 1]], 1e-9, &ctx);
        }
    }
}

#[test]
fn graphical_lasso_cv_matches_sklearn() {
    for c in golden()["cv"].as_array().unwrap() {
        let p = &c["params"];
        let alphas = match &p["alphas"] {
            Value::Array(a) => GlassoAlphas::List(a.iter().map(|v| v.as_f64().unwrap()).collect()),
            Value::Number(k) => GlassoAlphas::Grid(k.as_u64().unwrap() as usize),
            _ => GlassoAlphas::Grid(4),
        };
        let est = GraphicalLassoCV {
            alphas,
            n_refinements: p["n_refinements"].as_u64().map_or(4, |v| v as usize),
            cv: p["cv"].as_u64().map_or(5, |v| v as usize),
            ..Default::default()
        };
        let fit = est.fit(&json_to_array2(&c["X"])).unwrap();
        let ctx = format!("{p}");
        assert_close(fit.alpha, c["alpha"].as_f64().unwrap(), 1e-12, &ctx);
        let cv = fit.cv_results.as_ref().unwrap();
        assert_array1_close(
            &ndarray::Array1::from(cv.alphas.clone()),
            &json_to_array1(&c["alphas"]),
            1e-12,
            &ctx,
        );
        assert_array2_close(&cv.scores, &json_to_array2(&c["scores"]), 1e-9, &ctx);
        assert_array2_close(
            fit.cov.precision(),
            &json_to_array2(&c["precision"]),
            1e-9,
            &ctx,
        );
    }
}
