//! Golden tests for the gap batch 2 (#113) against scikit-learn 1.8.0:
//! univariate selection modes and TfidfTransformer
//! (`test_harness/generators/gen_univariate.py`).

mod common;

use anofox_ml_core::{Fit, FitUnsupervised, Transform};
use anofox_ml_preprocessing::select_k_best::ScoringFunction;
use anofox_ml_preprocessing::{GenericUnivariateSelect, SelectKBest, SelectionMode};
use anofox_ml_text::{TfidfNorm, TfidfTransformer};
use common::{assert_array1_close, assert_array2_close, json_to_array1, json_to_array2};
use serde_json::Value;

fn golden() -> Value {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden_data/univariate.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn selection_modes_match_sklearn() {
    for c in golden()["selection"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let scoring = match c["scoring"].as_str().unwrap() {
            "FClassif" => ScoringFunction::FClassif,
            "FRegression" => ScoringFunction::FRegression,
            _ => ScoringFunction::Chi2,
        };
        let param = &c["param"];
        let mode = match c["mode"].as_str().unwrap() {
            "percentile" => SelectionMode::Percentile(param.as_f64().unwrap()),
            "k_best" => SelectionMode::KBest(param.as_u64().unwrap() as usize),
            "fpr" => SelectionMode::Fpr(param.as_f64().unwrap()),
            "fdr" => SelectionMode::Fdr(param.as_f64().unwrap()),
            _ => SelectionMode::Fwe(param.as_f64().unwrap()),
        };
        let (x, y) = (json_to_array2(&c["X"]), json_to_array1(&c["y"]));
        let expected: Vec<usize> = c["support"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let fitted =
            Fit::fit(&GenericUnivariateSelect::new(scoring.clone(), mode), &x, &y).unwrap();
        assert_eq!(fitted.selected_indices(), expected.as_slice(), "{name}");
        assert_eq!(
            fitted.transform(&x).unwrap().ncols(),
            expected.len(),
            "{name}"
        );
        if let SelectionMode::KBest(k) = mode {
            let kb = SelectKBest::new(k, scoring).fit(&x, &y).unwrap();
            assert_eq!(
                kb.selected_indices(),
                expected.as_slice(),
                "{name} (SelectKBest)"
            );
        }
    }
}

#[test]
fn tfidf_transformer_matches_sklearn() {
    for c in golden()["tfidf"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let cfg = TfidfTransformer {
            norm: match c["norm"].as_str() {
                Some("l2") => TfidfNorm::L2,
                Some("l1") => TfidfNorm::L1,
                _ => TfidfNorm::None,
            },
            use_idf: c["use_idf"].as_bool().unwrap(),
            smooth_idf: c["smooth_idf"].as_bool().unwrap(),
            sublinear_tf: c["sublinear_tf"].as_bool().unwrap(),
        };
        let x = json_to_array2(&c["X"]);
        let fitted = cfg.fit(&x).unwrap();
        if !c["idf"].is_null() {
            assert_array1_close(
                fitted.idf().unwrap(),
                &json_to_array1(&c["idf"]),
                1e-12,
                name,
            );
        }
        assert_array2_close(
            &fitted.transform(&x).unwrap(),
            &json_to_array2(&c["transform"]),
            1e-12,
            name,
        );
    }
}
