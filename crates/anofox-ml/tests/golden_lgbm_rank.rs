//! LightGBM's ranking objectives (`LgbmRegressor` with `LambdaRank` /
//! `RankXendcg` and query groups) against LightGBM 4.7.0 (#207).
//! Generator: `test_harness/generators/gen_lgbm_rank.py`.

mod common;

use anofox_ml::prelude::*;
use anofox_ml_ensemble::{LgbmFitOptions, LgbmObjective, LgbmRankingParams};
use common::{json_to_array1, json_to_array2, load_golden_data};

#[test]
fn ranking_matches_lightgbm() {
    let mut failures = Vec::new();
    for case in load_golden_data("lgbm_rank.json") {
        let name = case["name"].as_str().unwrap();
        let p = &case["params"];
        let mut ranking = LgbmRankingParams::default();
        if let Some(t) = p["lambdarank_truncation_level"].as_u64() {
            ranking.truncation_level = t as usize;
        }
        if let Some(b) = p["lambdarank_norm"].as_bool() {
            ranking.norm = b;
        }
        if let Some(s) = p["sigmoid"].as_f64() {
            ranking.sigmoid = s;
        }
        if let Some(g) = p["label_gain"].as_array() {
            ranking.label_gain = g.iter().map(|v| v.as_f64().unwrap()).collect();
        }
        let objective = if p["objective"] == "rank_xendcg" {
            LgbmObjective::RankXendcg
        } else {
            LgbmObjective::LambdaRank
        };
        let mut m = LgbmRegressor::new()
            .with_objective(objective)
            .with_ranking(ranking)
            .with_n_estimators(p["n_estimators"].as_u64().unwrap() as usize)
            .with_learning_rate(p["learning_rate"].as_f64().unwrap())
            .with_num_leaves(p["num_leaves"].as_u64().unwrap() as usize)
            .with_min_child_samples(p["min_child_samples"].as_u64().unwrap() as usize);
        if let Some(seed) = p["random_state"].as_u64() {
            m = m.with_seed(seed);
        }
        if let Some(f) = p["subsample"].as_f64() {
            m = m
                .with_subsample(f)
                .with_subsample_freq(p["subsample_freq"].as_u64().unwrap() as usize);
        }
        let group: Vec<usize> = case["group"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let sw = (!case["sample_weight"].is_null()).then(|| json_to_array1(&case["sample_weight"]));
        let opts = LgbmFitOptions {
            group: Some(&group),
            sample_weight: sw.as_ref().map(|w| w.view()),
            ..Default::default()
        };
        let x = json_to_array2(&case["X"]);
        let y = json_to_array1(&case["y"]);
        let fitted = m.fit_with_eval(&x, &y, &opts).unwrap();
        let pred = fitted.predict(&json_to_array2(&case["X_test"])).unwrap();
        let expected = json_to_array1(&case["predictions"]);
        let err = pred
            .iter()
            .zip(&expected)
            .map(|(a, e)| (a - e).abs() / e.abs().max(1.0))
            .fold(0.0, f64::max);
        eprintln!("{name}: {err:e}");
        if err > 1e-12 {
            failures.push(format!("{name}: max error {err:e}"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
