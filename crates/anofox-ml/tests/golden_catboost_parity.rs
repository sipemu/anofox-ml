//! CatBoost-style estimators against catboost 1.2.10 (#127), with its
//! randomness switched off. Generator: `test_harness/generators/gen_catboost_parity.py`.

mod common;

use anofox_ml::prelude::*;
use anofox_ml_ensemble::catboost::catboost_borders;
use anofox_ml_ensemble::{
    CatBoostAutoClassWeights, CatBoostBacktracking, CatBoostBoostingType, CatBoostBootstrap,
    CatBoostClassifier, CatBoostClassifierLoss, CatBoostEmbeddingCalcer, CatBoostEmbeddingOptions,
    CatBoostEvalSet, CatBoostGrowPolicy, CatBoostLeafEstimation, CatBoostMultiEvalSet,
    CatBoostOdType, CatBoostRegressionLoss, CatBoostRegressor, CatBoostScoreFunction,
    CatBoostShrinkMode,
};
use common::{json_to_array1, load_golden_file};
use ndarray::{Array1, Array2};
use serde_json::Value;

/// The models agree with CatBoost to rounding.
const TOL: f64 = 1e-12;

fn array2_with_nan(val: &Value) -> Array2<f64> {
    let rows = val.as_array().unwrap();
    let p = rows[0].as_array().unwrap().len();
    let flat: Vec<f64> = rows
        .iter()
        .flat_map(|r| r.as_array().unwrap().iter())
        .map(|v| v.as_f64().unwrap_or(f64::NAN))
        .collect();
    Array2::from_shape_vec((rows.len(), p), flat).unwrap()
}

fn worst(pred: impl IntoIterator<Item = f64>, expected: impl IntoIterator<Item = f64>) -> f64 {
    pred.into_iter()
        .zip(expected)
        .map(|(a, e)| (a - e).abs() / e.abs().max(1.0))
        // A NaN prediction is a mismatch (`f64::max` would skip it).
        .fold(
            0.0,
            |w, d| if d.is_nan() { f64::INFINITY } else { w.max(d) },
        )
}

/// Embedding columns of a golden case: one `rows x dim` array each.
fn embeddings_of(v: &Value) -> Vec<Array2<f64>> {
    v.as_array().unwrap().iter().map(array2_with_nan).collect()
}

fn views(e: &[Array2<f64>]) -> Vec<ndarray::ArrayView2<'_, f64>> {
    e.iter().map(|a| a.view()).collect()
}

/// CatBoost `embedding_calcers` strings (`LDA`, `LDA:reg=0.5`, `KNN:k=3`).
fn embedding_options(v: &Value) -> CatBoostEmbeddingOptions {
    let calcers = v
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            let c = c.as_str().unwrap();
            let (name, opts) = c.split_once(':').unwrap_or((c, ""));
            if name == "KNN" {
                let k = opts.strip_prefix("k=").map_or(5, |k| k.parse().unwrap());
                return CatBoostEmbeddingCalcer::Knn { k };
            }
            assert_eq!(name, "LDA");
            let (mut components, mut reg, mut likelihood) = (None, 0.00005f32, false);
            for kv in opts.split(',').filter(|s| !s.is_empty()) {
                let (k, val) = kv.split_once('=').unwrap();
                match k {
                    "components" => components = Some(val.parse().unwrap()),
                    "reg" => reg = val.parse().unwrap(),
                    "likelihood" => likelihood = val == "true",
                    _ => panic!("unknown LDA option {k}"),
                }
            }
            CatBoostEmbeddingCalcer::Lda {
                components,
                reg,
                likelihood,
            }
        })
        .collect();
    CatBoostEmbeddingOptions { calcers }
}

#[test]
fn quantization_matches_catboost() {
    let data = load_golden_file("catboost_parity.json");
    for case in data["quantization"].as_array().unwrap() {
        let x = array2_with_nan(&case["X"]);
        let count = case["border_count"].as_u64().unwrap() as usize;
        let ours = catboost_borders(x.view(), count);
        for (f, expected) in case["borders"].as_array().unwrap().iter().enumerate() {
            let expected: Vec<f32> = expected
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            assert_eq!(ours[f], expected, "{} feature {f}", case["name"]);
        }
    }
}

macro_rules! configure {
    ($m:expr, $p:expr) => {{
        let p = $p;
        let mut m = $m
            .with_iterations(p["iterations"].as_u64().unwrap() as usize)
            .with_depth(p["depth"].as_u64().unwrap_or(6) as usize);
        if let Some(lr) = p["learning_rate"].as_f64() {
            m = m.with_learning_rate(lr);
        }
        if let Some(l2) = p["l2_leaf_reg"].as_f64() {
            m = m.with_l2_leaf_reg(l2);
        }
        if let Some(b) = p["border_count"].as_u64() {
            m = m.with_border_count(b as usize);
        }
        if p["score_function"] == "L2" {
            m = m.with_score_function(CatBoostScoreFunction::L2);
        }
        if p["leaf_estimation_method"] == "Gradient" {
            m = m.with_leaf_estimation_method(CatBoostLeafEstimation::Gradient);
        }
        if let Some(b) = p["boost_from_average"].as_bool() {
            m = m.with_boost_from_average(b);
        }
        if let Some(c) = p["cat_features"].as_array() {
            m = m.with_cat_features(c.iter().map(|v| v.as_u64().unwrap() as usize).collect());
        }
        if p["boosting_type"] == "Ordered" {
            m = m.with_boosting_type(CatBoostBoostingType::Ordered);
        }
        if let Some(f) = p["fold_len_multiplier"].as_f64() {
            m = m.with_fold_len_multiplier(f);
        }
        // The generator's common settings (not stored) include `has_time`.
        m = m.with_has_time(p["has_time"].as_bool().unwrap_or(true));
        // Settings a case leaves at catboost's defaults.
        let default = |k: &str| {
            p["defaults"]
                .as_array()
                .is_some_and(|d| d.iter().any(|v| v == k))
        };
        if !default("random_strength") {
            m = m.with_random_strength(p["random_strength"].as_f64().unwrap_or(0.0));
        }
        if let Some(r) = p["bayesian_matrix_reg"].as_f64() {
            m = m.with_bayesian_matrix_reg(r);
        }
        if !default("bootstrap_type") {
            m = m.with_bootstrap_type(match p["bootstrap_type"].as_str().unwrap_or("No") {
                "Bayesian" => CatBoostBootstrap::Bayesian,
                "Bernoulli" => CatBoostBootstrap::Bernoulli,
                "MVS" => CatBoostBootstrap::Mvs,
                _ => CatBoostBootstrap::No,
            });
        }
        if let Some(v) = p["subsample"].as_f64() {
            m = m.with_subsample(v);
        }
        if let Some(v) = p["bagging_temperature"].as_f64() {
            m = m.with_bagging_temperature(v);
        }
        if let Some(v) = p["mvs_reg"].as_f64() {
            m = m.with_mvs_reg(v);
        }
        if let Some(n) = p["random_seed"].as_u64() {
            m = m.with_random_seed(n);
        }
        if let Some(n) = p["permutation_count"].as_u64() {
            m = m.with_permutation_count(n as usize);
        }
        if let Some(n) = p["leaf_estimation_iterations"].as_u64() {
            m = m.with_leaf_estimation_iterations(n as usize);
        }
        if p["leaf_estimation_backtracking"] == "No" {
            m = m.with_leaf_estimation_backtracking(CatBoostBacktracking::No);
        }
        if let Some(n) = p["max_ctr_complexity"].as_u64() {
            m = m.with_max_ctr_complexity(n as usize);
        }
        if p["grow_policy"] == "Depthwise" {
            m = m.with_grow_policy(CatBoostGrowPolicy::Depthwise);
        }
        if p["grow_policy"] == "Lossguide" {
            m = m.with_grow_policy(CatBoostGrowPolicy::Lossguide);
        }
        if let Some(n) = p["max_leaves"].as_u64() {
            m = m.with_max_leaves(n as usize);
        }
        if let Some(n) = p["min_data_in_leaf"].as_u64() {
            m = m.with_min_data_in_leaf(n as usize);
        }
        if let Some(n) = p["early_stopping_rounds"].as_u64() {
            m = m.with_early_stopping_rounds(n as usize);
        }
        if let Some(b) = p["use_best_model"].as_bool() {
            m = m.with_use_best_model(b);
        }
        if p["od_type"] == "IncToDec" {
            m = m.with_od_type(CatBoostOdType::IncToDec);
        }
        if let Some(v) = p["od_pval"].as_f64() {
            m = m.with_od_pval(v);
        }
        if let Some(n) = p["od_wait"].as_u64() {
            m = m.with_od_wait(n as usize);
        }
        if let Some(n) = p["one_hot_max_size"].as_u64() {
            m = m.with_one_hot_max_size(n as usize);
        }
        if let Some(c) = p["monotone_constraints"].as_array() {
            m = m.with_monotone_constraints(c.iter().map(|v| v.as_i64().unwrap() as i8).collect());
        }
        if let Some(r) = p["model_shrink_rate"].as_f64() {
            m = m.with_model_shrink_rate(r);
        }
        if p["model_shrink_mode"] == "Decreasing" {
            m = m.with_model_shrink_mode(CatBoostShrinkMode::Decreasing);
        }
        m
    }};
}

/// A MultiRMSE case (a target matrix): the worst prediction error, and an
/// eval set mismatch.
fn multi_rmse_case(case: &Value, x: &Array2<f64>) -> (f64, Option<String>) {
    let p = &case["params"];
    let m = configure!(CatBoostRegressor::new(), p)
        .with_loss_function(CatBoostRegressionLoss::MultiRmse);
    let y = array2_with_nan(&case["y"]);
    let sw: Option<Array1<f64>> =
        (!case["sample_weight"].is_null()).then(|| json_to_array1(&case["sample_weight"]));
    let sw = sw.as_ref().map(|w| w.view());
    let ev = &case["eval"];
    let mut eval_err = None;
    let f = if ev.is_null() {
        m.fit_multi(x.view(), y.view(), sw).unwrap()
    } else {
        let (xe, ye) = (array2_with_nan(&ev["X"]), array2_with_nan(&ev["y"]));
        let we: Option<Array1<f64>> =
            (!ev["sample_weight"].is_null()).then(|| json_to_array1(&ev["sample_weight"]));
        let e = CatBoostMultiEvalSet {
            x: xe.view(),
            y: ye.view(),
            sample_weight: we.as_ref().map(|w| w.view()),
        };
        let f = m.fit_multi_with_eval(x.view(), y.view(), sw, e).unwrap();
        let expected = json_to_array1(&ev["metrics"]);
        let e = worst(f.eval_metrics().iter().copied(), expected.iter().copied());
        let n = ev["tree_count"].as_u64().unwrap() as usize;
        if f.eval_metrics().len() != expected.len() || e > TOL || f.n_estimators() != n {
            eval_err = Some(format!(
                "eval {} vs {} metrics (worst {e:e}), {} vs {n} trees",
                f.eval_metrics().len(),
                expected.len(),
                f.n_estimators()
            ));
        }
        f
    };
    let x_test = array2_with_nan(&case["X_test"]);
    let err = worst(
        f.predict_multi(&x_test).unwrap(),
        array2_with_nan(&case["predictions"]),
    );
    (err, eval_err)
}

#[test]
fn catboost_models_match_catboost() {
    let data = load_golden_file("catboost_parity.json");
    let mut report = Vec::new();
    let mut failures = Vec::new();
    for case in data["models"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let x = array2_with_nan(&case["X"]);
        if case["params"]["loss_function"] == "MultiRMSE" {
            let (err, eval_err) = multi_rmse_case(case, &x);
            report.push(format!("{name}: {err:e}"));
            if err > TOL || eval_err.is_some() {
                failures.push(format!(
                    "{name}: worst {err:e} {}",
                    eval_err.unwrap_or_default()
                ));
            }
            continue;
        }
        let y = json_to_array1(&case["y"]);
        let sw: Option<Array1<f64>> =
            (!case["sample_weight"].is_null()).then(|| json_to_array1(&case["sample_weight"]));
        let x_test = array2_with_nan(&case["X_test"]);
        // An eval set: its data, and CatBoost's metrics and tree count.
        let ev = &case["eval"];
        let eval_data = (!ev.is_null()).then(|| {
            let w: Option<Array1<f64>> =
                (!ev["sample_weight"].is_null()).then(|| json_to_array1(&ev["sample_weight"]));
            (array2_with_nan(&ev["X"]), json_to_array1(&ev["y"]), w)
        });
        let eval_set = eval_data.as_ref().map(|(x, y, w)| CatBoostEvalSet {
            x: x.view(),
            y: y.view(),
            sample_weight: w.as_ref().map(|w| w.view()),
        });
        let mut check_eval = |metrics: &[f64], trees: usize| {
            if ev.is_null() {
                return;
            }
            let expected = json_to_array1(&ev["metrics"]);
            let e = worst(metrics.iter().copied(), expected.iter().copied());
            let n = ev["tree_count"].as_u64().unwrap() as usize;
            if metrics.len() != expected.len() || e > TOL || trees != n {
                failures.push(format!(
                    "{name}: eval {} vs {} metrics (worst {e:e}), {trees} vs {n} trees",
                    metrics.len(),
                    expected.len()
                ));
            }
        };
        let err = if case["kind"] == "regressor" {
            let p = &case["params"];
            let mut m = configure!(CatBoostRegressor::new(), p);
            if let Some(l) = p["loss_function"].as_str() {
                let (name, param) = l.split_once(':').unwrap_or((l, ""));
                let value = || param.split_once('=').unwrap().1.parse::<f64>().unwrap();
                m = m.with_loss_function(match name {
                    "MAE" => CatBoostRegressionLoss::Mae,
                    "Quantile" => CatBoostRegressionLoss::Quantile { alpha: value() },
                    "Expectile" => CatBoostRegressionLoss::Expectile { alpha: value() },
                    "Poisson" => CatBoostRegressionLoss::Poisson,
                    "Tweedie" => CatBoostRegressionLoss::Tweedie {
                        variance_power: value(),
                    },
                    "LogLinQuantile" => CatBoostRegressionLoss::LogLinQuantile { alpha: value() },
                    "Huber" => CatBoostRegressionLoss::Huber { delta: value() },
                    "LogCosh" => CatBoostRegressionLoss::LogCosh,
                    "Lq" => CatBoostRegressionLoss::Lq { q: value() },
                    "RMSEWithUncertainty" => CatBoostRegressionLoss::RmseWithUncertainty,
                    "QueryRMSE" => CatBoostRegressionLoss::QueryRmse,
                    "PairLogit" => CatBoostRegressionLoss::PairLogit,
                    "PairLogitPairwise" => CatBoostRegressionLoss::PairLogitPairwise,
                    "YetiRank" | "YetiRankPairwise" => {
                        let arg = |k: &str, d: f64| {
                            param
                                .split(';')
                                .filter_map(|kv| kv.split_once('='))
                                .find(|(key, _)| *key == k)
                                .map_or(d, |(_, v)| v.parse().unwrap())
                        };
                        let (permutations, decay) =
                            (arg("permutations", 10.0) as usize, arg("decay", 0.85));
                        if name == "YetiRank" {
                            CatBoostRegressionLoss::YetiRank {
                                permutations,
                                decay,
                            }
                        } else {
                            CatBoostRegressionLoss::YetiRankPairwise {
                                permutations,
                                decay,
                            }
                        }
                    }
                    other => panic!("unknown loss {other}"),
                });
            }
            let sizes = |v: &Value| -> Vec<usize> {
                v.as_array()
                    .unwrap()
                    .iter()
                    .map(|g| g.as_u64().unwrap() as usize)
                    .collect()
            };
            let w = sw.as_ref().map(|w| w.view());
            let text_of = |v: &Value| -> Vec<Vec<String>> {
                v.as_array()
                    .unwrap()
                    .iter()
                    .map(|c| {
                        c.as_array()
                            .unwrap()
                            .iter()
                            .map(|t| t.as_str().unwrap().to_string())
                            .collect()
                    })
                    .collect()
            };
            if let Some(e) = case.get("embeddings") {
                let m = m.with_embedding_options(embedding_options(&case["embedding_calcers"]));
                let test = embeddings_of(&case["embeddings_test"]);
                let pred = if let Some(t) = case.get("text") {
                    m.fit_with_text_and_embeddings(
                        x.view(),
                        &text_of(t),
                        &views(&embeddings_of(e)),
                        y.view(),
                        w,
                    )
                    .unwrap()
                    .predict_with_text_and_embeddings(
                        x_test.view(),
                        &text_of(&case["text_test"]),
                        &views(&test),
                    )
                    .unwrap()
                } else {
                    m.fit_with_embeddings(x.view(), &views(&embeddings_of(e)), y.view(), w)
                        .unwrap()
                        .predict_with_embeddings(x_test.view(), &views(&test))
                        .unwrap()
                };
                let err = worst(pred, json_to_array1(&case["predictions"]));
                report.push(format!("{name}: {err:e}"));
                if err > TOL {
                    failures.push(format!("{name}: worst {err:e}"));
                }
                continue;
            }
            if let Some(t) = case.get("text") {
                if let Some(o) = case.get("text_options") {
                    m = m.with_text_options(serde_json::from_value(o.clone()).unwrap());
                }
                let f = m.fit_with_text(x.view(), &text_of(t), y.view(), w).unwrap();
                let pred = f
                    .predict_with_text(x_test.view(), &text_of(&case["text_test"]))
                    .unwrap();
                let err = worst(pred, json_to_array1(&case["predictions"]));
                report.push(format!("{name}: {err:e}"));
                if err > TOL {
                    failures.push(format!("{name}: worst {err:e}"));
                }
                continue;
            }
            let f = match (eval_set, case.get("group")) {
                (Some(e), Some(g)) => m
                    .fit_with_groups_eval(x.view(), y.view(), w, &sizes(g), e, &sizes(&ev["group"]))
                    .unwrap(),
                (None, Some(g)) => m.fit_with_groups(x.view(), y.view(), w, &sizes(g)).unwrap(),
                (Some(e), None) => m.fit_with_eval(x.view(), y.view(), w, e).unwrap(),
                (None, None) => m.fit_weighted(&x, &y, sw.as_ref()).unwrap(),
            };
            check_eval(f.eval_metrics(), f.n_estimators());
            if p["loss_function"] == "RMSEWithUncertainty" {
                worst(
                    f.predict_uncertainty(&x_test).unwrap(),
                    array2_with_nan(&case["predictions"]),
                )
            } else {
                worst(
                    f.predict(&x_test).unwrap(),
                    json_to_array1(&case["predictions"]),
                )
            }
        } else {
            let p = &case["params"];
            let mut m = configure!(CatBoostClassifier::new(), p);
            if let Some(e) = case.get("embeddings") {
                let m = m.with_embedding_options(embedding_options(&case["embedding_calcers"]));
                let test = embeddings_of(&case["embeddings_test"]);
                let w = sw.as_ref().map(|w| w.view());
                let proba = if let Some(t) = case.get("text") {
                    let text_of = |v: &Value| -> Vec<Vec<String>> {
                        serde_json::from_value(v.clone()).unwrap()
                    };
                    m.fit_with_text_and_embeddings(
                        x.view(),
                        &text_of(t),
                        &views(&embeddings_of(e)),
                        y.view(),
                        w,
                    )
                    .unwrap()
                    .predict_proba_with_text_and_embeddings(
                        x_test.view(),
                        &text_of(&case["text_test"]),
                        &views(&test),
                    )
                    .unwrap()
                } else {
                    m.fit_with_embeddings(x.view(), &views(&embeddings_of(e)), y.view(), w)
                        .unwrap()
                        .predict_proba_with_embeddings(x_test.view(), &views(&test))
                        .unwrap()
                };
                let err = worst(proba, array2_with_nan(&case["predictions"]));
                report.push(format!("{name}: {err:e}"));
                if err > TOL {
                    failures.push(format!("{name}: worst {err:e}"));
                }
                continue;
            }
            if let Some(t) = case.get("text") {
                if let Some(o) = case.get("text_options") {
                    m = m.with_text_options(serde_json::from_value(o.clone()).unwrap());
                }
                let text_of = |v: &Value| -> Vec<Vec<String>> {
                    v.as_array()
                        .unwrap()
                        .iter()
                        .map(|c| {
                            c.as_array()
                                .unwrap()
                                .iter()
                                .map(|t| t.as_str().unwrap().to_string())
                                .collect()
                        })
                        .collect()
                };
                let f = m
                    .fit_with_text(
                        x.view(),
                        &text_of(t),
                        y.view(),
                        sw.as_ref().map(|w| w.view()),
                    )
                    .unwrap();
                let proba = f
                    .predict_proba_with_text(x_test.view(), &text_of(&case["text_test"]))
                    .unwrap();
                let err = worst(proba, array2_with_nan(&case["predictions"]));
                report.push(format!("{name}: {err:e}"));
                if err > TOL {
                    failures.push(format!("{name}: worst {err:e}"));
                }
                continue;
            }
            if p["loss_function"] == "CrossEntropy" {
                m = m.with_loss_function(CatBoostClassifierLoss::CrossEntropy);
            }
            if p["loss_function"] == "MultiClassOneVsAll" {
                m = m.with_loss_function(CatBoostClassifierLoss::MultiClassOneVsAll);
            }
            if let Some(w) = p["class_weights"].as_array() {
                m = m.with_class_weights(w.iter().map(|v| v.as_f64().unwrap()).collect());
            }
            match p["auto_class_weights"].as_str() {
                Some("Balanced") => {
                    m = m.with_auto_class_weights(CatBoostAutoClassWeights::Balanced)
                }
                Some("SqrtBalanced") => {
                    m = m.with_auto_class_weights(CatBoostAutoClassWeights::SqrtBalanced)
                }
                _ => {}
            }
            let f = match eval_set {
                Some(e) => m
                    .fit_with_eval(x.view(), y.view(), sw.as_ref().map(|w| w.view()), e)
                    .unwrap(),
                None => m.fit_weighted(&x, &y, sw.as_ref()).unwrap(),
            };
            check_eval(f.eval_metrics(), f.n_estimators());
            worst(
                f.predict_proba(&x_test).unwrap(),
                array2_with_nan(&case["predictions"]),
            )
        };
        report.push(format!("{name}: {err:e}"));
        if err > TOL {
            failures.push(format!("{name}: {err:e}"));
        }
    }
    eprintln!("{report:#?}");
    assert!(failures.is_empty(), "{failures:#?}");
}
