use anofox_ml_core::splitters::group_k_fold;
use anofox_ml_registry::Out;
use anofox_ml_tune::{tune, tune_with_splits, Sampler, TuneOptions};
use ndarray::{Array1, Array2};
use serde_json::json;

/// y is a step function of x0 with 8 levels: a tree needs depth >= 3 to
/// fit it, and deeper trees gain nothing.
fn step_data(n: usize) -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_fn((n, 2), |(i, j)| {
        if j == 0 {
            (i % 32) as f64
        } else {
            ((i * 7) % 5) as f64
        }
    });
    let y = Array1::from_shape_fn(n, |i| (x[[i, 0]] / 4.0).floor());
    (x, y)
}

fn binary_data(n: usize) -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_fn((n, 2), |(i, j)| ((i * (3 + j * 4)) % 17) as f64);
    let y = Array1::from_shape_fn(n, |i| {
        if x[[i, 0]] + 0.5 * x[[i, 1]] > 10.0 {
            1.0
        } else {
            0.0
        }
    });
    (x, y)
}

fn options(n_trials: usize) -> TuneOptions {
    TuneOptions {
        n_trials,
        cv: 4,
        ..Default::default()
    }
}

#[test]
fn finds_the_depth_a_step_function_needs() {
    let (x, y) = step_data(128);
    let result = tune(
        "DecisionTreeRegressor",
        &json!({"max_depth": {"int": [1, 10]}}),
        x.view(),
        y.view(),
        &options(20),
    )
    .unwrap();
    assert_eq!(result.metric, "r2");
    assert!(result.maximize);
    assert!(
        result.best_params["max_depth"].as_i64().unwrap() >= 3,
        "{result:?}"
    );
    assert!(result.best_score > 0.99, "{result:?}");
    // The best trial is the one with the highest mean score.
    let best = result
        .trials
        .iter()
        .filter_map(|t| t.score)
        .fold(f64::MIN, f64::max);
    assert_eq!(result.best_score, best);
    assert_eq!(result.trials[result.best_trial].score, Some(best));
    // Unsearched defaults are kept in best_params.
    assert_eq!(result.best_params["min_samples_leaf"], 1);
}

#[test]
fn same_seed_same_search_and_refit_matches_best_params() {
    let (x, y) = step_data(96);
    let space = json!({
        "max_depth": {"int": [1, 8]},
        "min_samples_leaf": {"int": [1, 10]},
        "ccp_alpha": {"float": [1e-4, 1.0], "log": true},
    });
    let run = |seed, sampler| {
        let o = TuneOptions {
            seed,
            sampler,
            n_startup_trials: 5,
            ..options(15)
        };
        tune("DecisionTreeRegressor", &space, x.view(), y.view(), &o).unwrap()
    };
    let (a, b, c) = (
        run(1, Sampler::Tpe),
        run(1, Sampler::Tpe),
        run(2, Sampler::Tpe),
    );
    assert_eq!(a.trials, b.trials);
    assert_ne!(a.trials, c.trials);
    let r = run(1, Sampler::Random);
    assert_ne!(a.trials[5..], r.trials[5..]);
    for t in &a.trials {
        let p = &t.params;
        let d = p["max_depth"].as_i64().unwrap();
        let alpha = p["ccp_alpha"].as_f64().unwrap();
        assert!((1..=8).contains(&d) && (1e-4..=1.0).contains(&alpha), "{p}");
    }

    // The returned model is the estimator fitted with best_params on all rows.
    let mut expected = anofox_ml_registry::make("DecisionTreeRegressor", &a.best_params).unwrap();
    expected.fit(x.view(), Some(y.view()), None).unwrap();
    let (Out::A1(p1), Out::A1(p2)) = (
        a.model.as_ref().unwrap().predict(x.view()).unwrap(),
        expected.predict(x.view()).unwrap(),
    ) else {
        panic!("1-D predictions expected")
    };
    assert_eq!(p1, p2);
}

#[test]
fn loss_metrics_are_minimised() {
    let (x, y) = step_data(96);
    let o = TuneOptions {
        metric: Some("rmse".into()),
        ..options(12)
    };
    let r = tune(
        "DecisionTreeRegressor",
        &json!({"max_depth": {"int": [1, 6]}}),
        x.view(),
        y.view(),
        &o,
    )
    .unwrap();
    assert!(!r.maximize);
    let best = r
        .trials
        .iter()
        .filter_map(|t| t.score)
        .fold(f64::MAX, f64::min);
    assert_eq!(r.best_score, best);
    assert!(r.best_score < 0.1, "{r:?}");
}

#[test]
fn probability_metrics_score_classifiers() {
    let (x, y) = binary_data(120);
    for (metric, good) in [("log_loss", 0.5), ("roc_auc", 0.9)] {
        let o = TuneOptions {
            metric: Some(metric.into()),
            params: json!({"n_estimators": 20}),
            ..options(6)
        };
        let r = tune(
            "RandomForestClassifier",
            &json!({"max_depth": {"int": [2, 8]}}),
            x.view(),
            y.view(),
            &o,
        )
        .unwrap();
        assert!(r.trials.iter().all(|t| t.error.is_none()), "{:?}", r.trials);
        if r.maximize {
            assert!(r.best_score > good, "{metric}: {r:?}");
        } else {
            assert!(r.best_score < good, "{metric}: {r:?}");
        }
    }
    // Default for a classifier: accuracy.
    let r = tune(
        "KnnClassifier",
        &json!({"n_neighbors": {"int": [1, 15]}, "weights": {"categorical": ["Uniform", "Distance"]}}),
        x.view(),
        y.view(),
        &options(8),
    )
    .unwrap();
    assert_eq!(r.metric, "accuracy");
}

#[test]
fn failing_trials_are_recorded_not_fatal() {
    let (x, y) = step_data(64);
    // "NoSuchCriterion" can't be deserialised into the config.
    let r = tune(
        "RandomForestRegressor",
        &json!({"criterion": {"categorical": ["Mse", "NoSuchCriterion"]}}),
        x.view(),
        y.view(),
        &TuneOptions {
            sampler: Sampler::Random,
            params: json!({"n_estimators": 5}),
            ..options(10)
        },
    )
    .unwrap();
    let failed: Vec<_> = r.trials.iter().filter(|t| t.score.is_none()).collect();
    assert!(!failed.is_empty() && failed.len() < 10);
    assert!(failed
        .iter()
        .all(|t| t.error.as_deref().unwrap().contains("NoSuchCriterion")));
    assert_eq!(r.best_params["criterion"], "Mse");

    let e = tune(
        "RandomForestRegressor",
        &json!({"criterion": {"categorical": ["NoSuchCriterion"]}}),
        x.view(),
        y.view(),
        &options(3),
    )
    .unwrap_err();
    assert!(e.to_string().contains("every trial failed"), "{e}");
}

#[test]
fn invalid_requests_fail_up_front() {
    let (x, y) = step_data(32);
    let space = json!({"max_depth": {"int": [1, 4]}});
    let cases = [
        (
            "DecisionTreeRegressor",
            json!({"depth": {"int": [1, 4]}}),
            options(3),
            "no parameter 'depth'",
        ),
        (
            "DecisionTreeRegressor",
            space.clone(),
            TuneOptions {
                params: json!({"max_depth": 3}),
                ..options(3)
            },
            "both fixed",
        ),
        (
            "DecisionTreeRegressor",
            space.clone(),
            TuneOptions {
                metric: Some("auc".into()),
                ..options(3)
            },
            "unknown metric",
        ),
        (
            "DecisionTreeRegressor",
            space.clone(),
            options(0),
            "n_trials",
        ),
        ("NoSuchModel", space.clone(), options(3), "NoSuchModel"),
    ];
    for (kind, space, o, msg) in cases {
        let e = tune(kind, &space, x.view(), y.view(), &o)
            .unwrap_err()
            .to_string();
        assert!(e.contains(msg), "{msg}: {e}");
    }
}

#[test]
fn custom_splits_and_sample_weights() {
    let (x, y) = step_data(96);
    let groups = Array1::from_shape_fn(96, |i| i % 6);
    let splits = group_k_fold(&groups, 3).unwrap();
    let w = Array1::from_shape_fn(96, |i| 1.0 + (i % 3) as f64);
    let r = tune_with_splits(
        "DecisionTreeRegressor",
        &json!({"max_depth": {"int": [1, 6]}}),
        x.view(),
        y.view(),
        Some(w.view()),
        &splits,
        &options(8),
    )
    .unwrap();
    assert!(r.trials.iter().all(|t| t.fold_scores.len() == 3));
    // KnnRegressor does not take weights: every trial fails with that error.
    let e = tune_with_splits(
        "KnnRegressor",
        &json!({"n_neighbors": {"int": [1, 5]}}),
        x.view(),
        y.view(),
        Some(w.view()),
        &splits,
        &options(2),
    )
    .unwrap_err();
    assert!(e.to_string().contains("sample_weight"), "{e}");
}

#[test]
fn result_serializes_without_the_model() {
    let (x, y) = step_data(48);
    let r = tune(
        "DecisionTreeRegressor",
        &json!({"max_depth": {"int": [1, 4]}}),
        x.view(),
        y.view(),
        &TuneOptions {
            refit: false,
            ..options(3)
        },
    )
    .unwrap();
    assert!(r.model.is_none());
    let v = serde_json::to_value(&r).unwrap();
    assert_eq!(v["trials"].as_array().unwrap().len(), 3);
    assert!(v.get("model").is_none());
    // Options round-trip through JSON with defaults for missing fields.
    let o: TuneOptions =
        serde_json::from_value(json!({"n_trials": 7, "sampler": "random"})).unwrap();
    assert_eq!((o.n_trials, o.sampler, o.cv), (7, Sampler::Random, 5));
    assert!(serde_json::from_value::<TuneOptions>(json!({"trials": 7})).is_err());
}

/// Whether `a` is at least as good as `b` everywhere and better somewhere.
fn dominates(a: &[f64], b: &[f64], maximize: &[bool]) -> bool {
    let ge = |i: usize| {
        if maximize[i] {
            a[i] >= b[i]
        } else {
            a[i] <= b[i]
        }
    };
    let gt = |i: usize| {
        if maximize[i] {
            a[i] > b[i]
        } else {
            a[i] < b[i]
        }
    };
    (0..a.len()).all(ge) && (0..a.len()).any(gt)
}

#[test]
fn multi_objective_reports_the_pareto_front() {
    let (x, y) = step_data(128);
    let space = json!({"max_depth": {"int": [1, 10]}, "min_samples_leaf": {"int": [1, 20]}});
    for sampler in [Sampler::Tpe, Sampler::Nsgaii] {
        let o = TuneOptions {
            sampler,
            objectives: vec!["model_size".into(), "mae".into()],
            population_size: 6,
            ..options(18)
        };
        let r = tune("DecisionTreeRegressor", &space, x.view(), y.view(), &o).unwrap();
        assert_eq!(
            r.objectives,
            [
                ("r2".to_string(), true),
                ("model_size".to_string(), false),
                ("mae".to_string(), false)
            ]
        );
        let maximize: Vec<bool> = r.objectives.iter().map(|o| o.1).collect();
        assert!(r
            .trials
            .iter()
            .all(|t| t.values.len() == 3 && t.values[0] == t.score.unwrap()));
        // Brute force: the front is exactly the non-dominated trials.
        let expected: Vec<usize> = r
            .trials
            .iter()
            .filter(|t| {
                !r.trials
                    .iter()
                    .any(|o| dominates(&o.values, &t.values, &maximize))
            })
            .map(|t| t.number)
            .collect();
        assert_eq!(r.pareto_front, expected, "{sampler:?}");
        // Bigger trees fit better: the front trades size for accuracy, and
        // best_* still refer to the primary metric.
        assert!(
            r.pareto_front.len() >= 2,
            "{sampler:?}: {:?}",
            r.pareto_front
        );
        assert!(r.pareto_front.contains(&r.best_trial));
        let sizes: Vec<f64> = r
            .pareto_front
            .iter()
            .map(|&n| r.trials[n].values[1])
            .collect();
        assert!(sizes.iter().any(|&s| s != sizes[0]));
    }
}

#[test]
fn nsgaii_is_seeded_and_fit_time_is_measured() {
    let (x, y) = step_data(64);
    let space = json!({"max_depth": {"int": [1, 8]}});
    let run = |seed| {
        let o = TuneOptions {
            sampler: Sampler::Nsgaii,
            population_size: 4,
            objectives: vec!["model_size".into()],
            seed,
            ..options(10)
        };
        tune("DecisionTreeRegressor", &space, x.view(), y.view(), &o).unwrap()
    };
    let (a, b) = (run(3), run(3));
    let params = |r: &anofox_ml_tune::TuneResult| {
        r.trials
            .iter()
            .map(|t| t.params.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(params(&a), params(&b));

    let o = TuneOptions {
        objectives: vec!["fit_time".into()],
        ..options(3)
    };
    let r = tune("DecisionTreeRegressor", &space, x.view(), y.view(), &o).unwrap();
    assert!(r
        .trials
        .iter()
        .all(|t| t.values[1] > 0.0 && t.values[1] < 10.0));
}

#[test]
fn invalid_objectives_fail_up_front() {
    let (x, y) = step_data(32);
    let space = json!({"max_depth": {"int": [1, 4]}});
    for (objectives, msg) in [
        (vec!["r2".to_string()], "listed twice"),
        (vec!["speed".to_string()], "fit_time / model_size"),
    ] {
        let o = TuneOptions {
            objectives,
            ..options(2)
        };
        let e = tune("DecisionTreeRegressor", &space, x.view(), y.view(), &o)
            .unwrap_err()
            .to_string();
        assert!(e.contains(msg), "{msg}: {e}");
    }
    let o = TuneOptions {
        sampler: Sampler::Nsgaii,
        population_size: 1,
        ..options(2)
    };
    let e = tune("DecisionTreeRegressor", &space, x.view(), y.view(), &o)
        .unwrap_err()
        .to_string();
    assert!(e.contains("population_size"), "{e}");
}

#[test]
fn mapped_search_spaces_translate_each_trial() {
    use anofox_ml_core::RustMlError;
    use serde_json::{Map, Value};
    let (x, y) = step_data(96);
    let splits = anofox_ml_core::utils::default_cv_splits(x.view(), y.view(), 4, 0).unwrap();
    // "depth" (outside vocabulary) -> max_depth; depth 2 is rejected by the map.
    let map = |s: &Map<String, Value>| -> anofox_ml_core::Result<Map<String, Value>> {
        let d = s["depth"].as_i64().unwrap();
        if d == 2 {
            return Err(RustMlError::InvalidParameter(
                "depth 2 is not allowed".into(),
            ));
        }
        Ok(Map::from_iter([("max_depth".to_string(), json!(d))]))
    };
    let r = anofox_ml_tune::tune_with_splits_mapped(
        "DecisionTreeRegressor",
        &json!({"depth": {"int": [1, 6]}}),
        x.view(),
        y.view(),
        None,
        &splits,
        &TuneOptions {
            sampler: Sampler::Random,
            ..options(12)
        },
        &map,
    )
    .unwrap();
    assert!(r.trials.iter().all(|t| t.params.get("depth").is_some()));
    assert_eq!(
        r.best_params["max_depth"],
        r.trials[r.best_trial].params["depth"]
    );
    assert_eq!(r.best_params["min_samples_leaf"], 1, "defaults merged");
    for t in &r.trials {
        assert_eq!(
            t.params["depth"] == 2,
            t.error.as_deref() == Some("Invalid parameter: depth 2 is not allowed"),
            "{t:?}"
        );
    }
}
