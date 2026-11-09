//! ExtraTreeRegressor / ExtraTreeClassifier against scikit-learn 1.8.0
//! (`gen_extra_tree.py`): same leaves, same predictions.

mod common;

use anofox_ml_core::{FitUnsupervisedWeighted, Transform};
use anofox_ml_ensemble::RandomTreesEmbedding;
use anofox_ml_trees::{ExtraTreeClassifier, ExtraTreeRegressor, MaxFeatures, SkCriterion};
use common::{json_to_array1, json_to_array2, load_golden_file};

fn params(
    p: &serde_json::Value,
    base: anofox_ml_trees::ExtraTreeParams,
) -> anofox_ml_trees::ExtraTreeParams {
    anofox_ml_trees::ExtraTreeParams {
        max_depth: p["max_depth"].as_u64().map(|v| v as usize),
        min_samples_split: p["min_samples_split"].as_u64().unwrap_or(2) as usize,
        min_samples_leaf: p["min_samples_leaf"].as_u64().unwrap_or(1) as usize,
        max_features: match p.get("max_features") {
            Some(v) if v.is_null() => None,
            Some(v) => Some(MaxFeatures::Fixed(v.as_u64().unwrap() as usize)),
            None => base.max_features,
        },
        seed: p["random_state"].as_u64().unwrap() as u32,
        ..base
    }
}

fn usizes(v: &serde_json::Value) -> Vec<usize> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect()
}

#[test]
fn extra_trees_match_scikit_learn() {
    let d = load_golden_file("extra_tree.json");
    let (x, y, labels, w) = (
        json_to_array2(&d["X"]),
        json_to_array1(&d["y"]),
        json_to_array1(&d["labels"]),
        json_to_array1(&d["w"]),
    );
    let probe = json_to_array2(&d["probe"]);
    for case in d["regressor"].as_array().unwrap() {
        let p = &case["params"];
        let m = ExtraTreeRegressor {
            params: params(p, Default::default()),
        };
        let weights = case["weighted"].as_bool().unwrap().then(|| w.view());
        let f = m.fit(x.view(), y.view(), weights).unwrap();
        assert_eq!(
            f.tree.nodes.len() as u64,
            case["node_count"].as_u64().unwrap(),
            "{p}"
        );
        assert_eq!(
            f.apply(probe.view()).unwrap().to_vec(),
            usizes(&case["apply"]),
            "{p}"
        );
        for (o, e) in f
            .predict(probe.view())
            .unwrap()
            .iter()
            .zip(&json_to_array1(&case["predict"]))
        {
            assert!((o - e).abs() < 1e-10, "{p}: {o} vs {e}");
        }
    }
    for case in d["classifier"].as_array().unwrap() {
        let p = &case["params"];
        let m = ExtraTreeClassifier {
            criterion: if p["criterion"] == "entropy" {
                SkCriterion::Entropy
            } else {
                SkCriterion::Gini
            },
            params: params(p, ExtraTreeClassifier::default().params),
        };
        let weights = case["weighted"].as_bool().unwrap().then(|| w.view());
        let f = m.fit(x.view(), labels.view(), weights).unwrap();
        assert_eq!(
            f.tree.nodes.len() as u64,
            case["node_count"].as_u64().unwrap(),
            "{p}"
        );
        assert_eq!(
            f.apply(probe.view()).unwrap().to_vec(),
            usizes(&case["apply"]),
            "{p}"
        );
        for (o, e) in f
            .predict_proba(probe.view())
            .unwrap()
            .iter()
            .zip(&json_to_array2(&case["proba"]))
        {
            assert!((o - e).abs() < 1e-12, "{p}: {o} vs {e}");
        }
    }
}

#[test]
fn random_trees_embedding_matches_scikit_learn() {
    let d = load_golden_file("extra_tree.json");
    let (x, probe) = (json_to_array2(&d["X"]), json_to_array2(&d["probe"]));
    for case in d["embedding"].as_array().unwrap() {
        let p = &case["params"];
        let m = RandomTreesEmbedding {
            n_estimators: p["n_estimators"].as_u64().unwrap() as usize,
            max_depth: Some(p["max_depth"].as_u64().unwrap_or(5) as usize),
            min_samples_leaf: p["min_samples_leaf"].as_u64().unwrap_or(1) as usize,
            seed: p["random_state"].as_u64().unwrap() as u32,
            ..Default::default()
        };
        let f = m.fit_unsupervised_weighted_view(x.view(), None).unwrap();
        let apply = f.apply(probe.view()).unwrap();
        let want = json_to_array2(&case["apply"]);
        for (o, e) in apply.iter().zip(&want) {
            assert_eq!(*o as f64, *e, "{p}: apply");
        }
        assert_eq!(
            f.transform(&probe).unwrap(),
            json_to_array2(&case["transform"]),
            "{p}"
        );
    }
}
