//! HDBSCAN vs sklearn 1.8 (`test_harness/generators/gen_hdbscan.py`).
//!
//! `labels` / `probabilities` come from sklearn's own Cython pipeline with a
//! stable MST-edge sort (the only remaining source of difference to the
//! public sklearn API is its unstable `argsort` on tied edge weights), so
//! they must match exactly. `labels_public` is compared by adjusted Rand
//! index.

mod common;

use anofox_ml::cluster::{ClusterSelectionMethod, Hdbscan};
use anofox_ml_core::FitUnsupervised;
use common::{json_to_array1, json_to_array2, load_golden_data};
use std::collections::BTreeMap;

fn comb2(x: usize) -> f64 {
    (x as f64) * (x as f64 - 1.0) / 2.0
}

/// Adjusted Rand index (noise `-1` treated as its own label, like sklearn).
fn ari(a: &[i64], b: &[i64]) -> f64 {
    let n = a.len();
    let mut cont: BTreeMap<(i64, i64), usize> = BTreeMap::new();
    let mut ra: BTreeMap<i64, usize> = BTreeMap::new();
    let mut rb: BTreeMap<i64, usize> = BTreeMap::new();
    for (&x, &y) in a.iter().zip(b) {
        *cont.entry((x, y)).or_default() += 1;
        *ra.entry(x).or_default() += 1;
        *rb.entry(y).or_default() += 1;
    }
    let sum_ij: f64 = cont.values().map(|&c| comb2(c)).sum();
    let sum_a: f64 = ra.values().map(|&c| comb2(c)).sum();
    let sum_b: f64 = rb.values().map(|&c| comb2(c)).sum();
    let expected = sum_a * sum_b / comb2(n);
    let max = 0.5 * (sum_a + sum_b);
    if (max - expected).abs() < 1e-12 {
        return 1.0;
    }
    (sum_ij - expected) / (max - expected)
}

#[test]
fn test_golden_hdbscan() {
    let cases = load_golden_data("hdbscan.json");
    assert!(!cases.is_empty());
    for case in &cases {
        let name = case["name"].as_str().unwrap();
        let x = json_to_array2(&case["X"]);
        let method = match case["cluster_selection_method"].as_str().unwrap() {
            "leaf" => ClusterSelectionMethod::Leaf,
            _ => ClusterSelectionMethod::Eom,
        };
        let model = Hdbscan::new(
            case["min_samples"].as_u64().unwrap() as usize,
            case["min_cluster_size"].as_u64().unwrap() as usize,
        )
        .with_cluster_selection_method(method)
        .with_allow_single_cluster(case["allow_single_cluster"].as_bool().unwrap());
        let fitted = model.fit(&x).unwrap();

        let ours: Vec<i64> = fitted.labels.iter().map(|&l| l as i64).collect();
        let expected: Vec<i64> = json_to_array1(&case["labels"])
            .iter()
            .map(|&l| l as i64)
            .collect();
        let public: Vec<i64> = json_to_array1(&case["labels_public"])
            .iter()
            .map(|&l| l as i64)
            .collect();

        assert_eq!(
            fitted.n_clusters,
            case["n_clusters"].as_u64().unwrap() as usize,
            "{name}: n_clusters"
        );
        assert_eq!(ours, expected, "{name}: labels differ from sklearn");
        let probs = json_to_array1(&case["probabilities"]);
        for (i, (&p, &q)) in fitted.probabilities.iter().zip(probs.iter()).enumerate() {
            assert!((p - q).abs() < 1e-9, "{name}: probability[{i}] {p} vs {q}");
        }
        let a = ari(&ours, &public);
        assert!(a > 0.97, "{name}: ARI vs public sklearn API = {a}");
    }
}
