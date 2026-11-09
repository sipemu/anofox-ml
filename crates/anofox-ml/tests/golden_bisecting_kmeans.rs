//! BisectingKMeans against scikit-learn 1.8.0 (`gen_bisecting_kmeans.py`):
//! seeded fits draw their inits from numpy's RandomState.

mod common;

use anofox_ml_cluster::{BisectingInit, BisectingKMeans, BisectingStrategy};
use anofox_ml_core::FitUnsupervisedWeighted;
use common::{json_to_array1, json_to_array2, load_golden_file};

#[test]
fn bisecting_kmeans_matches_scikit_learn() {
    let data = load_golden_file("bisecting_kmeans.json");
    let (x, w) = (json_to_array2(&data["X"]), json_to_array1(&data["w"]));
    for case in data["cases"].as_array().unwrap() {
        let p = &case["params"];
        let m = BisectingKMeans {
            n_clusters: p["n_clusters"].as_u64().unwrap() as usize,
            init: if p["init"] == "k-means++" {
                BisectingInit::KMeansPlusPlus
            } else {
                BisectingInit::Random
            },
            n_init: p["n_init"].as_u64().unwrap_or(1) as usize,
            bisecting_strategy: if p["bisecting_strategy"] == "largest_cluster" {
                BisectingStrategy::LargestCluster
            } else {
                BisectingStrategy::BiggestInertia
            },
            seed: p["random_state"].as_u64().unwrap() as u32,
            ..Default::default()
        };
        let weights = case["weighted"].as_bool().unwrap().then(|| w.view());
        let f = m.fit_unsupervised_weighted_view(x.view(), weights).unwrap();
        let labels: Vec<usize> = case["labels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(f.labels.to_vec(), labels, "{p}: labels");
        for (o, e) in f
            .cluster_centers
            .iter()
            .zip(&json_to_array2(&case["centers"]))
        {
            assert!((o - e).abs() < 1e-10, "{p}: centre {o} vs {e}");
        }
        let inertia = case["inertia"].as_f64().unwrap();
        assert!(
            (f.inertia - inertia).abs() < 1e-9 * inertia,
            "{p}: inertia {} vs {inertia}",
            f.inertia
        );
        let probe = &x.slice(ndarray::s![..;7, ..]) + 0.3;
        let pred: Vec<usize> = case["predict"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(
            f.predict(probe.view()).unwrap().to_vec(),
            pred,
            "{p}: predict"
        );
    }
}
