//! Behavioral parity test for MiniBatchKMeans against sklearn 1.8.0.
//!
//! Different RNG / sampling means exact centroid match isn't possible. We
//! compare centroid *sets* after greedy nearest-neighbor matching.

mod common;

use common::{json_to_array2, load_golden_data};
use ndarray::Array1;
use rustml::core::FitUnsupervised;
use rustml_cluster::MiniBatchKMeans;

fn matched_centroid_distance(ours: &ndarray::Array2<f64>, theirs: &ndarray::Array2<f64>) -> f64 {
    // Greedy matching: for each rustml centroid, find closest sklearn centroid
    // (without reuse). Return the maximum match distance.
    let n = ours.nrows();
    let mut used = vec![false; n];
    let mut max_d: f64 = 0.0;
    for i in 0..n {
        let mut best = f64::INFINITY;
        let mut best_j = 0usize;
        for j in 0..n {
            if used[j] {
                continue;
            }
            let d: f64 = (0..ours.ncols())
                .map(|k| (ours[[i, k]] - theirs[[j, k]]).powi(2))
                .sum::<f64>()
                .sqrt();
            if d < best {
                best = d;
                best_j = j;
            }
        }
        used[best_j] = true;
        if best > max_d {
            max_d = best;
        }
    }
    max_d
}

#[test]
fn test_mini_batch_kmeans_centroid_match() {
    let cases = load_golden_data("mini_batch_kmeans.json");
    let case = &cases[0];

    let x = json_to_array2(&case["X"]);
    let theirs = json_to_array2(&case["sklearn_centroids"]);
    let n_clusters = case["n_clusters"].as_u64().unwrap() as usize;
    let batch_size = case["batch_size"].as_u64().unwrap() as usize;

    let model = MiniBatchKMeans::new(n_clusters)
        .with_batch_size(batch_size)
        .with_max_iter(500)
        .with_seed(0);
    let fitted: rustml_cluster::FittedMiniBatchKMeans<f64> =
        FitUnsupervised::fit(&model, &x).unwrap();
    let ours = fitted.centroids().clone();

    let d = matched_centroid_distance(&ours, &theirs);
    // make_blobs cluster_std=0.6 — anything under 0.5 means we're in the
    // same basin of attraction.
    assert!(d < 0.5, "max matched centroid distance = {d}");

    // Sanity: each predicted cluster has at least some points.
    let preds = rustml::core::Predict::predict(&fitted, &x).unwrap();
    let mut counts = vec![0usize; n_clusters];
    for &p in preds.iter() {
        let k = p as usize;
        counts[k] += 1;
    }
    let _ = Array1::from_vec(counts.iter().map(|c| *c as f64).collect::<Vec<_>>());
    for &c in &counts {
        assert!(c > 0, "empty cluster: counts={:?}", counts);
    }
}
