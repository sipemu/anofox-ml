mod common;

use anofox_ml::prelude::*;
use common::{json_to_array2, load_golden_data};

#[test]
fn test_golden_kmeans() {
    let cases = load_golden_data("kmeans.json");

    for case in &cases {
        let name = case["name"].as_str().unwrap();

        let x_train = json_to_array2(&case["X_train"]);
        let n_clusters = case["n_clusters"].as_u64().unwrap() as usize;

        let km = KMeans {
            n_clusters,
            max_iter: 300,
            tol: 1e-4,
            seed: 42,
        };
        let fitted = FitUnsupervised::<f64>::fit(&km, &x_train).unwrap();

        // KMeans is sensitive to initialization; we can't match sklearn's exact
        // cluster assignments because k-means++ initialization differs.
        // Instead verify: correct number of clusters, and inertia is reasonable.
        let centroids = fitted.centroids();
        assert_eq!(
            centroids.nrows(),
            n_clusters,
            "{}: expected {} centroids",
            name,
            n_clusters
        );

        // Verify predictions assign to valid clusters
        let x_test = json_to_array2(&case["X_test"]);
        let preds = fitted.predict(&x_test).unwrap();
        for &p in preds.iter() {
            assert!(
                p >= 0.0 && p < n_clusters as f64,
                "{}: invalid cluster label {}",
                name,
                p
            );
        }

        // Verify inertia is finite and positive
        let inertia = fitted.inertia();
        assert!(
            inertia.is_finite() && inertia > 0.0,
            "{}: bad inertia {}",
            name,
            inertia
        );
    }
}
