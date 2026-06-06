mod common;

use anofox_ml::prelude::*;
use common::{json_to_array2, load_golden_data};

#[test]
fn test_golden_dbscan() {
    let cases = load_golden_data("dbscan.json");

    for case in &cases {
        let name = case["name"].as_str().unwrap();
        let x = json_to_array2(&case["X"]);
        let eps = case["eps"].as_f64().unwrap();
        let min_samples = case["min_samples"].as_u64().unwrap() as usize;
        let expected_n_clusters = case["n_clusters"].as_u64().unwrap() as usize;
        let expected_n_noise = case["n_noise"].as_u64().unwrap() as usize;

        let dbscan = Dbscan::new(eps, min_samples);
        let fitted = FitUnsupervised::<f64>::fit(&dbscan, &x).unwrap();

        let labels = fitted.labels();
        let n_clusters = fitted.n_clusters();

        // Verify number of clusters matches expected
        assert_eq!(
            n_clusters, expected_n_clusters,
            "{}: expected {} clusters, got {}",
            name, expected_n_clusters, n_clusters
        );

        // Count noise points (label == -1.0)
        let n_noise = labels.iter().filter(|&&l| l < 0.0).count();
        assert_eq!(
            n_noise, expected_n_noise,
            "{}: expected {} noise points, got {}",
            name, expected_n_noise, n_noise
        );

        // Labels should be in range -1..n_clusters
        for (i, &label) in labels.iter().enumerate() {
            let l = label as i64;
            assert!(
                l >= -1 && l < n_clusters as i64,
                "{}: label {} at index {} is out of range [-1, {})",
                name,
                l,
                i,
                n_clusters
            );
        }

        // Noise + clustered should equal total
        let n_clustered = labels.iter().filter(|&&l| l >= 0.0).count();
        assert_eq!(
            n_noise + n_clustered,
            labels.len(),
            "{}: noise ({}) + clustered ({}) != total ({})",
            name,
            n_noise,
            n_clustered,
            labels.len()
        );
    }
}
