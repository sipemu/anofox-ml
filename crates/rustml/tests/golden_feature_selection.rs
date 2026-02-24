mod common;

use common::{json_to_array2, load_golden_data};
use rustml::prelude::*;

#[test]
fn test_golden_variance_threshold() {
    let cases = load_golden_data("feature_selection.json");

    for case in &cases {
        let name = case["name"].as_str().unwrap();
        if case["algorithm"].as_str().unwrap() != "VarianceThreshold" {
            continue;
        }

        let x = json_to_array2(&case["X"]);
        let threshold = case["threshold"].as_f64().unwrap();
        let expected_selected: Vec<usize> = case["selected_indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let expected_shape: Vec<usize> = case["X_transformed_shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();

        let vt = VarianceThreshold::new(threshold);
        let fitted = FitUnsupervised::<f64>::fit(&vt, &x).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        // Verify selected indices match
        let actual_selected = fitted.selected_indices();
        assert_eq!(
            actual_selected, &expected_selected[..],
            "{}: selected indices mismatch",
            name
        );

        // Verify transformed shape matches
        assert_eq!(
            x_transformed.nrows(), expected_shape[0],
            "{}: row count mismatch", name
        );
        assert_eq!(
            x_transformed.ncols(), expected_shape[1],
            "{}: col count mismatch", name
        );

        // Variances should all be non-negative
        let variances = fitted.variances();
        for (i, &v) in variances.iter().enumerate() {
            assert!(
                v >= 0.0,
                "{}: variance at feature {} is negative: {}",
                name, i, v
            );
        }
    }
}

#[test]
fn test_golden_mutual_information() {
    let cases = load_golden_data("feature_selection.json");

    for case in &cases {
        let name = case["name"].as_str().unwrap();
        if case["algorithm"].as_str().unwrap() != "MutualInformation" {
            continue;
        }

        let x = json_to_array2(&case["X"]);
        let y = common::json_to_array1(&case["y"]);
        let n_features = case["n_features"].as_u64().unwrap() as usize;

        // Fit MutualInformationSelector to select top-1 feature
        let mi_selector = MutualInformationSelector::new(1);
        let fitted = Fit::fit(&mi_selector, &x, &y).unwrap();

        let mi_scores = fitted.mi_scores();

        // MI scores should be non-negative
        for (i, &score) in mi_scores.iter().enumerate() {
            assert!(
                score >= 0.0,
                "{}: MI score at feature {} is negative: {}",
                name, i, score
            );
        }

        // Should have the right number of scores
        assert_eq!(
            mi_scores.len(), n_features,
            "{}: MI scores length mismatch", name
        );

        // Transform should reduce to 1 feature
        let x_transformed = fitted.transform(&x).unwrap();
        assert_eq!(
            x_transformed.ncols(), 1,
            "{}: expected 1 column after selection, got {}",
            name, x_transformed.ncols()
        );
    }
}
