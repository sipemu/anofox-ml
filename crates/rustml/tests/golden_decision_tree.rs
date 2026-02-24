mod common;

use common::{assert_array1_close, json_to_array1, json_to_array2, load_golden_data};
use rustml::prelude::*;

// Decision trees can have different splitting strategies (sklearn uses random_state
// to break ties), so we use a looser tolerance for predictions where tree structure
// may differ slightly. Feature importances especially can vary.
const PRED_TOL: f64 = 1e-10;
const IMPORTANCE_TOL: f64 = 0.15;

#[test]
fn test_golden_decision_tree_classifier() {
    let cases = load_golden_data("decision_tree.json");

    for case in &cases {
        let name = case["name"].as_str().unwrap();
        if case["algorithm"].as_str().unwrap() != "DecisionTreeClassifier" {
            continue;
        }

        let x_train = json_to_array2(&case["X_train"]);
        let y_train = json_to_array1(&case["y_train"]);
        let x_test = json_to_array2(&case["X_test"]);
        let expected_pred = json_to_array1(&case["y_pred"]);

        let max_depth = case["max_depth"].as_u64().map(|d| d as usize);
        let min_samples_split = case["min_samples_split"].as_u64().unwrap() as usize;
        let min_samples_leaf = case["min_samples_leaf"].as_u64().unwrap() as usize;

        let criterion = match case["criterion"].as_str().unwrap() {
            "gini" => SplitCriterion::Gini,
            "entropy" => SplitCriterion::Entropy,
            c => panic!("unknown criterion: {}", c),
        };

        let tree = DecisionTreeClassifier {
            max_depth,
            min_samples_split,
            min_samples_leaf,
            criterion,
        };

        let fitted = Fit::fit(&tree, &x_train, &y_train).unwrap();
        let preds = fitted.predict(&x_test).unwrap();

        assert_array1_close(&preds, &expected_pred, PRED_TOL, &format!("{}/predict", name));

        // Check feature importances (looser tolerance due to different split strategies)
        let expected_importances = json_to_array1(&case["feature_importances"]);
        let actual_importances = fitted.feature_importances();
        assert_array1_close(
            &actual_importances,
            &expected_importances,
            IMPORTANCE_TOL,
            &format!("{}/feature_importances", name),
        );
    }
}

#[test]
fn test_golden_decision_tree_regressor() {
    let cases = load_golden_data("decision_tree.json");

    for case in &cases {
        let name = case["name"].as_str().unwrap();
        if case["algorithm"].as_str().unwrap() != "DecisionTreeRegressor" {
            continue;
        }

        let x_train = json_to_array2(&case["X_train"]);
        let y_train = json_to_array1(&case["y_train"]);
        let x_test = json_to_array2(&case["X_test"]);
        let expected_pred = json_to_array1(&case["y_pred"]);

        let max_depth = case["max_depth"].as_u64().map(|d| d as usize);
        let min_samples_split = case["min_samples_split"].as_u64().unwrap() as usize;
        let min_samples_leaf = case["min_samples_leaf"].as_u64().unwrap() as usize;

        let tree = DecisionTreeRegressor {
            max_depth,
            min_samples_split,
            min_samples_leaf,
        };

        let fitted = Fit::fit(&tree, &x_train, &y_train).unwrap();
        let preds = fitted.predict(&x_test).unwrap();

        assert_array1_close(&preds, &expected_pred, PRED_TOL, &format!("{}/predict", name));

        // Check feature importances
        let expected_importances = json_to_array1(&case["feature_importances"]);
        let actual_importances = fitted.feature_importances();
        assert_array1_close(
            &actual_importances,
            &expected_importances,
            IMPORTANCE_TOL,
            &format!("{}/feature_importances", name),
        );
    }
}
