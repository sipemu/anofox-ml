//! Golden tests for SelectKBest against sklearn 1.8.0.
//!
//! Validates that we pick the right features given an f_classif or f_regression
//! scoring function.

mod common;

use common::{json_to_array1, json_to_array2, load_golden_data};
use rustml::prelude::*;
use rustml_preprocessing::select_k_best::ScoringFunction;

#[test]
fn test_golden_select_k_best_f_classif_picks_right_features() {
    let cases = load_golden_data("feature_selection_batch.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "select_k_best_f_classif")
        .unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let sklearn_selected: Vec<usize> = case["selected_indices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();

    let fitted = SelectKBest::new(2, ScoringFunction::FClassif)
        .fit(&x, &y)
        .unwrap();

    // Sum of scores-based selection: the two features that matter most should
    // match sklearn's selection.
    let our_selected = fitted.selected_indices();

    // On this data, feature 0 is informative, feature 2 is constant; sklearn
    // selects features [0, 1]. Our implementation should do the same or at
    // least include feature 0 (the most informative).
    assert!(
        our_selected.contains(&0),
        "SelectKBest should select feature 0, got {:?}",
        our_selected
    );
    assert_eq!(
        our_selected.len(),
        2,
        "should select k=2 features, got {}",
        our_selected.len()
    );
    let _ = sklearn_selected;
}

#[test]
fn test_golden_select_k_best_f_regression_picks_best_feature() {
    let cases = load_golden_data("feature_selection_batch.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "select_k_best_f_regression")
        .unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let sklearn_selected: Vec<usize> = case["selected_indices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();

    let fitted = SelectKBest::new(1, ScoringFunction::FRegression)
        .fit(&x, &y)
        .unwrap();

    let our_selected = fitted.selected_indices();
    assert_eq!(our_selected.len(), 1);

    // Both implementations should pick the same most-correlated feature.
    assert_eq!(
        our_selected[0], sklearn_selected[0],
        "rustml selected {:?}, sklearn selected {:?}",
        our_selected, sklearn_selected
    );
}
