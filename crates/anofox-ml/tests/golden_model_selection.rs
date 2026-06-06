mod common;

use anofox_ml::prelude::*;
use common::load_golden_data;

#[test]
fn test_golden_k_fold() {
    let cases = load_golden_data("model_selection_extra.json");
    let case = cases.iter().find(|c| c["name"] == "k_fold").unwrap();

    let n = case["n_samples"].as_u64().unwrap() as usize;
    let k = case["k"].as_u64().unwrap() as usize;
    let expected_folds: Vec<(Vec<usize>, Vec<usize>)> = case["folds"]
        .as_array()
        .unwrap()
        .iter()
        .map(|fold| {
            let train: Vec<usize> = fold[0]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let test: Vec<usize> = fold[1]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            (train, test)
        })
        .collect();

    let actual_folds = k_fold(n, k).unwrap();

    assert_eq!(
        actual_folds.len(),
        expected_folds.len(),
        "wrong number of folds"
    );
    for (i, ((actual_train, actual_test), (expected_train, expected_test))) in
        actual_folds.iter().zip(expected_folds.iter()).enumerate()
    {
        assert_eq!(actual_train, expected_train, "fold {} train mismatch", i);
        assert_eq!(actual_test, expected_test, "fold {} test mismatch", i);
    }
}

#[test]
fn test_golden_leave_one_out() {
    let cases = load_golden_data("model_selection_extra.json");
    let case = cases.iter().find(|c| c["name"] == "leave_one_out").unwrap();

    let n = case["n_samples"].as_u64().unwrap() as usize;
    let expected_folds: Vec<(Vec<usize>, Vec<usize>)> = case["folds"]
        .as_array()
        .unwrap()
        .iter()
        .map(|fold| {
            let train: Vec<usize> = fold[0]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let test: Vec<usize> = fold[1]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            (train, test)
        })
        .collect();

    let actual_folds = leave_one_out(n);

    assert_eq!(actual_folds.len(), expected_folds.len());
    for (i, ((actual_train, actual_test), (expected_train, expected_test))) in
        actual_folds.iter().zip(expected_folds.iter()).enumerate()
    {
        assert_eq!(actual_train, expected_train, "fold {} train mismatch", i);
        assert_eq!(actual_test, expected_test, "fold {} test mismatch", i);
    }
}

#[test]
fn test_golden_time_series_split() {
    let cases = load_golden_data("model_selection_extra.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "time_series_split")
        .unwrap();

    let n = case["n_samples"].as_u64().unwrap() as usize;
    let n_splits = case["n_splits"].as_u64().unwrap() as usize;
    let expected_folds: Vec<(Vec<usize>, Vec<usize>)> = case["folds"]
        .as_array()
        .unwrap()
        .iter()
        .map(|fold| {
            let train: Vec<usize> = fold[0]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let test: Vec<usize> = fold[1]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            (train, test)
        })
        .collect();

    let actual_folds = time_series_split(n, n_splits).unwrap();

    assert_eq!(
        actual_folds.len(),
        expected_folds.len(),
        "wrong number of splits"
    );
    for (i, ((actual_train, actual_test), (expected_train, expected_test))) in
        actual_folds.iter().zip(expected_folds.iter()).enumerate()
    {
        assert_eq!(actual_train, expected_train, "split {} train mismatch", i);
        assert_eq!(actual_test, expected_test, "split {} test mismatch", i);
    }
}
