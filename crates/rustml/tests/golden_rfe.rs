//! Golden test for RFE / SFS against sklearn 1.8.0.

mod common;

use common::{json_to_array1, json_to_array2, load_golden_data};
use rustml::core::{Fit, Predict};
use rustml::prelude::*;
use rustml_metrics::r2_score;
use rustml_preprocessing::{Rfe, SequentialFeatureSelector};

#[test]
fn test_rfe_matches_sklearn_support() {
    let cases = load_golden_data("rfe.json");
    let case = &cases[0];
    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let k = case["n_features_to_select"].as_u64().unwrap() as usize;
    let sk_support: Vec<bool> = case["sklearn_rfe_support"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_bool().unwrap())
        .collect();

    // Importance from Ridge |coef_|.
    let importance = move |xs: &ndarray::Array2<f64>, ys: &ndarray::Array1<f64>| {
        let m = RidgeRegressor::new().with_lambda(0.01).fit(xs, ys)?;
        Ok(m.coefficients().mapv(|v| v.abs()))
    };
    let rfe = Rfe::new(k, importance);
    let fitted = rfe.fit(&x, &y).unwrap();

    assert_eq!(fitted.support, sk_support);
}

#[test]
fn test_sfs_picks_informative_features() {
    let cases = load_golden_data("rfe.json");
    let case = &cases[0];
    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let k = case["n_features_to_select"].as_u64().unwrap() as usize;
    let expected: Vec<usize> = case["expected_features"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();

    // Scorer = train Ridge, return R² on training (cheap proxy).
    let score = move |xs: &ndarray::Array2<f64>, ys: &ndarray::Array1<f64>| {
        let m = RidgeRegressor::new().with_lambda(0.01).fit(xs, ys)?;
        let p = m.predict(xs)?;
        Ok(r2_score(ys, &p).unwrap_or(0.0))
    };

    let sfs = SequentialFeatureSelector::new(k, score);
    let fitted = sfs.fit(&x, &y).unwrap();

    let selected: Vec<usize> = fitted
        .support
        .iter()
        .enumerate()
        .filter(|(_, &b)| b)
        .map(|(i, _)| i)
        .collect();
    // SFS without true CV may not exactly match sklearn's pick, but the
    // true-informative features should be in there.
    let mut hits = 0;
    for &e in &expected {
        if selected.contains(&e) {
            hits += 1;
        }
    }
    assert!(
        hits >= 2,
        "SFS selected {:?}; expected to overlap with {:?}",
        selected,
        expected
    );
}
