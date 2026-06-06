//! Golden test for GaussianMixture against sklearn 1.8.0.

mod common;

use common::{json_to_array1, json_to_array2, load_golden_data};
use ndarray::Array1;
use rustml::core::FitUnsupervised;
use rustml_cluster::{CovarianceType, GaussianMixture};

fn adjusted_rand(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    use std::collections::HashMap;
    let n = a.len();
    let mut contingency = HashMap::<(i64, i64), usize>::new();
    let mut row = HashMap::<i64, usize>::new();
    let mut col = HashMap::<i64, usize>::new();
    for i in 0..n {
        let ai = a[i] as i64;
        let bi = b[i] as i64;
        *contingency.entry((ai, bi)).or_default() += 1;
        *row.entry(ai).or_default() += 1;
        *col.entry(bi).or_default() += 1;
    }
    let c2 = |n: usize| -> f64 {
        if n < 2 {
            0.0
        } else {
            (n * (n - 1)) as f64 / 2.0
        }
    };
    let sum_nij: f64 = contingency.values().map(|&c| c2(c)).sum();
    let sum_ai: f64 = row.values().map(|&c| c2(c)).sum();
    let sum_bj: f64 = col.values().map(|&c| c2(c)).sum();
    let total = c2(n);
    let expected = sum_ai * sum_bj / total.max(1e-12);
    let max_val = 0.5 * (sum_ai + sum_bj);
    (sum_nij - expected) / (max_val - expected).max(1e-12)
}

#[test]
fn test_gmm_full_and_diag_match_sklearn() {
    let cases = load_golden_data("gmm.json");
    for case in &cases {
        let x = json_to_array2(&case["X"]);
        let y_true = json_to_array1(&case["y_true"]);
        let sklearn_ari = case["sklearn_ari"].as_f64().unwrap();
        let ct = match case["covariance_type"].as_str().unwrap() {
            "full" => CovarianceType::Full,
            "diag" => CovarianceType::Diag,
            o => panic!("unknown covariance_type: {o}"),
        };
        let fitted = GaussianMixture::new(3)
            .with_covariance_type(ct)
            .with_max_iter(200)
            .with_seed(0)
            .fit(&x)
            .unwrap();
        let preds = rustml::core::Predict::predict(&fitted, &x).unwrap();
        let ari = adjusted_rand(&preds, &y_true);
        assert!(
            ari >= sklearn_ari - 0.05,
            "{}: rustml ARI {} vs sklearn {}",
            case["name"].as_str().unwrap(),
            ari,
            sklearn_ari
        );
    }
}
