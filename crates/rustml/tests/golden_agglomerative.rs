//! Golden test for AgglomerativeClustering against sklearn 1.8.0.
//!
//! Hierarchical merges are deterministic given the linkage, but the labels
//! sklearn returns may be permuted. We use Adjusted Rand Index to compare
//! against the true labels and require ARI ≥ sklearn's ARI − 0.05.

mod common;

use common::{json_to_array1, json_to_array2, load_golden_data};
use ndarray::Array1;
use rustml::core::FitUnsupervised;
use rustml_cluster::{AgglomerativeClustering, Linkage};

fn adjusted_rand(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    use std::collections::HashMap;
    let n = a.len();
    let mut contingency = HashMap::<(i64, i64), usize>::new();
    let mut row_marg = HashMap::<i64, usize>::new();
    let mut col_marg = HashMap::<i64, usize>::new();
    for i in 0..n {
        let ai = a[i] as i64;
        let bi = b[i] as i64;
        *contingency.entry((ai, bi)).or_default() += 1;
        *row_marg.entry(ai).or_default() += 1;
        *col_marg.entry(bi).or_default() += 1;
    }
    let comb2 = |n: usize| -> f64 {
        if n < 2 {
            0.0
        } else {
            (n * (n - 1)) as f64 / 2.0
        }
    };
    let mut sum_nij_c2 = 0.0;
    for (_, &c) in &contingency {
        sum_nij_c2 += comb2(c);
    }
    let mut sum_ai_c2 = 0.0;
    for (_, &c) in &row_marg {
        sum_ai_c2 += comb2(c);
    }
    let mut sum_bj_c2 = 0.0;
    for (_, &c) in &col_marg {
        sum_bj_c2 += comb2(c);
    }
    let total_c2 = comb2(n);
    let expected = sum_ai_c2 * sum_bj_c2 / total_c2.max(1e-12);
    let max_val = 0.5 * (sum_ai_c2 + sum_bj_c2);
    (sum_nij_c2 - expected) / (max_val - expected).max(1e-12)
}

#[test]
fn test_agglomerative_all_linkages_match_sklearn() {
    let cases = load_golden_data("agglomerative.json");
    for case in &cases {
        let x = json_to_array2(&case["X"]);
        let y_true = json_to_array1(&case["y_true"]);
        let sklearn_ari = case["sklearn_ari"].as_f64().unwrap();
        let link = match case["linkage"].as_str().unwrap() {
            "ward" => Linkage::Ward,
            "complete" => Linkage::Complete,
            "average" => Linkage::Average,
            "single" => Linkage::Single,
            o => panic!("unknown linkage: {o}"),
        };

        let fitted = AgglomerativeClustering::new(4)
            .with_linkage(link)
            .fit(&x)
            .unwrap();
        let ari = adjusted_rand(&fitted.labels, &y_true);
        assert!(
            ari >= sklearn_ari - 0.05,
            "{}: rustml ARI {} vs sklearn {}",
            case["name"].as_str().unwrap(),
            ari,
            sklearn_ari
        );
    }
}
