//! Golden test for TruncatedSVD against sklearn 1.8.0.

mod common;

use common::{json_to_array1, json_to_array2, load_golden_data};
use rustml::core::{FitUnsupervised, Transform};

#[test]
fn test_truncated_svd_matches_sklearn_abs() {
    let cases = load_golden_data("truncated_svd.json");
    let case = &cases[0];

    let x = json_to_array2(&case["X"]);
    let k = case["n_components"].as_u64().unwrap() as usize;
    let sv_sk = json_to_array1(&case["sklearn_singular_values"]);
    let abs_sk = json_to_array2(&case["sklearn_transformed_abs"]);

    let svd = rustml::preprocessing::TruncatedSvd::new(k).fit(&x).unwrap();
    let t = svd.transform(&x).unwrap();

    // Singular values must match exactly (deterministic).
    for j in 0..k {
        assert!(
            (svd.singular_values[j] - sv_sk[j]).abs() < 1e-6,
            "σ[{}]: rustml={}, sklearn={}",
            j,
            svd.singular_values[j],
            sv_sk[j]
        );
    }

    // Transformed entries match in absolute value (SVD sign-ambiguous).
    for i in 0..t.nrows() {
        for j in 0..k {
            assert!(
                (t[[i, j]].abs() - abs_sk[[i, j]]).abs() < 1e-6,
                "|T[{},{}]| mismatch: {} vs {}",
                i,
                j,
                t[[i, j]].abs(),
                abs_sk[[i, j]]
            );
        }
    }
}
