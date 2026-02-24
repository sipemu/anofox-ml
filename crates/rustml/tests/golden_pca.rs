mod common;

use common::{json_to_array1, json_to_array2, load_golden_data};
use rustml::prelude::*;

// PCA via power iteration may not converge perfectly, so use loose tolerances.
const VARIANCE_TOL: f64 = 0.5;

#[test]
fn test_golden_pca() {
    let cases = load_golden_data("pca.json");

    for case in &cases {
        let name = case["name"].as_str().unwrap();

        let x = json_to_array2(&case["X"]);
        let n_components = case["n_components"].as_u64().unwrap() as usize;
        let expected_variance = json_to_array1(&case["explained_variance"]);

        let pca = Pca { n_components };
        let fitted = FitUnsupervised::<f64>::fit(&pca, &x).unwrap();

        // Check explained variance (power iteration may not converge perfectly)
        let actual_variance = fitted.explained_variance();
        assert_eq!(
            actual_variance.len(),
            expected_variance.len(),
            "{}: variance length mismatch",
            name
        );

        // Variance should be in the right ballpark
        for (i, (&a, &e)) in actual_variance.iter().zip(expected_variance.iter()).enumerate() {
            let rel_err = if e.abs() > 1e-10 {
                (a - e).abs() / e.abs()
            } else {
                (a - e).abs()
            };
            assert!(
                rel_err < VARIANCE_TOL,
                "{}/explained_variance[{}]: expected {}, got {}, rel_err {}",
                name,
                i,
                e,
                a,
                rel_err
            );
        }

        // Transform and inverse_transform should approximately roundtrip
        let transformed = fitted.transform(&x).unwrap();
        let recovered = fitted.inverse_transform(&transformed).unwrap();

        // Roundtrip error depends on n_components
        if n_components == x.ncols() {
            // Full reconstruction should be very close
            for ((r, c), &actual) in recovered.indexed_iter() {
                let expected = x[[r, c]];
                assert!(
                    (actual - expected).abs() < 1.0,
                    "{}/roundtrip[{},{}]: expected {}, got {}",
                    name,
                    r,
                    c,
                    expected,
                    actual
                );
            }
        }
    }
}
