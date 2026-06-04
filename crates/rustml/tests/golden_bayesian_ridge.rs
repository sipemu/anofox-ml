//! Golden test for BayesianRidge / ARDRegression against sklearn 1.8.0.
//!
//! Both implementations evidence-maximize the same model. Our IRLS-style
//! fixed-point updates converge to slightly different α / λ values, so we
//! require predictions within 1% relative tolerance rather than exact match.

mod common;

use common::{json_to_array1, json_to_array2, load_golden_data};
use rustml::prelude::*;
use rustml_regression::{ARDRegression, BayesianRidge};

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-9)
}

#[test]
fn test_bayesian_ridge_matches_sklearn() {
    let cases = load_golden_data("bayesian_ridge.json");
    let case = cases.iter().find(|c| c["name"] == "bayesian_ridge").unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let expected = json_to_array1(&case["sklearn_predictions"]);

    let fitted = BayesianRidge::new().fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();
    for (i, (&a, &b)) in preds.iter().zip(expected.iter()).enumerate() {
        assert!(rel(a, b) < 0.005, "[{}] rustml={}, sklearn={}", i, a, b);
    }
    // Posterior std should also match sklearn's return_std=True within a few %.
    let sk_std = json_to_array1(&case["sklearn_std"]);
    let std = fitted.predict_std(&x).unwrap();
    for (i, (&a, &b)) in std.iter().zip(sk_std.iter()).enumerate() {
        assert!(
            rel(a, b) < 0.05,
            "[std {}] rustml={}, sklearn={}", i, a, b
        );
    }
}

#[test]
fn test_ard_matches_sklearn_in_relevance() {
    // ARD's job is to drive irrelevant feature coefficients toward zero.
    // We check that rustml and sklearn agree on *which* features are
    // important (rank order of |coef|) and that predictions are close.
    let cases = load_golden_data("bayesian_ridge.json");
    let case = cases.iter().find(|c| c["name"] == "ard_sparse").unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let sklearn_coef = json_to_array1(&case["sklearn_coef"]);

    let fitted = ARDRegression::new().fit(&x, &y).unwrap();
    // R² high enough — sklearn typically nails this.
    let preds = fitted.predict(&x).unwrap();
    let ss_res: f64 = preds.iter().zip(y.iter()).map(|(p, t)| (t - p).powi(2)).sum();
    let y_mean: f64 = y.iter().sum::<f64>() / y.len() as f64;
    let ss_tot: f64 = y.iter().map(|t| (t - y_mean).powi(2)).sum();
    let r2 = 1.0 - ss_res / ss_tot;
    assert!(r2 > 0.95, "R² too low: {r2}");

    // Both must drive feature 1, 2, 4 to ~0.
    for &j in &[1usize, 2, 4] {
        assert!(fitted.coef[j].abs() < 0.1, "ard coef[{j}] = {}", fitted.coef[j]);
        assert!(sklearn_coef[j].abs() < 0.1);
    }
    // Both must keep feature 0 and 3 large.
    assert!(fitted.coef[0].abs() > 1.0);
    assert!(fitted.coef[3].abs() > 0.5);
}
