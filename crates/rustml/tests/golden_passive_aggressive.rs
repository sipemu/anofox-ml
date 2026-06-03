//! Behavioral parity test for PassiveAggressive {Classifier, Regressor}.

mod common;

use common::{json_to_array1, json_to_array2, load_golden_data};
use rustml::prelude::*;
use rustml_metrics::r2_score;

#[test]
fn test_pa_classifier_accuracy_band() {
    let cases = load_golden_data("passive_aggressive.json");
    let case = cases.iter().find(|c| c["name"] == "pa_classifier").unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let sklearn_acc = case["sklearn_accuracy"].as_f64().unwrap();

    let fitted = PassiveAggressiveClassifier::new()
        .with_c(1.0)
        .with_max_iter(2000)
        .with_seed(0)
        .fit(&x, &y)
        .unwrap();
    let preds = fitted.predict(&x).unwrap();
    let correct = preds.iter().zip(y.iter()).filter(|(p, t)| (*p - *t).abs() < 0.5).count();
    let acc = correct as f64 / y.len() as f64;
    // sklearn averages updates and converges to ~0.98+; we don't average.
    // Allow a wider band but still require strong accuracy.
    assert!(
        (acc - sklearn_acc).abs() < 0.15,
        "rustml acc {acc} vs sklearn {sklearn_acc}"
    );
    assert!(acc >= 0.85, "accuracy too low: {acc}");
}

#[test]
fn test_pa_regressor_r2_band() {
    let cases = load_golden_data("passive_aggressive.json");
    let case = cases.iter().find(|c| c["name"] == "pa_regressor").unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let sklearn_r2 = case["sklearn_r2"].as_f64().unwrap();

    let fitted = PassiveAggressiveRegressor::new()
        .with_c(1.0)
        .with_epsilon(0.1)
        .with_max_iter(500)
        .with_seed(0)
        .fit(&x, &y)
        .unwrap();
    let preds = fitted.predict(&x).unwrap();
    let r2 = r2_score(&y, &preds).unwrap();
    // sklearn typically scores in [0.7, 0.95]; allow ~0.1 deviation.
    assert!(
        (r2 - sklearn_r2).abs() < 0.15,
        "rustml R² {r2} vs sklearn {sklearn_r2}"
    );
    assert!(r2 > 0.6, "R² too low: {r2}");
}
