//! Behavioral parity test for StackingClassifier against sklearn 1.8.0.
//!
//! We don't pursue exact agreement (sklearn defaults to `predict_proba` +
//! StratifiedKFold; ours uses hard predictions + sequential KFold). The
//! test asserts both implementations land in the same accuracy band on a
//! well-separated synthetic problem.

mod common;

use anofox_ml::prelude::*;
use common::{json_to_array1, json_to_array2, load_golden_data};

#[test]
fn test_stacking_classifier_matches_sklearn_accuracy_band() {
    let cases = load_golden_data("stacking_classifier.json");
    let case = &cases[0];

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let sklearn_acc = case["sklearn_accuracy"].as_f64().unwrap();

    let sc = StackingClassifier::new(DecisionTreeClassifier {
        max_depth: Some(3),
        ..Default::default()
    })
    .push(
        "t1",
        DecisionTreeClassifier {
            max_depth: Some(3),
            ..Default::default()
        },
    )
    .push(
        "t2",
        DecisionTreeClassifier {
            max_depth: Some(5),
            ..Default::default()
        },
    )
    .with_cv_folds(2);

    let fitted: anofox_ml_ensemble::FittedStackingClassifier<f64> = sc.fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    let correct = preds
        .iter()
        .zip(y.iter())
        .filter(|(p, t)| (*p - *t).abs() < 0.5)
        .count();
    let acc = correct as f64 / y.len() as f64;

    // Sklearn typically scores 0.95+. Require ours within 0.10 of that.
    assert!(
        (acc - sklearn_acc).abs() < 0.10,
        "anofox-ml accuracy {acc} vs sklearn {sklearn_acc}"
    );
    assert!(acc >= 0.85, "anofox-ml accuracy {acc} too low");
}
