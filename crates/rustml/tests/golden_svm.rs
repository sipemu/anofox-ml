mod common;

use common::{json_to_array1, json_to_array2, load_golden_data};
use rustml::prelude::*;

// SVM implementations differ between sklearn and our simplified solver,
// so we use a generous tolerance for predictions on well-separated data.
const PRED_TOL: f64 = 0.15;

#[test]
fn test_golden_linear_svc() {
    let cases = load_golden_data("svm.json");

    for case in &cases {
        let name = case["name"].as_str().unwrap();
        if case["algorithm"].as_str().unwrap() != "LinearSvc" {
            continue;
        }

        let x_train = json_to_array2(&case["X_train"]);
        let y_train = json_to_array1(&case["y_train"]);
        let x_test = json_to_array2(&case["X_test"]);
        let expected_pred = json_to_array1(&case["y_pred"]);

        let c = case["C"].as_f64().unwrap();
        let max_iter = case["max_iter"].as_u64().unwrap() as usize;

        let svm = LinearSvc::new()
            .with_c(c)
            .with_max_iter(max_iter)
            .with_seed(42);

        let fitted = Fit::fit(&svm, &x_train, &y_train).unwrap();
        let preds = fitted.predict(&x_test).unwrap();

        // Verify predictions are valid class labels
        let classes = fitted.class_labels();
        for &p in preds.iter() {
            assert!(
                classes.iter().any(|&c| (c - p).abs() < 1e-10),
                "{}: prediction {} is not a valid class label",
                name,
                p
            );
        }

        // Check accuracy on well-separated data
        let mut correct = 0;
        for (&p, &e) in preds.iter().zip(expected_pred.iter()) {
            if (p - e).abs() < PRED_TOL {
                correct += 1;
            }
        }
        let accuracy = correct as f64 / preds.len() as f64;
        assert!(
            accuracy >= 0.7,
            "{}: accuracy {} is too low (expected >= 0.7)",
            name,
            accuracy
        );
    }
}

#[test]
fn test_golden_svc() {
    let cases = load_golden_data("svm.json");

    for case in &cases {
        let name = case["name"].as_str().unwrap();
        if case["algorithm"].as_str().unwrap() != "Svc" {
            continue;
        }

        let x_train = json_to_array2(&case["X_train"]);
        let y_train = json_to_array1(&case["y_train"]);
        let x_test = json_to_array2(&case["X_test"]);
        let expected_pred = json_to_array1(&case["y_pred"]);

        let c = case["C"].as_f64().unwrap();
        let max_iter = case["max_iter"].as_u64().unwrap() as usize;

        let kernel = match case["kernel"].as_str().unwrap() {
            "linear" => SvmKernel::Linear,
            "rbf" => {
                let gamma = case["gamma"].as_f64().unwrap();
                SvmKernel::Rbf { gamma }
            }
            k => panic!("unknown kernel: {}", k),
        };

        let svm = Svc::new()
            .with_c(c)
            .with_kernel(kernel)
            .with_max_iter(max_iter)
            .with_seed(42);

        let fitted = Fit::fit(&svm, &x_train, &y_train).unwrap();
        let preds = fitted.predict(&x_test).unwrap();

        // Verify predictions are valid class labels
        let classes = fitted.class_labels();
        for &p in preds.iter() {
            assert!(
                classes.iter().any(|&c| (c - p).abs() < 1e-10),
                "{}: prediction {} is not a valid class label",
                name,
                p
            );
        }

        // Check accuracy on well-separated data
        let mut correct = 0;
        for (&p, &e) in preds.iter().zip(expected_pred.iter()) {
            if (p - e).abs() < PRED_TOL {
                correct += 1;
            }
        }
        let accuracy = correct as f64 / preds.len() as f64;
        assert!(
            accuracy >= 0.7,
            "{}: accuracy {} is too low (expected >= 0.7)",
            name,
            accuracy
        );
    }
}
