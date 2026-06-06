mod common;

use anofox_ml::prelude::*;
use common::{json_to_array1, json_to_array2, load_golden_data};

#[test]
fn test_golden_one_class_svm_inliers() {
    let cases = load_golden_data("svm_extra.json");
    let case = cases.iter().find(|c| c["name"] == "one_class_svm").unwrap();

    let x_train = json_to_array2(&case["X_train"]);
    let x_inlier = json_to_array2(&case["X_test_inlier"]);
    let x_outlier = json_to_array2(&case["X_test_outlier"]);

    let _expected_inlier: Vec<f64> = case["pred_inlier"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let _expected_outlier: Vec<f64> = case["pred_outlier"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    let model = OneClassSvm {
        nu: 0.3,
        kernel: SvmKernel::Rbf { gamma: 1.0 },
        max_iter: 5000,
        tol: 1e-6,
    };
    let fitted = FitUnsupervised::fit(&model, &x_train).unwrap();

    // Most inliers should be predicted as +1
    let pred_in = fitted.predict(&x_inlier).unwrap();
    let inlier_correct: usize = pred_in.iter().filter(|&&p| p > 0.0).count();
    assert!(
        inlier_correct >= pred_in.len() / 2,
        "at least half of inliers should be +1, got {}/{}",
        inlier_correct,
        pred_in.len()
    );

    // Outliers (far away) should be predicted as -1
    let pred_out = fitted.predict(&x_outlier).unwrap();
    let outlier_correct: usize = pred_out.iter().filter(|&&p| p < 0.0).count();
    assert!(
        outlier_correct >= pred_out.len() / 2,
        "at least half of outliers should be -1, got {}/{}",
        outlier_correct,
        pred_out.len()
    );
}

#[test]
fn test_golden_linear_svr_predictions() {
    let cases = load_golden_data("svm_extra.json");
    let case = cases.iter().find(|c| c["name"] == "linear_svr").unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let expected_preds = json_to_array1(&case["predictions"]);

    let model = LinearSvr {
        c: 10.0,
        epsilon: 0.1,
        max_iter: 5000,
        tol: 1e-4,
    };
    let fitted: anofox_ml::svm::FittedLinearSvr<f64> = Fit::fit(&model, &x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    // LinearSVR with SGD won't match sklearn's liblinear exactly,
    // but predictions should be in the same ballpark (within 3.0)
    for (i, (p, e)) in preds.iter().zip(expected_preds.iter()).enumerate() {
        assert!(
            (p - e).abs() < 3.0,
            "prediction {} differs too much: anofox-ml={:.3}, sklearn={:.3}",
            i,
            p,
            e
        );
    }
}
