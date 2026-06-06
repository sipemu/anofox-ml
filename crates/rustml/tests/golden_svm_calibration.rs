//! Golden tests for NuSVC, NuSVR, and CalibratedClassifierCV against sklearn 1.8.0.

mod common;

use common::{json_to_array1, json_to_array2, load_golden_data};
use ndarray::Array1;
use rustml::prelude::*;
use rustml_ensemble::calibrated_classifier::FittedCalibratedClassifier;
use rustml_trees::DecisionTreeClassifier;

fn accuracy(pred: &Array1<f64>, target: &Array1<f64>) -> f64 {
    let correct: usize = pred
        .iter()
        .zip(target.iter())
        .filter(|(&p, &t)| (p - t).abs() < 1e-10)
        .count();
    correct as f64 / pred.len() as f64
}

fn r2(pred: &Array1<f64>, target: &Array1<f64>) -> f64 {
    let mean = target.iter().sum::<f64>() / target.len() as f64;
    let ss_res: f64 = pred
        .iter()
        .zip(target.iter())
        .map(|(&p, &t)| (p - t).powi(2))
        .sum();
    let ss_tot: f64 = target.iter().map(|&t| (t - mean).powi(2)).sum();
    if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    }
}

#[test]
fn test_golden_nu_svc_accuracy() {
    let cases = load_golden_data("svm_calibration_batch.json");
    let case = cases.iter().find(|c| c["name"] == "nu_svc").unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let sklearn_acc = case["accuracy"].as_f64().unwrap();

    let model = NuSvc::new().with_nu(0.5).with_kernel(SvmKernel::Linear);
    let fitted: rustml::svm::FittedNuSvc<f64> = Fit::fit(&model, &x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();
    let our_acc = accuracy(&preds, &y);

    assert!(
        our_acc >= sklearn_acc - 0.2,
        "NuSVC accuracy {} vs sklearn {}",
        our_acc,
        sklearn_acc
    );
}

#[test]
fn test_golden_nu_svr_r2() {
    let cases = load_golden_data("svm_calibration_batch.json");
    let case = cases.iter().find(|c| c["name"] == "nu_svr").unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let sklearn_r2 = case["r2"].as_f64().unwrap();

    let model = NuSvr::new()
        .with_nu(0.5)
        .with_c(10.0)
        .with_kernel(SvmKernel::Linear);
    let fitted: rustml::svm::FittedNuSvr<f64> = Fit::fit(&model, &x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();
    let our_r2 = r2(&preds, &y);

    // NuSVR is implemented via epsilon bisection on our SVR FISTA solver.
    // On this linearly separable data, both sklearn and rustml should attain
    // R² essentially equal to 1.0.
    assert!(
        our_r2 >= sklearn_r2 - 0.05,
        "NuSVR R² {:.4} vs sklearn {:.4}",
        our_r2,
        sklearn_r2
    );
}

#[test]
fn test_golden_calibrated_classifier_accuracy() {
    let cases = load_golden_data("svm_calibration_batch.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "calibrated_classifier_sigmoid")
        .unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let sklearn_acc = case["accuracy"].as_f64().unwrap();

    let cal = CalibratedClassifierCV::new(DecisionTreeClassifier {
        max_depth: Some(3),
        ..Default::default()
    })
    .with_method(CalibrationMethod::Sigmoid)
    .with_cv_folds(2);

    let fitted: FittedCalibratedClassifier<f64> = Fit::fit(&cal, &x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();
    let our_acc = accuracy(&preds, &y);

    assert!(
        our_acc >= sklearn_acc - 0.2,
        "CalibratedClassifierCV accuracy {} vs sklearn {}",
        our_acc,
        sklearn_acc
    );

    // Calibrated probabilities should be in [0, 1]
    let proba = fitted.predict_proba(&x).unwrap();
    for &p in proba.iter() {
        assert!(
            (0.0..=1.0).contains(&p),
            "calibrated prob {} out of [0,1]",
            p
        );
    }
}
