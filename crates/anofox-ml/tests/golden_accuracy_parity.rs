//! Tight numerical parity tests against sklearn 1.8.0 for models where we
//! previously diverged: NuSVR, NuSVC, and AdaBoostRegressor.

mod common;

use anofox_ml::prelude::*;
use common::{json_to_array1, json_to_array2, load_golden_data};
use ndarray::Array1;

fn r2(pred: &Array1<f64>, target: &Array1<f64>) -> f64 {
    let mean = target.iter().sum::<f64>() / target.len() as f64;
    let ss_res: f64 = pred
        .iter()
        .zip(target.iter())
        .map(|(&p, &t)| (p - t).powi(2))
        .sum();
    let ss_tot: f64 = target.iter().map(|&t| (t - mean).powi(2)).sum();
    1.0 - ss_res / ss_tot
}

fn accuracy(pred: &Array1<f64>, target: &Array1<f64>) -> f64 {
    let correct: usize = pred
        .iter()
        .zip(target.iter())
        .filter(|(&p, &t)| (p - t).abs() < 1e-10)
        .count();
    correct as f64 / pred.len() as f64
}

#[test]
fn test_parity_nu_svr_linear() {
    let cases = load_golden_data("accuracy_parity.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "nu_svr_exact_linear")
        .unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let sk_preds = json_to_array1(&case["predictions"]);
    let sk_r2 = case["r2"].as_f64().unwrap();

    let fitted: anofox_ml::svm::FittedNuSvr<f64> = NuSvr::new()
        .with_nu(0.5)
        .with_c(10.0)
        .with_kernel(SvmKernel::Linear)
        .with_max_iter(5000)
        .fit(&x, &y)
        .unwrap();

    let our_preds = fitted.predict(&x).unwrap();
    let our_r2 = r2(&our_preds, &y);

    // With the libsvm-style SMO port, R² matches sklearn to ~1e-5.
    assert!(
        (our_r2 - sk_r2).abs() < 1e-4,
        "NuSVR R² parity: ours={:.6}, sklearn={:.6}",
        our_r2,
        sk_r2
    );

    // Per-point predictions should match sklearn to ~1e-3.
    for (i, (&o, &s)) in our_preds.iter().zip(sk_preds.iter()).enumerate() {
        assert!(
            (o - s).abs() < 1e-2,
            "NuSVR pred[{}]: ours={:.6}, sklearn={:.6}",
            i,
            o,
            s
        );
    }
}

#[test]
fn test_parity_nu_svr_rbf() {
    let cases = load_golden_data("accuracy_parity.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "nu_svr_exact_rbf")
        .unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let sk_preds = json_to_array1(&case["predictions"]);
    let sk_r2 = case["r2"].as_f64().unwrap();

    let fitted: anofox_ml::svm::FittedNuSvr<f64> = NuSvr::new()
        .with_nu(0.5)
        .with_c(10.0)
        .with_kernel(SvmKernel::Rbf { gamma: 0.1 })
        .with_max_iter(20000)
        .fit(&x, &y)
        .unwrap();

    let our_preds = fitted.predict(&x).unwrap();
    let our_r2 = r2(&our_preds, &y);

    // libsvm-style SMO matches sklearn's R² to ~1e-4 on this RBF case.
    assert!(
        (our_r2 - sk_r2).abs() < 1e-3,
        "NuSVR RBF R² parity: ours={:.6}, sklearn={:.6}",
        our_r2,
        sk_r2
    );

    // Per-point predictions match sklearn to ~1e-3.
    for (i, (&o, &s)) in our_preds.iter().zip(sk_preds.iter()).enumerate() {
        assert!(
            (o - s).abs() < 5e-3,
            "NuSVR RBF pred[{}]: ours={:.6}, sklearn={:.6}",
            i,
            o,
            s
        );
    }
}

#[test]
fn test_parity_nu_svc_linear() {
    let cases = load_golden_data("accuracy_parity.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "nu_svc_exact_linear")
        .unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let sk_preds = json_to_array1(&case["predictions"]);
    let sk_acc = case["accuracy"].as_f64().unwrap();

    let fitted: anofox_ml::svm::FittedNuSvc<f64> = NuSvc::new()
        .with_nu(0.5)
        .with_kernel(SvmKernel::Linear)
        .with_max_iter(5000)
        .fit(&x, &y)
        .unwrap();

    let our_preds = fitted.predict(&x).unwrap();
    let our_acc = accuracy(&our_preds, &y);

    // Both should reach 100% accuracy on linearly separable data.
    assert!(
        (our_acc - sk_acc).abs() < 1e-10,
        "NuSVC accuracy parity: ours={}, sklearn={}",
        our_acc,
        sk_acc
    );

    // Class predictions should match element-wise.
    for (i, (&o, &s)) in our_preds.iter().zip(sk_preds.iter()).enumerate() {
        assert_eq!(
            o as i64, s as i64,
            "NuSVC pred[{}]: ours={}, sklearn={}",
            i, o, s
        );
    }
}

#[test]
fn test_parity_adaboost_regressor() {
    let cases = load_golden_data("accuracy_parity.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "adaboost_regressor_exact")
        .unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let sk_r2 = case["r2"].as_f64().unwrap();

    let fitted = AdaBoostRegressor::new()
        .with_n_estimators(20)
        .with_seed(42)
        .fit(&x, &y)
        .unwrap();

    let our_preds = fitted.predict(&x).unwrap();
    let our_r2 = r2(&our_preds, &y);

    // With matching default base tree depth (3), R² should be within 0.05.
    assert!(
        our_r2 >= sk_r2 - 0.05,
        "AdaBoostRegressor R² parity: ours={:.6}, sklearn={:.6}",
        our_r2,
        sk_r2
    );
}
