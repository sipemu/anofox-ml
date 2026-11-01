//! Parity of the liblinear-style dual coordinate descent solvers
//! (`LinearSvc`, `LinearSvr`) with scikit-learn 1.9.1.
//!
//! Reference values were produced with
//! `LinearSVC(loss=..., C=0.7, dual=True, tol=1e-9, max_iter=100_000)` and
//! `LinearSVR(loss=..., C=2.0, epsilon=0.2, dual=True, tol=1e-9,
//! max_iter=100_000)` on the fixed dataset below (default
//! `fit_intercept=True`, `intercept_scaling=1`). The optimum is unique, so
//! coefficients must agree independently of the coordinate order.

use anofox_ml_core::Fit;
use anofox_ml_svm::{LinearSvc, LinearSvcLoss, LinearSvr, LinearSvrLoss};
use ndarray::{array, Array1, Array2};

fn x_data() -> Array2<f64> {
    array![
        [2.041, -2.556, 0.418],
        [-0.568, -0.453, -0.216],
        [-2.02, -0.232, -0.865],
        [3.323, 0.226, -0.353],
        [-0.281, -0.668, -1.055],
        [-0.391, 0.482, -0.239],
        [0.958, -0.2, 0.024],
        [1.546, 0.545, -0.505],
        [-0.183, 0.541, 1.935],
        [-0.27, -0.244, 1.002],
        [-0.886, -0.292, 0.883],
        [0.58, 0.092, 0.67],
        [-2.828, 1.021, -0.96],
        [-1.669, 0.276, 0.701],
        [-0.445, -1.076, 0.026],
        [-0.053, 1.406, 0.747],
        [0.194, 1.112, -0.206],
        [-0.926, 0.584, 0.583],
        [-0.215, -0.783, 0.229],
        [-2.494, 0.69, 0.491],
        [-1.639, 0.061, -0.964],
        [0.757, -2.034, -0.914],
        [0.71, 1.156, -2.158],
        [-0.498, 0.328, -0.609],
        [1.591, -1.191, 0.355],
        [-1.048, 1.406, -0.022],
        [-0.372, -1.718, 1.682],
        [0.753, 0.754, 1.138],
        [0.349, -0.639, -0.8],
        [-0.8, 1.37, -1.46],
    ]
}

fn y_class() -> Array1<f64> {
    array![
        1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0
    ]
}

fn y_multiclass() -> Array1<f64> {
    array![
        2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 2.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0
    ]
}

fn y_reg() -> Array1<f64> {
    array![
        5.893, -0.707, -4.085, 4.813, -0.95, -0.961, 2.355, 1.474, 4.245, 2.492, 1.552, 2.322,
        -6.434, -1.039, 1.089, 1.128, -0.491, -0.268, 1.329, -2.952, -4.208, 1.376, -2.821, -1.768,
        4.263, -2.452, 4.108, 3.535, 0.126, -4.576
    ]
}

fn assert_rel_close(got: f64, expected: f64, rel: f64, what: &str) {
    let scale = expected.abs().max(1e-3);
    assert!(
        (got - expected).abs() <= rel * scale,
        "{what}: got {got}, sklearn {expected}"
    );
}

fn check_svc(loss: LinearSvcLoss, coef: [f64; 3], intercept: f64) {
    let model = LinearSvc::new()
        .with_c(0.7)
        .with_loss(loss)
        .with_tol(1e-9)
        .with_max_iter(100_000);
    let fitted = model.fit(&x_data(), &y_class()).unwrap();
    let w = fitted.weights();
    for j in 0..3 {
        assert_rel_close(w[[0, j]], coef[j], 1e-6, &format!("{loss:?} coef[{j}]"));
    }
    assert_rel_close(
        fitted.bias()[0],
        intercept,
        1e-6,
        &format!("{loss:?} intercept"),
    );
}

#[test]
fn linear_svc_hinge_matches_sklearn() {
    check_svc(
        LinearSvcLoss::Hinge,
        [1.1432176383265578, -1.3570542481812875, 0.807866836100978],
        0.38015284858139436,
    );
}

#[test]
fn linear_svc_squared_hinge_matches_sklearn() {
    check_svc(
        LinearSvcLoss::SquaredHinge,
        [0.8422615996621919, -1.0561883185194971, 0.7436503028962131],
        0.2659599359321163,
    );
}

#[test]
fn linear_svc_default_loss_is_squared_hinge() {
    assert_eq!(LinearSvc::new().loss, LinearSvcLoss::SquaredHinge);
}

#[test]
fn linear_svc_multiclass_ovr_matches_sklearn() {
    let coef = [
        [-1.0943528975028136, 1.3197048131823712, 0.12971330341827564],
        [
            0.01045581392953369,
            -0.08934457826369893,
            0.03578045203380652,
        ],
        [
            1.3991101008183362,
            -1.0523670441656388,
            -0.048881426429434695,
        ],
    ];
    let intercept = [
        -0.7274537254910586,
        -0.39138953548426475,
        -0.9302274613515671,
    ];
    let model = LinearSvc::new()
        .with_c(1.0)
        .with_tol(1e-9)
        .with_max_iter(100_000);
    let fitted = model.fit(&x_data(), &y_multiclass()).unwrap();
    let w = fitted.weights();
    let b = fitted.bias();
    for k in 0..3 {
        for j in 0..3 {
            // Absolute tolerance relative to the row scale (class 1 has
            // near-zero coefficients).
            assert!(
                (w[[k, j]] - coef[k][j]).abs() < 1e-6,
                "coef[{k},{j}]: got {}, sklearn {}",
                w[[k, j]],
                coef[k][j]
            );
        }
        assert_rel_close(b[k], intercept[k], 1e-6, &format!("intercept[{k}]"));
    }
}

fn check_svr(loss: LinearSvrLoss, coef: [f64; 3], intercept: f64) {
    let model = LinearSvr::new()
        .with_c(2.0)
        .with_epsilon(0.2)
        .with_loss(loss)
        .with_tol(1e-9)
        .with_max_iter(100_000);
    let fitted = model.fit(&x_data(), &y_reg()).unwrap();
    let w = fitted.weights();
    for j in 0..3 {
        assert_rel_close(w[j], coef[j], 1e-6, &format!("{loss:?} coef[{j}]"));
    }
    assert_rel_close(
        fitted.bias(),
        intercept,
        1e-6,
        &format!("{loss:?} intercept"),
    );
}

#[test]
fn linear_svr_epsilon_insensitive_matches_sklearn() {
    check_svr(
        LinearSvrLoss::EpsilonInsensitive,
        [1.519909828564615, -0.6485955930102497, 1.9193556345267093],
        0.5307630516847306,
    );
}

#[test]
fn linear_svr_squared_epsilon_insensitive_matches_sklearn() {
    check_svr(
        LinearSvrLoss::SquaredEpsilonInsensitive,
        [1.5178134743773026, -0.6694998909705105, 1.9013651001740286],
        0.5895552559729592,
    );
}
