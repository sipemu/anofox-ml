//! Parity of `HuberRegressor` (joint scale) and the L1-penalised
//! `QuantileRegressor` with sklearn 1.8.0.
//!
//! Reference script: rng = RandomState(0); X = round(normal(size=(40, 3)), 3);
//! y = round(1 + X @ [2, -1, 0.5] + standard_t(2, 40), 3); y[[3, 17]] += 15;
//! w = round(uniform(0.2, 3, 40), 3). Huber with `tol=1e-12`, Quantile with
//! `solver="highs"`.

use anofox_ml_core::{Fit, FitWeighted};
use anofox_ml_regression::{HuberRegressor, HuberScale, QuantileRegressor};
use ndarray::{Array1, Array2};

const X: [f64; 120] = [
    1.764, 0.4, 0.979, 2.241, 1.868, -0.977, 0.95, -0.151, -0.103, 0.411, 0.144, 1.454, 0.761,
    0.122, 0.444, 0.334, 1.494, -0.205, 0.313, -0.854, -2.553, 0.654, 0.864, -0.742, 2.27, -1.454,
    0.046, -0.187, 1.533, 1.469, 0.155, 0.378, -0.888, -1.981, -0.348, 0.156, 1.23, 1.202, -0.387,
    -0.302, -1.049, -1.42, -1.706, 1.951, -0.51, -0.438, -1.253, 0.777, -1.614, -0.213, -0.895,
    0.387, -0.511, -1.181, -0.028, 0.428, 0.067, 0.302, -0.634, -0.363, -0.672, -0.36, -0.813,
    -1.726, 0.177, -0.402, -1.63, 0.463, -0.907, 0.052, 0.729, 0.129, 1.139, -1.235, 0.402, -0.685,
    -0.871, -0.579, -0.312, 0.056, -1.165, 0.901, 0.466, -1.536, 1.488, 1.896, 1.179, -0.18,
    -1.071, 1.054, -0.403, 1.222, 0.208, 0.977, 0.356, 0.707, 0.011, 1.786, 0.127, 0.402, 1.883,
    -1.348, -1.27, 0.969, -1.173, 1.944, -0.414, -0.747, 1.923, 1.481, 1.868, 0.906, -0.861, 1.91,
    -0.268, 0.802, 0.947, -0.155, 0.614, 0.922,
];
const Y: [f64; 40] = [
    5.653, 2.325, 3.412, 22.359, 2.237, 1.375, 0.306, 3.354, 6.04, 0.244, 0.915, -4.633, 2.255,
    1.408, -5.899, 4.067, -2.597, 19.857, 0.388, 1.449, -7.649, -4.489, -3.744, 1.72, -3.769, 0.92,
    -1.324, 0.41, 4.298, 2.49, -1.527, 3.546, -1.641, -5.03, -3.613, 3.186, 4.742, 0.097, -0.848,
    0.61,
];
const W: [f64; 40] = [
    1.783, 0.713, 0.606, 1.567, 1.196, 2.833, 2.343, 2.296, 2.73, 0.434, 1.746, 1.837, 2.893,
    1.018, 0.874, 0.481, 0.246, 2.803, 2.076, 2.398, 0.989, 1.842, 0.379, 1.56, 2.937, 2.654,
    1.147, 2.892, 0.849, 2.858, 2.836, 2.438, 1.965, 2.648, 1.02, 2.577, 1.93, 0.237, 1.172, 0.615,
];

fn x() -> Array2<f64> {
    Array2::from_shape_vec((40, 3), X.to_vec()).unwrap()
}

fn close(got: &[f64], want: &[f64], tol: f64, ctx: &str) {
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() < tol, "{ctx}[{i}]: {g} vs sklearn {w}");
    }
}

#[test]
fn test_huber_joint_scale_matches_sklearn() {
    for (alpha, b0, coef, scale) in [
        (
            1e-4,
            0.7877531148288206,
            [2.1309291802860737, -0.9907096690491403, 0.8271360154713809],
            0.8078529777613566,
        ),
        (
            1.0,
            0.7796625669701165,
            [2.083432629725894, -0.9527816552498742, 0.8074375391437271],
            0.8133180708297761,
        ),
    ] {
        let f = HuberRegressor::new()
            .with_alpha(alpha)
            .with_tol(1e-10)
            .with_max_iter(1000)
            .fit(&x(), &Array1::from_vec(Y.to_vec()))
            .unwrap();
        let ctx = format!("huber alpha={alpha}");
        close(&[f.intercept().unwrap()], &[b0], 1e-5, &ctx);
        close(f.coefficients().as_slice().unwrap(), &coef, 1e-5, &ctx);
        close(&[f.scale()], &[scale], 1e-5, &ctx);
        assert_eq!(f.n_outliers(), 16, "{ctx}");
    }
}

#[test]
fn test_huber_mad_scale_is_opt_in() {
    let y = Array1::from_vec(Y.to_vec());
    let joint = HuberRegressor::new().fit(&x(), &y).unwrap();
    let mad = HuberRegressor::new()
        .with_scale_method(HuberScale::Mad)
        .fit(&x(), &y)
        .unwrap();
    assert!((joint.scale() - mad.scale()).abs() > 1e-3);
}

#[test]
fn test_quantile_l1_matches_sklearn() {
    let y = Array1::from_vec(Y.to_vec());
    let w = Array1::from_vec(W.to_vec());
    for (tau, alpha, weighted, b0, coef) in [
        (
            0.5,
            0.05,
            false,
            0.6715560616911778,
            [1.9991054479190593, -0.6287147953488585, 0.6195346755204127],
        ),
        (
            0.3,
            0.1,
            true,
            -0.6945291932465377,
            [
                1.9929238283139008,
                -0.02491354297635255,
                0.005393488337209817,
            ],
        ),
        (0.5, 1.0, false, 0.92, [0.0, 0.0, 0.0]),
    ] {
        let f = QuantileRegressor::new(tau)
            .with_alpha(alpha)
            .fit_weighted(&x(), &y, weighted.then_some(&w))
            .unwrap();
        let ctx = format!("quantile tau={tau} alpha={alpha} weighted={weighted}");
        close(&[f.intercept().unwrap()], &[b0], 1e-8, &ctx);
        close(f.coefficients().as_slice().unwrap(), &coef, 1e-8, &ctx);
    }
}

#[test]
fn test_quantile_default_alpha_is_sklearn_default() {
    // sklearn's default alpha = 1.0 removes every slope on this data.
    let f = QuantileRegressor::new(0.5)
        .fit(&x(), &Array1::from_vec(Y.to_vec()))
        .unwrap();
    assert!(f.coefficients().iter().all(|c| *c == 0.0));
}
