//! GPR / GPC tests, including comparisons against scikit-learn 1.8.0
//! (`GaussianProcessRegressor` / `GaussianProcessClassifier`, default
//! L-BFGS-B optimiser, `n_restarts_optimizer=0`, identical initial values).

use super::*;
use crate::classifier::classifier_log_marginal_likelihood;
use anofox_ml_core::PredictProba;
use ndarray::array;

fn data1() -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    let x = Array2::from_shape_fn((25, 1), |(i, _)| {
        0.1 * i as f64 + 0.05 * (3.0 * i as f64).sin()
    });
    let y = Array1::from_shape_fn(25, |i| {
        let v = x[[i, 0]];
        (2.0 * v).sin() + 0.3 * (7.0 * v).cos() + 0.5 + 0.1 * (37.0 * i as f64 + 1.0).sin()
    });
    let xq = array![[0.13], [0.77], [1.5], [2.31], [2.9]];
    (x, y, xq)
}

fn data2() -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    let x = Array2::from_shape_fn((30, 2), |(i, j)| {
        if j == 0 {
            (1.3 * i as f64).sin()
        } else {
            (0.7 * i as f64).cos() * 1.5
        }
    });
    let y = Array1::from_shape_fn(30, |i| {
        let (a, b) = (x[[i, 0]], x[[i, 1]]);
        a * a - 0.5 * b + 0.2 * (5.0 * a * b).sin()
    });
    let xq = array![[0.1, 0.2], [-0.5, 1.0], [0.9, -1.2]];
    (x, y, xq)
}

fn data_clf() -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    let x = Array2::from_shape_fn((40, 2), |(i, j)| {
        if j == 0 {
            (0.9 * i as f64).sin() * 2.0
        } else {
            (1.7 * i as f64).cos() * 2.0
        }
    });
    let y = Array1::from_shape_fn(40, |i| {
        let (a, b) = (x[[i, 0]], x[[i, 1]]);
        if a + 0.5 * b + 0.6 * (3.0 * b).sin() > 0.0 {
            1.0
        } else {
            0.0
        }
    });
    let xq = array![[0.0, 0.0], [1.0, -0.5], [-1.5, 0.7], [0.3, 0.3]];
    (x, y, xq)
}

fn assert_close(a: &[f64], b: &[f64], rtol: f64, atol: f64, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= atol + rtol * y.abs(),
            "{what}[{i}]: ours {x} vs sklearn {y}"
        );
    }
}

/// Compare fitted hyperparameters in log space (|Δθ| ≤ tol).
fn assert_theta(ours: &[f64], sk: &[f64], tol: f64, what: &str) {
    for (i, (a, b)) in ours.iter().zip(sk).enumerate() {
        assert!(
            (a.ln() - b.ln()).abs() <= tol,
            "{what} param {i}: ours {a} vs sklearn {b}"
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn check_regressor(
    fitted: &FittedGaussianProcessRegressor,
    xq: &Array2<f64>,
    params: &[f64],
    lml: f64,
    mean: &[f64],
    std: &[f64],
    cov: &[f64],
    name: &str,
) {
    assert_theta(&fitted.kernel.params(), params, 1e-4, name);
    assert!(
        (fitted.log_marginal_likelihood_value() - lml).abs() < 1e-6 * lml.abs().max(1.0),
        "{name} lml: {} vs {lml}",
        fitted.log_marginal_likelihood_value()
    );
    let (m, s) = fitted.predict_with_std(xq).unwrap();
    assert_close(
        m.as_slice().unwrap(),
        mean,
        1e-4,
        1e-5,
        &format!("{name} mean"),
    );
    assert_close(
        s.as_slice().unwrap(),
        std,
        1e-3,
        1e-6,
        &format!("{name} std"),
    );
    let (_, c) = fitted.predict_cov(xq).unwrap();
    let c: Vec<f64> = c.iter().copied().collect();
    assert_close(&c, cov, 2e-3, 1e-7, &format!("{name} cov"));
}

#[test]
fn sklearn_rbf_plus_white_optimised() {
    // sklearn: C(1.0) * RBF(1.0) + WhiteKernel(0.1)
    //  → 1.01**2 * RBF(length_scale=0.38) + WhiteKernel(noise_level=0.000485)
    let (x, y, xq) = data1();
    let kernel = GpKernel::Rbf {
        length_scale: 1.0,
        signal_var: 1.0,
    }
    .add(GpKernel::White { noise_level: 0.1 });
    let fitted = GaussianProcessRegressor::new(kernel).fit(&x, &y).unwrap();
    // ours: [ls, signal_var, noise]; sklearn theta: [c, ls, noise]
    check_regressor(
        &fitted,
        &xq,
        &[
            0.38028352853797115,
            1.0165906315281361,
            0.0004847127106496837,
        ],
        24.109772303456747,
        &[
            0.9423876847461392,
            1.7784271108994205,
            0.5024115128014707,
            -0.8180385149210254,
            0.36075956831945344,
        ],
        &[
            0.027054496330831616,
            0.025834528978624663,
            0.025830461038829595,
            0.027590329586731965,
            0.6578218266470507,
        ],
        &[
            0.0007319457717156475,
            1.850240505682499e-05,
            -4.312985044301358e-06,
            9.814366114137599e-07,
            1.7543128919929543e-05,
            1.850240505682499e-05,
            0.0006674228875471755,
            1.1475812886907466e-05,
            1.069658903916481e-06,
            -0.000132731859213891,
            -4.312985044301358e-06,
            1.1475812886907466e-05,
            0.0006672127174789377,
            -1.4513017793343908e-05,
            0.0004106286448582982,
            9.814366114137599e-07,
            1.069658903916481e-06,
            -1.4513017793343908e-05,
            0.0007612262867044972,
            -0.0038770554025647774,
            1.7543128919929543e-05,
            -0.000132731859213891,
            0.0004106286448582982,
            -0.0038770554025647774,
            0.43272955561326254,
        ],
        "rbf_white",
    );
}

#[test]
fn sklearn_rbf_plus_white_noise_hits_lower_bound() {
    // Noise-free targets: sklearn drives the White noise to its lower bound 1e-5.
    let (x, _, xq) = data1();
    let y = x.column(0).mapv(|v| (2.0 * v).sin());
    let kernel = GpKernel::Rbf {
        length_scale: 1.0,
        signal_var: 1.0,
    }
    .add(GpKernel::White { noise_level: 0.1 });
    let fitted = GaussianProcessRegressor::new(kernel).fit(&x, &y).unwrap();
    let p = fitted.kernel.params();
    assert!(
        (p[2] - 1e-5).abs() < 1e-12,
        "noise must sit on its bound: {p:?}"
    );
    assert_theta(
        &p,
        &[1.2086861162439841, 2.5714560346736777, 1e-5],
        1e-4,
        "bound",
    );
    assert!((fitted.log_marginal_likelihood_value() - 85.36699466427744).abs() < 1e-5);
    let (m, _) = fitted.predict_with_std(&xq).unwrap();
    assert_close(
        m.as_slice().unwrap(),
        &[
            0.2576350241518108,
            0.9993375918458611,
            0.140946944566392,
            -0.9956174652208675,
            -0.4703731231470899,
        ],
        1e-4,
        1e-5,
        "bound mean",
    );
}

#[test]
fn sklearn_normalize_y_respected_in_lml() {
    let (x, y, xq) = data1();
    let kernel = GpKernel::Rbf {
        length_scale: 1.0,
        signal_var: 1.0,
    }
    .add(GpKernel::White { noise_level: 0.1 });
    let fitted = GaussianProcessRegressor::new(kernel)
        .with_normalize_y(true)
        .fit(&x, &y)
        .unwrap();
    check_regressor(
        &fitted,
        &xq,
        &[
            0.36566498076806797,
            1.2864366765706827,
            0.0008328607335754002,
        ],
        18.08065691340357,
        &[
            0.9407727391889523,
            1.7789074894442063,
            0.5022523096558342,
            -0.8188884316253605,
            0.6091212740885765,
        ],
        &[
            0.027375795512431043,
            0.026206128292460065,
            0.02619939977079528,
            0.02791827826039372,
            0.6092476229699756,
        ],
        &[
            0.0007494341799384396,
            1.7513986753658668e-05,
            -4.263995753113496e-06,
            4.21582981357604e-07,
            2.337674418637403e-05,
            1.7513986753658668e-05,
            0.0006867611600810082,
            9.347357078351747e-06,
            2.1701271973041772e-06,
            -9.834417788196409e-05,
            -4.263995753113496e-06,
            9.347357078351747e-06,
            0.0006864085483499477,
            -1.535851106945427e-05,
            0.00028986535382979797,
            4.21582981357604e-07,
            2.1701271973041772e-06,
            -1.535851106945427e-05,
            0.0007794302610247727,
            -0.003463310390818428,
            2.337674418637403e-05,
            -9.834417788196409e-05,
            0.00028986535382979797,
            -0.003463310390818428,
            0.37118266609456546,
        ],
        "rbf_white_norm",
    );
}

#[test]
fn sklearn_matern15_optimised() {
    // sklearn: C(1.0) * Matern(1.0, nu=1.5), alpha=1e-2
    let (x, y, xq) = data2();
    let kernel = GpKernel::Matern {
        length_scale: 1.0,
        signal_var: 1.0,
        nu: 1.5,
    };
    let fitted = GaussianProcessRegressor::new(kernel)
        .with_alpha(1e-2)
        .fit(&x, &y)
        .unwrap();
    check_regressor(
        &fitted,
        &xq,
        &[2.9450278958404223, 3.67660522586456],
        -3.374252493405507,
        &[
            -0.06749687770805135,
            -0.2620660283183973,
            1.2045865565872784,
        ],
        &[
            0.08240653605327784,
            0.10814591129109705,
            0.25728194411372124,
        ],
        &[
            0.006790837184299736,
            0.00037268914983235035,
            -9.457188540418215e-05,
            0.00037268914983235035,
            0.011695538128980942,
            0.00014178378584484008,
            -9.457188540418215e-05,
            0.00014178378584484008,
            0.06619399876693643,
        ],
        "matern15",
    );
}

#[test]
fn sklearn_matern25_ard_plus_white_optimised() {
    // sklearn: C(1.0) * Matern([1.0, 1.0], nu=2.5) + WhiteKernel(0.1)
    let (x, y, xq) = data2();
    let kernel = GpKernel::MaternArd {
        length_scales: vec![1.0, 1.0],
        signal_var: 1.0,
        nu: 2.5,
    }
    .add(GpKernel::White { noise_level: 0.1 });
    let fitted = GaussianProcessRegressor::new(kernel).fit(&x, &y).unwrap();
    check_regressor(
        &fitted,
        &xq,
        &[
            0.7986430950204234,
            2.7115194138160605,
            1.2915974270660353,
            0.00014867313009909788,
        ],
        6.509265868545963,
        &[
            -0.06897586692579516,
            -0.32222482723832613,
            1.3354202956023773,
        ],
        &[
            0.040124963863410794,
            0.05185508363620012,
            0.1172647464090368,
        ],
        &[
            0.001610012725040022,
            2.9030686811370998e-05,
            0.00014505646121298899,
            2.9030686811370998e-05,
            0.002688949698917309,
            -2.4801823568398884e-05,
            0.00014505646121298899,
            -2.4801823568398884e-05,
            0.013751020750375487,
        ],
        "matern25_ard_white",
    );
}

#[test]
fn sklearn_rbf_ard_optimised() {
    // sklearn: C(1.0) * RBF([1.0, 1.0]), alpha=1e-3
    let (x, y, xq) = data2();
    let kernel = GpKernel::RbfArd {
        length_scales: vec![1.0, 1.0],
        signal_var: 1.0,
    };
    let fitted = GaussianProcessRegressor::new(kernel)
        .with_alpha(1e-3)
        .fit(&x, &y)
        .unwrap();
    check_regressor(
        &fitted,
        &xq,
        &[0.44939633033504156, 1.6174634076049887, 0.9147241753060528],
        5.125913879792403,
        &[
            -0.06098690169677745,
            -0.31978754138524934,
            1.2956136724417422,
        ],
        &[0.028782303185397467, 0.0423633239146666, 0.1022611311664003],
        &[
            0.0008284209766560302,
            6.696316110721412e-06,
            2.527015887787698e-05,
            6.696316110721412e-06,
            0.0017946512130987413,
            -0.00014060474547245494,
            2.527015887787698e-05,
            -0.00014060474547245494,
            0.010457338947431727,
        ],
        "rbf_ard",
    );
}

#[test]
fn sklearn_lml_gradient_matern05_white() {
    // sklearn: (C(1.3) * Matern(0.8, nu=0.5) + WhiteKernel(0.2)).theta, eval_gradient=True
    let (x, y, _) = data1();
    let kernel = GpKernel::Matern {
        length_scale: 0.8,
        signal_var: 1.3,
        nu: 0.5,
    }
    .add(GpKernel::White { noise_level: 0.2 });
    let (lml, g) = log_marginal_likelihood_with_gradient(&x, &y, &kernel, 1e-10).unwrap();
    assert!((lml - -18.42395783284335).abs() < 1e-9, "{lml}");
    // sklearn order [c, ls, noise] → ours [ls, c, noise]
    assert_close(
        &g,
        &[3.615504819227777, -4.103598953867644, -6.1488690893229005],
        1e-8,
        1e-9,
        "grad",
    );
}

#[test]
fn sklearn_rational_quadratic_fixed_and_gradient() {
    // sklearn: C(1.3) * RQ(length_scale=0.7, alpha=0.8) + WhiteKernel(0.05), optimizer=None
    let (x, y, xq) = data1();
    let kernel = GpKernel::RationalQuadratic {
        length_scale: 0.7,
        signal_var: 1.3,
        alpha: 0.8,
    }
    .add(GpKernel::White { noise_level: 0.05 });
    let fitted = GaussianProcessRegressor::new(kernel.clone())
        .with_optimizer(false)
        .fit(&x, &y)
        .unwrap();
    let (lml, g) = fitted.log_marginal_likelihood(&kernel.theta()).unwrap();
    assert!((lml - -5.649715359651832).abs() < 1e-9);
    assert!((fitted.log_marginal_likelihood_value() - lml).abs() < 1e-12);
    // sklearn order [c, alpha, ls, noise] → ours [ls, c, alpha, noise]
    assert_close(
        &g,
        &[
            -11.536264057107248,
            1.652950761793355,
            -2.121549214879439,
            -5.272023411696327,
        ],
        1e-8,
        1e-9,
        "rq grad",
    );
    let (m, s) = fitted.predict_with_std(&xq).unwrap();
    assert_close(
        m.as_slice().unwrap(),
        &[
            0.8463862570312979,
            1.6408601226253685,
            0.5647533837087902,
            -0.7616626038444945,
            -0.3755858958676849,
        ],
        1e-9,
        1e-10,
        "rq mean",
    );
    assert_close(
        s.as_slice().unwrap(),
        &[
            0.2512443750747579,
            0.24738473104430433,
            0.24734929767322397,
            0.2553645536900585,
            0.6919766302047058,
        ],
        1e-8,
        1e-10,
        "rq std",
    );
}

#[test]
fn sklearn_gpc_fixed_kernel() {
    // sklearn: GPC(C(1.0) * RBF(1.0), optimizer=None)
    let (x, y, xq) = data_clf();
    let fitted = GaussianProcessClassifier::new(GpKernel::Rbf {
        length_scale: 1.0,
        signal_var: 1.0,
    })
    .with_optimizer(false)
    .fit(&x, &y)
    .unwrap();
    assert!((fitted.log_marginal_likelihood_value() - -21.127808986993077).abs() < 1e-8);
    let p = fitted.predict_proba(&xq).unwrap();
    assert_close(
        &p.column(1).to_vec(),
        &[
            0.4138695657433402,
            0.5651178249491835,
            0.2329254023772478,
            0.5450560220580059,
        ],
        1e-8,
        1e-10,
        "gpc proba",
    );
}

#[test]
fn sklearn_gpc_gradient_and_optimised() {
    let (x, y, xq) = data_clf();
    let kernel = GpKernel::Rbf {
        length_scale: 1.0,
        signal_var: 1.0,
    };
    // log_marginal_likelihood(np.log([2.0, 0.7])) with sklearn order [c, ls]
    let (lml, g) =
        classifier_log_marginal_likelihood(&x, &y, &kernel, &[0.7_f64.ln(), 2.0_f64.ln()]).unwrap();
    assert!((lml - -20.90933501663524).abs() < 1e-8, "{lml}");
    assert_close(
        &g,
        &[5.106818314244068, 2.4271274802281013],
        1e-6,
        1e-8,
        "gpc grad",
    );

    let fitted = GaussianProcessClassifier::new(kernel).fit(&x, &y).unwrap();
    assert_theta(
        &fitted.kernel.params(),
        &[4.442358179193139, 88.33481697921015],
        1e-3,
        "gpc theta",
    );
    assert!((fitted.log_marginal_likelihood_value() - -14.153456367342756).abs() < 1e-6);
    let p = fitted.predict_proba(&xq).unwrap();
    assert_close(
        &p.column(1).to_vec(),
        &[
            0.27765836863943605,
            0.7533830777502999,
            0.035942969321581586,
            0.5237325704139124,
        ],
        1e-3,
        1e-5,
        "gpc opt proba",
    );
}

#[test]
fn sklearn_gpc_multiclass_fixed_kernel() {
    let (x, _, xq) = data_clf();
    let y = Array1::from_shape_fn(x.nrows(), |i| {
        if x[[i, 0]] > 0.7 {
            2.0
        } else if x[[i, 1]] > 0.0 {
            1.0
        } else {
            0.0
        }
    });
    let fitted = MulticlassGaussianProcessClassifier::new(GpKernel::Rbf {
        length_scale: 1.0,
        signal_var: 1.0,
    })
    .with_optimizer(false)
    .fit(&x, &y)
    .unwrap();
    let p: Vec<f64> = fitted.predict_proba(&xq).unwrap().iter().copied().collect();
    assert_close(
        &p,
        &[
            0.34585544121633577,
            0.37427182487605126,
            0.27987273390761297,
            0.27426453985522947,
            0.1716074677229654,
            0.5541279924218051,
            0.20097240588792772,
            0.628824416804861,
            0.17020317730721124,
            0.2740071851433423,
            0.3707367297562546,
            0.3552560851004031,
        ],
        1e-8,
        1e-10,
        "gpc multiclass",
    );
}

#[test]
fn gpc_objective_is_finite_for_large_latents() {
    // Regression test for the old `-(-f).ln_1p()` term (NaN for f > 1):
    // well-separated, many points → large latent values; the LML must be
    // finite and the Newton iteration must converge to a valid mode.
    let n = 30;
    let x = Array2::from_shape_fn((n, 1), |(i, _)| i as f64 * 0.2);
    let y = Array1::from_shape_fn(n, |i| if i >= n / 2 { 1.0 } else { 0.0 });
    let fitted = GaussianProcessClassifier::new(GpKernel::Rbf {
        length_scale: 1.0,
        signal_var: 100.0,
    })
    .with_optimizer(false)
    .fit(&x, &y)
    .unwrap();
    assert!(fitted.log_marginal_likelihood_value().is_finite());
    let (mean, _) = fitted.latent_mean_var(x.view()).unwrap();
    assert!(
        mean.iter().any(|&f| f > 1.0),
        "latents should exceed 1: {mean:?}"
    );
    let pred = fitted.predict(&x).unwrap();
    assert_eq!(pred, y);
}

#[test]
fn restarts_never_worse_and_deterministic() {
    let (x, y, _) = data1();
    let kernel = GpKernel::Rbf {
        length_scale: 50.0,
        signal_var: 1.0,
    }
    .add(GpKernel::White { noise_level: 1.0 });
    let base = GaussianProcessRegressor::new(kernel.clone())
        .fit(&x, &y)
        .unwrap();
    let r1 = GaussianProcessRegressor::new(kernel.clone())
        .with_n_restarts_optimizer(4)
        .with_seed(7)
        .fit(&x, &y)
        .unwrap();
    let r2 = GaussianProcessRegressor::new(kernel)
        .with_n_restarts_optimizer(4)
        .with_seed(7)
        .fit(&x, &y)
        .unwrap();
    assert!(r1.log_marginal_likelihood_value() >= base.log_marginal_likelihood_value() - 1e-9);
    assert_eq!(r1.kernel.params(), r2.kernel.params());
}

#[test]
fn fixed_kernel_is_not_optimised() {
    let (x, y, _) = data1();
    let kernel = GpKernel::Rbf {
        length_scale: 1.0,
        signal_var: 1.0,
    }
    .with_param_bounds(vec![ParamBounds::default(), ParamBounds::Fixed])
    .add(GpKernel::White { noise_level: 0.1 }.fixed());
    let fitted = GaussianProcessRegressor::new(kernel).fit(&x, &y).unwrap();
    let p = fitted.kernel.params();
    assert_eq!(p[1], 1.0);
    assert_eq!(p[2], 0.1);
    assert!((p[0] - 1.0).abs() > 1e-3, "length scale should move: {p:?}");
}

#[test]
fn white_kernel_does_not_add_noise_to_test_points_equal_to_training() {
    // sklearn semantics: K(X*, X) has no White contribution, so the mean
    // at a training point is not pulled towards that point's target.
    let (x, y, _) = data1();
    let rbf = GpKernel::Rbf {
        length_scale: 0.4,
        signal_var: 1.0,
    };
    let with_white =
        GaussianProcessRegressor::new(rbf.clone().add(GpKernel::White { noise_level: 0.05 }))
            .with_optimizer(false)
            .fit(&x, &y)
            .unwrap();
    let as_alpha = GaussianProcessRegressor::new(rbf)
        .with_alpha(0.05 + 1e-10)
        .with_optimizer(false)
        .fit(&x, &y)
        .unwrap();
    let a = with_white.predict(&x).unwrap();
    let b = as_alpha.predict(&x).unwrap();
    for (p, q) in a.iter().zip(b.iter()) {
        assert!((p - q).abs() < 1e-10, "{p} vs {q}");
    }
}

#[test]
fn unsupported_matern_nu_errors_in_fit() {
    let (x, y, _) = data1();
    let gp = GaussianProcessRegressor::new(GpKernel::Matern {
        length_scale: 1.0,
        signal_var: 1.0,
        nu: 3.5,
    });
    assert!(gp.fit(&x, &y).is_err());
}

#[test]
fn cholesky_jitter_retry_handles_duplicate_points() {
    // Duplicate rows, no noise: K is singular; the jitter retry rescues the fit.
    let x = array![[0.0], [0.0], [1.0], [1.0], [2.0]];
    let y = array![0.0, 0.0, 1.0, 1.0, 0.5];
    let fitted = GaussianProcessRegressor::new(GpKernel::Rbf {
        length_scale: 1.0,
        signal_var: 1.0,
    })
    .with_alpha(0.0)
    .with_optimizer(false)
    .fit(&x, &y)
    .unwrap();
    assert!(fitted.alpha > 0.0);
    let p = fitted.predict(&x).unwrap();
    for (a, b) in p.iter().zip(y.iter()) {
        assert!((a - b).abs() < 1e-3);
    }
}

#[test]
fn sample_y_is_seeded_and_matches_moments() {
    let (x, y, xq) = data1();
    let fitted = GaussianProcessRegressor::new(
        GpKernel::Rbf {
            length_scale: 0.4,
            signal_var: 1.0,
        }
        .add(GpKernel::White { noise_level: 0.01 }),
    )
    .with_optimizer(false)
    .fit(&x, &y)
    .unwrap();
    let s1 = fitted.sample_y(&xq, 4000, 3).unwrap();
    let s2 = fitted.sample_y(&xq, 4000, 3).unwrap();
    assert_eq!(s1, s2);
    assert_eq!(s1.dim(), (5, 4000));
    let (m, s) = fitted.predict_with_std(&xq).unwrap();
    for i in 0..5 {
        let row = s1.row(i);
        let mean = row.mean().unwrap();
        let sd = row.std(0.0);
        assert!((mean - m[i]).abs() < 5.0 * s[i] / 4000f64.sqrt() + 1e-9);
        assert!((sd - s[i]).abs() < 0.1 * s[i] + 1e-9, "{sd} vs {}", s[i]);
    }
}

#[test]
fn predict_cov_diagonal_matches_std() {
    let (x, y, xq) = data2();
    let fitted = GaussianProcessRegressor::new(GpKernel::RbfArd {
        length_scales: vec![0.5, 1.5],
        signal_var: 1.0,
    })
    .with_alpha(1e-3)
    .with_optimizer(false)
    .fit(&x, &y)
    .unwrap();
    let (_, s) = fitted.predict_with_std(&xq).unwrap();
    let (_, c) = fitted.predict_cov(&xq).unwrap();
    for i in 0..xq.nrows() {
        assert!((c[[i, i]].max(0.0).sqrt() - s[i]).abs() < 1e-10);
    }
}

// --- legacy behaviour (fixed kernels) --------------------------------------

#[test]
fn test_gp_rbf_interpolates_with_low_noise() {
    let x = Array2::from_shape_vec((6, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let y: Array1<f64> = x.column(0).mapv(|v: f64| v.sin());
    let kernel = GpKernel::Rbf {
        length_scale: 1.0,
        signal_var: 1.0,
    };
    let fitted = GaussianProcessRegressor::new(kernel)
        .with_alpha(1e-8)
        .with_optimizer(false)
        .fit(&x, &y)
        .unwrap();
    let p = fitted.predict(&x).unwrap();
    for i in 0..6 {
        assert!((p[i] - y[i]).abs() < 1e-4, "[{i}]: {} vs {}", p[i], y[i]);
    }
}

#[test]
fn test_gp_matern_nu_2p5_interpolates() {
    let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
    let y: Array1<f64> = x.column(0).mapv(|v: f64| v.cos());
    let kernel = GpKernel::Matern {
        length_scale: 1.0,
        signal_var: 1.0,
        nu: 2.5,
    };
    let fitted = GaussianProcessRegressor::new(kernel)
        .with_alpha(1e-8)
        .fit(&x, &y)
        .unwrap();
    let p = fitted.predict(&x).unwrap();
    for i in 0..5 {
        assert!((p[i] - y[i]).abs() < 1e-3, "[{i}]: {} vs {}", p[i], y[i]);
    }
}

#[test]
fn test_gp_rational_quadratic_runs() {
    let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
    let y = array![0.0, 1.0, 0.5, -0.5, 0.0];
    let kernel = GpKernel::RationalQuadratic {
        length_scale: 1.0,
        signal_var: 1.0,
        alpha: 0.5,
    };
    let fitted = GaussianProcessRegressor::new(kernel)
        .with_alpha(1e-6)
        .fit(&x, &y)
        .unwrap();
    let p = fitted.predict(&x).unwrap();
    assert!(p.iter().all(|v| v.is_finite()));
}

#[test]
fn test_optimize_rbf_length_scale_picks_sensible_value() {
    let x = Array2::from_shape_vec((20, 1), (0..20).map(|i| (i as f64) * 0.3).collect()).unwrap();
    let y: Array1<f64> = x.column(0).mapv(|v: f64| v.sin());
    let best = optimize_rbf_length_scale(&x, &y, 1.0, 1e-6, -2.0, 2.0, 30).unwrap();
    assert!(best > 0.3 && best < 4.0, "best length_scale = {best}");
}

#[test]
fn test_optimize_kernel_lbfgs_rbf_two_params() {
    let x = Array2::from_shape_vec((20, 1), (0..20).map(|i| (i as f64) * 0.3).collect()).unwrap();
    let y: Array1<f64> = x.column(0).mapv(|v: f64| v.sin());
    let res = optimize_kernel_lbfgs(
        &x,
        &y,
        1e-6,
        &[0.0, 0.0],
        |p| GpKernel::Rbf {
            length_scale: p[0].exp(),
            signal_var: p[1].exp(),
        },
        50,
        1e-4,
        1e-3,
    )
    .unwrap();
    let lml_init = log_marginal_likelihood(
        &x,
        &y,
        &GpKernel::Rbf {
            length_scale: 1.0,
            signal_var: 1.0,
        },
        1e-6,
    )
    .unwrap();
    assert!(res.log_marginal_likelihood >= lml_init - 1e-9);
    let ls_opt = res.log_params[0].exp();
    assert!(ls_opt > 0.1 && ls_opt < 20.0, "length_scale {ls_opt}");
}

#[test]
fn test_log_marginal_likelihood_monotonic_in_data() {
    let x = Array2::from_shape_vec((10, 1), (0..10).map(|i| i as f64 * 0.3).collect()).unwrap();
    let y: Array1<f64> = x.column(0).mapv(|v: f64| v.sin());
    let good = GpKernel::Rbf {
        length_scale: 1.0,
        signal_var: 1.0,
    };
    let bad = GpKernel::Rbf {
        length_scale: 100.0,
        signal_var: 1.0,
    };
    let lml_good = log_marginal_likelihood(&x, &y, &good, 1e-6).unwrap();
    let lml_bad = log_marginal_likelihood(&x, &y, &bad, 1e-6).unwrap();
    assert!(lml_good > lml_bad, "good={lml_good}, bad={lml_bad}");
}

#[test]
fn test_gp_sum_of_kernels() {
    let x = Array2::from_shape_vec((10, 1), (0..10).map(|i| i as f64).collect()).unwrap();
    let y: Array1<f64> = x.column(0).mapv(|v: f64| v.sin() + 0.05);
    let kernel = GpKernel::Rbf {
        length_scale: 2.0,
        signal_var: 1.0,
    }
    .add(GpKernel::White { noise_level: 0.01 });
    let fitted = GaussianProcessRegressor::new(kernel)
        .with_alpha(1e-8)
        .fit(&x, &y)
        .unwrap();
    let p = fitted.predict(&x).unwrap();
    for (a, b) in p.iter().zip(y.iter()) {
        assert!((a - b).abs() < 0.1, "predict {a} vs y {b}");
    }
}

#[test]
fn test_gpc_separates_two_clusters() {
    let mut x_data = Vec::new();
    let mut y_data = Vec::new();
    for i in 0..6 {
        let f = i as f64 * 0.1;
        x_data.extend([f, f + 0.1]);
        y_data.push(0.0);
        x_data.extend([5.0 + f, 5.0 - f]);
        y_data.push(1.0);
    }
    let x = Array2::from_shape_vec((12, 2), x_data).unwrap();
    let y = Array1::from_vec(y_data);
    let fitted = GaussianProcessClassifier::new(GpKernel::Rbf {
        length_scale: 2.0,
        signal_var: 1.0,
    })
    .with_max_iter(50)
    .fit(&x, &y)
    .unwrap();
    let preds = fitted.predict(&x).unwrap();
    assert_eq!(preds, y);
    let proba = fitted.predict_proba(&x).unwrap();
    for i in 0..12 {
        assert!((proba[[i, 0]] + proba[[i, 1]] - 1.0).abs() < 1e-9);
    }
}

#[test]
fn test_multiclass_gpc_three_classes() {
    let n_per = 6;
    let mut x_data = Vec::new();
    let mut y_data = Vec::new();
    for i in 0..n_per {
        let f = i as f64 * 0.1;
        x_data.extend([f, f]);
        y_data.push(0.0);
        x_data.extend([5.0 + f, f]);
        y_data.push(1.0);
        x_data.extend([f, 5.0 + f]);
        y_data.push(2.0);
    }
    let x = Array2::from_shape_vec((n_per * 3, 2), x_data).unwrap();
    let y = Array1::from_vec(y_data);
    let fitted = MulticlassGaussianProcessClassifier::new(GpKernel::Rbf {
        length_scale: 2.0,
        signal_var: 1.0,
    })
    .with_max_iter(50)
    .fit(&x, &y)
    .unwrap();
    let preds = fitted.predict(&x).unwrap();
    let correct = preds
        .iter()
        .zip(y.iter())
        .filter(|(p, t)| (*p - *t).abs() < 0.5)
        .count();
    assert!(
        correct >= (n_per * 3) * 9 / 10,
        "got {correct}/{}",
        n_per * 3
    );
    let p = fitted.predict_proba(&x).unwrap();
    for i in 0..(n_per * 3) {
        let s: f64 = (0..3).map(|c| p[[i, c]]).sum();
        assert!((s - 1.0).abs() < 1e-9);
    }
}
