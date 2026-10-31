//! Parity of `LogisticRegressor` with sklearn 1.8.0 `LogisticRegression`
//! (`tol=1e-12`; lbfgs, saga for L1 / elastic-net) and
//! `OneVsRestClassifier(LogisticRegression())`, plus statsmodels 0.14
//! `Logit` for the Wald inference.
//!
//! Reference script: rng = RandomState(42); X = round(normal(60, 3), 3);
//! binary / imbalanced / 3-class labels from a logistic / Gumbel-max model;
//! w = round(uniform(0.2, 3), 2).

use anofox_ml_core::{ClassifierScore, DecisionFunction, Fit, FitWeighted, Predict, PredictProba};
use anofox_ml_regression::{ClassWeight, LogisticRegressor, MultiClass, StatisticKind};
use ndarray::{Array1, Array2};

const X: [f64; 180] = [
    0.497, -0.138, 0.648, 1.523, -0.234, -0.234, 1.579, 0.767, -0.469, 0.543, -0.463, -0.466,
    0.242, -1.913, -1.725, -0.562, -1.013, 0.314, -0.908, -1.412, 1.466, -0.226, 0.068, -1.425,
    -0.544, 0.111, -1.151, 0.376, -0.601, -0.292, -0.602, 1.852, -0.013, -1.058, 0.823, -1.221,
    0.209, -1.96, -1.328, 0.197, 0.738, 0.171, -0.116, -0.301, -1.479, -0.72, -0.461, 1.057, 0.344,
    -1.763, 0.324, -0.385, -0.677, 0.612, 1.031, 0.931, -0.839, -0.309, 0.331, 0.976, -0.479,
    -0.186, -1.106, -1.196, 0.813, 1.356, -0.072, 1.004, 0.362, -0.645, 0.361, 1.538, -0.036,
    1.565, -2.62, 0.822, 0.087, -0.299, 0.092, -1.988, -0.22, 0.357, 1.478, -0.518, -0.808, -0.502,
    0.915, 0.329, -0.53, 0.513, 0.097, 0.969, -0.702, -0.328, -0.392, -1.464, 0.296, 0.261, 0.005,
    -0.235, -1.415, -0.421, -0.343, -0.802, -0.161, 0.404, 1.886, 0.175, 0.258, -0.074, -1.919,
    -0.027, 0.06, 2.463, -0.192, 0.302, -0.035, -1.169, 1.143, 0.752, 0.791, -0.909, 1.403, -1.402,
    0.587, 2.19, -0.991, -0.566, 0.1, -0.503, -1.551, 0.069, -1.062, 0.474, -0.919, 1.55, -0.783,
    -0.322, 0.814, -1.231, 0.227, 1.307, -1.607, 0.185, 0.26, 0.782, -1.237, -1.32, 0.522, 0.297,
    0.25, 0.346, -0.68, 0.232, 0.293, -0.714, 1.866, 0.474, -1.191, 0.657, -0.975, 0.787, 1.159,
    -0.821, 0.963, 0.413, 0.822, 1.897, -0.245, -0.754, -0.89, -0.816, -0.077, 0.341, 0.277, 0.827,
    0.013, 1.454, -0.265, 2.72,
];
const YB: [f64; 60] = [
    1., 1., 1., 1., 1., 1., 0., 0., 0., 1., 1., 0., 1., 0., 0., 0., 1., 1., 0., 1., 1., 0., 0., 0.,
    0., 1., 1., 0., 1., 1., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1., 1., 1.,
    1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 1.,
];
const YI: [f64; 60] = [
    1., 1., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0.,
    0., 1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 1., 1., 1.,
    0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 1.,
];
const Y3: [f64; 60] = [
    2., 0., 2., 1., 0., 0., 2., 1., 1., 2., 1., 1., 0., 1., 0., 1., 0., 0., 2., 1., 2., 2., 2., 2.,
    0., 0., 0., 2., 2., 0., 1., 0., 0., 2., 0., 0., 0., 2., 0., 2., 2., 2., 2., 2., 0., 0., 0., 2.,
    1., 2., 0., 0., 0., 0., 0., 0., 0., 1., 0., 2.,
];
const W: [f64; 60] = [
    0.4, 2.0, 0.27, 1.84, 2.83, 1.81, 1.29, 2.0, 1.48, 1.73, 2.84, 1.28, 2.89, 2.73, 0.75, 0.39,
    0.48, 0.25, 0.46, 2.11, 0.4, 1.09, 2.57, 0.27, 2.48, 0.99, 0.53, 2.15, 1.96, 2.66, 2.26, 2.45,
    0.99, 0.7, 2.3, 2.46, 2.97, 1.36, 1.24, 2.37, 1.15, 2.81, 2.6, 1.4, 2.3, 2.31, 0.49, 2.73,
    1.61, 2.51, 1.1, 2.71, 1.29, 0.23, 2.74, 0.46, 1.09, 2.86, 2.86, 1.81,
];

fn x() -> Array2<f64> {
    Array2::from_shape_vec((60, 3), X.to_vec()).unwrap()
}
fn a(v: &[f64]) -> Array1<f64> {
    Array1::from_vec(v.to_vec())
}

fn assert_close(actual: &[f64], expected: &[f64], tol: f64, ctx: &str) {
    assert_eq!(actual.len(), expected.len(), "{ctx}: length");
    for (i, (u, v)) in actual.iter().zip(expected).enumerate() {
        assert!((u - v).abs() < tol, "{ctx}[{i}]: {u} vs sklearn {v}");
    }
}

fn binary_check(
    m: LogisticRegressor,
    y: &[f64],
    w: Option<&Array1<f64>>,
    b: f64,
    coef: [f64; 3],
    tol: f64,
    ctx: &str,
) {
    let f = m.fit_weighted(&x(), &a(y), w).unwrap();
    assert!(f.converged(), "{ctx}: not converged");
    assert_close(&[f.intercept().unwrap()], &[b], tol, ctx);
    assert_close(f.coefficients().as_slice().unwrap(), &coef, tol, ctx);
}

#[test]
fn test_binary_l2_c1_matches_sklearn() {
    binary_check(
        LogisticRegressor::new(),
        &YB,
        None,
        0.337744297594217,
        [1.1165985805533376, -0.9543015443975524, 0.26609388872584333],
        1e-8,
        "C=1",
    );
}

#[test]
fn test_binary_l2_sample_weight_matches_sklearn() {
    binary_check(
        LogisticRegressor::new().with_c(0.1),
        &YB,
        Some(&a(&W)),
        0.32053997286316904,
        [0.5532091081819757, -0.590022629125072, 0.2265919455301859],
        1e-6,
        "C=0.1 weighted",
    );
}

#[test]
fn test_binary_balanced_class_weight_matches_sklearn() {
    binary_check(
        LogisticRegressor::new().with_class_weight(ClassWeight::Balanced),
        &YI,
        None,
        -0.5274226209076037,
        [1.229957939689, -1.1642419324118447, 0.9051691078490398],
        1e-8,
        "balanced",
    );
}

#[test]
#[ignore = "L1 / elastic-net logistic needs anofox-regression support (#72)"]
fn test_binary_l1_and_elastic_net_match_sklearn_saga() {
    binary_check(
        LogisticRegressor::new().with_c(0.5).with_l1_ratio(1.0),
        &YB,
        None,
        0.3132634648349461,
        [0.8906884616282699, -0.8214379009126599, 0.10453903054606069],
        1e-6,
        "L1",
    );
    binary_check(
        LogisticRegressor::new().with_c(0.5).with_l1_ratio(0.5),
        &YB,
        None,
        0.31857543987621756,
        [0.9231724614391265, -0.8374083313220979, 0.17026742083773183],
        1e-6,
        "elastic-net",
    );
}

#[test]
#[ignore = "L1 / elastic-net logistic needs anofox-regression support (#72)"]
fn test_l1_produces_exact_zeros() {
    let f = LogisticRegressor::new()
        .with_c(0.05)
        .with_l1_ratio(1.0)
        .fit(&x(), &a(&YB))
        .unwrap();
    assert!(
        f.coefficients().iter().any(|c| *c == 0.0),
        "{:?}",
        f.coefficients()
    );
}

#[test]
#[ignore = "multinomial logistic needs anofox-regression support (#72)"]
fn test_multinomial_matches_sklearn() {
    let f = LogisticRegressor::new()
        .with_multi_class(MultiClass::Multinomial)
        .fit(&x(), &a(&Y3))
        .unwrap();
    assert!(f.converged());
    assert_eq!(f.classes(), &[0.0, 1.0, 2.0]);
    assert_eq!(f.coef().dim(), (3, 3));
    assert_close(
        f.coef().as_slice().unwrap(),
        &[
            0.5453637893962024,
            -0.35904004469501577,
            -0.34214973475422816,
            -0.4684985998162452,
            0.4253202613010895,
            -0.2564204079806024,
            -0.07686518957995704,
            -0.0662802166060746,
            0.5985701427348307,
        ],
        1e-6,
        "multinomial coef",
    );
    assert_close(
        f.intercepts().as_slice().unwrap(),
        &[0.3925209572267445, -0.5389281686445537, 0.1464072114178093],
        1e-6,
        "multinomial intercept",
    );
    let x2 = x().slice(ndarray::s![0..2, ..]).to_owned();
    let p = PredictProba::predict_proba(&f, &x2).unwrap();
    assert_close(
        p.as_slice().unwrap(),
        &[
            0.44648086694812456,
            0.10082322102621634,
            0.45269591202565906,
            0.7717764494031475,
            0.05295976191565734,
            0.175263788681195,
        ],
        1e-7,
        "multinomial proba",
    );
    // predict = argmax proba; decision_function is (n, K).
    let pred = f.predict(&x2).unwrap();
    assert_eq!(pred.to_vec(), vec![2.0, 0.0]);
    assert_eq!(
        DecisionFunction::decision_function(&f, &x2).unwrap().dim(),
        (2, 3)
    );
    assert!(ClassifierScore::score(&f, &x(), &a(&Y3)).unwrap() > 0.5);
}

#[test]
#[ignore = "multinomial logistic needs anofox-regression support (#72)"]
fn test_multinomial_class_and_sample_weight_match_sklearn() {
    let f = LogisticRegressor::new()
        .with_multi_class(MultiClass::Multinomial)
        .with_c(0.3)
        .with_class_weight(ClassWeight::Balanced)
        .fit_weighted(&x(), &a(&Y3), Some(&a(&W)))
        .unwrap();
    assert_close(
        f.coef().as_slice().unwrap(),
        &[
            0.5403630186621949,
            -0.23958939966554169,
            -0.5387714689432208,
            -0.2741975726414061,
            0.4361152759177665,
            -0.23749995060850293,
            -0.26616544602078873,
            -0.19652587625222506,
            0.776271419551724,
        ],
        1e-6,
        "weighted multinomial coef",
    );
    assert_close(
        f.intercepts().as_slice().unwrap(),
        &[
            -0.006919217444412793,
            0.011212319124491712,
            -0.0042931016800790735,
        ],
        1e-6,
        "weighted multinomial intercept",
    );
}

#[test]
fn test_one_vs_rest_matches_sklearn() {
    let f = LogisticRegressor::new()
        .with_multi_class(MultiClass::OneVsRest)
        .fit(&x(), &a(&Y3))
        .unwrap();
    assert_close(
        f.coef().as_slice().unwrap(),
        &[
            0.7120195253401368,
            -0.47478327449176155,
            -0.6081769198665397,
            -0.6459321607583127,
            0.610942766687831,
            -0.3515056968359624,
            -0.28966692667345184,
            0.04166059660834627,
            0.8659566803338837,
        ],
        1e-8,
        "ovr coef",
    );
    assert_close(
        f.intercepts().as_slice().unwrap(),
        &[
            -0.27711830039791413,
            -1.6405603167756395,
            -0.6952561885226719,
        ],
        1e-8,
        "ovr intercept",
    );
    let x2 = x().slice(ndarray::s![0..2, ..]).to_owned();
    assert_close(
        f.predict_proba(&x2).unwrap().as_slice().unwrap(),
        &[
            0.45549593220184653,
            0.09719350763046679,
            0.4473105601676866,
            0.7334699167287528,
            0.06305867747763873,
            0.20347140579360842,
        ],
        1e-8,
        "ovr proba",
    );
}

#[test]
fn test_binary_inference_matches_statsmodels_logit() {
    let f = LogisticRegressor::new()
        .with_no_penalty()
        .with_inference(true)
        .fit(&x(), &a(&YB))
        .unwrap();
    let s = f.summary().unwrap();
    assert_eq!(s.statistic_kind, StatisticKind::Z);
    let ic = s.intercept.as_ref().unwrap();
    assert_close(&[ic.estimate], &[0.36936002523453193], 1e-8, "intercept");
    assert_close(&[ic.std_error], &[0.3246648766275954], 1e-7, "intercept se");
    assert_close(
        s.coefficients.as_slice().unwrap(),
        &[1.4199769708381815, -1.1198883590930893, 0.3341983639696518],
        1e-8,
        "coef",
    );
    assert_close(
        s.std_errors.as_slice().unwrap(),
        &[0.5232159507744215, 0.3954594419866708, 0.3162624587889278],
        1e-7,
        "se",
    );
    assert_close(
        s.p_values.as_slice().unwrap(),
        &[
            0.006648810238956723,
            0.004627716057326486,
            0.2906430053607826,
        ],
        1e-7,
        "p",
    );
    assert_close(
        s.conf_lower.as_slice().unwrap(),
        &[0.3944925511834336, -1.894974622733271, -0.2856646649187297],
        1e-7,
        "ci lo",
    );
    assert_close(
        s.conf_upper.as_slice().unwrap(),
        &[2.4454613904929294, -0.3448020954529075, 0.9540613928580333],
        1e-7,
        "ci hi",
    );
    let (or, lo, hi) = s.odds_ratios();
    assert!((or[0] - 1.4199769708381815_f64.exp()).abs() < 1e-7);
    assert!(lo[0] < or[0] && or[0] < hi[0]);
    // Inference is binary-only.
    assert!(LogisticRegressor::new()
        .with_inference(true)
        .fit(&x(), &a(&Y3))
        .is_err());
}

#[test]
fn test_blocked_features_return_errors() {
    // Multinomial loss and L1 penalties need anofox-regression support (#72).
    assert!(LogisticRegressor::new()
        .with_multi_class(MultiClass::Multinomial)
        .fit(&x(), &a(&Y3))
        .is_err());
    assert!(LogisticRegressor::new()
        .with_l1_ratio(1.0)
        .fit(&x(), &a(&Y3))
        .is_err());
}
