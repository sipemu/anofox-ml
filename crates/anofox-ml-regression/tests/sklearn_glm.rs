//! Parity of the GLM wrappers with sklearn 1.8.0 (`PoissonRegressor`,
//! `GammaRegressor`, `TweedieRegressor`, fitted with `tol=1e-12`) and with
//! statsmodels 0.14 `GLM` (prior weights = `var_weights`, offsets, Wald SEs).
//!
//! Cases that anofox-regression 0.7 cannot fit yet (penalised IRLS that
//! does not converge, sample weights for Poisson / Gamma / Tweedie) are kept
//! as reference data and ignored tests until upstream is fixed (#72).
//!
//! Reference script: rng = RandomState(0); X = round(normal(30, 3), 3);
//! y ~ Poisson / Gamma(shape 2) / zero-inflated Gamma around
//! exp(0.3 + 0.5 x0 - 0.4 x1 + 0.2 x2); w = round(uniform(0.2, 3), 3).

use anofox_ml_core::{Fit, FitWeighted, Predict};
use anofox_ml_regression::FittedTweedieRegressor;
use anofox_ml_regression::{
    BinomialLink, BinomialRegressor, GammaRegressor, PoissonRegressor, StatisticKind,
    TweedieRegressor,
};
use ndarray::{Array1, Array2};

const X: [f64; 90] = [
    1.764, 0.4, 0.979, 2.241, 1.868, -0.977, 0.95, -0.151, -0.103, 0.411, 0.144, 1.454, 0.761,
    0.122, 0.444, 0.334, 1.494, -0.205, 0.313, -0.854, -2.553, 0.654, 0.864, -0.742, 2.27, -1.454,
    0.046, -0.187, 1.533, 1.469, 0.155, 0.378, -0.888, -1.981, -0.348, 0.156, 1.23, 1.202, -0.387,
    -0.302, -1.049, -1.42, -1.706, 1.951, -0.51, -0.438, -1.253, 0.777, -1.614, -0.213, -0.895,
    0.387, -0.511, -1.181, -0.028, 0.428, 0.067, 0.302, -0.634, -0.363, -0.672, -0.36, -0.813,
    -1.726, 0.177, -0.402, -1.63, 0.463, -0.907, 0.052, 0.729, 0.129, 1.139, -1.235, 0.402, -0.685,
    -0.871, -0.579, -0.312, 0.056, -1.165, 0.901, 0.466, -1.536, 1.488, 1.896, 1.179, -0.18,
    -1.071, 1.054,
];
const YP: [f64; 30] = [
    9.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 1.0, 10.0, 0.0, 3.0, 0.0, 2.0, 3.0, 0.0, 2.0, 0.0, 0.0, 0.0,
    2.0, 2.0, 0.0, 0.0, 1.0, 7.0, 0.0, 2.0, 0.0, 1.0, 2.0,
];
const YG: [f64; 30] = [
    0.9815, 0.9377, 1.2455, 5.866, 0.5472, 1.2684, 0.4534, 0.2068, 11.9819, 0.9437, 0.3445, 0.8057,
    0.5928, 0.6151, 0.1488, 0.4675, 0.7159, 0.2666, 0.3461, 1.632, 0.2333, 0.4983, 0.5047, 0.7726,
    8.5931, 0.7462, 0.6156, 2.2804, 2.2434, 1.9117,
];
const YT: [f64; 30] = [
    0.9815, 0.9377, 1.2455, 0.0, 0.5472, 1.2684, 0.4534, 0.2068, 11.9819, 0.0, 0.3445, 0.8057, 0.0,
    0.6151, 0.0, 0.4675, 0.7159, 0.2666, 0.3461, 1.632, 0.0, 0.4983, 0.0, 0.7726, 0.0, 0.0, 0.6156,
    2.2804, 2.2434, 1.9117,
];
const YB: [f64; 30] = [
    1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
    1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0,
];
const W: [f64; 30] = [
    2.367, 1.133, 0.427, 1.34, 0.85, 0.571, 0.35, 2.232, 0.232, 2.358, 0.611, 0.423, 0.451, 2.082,
    0.887, 1.378, 1.761, 2.61, 2.236, 0.957, 0.568, 0.355, 1.044, 0.934, 1.477, 2.113, 2.148,
    0.994, 1.264, 0.707,
];
const OFF: [f64; 30] = [
    -0.221, 0.029, 0.272, 0.306, -0.034, 0.523, -0.097, 0.249, -0.062, 0.335, 0.319, 0.345, -0.232,
    -0.388, 0.203, 0.127, -0.146, -0.016, 0.17, 0.32, 0.081, 0.186, -0.018, 0.377, 0.089, 0.12,
    -0.316, -0.192, -0.045, -0.472,
];

fn x() -> Array2<f64> {
    Array2::from_shape_vec((30, 3), X.to_vec()).unwrap()
}
fn a(v: &[f64]) -> Array1<f64> {
    Array1::from_vec(v.to_vec())
}

/// `expected` = [intercept, coef...].
fn check(intercept: Option<f64>, coef: &Array1<f64>, expected: &[f64], tol: f64, ctx: &str) {
    let b = intercept.unwrap();
    assert!(
        (b - expected[0]).abs() < tol,
        "{ctx}: intercept {b} vs {}",
        expected[0]
    );
    for j in 0..coef.len() {
        assert!(
            (coef[j] - expected[j + 1]).abs() < tol,
            "{ctx}: coef[{j}] {} vs {}",
            coef[j],
            expected[j + 1]
        );
    }
}

// sklearn reference coefficients ([intercept, coef...]).
const SK: &[(&str, f64, bool, [f64; 4])] = &[
    (
        "pois",
        0.0,
        false,
        [
            0.2413425730713719,
            0.7282578351583869,
            -0.45669841562405855,
            0.1875066820397625,
        ],
    ),
    (
        "pois",
        0.0,
        true,
        [
            0.11066833871334188,
            0.9374456386369864,
            -0.5653452332822471,
            0.2510948450834683,
        ],
    ),
    (
        "pois",
        0.1,
        false,
        [
            0.276383380094477,
            0.6976607049144924,
            -0.44016839976497235,
            0.18031308528254616,
        ],
    ),
    (
        "pois",
        0.1,
        true,
        [
            0.17499630267225175,
            0.8725216954563558,
            -0.5251455328384002,
            0.24771999434881803,
        ],
    ),
    (
        "pois",
        1.0,
        false,
        [
            0.4623104337233664,
            0.507175345288268,
            -0.323219390744292,
            0.13656449887699115,
        ],
    ),
    (
        "pois",
        1.0,
        true,
        [
            0.43654475718293984,
            0.545348434323213,
            -0.3069755588005669,
            0.19892342285760833,
        ],
    ),
    (
        "gamma",
        0.0,
        false,
        [
            0.18289444147937425,
            0.4786744749205537,
            -0.4187311145653131,
            0.4386196425863946,
        ],
    ),
    (
        "gamma",
        0.0,
        true,
        [
            0.09527715967221176,
            0.4133025936516089,
            -0.41020571716063264,
            0.45371939198567274,
        ],
    ),
    (
        "gamma",
        0.1,
        false,
        [
            0.1727665507561198,
            0.454902820171914,
            -0.3799063213349591,
            0.39895126909643464,
        ],
    ),
    (
        "gamma",
        0.1,
        true,
        [
            0.0862270151565124,
            0.3844980268384014,
            -0.36196132110705576,
            0.4146634060842426,
        ],
    ),
    (
        "gamma",
        1.0,
        false,
        [
            0.1945332415812371,
            0.31174668506287445,
            -0.2196318701733411,
            0.22535551938896645,
        ],
    ),
    (
        "gamma",
        1.0,
        true,
        [
            0.10819991108851784,
            0.23639037188021017,
            -0.180897811192152,
            0.24041119243972917,
        ],
    ),
    (
        "tw15",
        0.0,
        false,
        [
            -0.36272183101225774,
            0.6647480198509604,
            -0.43169238863596426,
            0.08644883965244357,
        ],
    ),
    (
        "tw15",
        0.0,
        true,
        [
            -0.6494287448263492,
            0.5670087792798687,
            -0.2041238663342076,
            -0.07361172320809162,
        ],
    ),
    (
        "tw15",
        0.1,
        false,
        [
            -0.337458590661243,
            0.6185773205468904,
            -0.3930337512780813,
            0.08286997920215268,
        ],
    ),
    (
        "tw15",
        0.1,
        true,
        [
            -0.6222613247608524,
            0.4943184933487411,
            -0.16242934019366548,
            -0.05232758599992375,
        ],
    ),
    (
        "tw15",
        1.0,
        false,
        [
            -0.20501737795905875,
            0.3858392200738447,
            -0.22417567814266284,
            0.05761072516276587,
        ],
    ),
    (
        "tw15",
        1.0,
        true,
        [
            -0.5254572252716686,
            0.2408923371057144,
            -0.051342971485569294,
            -0.0021579864798448303,
        ],
    ),
];

/// Unweighted fit of one `SK` case through the wrappers.
fn fit_case(name: &str, alpha: f64) -> FittedTweedieRegressor {
    let x = x();
    match name {
        "pois" => TweedieRegressor::new(1.0)
            .with_alpha(alpha)
            .with_tol(1e-12)
            .with_max_iter(1000)
            .fit(&x, &a(&YP))
            .unwrap(),
        "gamma" => GammaRegressor::new()
            .with_alpha(alpha)
            .with_tol(1e-12)
            .with_max_iter(1000)
            .fit(&x, &a(&YG))
            .unwrap(),
        _ => TweedieRegressor::new(1.5)
            .with_alpha(alpha)
            .with_tol(1e-12)
            .with_max_iter(1000)
            .fit(&x, &a(&YT))
            .unwrap(),
    }
}

/// Unweighted cases upstream fits today.
const CONVERGING: &[(&str, f64)] = &[
    ("pois", 0.0),
    ("pois", 0.1),
    ("gamma", 0.0),
    ("gamma", 0.1),
    ("tw15", 0.0),
];

#[test]
fn test_glm_alpha_matches_sklearn() {
    for &(name, alpha, weighted, expected) in SK {
        if weighted || !CONVERGING.contains(&(name, alpha)) {
            continue;
        }
        let f = fit_case(name, alpha);
        assert!(f.converged());
        check(
            f.intercept(),
            &f.coefficients(),
            &expected,
            1e-6,
            &format!("{name} alpha={alpha}"),
        );
    }
    // PoissonRegressor is the same model as Tweedie(power = 1).
    let f = PoissonRegressor::new()
        .with_alpha(0.1)
        .with_tol(1e-12)
        .fit(&x(), &a(&YP))
        .unwrap();
    check(
        f.intercept(),
        &f.coefficients(),
        &SK[2].3,
        1e-6,
        "poisson 0.1",
    );
}

#[test]
#[ignore = "upstream anofox-regression 0.7 penalised IRLS does not converge (#72)"]
fn test_glm_large_alpha_matches_sklearn() {
    for &(name, alpha, weighted, expected) in SK {
        if weighted || CONVERGING.contains(&(name, alpha)) {
            continue;
        }
        let f = fit_case(name, alpha);
        check(
            f.intercept(),
            &f.coefficients(),
            &expected,
            1e-6,
            &format!("{name} alpha={alpha}"),
        );
    }
    // The weighted rows of `SK` additionally need upstream sample weights
    // for Poisson / Gamma / Tweedie (`FitWeighted`, #72).
}

#[test]
#[ignore = "sklearn's default alpha = 1.0 is blocked on upstream IRLS convergence (#72)"]
fn test_glm_default_alpha_is_sklearn_default() {
    // Once upstream converges at alpha = 1, switch the wrappers' default
    // alpha to 1.0 (sklearn) and re-enable this test.
    let x = x();
    let f = PoissonRegressor::new().fit(&x, &a(&YP)).unwrap();
    check(
        f.intercept(),
        &f.coefficients(),
        &SK[4].3,
        1e-6,
        "poisson default",
    );
    let f = GammaRegressor::new().fit(&x, &a(&YG)).unwrap();
    check(
        f.intercept(),
        &f.coefficients(),
        &SK[10].3,
        1e-6,
        "gamma default",
    );
    let f = TweedieRegressor::new(1.5).fit(&x, &a(&YT)).unwrap();
    check(
        f.intercept(),
        &f.coefficients(),
        &SK[16].3,
        1e-6,
        "tweedie default",
    );
}

#[test]
fn test_poisson_inference_matches_statsmodels() {
    // statsmodels 0.14.5 GLM(Poisson).fit()
    let f = PoissonRegressor::new()
        .with_inference(true)
        .with_tol(1e-12)
        .fit(&x(), &a(&YP))
        .unwrap();
    let s = f.summary().unwrap();
    assert_eq!(s.statistic_kind, StatisticKind::Z);
    let bse = [
        0.18474362129671337,
        0.1298015859458723,
        0.12541246273227527,
        0.15427921480358778,
    ];
    let p: [f64; 4] = [
        0.19142851290122243,
        2.0168865974231872e-08,
        0.00027097914991795795,
        0.22422413628760496,
    ];
    let ic = s.intercept.as_ref().unwrap();
    assert!(
        (ic.std_error - bse[0]).abs() < 1e-7,
        "se[b] {}",
        ic.std_error
    );
    assert!((ic.p_value - p[0]).abs() < 1e-7, "p[b] {}", ic.p_value);
    for j in 0..3 {
        assert!(
            (s.std_errors[j] - bse[j + 1]).abs() < 1e-7,
            "se[{j}] {}",
            s.std_errors[j]
        );
        assert!(
            (s.p_values[j] - p[j + 1]).abs() < 1e-7 * p[j + 1].max(1e-3),
            "p[{j}] {}",
            s.p_values[j]
        );
    }
}

#[test]
fn test_gamma_inference_matches_statsmodels_pearson_scale() {
    // statsmodels GLM(Gamma(log)).fit(): scale = Pearson chi2 / df; SEs
    // identical to R's summary.glm; p-values from t(n - p) as in R.
    let f = GammaRegressor::new()
        .with_inference(true)
        .with_tol(1e-12)
        .with_max_iter(1000)
        .fit(&x(), &a(&YG))
        .unwrap();
    let s = f.summary().unwrap();
    assert_eq!(s.statistic_kind, StatisticKind::T);
    assert!((s.dispersion - 0.5324834115572503).abs() < 1e-7);
    assert!((s.residual_df - 26.0).abs() < 1e-12);
    let bse = [
        0.14118272004734012,
        0.1207893466379367,
        0.1380066345748662,
        0.14512561383962763,
    ];
    let p = [
        0.20655106227197093,
        0.0005149777844881463,
        0.0054164831450305975,
        0.0055743046467707,
    ];
    let ic = s.intercept.as_ref().unwrap();
    assert!((ic.std_error - bse[0]).abs() < 1e-7);
    assert!((ic.p_value - p[0]).abs() < 1e-6);
    for j in 0..3 {
        assert!((s.std_errors[j] - bse[j + 1]).abs() < 1e-7, "se[{j}]");
        assert!((s.p_values[j] - p[j + 1]).abs() < 1e-6, "p[{j}]");
    }
}

#[test]
fn test_binomial_links_with_weights_match_statsmodels() {
    let cases: [(BinomialLink, [f64; 4], [f64; 4]); 3] = [
        (
            BinomialLink::Logit,
            [
                -0.18954121782262143,
                1.8442743093776794,
                -1.780163536428821,
                -0.6717972571158372,
            ],
            [
                0.4772970728815795,
                0.6964918041821602,
                0.6884965178367226,
                0.5080957827373649,
            ],
        ),
        (
            BinomialLink::Probit,
            [
                -0.1252708754150558,
                1.101794055467222,
                -1.0020237736601836,
                -0.36465103559741124,
            ],
            [
                0.2776389350020937,
                0.3779004422370377,
                0.3734689578940487,
                0.28705312911855774,
            ],
        ),
        (
            BinomialLink::Cloglog,
            [
                -0.6043032518835446,
                1.3080392248967976,
                -1.0967354688574629,
                -0.29333990502446317,
            ],
            [
                0.34156045408896935,
                0.43919952498723147,
                0.4229232167771257,
                0.2881760831462496,
            ],
        ),
    ];
    for (link, params, bse) in cases {
        let f = BinomialRegressor::new()
            .with_link(link)
            .with_tol(1e-12)
            .with_inference(true)
            .fit_weighted(&x(), &a(&YB), Some(&a(&W)))
            .unwrap();
        check(
            f.intercept(),
            &f.coefficients(),
            &params,
            1e-6,
            &format!("{link:?}"),
        );
        let s = f.summary().unwrap();
        assert!((s.intercept.as_ref().unwrap().std_error - bse[0]).abs() < 1e-6);
        for j in 0..3 {
            assert!(
                (s.std_errors[j] - bse[j + 1]).abs() < 1e-6,
                "{link:?} se[{j}]"
            );
        }
        let p = f.predict(&x()).unwrap();
        assert!(p.iter().all(|v| *v > 0.0 && *v < 1.0));
    }
}

#[test]
fn test_poisson_offset_matches_statsmodels() {
    let f = PoissonRegressor::new()
        .with_alpha(0.0)
        .with_tol(1e-12)
        .fit_with_offset(&x(), &a(&YP), Some(&a(&OFF)))
        .unwrap();
    check(
        f.intercept(),
        &f.coefficients(),
        &[
            0.14967662558431055,
            0.7846089500458493,
            -0.4872300568397825,
            0.1658645267348445,
        ],
        1e-7,
        "poisson offset",
    );
    // predict_with_offset = exp(eta + offset) = predict * exp(offset).
    let p0 = f.predict(&x()).unwrap();
    let p1 = f.predict_with_offset(&x(), &a(&OFF)).unwrap();
    for i in 0..30 {
        assert!((p1[i] - p0[i] * OFF[i].exp()).abs() < 1e-10);
    }
}
