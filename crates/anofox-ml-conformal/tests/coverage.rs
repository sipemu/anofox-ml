//! Empirical coverage over many seeded replications, and behaviour checks.

use anofox_ml_conformal::{
    conformalize, CrossConformalRegressor, CrossMethod, PointPrediction, SplitConformalRegressor,
};
use anofox_ml_core::utils::leave_one_out;
use anofox_ml_core::{Fit, Predict};
use anofox_ml_regression::OlsRegressor;
use anofox_ml_trees::DecisionTreeRegressor;
use ndarray::{Array1, Array2};
use rand::{Rng, SeedableRng};

/// Heteroscedastic linear data: noise sd grows with |x0|.
fn data(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut normal = || {
        let (u1, u2): (f64, f64) = (rng.gen_range(1e-12..1.0), rng.gen());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };
    let x = Array2::from_shape_fn((n, 2), |_| normal());
    let y = Array1::from_shape_fn(n, |i| {
        2.0 * x[[i, 0]] - x[[i, 1]] + (0.3 + x[[i, 0]].abs()) * normal()
    });
    (x, y)
}

fn coverage(y: &Array1<f64>, lo: &Array1<f64>, hi: &Array1<f64>) -> f64 {
    y.iter()
        .zip(lo.iter().zip(hi))
        .filter(|(v, (l, h))| l <= v && v <= h)
        .count() as f64
        / y.len() as f64
}

/// Mean coverage over `reps` independent train / test draws.
fn mean_coverage(
    reps: u64,
    alpha: f64,
    fit_interval: impl Fn(&Array2<f64>, &Array1<f64>, &Array2<f64>) -> (Array1<f64>, Array1<f64>),
) -> f64 {
    (0..reps)
        .map(|r| {
            let (x, y) = data(100, 2 * r);
            let (xt, yt) = data(500, 2 * r + 1);
            let (lo, hi) = fit_interval(&x, &y, &xt);
            coverage(&yt, &lo, &hi)
        })
        .sum::<f64>()
        / reps as f64
        - (1.0 - alpha)
}

#[test]
fn split_conformal_covers_at_the_nominal_level() {
    for alpha in [0.1, 0.3] {
        let excess = mean_coverage(200, alpha, |x, y, xt| {
            SplitConformalRegressor::new(OlsRegressor::new())
                .with_calibration_size(0.4)
                .fit(x, y)
                .unwrap()
                .predict_interval(xt.view(), alpha)
                .unwrap()
        });
        // E[coverage] ∈ [1 − α, 1 − α + 1/(n_cal + 1)], n_cal = 40
        assert!((-0.01..0.035).contains(&excess), "alpha {alpha}: {excess}");
    }
}

#[test]
fn cross_conformal_methods_cover() {
    let alpha = 0.1;
    for (method, min_excess) in [
        (CrossMethod::Plus, -0.02),
        (CrossMethod::MinMax, -0.01),
        (CrossMethod::Base, -0.03),
    ] {
        let excess = mean_coverage(100, alpha, |x, y, xt| {
            CrossConformalRegressor::new(OlsRegressor::new())
                .with_cv(5)
                .with_method(method)
                .fit(x, y)
                .unwrap()
                .predict_interval(xt.view(), alpha)
                .unwrap()
        });
        assert!(excess > min_excess && excess < 0.05, "{method:?}: {excess}");
    }
}

#[test]
fn interval_shapes_and_nesting() {
    let (x, y) = data(120, 7);
    let (xt, _) = data(30, 8);
    let tree = DecisionTreeRegressor {
        max_depth: Some(3),
        ..Default::default()
    };
    let plus = CrossConformalRegressor::new(tree.clone())
        .fit(&x, &y)
        .unwrap();
    let minmax = CrossConformalRegressor::new(tree)
        .with_method(CrossMethod::MinMax)
        .fit(&x, &y)
        .unwrap();
    let (l90, h90) = plus.predict_interval(xt.view(), 0.1).unwrap();
    let (l50, h50) = plus.predict_interval(xt.view(), 0.5).unwrap();
    let (ml, mh) = minmax.predict_interval(xt.view(), 0.1).unwrap();
    for i in 0..30 {
        // Lower alpha -> wider; minmax contains CV+ (it uses the extremes).
        assert!(l90[i] <= l50[i] && h50[i] <= h90[i]);
        assert!(ml[i] <= l90[i] + 1e-12 && h90[i] <= mh[i] + 1e-12);
    }
    assert_eq!(plus.fold_models().len(), 5);
    assert_eq!(plus.conformity_scores().len(), 120);
}

#[test]
fn median_point_prediction_is_the_median_over_rows() {
    let (x, y) = data(9, 3);
    // 3 folds of 2, 3 and 4 rows.
    let fold = [0, 0, 1, 1, 1, 2, 2, 2, 2];
    let splits: Vec<_> = (0..3)
        .map(|f| {
            let test = (0..9).filter(|&i| fold[i] == f).collect();
            let train = (0..9).filter(|&i| fold[i] != f).collect();
            (train, test)
        })
        .collect();
    let fit = |p| {
        CrossConformalRegressor::new(OlsRegressor::new())
            .with_point_prediction(p)
            .fit_with_splits(x.view(), y.view(), &splits)
            .unwrap()
    };
    let (xt, _) = data(4, 4);
    let median = fit(PointPrediction::Median).predict(&xt).unwrap();
    let mean = fit(PointPrediction::Mean).predict(&xt).unwrap();
    let m = fit(PointPrediction::Full);
    let per_fold: Vec<Array1<f64>> = m
        .fold_models()
        .iter()
        .map(|f| f.predict(&xt).unwrap())
        .collect();
    for r in 0..4 {
        // 9 per-row values: 2× fold 0, 3× fold 1, 4× fold 2 -> 5th smallest
        let mut v: Vec<f64> = fold.iter().map(|&k| per_fold[k][r]).collect();
        v.sort_by(f64::total_cmp);
        assert_eq!(median[r], v[4]);
        let w = (2.0 * per_fold[0][r] + 3.0 * per_fold[1][r] + 4.0 * per_fold[2][r]) / 9.0;
        assert!((mean[r] - w).abs() < 1e-12);
    }
    assert_eq!(
        m.predict(&xt).unwrap(),
        m.full_model().predict(&xt).unwrap()
    );
}

#[test]
fn too_few_rows_error_or_infinite_bounds() {
    let (x, y) = data(30, 5);
    let model = OlsRegressor::new().fit(&x, &y).unwrap();
    let (xc, yc) = data(8, 6);
    let fitted = conformalize(model, xc.view(), yc.view()).unwrap();
    let e = fitted
        .predict_interval(x.view(), 0.1)
        .unwrap_err()
        .to_string();
    assert!(e.contains("at least 9"), "{e}");
    let (lo, hi) = fitted
        .clone()
        .with_allow_infinite_bounds(true)
        .predict_interval(x.view(), 0.1)
        .unwrap();
    assert!(lo.iter().all(|v| *v == f64::NEG_INFINITY) && hi.iter().all(|v| *v == f64::INFINITY));
    assert!(fitted
        .predict_interval(x.view(), 0.2)
        .unwrap()
        .0
        .iter()
        .all(|v| v.is_finite()));

    // Jackknife+ on 8 rows at alpha 0.1: unbounded unless allowed.
    let (x8, y8) = (xc.view(), yc.view());
    let jk = CrossConformalRegressor::new(OlsRegressor::new());
    let e = jk
        .fit_with_splits(x8, y8, &leave_one_out(8))
        .unwrap()
        .predict_interval(x.view(), 0.1)
        .unwrap_err();
    assert!(e.to_string().contains("too few"));
    let (lo, hi) = jk
        .with_allow_infinite_bounds(true)
        .fit_with_splits(x8, y8, &leave_one_out(8))
        .unwrap()
        .predict_interval(x.view(), 0.1)
        .unwrap();
    assert!(lo.iter().all(|v| v.is_infinite()) && hi.iter().all(|v| v.is_infinite()));
}

#[test]
fn invalid_inputs_are_rejected() {
    let (x, y) = data(20, 9);
    let cc = CrossConformalRegressor::new(OlsRegressor::new());
    let overlapping = vec![
        ((10..20).collect(), (0..10).collect()),
        ((0..10).collect(), (5..20).collect()),
    ];
    assert!(cc
        .fit_with_splits(x.view(), y.view(), &overlapping)
        .is_err());
    let missing = vec![
        ((10..20).collect(), (0..10).collect()),
        ((0..10).collect(), (10..19).collect()),
    ];
    assert!(cc.fit_with_splits(x.view(), y.view(), &missing).is_err());
    let fitted = cc.fit(&x, &y).unwrap();
    assert!(fitted.predict_interval(x.view(), 1.0).is_err());
    assert!(SplitConformalRegressor::new(OlsRegressor::new())
        .with_calibration_size(1.0)
        .fit(&x, &y)
        .is_err());
}

/// Mean coverage and the correlation between interval width and the noise
/// scale |x0| (adaptive methods should track it).
fn coverage_and_adaptivity(
    reps: u64,
    fit_interval: impl Fn(&Array2<f64>, &Array1<f64>, &Array2<f64>) -> (Array1<f64>, Array1<f64>),
) -> (f64, f64) {
    let mut cov = 0.0;
    let mut corr = 0.0;
    for r in 0..reps {
        let (x, y) = data(300, 100 + 2 * r);
        let (xt, yt) = data(400, 101 + 2 * r);
        let (lo, hi) = fit_interval(&x, &y, &xt);
        cov += coverage(&yt, &lo, &hi);
        let w = &hi - &lo;
        let s = xt.column(0).mapv(f64::abs);
        let (mw, ms) = (w.mean().unwrap(), s.mean().unwrap());
        let cv: f64 = w.iter().zip(&s).map(|(a, b)| (a - mw) * (b - ms)).sum();
        let (vw, vs): (f64, f64) = (
            w.iter().map(|a| (a - mw).powi(2)).sum(),
            s.iter().map(|b| (b - ms).powi(2)).sum(),
        );
        corr += cv / (vw * vs).sqrt();
    }
    (cov / reps as f64, corr / reps as f64)
}

#[test]
fn cqr_covers_and_adapts_to_the_noise() {
    use anofox_ml_conformal::ConformalizedQuantileRegressor;
    use anofox_ml_ensemble::{GbLoss, GradientBoostingRegressor};
    let alpha = 0.2;
    for symmetric in [false, true] {
        let (cov, corr) = coverage_and_adaptivity(20, |x, y, xt| {
            ConformalizedQuantileRegressor::new(alpha, |q| {
                GradientBoostingRegressor::new()
                    .with_loss(GbLoss::Quantile)
                    .with_alpha(q)
                    .with_n_estimators(60)
                    .with_max_depth(Some(3))
            })
            .with_calibration_size(0.3)
            .with_symmetric_correction(symmetric)
            .fit(x, y)
            .unwrap()
            .predict_interval(xt.view())
            .unwrap()
        });
        // 20 reps x 400 test rows: the mean is close to 1 - alpha.
        assert!(
            (cov - (1.0 - alpha)).abs() < 0.04,
            "symmetric={symmetric}: {cov}"
        );
        assert!(
            corr > 0.3,
            "symmetric={symmetric}: width/noise correlation {corr}"
        );
    }
}

#[test]
fn normalized_scores_cover_and_adapt_to_the_noise() {
    use anofox_ml_conformal::NormalizedConformalRegressor;
    let alpha = 0.1;
    let tree = DecisionTreeRegressor {
        max_depth: Some(3),
        ..Default::default()
    };
    let (cov, corr) = coverage_and_adaptivity(30, |x, y, xt| {
        NormalizedConformalRegressor::new(OlsRegressor::new(), tree.clone())
            .fit(x, y)
            .unwrap()
            .predict_interval(xt.view(), alpha)
            .unwrap()
    });
    assert!((cov - (1.0 - alpha)).abs() < 0.03, "{cov}");
    assert!(corr > 0.3, "width/noise correlation {corr}");
    // Plain split conformal has constant width (up to rounding): no correlation.
    let (_, flat) = coverage_and_adaptivity(5, |x, y, xt| {
        SplitConformalRegressor::new(OlsRegressor::new())
            .fit(x, y)
            .unwrap()
            .predict_interval(xt.view(), alpha)
            .unwrap()
    });
    assert!(flat.is_nan() || flat.abs() < 0.05, "{flat}");
}

#[test]
fn cqr_and_normalized_reject_bad_input() {
    use anofox_ml_conformal::{
        conformalize_quantiles, ConformalizedQuantileRegressor, NormalizedConformalRegressor,
    };
    let (x, y) = data(40, 11);
    let make = |_q: f64| OlsRegressor::new();
    assert!(ConformalizedQuantileRegressor::new(0.0, make)
        .fit(&x, &y)
        .is_err());
    let m = || OlsRegressor::new().fit(&x, &y).unwrap();
    let (xc, yc) = data(5, 12);
    let fitted = conformalize_quantiles(m(), m(), m(), xc.view(), yc.view(), 0.1).unwrap();
    assert!(fitted
        .predict_interval(x.view())
        .unwrap_err()
        .to_string()
        .contains("too few"));
    assert!(fitted
        .with_allow_infinite_bounds(true)
        .predict_interval(x.view())
        .unwrap()
        .0[0]
        .is_infinite());
    assert!(
        NormalizedConformalRegressor::new(OlsRegressor::new(), OlsRegressor::new())
            .with_residual_size(1.0)
            .fit(&x, &y)
            .is_err()
    );
}
