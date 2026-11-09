//! GLM offsets through the registry (#125): `fit_with_offset` /
//! `predict_with_offset` with `offset = ln(exposure)`.

use anofox_ml_registry::{make, Out};
use ndarray::{Array1, Array2};
use serde_json::json;

fn out1(o: Out) -> Array1<f64> {
    match o {
        Out::A1(a) => a,
        Out::A2(a) => panic!("expected a 1-D output, got {:?}", a.dim()),
    }
}

/// Counts proportional to exposure at a constant rate of 0.3, with a
/// feature that carries nothing: with `offset = ln(exposure)` the fitted
/// rate is exactly 0.3 (the deviance is 0 there), whatever the penalty.
fn exposure_data() -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let n = 40;
    let exposure = Array1::from_shape_fn(n, |i| 1.0 + (i % 7) as f64 * 2.5);
    let x = Array2::from_shape_fn((n, 1), |(i, _)| ((i * 5) % 9) as f64 / 4.0 - 1.0);
    let y = &exposure * 0.3;
    (x, y, exposure)
}

#[test]
fn poisson_recovers_the_rate_from_the_exposure() {
    let (x, y, exposure) = exposure_data();
    let offset = exposure.mapv(f64::ln);
    let mut m = make("PoissonRegressor", &json!({})).unwrap();
    m.fit_with_offset(x.view(), Some(y.view()), None, offset.view())
        .unwrap();
    // Rate per unit of exposure, and expected counts at the exposure.
    for r in out1(m.predict(x.view()).unwrap()) {
        assert!((r - 0.3).abs() < 1e-6, "rate {r}");
    }
    let counts = out1(m.predict_with_offset(x.view(), offset.view()).unwrap());
    for (c, e) in counts.iter().zip(&exposure) {
        assert!((c - 0.3 * e).abs() < 1e-5, "{c} vs {}", 0.3 * e);
    }
}

#[test]
fn tweedie_and_gamma_recover_the_rate_weighted_or_not() {
    let (x, y, exposure) = exposure_data();
    let offset = exposure.mapv(f64::ln);
    let weights = Array1::from_shape_fn(y.len(), |i| 0.5 + (i % 3) as f64);
    for (kind, params) in [
        ("TweedieRegressor", json!({"power": 1.5, "link_power": 0.0})),
        ("GammaRegressor", json!({})),
    ] {
        for w in [None, Some(weights.view())] {
            let mut m = make(kind, &params).unwrap();
            m.fit_with_offset(x.view(), Some(y.view()), w, offset.view())
                .unwrap();
            for r in out1(m.predict(x.view()).unwrap()) {
                assert!(
                    (r - 0.3).abs() < 1e-6,
                    "{kind} (weighted: {}): rate {r}",
                    w.is_some()
                );
            }
            let counts = out1(m.predict_with_offset(x.view(), offset.view()).unwrap());
            for (c, e) in counts.iter().zip(&exposure) {
                assert!((c - 0.3 * e).abs() < 1e-5, "{kind}: {c} vs {}", 0.3 * e);
            }
        }
    }
}

#[test]
fn registry_matches_the_crate_on_noisy_counts() {
    let (x, _, exposure) = exposure_data();
    let offset = exposure.mapv(f64::ln);
    let y = Array1::from_shape_fn(x.nrows(), |i| ((i * 3) % 5) as f64 + (i % 2) as f64);
    let w = Array1::from_shape_fn(x.nrows(), |i| 1.0 + (i % 4) as f64 / 2.0);
    let mut m = make("PoissonRegressor", &json!({})).unwrap();
    m.fit_with_offset(x.view(), Some(y.view()), Some(w.view()), offset.view())
        .unwrap();
    let theirs = anofox_ml_regression::PoissonRegressor::default()
        .fit_weighted_with_offset(&x, &y, Some(&w), Some(&offset))
        .unwrap();
    let half = Array1::from_elem(x.nrows(), 0.5);
    assert_eq!(
        out1(m.predict_with_offset(x.view(), half.view()).unwrap()),
        theirs.predict_with_offset(&x, &half).unwrap()
    );
}

#[test]
fn only_the_glms_list_it_and_lengths_are_checked() {
    for kind in ["PoissonRegressor", "GammaRegressor", "TweedieRegressor"] {
        let caps = make(kind, &json!({})).unwrap().capabilities();
        assert!(
            caps.contains(&"fit_with_offset") && caps.contains(&"predict_with_offset"),
            "{kind}"
        );
    }
    let (x, y, exposure) = exposure_data();
    let mut ridge = make("RidgeRegressor", &json!({})).unwrap();
    let err = ridge
        .fit_with_offset(x.view(), Some(y.view()), None, exposure.view())
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("RidgeRegressor does not support fit_with_offset"),
        "{err}"
    );

    let mut m = make("PoissonRegressor", &json!({})).unwrap();
    let short = exposure.slice(ndarray::s![..5]);
    assert!(m
        .fit_with_offset(x.view(), Some(y.view()), None, short)
        .is_err());
    m.fit_with_offset(
        x.view(),
        Some(y.view()),
        None,
        exposure.mapv(f64::ln).view(),
    )
    .unwrap();
    assert!(m.predict_with_offset(x.view(), short).is_err());
}
