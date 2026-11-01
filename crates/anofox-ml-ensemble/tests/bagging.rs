//! Bagging: per-estimator feature subsets, out-of-bag estimates and sample
//! weights (#50).

use anofox_ml_core::{Fit, FitWeighted, Predict};
use anofox_ml_ensemble::{BaggingClassifier, BaggingRegressor};
use anofox_ml_trees::MaxFeatures;
use ndarray::{Array1, Array2};

/// 120 rows, 6 features; only features 0 and 3 carry signal.
fn data() -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let x = Array2::from_shape_fn((120, 6), |(i, j)| ((i * (2 * j + 3) + j * 7) % 17) as f64);
    let yr = Array1::from_shape_fn(120, |i| {
        2.0 * x[[i, 0]] - x[[i, 3]] + ((i % 5) as f64) * 0.1
    });
    let yc = yr.mapv(|v| if v > 8.0 { 1.0 } else { 0.0 });
    (x, yr, yc)
}

#[test]
fn max_features_draws_one_subset_per_estimator() {
    let (x, yr, yc) = data();
    let f = BaggingRegressor::new(25)
        .with_max_features(Some(MaxFeatures::Fixed(3)), false)
        .with_seed(1)
        .fit(&x, &yr)
        .unwrap();
    let feats = f.estimators_features();
    assert_eq!(feats.len(), 25);
    for s in feats {
        assert_eq!(s.len(), 3);
        assert!(s.windows(2).all(|w| w[0] < w[1]), "{s:?}");
    }
    // Different estimators see different subsets.
    assert!(feats.iter().any(|s| s != &feats[0]));
    assert_eq!(f.predict(&x).unwrap().len(), 120);
    let imp = f.feature_importances();
    assert!((imp.sum() - 1.0).abs() < 1e-12);

    let c = BaggingClassifier::new(25)
        .with_max_features(Some(MaxFeatures::Fraction(0.5)), true)
        .with_seed(1)
        .fit(&x, &yc)
        .unwrap();
    for s in c.estimators_features() {
        assert_eq!(s.len(), 3);
    }
    // Drawn with replacement: some subset repeats a feature.
    assert!(c
        .estimators_features()
        .iter()
        .any(|s| s.windows(2).any(|w| w[0] == w[1])));
    let acc = c
        .predict(&x)
        .unwrap()
        .iter()
        .zip(&yc)
        .filter(|(p, t)| p == t)
        .count();
    assert!(acc >= 100, "{acc}/120");
}

#[test]
fn out_of_bag_estimates() {
    let (x, yr, yc) = data();
    let c = BaggingClassifier::new(40)
        .with_oob_score(true)
        .with_seed(3)
        .fit(&x, &yc)
        .unwrap();
    let dec = c.oob_decision_function().unwrap();
    assert_eq!(dec.dim(), (120, 2));
    for row in dec.rows() {
        // With 40 bootstrap draws every sample is out of bag somewhere.
        assert!((row.sum() - 1.0).abs() < 1e-12);
    }
    let score = c.oob_score().unwrap();
    assert!((0.75..=1.0).contains(&score), "{score}");
    // The OOB accuracy is the accuracy of the OOB decision function.
    let classes = c.classes();
    let manual = dec
        .rows()
        .into_iter()
        .zip(&yc)
        .filter(|(r, &t)| classes[if r[1] > r[0] { 1 } else { 0 }] == t)
        .count() as f64
        / 120.0;
    assert!((manual - score).abs() < 1e-12);

    let r = BaggingRegressor::new(40)
        .with_oob_score(true)
        .with_seed(3)
        .fit(&x, &yr)
        .unwrap();
    assert!(r.oob_prediction().unwrap().iter().all(|v| v.is_finite()));
    let r2 = r.oob_score().unwrap();
    assert!((0.7..1.0).contains(&r2), "{r2}");

    // Too few estimators: samples never out of bag get NaN.
    let few = BaggingRegressor::new(1)
        .with_oob_score(true)
        .fit(&x, &yr)
        .unwrap();
    assert!(few.oob_prediction().unwrap().iter().any(|v| v.is_nan()));
    assert!(few.oob_score().unwrap().is_finite());

    // Without OOB the accessors are empty; OOB needs bootstrap.
    assert!(BaggingClassifier::new(5)
        .fit(&x, &yc)
        .unwrap()
        .oob_score()
        .is_none());
    assert!(BaggingClassifier::new(5)
        .with_bootstrap(false)
        .with_oob_score(true)
        .fit(&x, &yc)
        .is_err());
}

#[test]
fn sample_weights() {
    let (x, yr, yc) = data();
    // Uniformly scaled weights change nothing.
    let w2 = Array1::from_elem(120, 2.0);
    let c = BaggingClassifier::new(10).with_seed(5);
    let a = c.fit(&x, &yc).unwrap().predict_proba(&x).unwrap();
    let b = c
        .fit_weighted(&x, &yc, Some(&w2))
        .unwrap()
        .predict_proba(&x)
        .unwrap();
    assert_eq!(a, b);
    let r = BaggingRegressor::new(10).with_seed(5);
    let a = r.fit(&x, &yr).unwrap().predict(&x).unwrap();
    let b = r
        .fit_weighted(&x, &yr, Some(&w2))
        .unwrap()
        .predict(&x)
        .unwrap();
    assert_eq!(a, b);

    // Zero-weight rows do not influence the leaves.
    let mut w = Array1::ones(120);
    let mut y_bad = yr.clone();
    for i in (0..120).step_by(4) {
        w[i] = 0.0;
        y_bad[i] = 1e6;
    }
    let p = BaggingRegressor::new(10)
        .with_seed(5)
        .fit_weighted(&x, &y_bad, Some(&w))
        .unwrap()
        .predict(&x)
        .unwrap();
    assert!(p.iter().all(|&v| v < 1e3), "{p:?}");
}

#[test]
fn invalid_max_features() {
    let (x, yr, _) = data();
    for mf in [
        MaxFeatures::Fixed(0),
        MaxFeatures::Fixed(7),
        MaxFeatures::Fraction(0.0),
    ] {
        assert!(BaggingRegressor::new(3)
            .with_max_features(Some(mf), false)
            .fit(&x, &yr)
            .is_err());
    }
}
