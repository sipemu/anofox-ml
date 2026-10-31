//! `FitWeighted` and `class_weight` for the SGD estimators (#37), checked
//! against sklearn 1.8 with `shuffle=False` (deterministic sample order).

use anofox_ml_core::{ClassWeight, FitWeighted};
use anofox_ml_linear::{LearningRate, SgdClassifier, SgdRegressor};
use ndarray::{Array1, Array2};

fn data() -> (
    Array2<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
) {
    let x = Array2::from_shape_fn((30, 3), |(i, j)| match j {
        0 => (i % 7) as f64 / 3.0,
        1 => ((i * 3) % 5) as f64 / 2.0,
        _ => ((i * i) % 11) as f64 / 5.0,
    });
    let yr = Array1::from_shape_fn(30, |i| x[[i, 0]] * 1.5 - x[[i, 2]] * 0.3 + (i as f64).sin());
    let yb = yr.mapv(|v| if v > 1.0 { 1.0 } else { 0.0 });
    let yc = Array1::from_shape_fn(30, |i| ((i * 7) % 3) as f64);
    let w = Array1::from_shape_fn(30, |i| 1.0 + (i % 4) as f64 * 0.5);
    (x, yr, yb, yc, w)
}

fn assert_close(a: &[f64], b: &[f64], tol: f64) {
    for (u, v) in a.iter().zip(b) {
        assert!((u - v).abs() < tol * v.abs().max(1.0), "{a:?} vs {b:?}");
    }
}

#[test]
fn sgd_regressor_sample_weight_matches_sklearn() {
    let (x, yr, _, _, w) = data();
    // SGDRegressor(shuffle=False, learning_rate='constant', eta0=0.01,
    //              max_iter=5, tol=None).fit(X, y, sample_weight=w)
    let f = SgdRegressor::new()
        .with_shuffle(false)
        .with_learning_rate(LearningRate::Constant)
        .with_eta0(0.01)
        .with_max_iter(5)
        .with_tol(None)
        .fit_weighted(&x, &yr, Some(&w))
        .unwrap();
    assert_close(
        f.weights().as_slice().unwrap(),
        &[
            1.0027785924227375,
            -0.01905109901329196,
            0.17359401570801608,
        ],
        1e-10,
    );
    assert_close(&[f.bias()], &[0.2680374931882771], 1e-10);
}

#[test]
fn sgd_classifier_sample_and_class_weight_match_sklearn() {
    let (x, _, yb, yc, w) = data();
    let base = SgdClassifier::new()
        .with_shuffle(false)
        .with_max_iter(5)
        .with_tol(None);

    // SGDClassifier(shuffle=False, max_iter=5, tol=None,
    //               class_weight={0: 2}).fit(X, yb, sample_weight=w)
    let b = base
        .clone()
        .with_class_weight(Some(ClassWeight::Manual(vec![(0.0, 2.0)])))
        .fit_weighted(&x, &yb, Some(&w))
        .unwrap();
    assert_close(
        b.weights()[0].as_slice().unwrap(),
        &[100.0870322019147, -10.879025239338484, 21.7580504786771],
        1e-9,
    );
    assert_close(b.biases(), &[-161.26005916467582], 1e-9);

    // ... class_weight='balanced' on three classes (one-vs-rest).
    let c = base
        .with_class_weight(Some(ClassWeight::Balanced))
        .fit_weighted(&x, &yc, Some(&w))
        .unwrap();
    let expected = [
        [17.4064403829417, -17.40644038294168, 9.573542210617932],
        [-21.758050478677106, -41.340295909486485, 9.573542210617912],
        [-36.263417464461824, 30.461270670147925, -6.092254134029586],
    ];
    for (k, e) in expected.iter().enumerate() {
        assert_close(c.weights()[k].as_slice().unwrap(), e, 1e-9);
    }
    assert_close(
        c.biases(),
        &[-16.149816701589224, -14.654571118403434, 9.267335494485184],
        1e-9,
    );
}

#[test]
fn default_learning_rate_schedules_match_sklearn() {
    use anofox_ml_core::Fit;
    let (x, yr, yb, _, _) = data();
    // SGDClassifier(shuffle=False, max_iter=5, tol=None): 'optimal' schedule.
    let c = Fit::fit(
        &SgdClassifier::new()
            .with_shuffle(false)
            .with_max_iter(5)
            .with_tol(None),
        &x,
        &yb,
    )
    .unwrap();
    assert_close(
        c.weights()[0].as_slice().unwrap(),
        &[40.615027560197255, -4.351610095735404, 1.7406440382941717],
        1e-9,
    );
    assert_close(c.biases(), &[-37.65225127784046], 1e-9);
    // SGDRegressor(shuffle=False, max_iter=5, tol=None): 'invscaling'.
    let r = Fit::fit(
        &SgdRegressor::new()
            .with_shuffle(false)
            .with_max_iter(5)
            .with_tol(None),
        &x,
        &yr,
    )
    .unwrap();
    assert_close(
        r.weights().as_slice().unwrap(),
        &[0.4954621275845946, 0.20882248514618207, 0.21056303540688862],
        1e-10,
    );
    assert_close(&[r.bias()], &[0.2743955052706654], 1e-10);
}
