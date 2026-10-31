//! `FitWeighted` for the forest ensembles.

use anofox_ml_core::{ClassWeight, Fit, FitWeighted, Predict};
use anofox_ml_ensemble::{
    ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor,
};
use ndarray::{Array1, Array2};

fn data() -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let n = 60;
    let x = Array2::from_shape_fn((n, 3), |(i, j)| ((i * (j + 3) * 7919) % 101) as f64 / 10.0);
    let yr = x.column(0).mapv(|v| v * 2.0) - x.column(1);
    let yc = yr.mapv(|v| if v > 5.0 { 1.0 } else { 0.0 });
    (x, yr, yc)
}

#[test]
fn unit_weights_match_unweighted_fit() {
    let (x, yr, yc) = data();
    let ones = Array1::ones(x.nrows());

    let rf = RandomForestRegressor::new(8);
    let a = Fit::fit(&rf, &x, &yr).unwrap().predict(&x).unwrap();
    let b = rf
        .fit_weighted(&x, &yr, Some(&ones))
        .unwrap()
        .predict(&x)
        .unwrap();
    assert!((&a - &b).iter().all(|d| d.abs() < 1e-9));

    let et = ExtraTreesRegressor::new(8);
    let a = Fit::fit(&et, &x, &yr).unwrap().predict(&x).unwrap();
    let b = et
        .fit_weighted(&x, &yr, Some(&ones))
        .unwrap()
        .predict(&x)
        .unwrap();
    assert!((&a - &b).iter().all(|d| d.abs() < 1e-9));

    let rfc = RandomForestClassifier::new(8);
    let a = Fit::fit(&rfc, &x, &yc).unwrap().predict(&x).unwrap();
    let b = rfc
        .fit_weighted(&x, &yc, Some(&ones))
        .unwrap()
        .predict(&x)
        .unwrap();
    assert_eq!(a, b);

    let etc = ExtraTreesClassifier::new(8);
    let a = Fit::fit(&etc, &x, &yc).unwrap().predict(&x).unwrap();
    let b = etc
        .fit_weighted(&x, &yc, Some(&ones))
        .unwrap()
        .predict(&x)
        .unwrap();
    assert_eq!(a, b);
}

#[test]
fn weights_shift_the_fit() {
    let (x, _, yc) = data();
    // Weight class 1 to zero: every tree sees only class 0 mass.
    let w = yc.mapv(|v| if v == 1.0 { 0.0 } else { 1.0 });
    for pred in [
        RandomForestClassifier::new(5)
            .fit_weighted(&x, &yc, Some(&w))
            .unwrap()
            .predict(&x)
            .unwrap(),
        ExtraTreesClassifier::new(5)
            .fit_weighted(&x, &yc, Some(&w))
            .unwrap()
            .predict(&x)
            .unwrap(),
    ] {
        assert!(pred.iter().all(|&p| p == 0.0), "{pred}");
    }
    assert!(RandomForestClassifier::new(5)
        .fit_weighted(&x, &yc, Some(&Array1::ones(3)))
        .is_err());
}

#[test]
fn class_weight_is_computed_on_the_full_target() {
    let (x, _, yc) = data();
    let balanced = RandomForestClassifier::new(6).with_class_weight(Some(ClassWeight::Balanced));
    let manual = {
        let n = yc.len() as f64;
        let n1 = yc.sum();
        let w = yc.mapv(|v| {
            if v == 1.0 {
                n / (2.0 * n1)
            } else {
                n / (2.0 * (n - n1))
            }
        });
        RandomForestClassifier::new(6)
            .fit_weighted(&x, &yc, Some(&w))
            .unwrap()
            .predict(&x)
            .unwrap()
    };
    assert_eq!(
        Fit::fit(&balanced, &x, &yc).unwrap().predict(&x).unwrap(),
        manual
    );
}

fn gb_data() -> (
    Array2<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
) {
    let x = Array2::from_shape_fn((30, 3), |(i, j)| match j {
        0 => (i % 7) as f64,
        1 => ((i * 3) % 5) as f64,
        _ => ((i * i) % 11) as f64,
    });
    let yr = Array1::from_shape_fn(30, |i| x[[i, 0]] * 1.5 - x[[i, 2]] * 0.3 + (i as f64).sin());
    let yb = yr.mapv(|v| if v > 2.0 { 1.0 } else { 0.0 });
    let yc = Array1::from_shape_fn(30, |i| ((i * 7) % 3) as f64);
    let w = Array1::from_shape_fn(30, |i| 1.0 + (i % 4) as f64 * 0.5);
    (x, yr, yb, yc, w)
}

fn assert_close(a: &[f64], b: &[f64]) {
    for (u, v) in a.iter().zip(b) {
        assert!((u - v).abs() < 1e-9, "{a:?} vs {b:?}");
    }
}

#[test]
fn gradient_boosting_weights_match_sklearn() {
    use anofox_ml_ensemble::{GradientBoostingClassifier, GradientBoostingRegressor};
    let (x, yr, yb, yc, w) = gb_data();
    let q = x.slice(ndarray::s![..4, ..]).to_owned();

    // sklearn 1.8, n_estimators=5, learning_rate=0.3, max_depth=2.
    let r = GradientBoostingRegressor {
        n_estimators: 5,
        learning_rate: 0.3,
        max_depth: Some(2),
        ..Default::default()
    }
    .fit_weighted(&x, &yr, Some(&w))
    .unwrap();
    assert_close(
        r.predict(&q).unwrap().as_slice().unwrap(),
        &[
            0.7807204558074653,
            1.2774318007543766,
            2.560236412079404,
            2.4539576452723706,
        ],
    );

    let gb = GradientBoostingClassifier {
        n_estimators: 5,
        learning_rate: 0.3,
        max_depth: Some(2),
        ..Default::default()
    };
    let b = gb.fit_weighted(&x, &yb, Some(&w)).unwrap();
    assert_close(
        b.decision_function(&q)
            .unwrap()
            .column(0)
            .to_vec()
            .as_slice(),
        &[
            -0.9472047300893034,
            -0.5525599456262328,
            0.35134376967658876,
            -0.9496502863739084,
        ],
    );

    let c = GradientBoostingClassifier {
        n_estimators: 4,
        ..gb
    }
    .fit_weighted(&x, &yc, Some(&w))
    .unwrap();
    let d = c.decision_function(&q).unwrap();
    assert_close(
        d.row(0).to_vec().as_slice(),
        &[
            -0.36324376253211177,
            0.1099665039678434,
            -0.5717739929407619,
        ],
    );
    assert_close(
        d.row(2).to_vec().as_slice(),
        &[
            -0.04580799802437626,
            -0.27635526480661804,
            0.2880490549975169,
        ],
    );
}

#[test]
fn adaboost_weights_match_sklearn() {
    use anofox_ml_ensemble::AdaBoostClassifier;
    let (x, _, _, yc, w) = gb_data();
    // sklearn 1.8: AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
    // n_estimators=4, learning_rate=0.7).fit(X, y, sample_weight=w)
    let f = AdaBoostClassifier::new()
        .with_n_estimators(4)
        .with_learning_rate(0.7)
        .fit_weighted(&x, &yc, Some(&w))
        .unwrap();
    let q = x.slice(ndarray::s![..2, ..]).to_owned();
    let d = f.decision_function(&q).unwrap();
    assert_close(
        d.row(0).to_vec().as_slice(),
        &[-0.5, 0.27539872176749064, 0.22460127823250933],
    );
    assert_close(
        f.estimator_weights().to_vec().as_slice(),
        &[
            0.2239600542690014,
            0.18174037655707784,
            0.10148989474692253,
            0.1665249136750869,
        ],
    );
}
