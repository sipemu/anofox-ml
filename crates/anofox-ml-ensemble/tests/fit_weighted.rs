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

#[test]
fn gradient_boosting_losses_match_sklearn() {
    use anofox_ml_ensemble::{GbLoss, GradientBoostingRegressor};
    // Continuous features (no split ties, which sklearn breaks randomly).
    let x = Array2::from_shape_fn((40, 3), |(i, j)| {
        (((i + 1) * (j + 1)) as f64 * 0.6180339887498949) % 1.0 * 10.0
    });
    let y = Array1::from_shape_fn(40, |i| x[[i, 0]] * 1.5 - x[[i, 2]] * 0.3 + (i as f64).sin());
    let w = Array1::from_shape_fn(40, |i| 1.0 + (i % 4) as f64 * 0.5);
    let q = x.slice(ndarray::s![..4, ..]).to_owned();
    // sklearn 1.8: GradientBoostingRegressor(loss=..., learning_rate=0.3,
    // max_depth=2, alpha=0.8). Absolute and pinball gradients take two
    // values, so later trees hit exact split ties that sklearn breaks with
    // its random feature order: those are compared after one stage.
    let cases: [(GbLoss, usize, [f64; 4], [f64; 4]); 3] = [
        (
            GbLoss::AbsoluteError,
            1,
            [
                7.1493319910508335,
                4.4697537377993,
                7.1493319910508335,
                5.666265386826486,
            ],
            [
                7.219376161585704,
                4.648913126393593,
                7.219376161585704,
                5.937690239021798,
            ],
        ),
        (
            GbLoss::Quantile,
            1,
            [
                8.96625347076722,
                8.96625347076722,
                10.561658555227677,
                8.96625347076722,
            ],
            [
                8.973122457676933,
                8.973122457676933,
                10.568527542137389,
                8.973122457676933,
            ],
        ),
        (
            GbLoss::Huber,
            6,
            [
                7.0924484882073395,
                2.241821093617352,
                10.562177310977049,
                5.600735664537336,
            ],
            [
                7.258995742549587,
                1.7476201235155198,
                10.857367055141799,
                5.682123558434269,
            ],
        ),
    ];
    for (loss, n_estimators, unweighted, weighted) in cases {
        let gb = GradientBoostingRegressor {
            n_estimators,
            learning_rate: 0.3,
            max_depth: Some(2),
            ..Default::default()
        }
        .with_loss(loss)
        .with_alpha(0.8);
        let a = Fit::fit(&gb, &x, &y).unwrap().predict(&q).unwrap();
        assert_close(a.as_slice().unwrap(), &unweighted);
        let b = gb
            .fit_weighted(&x, &y, Some(&w))
            .unwrap()
            .predict(&q)
            .unwrap();
        assert_close(b.as_slice().unwrap(), &weighted);
    }
    // The 0.8 quantile model leaves about 80 % of the targets below it.
    let f = Fit::fit(
        &GradientBoostingRegressor {
            n_estimators: 50,
            max_depth: Some(2),
            ..Default::default()
        }
        .with_loss(GbLoss::Quantile)
        .with_alpha(0.8),
        &x,
        &y,
    )
    .unwrap();
    let below = (&y - &f.predict(&x).unwrap())
        .iter()
        .filter(|&&d| d <= 0.0)
        .count();
    assert!((24..=40).contains(&below), "{below}");
}

#[test]
fn gradient_boosting_early_stopping_stops() {
    use anofox_ml_ensemble::GradientBoostingRegressor;
    let (x, yr, _, _, _) = gb_data();
    let gb = GradientBoostingRegressor {
        n_estimators: 500,
        learning_rate: 0.5,
        ..Default::default()
    }
    .with_early_stopping(Some(3), 0.2, 1e-4);
    let f = Fit::fit(&gb, &x, &yr).unwrap();
    assert!(f.n_estimators() < 500, "{}", f.n_estimators());
}

#[test]
fn gradient_boosting_classifier_early_stopping_stops() {
    use anofox_ml_ensemble::GradientBoostingClassifier;
    let (x, _, yb, yc, w) = gb_data();
    for y in [&yb, &yc] {
        let gb = GradientBoostingClassifier {
            n_estimators: 500,
            learning_rate: 0.5,
            ..Default::default()
        }
        .with_early_stopping(Some(3), 0.2, 1e-4);
        let f = FitWeighted::fit_weighted(&gb, &x, y, Some(&w)).unwrap();
        assert!(f.n_estimators() < 500, "{}", f.n_estimators());
        // Deterministic per seed.
        let g = FitWeighted::fit_weighted(&gb, &x, y, Some(&w)).unwrap();
        assert_eq!(f.n_estimators(), g.n_estimators());
        assert_eq!(f.predict_proba(&x).unwrap(), g.predict_proba(&x).unwrap());
    }
}

#[test]
fn gradient_boosting_max_features_draws_per_round() {
    use anofox_ml_ensemble::{GradientBoostingClassifier, GradientBoostingRegressor};
    use anofox_ml_trees::MaxFeatures;
    let (x, yr, yb, _, _) = gb_data();
    // With one feature per split, every round must draw its own features:
    // a fixed per-tree seed would make all rounds' stumps split on the same
    // feature.
    let stump_features = |imp: Array1<f64>| imp.iter().filter(|&&v| v > 0.0).count();
    let reg = GradientBoostingRegressor {
        n_estimators: 30,
        max_depth: Some(1),
        ..Default::default()
    }
    .with_max_features(Some(MaxFeatures::Fixed(1)));
    let fr = Fit::fit(&reg, &x, &yr).unwrap();
    assert!(stump_features(fr.feature_importances()) >= 2);
    let clf = GradientBoostingClassifier {
        n_estimators: 30,
        max_depth: Some(1),
        ..Default::default()
    }
    .with_max_features(Some(MaxFeatures::Fixed(1)));
    let fc = Fit::fit(&clf, &x, &yb).unwrap();
    assert!(stump_features(fc.feature_importances()) >= 2);
    let acc = fc
        .predict(&x)
        .unwrap()
        .iter()
        .zip(&yb)
        .filter(|(p, t)| p == t)
        .count();
    assert!(acc >= 24, "{acc}/30");
}
