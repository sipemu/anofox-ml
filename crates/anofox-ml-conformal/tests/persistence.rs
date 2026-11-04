//! Every fitted conformal model survives a round trip through the binary
//! persistence format (v2) with identical intervals and sets.

use anofox_ml_conformal::*;
use anofox_ml_core::persistence::{from_binary_bytes, to_binary_bytes};
use anofox_ml_core::Fit;
use anofox_ml_ensemble::{FittedRandomForestClassifier, RandomForestClassifier};
use anofox_ml_trees::{DecisionTreeRegressor, FittedDecisionTreeRegressor};
use ndarray::{Array1, Array2};
use serde::{de::DeserializeOwned, Serialize};

fn data() -> (Array2<f64>, Array1<f64>, Array1<f64>, Vec<usize>) {
    let x = Array2::from_shape_fn((200, 2), |(i, j)| ((i * (3 + 2 * j)) % 29) as f64 / 5.0);
    let y = Array1::from_shape_fn(200, |i| {
        x[[i, 0]] - 0.5 * x[[i, 1]] + ((i * 7) % 11) as f64 / 10.0
    });
    let c = Array1::from_shape_fn(200, |i| ((x[[i, 0]] + x[[i, 1]]) as usize % 3) as f64);
    let g = (0..200).map(|i| i % 2).collect();
    (x, y, c, g)
}

fn round_trip<T: Serialize + DeserializeOwned>(t: &T) -> T {
    from_binary_bytes(&to_binary_bytes(t).unwrap()).unwrap()
}

fn tree() -> DecisionTreeRegressor {
    DecisionTreeRegressor {
        max_depth: Some(4),
        ..Default::default()
    }
}

#[test]
fn regressors_round_trip() {
    let (x, y, _, g) = data();
    let v = x.view();
    let split = SplitConformalRegressor::new(tree()).fit(&x, &y).unwrap();
    assert_eq!(
        split.predict_interval(v, 0.1).unwrap(),
        round_trip(&split).predict_interval(v, 0.1).unwrap()
    );
    for method in [CrossMethod::Plus, CrossMethod::MinMax, CrossMethod::Base] {
        let cross = CrossConformalRegressor::new(tree())
            .with_method(method)
            .fit(&x, &y)
            .unwrap();
        let back: FittedCrossConformalRegressor<FittedDecisionTreeRegressor<f64>, f64> =
            round_trip(&cross);
        assert_eq!(
            cross.predict_interval(v, 0.2).unwrap(),
            back.predict_interval(v, 0.2).unwrap()
        );
    }
    let cqr = ConformalizedQuantileRegressor::new(0.2, |_q| tree())
        .fit(&x, &y)
        .unwrap();
    assert_eq!(
        cqr.predict_interval(v).unwrap(),
        round_trip(&cqr).predict_interval(v).unwrap()
    );
    let norm = NormalizedConformalRegressor::new(tree(), tree())
        .fit(&x, &y)
        .unwrap();
    assert_eq!(
        norm.predict_interval(v, 0.1).unwrap(),
        round_trip(&norm).predict_interval(v, 0.1).unwrap()
    );
    let mondrian = MondrianConformalRegressor::new(tree())
        .fit_with_groups(v, y.view(), &g)
        .unwrap();
    assert_eq!(
        mondrian.predict_interval(v, &g, 0.2).unwrap(),
        round_trip(&mondrian).predict_interval(v, &g, 0.2).unwrap()
    );
}

#[test]
fn classifiers_round_trip() {
    let (x, _, c, g) = data();
    let v = x.view();
    for score in [
        ClassificationScore::Lac,
        ClassificationScore::Aps(ApsMode::Randomized),
        ClassificationScore::Raps(RapsParams::new(ApsMode::Include)),
    ] {
        let fitted = SplitConformalClassifier::new(RandomForestClassifier::new(10))
            .with_score(score)
            .fit(&x, &c)
            .unwrap();
        let back: FittedSplitConformalClassifier<FittedRandomForestClassifier<f64>, f64> =
            round_trip(&fitted);
        assert_eq!(
            fitted.predict_set(v, 0.2).unwrap(),
            back.predict_set(v, 0.2).unwrap(),
            "{score:?}"
        );
    }
    let model = RandomForestClassifier::new(10).fit(&x, &c).unwrap();
    let m = conformalize_classifier_mondrian(
        model,
        vec![0.0, 1.0, 2.0],
        v,
        c.view(),
        &g,
        ClassificationScore::Lac,
        0,
    )
    .unwrap();
    assert_eq!(
        m.predict_set_mondrian(v, &g, 0.2).unwrap(),
        round_trip(&m).predict_set_mondrian(v, &g, 0.2).unwrap()
    );
}
