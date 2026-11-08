//! Kinds added to the registry for the DuckDB extension (#125):
//! LassoLarsIC and BinomialRegressor.

use anofox_ml_registry::{make, DynModel, Out};
use ndarray::{array, Array1, Array2};
use serde_json::{json, Value};

fn out1(o: Out) -> Array1<f64> {
    match o {
        Out::A1(a) => a,
        Out::A2(a) => panic!("expected a 1-D output, got {:?}", a.dim()),
    }
}

/// 40 × 6, two strong features and two weak ones (AIC keeps one of the
/// weak ones, BIC none).
fn lars_data() -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_fn((40, 6), |(i, j)| {
        ((i * 7 + j * 3 + i * j) % 13) as f64 / 6.0 - 1.0
    });
    let y = Array1::from_shape_fn(40, |i| {
        let r = x.row(i);
        2.0 * r[0] - r[1] + 0.2 * (r[3] + r[4]) + (((i * 13) % 7) as f64 - 3.0) / 2.0
    });
    (x, y)
}

fn fitted_field(m: &dyn DynModel, field: &str) -> f64 {
    let v: Value = serde_json::from_str(&m.fitted_json().unwrap()).unwrap();
    v[field].as_f64().unwrap()
}

#[test]
fn lasso_lars_ic_matches_scikit_learn() {
    // scikit-learn 1.8.0 `LassoLarsIC(criterion=...)` on the same data:
    // `alpha_` and `predict(X[:3])`.
    let expected = [
        (
            "Aic",
            0.010422461019806168,
            [
                -1.5292035398230097,
                -0.8197218710493035,
                -1.6744627054361574,
            ],
        ),
        (
            "Bic",
            0.09725750202757497,
            [
                -1.2870072992700743,
                -0.4907866991078663,
                -1.4839253852392549,
            ],
        ),
    ];
    let (x, y) = lars_data();
    for (criterion, alpha, pred) in expected {
        let mut m = make("LassoLarsIC", &json!({"criterion": criterion})).unwrap();
        m.fit(x.view(), Some(y.view()), None).unwrap();
        assert!(
            (fitted_field(m.as_ref(), "alpha") - alpha).abs() < 1e-12,
            "{criterion} alpha"
        );
        let ours = out1(m.predict(x.slice(ndarray::s![..3, ..])).unwrap());
        for (o, e) in ours.iter().zip(pred) {
            assert!((o - e).abs() < 1e-12, "{criterion}: {o} vs {e}");
        }
    }
}

/// One binary feature: 3 of 10 successes at x = 0, 8 of 10 at x = 1.
fn trials() -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_fn((20, 1), |(i, _)| f64::from(i >= 10));
    let y = Array1::from_shape_fn(20, |i| f64::from(if i < 10 { i < 3 } else { i < 18 }));
    (x, y)
}

#[test]
fn binomial_regressor_recovers_the_group_proportions() {
    // With one binary feature the logit MLE is saturated: the fitted
    // probabilities are the observed proportions 0.3 and 0.8.
    let at = array![[0.0], [1.0]];
    let (x, y) = trials();
    let mut m = make("BinomialRegressor", &json!({})).unwrap();
    m.fit(x.view(), Some(y.view()), None).unwrap();
    let p = out1(m.predict(at.view()).unwrap());
    assert!(
        (p[0] - 0.3).abs() < 1e-8 && (p[1] - 0.8).abs() < 1e-8,
        "{p}"
    );

    // The same data as proportions with the trial counts as weights.
    let (xs, ys, ws) = (array![[0.0], [1.0]], array![0.3, 0.8], array![10.0, 10.0]);
    let mut grouped = make("BinomialRegressor", &json!({})).unwrap();
    grouped
        .fit(xs.view(), Some(ys.view()), Some(ws.view()))
        .unwrap();
    let q = out1(grouped.predict(at.view()).unwrap());
    assert!(
        (q[0] - 0.3).abs() < 1e-8 && (q[1] - 0.8).abs() < 1e-8,
        "{q}"
    );
}

#[test]
fn binomial_offset_shifts_the_logit() {
    // A constant offset c on the logit scale: the fit absorbs it, so at
    // offset c the proportions come back and at offset 0 the logits are
    // shifted by -c.
    let logit = |p: f64| (p / (1.0 - p)).ln();
    let expit = |z: f64| 1.0 / (1.0 + (-z).exp());
    let (x, y) = trials();
    let offset = Array1::from_elem(20, 0.5);
    let mut m = make("BinomialRegressor", &json!({})).unwrap();
    m.fit_with_offset(x.view(), Some(y.view()), None, offset.view())
        .unwrap();
    let at = array![[0.0], [1.0]];
    let with = out1(
        m.predict_with_offset(at.view(), array![0.5, 0.5].view())
            .unwrap(),
    );
    let without = out1(m.predict(at.view()).unwrap());
    for (k, p) in [0.3, 0.8].into_iter().enumerate() {
        assert!((with[k] - p).abs() < 1e-8, "{with}");
        assert!(
            (without[k] - expit(logit(p) - 0.5)).abs() < 1e-8,
            "{without}"
        );
    }
}

#[test]
fn new_kinds_are_listed_and_round_trip() {
    for kind in ["LassoLarsIC", "BinomialRegressor"] {
        assert!(anofox_ml_registry::kinds().contains(&kind), "{kind}");
        assert!(anofox_ml_registry::default_params(kind).is_ok(), "{kind}");
    }
    let (x, y) = lars_data();
    let mut lars = make("LassoLarsIC", &json!({"criterion": "Bic"})).unwrap();
    lars.fit(x.view(), Some(y.view()), None).unwrap();
    let back = anofox_ml_registry::load("LassoLarsIC", &lars.to_bytes().unwrap()).unwrap();
    assert_eq!(
        out1(back.predict(x.view()).unwrap()),
        out1(lars.predict(x.view()).unwrap())
    );

    let (x, y) = trials();
    let mut bin = make("BinomialRegressor", &json!({})).unwrap();
    bin.fit(x.view(), Some(y.view()), None).unwrap();
    let back = anofox_ml_registry::load("BinomialRegressor", &bin.to_bytes().unwrap()).unwrap();
    assert_eq!(
        out1(back.predict(x.view()).unwrap()),
        out1(bin.predict(x.view()).unwrap())
    );
}
