//! `DynModel::partial_fit` (#125): batches folded into the stored model.

use anofox_ml_registry::{make, DynModel, Out};
use ndarray::{array, s, Array1, Array2};
use serde_json::json;

fn out2(o: Out) -> Array2<f64> {
    match o {
        Out::A2(a) => a,
        Out::A1(a) => a.insert_axis(ndarray::Axis(1)),
    }
}

/// Twelve rows of non-negative counts, three classes, all classes in both
/// halves.
fn data() -> (Array2<f64>, Array1<f64>) {
    let x = array![
        [3.0, 0.0, 1.0],
        [0.0, 3.0, 1.0],
        [1.0, 0.0, 4.0],
        [4.0, 1.0, 0.0],
        [1.0, 4.0, 0.0],
        [0.0, 1.0, 3.0],
        [2.0, 0.0, 2.0],
        [0.0, 2.0, 2.0],
        [2.0, 1.0, 3.0],
        [5.0, 1.0, 1.0],
        [1.0, 5.0, 2.0],
        [1.0, 2.0, 5.0]
    ];
    let y = array![0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0];
    (x, y)
}

/// `fit` on all rows and `partial_fit` on the two halves.
fn whole_and_batched(
    kind: &str,
    params: serde_json::Value,
    supervised: bool,
) -> (Box<dyn DynModel>, Box<dyn DynModel>) {
    let (x, y) = data();
    let y = supervised.then_some(y);
    let mut whole = make(kind, &params).unwrap();
    whole
        .fit(x.view(), y.as_ref().map(|y| y.view()), None)
        .unwrap();
    let mut batched = make(kind, &params).unwrap();
    for (a, b) in [(0, 6), (6, 12)] {
        let yb = y.as_ref().map(|y| y.slice(s![a..b]));
        batched
            .partial_fit(x.slice(s![a..b, ..]), yb, None, Some(&[0.0, 1.0, 2.0]))
            .unwrap();
    }
    (whole, batched)
}

fn assert_close(a: &Array2<f64>, b: &Array2<f64>, what: &str) {
    assert_eq!(a.dim(), b.dim(), "{what}");
    for (u, v) in a.iter().zip(b) {
        assert!((u - v).abs() < 1e-10, "{what}: {u} vs {v}");
    }
}

#[test]
fn gaussian_nb_batches_equal_one_fit() {
    let (x, _) = data();
    let (whole, batched) = whole_and_batched("GaussianNB", json!({}), true);
    assert_close(
        &out2(whole.predict_proba(x.view()).unwrap()),
        &out2(batched.predict_proba(x.view()).unwrap()),
        "GaussianNB",
    );
}

#[test]
fn batches_equal_one_fit_for_the_exact_kinds() {
    let (x, _) = data();
    for kind in [
        "MultinomialNB",
        "BernoulliNB",
        "ComplementNB",
        "CategoricalNB",
    ] {
        let (whole, batched) = whole_and_batched(kind, json!({}), true);
        assert_close(
            &out2(whole.predict_proba(x.view()).unwrap()),
            &out2(batched.predict_proba(x.view()).unwrap()),
            kind,
        );
    }
    // IncrementalPca's fit runs in batches too: of 6 rows here, as above.
    for (kind, params) in [
        ("StandardScaler", json!({})),
        ("MinMaxScaler", json!({})),
        ("MaxAbsScaler", json!({})),
        ("IncrementalPca", json!({"batch_size": 6})),
    ] {
        let (whole, batched) = whole_and_batched(kind, params, false);
        assert_close(
            &out2(whole.transform(x.view()).unwrap()),
            &out2(batched.transform(x.view()).unwrap()),
            kind,
        );
    }
}

/// The predictions after the registry's `partial_fit` on both halves, and
/// after the crate's own `PartialFit` chain on them.
fn registry_and_crate<P: anofox_ml_core::PartialFit<f64>>(
    kind: &str,
    config: &P,
    classes: Option<&[f64]>,
    predict: impl Fn(&P::Fitted, &Array2<f64>) -> Array1<f64>,
) -> (Array1<f64>, Array1<f64>) {
    let (x, y) = data();
    let mut m = make(kind, &json!({})).unwrap();
    let mut state = None;
    for (a, b) in [(0, 6), (6, 12)] {
        let (xb, yb) = (x.slice(s![a..b, ..]), y.slice(s![a..b]));
        m.partial_fit(xb, Some(yb), None, classes).unwrap();
        state = Some(config.partial_fit_view(state, xb, yb, classes).unwrap());
    }
    let ours = match m.predict(x.view()).unwrap() {
        Out::A1(a) => a,
        Out::A2(a) => panic!("{kind}: 2-D predict {:?}", a.dim()),
    };
    (ours, predict(&state.unwrap(), &x))
}

#[test]
fn epoch_kinds_follow_the_crates_state_chain() {
    use anofox_ml_core::Predict;
    let classes = Some(&[0.0, 1.0, 2.0][..]);
    let pairs = [
        registry_and_crate(
            "SgdClassifier",
            &anofox_ml_linear::SgdClassifier::default(),
            classes,
            |f, x| f.predict(x).unwrap(),
        ),
        registry_and_crate(
            "SgdRegressor",
            &anofox_ml_linear::SgdRegressor::default(),
            None,
            |f, x| f.predict(x).unwrap(),
        ),
        registry_and_crate(
            "PassiveAggressiveRegressor",
            &anofox_ml_linear::PassiveAggressiveRegressor::default(),
            None,
            |f, x| f.predict(x).unwrap(),
        ),
        registry_and_crate(
            "MlpClassifier",
            &anofox_ml_neural_networks::MlpClassifier::default(),
            classes,
            |f, x| f.predict(x).unwrap(),
        ),
        registry_and_crate(
            "MlpRegressor",
            &anofox_ml_neural_networks::MlpRegressor::default(),
            None,
            |f, x| f.predict(x).unwrap(),
        ),
    ];
    for (k, (ours, theirs)) in pairs.into_iter().enumerate() {
        assert_eq!(ours, theirs, "kind #{k}");
    }
}

#[test]
fn exactly_the_incremental_kinds_list_it() {
    let mut expected = vec![
        "BernoulliNB",
        "CategoricalNB",
        "ComplementNB",
        "GaussianNB",
        "IncrementalPca",
        "MaxAbsScaler",
        "MinMaxScaler",
        "MlpClassifier",
        "MlpRegressor",
        "MultinomialNB",
        "PassiveAggressiveRegressor",
        "SgdClassifier",
        "SgdRegressor",
        "StandardScaler",
    ];
    expected.sort_unstable();
    let mut listed: Vec<&str> = anofox_ml_registry::kinds()
        .into_iter()
        .filter(|k| {
            make(k, &json!({}))
                .map(|m| m.capabilities().contains(&"partial_fit"))
                .unwrap_or(false)
        })
        .collect();
    listed.sort_unstable();
    assert_eq!(listed, expected);

    let (x, y) = data();
    let mut m = make("DecisionTreeClassifier", &json!({})).unwrap();
    let err = m
        .partial_fit(x.view(), Some(y.view()), None, None)
        .unwrap_err();
    assert!(err
        .to_string()
        .contains("DecisionTreeClassifier does not support partial_fit"));
}

#[test]
fn a_failed_batch_leaves_the_model_unchanged() {
    let (x, y) = data();
    let mut m = make("GaussianNB", &json!({})).unwrap();
    m.partial_fit(x.view(), Some(y.view()), None, None).unwrap();
    let before = out2(m.predict_proba(x.view()).unwrap());
    // A label outside the classes of the first batch.
    let bad = array![5.0, 5.0];
    assert!(m
        .partial_fit(x.slice(s![..2, ..]), Some(bad.view()), None, None)
        .is_err());
    assert_eq!(out2(m.predict_proba(x.view()).unwrap()), before);

    let mut sgd = make("SgdRegressor", &json!({})).unwrap();
    sgd.partial_fit(x.view(), Some(y.view()), None, None)
        .unwrap();
    let before = out2(sgd.predict(x.view()).unwrap());
    let w = Array1::ones(12);
    let err = sgd
        .partial_fit(x.view(), Some(y.view()), Some(w.view()), None)
        .unwrap_err();
    assert!(err.to_string().contains("sample_weight"), "{err}");
    assert_eq!(out2(sgd.predict(x.view()).unwrap()), before);
}

#[test]
fn partial_fit_survives_a_round_trip_between_batches() {
    let (x, y) = data();
    let classes = Some(&[0.0, 1.0, 2.0][..]);
    let (first, second) = (
        (x.slice(s![..6, ..]), y.slice(s![..6])),
        (x.slice(s![6.., ..]), y.slice(s![6..])),
    );
    for kind in ["SgdClassifier", "GaussianNB", "StandardScaler"] {
        let sup = kind != "StandardScaler";
        let ys = |b| sup.then_some(b);
        let mut straight = make(kind, &json!({})).unwrap();
        straight
            .partial_fit(first.0, ys(first.1), None, classes)
            .unwrap();
        straight
            .partial_fit(second.0, ys(second.1), None, classes)
            .unwrap();
        let mut stored = make(kind, &json!({})).unwrap();
        stored
            .partial_fit(first.0, ys(first.1), None, classes)
            .unwrap();
        let mut stored = anofox_ml_registry::load(kind, &stored.to_bytes().unwrap()).unwrap();
        stored
            .partial_fit(second.0, ys(second.1), None, classes)
            .unwrap();
        let out = |m: &dyn DynModel| {
            out2(
                if sup {
                    m.predict(x.view())
                } else {
                    m.transform(x.view())
                }
                .unwrap(),
            )
        };
        assert_eq!(out(stored.as_ref()), out(straight.as_ref()), "{kind}");
    }
}
