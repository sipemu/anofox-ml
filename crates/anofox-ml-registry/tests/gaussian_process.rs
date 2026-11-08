//! Gaussian processes through the registry (#125), against scikit-learn
//! 1.8.0 on the same inline data.

use anofox_ml_registry::{make, Out};
use ndarray::{array, Array1, Array2};
use serde_json::json;

fn out1(o: Out) -> Array1<f64> {
    match o {
        Out::A1(a) => a,
        Out::A2(a) => panic!("expected a 1-D output, got {:?}", a.dim()),
    }
}

fn out2(o: Out) -> Array2<f64> {
    match o {
        Out::A2(a) => a,
        Out::A1(a) => panic!("expected a 2-D output, got {} values", a.len()),
    }
}

fn data() -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_fn((15, 2), |(i, j)| {
        if j == 0 {
            ((i * 7) % 11) as f64 / 4.0 - 1.2
        } else {
            ((i * 5 + 2) % 9) as f64 / 3.0 - 1.0
        }
    });
    let y = x
        .rows()
        .into_iter()
        .map(|r| (2.0 * r[0]).sin() + 0.5 * r[1])
        .collect();
    (x, y)
}

fn queries() -> Array2<f64> {
    array![[0.1, 0.2], [-0.7, 0.5], [1.1, -0.4]]
}

fn assert_close(ours: &[f64], expected: &[f64], tol: f64, what: &str) {
    assert_eq!(ours.len(), expected.len(), "{what}");
    for (o, e) in ours.iter().zip(expected) {
        assert!((o - e).abs() < tol, "{what}: {o} vs {e}");
    }
}

#[test]
fn regressor_default_kernel_matches_scikit_learn() {
    // `GaussianProcessRegressor()`: 1.0 * RBF(1.0), both fixed.
    let (x, y) = data();
    let mut m = make("GaussianProcessRegressor", &json!({})).unwrap();
    m.fit(x.view(), Some(y.view()), None).unwrap();
    let q = queries();
    assert_close(
        out1(m.predict(q.view()).unwrap()).as_slice().unwrap(),
        &[0.3103492475016276, -0.710188176381334, 0.6227989732805139],
        1e-9,
        "mean",
    );
    assert_close(
        out1(m.predict_std(q.view()).unwrap()).as_slice().unwrap(),
        &[
            0.0306896818176486,
            0.032477016557222196,
            0.09778066959117146,
        ],
        1e-9,
        "std",
    );
}

#[test]
fn regressor_optimised_kernel_matches_scikit_learn() {
    // `GaussianProcessRegressor(kernel=C(1.0) * RBF(1.0))`, optimised to
    // `0.941**2 * RBF(length_scale=1)`.
    let (x, y) = data();
    let params = json!({"kernel": {"Rbf": {"length_scale": 1.0, "signal_var": 1.0}}});
    let mut m = make("GaussianProcessRegressor", &params).unwrap();
    m.fit(x.view(), Some(y.view()), None).unwrap();
    let q = queries();
    assert_close(
        out1(m.predict(q.view()).unwrap()).as_slice().unwrap(),
        &[0.3103288501292587, -0.7102116747160396, 0.6227838687217313],
        1e-6,
        "mean",
    );
    assert_close(
        out1(m.predict_std(q.view()).unwrap()).as_slice().unwrap(),
        &[
            0.028832595702545566,
            0.030514646986010058,
            0.09191262293732114,
        ],
        1e-6,
        "std",
    );
}

#[test]
fn classifier_default_kernel_matches_scikit_learn() {
    // `GaussianProcessClassifier()`: binary, and one-vs-rest for 3 classes.
    let (x, _) = data();
    let q = queries();
    let y2: Array1<f64> = x
        .rows()
        .into_iter()
        .map(|r| f64::from(r[0] + 0.3 * r[1] > 0.0))
        .collect();
    let mut m = make("GaussianProcessClassifier", &json!({})).unwrap();
    m.fit(x.view(), Some(y2.view()), None).unwrap();
    let p = out2(m.predict_proba(q.view()).unwrap());
    assert_close(
        &p.column(1).to_vec(),
        &[0.5069219010275212, 0.3459058092768146, 0.6451285836385523],
        1e-9,
        "binary",
    );

    let y3: Array1<f64> = x
        .rows()
        .into_iter()
        .map(|r| {
            if r[0] < -0.4 {
                0.0
            } else if r[0] < 0.6 {
                1.0
            } else {
                2.0
            }
        })
        .collect();
    let mut m = make("GaussianProcessClassifier", &json!({})).unwrap();
    m.fit(x.view(), Some(y3.view()), None).unwrap();
    let p = out2(m.predict_proba(q.view()).unwrap());
    let expected = [
        0.36568223300113833,
        0.37239497418263046,
        0.26192279281623126,
        0.5211539407624358,
        0.2965886983859649,
        0.1822573608515993,
        0.23580207686220367,
        0.30760855334901765,
        0.4565893697887787,
    ];
    assert_close(p.as_slice().unwrap(), &expected, 1e-9, "3 classes");
}

#[test]
fn fitted_processes_round_trip_exactly() {
    let (x, y) = data();
    let q = queries();
    let params = json!({"kernel": {"Rbf": {"length_scale": 1.0, "signal_var": 1.0}}});
    let mut m = make("GaussianProcessRegressor", &params).unwrap();
    m.fit(x.view(), Some(y.view()), None).unwrap();
    let back =
        anofox_ml_registry::load("GaussianProcessRegressor", &m.to_bytes().unwrap()).unwrap();
    assert_eq!(
        out1(back.predict(q.view()).unwrap()),
        out1(m.predict(q.view()).unwrap())
    );
    assert_eq!(
        out1(back.predict_std(q.view()).unwrap()),
        out1(m.predict_std(q.view()).unwrap())
    );

    let yc: Array1<f64> = x
        .rows()
        .into_iter()
        .map(|r| f64::from(r[0] > 0.0))
        .collect();
    let mut c = make("GaussianProcessClassifier", &json!({})).unwrap();
    c.fit(x.view(), Some(yc.view()), None).unwrap();
    let back =
        anofox_ml_registry::load("GaussianProcessClassifier", &c.to_bytes().unwrap()).unwrap();
    assert_eq!(
        out2(back.predict_proba(q.view()).unwrap()),
        out2(c.predict_proba(q.view()).unwrap())
    );
}

#[test]
fn only_the_regressor_has_predict_std() {
    assert!(make("GaussianProcessRegressor", &json!({}))
        .unwrap()
        .capabilities()
        .contains(&"predict_std"));
    let (x, y) = data();
    let yc: Array1<f64> = y.mapv(|v| f64::from(v > 0.0));
    let mut c = make("GaussianProcessClassifier", &json!({})).unwrap();
    c.fit(x.view(), Some(yc.view()), None).unwrap();
    assert!(!c.capabilities().contains(&"predict_std"));
    let err = c.predict_std(x.view()).unwrap_err().to_string();
    assert!(
        err.contains("GaussianProcessClassifier does not support predict_std"),
        "{err}"
    );
}
