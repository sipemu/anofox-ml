//! `Cca` through the registry (#125): fitted on `[X | Y]`, the last
//! `y_columns` columns being `Y`; `transform` gives the X scores and
//! `transform_y` the Y scores.

use anofox_ml_registry::{make, Out};
use ndarray::{concatenate, Array2, Axis};
use serde_json::json;

fn out2(o: Out) -> Array2<f64> {
    match o {
        Out::A2(a) => a,
        Out::A1(a) => panic!("expected a 2-D output, got {} values", a.len()),
    }
}

/// X: 20 × 3; Y: 20 × 2, linear in X plus small deterministic noise.
fn data() -> (Array2<f64>, Array2<f64>) {
    let x = Array2::from_shape_fn((20, 3), |(i, j)| {
        ((i * 7 + j * 3 + i * j) % 13) as f64 / 6.0 - 1.0
    });
    let y = Array2::from_shape_fn((20, 2), |(i, k)| {
        let r = x.row(i);
        if k == 0 {
            r[0] + 0.5 * r[1] + (((i * 5) % 7) as f64 - 3.0) / 10.0
        } else {
            r[2] - r[0] + (((i * 3) % 5) as f64 - 2.0) / 10.0
        }
    });
    (x, y)
}

fn assert_close(a: &Array2<f64>, b: &[[f64; 2]; 2], what: &str) {
    for (i, row) in b.iter().enumerate() {
        for (j, e) in row.iter().enumerate() {
            assert!(
                (a[[i, j]] - e).abs() < 1e-9,
                "{what}[{i},{j}]: {} vs {e}",
                a[[i, j]]
            );
        }
    }
}

#[test]
fn cca_scores_match_scikit_learn() {
    // scikit-learn 1.8.0 `CCA(n_components=2).fit(X, Y).transform(X[:2], Y[:2])`.
    let (x, y) = data();
    let xy = concatenate(Axis(1), &[x.view(), y.view()]).unwrap();
    let mut m = make("Cca", &json!({"n_components": 2, "y_columns": 2})).unwrap();
    m.fit(xy.view(), None, None).unwrap();
    let xs = out2(m.transform(x.slice(ndarray::s![..2, ..])).unwrap());
    let ys = out2(m.transform_y(y.slice(ndarray::s![..2, ..])).unwrap());
    assert_close(
        &xs,
        &[
            [-1.2472846665634445, -1.1863803207797634],
            [1.1519267111428795, 0.03201131317771875],
        ],
        "x scores",
    );
    assert_close(
        &ys,
        &[
            [-1.3699713775002949, -1.2436787412113728],
            [1.3600215627890784, 0.34572336598576175],
        ],
        "y scores",
    );
}

#[test]
fn cca_checks_columns_and_round_trips() {
    let (x, y) = data();
    let xy = concatenate(Axis(1), &[x.view(), y.view()]).unwrap();
    for bad in [0, 5] {
        let mut m = make("Cca", &json!({"y_columns": bad})).unwrap();
        let err = m.fit(xy.view(), None, None).unwrap_err().to_string();
        assert!(err.contains("y_columns"), "{bad}: {err}");
    }
    let mut m = make("Cca", &json!({"n_components": 2, "y_columns": 2})).unwrap();
    m.fit(xy.view(), None, None).unwrap();
    // transform takes the X columns only, transform_y the Y columns only.
    assert!(m.transform(xy.view()).is_err());
    assert!(m.transform_y(x.view()).is_err());

    let back = anofox_ml_registry::load("Cca", &m.to_bytes().unwrap()).unwrap();
    assert_eq!(
        out2(back.transform(x.view()).unwrap()),
        out2(m.transform(x.view()).unwrap())
    );
    assert_eq!(
        out2(back.transform_y(y.view()).unwrap()),
        out2(m.transform_y(y.view()).unwrap())
    );

    assert!(m.capabilities().contains(&"transform_y"));
    let pca = make("Pca", &json!({})).unwrap();
    assert!(!pca.capabilities().contains(&"transform_y"));
}
