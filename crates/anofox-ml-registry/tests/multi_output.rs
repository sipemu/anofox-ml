//! `MultiOutputRegressor` / `RegressorChain` through the registry (#125):
//! fitted on `[X | Y]`, the last `y_columns` columns being the targets,
//! against scikit-learn 1.8.0 on the same inline data.

use anofox_ml_registry::{make, Out};
use ndarray::{array, concatenate, Array2, Axis};
use serde_json::{json, Value};

fn out2(o: Out) -> Array2<f64> {
    match o {
        Out::A2(a) => a,
        Out::A1(a) => panic!("expected a 2-D output, got {} values", a.len()),
    }
}

fn data() -> (Array2<f64>, Array2<f64>) {
    let x = Array2::from_shape_fn((30, 3), |(i, j)| {
        ((i * 7 + j * 3 + i * j) % 13) as f64 / 6.0 - 1.0
    });
    let y = Array2::from_shape_fn((30, 2), |(i, k)| {
        let r = x.row(i);
        if k == 0 {
            r[0] - 2.0 * r[1] + r[2] * r[2]
        } else {
            (2.0 * r[0]).sin() + r[1] * r[2]
        }
    });
    (x, y)
}

fn fit_predict(kind: &str, spec: Value) -> Array2<f64> {
    let (x, y) = data();
    let xy = concatenate(Axis(1), &[x.view(), y.view()]).unwrap();
    let mut m = make(kind, &spec).unwrap();
    m.fit(xy.view(), None, None).unwrap();
    out2(
        m.predict(array![[-0.9, -0.6, -0.4], [-0.2, 0.3, 0.7], [0.4, -0.6, 0.7]].view())
            .unwrap(),
    )
}

fn close(a: &Array2<f64>, e: [[f64; 2]; 3], what: &str) {
    for (i, row) in e.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            assert!(
                (a[[i, j]] - v).abs() < 1e-12,
                "{what}[{i},{j}]: {} vs {v}",
                a[[i, j]]
            );
        }
    }
}

#[test]
fn multi_output_and_chains_match_scikit_learn() {
    let knn = json!({"n_neighbors": 3});
    close(
        &fit_predict(
            "MultiOutputRegressor",
            json!({"estimator": "KnnRegressor", "estimator_params": knn, "y_columns": 2}),
        ),
        [
            [0.0, -0.9092974268256816],
            [-0.19444444444444434, -0.1836545044357735],
            [2.1203703703703707, 0.4985854057099914],
        ],
        "MultiOutputRegressor(KNN(3))",
    );
    close(
        &fit_predict(
            "RegressorChain",
            json!({"estimator": "KnnRegressor", "estimator_params": knn, "y_columns": 2, "order": [1, 0]}),
        ),
        [
            [0.0, -0.9092974268256816],
            [-0.9444444444444443, -0.1836545044357735],
            [2.1203703703703707, 0.4985854057099914],
        ],
        "RegressorChain(KNN(3), order=[1, 0])",
    );
    close(
        &fit_predict(
            "RegressorChain",
            json!({"estimator": "OlsRegressor", "y_columns": 2}),
        ),
        [
            [0.6539162428425835, -0.9377306905715634],
            [-0.43807582426307734, -0.23111166343160577],
            [2.07957861670227, 0.7462478405017321],
        ],
        "RegressorChain(LinearRegression())",
    );
}

#[test]
fn multi_output_checks_columns_and_round_trips() {
    let (x, y) = data();
    let xy = concatenate(Axis(1), &[x.view(), y.view()]).unwrap();
    let mut bad = make("MultiOutputRegressor", &json!({"y_columns": 3})).unwrap();
    assert!(bad
        .fit(x.view(), None, None)
        .unwrap_err()
        .to_string()
        .contains("y_columns"));
    let mut bad = make("RegressorChain", &json!({"y_columns": 2, "order": [0, 0]})).unwrap();
    assert!(bad
        .fit(xy.view(), None, None)
        .unwrap_err()
        .to_string()
        .contains("order"));

    let mut m = make(
        "RegressorChain",
        &json!({"estimator": "KnnRegressor", "y_columns": 2, "order": [1, 0]}),
    )
    .unwrap();
    m.fit(xy.view(), None, None).unwrap();
    let back = anofox_ml_registry::load("RegressorChain", &m.to_bytes().unwrap()).unwrap();
    assert_eq!(
        out2(back.predict(x.view()).unwrap()),
        out2(m.predict(x.view()).unwrap())
    );
}
