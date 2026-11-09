//! `DynModel::kneighbors` (#125): distances and training-row indices of the
//! nearest neighbours of the fitted kNN models.

use anofox_ml_registry::make;
use ndarray::{array, Array1, Array2};
use serde_json::json;

/// Five points on a line: 0, 1, 3, 6, 10.
fn line() -> (Array2<f64>, Array1<f64>) {
    let x = array![[0.0], [1.0], [3.0], [6.0], [10.0]];
    let y = array![0.0, 0.0, 1.0, 1.0, 1.0];
    (x, y)
}

#[test]
fn knn_classifier_finds_the_nearest_training_rows() {
    let (x, y) = line();
    let mut m = make("KnnClassifier", &json!({"n_neighbors": 2})).unwrap();
    m.fit(x.view(), Some(y.view()), None).unwrap();
    // 2.4 is 0.6 from 3, 1.4 from 1 and 2.4 from 0.
    let (dist, idx) = m.kneighbors(array![[2.4]].view(), Some(3)).unwrap();
    assert_eq!(idx, array![[2, 1, 0]]);
    for (d, e) in dist.iter().zip([0.6, 1.4, 2.4]) {
        assert!((d - e).abs() < 1e-12, "{d} vs {e}");
    }
}

#[test]
fn knn_regressor_defaults_to_n_neighbors_and_finds_itself() {
    let (x, y) = line();
    let mut m = make("KnnRegressor", &json!({"n_neighbors": 2})).unwrap();
    m.fit(x.view(), Some(y.view()), None).unwrap();
    let (dist, idx) = m.kneighbors(x.view(), None).unwrap();
    assert_eq!(dist.dim(), (5, 2));
    // Every training row is its own nearest neighbour.
    for i in 0..5 {
        assert_eq!(idx[[i, 0]], i);
        assert_eq!(dist[[i, 0]], 0.0);
    }
    // 6's next neighbour is 3 (distance 3), before 10 (distance 4).
    assert_eq!((idx[[3, 1]], dist[[3, 1]]), (2, 3.0));
}

#[test]
fn only_the_knn_kinds_list_it() {
    let (x, y) = line();
    for kind in ["KnnClassifier", "KnnRegressor"] {
        let m = make(kind, &json!({})).unwrap();
        assert!(m.capabilities().contains(&"kneighbors"), "{kind}");
    }
    let mut m = make("DecisionTreeRegressor", &json!({})).unwrap();
    m.fit(x.view(), Some(y.view()), None).unwrap();
    assert!(!m.capabilities().contains(&"kneighbors"));
    let err = m.kneighbors(x.view(), Some(1)).unwrap_err().to_string();
    assert!(
        err.contains("DecisionTreeRegressor does not support kneighbors"),
        "{err}"
    );
}

#[test]
fn kneighbors_survives_a_round_trip() {
    let (x, y) = line();
    let mut m = make("KnnClassifier", &json!({"n_neighbors": 3})).unwrap();
    m.fit(x.view(), Some(y.view()), None).unwrap();
    let restored = anofox_ml_registry::load("KnnClassifier", &m.to_bytes().unwrap()).unwrap();
    let q = array![[2.0], [7.5]];
    assert_eq!(
        restored.kneighbors(q.view(), None).unwrap(),
        m.kneighbors(q.view(), None).unwrap()
    );
}
