//! `warm_start` of the MLPs through the registry (#125): sklearn's
//! `warm_start=True` refit continues training where the fitted model left
//! off.

use anofox_ml_registry::{make, Out};
use ndarray::{Array1, Array2};
use serde_json::{json, Value};

fn out2(o: Out) -> Array2<f64> {
    match o {
        Out::A2(a) => a,
        Out::A1(a) => a.insert_axis(ndarray::Axis(1)),
    }
}

fn data() -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let x = Array2::from_shape_fn((24, 3), |(i, j)| ((i * 7 + j * 3) % 11) as f64 / 5.0 - 1.0);
    let y_reg = x
        .rows()
        .into_iter()
        .map(|r| r[0] - 0.5 * r[1] + r[2] * r[2])
        .collect();
    let y_clf = x
        .rows()
        .into_iter()
        .map(|r| f64::from(r[0] + r[2] > 0.0))
        .collect();
    (x, y_reg, y_clf)
}

/// Plain SGD without momentum or shuffling: no optimiser state and no draws
/// between epochs, so 3 + 3 epochs are 6 epochs.
fn plain(max_iter: usize) -> Value {
    json!({
        "hidden_layer_sizes": [4],
        "solver": "Sgd",
        "momentum": 0.0,
        "shuffle": false,
        "learning_rate": 0.05,
        "tol": 1e-12,
        "n_iter_no_change": 1000,
        "max_iter": max_iter,
    })
}

#[test]
fn warm_start_continues_the_training() {
    let (x, y_reg, y_clf) = data();
    for (kind, y) in [("MlpRegressor", &y_reg), ("MlpClassifier", &y_clf)] {
        let mut six = make(kind, &plain(6)).unwrap();
        six.fit(x.view(), Some(y.view()), None).unwrap();
        let mut warm = make(kind, &plain(3)).unwrap();
        warm.fit(x.view(), Some(y.view()), None).unwrap();
        warm.warm_start(x.view(), Some(y.view()), None).unwrap();
        let out = |m: &dyn anofox_ml_registry::DynModel| {
            out2(
                if kind == "MlpClassifier" {
                    m.predict_proba(x.view())
                } else {
                    m.predict(x.view())
                }
                .unwrap(),
            )
        };
        let (a, b) = (out(six.as_ref()), out(warm.as_ref()));
        for (u, v) in a.iter().zip(&b) {
            assert!((u - v).abs() < 1e-12, "{kind}: {u} vs {v}");
        }
    }
}

#[test]
fn warm_start_needs_a_fitted_model() {
    let (x, y_reg, _) = data();
    for kind in ["MlpRegressor", "MlpClassifier"] {
        let mut m = make(kind, &plain(3)).unwrap();
        assert!(m.capabilities().contains(&"warm_start"), "{kind}");
        let err = m
            .warm_start(x.view(), Some(y_reg.view()), None)
            .unwrap_err();
        assert!(err.to_string().contains("not fitted"), "{kind}: {err}");
        assert!(!m.is_fitted());
    }
}
