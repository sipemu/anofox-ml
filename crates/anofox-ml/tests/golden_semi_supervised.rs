//! Golden tests for LabelPropagation / LabelSpreading (#113, batch 8)
//! against scikit-learn 1.8.0 (`gen_semi_supervised.py`).

mod common;

use anofox_ml_core::{Fit, Predict, PredictProba};
use anofox_ml_neighbors::{GraphKernel, LabelPropagation, LabelSpreading};
use common::{assert_array2_close, json_to_array1, json_to_array2};
use serde_json::Value;

#[test]
fn label_propagation_and_spreading_match_sklearn() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden_data/semi_supervised.json");
    let cases: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    for c in cases.as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let p = &c["params"];
        let kernel = if p["kernel"] == "knn" {
            GraphKernel::Knn {
                n_neighbors: p["n_neighbors"].as_u64().unwrap() as usize,
            }
        } else {
            GraphKernel::Rbf {
                gamma: p["gamma"].as_f64().unwrap_or(20.0),
            }
        };
        let (x, y, xt) = (
            json_to_array2(&c["X"]),
            json_to_array1(&c["y"]),
            json_to_array2(&c["X_test"]),
        );
        let m = if c["variant"] == "propagation" {
            LabelPropagation {
                kernel,
                ..Default::default()
            }
            .fit(&x, &y)
        } else {
            LabelSpreading {
                kernel,
                alpha: p["alpha"].as_f64().unwrap(),
                ..Default::default()
            }
            .fit(&x, &y)
        }
        .unwrap();
        assert_eq!(m.n_iter() as u64, c["n_iter"].as_u64().unwrap(), "{name}");
        assert_array2_close(
            m.label_distributions(),
            &json_to_array2(&c["label_distributions"]),
            1e-10,
            name,
        );
        assert_eq!(
            m.transduction(),
            &json_to_array1(&c["transduction"]),
            "{name}"
        );
        assert_array2_close(
            &m.predict_proba(&xt).unwrap(),
            &json_to_array2(&c["predict_proba"]),
            1e-10,
            name,
        );
        assert_eq!(
            m.predict(&xt).unwrap(),
            json_to_array1(&c["predict"]),
            "{name}"
        );
    }
}
