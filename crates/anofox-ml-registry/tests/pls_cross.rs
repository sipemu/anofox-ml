//! PlsCanonical / PlsSvd through the registry (#113): fitted on `[X | Y]`
//! with `y_columns`, as Cca, against scikit-learn 1.8.0
//! (`test_harness/generators/gen_pls.py`).

use anofox_ml_registry::{load, make, Out};
use ndarray::{Array1, Array2, Axis};
use serde_json::{json, Value};

fn a2(v: &Value) -> Array2<f64> {
    let rows: Vec<Array1<f64>> = v
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect()
        })
        .collect();
    let views: Vec<_> = rows.iter().map(|r| r.view()).collect();
    ndarray::stack(Axis(0), &views).unwrap()
}

#[test]
fn pls_cross_kinds_match_scikit_learn() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../anofox-ml/tests/golden_data/pls_cross.json");
    let data: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let (x, y, xq, yq) = (
        a2(&data["X"]),
        a2(&data["Y"]),
        a2(&data["Xq"]),
        a2(&data["Yq"]),
    );
    let xy = ndarray::concatenate(Axis(1), &[x.view(), y.view()]).unwrap();
    for (kind, name) in [("PlsCanonical", "PLSCanonical"), ("PlsSvd", "PLSSVD")] {
        let mut m = make(kind, &json!({"n_components": 2, "y_columns": 3})).unwrap();
        m.fit(xy.view(), None, None).unwrap();
        let m = load(kind, &m.to_bytes().unwrap()).unwrap();
        let (Out::A2(xs), Out::A2(ys)) = (
            m.transform(xq.view()).unwrap(),
            m.transform_y(yq.view()).unwrap(),
        ) else {
            panic!()
        };
        let e = &data["models"][name];
        for (ours, key) in [(xs, "x_scores"), (ys, "y_scores")] {
            for (o, v) in ours.iter().zip(&a2(&e[key])) {
                assert!((o - v).abs() < 1e-10, "{kind} {key}: {o} vs {v}");
            }
        }
    }
}
