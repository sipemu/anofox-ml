//! MultiTaskLasso / MultiTaskElasticNet kinds (#113): fitted on `[X | Y]`
//! with `y_columns`, against scikit-learn 1.8.0 (`gen_multi_task.py`).

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
fn multi_task_kinds_match_scikit_learn() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../anofox-ml/tests/golden_data/multi_task.json");
    let data: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let (x, y) = (a2(&data["X"]), a2(&data["Y"]));
    let xy = ndarray::concatenate(Axis(1), &[x.view(), y.view()]).unwrap();
    for case in data["cases"].as_array().unwrap() {
        let kind = case["kind"].as_str().unwrap();
        let mut params = case["params"].clone();
        params["y_columns"] = json!(3);
        let mut m = make(kind, &params).unwrap();
        m.fit(xy.view(), None, None).unwrap();
        let m = load(kind, &m.to_bytes().unwrap()).unwrap();
        let (Out::A2(p), Out::A2(c)) = (
            m.predict(x.slice(ndarray::s![..3, ..])).unwrap(),
            m.coef().unwrap(),
        ) else {
            panic!()
        };
        for (o, e) in p
            .iter()
            .zip(&a2(&case["predict"]))
            .chain(c.iter().zip(&a2(&case["coef"])))
        {
            assert!((o - e).abs() < 1e-8, "{kind} {params}: {o} vs {e}");
        }
    }
}
