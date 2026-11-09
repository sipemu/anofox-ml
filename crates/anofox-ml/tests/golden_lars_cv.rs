//! LarsCV / LassoLarsCV against scikit-learn 1.8.0
//! (`test_harness/generators/gen_lars_cv.py`).

mod common;

use anofox_ml_core::{Fit, Predict};
use anofox_ml_regression::LarsCV;
use common::{json_to_array1, json_to_array2, load_golden_file};

fn close(o: f64, e: f64, tol: f64, what: &str) {
    assert!((o - e).abs() <= tol * (1.0 + e.abs()), "{what}: {o} vs {e}");
}

#[test]
fn lars_cv_matches_scikit_learn() {
    let data = load_golden_file("lars_cv.json");
    let (x, y) = (json_to_array2(&data["X"]), json_to_array1(&data["y"]));
    for case in data["cases"].as_array().unwrap() {
        let p = &case["params"];
        let mut m = if case["kind"] == "LassoLarsCV" {
            LarsCV::lasso()
        } else {
            LarsCV::default()
        };
        m.cv = p["cv"].as_u64().unwrap_or(5) as usize;
        m.fit_intercept = p["fit_intercept"].as_bool().unwrap_or(true);
        m.max_n_alphas = p["max_n_alphas"].as_u64().unwrap_or(1000) as usize;
        let f = m.fit(&x, &y).unwrap();
        let cv_alphas = json_to_array1(&case["cv_alphas"]);
        assert_eq!(f.cv_alphas.len(), cv_alphas.len(), "{p}: cv_alphas");
        for (o, e) in f.cv_alphas.iter().zip(&cv_alphas) {
            close(*o, *e, 1e-10, &format!("{p}: cv_alphas"));
        }
        for (o, e) in f.mse_path.iter().zip(&json_to_array2(&case["mse_path"])) {
            close(*o, *e, 1e-9, &format!("{p}: mse_path"));
        }
        close(
            f.alpha,
            case["alpha"].as_f64().unwrap(),
            1e-10,
            &format!("{p}: alpha"),
        );
        for (o, e) in f.fitted.alphas.iter().zip(&json_to_array1(&case["alphas"])) {
            close(*o, *e, 1e-10, &format!("{p}: alphas"));
        }
        for (o, e) in f.fitted.coef.iter().zip(&json_to_array1(&case["coef"])) {
            close(*o, *e, 1e-9, &format!("{p}: coef"));
        }
        close(
            f.fitted.intercept,
            case["intercept"].as_f64().unwrap(),
            1e-9,
            &format!("{p}: b"),
        );
        let pred = f.predict(&x.slice(ndarray::s![..5, ..])).unwrap();
        for (o, e) in pred.iter().zip(&json_to_array1(&case["predict"])) {
            close(*o, *e, 1e-9, &format!("{p}: predict"));
        }
    }
}
