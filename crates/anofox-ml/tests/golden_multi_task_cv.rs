//! MultiTaskLassoCV / MultiTaskElasticNetCV against scikit-learn 1.8.0
//! (`test_harness/generators/gen_multi_task_cv.py`, solved to tol=1e-12).

mod common;

use anofox_ml_regression::MultiTaskElasticNetCV;
use common::{json_to_array1, json_to_array2, load_golden_file};

fn close(o: f64, e: f64, tol: f64, what: &str) {
    assert!((o - e).abs() <= tol * (1.0 + e.abs()), "{what}: {o} vs {e}");
}

#[test]
fn multi_task_cv_matches_scikit_learn() {
    let data = load_golden_file("multi_task_cv.json");
    let (x, y) = (json_to_array2(&data["X"]), json_to_array2(&data["Y"]));
    for case in data["cases"].as_array().unwrap() {
        let p = &case["params"];
        let mut m = if case["kind"] == "MultiTaskLassoCV" {
            MultiTaskElasticNetCV::lasso()
        } else {
            let l1 = &p["l1_ratio"];
            MultiTaskElasticNetCV {
                l1_ratios: match l1.as_array() {
                    Some(a) => a.iter().map(|v| v.as_f64().unwrap()).collect(),
                    None => vec![l1.as_f64().unwrap()],
                },
                ..Default::default()
            }
        };
        match p["alphas"].as_array() {
            Some(a) => m.alphas = Some(a.iter().map(|v| v.as_f64().unwrap()).collect()),
            None => m.n_alphas = p["alphas"].as_u64().unwrap() as usize,
        }
        m.cv = p["cv"].as_u64().unwrap() as usize;
        m.eps = p["eps"].as_f64().unwrap_or(1e-3);
        m.fit_intercept = p["fit_intercept"].as_bool().unwrap_or(true);
        let f = m.fit(&x, &y).unwrap();

        let alphas = json_to_array2(&case["alphas"]);
        for (o, e) in f.alphas.iter().zip(&alphas) {
            close(*o, *e, 1e-14, &format!("{p}: alphas"));
        }
        let mse = &case["mse_path"];
        for (l, per_alpha) in mse.as_array().unwrap().iter().enumerate() {
            for (a, folds) in per_alpha.as_array().unwrap().iter().enumerate() {
                for (k, e) in folds.as_array().unwrap().iter().enumerate() {
                    let e = e.as_f64().unwrap();
                    close(f.mse_path[[l, a, k]], e, 1e-8, &format!("{p}: mse_path"));
                }
            }
        }
        close(
            f.alpha,
            case["alpha"].as_f64().unwrap(),
            1e-14,
            &format!("{p}: alpha"),
        );
        close(
            f.l1_ratio,
            case["l1_ratio"].as_f64().unwrap(),
            0.0,
            &format!("{p}: l1"),
        );
        for (o, e) in f.fitted.coef.iter().zip(&json_to_array2(&case["coef"])) {
            close(*o, *e, 1e-8, &format!("{p}: coef"));
        }
        for (o, e) in f
            .fitted
            .intercept
            .iter()
            .zip(&json_to_array1(&case["intercept"]))
        {
            close(*o, *e, 1e-8, &format!("{p}: intercept"));
        }
        let pred = f.predict(&x.slice(ndarray::s![..3, ..])).unwrap();
        for (o, e) in pred.iter().zip(&json_to_array2(&case["predict"])) {
            close(*o, *e, 1e-8, &format!("{p}: predict"));
        }
    }
}

#[test]
fn multi_task_cv_rejects_bad_input() {
    let x = ndarray::Array2::<f64>::zeros((10, 2));
    let y = ndarray::Array2::<f64>::zeros((10, 2));
    let bad_cv = MultiTaskElasticNetCV {
        cv: 1,
        ..Default::default()
    };
    assert!(bad_cv.fit(&x, &y).is_err());
    let no_grid = MultiTaskElasticNetCV {
        l1_ratios: vec![0.0],
        ..Default::default()
    };
    assert!(no_grid.fit(&x, &y).is_err());
    assert!(MultiTaskElasticNetCV::lasso()
        .fit(&x, &y.slice(ndarray::s![..5, ..]).to_owned())
        .is_err());
}
