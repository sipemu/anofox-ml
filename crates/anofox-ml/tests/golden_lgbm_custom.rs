//! Custom objectives (`LgbmRegressor::fit_custom`) against LightGBM 4.7.0
//! trained with a Python callable objective (#213).
//! Generator: `test_harness/generators/gen_lgbm_custom.py`.

mod common;

use anofox_ml::ensemble::{BoostingType, LgbmCustomObjective, LgbmRegressor};
use common::{json_to_array1, json_to_array2, load_golden_data};
use ndarray::{Array1, Array2, ArrayView2, ArrayViewMut2};
use serde_json::Value;

/// Pseudo-Huber loss against `y` (the generator's `pseudo_huber`).
struct PseudoHuber {
    y: Array1<f64>,
    delta: f64,
}

impl LgbmCustomObjective for PseudoHuber {
    fn gradients(
        &self,
        raw: ArrayView2<f64>,
        mut g: ArrayViewMut2<f64>,
        mut h: ArrayViewMut2<f64>,
    ) {
        for i in 0..raw.nrows() {
            let r = raw[[i, 0]] - self.y[i];
            let s = (1.0 + (r / self.delta).powi(2)).sqrt();
            g[[i, 0]] = r / s;
            h[[i, 0]] = 1.0 / s.powf(3.0);
        }
    }
}

/// FFORMA's softmax combination loss over per-method errors.
struct Fforma {
    errors: Array2<f64>,
}

impl LgbmCustomObjective for Fforma {
    fn n_outputs(&self) -> usize {
        self.errors.ncols()
    }

    fn gradients(
        &self,
        raw: ArrayView2<f64>,
        mut g: ArrayViewMut2<f64>,
        mut h: ArrayViewMut2<f64>,
    ) {
        let m = self.errors.ncols();
        for i in 0..raw.nrows() {
            let max = raw.row(i).fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let e: Vec<f64> = (0..m).map(|j| (raw[[i, j]] - max).exp()).collect();
            let sum: f64 = e.iter().sum();
            let p: Vec<f64> = e.iter().map(|v| v / sum).collect();
            let loss: f64 = (0..m).map(|j| p[j] * self.errors[[i, j]]).sum();
            for j in 0..m {
                let err = self.errors[[i, j]];
                let gj = p[j] * (err - loss);
                g[[i, j]] = gj;
                h[[i, j]] = (err * p[j] * (1.0 - p[j]) - gj * p[j]).max(1e-6);
            }
        }
    }
}

/// 2-D array with JSON `null` read as `NaN`.
fn array2_with_nan(val: &Value) -> Array2<f64> {
    let rows = val.as_array().unwrap();
    let p = rows[0].as_array().unwrap().len();
    let flat: Vec<f64> = rows
        .iter()
        .flat_map(|r| r.as_array().unwrap().iter())
        .map(|v| v.as_f64().unwrap_or(f64::NAN))
        .collect();
    Array2::from_shape_vec((rows.len(), p), flat).unwrap()
}

fn model(case: &Value) -> LgbmRegressor {
    let p = &case["params"];
    let mut m = LgbmRegressor::new()
        .with_n_estimators(case["rounds"].as_u64().unwrap() as usize)
        .with_num_leaves(p["num_leaves"].as_u64().unwrap() as usize)
        .with_learning_rate(p["learning_rate"].as_f64().unwrap())
        .with_min_child_samples(p["min_data_in_leaf"].as_u64().unwrap() as usize)
        .with_categorical_features(
            case["categorical"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect(),
        );
    if let Some(f) = p["bagging_fraction"].as_f64() {
        m = m
            .with_subsample(f)
            .with_subsample_freq(p["bagging_freq"].as_u64().unwrap() as usize);
    }
    if let Some(f) = p["feature_fraction"].as_f64() {
        m = m.with_colsample_bytree(f);
    }
    if let Some(s) = p["seed"].as_u64() {
        m = m.with_seed(s);
    }
    if p["boosting"] == "goss" {
        m = m.with_boosting_type(BoostingType::Goss);
    }
    m
}

#[test]
fn custom_objectives_match_lightgbm() {
    let mut failures = Vec::new();
    for case in load_golden_data("lgbm_custom.json") {
        let name = case["name"].as_str().unwrap();
        let x = array2_with_nan(&case["X"]);
        let x_test = array2_with_nan(&case["X_test"]);
        let side = &case["side"];
        let objective: Box<dyn LgbmCustomObjective> = if name.starts_with("fforma") {
            Box::new(Fforma {
                errors: json_to_array2(&side["errors"]),
            })
        } else {
            Box::new(PseudoHuber {
                y: json_to_array1(&side["y"]),
                delta: side["delta"].as_f64().unwrap(),
            })
        };
        let fitted = model(&case).fit_custom(&x, objective.as_ref()).unwrap();
        let k = case["outputs"].as_u64().unwrap() as usize;
        assert_eq!(fitted.n_outputs(), k, "{name}");
        assert_eq!(
            fitted.n_iterations() * k,
            case["n_trees"].as_u64().unwrap() as usize,
            "{name}"
        );
        let expected = json_to_array2(&case["raw"]);
        let raw = fitted.predict_raw(&x_test).unwrap();
        let err = raw
            .iter()
            .zip(&expected)
            .map(|(a, e)| (a - e).abs() / e.abs().max(1.0))
            .fold(0.0, f64::max);
        eprintln!("{name}: {err:e}");
        if err > 1e-10 {
            failures.push(format!("{name}: max error {err:e}"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
