//! Parity with MAPIE 1.6.0 (test_harness/generators/gen_conformal.py), with
//! ordinary least squares as the base estimator on both sides.

use anofox_ml_conformal::{conformalize, CrossConformalRegressor, CrossMethod, PointPrediction};
use anofox_ml_core::utils::CvSplit;
use anofox_ml_core::Fit;
use anofox_ml_regression::OlsRegressor;
use ndarray::{Array1, Array2, Axis};
use serde_json::Value;

/// Intervals at a given alpha.
type IntervalFn = Box<dyn Fn(f64) -> anofox_ml_core::Result<(Array1<f64>, Array1<f64>)>>;

fn golden() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden_data/mapie.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn matrix(v: &Value) -> Array2<f64> {
    let rows: Vec<Vec<f64>> = serde_json::from_value(v.clone()).unwrap();
    Array2::from_shape_fn((rows.len(), rows[0].len()), |(i, j)| rows[i][j])
}

fn vector(v: &Value) -> Array1<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| match x {
            Value::String(s) if s == "inf" => f64::INFINITY,
            Value::String(s) if s == "-inf" => f64::NEG_INFINITY,
            x => x.as_f64().unwrap(),
        })
        .collect()
}

fn indices(v: &Value) -> Vec<usize> {
    serde_json::from_value(v.clone()).unwrap()
}

fn assert_close(name: &str, got: &Array1<f64>, want: &Array1<f64>) {
    assert_eq!(got.len(), want.len(), "{name}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let ok = (g.is_infinite() && g == w) || (g - w).abs() <= 1e-9 * (1.0 + w.abs());
        assert!(ok, "{name}[{i}]: got {g}, MAPIE {w}");
    }
}

#[test]
fn matches_mapie() {
    let g = golden();
    let (x, y, x_test) = (matrix(&g["X"]), vector(&g["y"]), matrix(&g["X_test"]));
    let cases = g["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 9);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let levels: Vec<f64> = serde_json::from_value(case["levels"].clone()).unwrap();
        let error_levels: Vec<f64> = serde_json::from_value(case["error_levels"].clone()).unwrap();
        let (pred, interval): (Array1<f64>, IntervalFn) = match case["kind"].as_str().unwrap() {
            "split" => {
                let (train, cal) = (indices(&case["train"]), indices(&case["calibration"]));
                let model = OlsRegressor::new()
                    .fit(&x.select(Axis(0), &train), &y.select(Axis(0), &train))
                    .unwrap();
                let fitted = conformalize(
                    model,
                    x.select(Axis(0), &cal).view(),
                    y.select(Axis(0), &cal).view(),
                )
                .unwrap();
                let pred = anofox_ml_core::Predict::predict(&fitted, &x_test).unwrap();
                let xt = x_test.clone();
                (
                    pred,
                    Box::new(move |a| fitted.predict_interval(xt.view(), a)),
                )
            }
            _ => {
                let fold = indices(&case["fold"]);
                let n = fold.len();
                let k = fold.iter().max().unwrap() + 1;
                let splits: Vec<CvSplit> = (0..k)
                    .map(|f| {
                        let test: Vec<usize> = (0..n).filter(|&i| fold[i] == f).collect();
                        let train: Vec<usize> = (0..n).filter(|&i| fold[i] != f).collect();
                        (train, test)
                    })
                    .collect();
                let method = match case["method"].as_str().unwrap() {
                    "plus" => CrossMethod::Plus,
                    "minmax" => CrossMethod::MinMax,
                    _ => CrossMethod::Base,
                };
                let point = match case["point"].as_str().unwrap() {
                    "mean" => PointPrediction::Mean,
                    "median" => PointPrediction::Median,
                    _ => PointPrediction::Full,
                };
                let (xs, ys) = (x.slice(ndarray::s![..n, ..]), y.slice(ndarray::s![..n]));
                let fitted = CrossConformalRegressor::new(OlsRegressor::new())
                    .with_method(method)
                    .with_point_prediction(point)
                    .fit_with_splits(xs, ys, &splits)
                    .unwrap();
                let pred = anofox_ml_core::Predict::predict(&fitted, &x_test).unwrap();
                let xt = x_test.clone();
                (
                    pred,
                    Box::new(move |a| fitted.predict_interval(xt.view(), a)),
                )
            }
        };
        assert_close(&format!("{name} y_pred"), &pred, &vector(&case["y_pred"]));
        for (level, expected) in levels.iter().zip(case["intervals"].as_array().unwrap()) {
            let (lo, hi) = interval(1.0 - level).unwrap();
            assert_close(
                &format!("{name} lower@{level}"),
                &lo,
                &vector(&expected["lower"]),
            );
            assert_close(
                &format!("{name} upper@{level}"),
                &hi,
                &vector(&expected["upper"]),
            );
        }
        for level in error_levels {
            let e = interval(1.0 - level).unwrap_err().to_string();
            assert!(e.contains("too few"), "{name}@{level}: {e}");
        }
    }
}
