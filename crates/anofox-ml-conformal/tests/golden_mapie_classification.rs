//! Parity with MAPIE 1.6.0's SplitConformalClassifier
//! (test_harness/generators/gen_conformal_classification.py). Both sides use
//! the same fixed softmax model, so only the conformal logic is compared.

use anofox_ml_conformal::{
    conformalize_classifier, conformalize_raps, ApsMode, ClassificationScore, RapsParams,
};
use anofox_ml_core::{Predict, PredictProba, Result};
use ndarray::{Array1, Array2, ArrayView2, Axis};
use serde_json::Value;

/// P(y = k | x) = softmax(x W)_k, classes 0..K.
struct FixedSoftmax {
    w: Array2<f64>,
}

impl PredictProba<f64> for FixedSoftmax {
    fn predict_proba_view(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        let mut z = x.dot(&self.w);
        for mut row in z.rows_mut() {
            let m = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            row.mapv_inplace(|v| (v - m).exp());
            let s: f64 = row.iter().sum();
            row.mapv_inplace(|v| v / s);
        }
        Ok(z)
    }
}

impl Predict<f64> for FixedSoftmax {
    fn predict_view(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        let p = self.predict_proba_view(x)?;
        Ok(p.rows()
            .into_iter()
            .map(|r| {
                let mut best = 0;
                for k in 1..r.len() {
                    if r[k] > r[best] {
                        best = k;
                    }
                }
                best as f64
            })
            .collect())
    }
}

fn matrix(v: &Value) -> Array2<f64> {
    let rows: Vec<Vec<f64>> = serde_json::from_value(v.clone()).unwrap();
    Array2::from_shape_fn((rows.len(), rows[0].len()), |(i, j)| rows[i][j])
}

#[test]
fn sets_match_mapie() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/golden_data/mapie_classification.json"
    );
    let g: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let w = matrix(&g["W"]);
    let (x_cal, x_test) = (matrix(&g["X_cal"]), matrix(&g["X_test"]));
    let y_cal: Array1<f64> = serde_json::from_value::<Vec<f64>>(g["y_cal"].clone())
        .unwrap()
        .into();
    let classes: Vec<f64> = (0..w.ncols()).map(|k| k as f64).collect();
    let cases = g["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 6);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let include = case["include_last_label"].as_bool();
        let mode = if include == Some(false) {
            ApsMode::Exclude
        } else {
            ApsMode::Include
        };
        let fitted = match case["score"].as_str().unwrap() {
            "raps" => {
                // MAPIE's tuning rows and lambda are pinned in the generator.
                let tune: Vec<usize> = serde_json::from_value(case["tune_index"].clone()).unwrap();
                let cal: Vec<usize> = (0..x_cal.nrows()).filter(|i| !tune.contains(i)).collect();
                let params = RapsParams::new(mode).with_lambda(case["lambda"].as_f64().unwrap());
                conformalize_raps(
                    FixedSoftmax { w: w.clone() },
                    classes.clone(),
                    x_cal.select(Axis(0), &cal).view(),
                    y_cal.select(Axis(0), &cal).view(),
                    x_cal.select(Axis(0), &tune).view(),
                    y_cal.select(Axis(0), &tune).view(),
                    params,
                    0,
                )
                .unwrap()
            }
            other => {
                let score = match other {
                    "lac" => ClassificationScore::Lac,
                    "aps" => ClassificationScore::Aps(mode),
                    _ => panic!("unknown case {name}"),
                };
                conformalize_classifier(
                    FixedSoftmax { w: w.clone() },
                    classes.clone(),
                    x_cal.view(),
                    y_cal.view(),
                    score,
                    0,
                )
                .unwrap()
            }
        };
        let y_pred: Vec<f64> = serde_json::from_value(case["y_pred"].clone()).unwrap();
        assert_eq!(fitted.predict(&x_test).unwrap().to_vec(), y_pred, "{name}");
        let levels: Vec<f64> = serde_json::from_value(case["levels"].clone()).unwrap();
        for (j, level) in levels.iter().enumerate() {
            let expected: Vec<Vec<u8>> = serde_json::from_value(case["sets"][j].clone()).unwrap();
            let sets = fitted.predict_set(x_test.view(), 1.0 - level).unwrap();
            for (i, row) in expected.iter().enumerate() {
                let got: Vec<u8> = sets.row(i).iter().map(|&b| b as u8).collect();
                assert_eq!(&got, row, "{name} @ {level}, row {i}");
            }
        }
    }
}
