//! LgbmRegressor above LightGBM's `bin_construct_sample_cnt` (200 000
//! rows), from `f64` and `f32` features, against LightGBM 4.7.0 (#214).
//! Generator: `test_harness/generators/gen_lgbm_large.py`, whose hashed
//! features are recomputed here.

mod common;

use anofox_ml::ensemble::{LgbmFitOptions, LgbmObjective, LgbmRegressor};
use common::load_golden_data;
use ndarray::{Array1, Array2};

const N: usize = 260_000;

fn hash(i: usize, j: u64) -> u64 {
    (i as u64 * 2_654_435_761 + j * 40_503 + 17) % (1 << 32)
}

/// The generator's `data()`.
fn data() -> (Array2<f64>, Array1<f64>) {
    let mut x = Array2::zeros((N, 4));
    let mut y = Array1::zeros(N);
    for i in 0..N {
        let h: Vec<u64> = (0..5).map(|j| hash(i, j)).collect();
        let x0 = ((h[0] >> 8) % 4096) as f64 / 256.0 - 8.0;
        let x1 = ((h[1] >> 4) % 1024) as f64 / 128.0 - 4.0;
        let cat = ((h[2] >> 12) % 30) as f64;
        let mut x3 = ((h[3] >> 6) % 512) as f64 / 64.0 - 4.0;
        if (h[3] >> 20) % 97 == 0 {
            x3 = f64::NAN;
        }
        let noise = ((h[4] >> 9) % 1000) as f64 / 1000.0 - 0.5;
        x.row_mut(i).assign(&ndarray::arr1(&[x0, x1, cat, x3]));
        let x3z = if x3.is_nan() { 0.0 } else { x3 };
        y[i] = 0.3 * x0 * x0 - 0.5 * x1 + (cat % 5.0 - 2.0) * 0.4 + 0.2 * x3z + noise;
    }
    (x, y)
}

#[test]
fn sampled_bins_match_lightgbm_from_f64_and_f32() {
    let (x, y) = data();
    let x32 = x.mapv(|v| v as f32);
    for case in load_golden_data("lgbm_large.json") {
        let name = case["name"].as_str().unwrap();
        let p = &case["params"];
        let mut m = LgbmRegressor::new()
            .with_n_estimators(p["n_estimators"].as_u64().unwrap() as usize)
            .with_learning_rate(p["learning_rate"].as_f64().unwrap())
            .with_num_leaves(p["num_leaves"].as_u64().unwrap() as usize);
        let mut target = y.clone();
        if name.contains("poisson") {
            m = m
                .with_objective(LgbmObjective::Poisson)
                .with_categorical_features(vec![2]);
            target = y.mapv(|v| (0.3 * v).exp());
        }
        if let Some(seed) = p["random_state"].as_u64() {
            m = m
                .with_seed(seed)
                .with_subsample(p["subsample"].as_f64().unwrap())
                .with_subsample_freq(p["subsample_freq"].as_u64().unwrap() as usize)
                .with_colsample_bytree(p["colsample_bytree"].as_f64().unwrap());
        }
        let rows: Vec<usize> = case["test_rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let x_test = x.select(ndarray::Axis(0), &rows);
        let expected: Vec<f64> = case["predictions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();

        let from_f64 = m
            .fit_with_eval(&x, &target, &LgbmFitOptions::default())
            .unwrap();
        let data32 = m.dataset(&x32).unwrap();
        let from_f32 = m
            .fit_dataset(&data32, &target, &LgbmFitOptions::default())
            .unwrap();
        let pred64 = anofox_ml::prelude::Predict::predict(&from_f64, &x_test).unwrap();
        let pred32 = from_f32.predict_f32(&x_test.mapv(|v| v as f32)).unwrap();
        assert_eq!(pred64, pred32, "{name}: f32 and f64 input differ");
        let err = pred64
            .iter()
            .zip(&expected)
            .map(|(a, e)| (a - e).abs() / e.abs().max(1.0))
            .fold(0.0, f64::max);
        eprintln!("{name}: {err:e}");
        assert!(err < 1e-10, "{name}: max error {err:e}");

        // Streamed in blocks: the same model.
        let sample_rows = m.bin_sample_rows(N);
        assert_eq!(sample_rows.len(), 200_000);
        let sample = x32.select(ndarray::Axis(0), &sample_rows);
        let mut builder = m.dataset_builder(&sample, N).unwrap();
        for start in (0..N).step_by(70_000) {
            let end = (start + 70_000).min(N);
            builder
                .push_rows(x32.slice(ndarray::s![start..end, ..]))
                .unwrap();
        }
        let streamed = m
            .fit_dataset(
                &builder.finish().unwrap(),
                &target,
                &LgbmFitOptions::default(),
            )
            .unwrap();
        assert_eq!(
            streamed.predict_f32(&x_test.mapv(|v| v as f32)).unwrap(),
            pred32,
            "{name}: streamed"
        );
    }
}
