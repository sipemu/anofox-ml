//! Peak memory of an LgbmRegressor fit from a row-major `f32` matrix
//! (#214): bin it once into an `LgbmDataset`, drop the raw matrix, train.
//!
//! `cargo run --release --example lgbm_memory -- [rows] [features]`
//! (default 5 000 000 × 60). Linux only (reads `VmHWM` / `VmRSS`).

use anofox_ml::ensemble::LgbmRegressor;
use ndarray::{Array1, Array2};
use std::time::Instant;

fn status(field: &str) -> f64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    s.lines()
        .find(|l| l.starts_with(field))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|kb| kb.parse::<f64>().ok())
        .map_or(f64::NAN, |kb| kb / 1024.0 / 1024.0)
}

fn report(stage: &str) {
    println!(
        "{stage:<28} rss {:6.2} GB   peak {:6.2} GB",
        status("VmRSS:"),
        status("VmHWM:")
    );
}

fn main() {
    let mut args = std::env::args().skip(1);
    let n: usize = args
        .next()
        .and_then(|a| a.parse().ok())
        .unwrap_or(5_000_000);
    let p: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(60);
    println!("{n} rows x {p} features, f32 row-major");
    report("start");

    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32
    };
    let x = Array2::from_shape_simple_fn((n, p), &mut next);
    let y = Array1::from_shape_fn(n, |i| {
        let r = x.row(i);
        f64::from(r[0] * r[0] + r[1] - 0.5 * r[2]) + f64::from(next()) * 0.1
    });
    let raw_gb = (n * p * 4) as f64 / 1e9 * 1e9 / 1024f64.powi(3);
    println!("raw matrix {raw_gb:.2} GB, binned {:.2} GB", raw_gb / 4.0);
    report("raw matrix built");

    let model = LgbmRegressor::new()
        .with_n_estimators(50)
        .with_num_leaves(31)
        .with_subsample(0.9)
        .with_subsample_freq(5)
        .with_colsample_bytree(0.9);
    let t = Instant::now();
    let data = model.dataset(&x).expect("dataset");
    println!("binning {:.2} s", t.elapsed().as_secs_f64());
    report("dataset built");
    drop(x);
    report("raw matrix dropped");

    let t = Instant::now();
    let fitted = model
        .fit_dataset(&data, &y, &Default::default())
        .expect("fit");
    println!("fit {:.2} s", t.elapsed().as_secs_f64());
    report("fitted");
    println!("{} trees", fitted.n_estimators());
}
