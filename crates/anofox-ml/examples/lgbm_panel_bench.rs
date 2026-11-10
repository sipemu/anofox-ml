//! Panel-scale LgbmRegressor benchmark (#215): fit and predict time and
//! peak memory on `rows × 60` f32 features, as anofox-orchestration's
//! global model fits them (31 leaves, 200 iterations, bagging 0.9 every 5,
//! feature fraction 0.9). `benchmarks/lgbm_panel_benchmark.py` runs the
//! same shapes with LightGBM.
//!
//! `cargo run --release --example lgbm_panel_bench -- ROWS [cat] [poisson] [threads=N]`
//!
//! `cat`: three categorical features with ~10, ~100 and ~1000 levels.

use anofox_ml::ensemble::{LgbmObjective, LgbmRegressor};
use ndarray::{Array1, Array2};
use std::time::Instant;

fn peak_gb() -> f64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    s.lines()
        .find(|l| l.starts_with("VmHWM:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|kb| kb.parse::<f64>().ok())
        .map_or(f64::NAN, |kb| kb / 1024.0 / 1024.0)
}

/// xorshift64* uniforms and Box–Muller normals.
struct Rng(u64);

impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform().max(1e-300), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let n: usize = args
        .first()
        .and_then(|a| a.parse().ok())
        .unwrap_or(1_000_000);
    let cat = args.iter().any(|a| a == "cat");
    let poisson = args.iter().any(|a| a == "poisson");
    let threads: Option<usize> = args
        .iter()
        .find_map(|a| a.strip_prefix("threads=").and_then(|t| t.parse().ok()));
    let p = 60;

    let mut rng = Rng(42);
    let mut x = Array2::<f32>::zeros((n, p));
    for v in x.iter_mut() {
        *v = rng.normal() as f32;
    }
    let mut cats = Vec::new();
    if cat {
        for (j, k) in [(57, 10.0), (58, 100.0), (59, 1000.0)] {
            for i in 0..n {
                x[[i, j]] = (rng.uniform() * k).floor() as f32;
            }
            cats.push(j);
        }
    }
    let y = Array1::from_shape_fn(n, |i| {
        let r = x.row(i);
        let mut eta = (2.0 * f64::from(r[0])).sin()
            + f64::from(r[1] * r[2])
            + 0.5 * r.iter().skip(3).take(5).map(|&v| f64::from(v)).sum::<f64>();
        if cat {
            eta += f64::from(r[58]).sin() + f64::from(r[59] % 7.0) * 0.1;
        }
        let noise = 0.5 * rng.normal();
        if poisson {
            (0.3 * eta + 0.1 * noise).exp().round()
        } else {
            eta + noise
        }
    });

    let mut model = LgbmRegressor::new()
        .with_n_estimators(200)
        .with_num_leaves(31)
        .with_learning_rate(0.1)
        .with_subsample(0.9)
        .with_subsample_freq(5)
        .with_colsample_bytree(0.9)
        .with_categorical_features(cats);
    if poisson {
        model = model.with_objective(LgbmObjective::Poisson);
    }
    model.n_jobs = threads;

    let t = Instant::now();
    let data = model.dataset(&x).expect("dataset");
    let bin = t.elapsed().as_secs_f64();
    let fitted = model
        .fit_dataset(&data, &y, &Default::default())
        .expect("fit");
    let fit = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let pred = fitted.predict_f32(&x).expect("predict");
    let predict = t.elapsed().as_secs_f64();
    let mse = pred
        .iter()
        .zip(&y)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / n as f64;
    println!(
        "anofox n={n} cat={cat} poisson={poisson} threads={} fit {fit:.2}s (binning {bin:.2}s) \
         predict {predict:.2}s peak {:.2}GB mse {mse:.5}",
        threads.map_or("default".into(), |t| t.to_string()),
        peak_gb()
    );
}
