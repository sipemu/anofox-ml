//! FFORMA-style forecast combination with a custom multi-output LightGBM
//! objective (#213).
//!
//! Each row describes a time series (its features) and the errors that `M`
//! forecasting methods made on it. One booster with `M` raw scores learns
//! softmax weights per series that minimise the weighted error
//! `L = Σ_j softmax(score)_j · e_j`, the meta-learner of FFORMA
//! (Montero-Manso et al., 2020).

use anofox_ml::ensemble::{LgbmCustomObjective, LgbmRegressor};
use ndarray::{Array2, ArrayView2, ArrayViewMut2, Axis};

/// FFORMA's combination loss; the per-method errors are side data.
struct Fforma {
    errors: Array2<f64>,
}

fn softmax(scores: &[f64]) -> Vec<f64> {
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
    let sum: f64 = e.iter().sum();
    e.iter().map(|v| v / sum).collect()
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
        for (i, scores) in raw.axis_iter(Axis(0)).enumerate() {
            let p = softmax(scores.as_slice().unwrap());
            let e = self.errors.row(i);
            let loss: f64 = p.iter().zip(e).map(|(p, e)| p * e).sum();
            for j in 0..p.len() {
                let gj = p[j] * (e[j] - loss);
                g[[i, j]] = gj;
                h[[i, j]] = (e[j] * p[j] * (1.0 - p[j]) - gj * p[j]).max(1e-6);
            }
        }
    }
}

fn main() {
    // 600 series, 3 features; method 0 is best on trending series
    // (feature 0 high), method 2 on seasonal ones (feature 1 high).
    let n = 600;
    let x = Array2::from_shape_fn((n, 3), |(i, j)| {
        (((i * (7 + 5 * j) + 3 * j) % 101) as f64) / 50.0 - 1.0
    });
    let errors = Array2::from_shape_fn((n, 3), |(i, j)| {
        let (trend, season) = (x[[i, 0]], x[[i, 1]]);
        match j {
            0 => 1.0 - 0.6 * trend,
            1 => 1.0,
            _ => 1.0 - 0.6 * season,
        }
        .max(0.05)
    });

    let booster = LgbmRegressor::new()
        .with_n_estimators(200)
        .with_learning_rate(0.05)
        .with_num_leaves(15)
        .with_subsample(0.9)
        .with_subsample_freq(5)
        .with_colsample_bytree(0.9)
        .fit_custom(
            &x,
            &Fforma {
                errors: errors.clone(),
            },
        )
        .expect("fit");

    let raw = booster.predict_raw(&x).expect("predict");
    let (mut combined, mut best_single) = (0.0, f64::INFINITY);
    for (scores, e) in raw.axis_iter(Axis(0)).zip(errors.axis_iter(Axis(0))) {
        let w = softmax(scores.as_slice().unwrap());
        combined += w.iter().zip(e).map(|(w, e)| w * e).sum::<f64>() / n as f64;
    }
    for j in 0..errors.ncols() {
        best_single = best_single.min(errors.column(j).mean().unwrap());
    }
    println!(
        "{} iterations x {} outputs",
        booster.n_iterations(),
        booster.n_outputs()
    );
    println!("mean error, best single method: {best_single:.4}");
    println!("mean error, FFORMA combination:  {combined:.4}");
    assert!(combined < best_single);
}
