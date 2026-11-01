//! sklearn 1.8 comparison of full t-SNE runs (#61). The affinities and the
//! first optimiser steps match sklearn to rounding (unit test in
//! `tsne.rs`); after that t-SNE amplifies rounding differences chaotically,
//! so full runs are compared by iteration count and by the mean KL
//! divergence over several initialisations.

use anofox_ml_core::FitUnsupervised;
use anofox_ml_manifold::{TSne, TSneInit, TSneMethod};
use ndarray::Array2;

fn data() -> (Array2<f64>, Array2<f64>) {
    let x = Array2::from_shape_fn((40, 4), |(i, j)| {
        let (fi, fj) = (i as f64, j as f64);
        (fi * 0.6180339887 * (fj + 1.0) + fi * fi * 0.0137 * (fj + 3.0)) % 1.0
            + 3.0 * (i % 2) as f64
    });
    let y0 = Array2::from_shape_fn((40, 2), |(i, c)| {
        1e-4 * (((i as f64) * 0.7548776662 * (c as f64 + 1.0)) % 1.0 - 0.5)
    });
    (x, y0)
}

fn tsne(method: TSneMethod, max_iter: usize, init: TSneInit) -> TSne {
    TSne::new(2)
        .with_perplexity(8.0)
        .with_learning_rate(50.0)
        .with_method(method)
        .with_n_iter(max_iter)
        .with_init(init)
}

#[test]
fn iteration_counts_match_sklearn() {
    let (x, y0) = data();
    // TSNE(perplexity=8, learning_rate=50.0, init=Y0, method="exact", max_iter=m).n_iter_
    for (max_iter, n_iter) in [(300, 299), (600, 599)] {
        let f = tsne(TSneMethod::Exact, max_iter, TSneInit::Array(y0.clone()))
            .fit(&x)
            .unwrap();
        assert_eq!(f.n_iter, n_iter);
    }
    // max_iter = 250: sklearn's second phase runs no iteration and reports
    // n_iter_ = 250 with the initial (maximal) error.
    let f = tsne(TSneMethod::Exact, 250, TSneInit::Array(y0))
        .fit(&x)
        .unwrap();
    assert_eq!(f.n_iter, 250);
    assert_eq!(f.kl_divergence, f64::MAX);
}

#[test]
fn mean_kl_over_initialisations_matches_sklearn() {
    let (x, _) = data();
    let init = |s: usize| {
        Array2::from_shape_fn((40, 2), |(i, c)| {
            1e-4 * (((i as f64) * 0.7548776662 * (c as f64 + 1.0) + s as f64 * 0.31) % 1.0 - 0.5)
        })
    };
    // Mean KL of TSNE(perplexity=8, learning_rate=50.0, init=Y0(s), method, max_iter)
    // over s = 0..8 (std ≈ 0.011 for both).
    for (method, max_iter, sk_mean) in [
        (TSneMethod::Exact, 600, 0.11407818436792787),
        (TSneMethod::BarnesHut, 1000, 0.10839082207530737),
    ] {
        let mean = (0..8)
            .map(|s| {
                tsne(method, max_iter, TSneInit::Array(init(s)))
                    .fit(&x)
                    .unwrap()
                    .kl_divergence
            })
            .sum::<f64>()
            / 8.0;
        assert!(
            (mean - sk_mean).abs() < 0.015,
            "{method:?}: {mean} vs {sk_mean}"
        );
    }
}
