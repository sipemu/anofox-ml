//! sklearn 1.8 parity of spectral embedding / clustering (#61): k-NN
//! affinity `0.5·(C + Cᵀ)` with self loops, embedding divided by
//! `√degree` with the deterministic sign flip, k-means on the embedding.

use anofox_ml_cluster::{Affinity, SpectralClustering};
use anofox_ml_core::FitUnsupervised;
use anofox_ml_manifold::spectral_embedding;
use anofox_ml_neighbors::{DistanceMetric, NeighborAlgorithm, NeighborIndex};
use ndarray::Array2;

fn spiral(n: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, 2), |(i, j)| {
        let t = 1.0 + i as f64 * (3.0 * std::f64::consts::PI / n as f64);
        let jitter = |c: f64| (((i as f64) * c) % 1.0 - 0.5) * 0.3;
        if j == 0 {
            t * t.cos() + jitter(0.618)
        } else {
            t * t.sin() + jitter(0.414)
        }
    })
}

/// kneighbors_graph(X, 8, include_self=True) symmetrised.
fn knn_graph(x: &Array2<f64>) -> Vec<(usize, usize, f64)> {
    let index = NeighborIndex::new(
        x,
        DistanceMetric::Euclidean,
        NeighborAlgorithm::Auto,
        Some(8),
    )
    .unwrap();
    let (_, nb) = index.kneighbors(x, 8).unwrap();
    let mut e = Vec::new();
    for (i, row) in nb.rows().into_iter().enumerate() {
        for &j in row {
            e.push((i, j, 0.5));
            e.push((j, i, 0.5));
        }
    }
    e
}

#[test]
fn spectral_embedding_matches_sklearn() {
    // spectral_embedding(A, n_components=3, drop_first=False)
    let cases: [(usize, [f64; 6]); 2] = [
        (
            60,
            [
                0.048795003647425186,
                0.0892531782156828,
                0.019217415768626223,
                0.048795003647425186,
                0.08916927899036595,
                0.019158582396930947,
            ],
        ),
        (
            700,
            [
                0.014285714285789702,
                -0.018947130563201877,
                0.017466618461376336,
                0.014285714285789706,
                -0.018947443015667913,
                0.017467823197596873,
            ],
        ),
    ];
    for (n, expected) in cases {
        let x = spiral(n);
        let e = spectral_embedding(n, &knn_graph(&x), 3, false).unwrap();
        for i in 0..2 {
            for c in 0..3 {
                let (a, b) = (e[[i, c]], expected[3 * i + c]);
                assert!((a - b).abs() < 1e-8, "{n} [{i},{c}]: {a} vs {b}");
            }
        }
    }
}

#[test]
fn two_moons_are_separated_like_sklearn() {
    // SpectralClustering(2, affinity="nearest_neighbors", n_neighbors=8)
    // recovers the two moons exactly (ARI 1).
    for n in [60, 700] {
        let x = Array2::from_shape_fn((n, 2), |(i, j)| {
            let t = (i / 2) as f64 * (std::f64::consts::PI / (n / 2) as f64);
            let up = i % 2 == 0;
            let jit = |c: f64| (((i as f64) * c) % 1.0 - 0.5) * 0.1;
            match (j, up) {
                (0, true) => t.cos() + jit(0.618),
                (0, false) => 1.0 - t.cos() + jit(0.618),
                (_, true) => t.sin() + jit(0.414),
                (_, false) => 0.5 - t.sin() + jit(0.414),
            }
        });
        let f = SpectralClustering::new(2)
            .with_affinity(Affinity::KNearest(8))
            .fit(&x)
            .unwrap();
        for i in 0..n {
            assert_eq!(f.labels[i] == f.labels[0], i % 2 == 0, "{n}: sample {i}");
        }
    }
}
