//! sklearn 1.8 parity of ClassicalMds (#61) with Euclidean features (via
//! the covariance), another metric and a precomputed dissimilarity matrix.

use anofox_ml_core::FitUnsupervised;
use anofox_ml_manifold::ClassicalMds;
use anofox_ml_neighbors::distance::compute_distance;
use anofox_ml_neighbors::DistanceMetric;
use ndarray::Array2;

fn data() -> Array2<f64> {
    Array2::from_shape_fn((25, 4), |(i, j)| {
        let (fi, fj) = (i as f64, j as f64);
        (fi * 0.6180339887 * (fj + 1.0) + fi * fi * 0.0137 * (fj + 3.0)) % 1.0
            + 2.0 * (i % 3) as f64
    })
}

fn check(f: &anofox_ml_manifold::FittedClassicalMds, emb: &[f64], eig: &[f64]) {
    let k = eig.len();
    for (a, b) in f.eigenvalues.iter().zip(eig) {
        assert!((a - b).abs() < 1e-9 * b.abs().max(1.0), "{a} vs {b}");
    }
    for i in 0..2 {
        for c in 0..k {
            let (a, b) = (f.embedding[[i, c]], emb[k * i + c]);
            assert!((a - b).abs() < 1e-9, "[{i},{c}] {a} vs {b}");
        }
    }
}

#[test]
fn classical_mds_matches_sklearn() {
    let x = data();
    // ClassicalMDS(n_components=3, metric="euclidean")
    let f = ClassicalMds::new(3).fit(&x).unwrap();
    check(
        &f,
        &[
            4.76871291950607,
            -0.053633750783197404,
            -0.004390579470129769,
            -0.44992818191269074,
            -0.3519449256806256,
            0.03910129391645373,
        ],
        &[294.8871599103329, 2.5799052458919802, 1.4872170256095114],
    );
    // metric="manhattan"
    let f = ClassicalMds::new(3)
        .with_metric(DistanceMetric::Manhattan)
        .fit(&x)
        .unwrap();
    check(
        &f,
        &[
            9.545956019756925,
            -0.009149927385919602,
            -0.02641498522796559,
            -0.895476841078843,
            -0.6717106707307992,
            0.028985452542407517,
        ],
        &[1174.6470571435755, 4.366535695290593, 2.96866424106556],
    );
    // metric="precomputed" on Chebyshev distances
    let d = Array2::from_shape_fn((25, 25), |(i, j)| {
        compute_distance(&x.row(i), &x.row(j), DistanceMetric::Chebyshev)
    });
    let f = ClassicalMds::new(2)
        .with_metric(DistanceMetric::Precomputed)
        .fit(&d)
        .unwrap();
    check(
        &f,
        &[
            -2.4767966862381,
            0.012846021328988533,
            0.26493872810837427,
            -0.6121412220894785,
        ],
        &[87.40066484274745, 6.31254671217256],
    );
}
