//! sklearn 1.8 parity of Isomap with the shared neighbour index and
//! Dijkstra geodesics (#60).

use anofox_ml_core::FitUnsupervised;
use anofox_ml_manifold::Isomap;
use ndarray::Array2;

#[test]
fn isomap_matches_sklearn() {
    let x = Array2::from_shape_fn((60, 2), |(i, j)| {
        let (i, j) = (i as f64, j as f64);
        (i * 0.6180339887 * (j + 1.0) + i * i * 0.0137 * (j + 3.0)) % 1.0
    });
    let e = Isomap::new(2, 8).fit(&x).unwrap().embedding;
    // Isomap(n_neighbors=8, n_components=2).fit(X).embedding_[:6]
    let expected = [
        -0.016842877493726018,
        0.7016304955571466,
        0.26341680072586315,
        0.00825248103023137,
        -0.21888788664159425,
        -0.08970612469415314,
        -0.026182588758914164,
        0.40691139615677824,
        -0.5180923518371119,
        0.007476670983943201,
        -0.32265673307949955,
        0.22808865174650067,
    ];
    // Eigenvector signs are arbitrary: align each column first.
    for c in 0..2 {
        let sign = if (e[[0, c]] * expected[c]) < 0.0 {
            -1.0
        } else {
            1.0
        };
        for i in 0..6 {
            let got = sign * e[[i, c]];
            assert!((got - expected[2 * i + c]).abs() < 1e-9, "{i},{c}: {got}");
        }
    }
}
