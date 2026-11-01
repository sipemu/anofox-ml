//! sklearn 1.8 parity of LLE (#61): `reg · trace` regularisation and the
//! bottom eigenvectors of the sparse `(I − W)ᵀ(I − W)`, from the dense
//! solver (n = 100) and sparse shift-invert Lanczos (n = 800).

use anofox_ml_core::FitUnsupervised;
use anofox_ml_manifold::LocallyLinearEmbedding;
use ndarray::Array2;

fn swiss_roll(n: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, 3), |(i, j)| {
        let t = i as f64 * (3.0 * std::f64::consts::PI / n as f64);
        match j {
            0 => t * t.cos(),
            1 => t * t.sin(),
            _ => ((i as f64) * 0.6180339887) % 1.0 * 2.0,
        }
    })
}

#[test]
fn lle_matches_sklearn() {
    // LocallyLinearEmbedding(n_neighbors=10, n_components=2, eigen_solver="dense")
    let cases: [(usize, f64, [f64; 6]); 2] = [
        (
            100,
            4.410605310983665e-06,
            [
                -0.2524289690347846,
                0.12927805588879704,
                -0.27088847420014456,
                0.17860591628556627,
                -0.2570920449191203,
                0.13807777786864767,
            ],
        ),
        (
            800,
            2.4781974669494356e-07,
            [
                0.04196310389614063,
                0.04953047489320637,
                0.041841202576248696,
                0.05888004816747721,
                0.041870790062742136,
                0.052842010224232586,
            ],
        ),
    ];
    for (n, err, emb) in cases {
        let f = LocallyLinearEmbedding::new(2, 10)
            .fit(&swiss_roll(n))
            .unwrap();
        assert!(
            (f.reconstruction_error - err).abs() < 1e-6 * err.max(1e-6),
            "{n}: {}",
            f.reconstruction_error
        );
        for c in 0..2 {
            // Eigenvector signs are arbitrary.
            let sign = if f.embedding[[0, c]] * emb[c] < 0.0 {
                -1.0
            } else {
                1.0
            };
            for i in 0..3 {
                let got = sign * f.embedding[[i, c]];
                assert!((got - emb[2 * i + c]).abs() < 1e-6, "{n} {i},{c}: {got}");
            }
        }
    }
}
