//! sklearn 1.8 parity of MeanShift (#61): estimated bandwidth, flat-kernel
//! shifts, intensity-sorted centre suppression, bin seeding, cluster_all.

use anofox_ml_cluster::{estimate_bandwidth, MeanShift};
use anofox_ml_core::FitUnsupervised;
use ndarray::Array2;

fn data() -> Array2<f64> {
    Array2::from_shape_fn((90, 2), |(i, j)| {
        let (fi, fj) = (i as f64, j as f64);
        (fi * 0.6180339887 * (fj + 1.0) + fi * fi * 0.0137 * (fj + 3.0)) % 1.0
            + 3.0 * (i % 3) as f64
    })
}

fn labels(pattern: impl Fn(usize) -> f64) -> Vec<f64> {
    (0..90).map(pattern).collect()
}

#[test]
fn mean_shift_matches_sklearn() {
    let x = data();
    // estimate_bandwidth(X)
    let bw = estimate_bandwidth(&x, 0.3, None, 0).unwrap();
    assert!((bw - 0.7512068551407614).abs() < 1e-12, "{bw}");

    // MeanShift() and MeanShift(bin_seeding=True): n_iter_ 3, same centres.
    let centers = [
        6.55282981918334,
        6.439692971700009,
        3.4691291638166706,
        3.5049583276333443,
        0.4342951751166721,
        0.44649035023333716,
    ];
    let expected = labels(|i| [2.0, 1.0, 0.0][i % 3]);
    for ms in [
        MeanShift::auto(),
        MeanShift::auto().with_bin_seeding(true, 1),
    ] {
        let f = ms.fit(&x).unwrap();
        assert_eq!(f.n_iter, 3);
        for (a, b) in f.cluster_centers.iter().zip(centers) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
        assert_eq!(f.labels.to_vec(), expected);
    }

    // MeanShift(bandwidth=0.6, cluster_all=False)
    let f = MeanShift::new(0.6).with_cluster_all(false).fit(&x).unwrap();
    assert_eq!(f.n_iter, 5);
    let centers = [
        6.5696338406137995,
        6.4545918191586305,
        3.4520382575241415,
        3.517469618496562,
        0.44927087081035044,
        0.4618865692069005,
    ];
    for (a, b) in f.cluster_centers.iter().zip(centers) {
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
    }
    let expected = labels(|i| {
        if [0, 49, 71].contains(&i) {
            -1.0
        } else {
            [2.0, 1.0, 0.0][i % 3]
        }
    });
    assert_eq!(f.labels.to_vec(), expected);
}
