//! sklearn 1.8 parity of OPTICS ξ extraction (#61): steep areas,
//! predecessor correction, cluster hierarchy and leaf labels.

use anofox_ml_cluster::{Optics, OpticsClusterMethod};
use anofox_ml_core::FitUnsupervised;
use ndarray::Array2;

fn data() -> Array2<f64> {
    let centres = [(0.0, 0.0), (4.0, 0.0), (2.0, 4.0), (8.0, 8.0)];
    let spread = [0.4, 0.6, 0.3, 1.2];
    Array2::from_shape_fn((120, 2), |(i, j)| {
        let fi = i as f64;
        let (c, s) = (centres[i % 4], spread[i % 4]);
        if j == 0 {
            c.0 + ((fi * 0.6180339887) % 1.0 - 0.5) * 2.0 * s
        } else {
            c.1 + ((fi * 0.4142135623 + fi * fi * 0.001) % 1.0 - 0.5) * 2.0 * s
        }
    })
}

#[test]
#[allow(clippy::type_complexity)]
fn xi_extraction_matches_sklearn() {
    // OPTICS(min_samples=5, **kw): labels_ and cluster_hierarchy_
    let cases: [(OpticsClusterMethod, &[i64], &[usize]); 3] = [
        (
            OpticsClusterMethod::Xi {
                xi: 0.05,
                min_cluster_size: None,
                predecessor_correction: true,
            },
            &[
                -1, -1, -1, 6, -1, 3, 4, 8, 1, -1, 5, 7, -1, -1, -1, -1, -1, 3, -1, 6, -1, -1, 4,
                -1, 1, -1, 5, 7, 0, 2, 5, 8, 1, -1, 5, 7, 0, 2, 5, 8, 1, -1, -1, -1, 0, 2, 5, 8, 1,
                -1, -1, 7, 0, 2, 5, -1, 1, -1, -1, 7, -1, -1, 5, -1, -1, 3, 4, -1, -1, 3, -1, 6,
                -1, -1, 5, 7, 1, -1, 5, -1, -1, 3, 4, -1, -1, 3, -1, 7, 0, -1, 4, 8, 1, 3, -1, 6,
                -1, 2, 5, 8, 1, 3, -1, 6, -1, 2, 5, 8, 1, 3, 4, 6, -1, 2, 5, 8, 1, 3, 4, 6,
            ],
            &[
                2, 6, 12, 22, 0, 29, 38, 44, 32, 47, 49, 58, 30, 59, 0, 59, 63, 69, 70, 82, 62, 84,
                60, 89, 0, 89, 94, 100, 103, 109, 94, 109, 110, 117, 90, 119, 0, 119,
            ],
        ),
        (
            OpticsClusterMethod::Xi {
                xi: 0.1,
                min_cluster_size: Some(8),
                predecessor_correction: true,
            },
            &[
                -1, -1, -1, -1, -1, 2, -1, 4, 0, 1, 3, -1, -1, -1, -1, -1, -1, 2, -1, -1, -1, -1,
                -1, -1, 0, 1, 3, -1, -1, 1, 3, 4, 0, 1, 3, -1, -1, 1, 3, 4, 0, 1, -1, -1, -1, 1, 3,
                4, 0, 1, -1, -1, -1, 1, 3, -1, 0, 1, -1, -1, -1, 1, 3, -1, -1, 2, -1, -1, -1, 2,
                -1, -1, -1, 1, 3, -1, 0, 1, 3, -1, -1, 2, -1, -1, -1, 2, -1, -1, -1, -1, -1, 4, 0,
                2, -1, -1, -1, 1, 3, 4, 0, 2, -1, -1, -1, 1, 3, 4, 0, 2, -1, -1, -1, 1, 3, 4, 0, 2,
                -1, -1,
            ],
            &[
                12, 22, 0, 29, 32, 47, 49, 58, 30, 59, 0, 59, 70, 82, 62, 84, 60, 89, 0, 89, 110,
                117, 90, 119, 0, 119,
            ],
        ),
        (
            OpticsClusterMethod::Xi {
                xi: 0.05,
                min_cluster_size: None,
                predecessor_correction: false,
            },
            &[
                -1, -1, -1, 6, -1, 3, 4, 8, 1, -1, 5, 7, -1, -1, -1, -1, -1, 3, -1, 6, -1, 3, 4,
                -1, 1, 2, 5, 7, 0, 2, 5, 8, 1, -1, 5, 7, 0, 2, 5, 8, 1, -1, -1, -1, 0, 2, 5, 8, 1,
                -1, -1, 7, 0, 2, 5, 8, 1, -1, -1, 7, -1, 2, 5, -1, -1, 3, 4, -1, -1, 3, -1, 6, 1,
                -1, 5, 7, 1, -1, 5, -1, -1, 3, 4, -1, -1, 3, -1, 7, 0, -1, 4, 8, 1, 3, -1, 6, -1,
                2, 5, 8, 1, 3, -1, 6, -1, 2, 5, 8, 1, 3, 4, 6, -1, 2, 5, 8, 1, 3, 4, 6,
            ],
            &[
                2, 6, 12, 23, 0, 29, 38, 46, 32, 47, 49, 59, 30, 59, 0, 59, 63, 69, 70, 82, 62, 84,
                60, 89, 0, 89, 94, 100, 103, 109, 94, 109, 110, 118, 90, 119, 0, 119,
            ],
        ),
    ];
    for (method, labels, hierarchy) in cases {
        let f = Optics::new(5, 0.0)
            .with_cluster_method(method)
            .fit(&data())
            .unwrap();
        let l: Vec<i64> = f.labels.iter().map(|&v| v as i64).collect();
        assert_eq!(l, labels, "{method:?}");
        let h: Vec<usize> = f.cluster_hierarchy.iter().flatten().copied().collect();
        assert_eq!(h, hierarchy, "{method:?}");
    }
}
