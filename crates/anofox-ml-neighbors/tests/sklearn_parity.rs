//! sklearn 1.8 parity of the shared neighbour index and the neighbour
//! estimators (#60), for brute force and the KD-tree, plus exact agreement
//! between the two algorithms on tie-heavy data.

use anofox_ml_core::{Fit, Predict};
use anofox_ml_neighbors::{
    DistanceMetric, KnnClassifier, NeighborAlgorithm, NeighborIndex, RadiusNeighborsClassifier,
    RadiusNeighborsRegressor, WeightFunction,
};
use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2};

fn data() -> (Array2<f64>, Array2<f64>) {
    let x = Array2::from_shape_fn((200, 3), |(i, j)| {
        let (i, j) = (i as f64, j as f64);
        (i * 0.6180339887 * (j + 1.0) + i * i * 0.0137 * (j + 3.0)) % 1.0
    });
    let q = Array2::from_shape_fn((4, 3), |(i, j)| {
        (((i + 1) as f64) * 0.4142135623 * (j as f64 + 2.0) + 0.05) % 1.0
    });
    (x, q)
}

fn check_knn(metric: DistanceMetric, algo: NeighborAlgorithm, idx: &[usize], dist: &[f64]) {
    let (x, q) = data();
    let index = NeighborIndex::new(&x, metric, algo, Some(4)).unwrap();
    let (d, i) = index.kneighbors(&q, 4).unwrap();
    assert_eq!(
        i.iter().copied().collect::<Vec<_>>(),
        idx,
        "{metric:?} {algo:?}"
    );
    for (a, b) in d.iter().zip(dist) {
        assert_abs_diff_eq!(*a, *b, epsilon = 1e-12);
    }
}

#[test]
fn kneighbors_match_sklearn() {
    let cases: [(DistanceMetric, &[usize], &[f64]); 4] = [
        (
            DistanceMetric::Euclidean,
            &[
                104, 66, 87, 197, 92, 97, 74, 100, 149, 171, 167, 125, 181, 169, 18, 128,
            ],
            &[
                0.07158741716213626,
                0.07550199825017166,
                0.08432798458914148,
                0.09855383197345292,
                0.024958973285281863,
                0.05313524002611327,
                0.10773230503418721,
                0.12882778799626163,
                0.019247737743163938,
                0.06093124836722241,
                0.09413894249747304,
                0.11007243258891292,
                0.05011033239594062,
                0.112581439165381,
                0.13507183573953233,
                0.14151464960144508,
            ],
        ),
        (
            DistanceMetric::Manhattan,
            &[
                104, 66, 87, 197, 92, 97, 74, 57, 149, 171, 167, 125, 181, 169, 18, 128,
            ],
            &[
                0.10581356230009631,
                0.10981356230003014,
                0.126560256099976,
                0.16214741729910875,
                0.035172875399974046,
                0.07182712459999685,
                0.1639728753999865,
                0.2125861993999516,
                0.033019715700128405,
                0.08924068689967113,
                0.1340406868997941,
                0.16764068689988632,
                0.07653657280032222,
                0.17090511520002893,
                0.2030240595999966,
                0.2147457507997852,
            ],
        ),
        (
            DistanceMetric::Chebyshev,
            &[
                66, 104, 188, 197, 92, 97, 164, 74, 149, 171, 167, 125, 181, 169, 83, 18,
            ],
            &[
                0.056583870400009784,
                0.06529229980001194,
                0.07168462239976581,
                0.07583135069993407,
                0.022672711199994033,
                0.04765734529998955,
                0.08501394200016099,
                0.09126091459999142,
                0.012882942500017425,
                0.046369306099915075,
                0.08570526089989583,
                0.09353278629999862,
                0.043538867300233264,
                0.09431527409997464,
                0.10954618949999961,
                0.1095816069999973,
            ],
        ),
        (
            DistanceMetric::Minkowski(3.0),
            &[
                104, 66, 87, 197, 92, 97, 74, 100, 149, 171, 167, 125, 181, 169, 18, 128,
            ],
            &[
                0.06675333315001632,
                0.06733322766083182,
                0.07827777752564656,
                0.08561488360575867,
                0.023343442033646043,
                0.04948784400005077,
                0.09744696899899927,
                0.11095920709821618,
                0.016176012996687052,
                0.054376594005320446,
                0.08806763092401823,
                0.0996355057249729,
                0.045633790164909926,
                0.10159136729580774,
                0.12123771934705033,
                0.13053881766144476,
            ],
        ),
    ];
    for (metric, idx, dist) in cases {
        for algo in [NeighborAlgorithm::KdTree, NeighborAlgorithm::BruteForce] {
            check_knn(metric, algo, idx, dist);
        }
    }
    check_knn(
        DistanceMetric::Cosine,
        NeighborAlgorithm::Auto,
        &[
            197, 188, 111, 35, 30, 100, 92, 134, 149, 171, 167, 101, 181, 83, 169, 151,
        ],
        &[
            0.0006710033793621006,
            0.0008891113139223306,
            0.0010893540941439017,
            0.001126074808138644,
            2.4020670896796048e-08,
            3.5094740041818895e-06,
            8.320166627218128e-05,
            0.00013049825574940055,
            5.2001858629524556e-05,
            0.001729729376049205,
            0.0017336830442932216,
            0.002444793991416394,
            0.00135572254770977,
            0.00466876605526878,
            0.00696045878072149,
            0.0073159203574985066,
        ],
    );
    // The KD-tree cannot serve cosine distances.
    let (x, _) = data();
    assert!(
        NeighborIndex::new(&x, DistanceMetric::Cosine, NeighborAlgorithm::KdTree, None).is_err()
    );
}

#[test]
fn self_and_radius_queries_match_sklearn() {
    let (x, q) = data();
    for algo in [NeighborAlgorithm::KdTree, NeighborAlgorithm::BruteForce] {
        let index = NeighborIndex::new(&x, DistanceMetric::Euclidean, algo, None).unwrap();
        let (_, i) = index.kneighbors_self(3).unwrap();
        let first: Vec<usize> = i.rows().into_iter().take(3).flatten().copied().collect();
        assert_eq!(first, [21, 190, 140, 158, 61, 27, 79, 22, 146]);

        let r = index.radius_neighbors(&q, 0.12).unwrap();
        let ids: Vec<Vec<usize>> = r.iter().map(|v| v.iter().map(|p| p.0).collect()).collect();
        assert_eq!(
            ids,
            [
                vec![104, 66, 87, 197, 188, 13],
                vec![92, 97, 74],
                vec![149, 171, 167, 125],
                vec![181, 169]
            ]
        );
    }
}

fn labels(x: &Array2<f64>) -> Array1<f64> {
    Array1::from_shape_fn(x.nrows(), |i| {
        let a = if x[[i, 0]] + x[[i, 1]] > 1.0 {
            1.0
        } else {
            0.0
        };
        a + if x[[i, 2]] > 0.7 { 1.0 } else { 0.0 }
    })
}

#[test]
fn estimators_match_sklearn() {
    let (x, q) = data();
    let y = labels(&x);

    // Distance weights; the last query is training row 5 (distance 0).
    let mut qz = Array2::zeros((5, 3));
    qz.slice_mut(ndarray::s![..4, ..]).assign(&q);
    qz.row_mut(4).assign(&x.row(5));
    let knn = KnnClassifier::new(7)
        .with_weights(WeightFunction::Distance)
        .fit(&x, &y)
        .unwrap();
    let expected = [
        0.10411712788383724,
        0.37784866173077125,
        0.5180342103853914,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.33506564971462993,
        0.6649343502853701,
        0.0,
        0.0,
        1.0,
        0.0,
    ];
    for (a, b) in knn.predict_proba(&qz).unwrap().iter().zip(expected) {
        assert_abs_diff_eq!(*a, b, epsilon = 1e-12);
    }

    let mut qo = Array2::from_elem((5, 3), 5.0);
    qo.slice_mut(ndarray::s![..4, ..]).assign(&q);
    let rc = RadiusNeighborsClassifier::new(0.15)
        .with_weights(WeightFunction::Distance)
        .with_outlier_label(Some(5.0))
        .fit(&x, &y)
        .unwrap();
    let expected = [
        0.09508857820798934,
        0.3450833955183281,
        0.5598280262736824,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.37214195623174856,
        0.6278580437682514,
        0.0,
        0.0,
        0.0,
        0.0,
    ];
    for (a, b) in rc.predict_proba(&qo).unwrap().iter().zip(expected) {
        assert_abs_diff_eq!(*a, b, epsilon = 1e-12);
    }
    assert_eq!(rc.predict(&qo).unwrap().to_vec(), [2.0, 1.0, 1.0, 1.0, 5.0]);
    // Without an outlier label a query without neighbours is an error.
    let strict = RadiusNeighborsClassifier::new(0.15).fit(&x, &y).unwrap();
    assert!(strict.predict(&qo).is_err());

    let yr = Array1::from_shape_fn(200, |i| x[[i, 0]] * 2.0 - x[[i, 2]]);
    let rr = RadiusNeighborsRegressor::new(0.15).fit(&x, &yr).unwrap();
    let expected = [
        0.9561566780250117,
        1.1559917665333235,
        0.9150997289002021,
        0.10094991343333604,
    ];
    for (a, b) in rr.predict(&q).unwrap().iter().zip(expected) {
        assert_abs_diff_eq!(*a, b, epsilon = 1e-12);
    }
    assert!(rr.predict(&qo).unwrap()[4].is_nan());
}

/// Coarse integer grid: many exact ties. KD-tree and brute force must agree
/// exactly (ties go to the lower index) for every Minkowski metric, and a
/// precomputed distance matrix must give the same neighbours.
#[test]
fn kdtree_matches_brute_force_with_ties() {
    let x = Array2::from_shape_fn((300, 2), |(i, j)| ((i * (3 + j * 4) + j) % 7) as f64);
    let q = Array2::from_shape_fn((40, 2), |(i, j)| {
        ((i * (2 + j) + 1) % 8) as f64 - 0.5 * j as f64
    });
    for metric in [
        DistanceMetric::Euclidean,
        DistanceMetric::Manhattan,
        DistanceMetric::Chebyshev,
        DistanceMetric::Minkowski(1.5),
    ] {
        let kd =
            NeighborIndex::with_leaf_size(&x, metric, NeighborAlgorithm::KdTree, None, 5).unwrap();
        let bf = NeighborIndex::new(&x, metric, NeighborAlgorithm::BruteForce, None).unwrap();
        for k in [1, 7, 50] {
            assert_eq!(
                kd.kneighbors(&q, k).unwrap(),
                bf.kneighbors(&q, k).unwrap(),
                "{metric:?} k={k}"
            );
            assert_eq!(
                kd.kneighbors_self(k).unwrap(),
                bf.kneighbors_self(k).unwrap()
            );
        }
        for r in [0.0, 1.0, 2.5] {
            assert_eq!(
                kd.radius_neighbors(&q, r).unwrap(),
                bf.radius_neighbors(&q, r).unwrap()
            );
            assert_eq!(kd.radius_neighbors_self(r), bf.radius_neighbors_self(r));
        }
    }

    // Precomputed distances.
    let bf = NeighborIndex::new(
        &x,
        DistanceMetric::Euclidean,
        NeighborAlgorithm::BruteForce,
        None,
    )
    .unwrap();
    let dist = |a: &Array2<f64>| {
        Array2::from_shape_fn((a.nrows(), x.nrows()), |(i, j)| {
            ((a[[i, 0]] - x[[j, 0]]).powi(2) + (a[[i, 1]] - x[[j, 1]]).powi(2)).sqrt()
        })
    };
    let pre = NeighborIndex::new(
        &dist(&x),
        DistanceMetric::Precomputed,
        NeighborAlgorithm::Auto,
        None,
    )
    .unwrap();
    assert!(!pre.uses_tree());
    let (dp, ip) = pre.kneighbors(&dist(&q), 7).unwrap();
    let (db, ib) = bf.kneighbors(&q, 7).unwrap();
    assert_eq!(ip, ib);
    for (a, b) in dp.iter().zip(&db) {
        assert_abs_diff_eq!(*a, *b, epsilon = 1e-12);
    }
    assert_eq!(
        pre.kneighbors_self(7).unwrap().1,
        bf.kneighbors_self(7).unwrap().1
    );
}

#[test]
fn auto_rule_and_validation() {
    let (x, _) = data();
    let auto = |m, k| {
        NeighborIndex::new(&x, m, NeighborAlgorithm::Auto, k)
            .unwrap()
            .uses_tree()
    };
    assert!(auto(DistanceMetric::Euclidean, Some(5)));
    assert!(auto(DistanceMetric::Minkowski(3.0), None));
    assert!(!auto(DistanceMetric::Cosine, Some(5)));
    assert!(!auto(DistanceMetric::Euclidean, Some(100)));
    let wide = Array2::from_shape_fn((50, 16), |(i, j)| ((i * j) % 5) as f64);
    assert!(!NeighborIndex::new(
        &wide,
        DistanceMetric::Euclidean,
        NeighborAlgorithm::Auto,
        Some(3)
    )
    .unwrap()
    .uses_tree());
    assert!(NeighborIndex::new(
        &x,
        DistanceMetric::Minkowski(0.5),
        NeighborAlgorithm::Auto,
        None
    )
    .is_err());
}
