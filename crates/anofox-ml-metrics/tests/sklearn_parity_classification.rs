//! Classification metric parity with scikit-learn 1.8.0.
//!
//! Expected values were generated with sklearn 1.8.0 and are hard-coded.

use anofox_ml_metrics::*;
use approx::assert_abs_diff_eq;
use ndarray::{array, Array1, Array2};

const TOL: f64 = 1e-12;

fn mc() -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let yt = array![0., 1., 2., 2., 1., 0., 2., 1., 1., 0., 2., 2.];
    let yp = array![0., 2., 2., 1., 1., 0., 0., 1., 2., 0., 2., 1.];
    let w = array![1., 2., 0.5, 1., 3., 1., 2., 1., 0.5, 1., 1., 2.];
    (yt, yp, w)
}

fn bin() -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    let bt = array![0., 1., 1., 0., 1., 1., 0., 0., 1., 1.];
    let bp = array![0., 1., 0., 0., 1., 1., 1., 0., 0., 1.];
    let bw = array![1., 2., 1., 1., 0.5, 1., 3., 1., 1., 2.];
    let s = array![0.1, 0.8, 0.4, 0.3, 0.7, 0.9, 0.6, 0.2, 0.5, 0.65];
    (bt, bp, bw, s)
}

/// `rng.dirichlet([1,1,1], 12).round(6)` with the last column recomputed as
/// `1 - p0 - p1` (same float ops as numpy, so rows sum to exactly 1).
fn proba() -> Array2<f64> {
    let p01 = [
        [0.267518, 0.422158],
        [0.331257, 0.231883],
        [0.094132, 0.36368],
        [0.172397, 0.559287],
        [0.239061, 0.739954],
        [0.048003, 0.010756],
        [0.203723, 0.276042],
        [0.429128, 0.165458],
        [0.096675, 0.784488],
        [0.69447, 0.17703],
        [0.127788, 0.618915],
        [0.461549, 0.010418],
    ];
    let mut p = Array2::zeros((12, 3));
    for (i, r) in p01.iter().enumerate() {
        p[[i, 0]] = r[0];
        p[[i, 1]] = r[1];
        p[[i, 2]] = 1.0 - r[0] - r[1];
    }
    p
}

#[test]
fn confusion_matrix_weighted_and_labels() {
    let (yt, yp, w) = mc();
    let cm = confusion_matrix_with(
        &yt,
        &yp,
        &ClassificationOptions::new().with_sample_weight(&w),
    )
    .unwrap();
    let expected = array![[3.0, 0.0, 0.0], [0.0, 4.0, 2.5], [2.0, 3.0, 1.5]];
    assert_eq!(cm, expected);

    let labels = [2.0, 0.0];
    let cm = confusion_matrix_with(&yt, &yp, &ClassificationOptions::new().with_labels(&labels))
        .unwrap();
    assert_eq!(cm, array![[2.0, 1.0], [0.0, 3.0]]);
}

#[test]
fn confusion_matrix_exact_labels_no_epsilon() {
    // Labels closer than the old 1e-9 epsilon must stay distinct; -0.0 == 0.0.
    let yt = array![0.0, 1e-12, -0.0];
    let yp = array![0.0, 0.0, 0.0];
    let cm = confusion_matrix(&yt, &yp).unwrap();
    assert_eq!(cm, array![[2.0, 0.0], [1.0, 0.0]]);
    assert_abs_diff_eq!(accuracy_score(&yt, &yp).unwrap(), 2.0 / 3.0, epsilon = TOL);
    // NaN labels are rejected instead of panicking.
    assert!(confusion_matrix(&array![f64::NAN, 1.0], &array![1.0, 1.0]).is_err());
}

#[test]
fn accuracy_weighted() {
    let (yt, yp, w) = mc();
    assert_abs_diff_eq!(
        accuracy_score_with(&yt, &yp, Some(&w)).unwrap(),
        0.53125,
        epsilon = TOL
    );
}

#[test]
fn prf_weighted_averages() {
    let (yt, yp, w) = mc();
    let o = ClassificationOptions::new().with_sample_weight(&w);
    let cases = [
        (
            Average::Macro,
            [0.5154761904761904, 0.6153846153846154, 0.5427689594356261],
        ),
        (Average::Micro, [0.53125, 0.53125, 0.53125]),
        (
            Average::Weighted,
            [0.49698660714285714, 0.53125, 0.49743716931216925],
        ),
    ];
    for (avg, [p, r, f]) in cases {
        assert_abs_diff_eq!(
            precision_score_with(&yt, &yp, avg, &o).unwrap(),
            p,
            epsilon = TOL
        );
        assert_abs_diff_eq!(
            recall_score_with(&yt, &yp, avg, &o).unwrap(),
            r,
            epsilon = TOL
        );
        assert_abs_diff_eq!(f1_score_with(&yt, &yp, avg, &o).unwrap(), f, epsilon = TOL);
    }
    assert_abs_diff_eq!(
        fbeta_score_with(&yt, &yp, 2.0, Average::Macro, &o).unwrap(),
        0.5794711824123588,
        epsilon = TOL
    );
}

#[test]
fn prf_per_class_weighted() {
    let (yt, yp, w) = mc();
    let o = ClassificationOptions::new().with_sample_weight(&w);
    let s = precision_recall_fscore_support_per_class(&yt, &yp, 1.0, &o).unwrap();
    assert_eq!(s.labels, vec![0.0, 1.0, 2.0]);
    let exp_p = [0.6, 4.0 / 7.0, 0.375];
    let exp_r = [1.0, 4.0 / 6.5, 1.5 / 6.5];
    let exp_f = [0.75, 0.5925925925925926, 0.2857142857142857];
    for i in 0..3 {
        assert_abs_diff_eq!(s.precision[i], exp_p[i], epsilon = TOL);
        assert_abs_diff_eq!(s.recall[i], exp_r[i], epsilon = TOL);
        assert_abs_diff_eq!(s.fscore[i], exp_f[i], epsilon = 1e-12);
    }
    assert_eq!(s.support, array![3.0, 6.5, 6.5]);
    assert_eq!(precision_per_class(&yt, &yp, &o).unwrap(), s.precision);
    assert_eq!(recall_per_class(&yt, &yp, &o).unwrap(), s.recall);
    assert_eq!(f1_per_class(&yt, &yp, &o).unwrap(), s.fscore);
}

#[test]
fn prf_labels_and_zero_division() {
    // labels=[1, 3]: class 3 never occurs -> 0/0 for every score.
    let (yt, yp, _) = mc();
    let labels = [1.0, 3.0];
    let check = |zd: ZeroDivision, avg: Average, expected: f64| {
        let o = ClassificationOptions::new()
            .with_labels(&labels)
            .with_zero_division(zd);
        assert_abs_diff_eq!(
            precision_score_with(&yt, &yp, avg, &o).unwrap(),
            expected,
            epsilon = TOL
        );
        assert_abs_diff_eq!(
            recall_score_with(&yt, &yp, avg, &o).unwrap(),
            expected,
            epsilon = TOL
        );
        assert_abs_diff_eq!(
            f1_score_with(&yt, &yp, avg, &o).unwrap(),
            expected,
            epsilon = TOL
        );
    };
    check(ZeroDivision::Zero, Average::Macro, 0.25);
    check(ZeroDivision::One, Average::Macro, 0.75);
    check(ZeroDivision::NaN, Average::Macro, 0.5);
    check(ZeroDivision::Zero, Average::Micro, 0.5);
    check(ZeroDivision::Zero, Average::Weighted, 0.5);
}

#[test]
fn prf_binary_pos_label() {
    let (bt, bp, bw, _) = bin();
    let o = ClassificationOptions::new();
    let avg = Average::binary();
    assert_abs_diff_eq!(
        precision_score_with(&bt, &bp, avg, &o).unwrap(),
        0.8,
        epsilon = TOL
    );
    assert_abs_diff_eq!(
        recall_score_with(&bt, &bp, avg, &o).unwrap(),
        0.6666666666666666,
        epsilon = TOL
    );
    assert_abs_diff_eq!(
        f1_score_avg(&bt, &bp, avg).unwrap(),
        0.7272727272727273,
        epsilon = TOL
    );

    let o = ClassificationOptions::new().with_sample_weight(&bw);
    let avg = Average::Binary { pos_label: 0.0 };
    assert_abs_diff_eq!(
        precision_score_with(&bt, &bp, avg, &o).unwrap(),
        0.6,
        epsilon = TOL
    );
    assert_abs_diff_eq!(
        recall_score_with(&bt, &bp, avg, &o).unwrap(),
        0.5,
        epsilon = TOL
    );
    assert_abs_diff_eq!(
        f1_score_with(&bt, &bp, avg, &o).unwrap(),
        0.5454545454545454,
        epsilon = TOL
    );

    // Single class present, pos_label absent: 0/0 -> zero_division.
    let z = Array1::<f64>::zeros(4);
    assert_eq!(precision_score(&z, &z, Average::binary()).unwrap(), 0.0);
    let o1 = ClassificationOptions::new().with_zero_division(ZeroDivision::One);
    assert_eq!(f1_score_with(&z, &z, Average::binary(), &o1).unwrap(), 1.0);

    // Errors: multiclass target, or pos_label not among two present labels.
    let (yt, yp, _) = mc();
    assert!(precision_score(&yt, &yp, Average::binary()).is_err());
    assert!(precision_score(&bt, &bp, Average::Binary { pos_label: 2.0 }).is_err());
}

#[test]
fn roc_auc_binary_weighted() {
    let (bt, _, bw, s) = bin();
    assert_abs_diff_eq!(
        roc_auc_score_with(&bt, &s, Some(&bw)).unwrap(),
        0.8666666666666667,
        epsilon = TOL
    );
    assert_abs_diff_eq!(
        roc_auc_score_with(&bt, &s, None).unwrap(),
        0.9166666666666666,
        epsilon = TOL
    );
    // Greater label is positive: {1, 2} works like {0, 1}.
    let bt12 = bt.mapv(|v| v + 1.0);
    assert_abs_diff_eq!(
        roc_auc_score_with(&bt12, &s, None).unwrap(),
        0.9166666666666666,
        epsilon = TOL
    );
}

#[test]
fn roc_auc_multiclass_ovr_ovo() {
    let (yt, _, w) = mc();
    let p = proba();
    let cases = [
        (MultiClass::Ovr, Average::Macro, 0.6325066137566138),
        (MultiClass::Ovr, Average::Weighted, 0.6433531746031746),
        (MultiClass::Ovr, Average::Micro, 0.6597222222222222),
        (MultiClass::Ovo, Average::Macro, 0.6305555555555555),
        (MultiClass::Ovo, Average::Weighted, 0.6336805555555556),
    ];
    for (m, avg, e) in cases {
        let got = roc_auc_score_multiclass(&yt, &p, m, avg, None, None).unwrap();
        assert_abs_diff_eq!(got, e, epsilon = TOL);
    }
    let weighted = [
        (Average::Macro, 0.6734143049932525),
        (Average::Weighted, 0.7029352226720649),
        (Average::Micro, 0.70703125),
    ];
    for (avg, e) in weighted {
        let got = roc_auc_score_multiclass(&yt, &p, MultiClass::Ovr, avg, None, Some(&w)).unwrap();
        assert_abs_diff_eq!(got, e, epsilon = TOL);
    }
    let per = roc_auc_score_ovr_per_class(&yt, &p, None, None).unwrap();
    let exp = [0.5555555555555556, 0.65625, 0.6857142857142857];
    for i in 0..3 {
        assert_abs_diff_eq!(per[i], exp[i], epsilon = TOL);
    }
    // sklearn rejects: ovo + sample_weight, rows not summing to 1.
    assert!(
        roc_auc_score_multiclass(&yt, &p, MultiClass::Ovo, Average::Macro, None, Some(&w)).is_err()
    );
    let bad = p.mapv(|v| v * 0.5);
    assert!(
        roc_auc_score_multiclass(&yt, &bad, MultiClass::Ovr, Average::Macro, None, None).is_err()
    );
}

#[test]
fn log_loss_multiclass_and_weighted() {
    let (yt, _, w) = mc();
    let p = proba();
    assert_abs_diff_eq!(
        log_loss_multiclass(&yt, &p, None, None).unwrap(),
        1.0930928399575446,
        epsilon = TOL
    );
    assert_abs_diff_eq!(
        log_loss_multiclass(&yt, &p, None, Some(&w)).unwrap(),
        1.0028505526878067,
        epsilon = TOL
    );

    let (bt, _, bw, s) = bin();
    assert_abs_diff_eq!(
        log_loss_with(&bt, &s, None).unwrap(),
        0.4326869582222247,
        epsilon = TOL
    );
    assert_abs_diff_eq!(
        log_loss_with(&bt, &s, Some(&bw)).unwrap(),
        0.4914844475116929,
        epsilon = TOL
    );

    // Single class in y_true needs explicit labels.
    let y1 = array![1.0, 1.0, 1.0];
    let p1 = array![[0.2, 0.8], [0.3, 0.7], [0.1, 0.9]];
    assert!(log_loss_multiclass(&y1, &p1, None, None).is_err());
    assert_abs_diff_eq!(
        log_loss_multiclass(&y1, &p1, Some(&[0.0, 1.0]), None).unwrap(),
        0.22839300363692283,
        epsilon = TOL
    );

    // Clipping uses machine epsilon, like sklearn >= 1.5.
    let y01 = array![0.0, 1.0];
    assert_abs_diff_eq!(
        log_loss_with(&y01, &array![0.0, 1.0], None).unwrap(),
        2.2204460492503136e-16,
        epsilon = 1e-20
    );
    assert_abs_diff_eq!(
        log_loss_with(&y01, &array![1.0, 0.0], None).unwrap(),
        36.04365338911715,
        epsilon = 1e-10
    );
}

#[test]
fn brier_weighted_and_other_cm_metrics() {
    let (bt, _, bw, s) = bin();
    assert_abs_diff_eq!(
        brier_score_loss_with(&bt, &s, Some(&bw)).unwrap(),
        0.1637037037037037,
        epsilon = TOL
    );
    let (yt, yp, _) = mc();
    assert_abs_diff_eq!(
        balanced_accuracy_score(&yt, &yp).unwrap(),
        0.6333333333333333,
        epsilon = TOL
    );
    assert_abs_diff_eq!(cohen_kappa_score(&yt, &yp).unwrap(), 0.375, epsilon = TOL);
}
