//! Mondrian (group-conditional) conformal prediction. No reference
//! implementation (MAPIE 1.6.0 has none): checked by definition — one group
//! reproduces plain split conformal exactly — and by per-group coverage.

use anofox_ml_conformal::{
    conformalize, conformalize_classifier, conformalize_classifier_mondrian, conformalize_mondrian,
    ApsMode, ClassificationScore, MondrianConformalRegressor, RapsParams,
};
use anofox_ml_core::{Fit, Predict, PredictProba, Result};
use anofox_ml_regression::OlsRegressor;
use ndarray::{Array1, Array2, ArrayView2};
use rand::{Rng, SeedableRng};

fn normal(rng: &mut rand::rngs::StdRng) -> f64 {
    let (u1, u2): (f64, f64) = (rng.gen_range(1e-12..1.0), rng.gen());
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// y = 2 x0 − x1 + noise; group 0 has noise sd 0.2, group 1 sd 2.0.
fn grouped(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>, Vec<usize>) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let groups: Vec<usize> = (0..n).map(|i| i % 2).collect();
    let x = Array2::from_shape_fn((n, 2), |_| normal(&mut rng));
    let y = Array1::from_shape_fn(n, |i| {
        2.0 * x[[i, 0]] - x[[i, 1]] + [0.2, 2.0][groups[i]] * normal(&mut rng)
    });
    (x, y, groups)
}

fn group_coverage(
    y: &Array1<f64>,
    lo: &Array1<f64>,
    hi: &Array1<f64>,
    groups: &[usize],
    g: usize,
) -> f64 {
    let idx: Vec<usize> = (0..y.len()).filter(|&i| groups[i] == g).collect();
    idx.iter()
        .filter(|&&i| lo[i] <= y[i] && y[i] <= hi[i])
        .count() as f64
        / idx.len() as f64
}

#[test]
fn one_group_is_plain_split_conformal() {
    let (x, y, _) = grouped(200, 1);
    let (xt, _, _) = grouped(50, 2);
    let model = || {
        OlsRegressor::new()
            .fit(x.slice(ndarray::s![..100, ..]), y.slice(ndarray::s![..100]))
            .unwrap()
    };
    let (xc, yc) = (x.slice(ndarray::s![100.., ..]), y.slice(ndarray::s![100..]));
    let plain = conformalize(model(), xc, yc).unwrap();
    let mondrian = conformalize_mondrian(model(), xc, yc, &[7; 100]).unwrap();
    for alpha in [0.05, 0.1, 0.5] {
        assert_eq!(
            plain.predict_interval(xt.view(), alpha).unwrap(),
            mondrian
                .predict_interval(xt.view(), &[7; 50], alpha)
                .unwrap()
        );
    }
    assert_eq!(
        mondrian.group_sizes().into_iter().collect::<Vec<_>>(),
        vec![(7, 100)]
    );
}

#[test]
fn mondrian_covers_every_group_where_marginal_does_not() {
    let alpha = 0.1;
    let (mut plain, mut mondrian) = ([0.0; 2], [0.0; 2]);
    let reps = 100;
    for r in 0..reps {
        let (x, y, g) = grouped(400, 10 + 2 * r);
        let (xt, yt, gt) = grouped(400, 11 + 2 * r);
        let fitted = MondrianConformalRegressor::new(OlsRegressor::new())
            .with_calibration_size(0.5)
            .with_seed(r)
            .fit_with_groups(x.view(), y.view(), &g)
            .unwrap();
        let (lo, hi) = fitted.predict_interval(xt.view(), &gt, alpha).unwrap();
        let p = anofox_ml_conformal::SplitConformalRegressor::new(OlsRegressor::new())
            .with_calibration_size(0.5)
            .with_seed(r)
            .fit(&x, &y)
            .unwrap();
        let (plo, phi) = p.predict_interval(xt.view(), alpha).unwrap();
        for k in 0..2 {
            mondrian[k] += group_coverage(&yt, &lo, &hi, &gt, k) / reps as f64;
            plain[k] += group_coverage(&yt, &plo, &phi, &gt, k) / reps as f64;
        }
    }
    // Marginal calibration over-covers the quiet group and under-covers the
    // noisy one; Mondrian covers both (≈100 calibration rows per group).
    assert!(plain[0] > 0.97 && plain[1] < 0.86, "plain {plain:?}");
    for k in 0..2 {
        assert!(
            mondrian[k] - (1.0 - alpha) > -0.01 && mondrian[k] < 0.93,
            "group {k}: {mondrian:?}"
        );
    }
}

/// Softmax model on 3 features, 3 classes.
#[derive(Debug)]
struct Softmax;

impl PredictProba<f64> for Softmax {
    fn predict_proba_view(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        let w = ndarray::array![[2.0, -1.0, 0.0], [0.0, 1.5, -1.5], [-0.5, 0.0, 1.0]];
        let mut z = x.dot(&w);
        for mut row in z.rows_mut() {
            let m = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            row.mapv_inplace(|v| (v - m).exp());
            let s: f64 = row.iter().sum();
            row.mapv_inplace(|v| v / s);
        }
        Ok(z)
    }
}

impl Predict<f64> for Softmax {
    fn predict_view(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        let p = self.predict_proba_view(x)?;
        Ok(p.rows()
            .into_iter()
            .map(|r| (0..3).fold(0, |b, k| if r[k] > r[b] { k } else { b }) as f64)
            .collect())
    }
}

/// Labels from the model in group 0; in group 1, 40 % of labels are
/// replaced by a uniform draw (the model is overconfident there).
fn class_data(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>, Vec<usize>) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let x = Array2::from_shape_fn((n, 3), |_| rng.gen_range(-2.0..2.0));
    let p = Softmax.predict_proba_view(x.view()).unwrap();
    let groups: Vec<usize> = (0..n).map(|i| i % 2).collect();
    let y = (0..n)
        .map(|i| {
            if groups[i] == 1 && rng.gen::<f64>() < 0.4 {
                return rng.gen_range(0..3) as f64;
            }
            let u: f64 = rng.gen();
            let mut acc = 0.0;
            for k in 0..3 {
                acc += p[[i, k]];
                if u < acc {
                    return k as f64;
                }
            }
            2.0
        })
        .collect();
    (x, y, groups)
}

const CLASSES: [f64; 3] = [0.0, 1.0, 2.0];

fn set_coverage(sets: &Array2<bool>, y: &Array1<f64>, groups: &[usize], g: usize) -> f64 {
    let idx: Vec<usize> = (0..y.len()).filter(|&i| groups[i] == g).collect();
    idx.iter().filter(|&&i| sets[[i, y[i] as usize]]).count() as f64 / idx.len() as f64
}

#[test]
fn classification_one_group_is_plain_and_groups_are_covered() {
    let (xc, yc, _) = class_data(400, 5);
    let (xt, _, _) = class_data(400, 6);
    // One group: identical to the plain sets, for every score.
    for score in [
        ClassificationScore::Lac,
        ClassificationScore::Aps(ApsMode::Include),
        ClassificationScore::Aps(ApsMode::Randomized),
        ClassificationScore::Raps(RapsParams::new(ApsMode::Include)),
    ] {
        let plain =
            conformalize_classifier(Softmax, CLASSES.to_vec(), xc.view(), yc.view(), score, 3)
                .unwrap();
        let mondrian = conformalize_classifier_mondrian(
            Softmax,
            CLASSES.to_vec(),
            xc.view(),
            yc.view(),
            &[0; 400],
            score,
            3,
        )
        .unwrap();
        assert_eq!(
            plain.predict_set(xt.view(), 0.1).unwrap(),
            mondrian
                .predict_set_mondrian(xt.view(), &[0; 400], 0.1)
                .unwrap(),
            "{score:?}"
        );
    }
    // Two groups: mean per-group coverage over replications.
    let alpha = 0.1;
    let (mut plain, mut mondrian) = ([0.0; 2], [0.0; 2]);
    let reps = 60;
    for r in 0..reps {
        let (xc, yc, gc) = class_data(400, 100 + 2 * r);
        let (xt, yt, gt) = class_data(400, 101 + 2 * r);
        let p = conformalize_classifier(
            Softmax,
            CLASSES.to_vec(),
            xc.view(),
            yc.view(),
            ClassificationScore::Lac,
            r,
        )
        .unwrap()
        .predict_set(xt.view(), alpha)
        .unwrap();
        let m = conformalize_classifier_mondrian(
            Softmax,
            CLASSES.to_vec(),
            xc.view(),
            yc.view(),
            &gc,
            ClassificationScore::Lac,
            r,
        )
        .unwrap()
        .predict_set_mondrian(xt.view(), &gt, alpha)
        .unwrap();
        for k in 0..2 {
            plain[k] += set_coverage(&p, &yt, &gt, k) / reps as f64;
            mondrian[k] += set_coverage(&m, &yt, &gt, k) / reps as f64;
        }
    }
    assert!(
        plain[1] < 0.87,
        "plain under-covers the noisy group: {plain:?}"
    );
    for k in 0..2 {
        assert!(
            mondrian[k] - (1.0 - alpha) > -0.01,
            "group {k}: {mondrian:?}"
        );
    }
}

#[test]
fn mondrian_errors() {
    let (x, y, g) = grouped(60, 3);
    let model = OlsRegressor::new().fit(&x, &y).unwrap();
    let fitted = conformalize_mondrian(model, x.view(), y.view(), &g).unwrap();
    // Unseen group, wrong number of ids, too few rows in a group.
    assert!(fitted
        .predict_interval(x.view(), &vec![5; 60], 0.1)
        .unwrap_err()
        .to_string()
        .contains("group 5"));
    assert!(fitted.predict_interval(x.view(), &[0; 3], 0.1).is_err());
    let e = fitted
        .predict_interval(x.view(), &g, 0.01)
        .unwrap_err()
        .to_string();
    assert!(e.contains("group 0") && e.contains("too few"), "{e}");
    let (lo, _) = fitted
        .with_allow_infinite_bounds(true)
        .predict_interval(x.view(), &g, 0.01)
        .unwrap();
    assert!(lo.iter().all(|v| v.is_infinite()));
    // A classifier calibrated without groups cannot give Mondrian sets.
    let (xc, yc, _) = class_data(50, 4);
    let plain = conformalize_classifier(
        Softmax,
        CLASSES.to_vec(),
        xc.view(),
        yc.view(),
        ClassificationScore::Lac,
        0,
    )
    .unwrap();
    assert!(plain
        .predict_set_mondrian(xc.view(), &[0; 50], 0.1)
        .unwrap_err()
        .to_string()
        .contains("without groups"));
}

#[test]
fn classifier_fit_with_groups_one_group_is_plain_fit() {
    use anofox_ml_conformal::SplitConformalClassifier;
    use anofox_ml_trees::DecisionTreeClassifier;
    let (x, y, groups) = class_data(400, 9);
    let (xt, _, _) = class_data(100, 10);
    let conformal = SplitConformalClassifier::new(DecisionTreeClassifier::default())
        .with_score(ClassificationScore::Aps(ApsMode::Include))
        .with_seed(4);
    let plain = conformal.fit(&x, &y).unwrap();
    let one = conformal
        .fit_with_groups(x.view(), y.view(), &[3; 400])
        .unwrap();
    assert_eq!(
        plain.predict_set(xt.view(), 0.2).unwrap(),
        one.predict_set_mondrian(xt.view(), &[3; 100], 0.2).unwrap()
    );
    // Two groups calibrate separately; an unseen group is an error.
    let two = conformal
        .fit_with_groups(x.view(), y.view(), &groups)
        .unwrap();
    let tg: Vec<usize> = (0..100).map(|i| i % 2).collect();
    assert_eq!(
        two.predict_set_mondrian(xt.view(), &tg, 0.2).unwrap().dim(),
        (100, 3)
    );
    assert!(two.predict_set_mondrian(xt.view(), &[5; 100], 0.2).is_err());
    assert!(conformal
        .fit_with_groups(x.view(), y.view(), &[0; 10])
        .is_err());
}
