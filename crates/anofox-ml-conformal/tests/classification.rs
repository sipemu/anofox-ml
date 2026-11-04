//! Coverage and behaviour of conformal classification sets.

use anofox_ml_conformal::{
    conformalize_classifier, ApsMode, ClassificationScore, SplitConformalClassifier,
};
use anofox_ml_core::{Fit, Predict, PredictProba, Result};
use anofox_ml_ensemble::RandomForestClassifier;
use ndarray::{Array1, Array2, ArrayView2};
use rand::{Rng, SeedableRng};

/// P(y = k | x) = softmax(x W)_k: the true model of the simulated data.
#[derive(Debug)]
struct Softmax {
    w: Array2<f64>,
}

impl PredictProba<f64> for Softmax {
    fn predict_proba_view(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        let mut z = x.dot(&self.w);
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
            .map(|r| (0..r.len()).fold(0, |b, k| if r[k] > r[b] { k } else { b }) as f64)
            .collect())
    }
}

fn w() -> Array2<f64> {
    ndarray::array![
        [2.0, -1.0, 0.0, -1.0],
        [0.0, 1.5, -1.5, 0.0],
        [-0.5, 0.0, 1.0, 1.0]
    ]
}

/// Features and labels drawn from the true model.
fn data(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let x = Array2::from_shape_fn((n, 3), |_| rng.gen_range(-2.0..2.0));
    let p = Softmax { w: w() }.predict_proba_view(x.view()).unwrap();
    let y = p
        .rows()
        .into_iter()
        .map(|r| {
            let u: f64 = rng.gen();
            let mut acc = 0.0;
            for (k, &v) in r.iter().enumerate() {
                acc += v;
                if u < acc {
                    return k as f64;
                }
            }
            (r.len() - 1) as f64
        })
        .collect();
    (x, y)
}

const CLASSES: [f64; 4] = [0.0, 1.0, 2.0, 3.0];

fn covered(sets: &Array2<bool>, y: &Array1<f64>) -> f64 {
    anofox_ml_metrics::set_coverage_score(y, sets, &CLASSES).unwrap()
}

/// Mean coverage and mean set size over `reps` calibration / test draws.
fn mean_coverage(score: ClassificationScore, alpha: f64, reps: u64) -> (f64, f64) {
    let (mut cov, mut size) = (0.0, 0.0);
    for r in 0..reps {
        let (xc, yc) = data(100, 2 * r);
        let (xt, yt) = data(400, 2 * r + 1);
        let fitted = conformalize_classifier(
            Softmax { w: w() },
            CLASSES.to_vec(),
            xc.view(),
            yc.view(),
            score,
            r,
        )
        .unwrap();
        let sets = fitted.predict_set(xt.view(), alpha).unwrap();
        cov += covered(&sets, &yt);
        size += anofox_ml_metrics::mean_set_size(&sets).unwrap();
    }
    (cov / reps as f64, size / reps as f64)
}

#[test]
fn sets_cover_at_the_nominal_level() {
    for alpha in [0.1, 0.3] {
        // LAC and APS (include) are exact up to 1/(n+1) = 0.0099 above.
        for score in [
            ClassificationScore::Lac,
            ClassificationScore::Aps(ApsMode::Include),
        ] {
            let (cov, _) = mean_coverage(score, alpha, 150);
            let excess = cov - (1.0 - alpha);
            assert!(excess > -0.01, "{score:?} alpha {alpha}: {cov}");
        }
        // Randomised APS: coverage 1 − α on average.
        let (cov, _) = mean_coverage(ClassificationScore::Aps(ApsMode::Randomized), alpha, 150);
        assert!(
            (cov - (1.0 - alpha)).abs() < 0.015,
            "randomized alpha {alpha}: {cov}"
        );
    }
    // LAC gives the smallest sets.
    let (_, lac) = mean_coverage(ClassificationScore::Lac, 0.1, 30);
    let (_, aps) = mean_coverage(ClassificationScore::Aps(ApsMode::Include), 0.1, 30);
    assert!(lac < aps, "LAC {lac} vs APS {aps}");
}

#[test]
fn aps_sets_grow_on_uncertain_inputs_and_exclude_is_a_subset() {
    let (xc, yc) = data(300, 7);
    let (xt, _) = data(500, 8);
    let fit = |mode| {
        conformalize_classifier(
            Softmax { w: w() },
            CLASSES.to_vec(),
            xc.view(),
            yc.view(),
            ClassificationScore::Aps(mode),
            0,
        )
        .unwrap()
    };
    let include = fit(ApsMode::Include).predict_set(xt.view(), 0.1).unwrap();
    let exclude = fit(ApsMode::Exclude).predict_set(xt.view(), 0.1).unwrap();
    assert!(include.iter().zip(exclude.iter()).all(|(&i, &e)| i || !e));
    // Larger sets where the top probability is lower.
    let top = Softmax { w: w() }
        .predict_proba_view(xt.view())
        .unwrap()
        .map_axis(ndarray::Axis(1), |r| r.iter().copied().fold(0.0, f64::max));
    let size = include.map_axis(ndarray::Axis(1), |r| {
        r.iter().filter(|&&b| b).count() as f64
    });
    let (mt, ms) = (top.mean().unwrap(), size.mean().unwrap());
    let cov: f64 = top
        .iter()
        .zip(&size)
        .map(|(a, b)| (a - mt) * (b - ms))
        .sum();
    assert!(cov < 0.0, "set size should fall as confidence rises");
}

#[test]
fn split_classifier_fits_a_real_model() {
    let (x, y) = data(600, 21);
    let (xt, yt) = data(1000, 22);
    let fitted = SplitConformalClassifier::new(RandomForestClassifier::new(30))
        .with_score(ClassificationScore::Aps(ApsMode::Randomized))
        .with_calibration_size(0.4)
        .fit(&x, &y)
        .unwrap();
    assert_eq!(fitted.classes(), &CLASSES);
    let sets = fitted.predict_set(xt.view(), 0.2).unwrap();
    let cov = covered(&sets, &yt);
    assert!((cov - 0.8).abs() < 0.06, "{cov}");
    // Same seed, same randomised sets on every call.
    assert_eq!(sets, fitted.predict_set(xt.view(), 0.2).unwrap());
    let labels = fitted.predict_set_labels(xt.view(), 0.2).unwrap();
    assert_eq!(labels.len(), 1000);
    assert!(labels[0].iter().all(|l| CLASSES.contains(l)));
}

#[test]
fn errors_and_full_sets() {
    let (x, y) = data(8, 31);
    let fit = |classes: Vec<f64>| {
        conformalize_classifier(
            Softmax { w: w() },
            classes,
            x.view(),
            y.view(),
            ClassificationScore::Lac,
            0,
        )
    };
    // 8 calibration rows cannot support alpha = 0.1.
    let fitted = fit(CLASSES.to_vec()).unwrap();
    assert!(fitted
        .predict_set(x.view(), 0.1)
        .unwrap_err()
        .to_string()
        .contains("at least 9"));
    let full = fitted
        .with_allow_full_sets(true)
        .predict_set(x.view(), 0.1)
        .unwrap();
    assert!(full.iter().all(|&b| b));
    // Labels outside the classes, and a class count that does not match.
    let threes = y.mapv(|_| 3.0);
    let e = conformalize_classifier(
        Softmax { w: w() },
        vec![0.0, 1.0, 2.0, 9.0],
        x.view(),
        threes.view(),
        ClassificationScore::Lac,
        0,
    )
    .unwrap_err();
    assert!(e.to_string().contains("not one of the classes"), "{e}");
    assert!(fit(vec![0.0, 1.0, 2.0]).is_err());
    assert!(fit(vec![0.0]).is_err());
}

#[test]
fn raps_covers_and_tunes_its_regularization() {
    use anofox_ml_conformal::RapsParams;
    let alpha = 0.1;
    for (mode, check) in [
        (ApsMode::Include, "at_least"),
        (ApsMode::Randomized, "about"),
    ] {
        let (cov, _) = mean_coverage(ClassificationScore::Raps(RapsParams::new(mode)), alpha, 150);
        match check {
            // 80 calibration rows after the 20 % tuning split.
            "at_least" => assert!(cov - (1.0 - alpha) > -0.01, "{mode:?}: {cov}"),
            _ => assert!((cov - (1.0 - alpha)).abs() < 0.02, "{mode:?}: {cov}"),
        }
    }
    let (xc, yc) = data(300, 41);
    let fitted = conformalize_classifier(
        Softmax { w: w() },
        CLASSES.to_vec(),
        xc.view(),
        yc.view(),
        ClassificationScore::Raps(RapsParams::new(ApsMode::Include)),
        0,
    )
    .unwrap();
    let (k, lambda) = fitted.raps_regularization(alpha).unwrap().unwrap();
    assert!((1..=4).contains(&k), "{k}");
    assert!([0.001, 0.01, 0.1, 0.2, 0.5].contains(&lambda), "{lambda}");
    // 300 rows, 20 % tuning: 240 calibration scores.
    assert_eq!(fitted.conformity_scores().len(), 240);
    // Fixed k_reg and lambda: no tuning split, all rows calibrate.
    let fixed = conformalize_classifier(
        Softmax { w: w() },
        CLASSES.to_vec(),
        xc.view(),
        yc.view(),
        ClassificationScore::Raps(
            RapsParams::new(ApsMode::Include)
                .with_k_reg(2)
                .with_lambda(0.2),
        ),
        0,
    )
    .unwrap();
    assert_eq!(fixed.conformity_scores().len(), 300);
    assert_eq!(fixed.raps_regularization(alpha).unwrap(), Some((2, 0.2)));
    assert_eq!(
        conformalize_classifier(
            Softmax { w: w() },
            CLASSES.to_vec(),
            xc.view(),
            yc.view(),
            ClassificationScore::Lac,
            0
        )
        .unwrap()
        .raps_regularization(alpha)
        .unwrap(),
        None
    );
}

#[test]
fn raps_sets_are_no_larger_than_aps_with_many_classes() {
    use anofox_ml_conformal::RapsParams;
    // 10 classes, two of them likely and eight with small tails.
    let w10 = Array2::from_shape_fn((3, 10), |(i, j)| {
        if j < 2 {
            [2.5, -2.5][j] * [1.0, 0.5, -0.5][i]
        } else {
            0.15 * ((i * 7 + j * 3) % 5) as f64 - 0.3
        }
    });
    let gen = |n: usize, seed: u64| {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let x = Array2::from_shape_fn((n, 3), |_| rng.gen_range(-2.0..2.0));
        let p = Softmax { w: w10.clone() }
            .predict_proba_view(x.view())
            .unwrap();
        let y: Array1<f64> = p
            .rows()
            .into_iter()
            .map(|r| {
                let u: f64 = rng.gen();
                let mut acc = 0.0;
                for (k, &v) in r.iter().enumerate() {
                    acc += v;
                    if u < acc {
                        return k as f64;
                    }
                }
                9.0
            })
            .collect();
        (x, y)
    };
    let classes: Vec<f64> = (0..10).map(f64::from).collect();
    let (mut aps, mut raps) = (0.0, 0.0);
    for r in 0..20 {
        let (xc, yc) = gen(500, 100 + 2 * r);
        let (xt, _) = gen(300, 101 + 2 * r);
        let size = |score| {
            let f = conformalize_classifier(
                Softmax { w: w10.clone() },
                classes.clone(),
                xc.view(),
                yc.view(),
                score,
                r,
            )
            .unwrap();
            anofox_ml_metrics::mean_set_size(&f.predict_set(xt.view(), 0.1).unwrap()).unwrap()
        };
        aps += size(ClassificationScore::Aps(ApsMode::Include));
        raps += size(ClassificationScore::Raps(RapsParams::new(ApsMode::Include)));
    }
    assert!(raps <= aps, "RAPS {} vs APS {}", raps / 20.0, aps / 20.0);
}
