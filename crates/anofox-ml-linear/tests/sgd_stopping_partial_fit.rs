//! Regression tests for issue #31: SGD stopping rule, incremental
//! `partial_fit`, stable log loss, single-model binary classification,
//! truncated-gradient L1, averaging and `predict_proba`.

use anofox_ml_core::{Fit, PartialFit, Predict, PredictProba};
use anofox_ml_linear::{
    ClassifierLoss, LearningRate, RegressorLoss, SgdClassifier, SgdPenalty, SgdRegressor,
};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// y = 1.5 + 2 x0 - x1 + 0.5 x2 + noise, x ~ U(-1, 1).
fn noisy_regression(n: usize, noise: f64, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let x = Array2::from_shape_fn((n, 3), |_| rng.gen_range(-1.0..1.0));
    let y = Array1::from_shape_fn(n, |i| {
        // Sum of uniforms ~ approx. normal noise.
        let e: f64 = (0..4).map(|_| rng.gen_range(-1.0..1.0)).sum::<f64>() * 0.5;
        1.5 + 2.0 * x[[i, 0]] - x[[i, 1]] + 0.5 * x[[i, 2]] + noise * e
    });
    (x, y)
}

/// Two Gaussian-ish blobs in 2-D, labels {0, 1}.
fn blobs(n: usize, k: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let centers = [(-2.0, -2.0), (2.0, 2.0), (-2.0, 2.5), (2.5, -2.0)];
    let mut x = Array2::zeros((n, 2));
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let c = i % k;
        x[[i, 0]] = centers[c].0 + rng.gen_range(-1.0..1.0);
        x[[i, 1]] = centers[c].1 + rng.gen_range(-1.0..1.0);
        y[i] = c as f64;
    }
    (x, y)
}

fn accuracy(pred: &Array1<f64>, y: &Array1<f64>) -> f64 {
    pred.iter().zip(y).filter(|(a, b)| a == b).count() as f64 / y.len() as f64
}

#[test]
fn regressor_default_tol_stops_early_on_noisy_data() {
    let (x, y) = noisy_regression(300, 1.0, 1);
    let fitted = SgdRegressor::new().with_seed(3).fit(&x, &y).unwrap();
    assert!(
        fitted.n_iter() < 100,
        "noisy data should stop well before max_iter=1000, n_iter={}",
        fitted.n_iter()
    );
    assert!(
        fitted.n_iter() >= 6,
        "needs n_iter_no_change+1 epochs at least"
    );
    assert_eq!(fitted.t(), 1 + fitted.n_iter() * x.nrows());
    // Still learns the trend.
    assert!(
        (fitted.weights()[0] - 2.0).abs() < 0.5,
        "{:?}",
        fitted.weights()
    );
}

#[test]
fn regressor_tol_none_runs_all_epochs() {
    let (x, y) = noisy_regression(50, 1.0, 2);
    let fitted = SgdRegressor::new()
        .with_tol(None)
        .with_max_iter(37)
        .fit(&x, &y)
        .unwrap();
    assert_eq!(fitted.n_iter(), 37);
    assert_eq!(fitted.t(), 1 + 37 * 50);
}

#[test]
fn regressor_n_iter_no_change_controls_patience() {
    let (x, y) = noisy_regression(300, 1.0, 1);
    let short = SgdRegressor::new()
        .with_n_iter_no_change(1)
        .fit(&x, &y)
        .unwrap();
    let long = SgdRegressor::new()
        .with_n_iter_no_change(20)
        .fit(&x, &y)
        .unwrap();
    assert!(short.n_iter() < long.n_iter());
}

#[test]
fn regressor_early_stopping_uses_validation_split() {
    let (x, y) = noisy_regression(400, 1.0, 4);
    let fitted = SgdRegressor::new()
        .with_early_stopping(true)
        .with_validation_fraction(0.2)
        .fit(&x, &y)
        .unwrap();
    assert!(fitted.n_iter() < 1000);
    // Trains on the 80 % split only.
    assert_eq!(fitted.t(), 1 + fitted.n_iter() * 320);

    // Too little data for a validation set -> error, not silent fallback.
    let (xs, ys) = noisy_regression(3, 1.0, 4);
    assert!(SgdRegressor::new()
        .with_early_stopping(true)
        .fit(&xs, &ys)
        .is_err());
}

#[test]
fn regressor_partial_fit_continues_lr_schedule_and_matches_fit() {
    let (x, y) = noisy_regression(40, 0.3, 5);
    for average in [0usize, 1, 30] {
        let base = SgdRegressor::new()
            .with_shuffle(false)
            .with_learning_rate(LearningRate::InvScaling)
            .with_penalty(SgdPenalty::ElasticNet)
            .with_average(average);
        let epochs = 12;
        let full = base
            .clone()
            .with_tol(None)
            .with_max_iter(epochs)
            .fit(&x, &y)
            .unwrap();

        let mut state = None;
        let mut last_t = 1;
        for _ in 0..epochs {
            let s = base.partial_fit(state, &x, &y, None).unwrap();
            assert_eq!(s.n_iter(), 1, "partial_fit must run exactly one epoch");
            assert_eq!(s.t(), last_t + x.nrows(), "t_ must persist and grow");
            last_t = s.t();
            state = Some(s);
        }
        let inc = state.unwrap();
        assert_eq!(inc.t(), full.t());
        for (a, b) in inc.weights().iter().zip(full.weights()) {
            assert!((a - b).abs() < 1e-12, "average={average}: {a} vs {b}");
        }
        assert!((inc.bias() - full.bias()).abs() < 1e-12);
    }
}

#[test]
fn regressor_partial_fit_rejects_feature_change() {
    let (x, y) = noisy_regression(20, 0.1, 6);
    let reg = SgdRegressor::new();
    let s = reg.partial_fit(None, &x, &y, None).unwrap();
    let x2 = Array2::zeros((20, 2));
    assert!(reg.partial_fit(Some(s), &x2, &y, None).is_err());
}

#[test]
fn regressor_l1_gives_exact_zeros() {
    // Two informative features plus three pure-noise features.
    let mut rng = StdRng::seed_from_u64(7);
    let n = 400;
    let x = Array2::from_shape_fn((n, 5), |_| rng.gen_range(-1.0..1.0));
    let y = Array1::from_shape_fn(n, |i| {
        3.0 * x[[i, 0]] - 2.0 * x[[i, 1]] + 0.1 * rng.gen_range(-1.0..1.0)
    });
    let fitted = SgdRegressor::new()
        .with_penalty(SgdPenalty::L1)
        .with_alpha(0.05)
        .fit(&x, &y)
        .unwrap();
    let w = fitted.weights();
    assert!(w[0] > 1.0 && w[1] < -1.0, "{w:?}");
    let zeros = w.iter().filter(|v| **v == 0.0).count();
    assert!(
        zeros >= 2,
        "truncated gradient should zero noise features: {w:?}"
    );
}

#[test]
fn regressor_average_smooths_iterates() {
    let (x, y) = noisy_regression(200, 1.0, 8);
    let plain = SgdRegressor::new()
        .with_learning_rate(LearningRate::Constant)
        .with_eta0(0.05)
        .fit(&x, &y)
        .unwrap();
    let avg = SgdRegressor::new()
        .with_learning_rate(LearningRate::Constant)
        .with_eta0(0.05)
        .with_average(1)
        .fit(&x, &y)
        .unwrap();
    let err = |w: &Array1<f64>| {
        ((w[0] - 2.0).powi(2) + (w[1] + 1.0).powi(2) + (w[2] - 0.5).powi(2)).sqrt()
    };
    assert_ne!(plain.weights(), avg.weights());
    assert!(
        err(avg.weights()) < 0.5,
        "averaged weights {:?}",
        avg.weights()
    );
}

#[test]
fn regressor_huber_and_epsilon_losses_still_stop() {
    let (x, y) = noisy_regression(300, 1.0, 9);
    for loss in [RegressorLoss::Huber, RegressorLoss::EpsilonInsensitive] {
        let f = SgdRegressor::new().with_loss(loss).fit(&x, &y).unwrap();
        assert!(f.n_iter() < 1000, "{loss:?} n_iter={}", f.n_iter());
        assert!(f.predict(&x).unwrap().iter().all(|v| v.is_finite()));
    }
}

#[test]
fn classifier_default_tol_stops_early() {
    let (x, y) = blobs(200, 2, 10);
    let f = SgdClassifier::new().fit(&x, &y).unwrap();
    assert!(f.n_iter() < 1000, "n_iter={}", f.n_iter());
}

#[test]
fn classifier_binary_trains_single_model() {
    let (x, y) = blobs(200, 2, 11);
    for loss in [
        ClassifierLoss::Hinge,
        ClassifierLoss::Log,
        ClassifierLoss::ModifiedHuber,
        ClassifierLoss::Perceptron,
    ] {
        let f = SgdClassifier::new().with_loss(loss).fit(&x, &y).unwrap();
        assert_eq!(f.weights().len(), 1, "{loss:?}");
        assert_eq!(f.biases().len(), 1);
        assert_eq!(f.decision_function(&x).unwrap().ncols(), 1);
        let acc = accuracy(&f.predict(&x).unwrap(), &y);
        assert!(acc >= 0.97, "{loss:?} accuracy {acc}");
    }
    // Labels other than {0, 1}: positive class is classes[1].
    let y2 = y.mapv(|v| if v == 1.0 { 7.0 } else { -3.0 });
    let f = SgdClassifier::new().fit(&x, &y2).unwrap();
    assert_eq!(f.classes(), &[-3.0, 7.0]);
    assert!(accuracy(&f.predict(&x).unwrap(), &y2) >= 0.97);
}

#[test]
fn classifier_multiclass_keeps_ovr() {
    let (x, y) = blobs(300, 3, 12);
    let f = SgdClassifier::new().fit(&x, &y).unwrap();
    assert_eq!(f.weights().len(), 3);
    assert_eq!(f.decision_function(&x).unwrap().ncols(), 3);
    assert!(accuracy(&f.predict(&x).unwrap(), &y) >= 0.9);
}

#[test]
fn classifier_log_loss_huge_margin_is_finite() {
    // Huge feature scale + large constant LR -> margins of order 1e6+.
    // ln(sigmoid) used to give -inf / NaN here.
    let (x, y) = blobs(100, 2, 13);
    let x = x.mapv(|v| v * 1e4);
    let f = SgdClassifier::new()
        .with_loss(ClassifierLoss::Log)
        .with_learning_rate(LearningRate::Constant)
        .with_eta0(10.0)
        .with_penalty(SgdPenalty::None)
        .fit(&x, &y)
        .unwrap();
    assert!(f.weights()[0].iter().all(|w| w.is_finite()));
    assert!(f.biases()[0].is_finite());
    assert!(
        f.n_iter() < 1000,
        "loss must stay comparable: n_iter={}",
        f.n_iter()
    );
    let scores = f.decision_function(&x).unwrap();
    assert!(scores.iter().any(|s| s.abs() > 1e3), "margins not huge");
    let p = f.predict_proba(&x).unwrap();
    assert!(p.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
    assert!(accuracy(&f.predict(&x).unwrap(), &y) >= 0.95);
}

#[test]
fn classifier_partial_fit_matches_fit_and_persists_t() {
    for k in [2usize, 3] {
        let (x, y) = blobs(60, k, 14);
        let base = SgdClassifier::new()
            .with_loss(ClassifierLoss::Log)
            .with_shuffle(false)
            .with_penalty(SgdPenalty::ElasticNet)
            .with_average(10);
        let epochs = 8;
        let full = base
            .clone()
            .with_tol(None)
            .with_max_iter(epochs)
            .fit(&x, &y)
            .unwrap();
        let classes: Vec<f64> = (0..k).map(|c| c as f64).collect();
        let mut state = None;
        let mut last_t = 1;
        for _ in 0..epochs {
            let s = base.partial_fit(state, &x, &y, Some(&classes)).unwrap();
            assert_eq!(s.n_iter(), 1);
            assert_eq!(s.t(), last_t + x.nrows());
            last_t = s.t();
            state = Some(s);
        }
        let inc = state.unwrap();
        assert_eq!(inc.t(), full.t());
        assert_eq!(inc.weights().len(), full.weights().len());
        for (wa, wb) in inc.weights().iter().zip(full.weights()) {
            for (a, b) in wa.iter().zip(wb) {
                assert!((a - b).abs() < 1e-12, "k={k}: {a} vs {b}");
            }
        }
    }
}

#[test]
fn classifier_partial_fit_rejects_unknown_label() {
    let (x, y) = blobs(20, 2, 15);
    let clf = SgdClassifier::new();
    let s = clf.partial_fit(None, &x, &y, Some(&[0.0, 1.0])).unwrap();
    let y_bad = y.mapv(|v| v + 5.0);
    assert!(clf.partial_fit(Some(s), &x, &y_bad, None).is_err());
}

#[test]
fn classifier_early_stopping() {
    let (x, y) = blobs(200, 3, 16);
    let f = SgdClassifier::new()
        .with_early_stopping(true)
        .fit(&x, &y)
        .unwrap();
    assert!(f.n_iter() < 1000);
    assert!(accuracy(&f.predict(&x).unwrap(), &y) >= 0.85);
}

#[test]
fn classifier_predict_proba() {
    for k in [2usize, 3] {
        let (x, y) = blobs(150, k, 17);
        for loss in [ClassifierLoss::Log, ClassifierLoss::ModifiedHuber] {
            let f = SgdClassifier::new().with_loss(loss).fit(&x, &y).unwrap();
            let p = f.predict_proba(&x).unwrap();
            assert_eq!(p.dim(), (150, k));
            for row in p.rows() {
                assert!((row.sum() - 1.0).abs() < 1e-12, "{loss:?} {row:?}");
            }
            // argmax(proba) agrees with predict.
            let pred = f.predict(&x).unwrap();
            for (i, row) in p.rows().into_iter().enumerate() {
                let am = row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0;
                if row[am] > 0.5 + 1e-9 {
                    assert_eq!(f.classes()[am], pred[i]);
                }
            }
        }
        let hinge = SgdClassifier::new().fit(&x, &y).unwrap();
        assert!(hinge.predict_proba(&x).is_err());
    }
}

#[test]
fn serde_roundtrip_keeps_partial_fit_state() {
    let (x, y) = blobs(40, 2, 18);
    let clf = SgdClassifier::new().with_shuffle(false);
    let s = clf.partial_fit(None, &x, &y, Some(&[0.0, 1.0])).unwrap();
    let json = serde_json::to_string(&s).unwrap();
    let back: anofox_ml_linear::FittedSgdClassifier<f64> = serde_json::from_str(&json).unwrap();
    assert_eq!(back.t(), s.t());
    let a = clf.partial_fit(Some(s), &x, &y, None).unwrap();
    let b = clf.partial_fit(Some(back), &x, &y, None).unwrap();
    assert_eq!(a.weights(), b.weights());
}
