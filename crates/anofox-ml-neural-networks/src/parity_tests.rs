//! Trajectory parity with scikit-learn 1.8.0 `MLPRegressor` / `MLPClassifier`.
//!
//! Reference values come from sklearn with `coefs_` / `intercepts_`
//! injected (fit once, overwrite the weights, reset `loss_curve_`,
//! `best_loss_`, `t_`, then `warm_start=True` refit) and `shuffle=False`, so
//! both implementations follow the identical deterministic trajectory.

use crate::{Activation, LearningRateSchedule, MlpClassifier, MlpRegressor, Solver};
use anofox_ml_core::{Fit, PartialFit, Predict};
use ndarray::{Array1, Array2};

fn x_data(n: usize, d: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, d), |(i, k)| {
        (0.37 * i as f64 * (k + 1) as f64 + k as f64).sin()
    })
}

fn y_reg(x: &Array2<f64>) -> Array1<f64> {
    Array1::from_shape_fn(x.nrows(), |i| {
        x[[i, 0]] * 2.0 - x[[i, 1]] + 0.5 * (3.0 * x[[i, 2]]).sin()
    })
}

fn init_weights(sizes: &[usize]) -> Vec<(Array2<f64>, Array1<f64>)> {
    (0..sizes.len() - 1)
        .map(|l| {
            let w = Array2::from_shape_fn((sizes[l], sizes[l + 1]), |(i, j)| {
                0.4 * (1.7 * (i + 1) as f64 + 0.9 * (j + 1) as f64 + l as f64).sin()
            });
            let b = Array1::from_shape_fn(sizes[l + 1], |j| {
                0.1 * (2.3 * (j + 1) as f64 + l as f64).cos()
            });
            (w, b)
        })
        .collect()
}

fn assert_close(a: &[f64], b: &[f64], tol: f64, what: &str) {
    assert_eq!(a.len(), b.len(), "{what} length");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= tol * (1.0 + y.abs()),
            "{what}[{i}]: ours {x} vs sklearn {y}"
        );
    }
}

#[test]
fn sklearn_regressor_sgd_nesterov_minibatch() {
    let x = x_data(20, 3);
    let y = y_reg(&x);
    let mlp = MlpRegressor {
        hidden_layer_sizes: vec![5],
        activation: Activation::Tanh,
        solver: Solver::Sgd,
        learning_rate: 0.05,
        momentum: 0.9,
        nesterovs_momentum: true,
        batch_size: Some(10),
        shuffle: false,
        alpha: 0.01,
        tol: 1e-12,
        n_iter_no_change: 1000,
        max_iter: 6,
        ..Default::default()
    }
    .with_initial_weights(init_weights(&[3, 5, 1]));
    let fitted = mlp.fit(&x, &y).unwrap();
    assert_close(
        fitted.loss_curve(),
        &[
            1.3407858357471043,
            1.148218939633115,
            0.9329020880215871,
            0.7079621632254384,
            0.4671207069708648,
            0.24868134468269157,
        ],
        1e-12,
        "loss_curve",
    );
    assert_eq!(fitted.n_iter(), 6);
    let p = fitted
        .predict(&x.slice(ndarray::s![..4, ..]).to_owned())
        .unwrap();
    assert_close(
        p.as_slice().unwrap(),
        &[
            -0.9584800332218424,
            -0.39301776429395985,
            0.6067857050964376,
            1.6687075982990345,
        ],
        1e-12,
        "pred",
    );
    let w0: Vec<f64> = fitted.coefs()[0].iter().copied().collect();
    assert_close(
        &w0,
        &[
            0.24586146430764,
            -0.8684651088797415,
            -0.06453554619238043,
            -0.2575420747929937,
            -1.0035496996353737,
            -0.34425194033484036,
            0.017585872062633658,
            -0.24791710964115576,
            0.1817258515843853,
            0.6783828916559514,
            -0.06756717102964331,
            0.09187549340829837,
            0.454542887164793,
            0.24321695090473397,
            -0.16113453663185834,
        ],
        1e-12,
        "coefs_[0]",
    );
}

#[test]
fn sklearn_classifier_adam_multiclass_minibatch() {
    let x = x_data(24, 2);
    let y = Array1::from_shape_fn(24, |i| (i % 3) as f64);
    let mlp = MlpClassifier {
        hidden_layer_sizes: vec![6, 4],
        activation: Activation::Relu,
        solver: Solver::Adam,
        learning_rate: 0.01,
        batch_size: Some(8),
        shuffle: false,
        alpha: 0.1,
        tol: 1e-12,
        n_iter_no_change: 1000,
        max_iter: 5,
        ..Default::default()
    }
    .with_initial_weights(init_weights(&[2, 6, 4, 3]));
    let fitted = mlp.fit(&x, &y).unwrap();
    assert_close(
        fitted.loss_curve(),
        &[
            1.1272037834052349,
            1.120741623606775,
            1.1166069791389117,
            1.1133299812164823,
            1.1107091702388192,
        ],
        1e-12,
        "loss_curve",
    );
    let p: Vec<f64> = fitted
        .predict_proba(&x.slice(ndarray::s![..3, ..]).to_owned())
        .unwrap()
        .iter()
        .copied()
        .collect();
    assert_close(
        &p,
        &[
            0.33187464547505774,
            0.34807991825094964,
            0.32004543627399273,
            0.33293944595047215,
            0.3477190295635841,
            0.3193415244859437,
            0.33157603617072273,
            0.3484314859566093,
            0.31999247787266794,
        ],
        1e-12,
        "proba",
    );
}

#[test]
fn sklearn_binary_classifier_sgd_invscaling_logistic() {
    let x = x_data(24, 2);
    let y = Array1::from_shape_fn(24, |i| {
        if x[[i, 0]] + 0.3 > x[[i, 1]] {
            1.0
        } else {
            0.0
        }
    });
    let mlp = MlpClassifier {
        hidden_layer_sizes: vec![4],
        activation: Activation::Sigmoid,
        solver: Solver::Sgd,
        learning_rate_schedule: LearningRateSchedule::InvScaling,
        learning_rate: 0.5,
        power_t: 0.25,
        momentum: 0.5,
        nesterovs_momentum: false,
        batch_size: Some(24),
        shuffle: false,
        alpha: 1e-3,
        tol: 1e-12,
        n_iter_no_change: 1000,
        max_iter: 7,
        ..Default::default()
    }
    .with_initial_weights(init_weights(&[2, 4, 1]));
    let fitted = mlp.fit(&x, &y).unwrap();
    assert_close(
        fitted.loss_curve(),
        &[
            0.7247337930098242,
            0.7122560309463354,
            0.7039331239352672,
            0.6988429133819835,
            0.6956583008839498,
            0.6935186254231661,
            0.6919483247598507,
        ],
        1e-12,
        "loss_curve",
    );
    let p = fitted
        .predict_proba(&x.slice(ndarray::s![..3, ..]).to_owned())
        .unwrap();
    assert_close(
        &p.column(1).to_vec(),
        &[0.5786022884836455, 0.5756774957036349, 0.5709776488078241],
        1e-12,
        "proba",
    );
}

#[test]
fn sklearn_multi_output_regression_trajectory() {
    let x = x_data(20, 3);
    let y = y_reg(&x);
    let mut yy = Array2::zeros((20, 2));
    for i in 0..20 {
        yy[[i, 0]] = y[i];
        yy[[i, 1]] = x[[i, 2]].powi(2) - 0.3;
    }
    let mlp = MlpRegressor {
        hidden_layer_sizes: vec![5],
        activation: Activation::Relu,
        solver: Solver::Sgd,
        learning_rate: 0.02,
        momentum: 0.0,
        batch_size: Some(20),
        shuffle: false,
        alpha: 0.05,
        tol: 1e-12,
        n_iter_no_change: 1000,
        max_iter: 4,
        ..Default::default()
    }
    .with_initial_weights(init_weights(&[3, 5, 2]));
    let fitted = mlp.fit_multi(&x, &yy).unwrap();
    assert_eq!(fitted.n_outputs(), 2);
    assert!(
        fitted.predict(&x).is_err(),
        "1-D predict on a 2-output model"
    );
    let p: Vec<f64> = fitted
        .predict_multi(&x.slice(ndarray::s![..3, ..]).to_owned())
        .unwrap()
        .iter()
        .copied()
        .collect();
    assert_close(
        &p,
        &[
            -0.017584704776118523,
            0.09406472029105538,
            -0.22438129895076886,
            -0.07421974605294879,
            -0.19782900263115885,
            -0.03092755621292216,
        ],
        1e-12,
        "multi pred",
    );
}

#[test]
fn sklearn_lbfgs_regressor_converges_to_same_minimum() {
    // L-BFGS paths differ from SciPy's L-BFGS-B, so compare the converged
    // optimum on regularised problems with a unique minimum from the same
    // start (weakly regularised nets can end in different local minima).
    let x = x_data(20, 3);
    let y = y_reg(&x);
    let cases = [
        (
            Activation::Tanh,
            1.0,
            4,
            0.1495296019389109,
            [
                -0.568660622205806,
                -0.2845589623781922,
                0.42904043769370065,
                1.6188711611384257,
            ],
        ),
        (
            Activation::Identity,
            0.1,
            3,
            0.03198398499326887,
            [
                -0.5490989381143667,
                -0.2535691216622357,
                0.4482550106300896,
                1.6032944138876353,
            ],
        ),
        (
            Activation::Sigmoid,
            0.5,
            4,
            0.2662744387186919,
            [
                -0.5397952300015539,
                -0.26387111240480443,
                0.41190259230645754,
                1.5119305524653468,
            ],
        ),
    ];
    for (act, alpha, h, loss, pred) in cases {
        let mlp = MlpRegressor {
            hidden_layer_sizes: vec![h],
            activation: act,
            solver: Solver::Lbfgs,
            alpha,
            tol: 1e-10,
            max_iter: 5000,
            max_fun: 100_000,
            ..Default::default()
        }
        .with_initial_weights(init_weights(&[3, h, 1]));
        let fitted = mlp.fit(&x, &y).unwrap();
        assert!(
            (fitted.loss() - loss).abs() < 1e-7,
            "{act:?}: loss {} vs {loss} (n_iter {})",
            fitted.loss(),
            fitted.n_iter()
        );
        assert!(fitted.loss_curve().is_empty());
        assert!(fitted.n_iter() > 0);
        let p = fitted
            .predict(&x.slice(ndarray::s![..4, ..]).to_owned())
            .unwrap();
        assert_close(p.as_slice().unwrap(), &pred, 1e-4, "lbfgs pred");
    }
}

#[test]
fn sklearn_adaptive_learning_rate_stopping() {
    let x = x_data(20, 3);
    let y = y_reg(&x);
    let mlp = MlpRegressor {
        hidden_layer_sizes: vec![5],
        activation: Activation::Tanh,
        solver: Solver::Sgd,
        learning_rate_schedule: LearningRateSchedule::Adaptive,
        learning_rate: 0.05,
        batch_size: Some(20),
        shuffle: false,
        alpha: 0.0,
        tol: 1e-3,
        n_iter_no_change: 3,
        max_iter: 300,
        ..Default::default()
    }
    .with_initial_weights(init_weights(&[3, 5, 1]));
    let fitted = mlp.fit(&x, &y).unwrap();
    assert_eq!(fitted.n_iter(), 47);
    let c = fitted.loss_curve();
    assert_close(
        &c[c.len() - 3..],
        &[0.21670521309879592, 0.2182983139788043, 0.21974261438729842],
        1e-12,
        "adaptive tail",
    );
}

#[test]
fn stopping_requires_n_iter_no_change_epochs() {
    // tol = 1 → every epoch after the first counts as "no improvement";
    // sklearn stops once the counter exceeds n_iter_no_change (= 2) → 4 epochs.
    let x = x_data(20, 3);
    let y = y_reg(&x);
    let fitted = MlpRegressor {
        hidden_layer_sizes: vec![5],
        solver: Solver::Sgd,
        learning_rate: 0.01,
        tol: 1.0,
        n_iter_no_change: 2,
        max_iter: 100,
        ..Default::default()
    }
    .fit(&x, &y)
    .unwrap();
    assert_eq!(fitted.n_iter(), 4);
    let min = fitted
        .loss_curve()
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    assert_eq!(fitted.best_loss(), Some(min));
}

#[test]
fn partial_fit_equals_fit_epochs_and_warm_start_continues() {
    let x = x_data(20, 3);
    let y = y_reg(&x);
    let base = MlpRegressor {
        hidden_layer_sizes: vec![5],
        activation: Activation::Tanh,
        solver: Solver::Sgd,
        learning_rate: 0.05,
        momentum: 0.9,
        batch_size: Some(7),
        shuffle: false,
        tol: 1e-12,
        n_iter_no_change: 1000,
        ..Default::default()
    }
    .with_initial_weights(init_weights(&[3, 5, 1]));
    let full = base.clone().with_max_iter(5).fit(&x, &y).unwrap();
    let mut state = None;
    for _ in 0..5 {
        state = Some(base.partial_fit(state, &x, &y, None).unwrap());
    }
    let inc = state.unwrap();
    assert_eq!(inc.n_iter(), 1);
    assert_close(
        inc.loss_curve(),
        full.loss_curve(),
        1e-13,
        "partial_fit curve",
    );
    assert_eq!(inc.t(), 100);

    // Warm start with plain SGD (no optimiser state) continues the trajectory.
    let plain = base.clone().with_momentum(0.0);
    let six = plain.clone().with_max_iter(6).fit(&x, &y).unwrap();
    let three = plain.clone().with_max_iter(3).fit(&x, &y).unwrap();
    let cont = plain
        .with_max_iter(3)
        .fit_warm_start(three, &x, &y)
        .unwrap();
    assert_eq!(cont.loss_curve().len(), 6);
    assert_close(
        cont.loss_curve(),
        six.loss_curve(),
        1e-13,
        "warm start curve",
    );
}

#[test]
fn partial_fit_classifier_requires_known_classes_and_rejects_lbfgs() {
    let x = x_data(24, 2);
    let y = Array1::from_shape_fn(24, |i| (i % 3) as f64);
    let mlp = MlpClassifier::default().with_hidden_layer_sizes(vec![4]);
    let s = mlp
        .partial_fit(
            None,
            &x.slice(ndarray::s![..2, ..]).to_owned(),
            &ndarray::array![0.0, 1.0],
            Some(&[0.0, 1.0, 2.0]),
        )
        .unwrap();
    assert_eq!(s.classes(), &[0.0, 1.0, 2.0]);
    let s = mlp.partial_fit(Some(s), &x, &y, None).unwrap();
    assert_eq!(s.loss_curve().len(), 2);
    assert!(mlp
        .partial_fit(Some(s), &x, &y.mapv(|v| v + 10.0), None)
        .is_err());
    let lb = mlp.clone().with_solver(Solver::Lbfgs);
    assert!(lb.partial_fit(None, &x, &y, None).is_err());
}

#[test]
fn early_stopping_keeps_best_validation_weights() {
    let x = x_data(60, 3);
    let y = y_reg(&x);
    let fitted = MlpRegressor {
        hidden_layer_sizes: vec![8],
        solver: Solver::Adam,
        learning_rate: 0.01,
        early_stopping: true,
        validation_fraction: 0.2,
        n_iter_no_change: 5,
        max_iter: 400,
        seed: 3,
        ..Default::default()
    }
    .fit(&x, &y)
    .unwrap();
    let scores = fitted.validation_scores();
    assert_eq!(scores.len(), fitted.n_iter());
    let best = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert_eq!(fitted.best_validation_score(), Some(best));
    assert_eq!(fitted.best_loss(), None);
    assert!(fitted.n_iter() < 400, "early stopping should trigger");

    let xc = x_data(60, 2);
    let yc = Array1::from_shape_fn(60, |i| if xc[[i, 0]] > xc[[i, 1]] { 1.0 } else { 0.0 });
    let clf = MlpClassifier::default()
        .with_hidden_layer_sizes(vec![6])
        .with_early_stopping(true)
        .with_validation_fraction(0.25)
        .fit(&xc, &yc)
        .unwrap();
    assert!(clf
        .validation_scores()
        .iter()
        .all(|s| (0.0..=1.0).contains(s)));
    // partial_fit with early stopping is rejected (sklearn).
    let es = MlpClassifier::default().with_early_stopping(true);
    assert!(es.partial_fit(None, &xc, &yc, None).is_err());
}

#[test]
fn l2_penalty_is_scaled_by_batch_size() {
    // Loss at the injected start = data loss + alpha/(2n)·||W||²: with
    // learning_rate → 0 the first epoch's loss isolates the penalty.
    let x = x_data(50, 3);
    let y = y_reg(&x);
    let w = init_weights(&[3, 4, 1]);
    let sq: f64 = w
        .iter()
        .map(|(m, _)| m.iter().map(|v| v * v).sum::<f64>())
        .sum();
    let run = |alpha: f64| {
        MlpRegressor {
            hidden_layer_sizes: vec![4],
            solver: Solver::Sgd,
            learning_rate: 1e-300,
            momentum: 0.0,
            batch_size: None,
            alpha,
            max_iter: 1,
            ..Default::default()
        }
        .with_initial_weights(w.clone())
        .fit(&x, &y)
        .unwrap()
        .loss_curve()[0]
    };
    let diff = run(2.0) - run(0.0);
    assert!((diff - 2.0 * sq / (2.0 * 50.0)).abs() < 1e-12, "{diff}");
}

#[test]
fn f32_training_works() {
    let x = x_data(30, 3).mapv(|v| v as f32);
    let y = y_reg(&x_data(30, 3)).mapv(|v| v as f32);
    for solver in [Solver::Adam, Solver::Sgd, Solver::Lbfgs] {
        let f = MlpRegressor::default()
            .with_hidden_layer_sizes(vec![6])
            .with_solver(solver)
            .with_max_iter(50)
            .fit(&x, &y)
            .unwrap();
        assert!(f.predict(&x).unwrap().iter().all(|v| v.is_finite()));
    }
}
