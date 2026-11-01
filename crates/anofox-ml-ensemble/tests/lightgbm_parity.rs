//! LightGBM 4.7 parity of the L1 objective (median baseline and leaf
//! refit), weighted baselines, the Huber default and categorical splits
//! (#50). Reference predictions come from `lightgbm.LGBMRegressor` with
//! `min_data_in_bin=3, deterministic=True, force_row_wise=True` on the same
//! integer-formula data.

use anofox_ml_core::{Fit, Predict};
use anofox_ml_ensemble::{CategoricalSplitParams, LgbmFitOptions, LgbmObjective, LgbmRegressor};
use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2};

/// Numeric data: 20-level and 7-level features, every 17th row an outlier.
fn data_num() -> (Array2<f64>, Array1<f64>) {
    let n = 200;
    let mut x = Array2::zeros((n, 2));
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let x0 = ((i * 7) % 20) as f64;
        let x1 = ((i * 3) % 7) as f64;
        x[[i, 0]] = x0;
        x[[i, 1]] = x1;
        let noise = (((i * 37) % 11) as f64 - 5.0) / 10.0;
        y[i] = 0.5 * x0 + if x1 == 3.0 { 4.0 } else { 0.0 } + noise;
        if i % 17 == 0 {
            y[i] += 25.0;
        }
    }
    (x, y)
}

fn weights_num() -> Array1<f64> {
    Array1::from_iter((0..200).map(|i| 1.0 + ((i * 13) % 5) as f64 / 2.0))
}

/// Categorical data: 12 categories with distinct effects, two rare
/// categories (40, 41) that fall into the "other" bin, and a numeric feature.
fn data_cat() -> (Array2<f64>, Array1<f64>) {
    let n = 600;
    let eff = [
        3.0, -1.0, 4.0, 1.0, -5.0, 9.0, 2.0, -6.0, 5.0, 3.0, -5.0, 8.0,
    ];
    let mut x = Array2::zeros((n, 2));
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let c = if i % 97 == 0 {
            40.0 + (i % 2) as f64
        } else {
            ((i * 5) % 12) as f64
        };
        let x1 = ((i * 3) % 10) as f64;
        x[[i, 0]] = c;
        x[[i, 1]] = x1;
        let e = if c < 12.0 { eff[c as usize] } else { 0.0 };
        y[i] = e + 0.3 * x1 + (((i * 37) % 11) as f64 - 5.0) / 10.0;
    }
    (x, y)
}

fn base() -> LgbmRegressor {
    LgbmRegressor::new()
        .with_n_estimators(5)
        .with_learning_rate(0.3)
        .with_num_leaves(4)
        .with_min_child_samples(5)
}

fn check(pred: &Array1<f64>, rows: impl Iterator<Item = usize>, expected: &[f64]) {
    let rows: Vec<usize> = rows.collect();
    assert_eq!(rows.len(), expected.len());
    for (&r, &e) in rows.iter().zip(expected) {
        assert_abs_diff_eq!(pred[r], e, epsilon = 1e-6);
    }
}

#[test]
fn l1_matches_lightgbm() {
    let (x, y) = data_num();
    let m = base()
        .with_objective(LgbmObjective::RegressionL1)
        .fit(&x, &y)
        .unwrap();
    let expected = [
        2.771985021,
        5.80969298,
        2.771985021,
        6.544692994,
        2.771985021,
        7.608230508,
        3.22415423,
        8.298528145,
        4.088442962,
        8.298528145,
        4.849692966,
        2.771985021,
        6.544692994,
        2.771985021,
        6.837236285,
        2.44198505,
    ];
    check(&m.predict(&x).unwrap(), (0..200).step_by(13), &expected);
}

#[test]
fn weighted_l1_matches_lightgbm() {
    let (x, y) = data_num();
    let w = weights_num();
    let opts = LgbmFitOptions {
        sample_weight: Some(&w),
        ..Default::default()
    };
    let m = base()
        .with_objective(LgbmObjective::RegressionL1)
        .fit_with_eval(&x, &y, &opts)
        .unwrap();
    let expected = [
        2.509765248,
        5.687987227,
        2.509765248,
        6.918429804,
        2.509765248,
        7.398429776,
        3.344098698,
        8.154846953,
        4.01086219,
        8.154846953,
        4.636154848,
        2.509765248,
        6.380654886,
        2.791765251,
        7.395484461,
        2.476765241,
    ];
    check(&m.predict(&x).unwrap(), (0..200).step_by(13), &expected);
}

#[test]
fn huber_default_delta_matches_lightgbm() {
    let (x, y) = data_num();
    let m = base()
        .with_objective(LgbmObjective::Huber)
        .fit(&x, &y)
        .unwrap();
    let expected = [
        5.660913973,
        6.178391714,
        5.660913973,
        6.747684451,
        5.660913973,
        7.304268774,
        5.660913973,
        8.057858852,
        5.7471222,
        8.057858852,
        6.178391714,
        5.660913973,
        6.556652264,
        5.660913973,
        7.304268774,
        5.660913973,
    ];
    check(&m.predict(&x).unwrap(), (0..200).step_by(13), &expected);
}

#[test]
fn weighted_l2_baseline_matches_lightgbm() {
    let (x, y) = data_num();
    let w = weights_num();
    let opts = LgbmFitOptions {
        sample_weight: Some(&w),
        ..Default::default()
    };
    let m = base().fit_with_eval(&x, &y, &opts).unwrap();
    let expected = [
        2.805124559,
        8.747801262,
        2.805124559,
        8.732738815,
        2.805124559,
        9.746978473,
        5.061457863,
        9.523810316,
        4.391721152,
        10.78179997,
        8.320698406,
        2.805124559,
        8.732738815,
        3.810107566,
        8.916091676,
        3.871623064,
    ];
    check(&m.predict(&x).unwrap(), (0..200).step_by(13), &expected);
}

fn cat_rows() -> impl Iterator<Item = usize> {
    (0..600).step_by(41).chain([97, 194])
}

#[test]
fn categorical_defaults_match_lightgbm() {
    let (x, y) = data_cat();
    let m = base()
        .with_categorical_features(vec![0])
        .fit(&x, &y)
        .unwrap();
    let expected = [
        1.5706436,
        1.449298573,
        5.446057365,
        2.869011571,
        -2.951168585,
        7.940661392,
        3.161154949,
        -2.951168585,
        5.831591662,
        4.448672248,
        -1.913425636,
        7.469264646,
        4.605646631,
        1.914774937,
        4.509184256,
        1.5706436,
        1.5706436,
    ];
    check(&m.predict(&x).unwrap(), cat_rows(), &expected);
}

#[test]
fn categorical_params_and_unseen_categories_match_lightgbm() {
    let (x, y) = data_cat();
    let m = base()
        .with_categorical_features(vec![0])
        .with_categorical_split(CategoricalSplitParams {
            min_data_per_group: 20,
            cat_smooth: 5.0,
            cat_l2: 1.0,
            max_cat_threshold: 4,
            ..Default::default()
        })
        .fit(&x, &y)
        .unwrap();
    let expected = [
        3.079406664,
        0.5484417842,
        4.756632438,
        2.867874037,
        -2.695296997,
        8.597923145,
        3.94380759,
        -2.695296997,
        5.402707638,
        4.459244198,
        -2.695296997,
        8.597923145,
        3.94380759,
        1.41284271,
        4.281143887,
        3.079406664,
        3.079406664,
    ];
    check(&m.predict(&x).unwrap(), cat_rows(), &expected);

    // An unseen category, a missing value and a rare training category all
    // take the "other" branch.
    let xt = ndarray::array![[100.0, 0.0], [f64::NAN, 0.0], [40.0, 0.0]];
    let p = m.predict(&xt).unwrap();
    for v in p {
        assert_abs_diff_eq!(v, 3.079406664, epsilon = 1e-6);
    }
}

fn classifier() -> anofox_ml_ensemble::LgbmClassifier {
    anofox_ml_ensemble::LgbmClassifier::new()
        .with_n_estimators(5)
        .with_learning_rate(0.3)
        .with_num_leaves(4)
        .with_min_child_samples(5)
}

#[test]
fn weighted_binary_baseline_matches_lightgbm() {
    let (x, y) = data_num();
    let yb = y.mapv(|v| if v > 6.0 { 1.0 } else { 0.0 });
    let w = weights_num();
    let opts = LgbmFitOptions {
        sample_weight: Some(&w),
        ..Default::default()
    };
    let m = classifier().fit_with_eval(&x, &yb, &opts).unwrap();
    let proba = m.predict_proba(&x).unwrap();
    let expected = [
        0.1017107024,
        0.2647586065,
        0.1017107024,
        0.8795232679,
        0.1017107024,
        0.8795232679,
        0.3334501292,
        0.8795232679,
        0.2010967846,
        0.8795232679,
        0.2647586065,
        0.1017107024,
        0.7309036956,
        0.1860783422,
        0.8795232679,
        0.2010967846,
    ];
    check(&proba.column(1).to_owned(), (0..200).step_by(13), &expected);
}

#[test]
fn weighted_multiclass_baseline_matches_lightgbm() {
    let (x, y) = data_num();
    let y3 = y.mapv(|v| {
        if v < 4.0 {
            0.0
        } else if v < 8.0 {
            1.0
        } else {
            2.0
        }
    });
    let w = weights_num();
    let opts = LgbmFitOptions {
        sample_weight: Some(&w),
        ..Default::default()
    };
    let m = classifier().fit_with_eval(&x, &y3, &opts).unwrap();
    let proba = m.predict_proba(&x).unwrap();
    let expected = [
        0.849266948,
        0.1085960702,
        0.04213698181,
        0.08860913426,
        0.6790550613,
        0.2323358045,
        0.849266948,
        0.1085960702,
        0.04213698181,
        0.08860913426,
        0.6790550613,
        0.2323358045,
        0.849266948,
        0.1085960702,
        0.04213698181,
        0.08860913426,
        0.6790550613,
        0.2323358045,
        0.239194097,
        0.6146545845,
        0.1461513185,
        0.05722579123,
        0.05818567722,
        0.8845885315,
        0.2297508804,
        0.6979063725,
        0.0723427471,
        0.05722579123,
        0.05818567722,
        0.8845885315,
        0.08860913426,
        0.6790550613,
        0.2323358045,
        0.849266948,
        0.1085960702,
        0.04213698181,
        0.08860913426,
        0.6790550613,
        0.2323358045,
        0.239194097,
        0.6146545845,
        0.1461513185,
        0.09371371708,
        0.7181739718,
        0.1881123111,
        0.7930095133,
        0.1370531766,
        0.06993731015,
    ];
    let rows: Vec<usize> = (0..200).step_by(13).collect();
    for (k, &r) in rows.iter().enumerate() {
        for c in 0..3 {
            assert_abs_diff_eq!(proba[[r, c]], expected[3 * k + c], epsilon = 1e-6);
        }
    }
}
