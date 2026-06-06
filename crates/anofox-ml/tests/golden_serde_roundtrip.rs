//! Persistence round-trip tests for the new fitted models.
//!
//! Asserts that fitted estimators can be JSON-serialised and re-loaded
//! producing identical predictions. Covers a representative sample;
//! anofox-backed fitted models (Tweedie/Gamma) are excluded because
//! `FittedTweedie` does not implement `Serialize`.

use anofox_ml::core::{Fit, FitUnsupervised, Predict, PredictProba, Transform};
use anofox_ml::prelude::*;
use anofox_ml_cluster::{AgglomerativeClustering, Linkage, MiniBatchKMeans};
use anofox_ml_preprocessing::{KernelPca, KpcaKernel, Nmf, PlsRegression, TruncatedSvd};
use anofox_ml_regression::{
    ARDRegression, BayesianRidge, KernelRidge, Lars, OrthogonalMatchingPursuit, RansacRegressor,
    TheilSenRegressor,
};
use ndarray::array;

fn assert_array_eq(a: &ndarray::Array1<f64>, b: &ndarray::Array1<f64>) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!((x - y).abs() < 1e-12, "[{i}] {} vs {}", x, y);
    }
}

#[test]
fn test_kernel_ridge_roundtrip() {
    let x = array![[0.0_f64, 1.0], [1.0, 0.0], [1.0, 1.0], [2.0, 3.0]];
    let y = array![1.0, 2.0, 3.0, 4.0];
    let fitted = KernelRidge::new()
        .with_alpha(0.5)
        .with_kernel(anofox_ml_svm::SvmKernel::Rbf { gamma: 0.5 })
        .fit(&x, &y)
        .unwrap();
    let json = serde_json::to_string(&fitted).unwrap();
    let back: anofox_ml_regression::FittedKernelRidge = serde_json::from_str(&json).unwrap();
    assert_array_eq(&fitted.predict(&x).unwrap(), &back.predict(&x).unwrap());
}

#[test]
fn test_bayesian_ridge_roundtrip() {
    let x = array![[1.0_f64], [2.0], [3.0], [4.0], [5.0]];
    let y = array![1.5, 3.0, 4.5, 6.0, 7.5];
    let fitted = BayesianRidge::new().fit(&x, &y).unwrap();
    let json = serde_json::to_string(&fitted).unwrap();
    let back: anofox_ml_regression::FittedBayesianRidge = serde_json::from_str(&json).unwrap();
    assert_array_eq(&fitted.predict(&x).unwrap(), &back.predict(&x).unwrap());
    assert_array_eq(
        &fitted.predict_std(&x).unwrap(),
        &back.predict_std(&x).unwrap(),
    );
}

#[test]
fn test_lars_roundtrip() {
    let x = array![
        [1.0_f64, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [2.0, 4.0, 6.0]
    ];
    let y = array![1.0, 2.0, 3.0, 4.0];
    let fitted = Lars::new(2).fit(&x, &y).unwrap();
    let json = serde_json::to_string(&fitted).unwrap();
    let back: anofox_ml_regression::FittedLars = serde_json::from_str(&json).unwrap();
    assert_array_eq(&fitted.predict(&x).unwrap(), &back.predict(&x).unwrap());
}

#[test]
fn test_ard_roundtrip() {
    let x = array![
        [1.0_f64, 0.0],
        [2.0, 0.5],
        [3.0, -0.3],
        [4.0, 0.2],
        [5.0, 0.1]
    ];
    let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let fitted = ARDRegression::new().fit(&x, &y).unwrap();
    let json = serde_json::to_string(&fitted).unwrap();
    let back: anofox_ml_regression::FittedARDRegression = serde_json::from_str(&json).unwrap();
    assert_array_eq(&fitted.predict(&x).unwrap(), &back.predict(&x).unwrap());
}

#[test]
fn test_lda_roundtrip() {
    let x = array![[0.0_f64, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 4.9]];
    let y = array![0.0, 0.0, 1.0, 1.0];
    let fitted = LinearDiscriminantAnalysis::new().fit(&x, &y).unwrap();
    let json = serde_json::to_string(&fitted).unwrap();
    let back: anofox_ml::discriminant::FittedLinearDiscriminantAnalysis =
        serde_json::from_str(&json).unwrap();
    assert_array_eq(&fitted.predict(&x).unwrap(), &back.predict(&x).unwrap());
    // PredictProba should also round-trip.
    let p1 = fitted.predict_proba(&x).unwrap();
    let p2 = back.predict_proba(&x).unwrap();
    for ((r, c), &v) in p1.indexed_iter() {
        assert!((v - p2[[r, c]]).abs() < 1e-12);
    }
}

#[test]
fn test_truncated_svd_and_nmf_roundtrip() {
    let x = array![
        [1.0_f64, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [2.0, 3.0, 4.0],
    ];
    let svd = TruncatedSvd::new(2).fit(&x).unwrap();
    let json = serde_json::to_string(&svd).unwrap();
    let back: anofox_ml_preprocessing::FittedTruncatedSvd = serde_json::from_str(&json).unwrap();
    let t1 = svd.transform(&x).unwrap();
    let t2 = back.transform(&x).unwrap();
    for ((r, c), &v) in t1.indexed_iter() {
        assert!((v - t2[[r, c]]).abs() < 1e-12);
    }

    let nmf = Nmf::new(2).fit(&x).unwrap();
    let json = serde_json::to_string(&nmf).unwrap();
    let _back: anofox_ml_preprocessing::FittedNmf = serde_json::from_str(&json).unwrap();
}

#[test]
fn test_agglomerative_kmeansmini_roundtrip() {
    let x = array![[0.0_f64, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.0],];
    let a = AgglomerativeClustering::new(2)
        .with_linkage(Linkage::Ward)
        .fit(&x)
        .unwrap();
    let json = serde_json::to_string(&a).unwrap();
    let _back: anofox_ml_cluster::FittedAgglomerativeClustering =
        serde_json::from_str(&json).unwrap();

    let m: anofox_ml_cluster::FittedMiniBatchKMeans<f64> =
        FitUnsupervised::fit(&MiniBatchKMeans::new(2).with_seed(0), &x).unwrap();
    let json = serde_json::to_string(&m).unwrap();
    let _back: anofox_ml_cluster::FittedMiniBatchKMeans<f64> = serde_json::from_str(&json).unwrap();
}

#[test]
fn test_robust_omp_roundtrip() {
    let x = ndarray::Array2::from_shape_vec((10, 1), (0..10).map(|i| i as f64).collect()).unwrap();
    let y = ndarray::Array1::from_vec((0..10).map(|i| 2.0 * i as f64 + 1.0).collect());
    let r = RansacRegressor::new()
        .with_min_samples(2)
        .with_residual_threshold(0.5)
        .with_max_trials(50)
        .with_seed(0)
        .fit(&x, &y)
        .unwrap();
    let json = serde_json::to_string(&r).unwrap();
    let back: anofox_ml_regression::FittedRansacRegressor = serde_json::from_str(&json).unwrap();
    assert_array_eq(&r.predict(&x).unwrap(), &back.predict(&x).unwrap());

    let t = TheilSenRegressor::new().with_seed(0).fit(&x, &y).unwrap();
    let json = serde_json::to_string(&t).unwrap();
    let _back: anofox_ml_regression::FittedTheilSenRegressor = serde_json::from_str(&json).unwrap();

    let o = OrthogonalMatchingPursuit::new()
        .with_n_nonzero_coefs(1)
        .fit(&x, &y)
        .unwrap();
    let json = serde_json::to_string(&o).unwrap();
    let _back: anofox_ml_regression::FittedOrthogonalMatchingPursuit =
        serde_json::from_str(&json).unwrap();
}

#[test]
fn test_kernel_pca_roundtrip() {
    let x = array![[0.0_f64, 1.0], [1.0, 0.0], [2.0, 2.0], [3.0, 1.0]];
    let kpca = KernelPca::new(2, KpcaKernel::Rbf { gamma: 0.5 })
        .fit(&x)
        .unwrap();
    let json = serde_json::to_string(&kpca).unwrap();
    let back: anofox_ml_preprocessing::FittedKernelPca = serde_json::from_str(&json).unwrap();
    let t1 = kpca.transform(&x).unwrap();
    let t2 = back.transform(&x).unwrap();
    for ((r, c), &v) in t1.indexed_iter() {
        assert!((v - t2[[r, c]]).abs() < 1e-12);
    }
}

#[test]
fn test_pls_roundtrip() {
    let x = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
    let y = array![1.0, 2.0, 3.0, 4.0];
    let pls = PlsRegression::new(1).fit(&x, &y).unwrap();
    let json = serde_json::to_string(&pls).unwrap();
    let back: anofox_ml_preprocessing::FittedPlsRegression = serde_json::from_str(&json).unwrap();
    assert_array_eq(&pls.predict(&x).unwrap(), &back.predict(&x).unwrap());
}
