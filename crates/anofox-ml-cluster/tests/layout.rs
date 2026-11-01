//! Regression tests for #42: estimators must accept non-standard-layout
//! (Fortran-order / transposed) input and give the same result as for the
//! equivalent C-order array.

use anofox_ml_cluster::{
    AgglomerativeClustering, BayesianGaussianMixture, Dbscan, GaussianMixture, Hdbscan, KMeans,
    MiniBatchKMeans,
};
use anofox_ml_core::{FitUnsupervised, Predict};
use ndarray::{Array2, ShapeBuilder};

/// Two blobs, 3 features, in C order.
fn blobs() -> Array2<f64> {
    let mut v = Vec::new();
    for i in 0..30 {
        let t = i as f64;
        let off = if i < 15 { 0.0 } else { 8.0 };
        v.push(off + (t * 0.7).sin() * 0.5);
        v.push(off + (t * 1.3).cos() * 0.5);
        v.push((t * 0.4).sin() * 0.3);
    }
    Array2::from_shape_vec((30, 3), v).unwrap()
}

/// The same values in Fortran (column-major) order.
fn fortran(x: &Array2<f64>) -> Array2<f64> {
    let v: Vec<f64> = x.t().iter().copied().collect();
    let xf = Array2::from_shape_vec((x.nrows(), x.ncols()).f(), v).unwrap();
    assert!(!xf.is_standard_layout());
    assert_eq!(&xf, x);
    xf
}

#[test]
fn kmeans_fortran_order() {
    let x = blobs();
    let xf = fortran(&x);
    let km = KMeans::new(2).with_seed(1);
    let a = FitUnsupervised::<f64>::fit(&km, &x).unwrap();
    let b = FitUnsupervised::<f64>::fit(&km, &xf).unwrap();
    assert_eq!(a.labels(), b.labels());
    assert_eq!(a.centroids(), b.centroids());
    assert_eq!(a.predict(&x).unwrap(), b.predict(&xf).unwrap());
}

#[test]
fn mini_batch_kmeans_fortran_order() {
    let x = blobs();
    let xf = fortran(&x);
    let m = MiniBatchKMeans::new(2);
    let a = FitUnsupervised::<f64>::fit(&m, &x).unwrap();
    let b = FitUnsupervised::<f64>::fit(&m, &xf).unwrap();
    assert_eq!(a.centroids(), b.centroids());
    assert_eq!(a.predict(&x).unwrap(), b.predict(&xf).unwrap());
}

#[test]
fn dbscan_fortran_order() {
    let x = blobs();
    let xf = fortran(&x);
    let d = Dbscan::new(1.5, 3);
    let a = FitUnsupervised::<f64>::fit(&d, &x).unwrap();
    let b = FitUnsupervised::<f64>::fit(&d, &xf).unwrap();
    assert_eq!(a.labels(), b.labels());
    assert_eq!(b.predict(&xf).unwrap().len(), xf.nrows());
}

#[test]
fn gmm_bgmm_fortran_order() {
    let x = blobs();
    let xf = fortran(&x);
    let g = GaussianMixture::new(2);
    let a = g.fit(&x).unwrap();
    let b = g.fit(&xf).unwrap();
    assert_eq!(a.predict(&x).unwrap(), b.predict(&xf).unwrap());

    let bg = BayesianGaussianMixture::new(2);
    let a = bg.fit(&x).unwrap();
    let b = bg.fit(&xf).unwrap();
    assert_eq!(a.predict(&x).unwrap(), b.predict(&xf).unwrap());
}

#[test]
fn agglomerative_hdbscan_fortran_order() {
    let x = blobs();
    let xf = fortran(&x);
    let ag = AgglomerativeClustering::new(2);
    assert_eq!(ag.fit(&x).unwrap().labels, ag.fit(&xf).unwrap().labels);
    let h = Hdbscan::new(3, 5);
    assert_eq!(h.fit(&x).unwrap().labels, h.fit(&xf).unwrap().labels);
}
