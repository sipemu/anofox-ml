//! Every clustering estimator accepts `f32` input and, since it computes in
//! `f64`, gives the same result as for the same data in `f64` (the values
//! below are exact in both types).

use anofox_ml_cluster::*;
use anofox_ml_core::{FitUnsupervised, Predict};
use ndarray::Array2;

fn blobs() -> (Array2<f64>, Array2<f32>) {
    // Three blobs on a 1/8 grid: exactly representable in f32.
    let x = Array2::from_shape_fn((30, 2), |(i, j)| {
        let centre = [(0.0, 0.0), (6.0, 6.0), (0.0, 7.0)][i % 3];
        let off = ((i * (j + 3)) % 7) as f64 / 8.0;
        if j == 0 {
            centre.0 + off
        } else {
            centre.1 + off
        }
    });
    let x32 = x.mapv(|v| v as f32);
    (x, x32)
}

#[test]
fn test_f32_matches_f64() {
    let (x, x32) = blobs();

    let a = FitUnsupervised::<f64>::fit(&AgglomerativeClustering::new(3), &x).unwrap();
    let b = FitUnsupervised::<f32>::fit(&AgglomerativeClustering::new(3), &x32).unwrap();
    assert_eq!(a.labels, b.labels);

    let ap = AffinityPropagation::new();
    assert_eq!(
        FitUnsupervised::<f64>::fit(&ap, &x).unwrap().labels,
        FitUnsupervised::<f32>::fit(&ap, &x32).unwrap().labels
    );

    let sc = SpectralClustering::new(3).with_seed(0);
    assert_eq!(
        FitUnsupervised::<f64>::fit(&sc, &x).unwrap().labels,
        FitUnsupervised::<f32>::fit(&sc, &x32).unwrap().labels
    );

    let op = Optics::new(4, f64::INFINITY);
    assert_eq!(
        FitUnsupervised::<f64>::fit(&op, &x).unwrap().labels,
        FitUnsupervised::<f32>::fit(&op, &x32).unwrap().labels
    );

    let hd = Hdbscan::new(4, 5);
    assert_eq!(
        FitUnsupervised::<f64>::fit(&hd, &x).unwrap().labels,
        FitUnsupervised::<f32>::fit(&hd, &x32).unwrap().labels
    );

    // Estimators with predict: f32 in, f32 out.
    let gmm = GaussianMixture::new(3).with_seed(0);
    let g64 = FitUnsupervised::<f64>::fit(&gmm, &x).unwrap();
    let g32 = FitUnsupervised::<f32>::fit(&gmm, &x32).unwrap();
    let p32: ndarray::Array1<f32> = g32.predict(&x32).unwrap();
    assert_eq!(p32.mapv(f64::from), g64.predict(&x).unwrap());

    let bgmm = BayesianGaussianMixture::new(3).with_seed(0);
    let b64 = FitUnsupervised::<f64>::fit(&bgmm, &x).unwrap();
    let b32 = FitUnsupervised::<f32>::fit(&bgmm, &x32).unwrap();
    assert_eq!(
        b32.predict(&x32).unwrap().mapv(f64::from),
        b64.predict(&x).unwrap()
    );

    let birch = Birch::new(0.5);
    let r64 = FitUnsupervised::<f64>::fit(&birch, &x).unwrap();
    let r32 = FitUnsupervised::<f32>::fit(&birch, &x32).unwrap();
    assert_eq!(
        r32.predict(&x32).unwrap().mapv(f64::from),
        r64.predict(&x).unwrap()
    );

    let ms = MeanShift::new(2.0);
    let m64 = FitUnsupervised::<f64>::fit(&ms, &x).unwrap();
    let m32 = FitUnsupervised::<f32>::fit(&ms, &x32).unwrap();
    assert_eq!(
        m32.predict(&x32).unwrap().mapv(f64::from),
        m64.predict(&x).unwrap()
    );
}
