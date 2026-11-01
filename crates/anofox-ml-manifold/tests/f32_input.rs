//! The manifold learners accept `f32` input and, since they compute in
//! `f64`, embed it exactly as the same data in `f64`.

use anofox_ml_core::FitUnsupervised;
use anofox_ml_manifold::*;
use ndarray::Array2;

#[test]
fn test_f32_matches_f64() {
    // A noisy curve on a 1/16 grid: exactly representable in f32.
    let x = Array2::from_shape_fn((40, 3), |(i, j)| {
        let t = i as f64 / 4.0;
        (((t * (j as f64 + 1.0)).sin() * 16.0).round()) / 16.0 + t * j as f64 / 8.0
    });
    let x32 = x.mapv(|v| v as f32);

    let iso = Isomap::new(2, 6);
    assert_eq!(
        FitUnsupervised::<f64>::fit(&iso, &x).unwrap().embedding,
        FitUnsupervised::<f32>::fit(&iso, &x32).unwrap().embedding
    );
    let lle = LocallyLinearEmbedding::new(2, 8);
    assert_eq!(
        FitUnsupervised::<f64>::fit(&lle, &x).unwrap().embedding,
        FitUnsupervised::<f32>::fit(&lle, &x32).unwrap().embedding
    );
    let mds = ClassicalMds::new(2);
    assert_eq!(
        FitUnsupervised::<f64>::fit(&mds, &x).unwrap().embedding,
        FitUnsupervised::<f32>::fit(&mds, &x32).unwrap().embedding
    );
    let tsne = TSne::new(2).with_perplexity(5.0).with_seed(0);
    assert_eq!(
        FitUnsupervised::<f64>::fit(&tsne, &x).unwrap().embedding,
        FitUnsupervised::<f32>::fit(&tsne, &x32).unwrap().embedding
    );
}
