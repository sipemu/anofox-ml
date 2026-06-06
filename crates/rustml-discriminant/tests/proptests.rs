//! Property tests for LDA / QDA.

use ndarray::{Array1, Array2};
use proptest::prelude::*;
use rustml_core::{Fit, Predict, PredictProba};
use rustml_discriminant::{LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis};

/// Generate a synthetic 2-class problem with well-separated classes and
/// `n_per_class` samples per class.
fn two_class(n_per_class: usize, offset: f64, jitter: u64) -> (Array2<f64>, Array1<f64>) {
    let n = 2 * n_per_class;
    let mut x = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);
    let mut s = jitter | 1;
    for i in 0..n {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let jx = ((s >> 16) as f64 / u64::MAX as f64 - 0.5) * 0.4;
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let jy = ((s >> 16) as f64 / u64::MAX as f64 - 0.5) * 0.4;
        if i < n_per_class {
            x[[i, 0]] = jx;
            x[[i, 1]] = jy;
            y[i] = 0.0;
        } else {
            x[[i, 0]] = offset + jx;
            x[[i, 1]] = offset + jy;
            y[i] = 1.0;
        }
    }
    (x, y)
}

proptest! {
    /// LDA predict_proba columns sum to 1.0 per row.
    #[test]
    fn lda_predict_proba_sums_to_one(
        n_per in 5_usize..30,
        offset in 3.0_f64..=8.0,
        seed in 0u64..1_000_000,
    ) {
        let (x, y) = two_class(n_per, offset, seed);
        let fitted = LinearDiscriminantAnalysis::new().fit(&x, &y).unwrap();
        let p = fitted.predict_proba(&x).unwrap();
        for i in 0..p.nrows() {
            let s: f64 = (0..p.ncols()).map(|c| p[[i, c]]).sum();
            prop_assert!((s - 1.0).abs() < 1e-9, "row {} sum = {}", i, s);
        }
    }

    /// LDA predictions on training data of two well-separated classes are
    /// always perfect (Bayes rule on Gaussian-like classes is provably so).
    #[test]
    fn lda_perfect_on_well_separated(
        n_per in 5_usize..30,
        seed in 0u64..1_000_000,
    ) {
        let (x, y) = two_class(n_per, 10.0, seed);
        let fitted = LinearDiscriminantAnalysis::new().fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        for (p, t) in preds.iter().zip(y.iter()) {
            prop_assert_eq!(p, t);
        }
    }

    /// QDA predict_proba columns sum to 1.
    #[test]
    fn qda_predict_proba_sums_to_one(
        n_per in 6_usize..25,
        offset in 4.0_f64..=8.0,
        seed in 0u64..1_000_000,
    ) {
        let (x, y) = two_class(n_per, offset, seed);
        let fitted = QuadraticDiscriminantAnalysis::new().fit(&x, &y).unwrap();
        let p = fitted.predict_proba(&x).unwrap();
        for i in 0..p.nrows() {
            let s: f64 = (0..p.ncols()).map(|c| p[[i, c]]).sum();
            prop_assert!((s - 1.0).abs() < 1e-9);
        }
    }
}
