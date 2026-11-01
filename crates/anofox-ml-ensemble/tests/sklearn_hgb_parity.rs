//! sklearn 1.8 parity of the multinomial HistGradientBoostingClassifier: the
//! facade over the LGBM engine must use sklearn's softmax hessian p(1 - p),
//! not LightGBM's K / (K - 1) scaling.

use anofox_ml_core::Fit;
use anofox_ml_ensemble::HistGradientBoostingClassifier;
use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2};

#[test]
fn multiclass_matches_sklearn() {
    let n = 200;
    let mut x = Array2::zeros((n, 2));
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let x0 = ((i * 7) % 20) as f64;
        let x1 = ((i * 3) % 7) as f64;
        x[[i, 0]] = x0;
        x[[i, 1]] = x1;
        let mut v =
            0.5 * x0 + if x1 == 3.0 { 4.0 } else { 0.0 } + (((i * 37) % 11) as f64 - 5.0) / 10.0;
        if i % 17 == 0 {
            v += 25.0;
        }
        y[i] = if v < 4.0 {
            0.0
        } else if v < 8.0 {
            1.0
        } else {
            2.0
        };
    }
    let mut m = HistGradientBoostingClassifier::new();
    m.n_estimators = 5;
    m.learning_rate = 0.3;
    m.max_leaf_nodes = Some(4);
    m.min_samples_leaf = 5;
    let proba = m.fit(&x, &y).unwrap().predict_proba(&x).unwrap();

    // HistGradientBoostingClassifier(max_iter=5, learning_rate=0.3,
    // max_leaf_nodes=4, min_samples_leaf=5, early_stopping=False)
    let expected = [
        [0.92683966, 0.05100865, 0.02215169],
        [0.04285872, 0.77048973, 0.18665155],
    ];
    for (r, e) in [0, 13].into_iter().zip(expected) {
        for c in 0..3 {
            assert_abs_diff_eq!(proba[[r, c]], e[c], epsilon = 1e-7);
        }
    }
}
