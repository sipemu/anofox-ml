//! Fit a RandomForestClassifier on a small synthetic 3-class dataset and
//! evaluate train/test accuracy.

use ndarray::{Array1, Array2};
use rustml::prelude::*;

fn main() {
    // 3 well-separated clusters of 20 points each in 4 features.
    let n_per = 20;
    let mut x = Array2::<f64>::zeros((3 * n_per, 4));
    let mut y = Array1::<f64>::zeros(3 * n_per);
    let centres = [
        [0.0, 0.0, 0.0, 0.0],
        [5.0, 5.0, 0.0, 0.0],
        [0.0, 0.0, 5.0, 5.0],
    ];
    for (cls, c) in centres.iter().enumerate() {
        for i in 0..n_per {
            let row = cls * n_per + i;
            let t = (i as f64) * 0.1;
            for j in 0..4 {
                x[[row, j]] = c[j] + t.sin() * 0.3;
            }
            y[row] = cls as f64;
        }
    }
    // 80/20 split.
    let n = x.nrows();
    let split = (n * 4) / 5;
    let x_train = x.slice(ndarray::s![..split, ..]).to_owned();
    let y_train = y.slice(ndarray::s![..split]).to_owned();
    let x_test = x.slice(ndarray::s![split.., ..]).to_owned();
    let y_test = y.slice(ndarray::s![split..]).to_owned();

    let rf = RandomForestClassifier::new(50).with_seed(0);
    let fitted = Fit::fit(&rf, &x_train, &y_train).unwrap();
    let pred_train = fitted.predict(&x_train).unwrap();
    let pred_test = fitted.predict(&x_test).unwrap();

    let acc = |p: &Array1<f64>, y: &Array1<f64>| -> f64 {
        let n = p.len();
        let correct = p.iter().zip(y.iter()).filter(|(a, b)| a == b).count();
        correct as f64 / n as f64
    };
    println!("Train accuracy: {:.4}", acc(&pred_train, &y_train));
    println!("Test accuracy:  {:.4}", acc(&pred_test, &y_test));
}
