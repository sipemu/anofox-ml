//! Golden test for GaussianProcessRegressor against sklearn 1.8.0.

mod common;

use common::{assert_array1_close, json_to_array1, json_to_array2, load_golden_data};
use rustml::core::{Fit, Predict};
use rustml_gaussian_process::{GaussianProcessRegressor, GpKernel};

#[test]
fn test_gp_mean_matches_sklearn() {
    let cases = load_golden_data("gp.json");
    let case = &cases[0];
    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let xq = json_to_array2(&case["Xq"]);
    let length_scale = case["length_scale"].as_f64().unwrap();
    let signal_var = case["signal_var"].as_f64().unwrap();
    let alpha = case["alpha"].as_f64().unwrap();
    let sk_pred = json_to_array1(&case["sklearn_pred"]);
    let sk_std = json_to_array1(&case["sklearn_std"]);

    let gp = GaussianProcessRegressor::new(GpKernel::Rbf { length_scale, signal_var })
        .with_alpha(alpha);
    let fitted = gp.fit(&x, &y).unwrap();
    let mean = fitted.predict(&xq).unwrap();
    let std = fitted.predict_std(&xq).unwrap();

    // Closed-form Cholesky solve: mean should match sklearn to 1e-6.
    assert_array1_close(&mean, &sk_pred, 1e-6, "gp mean");
    // Std involves the same factor; expect tight match.
    for i in 0..std.len() {
        assert!(
            (std[i] - sk_std[i]).abs() < 1e-4,
            "std[{i}]: {} vs {}", std[i], sk_std[i]
        );
    }
}
