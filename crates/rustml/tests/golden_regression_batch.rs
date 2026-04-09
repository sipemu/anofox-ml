//! Golden tests for regression models against sklearn 1.8.0.
//!
//! Validates OLS, Ridge, Lasso, ElasticNet, LogisticRegression, HuberRegressor,
//! and SGDRegressor predictions + fitted coefficients.

mod common;

use common::{assert_close, json_to_array1, json_to_array2, load_golden_data};
use ndarray::Array1;
use rustml::prelude::*;

/// Compute RMSE for behavioral equivalence checks.
fn rmse(pred: &Array1<f64>, target: &Array1<f64>) -> f64 {
    let mse: f64 = pred
        .iter()
        .zip(target.iter())
        .map(|(&p, &t)| (p - t).powi(2))
        .sum::<f64>()
        / pred.len() as f64;
    mse.sqrt()
}

#[test]
fn test_golden_ols_matches_sklearn() {
    let cases = load_golden_data("regression_batch.json");
    let case = cases.iter().find(|c| c["name"] == "ols").unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let sklearn_intercept = case["intercept"].as_f64().unwrap();
    let sklearn_coef: Vec<f64> = case["coef"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    let fitted = OlsRegressor::new().fit(&x, &y).unwrap();
    let coef = fitted.coefficients();
    let intercept = fitted.intercept().unwrap_or(0.0);

    // OLS is a closed-form solution, so these should match exactly.
    assert_close(intercept, sklearn_intercept, 1e-8, "ols intercept");
    for (i, (&c, &s)) in coef.iter().zip(sklearn_coef.iter()).enumerate() {
        assert_close(c, s, 1e-8, &format!("ols coef[{}]", i));
    }

    let preds = fitted.predict(&x).unwrap();
    let expected = json_to_array1(&case["predictions"]);
    for (i, (&p, &e)) in preds.iter().zip(expected.iter()).enumerate() {
        assert_close(p, e, 1e-8, &format!("ols pred[{}]", i));
    }
}

#[test]
fn test_golden_ridge_close_to_sklearn() {
    let cases = load_golden_data("regression_batch.json");
    let case = cases.iter().find(|c| c["name"] == "ridge").unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let expected = json_to_array1(&case["predictions"]);

    let fitted = RidgeRegressor::new().with_lambda(1.0).fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    // Ridge is closed-form but may use slightly different solvers;
    // allow small tolerance.
    let err = rmse(&preds, &expected);
    assert!(
        err < 0.5,
        "Ridge predictions deviate from sklearn: rmse={:.4}",
        err
    );
}

#[test]
fn test_golden_lasso_close_to_sklearn() {
    let cases = load_golden_data("regression_batch.json");
    let case = cases.iter().find(|c| c["name"] == "lasso").unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let expected = json_to_array1(&case["predictions"]);

    let fitted = LassoRegressor::new().with_lambda(0.1).fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    // Lasso uses coordinate descent — allow small tolerance.
    let err = rmse(&preds, &expected);
    assert!(
        err < 0.5,
        "Lasso predictions deviate from sklearn: rmse={:.4}",
        err
    );
}

#[test]
fn test_golden_elastic_net_close_to_sklearn() {
    let cases = load_golden_data("regression_batch.json");
    let case = cases.iter().find(|c| c["name"] == "elastic_net").unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let expected = json_to_array1(&case["predictions"]);

    // Note: anofox uses `lambda` for overall strength and `alpha` for the L1/L2 mix,
    // whereas sklearn uses `alpha` (strength) and `l1_ratio` (mix).
    let fitted = ElasticNetRegressor::new()
        .with_lambda(0.1)
        .with_alpha(0.5)
        .fit(&x, &y)
        .unwrap();
    let preds = fitted.predict(&x).unwrap();

    let err = rmse(&preds, &expected);
    assert!(
        err < 0.5,
        "ElasticNet predictions deviate from sklearn: rmse={:.4}",
        err
    );
}

#[test]
fn test_golden_logistic_regression_matches_sklearn() {
    let cases = load_golden_data("regression_batch.json");
    let case = cases.iter().find(|c| c["name"] == "logistic_regression").unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let expected_preds = json_to_array1(&case["predictions"]);

    let fitted = LogisticRegressor::new()
        .with_c(1e6)
        .with_max_iter(1000)
        .fit(&x, &y)
        .unwrap();

    let preds = fitted.predict(&x).unwrap();

    // Predictions (class labels) should match exactly on this well-separated data.
    for (i, (&p, &e)) in preds.iter().zip(expected_preds.iter()).enumerate() {
        assert_close(p, e, 1e-10, &format!("logreg pred[{}]", i));
    }

    // Probabilities should be close (same solver family).
    let proba = fitted.predict_proba(&x).unwrap();
    let expected_proba: Vec<Vec<f64>> = case["predict_proba"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect()
        })
        .collect();
    for i in 0..x.nrows() {
        // sklearn returns [p0, p1]; our predict_proba returns p1 (class 1 probability)
        assert_close(
            proba[i],
            expected_proba[i][1],
            0.05,
            &format!("logreg proba[{}]", i),
        );
    }
}

#[test]
fn test_golden_huber_regressor_close_to_sklearn() {
    let cases = load_golden_data("regression_batch.json");
    let case = cases.iter().find(|c| c["name"] == "huber_regressor").unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let expected = json_to_array1(&case["predictions"]);

    let fitted = HuberRegressor::new()
        .with_epsilon(1.35)
        .with_alpha(0.0001)
        .fit(&x, &y)
        .unwrap();
    let preds = fitted.predict(&x).unwrap();

    let err = rmse(&preds, &expected);
    assert!(
        err < 0.5,
        "Huber predictions deviate from sklearn: rmse={:.4}",
        err
    );
}

#[test]
fn test_golden_sgd_regressor_learns_linear_trend() {
    let cases = load_golden_data("regression_batch.json");
    let case = cases.iter().find(|c| c["name"] == "sgd_regressor").unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);

    // SGD is stochastic — our implementation uses a different RNG so exact
    // match is impossible. Validate that we learn the underlying linear trend.
    let fitted = SgdRegressor::new()
        .with_max_iter(2000)
        .with_alpha(0.0001)
        .with_eta0(0.001)
        .with_seed(42)
        .fit(&x, &y)
        .unwrap();

    let preds = fitted.predict(&x).unwrap();

    // R² on training data should be reasonable
    let mean: f64 = y.iter().sum::<f64>() / y.len() as f64;
    let ss_res: f64 = preds
        .iter()
        .zip(y.iter())
        .map(|(&p, &t)| (p - t).powi(2))
        .sum();
    let ss_tot: f64 = y.iter().map(|&t| (t - mean).powi(2)).sum();
    let r2 = 1.0 - ss_res / ss_tot;
    assert!(r2 > 0.8, "SGDRegressor R² = {:.4}", r2);
}
