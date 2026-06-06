mod common;

use anofox_ml::prelude::*;
use common::{assert_close, json_to_array1, json_to_array2, load_golden_data};

const TOL: f64 = 1e-4;

#[test]
fn test_golden_multinomial_nb() {
    let cases = load_golden_data("naive_bayes_extra.json");
    let case = cases
        .iter()
        .find(|c| c["name"] == "multinomial_nb")
        .unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let expected_preds = json_to_array1(&case["predictions"]);
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

    let fitted = MultinomialNB::new().fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    // Check predictions match sklearn
    for (i, (p, e)) in preds.iter().zip(expected_preds.iter()).enumerate() {
        assert_close(*p, *e, 1e-10, &format!("multinomial_nb pred[{}]", i));
    }

    // Check probabilities match sklearn
    let proba = fitted.predict_proba(&x).unwrap();
    for i in 0..x.nrows() {
        for c in 0..proba.ncols() {
            assert_close(
                proba[[i, c]],
                expected_proba[i][c],
                TOL,
                &format!("multinomial_nb proba[{},{}]", i, c),
            );
        }
    }
}

#[test]
fn test_golden_bernoulli_nb() {
    let cases = load_golden_data("naive_bayes_extra.json");
    let case = cases.iter().find(|c| c["name"] == "bernoulli_nb").unwrap();

    let x = json_to_array2(&case["X"]);
    let y = json_to_array1(&case["y"]);
    let expected_preds = json_to_array1(&case["predictions"]);
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

    let fitted = BernoulliNB::new().fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    for (i, (p, e)) in preds.iter().zip(expected_preds.iter()).enumerate() {
        assert_close(*p, *e, 1e-10, &format!("bernoulli_nb pred[{}]", i));
    }

    let proba = fitted.predict_proba(&x).unwrap();
    for i in 0..x.nrows() {
        for c in 0..proba.ncols() {
            assert_close(
                proba[[i, c]],
                expected_proba[i][c],
                TOL,
                &format!("bernoulli_nb proba[{},{}]", i, c),
            );
        }
    }
}
