//! PLSCanonical and PLSSVD against scikit-learn 1.8.0
//! (`test_harness/generators/gen_pls.py`).

mod common;

use anofox_ml_preprocessing::{PlsCanonical, PlsSvd};
use common::{json_to_array2, load_golden_file};
use ndarray::Array2;

fn close(ours: &Array2<f64>, expected: &Array2<f64>, what: &str) {
    assert_eq!(ours.dim(), expected.dim(), "{what}");
    for (o, e) in ours.iter().zip(expected) {
        assert!((o - e).abs() < 1e-10, "{what}: {o} vs {e}");
    }
}

#[test]
fn pls_canonical_and_svd_match_scikit_learn() {
    let data = load_golden_file("pls_cross.json");
    let (x, y) = (json_to_array2(&data["X"]), json_to_array2(&data["Y"]));
    let (xq, yq) = (json_to_array2(&data["Xq"]), json_to_array2(&data["Yq"]));

    let m = &data["models"]["PLSCanonical"];
    let f = PlsCanonical::new(2).fit(&x, &y).unwrap();
    close(
        &f.x_weights,
        &json_to_array2(&m["x_weights"]),
        "PLSCanonical x_weights",
    );
    close(
        &f.y_weights,
        &json_to_array2(&m["y_weights"]),
        "PLSCanonical y_weights",
    );
    close(
        &f.transform_x(&xq).unwrap(),
        &json_to_array2(&m["x_scores"]),
        "PLSCanonical x scores",
    );
    close(
        &f.transform_y(&yq).unwrap(),
        &json_to_array2(&m["y_scores"]),
        "PLSCanonical y scores",
    );

    let m = &data["models"]["PLSSVD"];
    let f = PlsSvd::new(2).fit(&x, &y).unwrap();
    close(
        &f.x_weights,
        &json_to_array2(&m["x_weights"]),
        "PLSSVD x_weights",
    );
    close(
        &f.y_weights,
        &json_to_array2(&m["y_weights"]),
        "PLSSVD y_weights",
    );
    close(
        &f.transform_x(&xq).unwrap(),
        &json_to_array2(&m["x_scores"]),
        "PLSSVD x scores",
    );
    close(
        &f.transform_y(&yq).unwrap(),
        &json_to_array2(&m["y_scores"]),
        "PLSSVD y scores",
    );
}
