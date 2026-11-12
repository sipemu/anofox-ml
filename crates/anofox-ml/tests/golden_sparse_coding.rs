//! Golden tests for sparse coding (#258) against scikit-learn 1.8.0
//! (`test_harness/generators/gen_sparse_coding.py`). sklearn's
//! DictionaryLearning starts from LAPACK's SVD, whose signs are arbitrary,
//! and the problem is symmetric in each atom's sign (with its code's), so
//! atoms and codes are compared after aligning each atom's sign.

mod common;

use anofox_ml_core::{FitUnsupervised, Transform};
use anofox_ml_preprocessing::{
    CodingAlgorithm, CodingParams, DictionaryLearning, FitAlgorithm, SparseCoder,
};
use common::{assert_array2_close, assert_close, json_to_array2};
use ndarray::Array2;
use serde_json::Value;

fn golden() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden_data/sparse_coding.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn algorithm(name: &str) -> CodingAlgorithm {
    serde_json::from_value(Value::String(name.into())).unwrap()
}

fn coding(p: &Value, alpha_default: Option<f64>) -> CodingParams {
    CodingParams {
        algorithm: algorithm(p["transform_algorithm"].as_str().unwrap_or("omp")),
        n_nonzero_coefs: p["transform_n_nonzero_coefs"].as_u64().map(|v| v as usize),
        alpha: p["transform_alpha"].as_f64().or(alpha_default),
        split_sign: p["split_sign"].as_bool().unwrap_or(false),
        max_iter: 1000,
    }
}

#[test]
fn sparse_coder_matches_sklearn() {
    for c in golden()["coder"].as_array().unwrap() {
        let p = &c["params"];
        let est = SparseCoder {
            dictionary: json_to_array2(&c["dictionary"]),
            coding: coding(p, None),
        };
        let x = json_to_array2(&c["X"]);
        let out = est.fit(&x).unwrap().transform(&x).unwrap();
        // Lasso by coordinate descent stops at sklearn's tol=1e-8.
        let tol = if p["transform_algorithm"] == "lasso_cd" {
            1e-6
        } else {
            1e-12
        };
        assert_array2_close(&out, &json_to_array2(&c["transform"]), tol, &format!("{p}"));
    }
}

/// Each atom's sign, so that `ours * sign` points as sklearn's.
fn signs(ours: &Array2<f64>, sklearn: &Array2<f64>) -> Vec<f64> {
    (0..ours.nrows())
        .map(|k| {
            if ours.row(k).dot(&sklearn.row(k)) < 0.0 {
                -1.0
            } else {
                1.0
            }
        })
        .collect()
}

#[test]
fn dictionary_learning_matches_sklearn_up_to_atom_signs() {
    for c in golden()["learning"].as_array().unwrap() {
        let p = &c["params"];
        let alpha = p["alpha"].as_f64().unwrap();
        let cd = p["fit_algorithm"] == "cd";
        let est = DictionaryLearning {
            n_components: p["n_components"].as_u64().map(|v| v as usize),
            alpha,
            fit_algorithm: if cd {
                FitAlgorithm::Cd
            } else {
                FitAlgorithm::Lars
            },
            transform: coding(p, None),
            ..Default::default()
        };
        let x = json_to_array2(&c["X"]);
        let fitted = est.fit(&x).unwrap();
        let ctx = format!("{p}");
        // LARS: rounding builds up over the iterations. Coordinate descent:
        // sklearn stops each code at tol=1e-8 (~1e-7 from the optimum; ours
        // runs to 1e-14), and that compounds over the dictionary updates.
        let tol = if cd { 1e-3 } else { 1e-5 };
        let want = json_to_array2(&c["components"]);
        let s = signs(&fitted.components, &want);
        let flip = |a: &Array2<f64>, by_column: bool| {
            Array2::from_shape_fn(a.dim(), |(i, j)| {
                a[[i, j]] * s[if by_column { j } else { i }]
            })
        };
        assert_array2_close(&flip(&fitted.components, false), &want, tol, &ctx);
        assert_array2_close(
            &flip(&fitted.train_code, true),
            &json_to_array2(&c["code"]),
            tol,
            &ctx,
        );
        assert_eq!(fitted.n_iter as u64, c["n_iter"].as_u64().unwrap(), "{ctx}");
        assert_close(
            *fitted.error.last().unwrap(),
            c["error"]
                .as_array()
                .unwrap()
                .last()
                .unwrap()
                .as_f64()
                .unwrap(),
            tol,
            &ctx,
        );
        // The transform codes flip with the atoms (threshold, LassoLars, OMP).
        let out = fitted.transform(&x).unwrap();
        assert_array2_close(
            &flip(&out, true),
            &json_to_array2(&c["transform"]),
            tol,
            &ctx,
        );
    }
}
