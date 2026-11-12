//! Golden tests for the kernel approximation samplers (#257) against
//! scikit-learn 1.8.0 (`test_harness/generators/gen_kernel_samplers.py`).
//! The random ones draw from numpy's `RandomState`, so they are compared
//! draw for draw.

mod common;

use anofox_ml_core::{FitUnsupervised, Transform};
use anofox_ml_preprocessing::{AdditiveChi2Sampler, PolynomialCountSketch, SkewedChi2Sampler};
use common::{assert_array1_close, assert_array2_close, json_to_array1, json_to_array2};
use serde_json::Value;

fn golden() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden_data/kernel_samplers.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn seed(p: &Value) -> u32 {
    p["random_state"].as_u64().unwrap() as u32
}

#[test]
fn additive_chi2_sampler_matches_sklearn() {
    for c in golden()["additive"].as_array().unwrap() {
        let p = &c["params"];
        let est = AdditiveChi2Sampler {
            sample_steps: p["sample_steps"].as_u64().map_or(2, |v| v as usize),
            sample_interval: p["sample_interval"].as_f64(),
        };
        let x = json_to_array2(&c["X"]);
        let out = est.fit(&x).unwrap().transform(&x).unwrap();
        assert_array2_close(
            &out,
            &json_to_array2(&c["transform"]),
            1e-12,
            &format!("{p}"),
        );
    }
}

#[test]
fn skewed_chi2_sampler_matches_sklearn() {
    for c in golden()["skewed"].as_array().unwrap() {
        let p = &c["params"];
        let est = SkewedChi2Sampler {
            skewedness: p["skewedness"].as_f64().unwrap_or(1.0),
            n_components: p["n_components"].as_u64().map_or(100, |v| v as usize),
            seed: seed(p),
        };
        let fitted = est.fit(&json_to_array2(&c["X"])).unwrap();
        let ctx = format!("{p}");
        assert_array2_close(
            fitted.random_weights(),
            &json_to_array2(&c["random_weights"]),
            1e-12,
            &ctx,
        );
        assert_array1_close(
            fitted.random_offset(),
            &json_to_array1(&c["random_offset"]),
            1e-12,
            &ctx,
        );
        let out = fitted.transform(&json_to_array2(&c["X_test"])).unwrap();
        assert_array2_close(&out, &json_to_array2(&c["transform"]), 1e-10, &ctx);
    }
}

#[test]
fn polynomial_count_sketch_matches_sklearn() {
    for c in golden()["sketch"].as_array().unwrap() {
        let p = &c["params"];
        let est = PolynomialCountSketch {
            gamma: p["gamma"].as_f64().unwrap_or(1.0),
            degree: p["degree"].as_u64().map_or(2, |v| v as usize),
            coef0: p["coef0"].as_f64().unwrap_or(0.0),
            n_components: p["n_components"].as_u64().unwrap() as usize,
            seed: seed(p),
        };
        let x = json_to_array2(&c["X"]);
        let fitted = est.fit(&x).unwrap();
        let ctx = format!("{p}");
        assert_array2_close(
            &fitted.index_hash().mapv(|v| v as f64),
            &json_to_array2(&c["index_hash"]),
            1e-12,
            &ctx,
        );
        assert_array2_close(
            fitted.bit_hash(),
            &json_to_array2(&c["bit_hash"]),
            1e-12,
            &ctx,
        );
        // sklearn multiplies the sketches' FFTs; this is the direct
        // circular convolution, equal up to the FFT's rounding.
        let out = fitted.transform(&x).unwrap();
        assert_array2_close(&out, &json_to_array2(&c["transform"]), 1e-10, &ctx);
    }
}
