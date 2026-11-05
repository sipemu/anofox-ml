//! Golden tests for KernelDensity and FactorAnalysis (#113, batch 6)
//! against scikit-learn 1.8.0 (`test_harness/generators/gen_density_fa.py`).

mod common;

use anofox_ml_core::{FitUnsupervised, Transform};
use anofox_ml_neighbors::{Bandwidth, DensityKernel, KernelDensity};
use anofox_ml_preprocessing::{FactorAnalysis, FactorRotation};
use common::{
    assert_array1_close, assert_array2_close, assert_close, json_to_array1, json_to_array2,
};
use ndarray::Array2;
use serde_json::Value;

fn golden() -> Value {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden_data/density_fa.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn kernel_density_matches_sklearn() {
    for c in golden()["kde"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let kernel = match c["kernel"].as_str().unwrap() {
            "gaussian" => DensityKernel::Gaussian,
            "tophat" => DensityKernel::Tophat,
            "epanechnikov" => DensityKernel::Epanechnikov,
            "exponential" => DensityKernel::Exponential,
            "linear" => DensityKernel::Linear,
            _ => DensityKernel::Cosine,
        };
        let bw = match &c["bandwidth"] {
            Value::String(s) if s == "scott" => Bandwidth::Scott,
            Value::String(_) => Bandwidth::Silverman,
            v => Bandwidth::Fixed(v.as_f64().unwrap()),
        };
        let w = (!c["sample_weight"].is_null()).then(|| json_to_array1(&c["sample_weight"]));
        let f = KernelDensity::new(bw, kernel)
            .fit_weighted(json_to_array2(&c["X"]).view(), w.as_ref().map(|w| w.view()))
            .unwrap();
        assert_close(
            f.bandwidth(),
            c["bandwidth_"].as_f64().unwrap(),
            1e-14,
            name,
        );
        let got = f
            .score_samples(json_to_array2(&c["X_test"]).view())
            .unwrap();
        let expected: Vec<f64> = c["score_samples"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap_or(f64::NEG_INFINITY))
            .collect();
        for (i, (&a, &b)) in got.iter().zip(&expected).enumerate() {
            if b == f64::NEG_INFINITY {
                assert_eq!(a, b, "{name}[{i}]");
            } else {
                assert!(
                    (a - b).abs() <= 1e-10 * b.abs().max(1.0),
                    "{name}[{i}]: {a} vs {b}"
                );
            }
        }
    }
}

/// Flip each row of `ours` to agree in sign with `theirs`.
fn align_rows(ours: &Array2<f64>, theirs: &Array2<f64>) -> Array2<f64> {
    let mut out = ours.clone();
    for (mut r, t) in out.rows_mut().into_iter().zip(theirs.rows()) {
        if r.dot(&t) < 0.0 {
            r.mapv_inplace(|v| -v);
        }
    }
    out
}

#[test]
fn factor_analysis_matches_sklearn() {
    for c in golden()["fa"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let fa = FactorAnalysis {
            n_components: c["n_components"].as_u64().map(|v| v as usize),
            rotation: match c["rotation"].as_str() {
                Some("varimax") => Some(FactorRotation::Varimax),
                Some(_) => Some(FactorRotation::Quartimax),
                None => None,
            },
            ..Default::default()
        };
        let f = fa.fit(&json_to_array2(&c["X"])).unwrap();
        let xt = json_to_array2(&c["X_test"]);
        assert_eq!(f.n_iter() as u64, c["n_iter"].as_u64().unwrap(), "{name}");
        assert_array1_close(
            &ndarray::Array1::from(f.loglike().to_vec()),
            &json_to_array1(&c["loglike"]),
            1e-8,
            name,
        );
        assert_array1_close(
            f.noise_variance(),
            &json_to_array1(&c["noise_variance"]),
            1e-9,
            name,
        );
        assert_array2_close(
            &f.covariance(),
            &json_to_array2(&c["covariance"]),
            1e-9,
            name,
        );
        assert_array2_close(
            &f.precision().unwrap(),
            &json_to_array2(&c["precision"]),
            1e-8,
            name,
        );
        assert_array1_close(
            &f.score_samples(xt.view()).unwrap(),
            &json_to_array1(&c["score_samples"]),
            1e-8,
            name,
        );
        let theirs = json_to_array2(&c["components"]);
        assert_array2_close(&align_rows(f.components(), &theirs), &theirs, 1e-7, name);
        let t = f.transform(&xt).unwrap();
        let tt = json_to_array2(&c["transform"]);
        let flipped = align_rows(&t.t().to_owned(), &tt.t().to_owned())
            .t()
            .to_owned();
        assert_array2_close(&flipped, &tt, 1e-7, name);
    }
}
