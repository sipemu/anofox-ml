//! Golden tests for kernel approximation (#113, batch 5) against
//! scikit-learn 1.8.0 (`test_harness/generators/gen_kernel_approx.py`).

mod common;

use anofox_ml_core::{FitUnsupervised, Transform};
use anofox_ml_preprocessing::{
    johnson_lindenstrauss_min_dim, KernelCenterer, Nystroem, NystroemKernel,
};
use common::{assert_array2_close, json_to_array2};
use serde_json::Value;

fn golden() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden_data/kernel_approx.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn kernel_centerer_matches_sklearn() {
    let g = golden();
    let c = &g["centerer"];
    let f = KernelCenterer.fit(&json_to_array2(&c["K"])).unwrap();
    assert_array2_close(
        &f.transform(&json_to_array2(&c["K"])).unwrap(),
        &json_to_array2(&c["fit"]),
        1e-12,
        "fit",
    );
    assert_array2_close(
        &f.transform(&json_to_array2(&c["K_test"])).unwrap(),
        &json_to_array2(&c["test"]),
        1e-12,
        "test",
    );
}

#[test]
fn nystroem_gram_matches_sklearn() {
    for c in golden()["nystroem"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let p = &c["params"];
        let gamma = p["gamma"].as_f64();
        let kernel = match p["kernel"].as_str().unwrap() {
            "rbf" => NystroemKernel::Rbf { gamma },
            "laplacian" => NystroemKernel::Laplacian { gamma },
            "poly" => NystroemKernel::Polynomial {
                degree: p["degree"].as_f64().unwrap(),
                gamma,
                coef0: p["coef0"].as_f64().unwrap(),
            },
            _ => NystroemKernel::Linear,
        };
        let x = json_to_array2(&c["X"]);
        let f = Nystroem {
            kernel,
            n_components: 20,
            seed: 0,
        }
        .fit(&x)
        .unwrap();
        let z = f.transform(&x).unwrap();
        let zy = f.transform(&json_to_array2(&c["X_test"])).unwrap();
        assert_array2_close(&z.dot(&z.t()), &json_to_array2(&c["gram"]), 1e-8, name);
        assert_array2_close(
            &zy.dot(&z.t()),
            &json_to_array2(&c["gram_test"]),
            1e-8,
            name,
        );
    }
}

#[test]
fn johnson_lindenstrauss_matches_sklearn() {
    for c in golden()["jl"].as_array().unwrap() {
        let (n, eps) = (
            c["n"].as_u64().unwrap() as usize,
            c["eps"].as_f64().unwrap(),
        );
        assert_eq!(
            johnson_lindenstrauss_min_dim(n, eps).unwrap() as u64,
            c["dim"].as_u64().unwrap(),
            "{n} {eps}"
        );
    }
}
