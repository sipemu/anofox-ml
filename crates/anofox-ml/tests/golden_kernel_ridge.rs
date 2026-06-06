//! Golden test for KernelRidge against sklearn 1.8.0.

mod common;

use anofox_ml::prelude::*;
use anofox_ml_svm::SvmKernel;
use common::{assert_array1_close, json_to_array1, json_to_array2, load_golden_data};

fn kernel_for(case: &serde_json::Value) -> SvmKernel {
    let name = case["kernel"].as_str().unwrap();
    match name {
        "linear" => SvmKernel::Linear,
        "rbf" => SvmKernel::Rbf {
            gamma: case["gamma"].as_f64().unwrap(),
        },
        "polynomial" => SvmKernel::Polynomial {
            degree: case["degree"].as_u64().unwrap() as usize,
            coef0: case["coef0"].as_f64().unwrap(),
        },
        _ => panic!("unknown kernel: {name}"),
    }
}

#[test]
fn test_golden_kernel_ridge() {
    let cases = load_golden_data("kernel_ridge.json");
    for case in &cases {
        let name = case["name"].as_str().unwrap();
        let x = json_to_array2(&case["X"]);
        let y = json_to_array1(&case["y"]);
        let alpha = case["alpha"].as_f64().unwrap();
        let kernel = kernel_for(case);
        let expected = json_to_array1(&case["predictions"]);

        let fitted = KernelRidge::new()
            .with_alpha(alpha)
            .with_kernel(kernel)
            .fit(&x, &y)
            .unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_array1_close(
            &preds,
            &expected,
            1e-7,
            &format!("kernel_ridge {name} predictions"),
        );
    }
}
