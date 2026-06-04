//! Golden test for KernelPCA and NMF against sklearn 1.8.0.

mod common;

use common::{json_to_array1, json_to_array2, load_golden_data};
use rustml::core::{FitUnsupervised, Transform};
use rustml_preprocessing::{KernelPca, KpcaKernel, Nmf};

#[test]
fn test_kernel_pca_eigenvalues_match_sklearn() {
    let cases = load_golden_data("decomposition.json");
    let case = cases.iter().find(|c| c["name"] == "kpca_rbf").unwrap();
    let x = json_to_array2(&case["X"]);
    let k = case["n_components"].as_u64().unwrap() as usize;
    let gamma = case["gamma"].as_f64().unwrap();
    let sk_evals = json_to_array1(&case["sklearn_eigenvalues"]);

    let kpca = KernelPca::new(k, KpcaKernel::Rbf { gamma });
    let fitted = kpca.fit(&x).unwrap();

    // Eigenvalues (in descending order) should match sklearn to 1e-6.
    for i in 0..k {
        assert!(
            (fitted.eigenvalues[i] - sk_evals[i]).abs() < 1e-6,
            "eig[{i}]: {} vs {}", fitted.eigenvalues[i], sk_evals[i]
        );
    }
    let t = fitted.transform(&x).unwrap();
    assert_eq!(t.shape(), &[40, k]);
}

#[test]
fn test_nmf_reaches_low_reconstruction_error() {
    let cases = load_golden_data("decomposition.json");
    let case = cases.iter().find(|c| c["name"] == "nmf_3").unwrap();
    let x = json_to_array2(&case["X"]);
    let k = case["n_components"].as_u64().unwrap() as usize;
    let sk_err = case["sklearn_reconstruction_err"].as_f64().unwrap();

    let fitted = Nmf::new(k).fit(&x).unwrap();
    // Both reach a similar order-of-magnitude error.
    assert!(
        fitted.reconstruction_err() < 2.0 * sk_err,
        "rustml err {} vs sklearn {}",
        fitted.reconstruction_err(),
        sk_err
    );
}
