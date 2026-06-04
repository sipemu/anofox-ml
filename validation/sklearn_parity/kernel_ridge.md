# KernelRidge — sklearn parity

Issue: [#6](https://github.com/sipemu/rustml/issues/6)

## What

Kernel ridge regression. Solves `(K + αI) c = y` via Cholesky and predicts
`K_test @ c`. Matches `sklearn.kernel_ridge.KernelRidge` for the linear, RBF,
and polynomial kernels. No intercept (sklearn also has no intercept by
default).

## Reference

`sklearn.kernel_ridge.KernelRidge` — sklearn 1.8.0.

## Golden test

- Generator: `test_harness/generators/gen_kernel_ridge.py`
- Fixture:   `crates/rustml/tests/golden_data/kernel_ridge.json`
- Rust test: `crates/rustml/tests/golden_kernel_ridge.rs`

Three cases, all 30×3 synthetic with a noisy non-linear target:

| Case | Kernel | α | Other |
|---|---|---|---|
| linear | linear | 0.5 | — |
| rbf_gamma0p5 | RBF | 0.1 | gamma=0.5 |
| poly_deg3 | polynomial | 1.0 | degree=3, gamma=1.0, coef0=1.0 |

All predictions match sklearn to within `1e-7` element-wise (Cholesky-based
closed-form solution agrees up to BLAS ordering).

## Differences from sklearn

- Polynomial kernel does not currently support a separate `gamma` parameter —
  uses the `(x·y + coef0)^degree` form, equivalent to sklearn with `gamma=1`.
  Test fixture pins `gamma=1` accordingly.
- Sigmoid, cosine, laplacian, chi-squared kernels not yet supported (the
  shared `SvmKernel` enum from `rustml-svm` only has linear / RBF / polynomial).
- No sample-weight support.

## Complexity

- `fit`: O(n³) Cholesky + O(n² · d) Gram matrix.
- `predict`: O(n_test · n_train · d) for the kernel matrix + O(n_test · n_train) for the dual product.
- Memory: O(n²) Cholesky factor (kept in `FittedKernelRidge`).
- Sample-weighted fit uses the `√W K √W` substitution; same asymptotic cost.
