# OrthogonalMatchingPursuit — sklearn parity

Issue: [#3](https://github.com/sipemu/rustml/issues/3)

## What

Greedy sparse regression. At each step pick the feature most correlated with
the current residual, add it to the active set, refit OLS on the active set.
Stop after `n_nonzero_coefs` features or when residual norm < `tol`.

## Reference

`sklearn.linear_model.OrthogonalMatchingPursuit` — sklearn 1.8.0.

## Golden test

- Generator: `test_harness/generators/gen_omp.py`
- Fixture:   `crates/rustml/tests/golden_data/omp.json`
- Rust test: `crates/rustml/tests/golden_omp.rs`

80×8 problem with 3 non-zero true coefficients at indices `{1, 3, 6}`. Test
asserts both the recovered active set and predictions match sklearn (`1e-6`
element-wise — both do the same OLS refit on the active set).

## Differences from sklearn

- `OrthogonalMatchingPursuitCV` (CV-selected `n_nonzero_coefs`) is not
  implemented.
- `precompute=False` only — sklearn can precompute `X'X` for repeated calls.
- No `return_path` (full path of fits).
