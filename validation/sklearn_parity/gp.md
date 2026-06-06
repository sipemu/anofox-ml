# GaussianProcessRegressor — sklearn parity

Issue: [#12](https://github.com/sipemu/anofox-ml/issues/12) (partial — no kernel learning, no GP classifier)

## What

New crate `anofox-ml-gaussian-process`. Closed-form GP regression with a fixed
RBF kernel `σ² exp(-||x-x'||² / (2 ℓ²))`, additive noise `α` on the diagonal,
optional `normalize_y`. Posterior mean via Cholesky solve, posterior std via
forward-substitution per query point.

## Reference

`sklearn.gaussian_process.GaussianProcessRegressor` — sklearn 1.8.0.

## Golden test

- Generator: `test_harness/generators/gen_gp.py`
- Fixture:   `crates/anofox-ml/tests/golden_data/gp.json`
- Rust test: `crates/anofox-ml/tests/golden_gp.rs`

20 training samples on `y = sin(x) + ε`, 40 query points. sklearn is run
with `optimizer=None` to fix the kernel (no hyperparameter learning) and
`normalize_y=False` so the math matches ours exactly. Posterior mean matches
sklearn to `1e-6`, posterior std to `1e-4`.

## Not yet implemented

- **GaussianProcessClassifier** — still pending in #12.
- Kernel hyperparameter learning (log marginal likelihood + L-BFGS).
- Composable kernels (Matern, RationalQuadratic, sums, products, WhiteKernel).
- Multi-output GP.

## Complexity

- `fit`: O(n³) Cholesky on the n×n kernel matrix.
- `predict` mean: O(n_train · n_test) for the kernel matrix + O(n_train · n_test) for the dot product.
- `predict_std`: additional O(n_train² · n_test) for the per-query forward solve. Could be reduced to O(n_train · n_test) by precomputing `L⁻¹ K_train_test`.
- Memory: O(n²) for the Cholesky factor.
- Hard ceiling: ~5,000 samples before n³ wall-clock dominates.
