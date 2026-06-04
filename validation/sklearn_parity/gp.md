# GaussianProcessRegressor — sklearn parity

Issue: [#12](https://github.com/sipemu/rustml/issues/12) (partial — no kernel learning, no GP classifier)

## What

New crate `rustml-gaussian-process`. Closed-form GP regression with a fixed
RBF kernel `σ² exp(-||x-x'||² / (2 ℓ²))`, additive noise `α` on the diagonal,
optional `normalize_y`. Posterior mean via Cholesky solve, posterior std via
forward-substitution per query point.

## Reference

`sklearn.gaussian_process.GaussianProcessRegressor` — sklearn 1.8.0.

## Golden test

- Generator: `test_harness/generators/gen_gp.py`
- Fixture:   `crates/rustml/tests/golden_data/gp.json`
- Rust test: `crates/rustml/tests/golden_gp.rs`

20 training samples on `y = sin(x) + ε`, 40 query points. sklearn is run
with `optimizer=None` to fix the kernel (no hyperparameter learning) and
`normalize_y=False` so the math matches ours exactly. Posterior mean matches
sklearn to `1e-6`, posterior std to `1e-4`.

## Not yet implemented

- **GaussianProcessClassifier** — still pending in #12.
- Kernel hyperparameter learning (log marginal likelihood + L-BFGS).
- Composable kernels (Matern, RationalQuadratic, sums, products, WhiteKernel).
- Multi-output GP.
