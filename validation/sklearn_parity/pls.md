# PLSRegression — sklearn parity

Issue: [#11](https://github.com/sipemu/anofox-ml/issues/11) (partial — PLSCanonical / CCA pending)

## What

PLS1 (1-D `y`) regression via NIPALS. Standardizes X and y to unit variance,
fits `n_components` latent variables, returns coefficients in the original
scale.

API: `PlsRegression` in `anofox-ml-preprocessing::pls`.

## Reference

`sklearn.cross_decomposition.PLSRegression` — sklearn 1.8.0.

## Golden test

- Generator: `test_harness/generators/gen_pls.py`
- Fixture:   `crates/anofox-ml/tests/golden_data/pls.json`
- Rust test: `crates/anofox-ml/tests/golden_pls.rs`

80×6 problem with multiple non-zero coefficients, 3 PLS components. NIPALS is
deterministic — predictions match sklearn element-wise to `1e-6`.

## Not yet implemented

- `PLSCanonical`, `CCA`, `PLSSVD` — issue #11 still tracks these.
- 2-D `y` (PLS2 / PLS-DA) — current code is PLS1 only.
- `transform` / `x_scores_` / `y_scores_` exposed publicly.

## Complexity

- Time per component: **O(n · p · q + n · p² + n · q²)** for PLSCanonical/PLSRegression NIPALS power iterations until convergence.
- Total: **O(k · iter · (np + p² + q²))** for `k = n_components`.
- Memory: **O(n·p + n·q + p·q)** for scores, loadings, weight matrices.
