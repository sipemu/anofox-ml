# LARS / LassoLars — sklearn parity

Issue: [#2](https://github.com/sipemu/anofox-ml/issues/2) (partial — LassoLarsIC pending)

## What

Least Angle Regression — walks the L1 regularisation path piecewise-linearly.
At each step a new feature joins the active set (LARS); in LassoLars a feature
can also leave when its coefficient crosses zero. Feature columns are
internally unit-normalised for stability.

API: `Lars::new(k)` / `Lars::lasso(k)` in `anofox-ml-regression::lars`.

## Reference

`sklearn.linear_model.Lars`, `LassoLars` — sklearn 1.8.0.

## Golden test

- Generator: `test_harness/generators/gen_lars.py`
- Fixture:   `crates/anofox-ml/tests/golden_data/lars.json`
- Rust test: `crates/anofox-ml/tests/golden_lars.rs`

60×6 problem with 3 non-zero true coefficients at `{1, 3, 5}`. LARS recovers
the same active set as sklearn; training-set R² within 0.10 of sklearn's
value (LARS stops at step `k` without refitting OLS, so neither implementation
hits an interpolation R²).

## Not yet implemented

- `LassoLarsIC` (AIC/BIC criterion selection).
- `LarsCV` / `LassoLarsCV`.
- Stopping by `alpha` instead of `n_nonzero_coefs` for LassoLars.

## Complexity

- LARS / LassoLars: each LARS iteration adds (or removes) one variable. Cost per step is **O(n·p + p²)** for Gram updates + Cholesky downdate.
- Total: **O(k · (np + p²))** for k iterations, where k ≤ min(n, p) (the active-set size).
- LassoLarsIC: same cost as LassoLars plus an O(k) model-selection sweep over the path.
- Memory: **O(p²)** for the Gram matrix scratch, **O(n·p)** for the data.
