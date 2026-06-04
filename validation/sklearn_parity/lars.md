# LARS / LassoLars — sklearn parity

Issue: [#2](https://github.com/sipemu/rustml/issues/2) (partial — LassoLarsIC pending)

## What

Least Angle Regression — walks the L1 regularisation path piecewise-linearly.
At each step a new feature joins the active set (LARS); in LassoLars a feature
can also leave when its coefficient crosses zero. Feature columns are
internally unit-normalised for stability.

API: `Lars::new(k)` / `Lars::lasso(k)` in `rustml-regression::lars`.

## Reference

`sklearn.linear_model.Lars`, `LassoLars` — sklearn 1.8.0.

## Golden test

- Generator: `test_harness/generators/gen_lars.py`
- Fixture:   `crates/rustml/tests/golden_data/lars.json`
- Rust test: `crates/rustml/tests/golden_lars.rs`

60×6 problem with 3 non-zero true coefficients at `{1, 3, 5}`. LARS recovers
the same active set as sklearn; training-set R² within 0.10 of sklearn's
value (LARS stops at step `k` without refitting OLS, so neither implementation
hits an interpolation R²).

## Not yet implemented

- `LassoLarsIC` (AIC/BIC criterion selection).
- `LarsCV` / `LassoLarsCV`.
- Stopping by `alpha` instead of `n_nonzero_coefs` for LassoLars.
