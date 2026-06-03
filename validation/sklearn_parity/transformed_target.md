# TransformedTargetRegressor — sklearn parity

Issue: [#8](https://github.com/sipemu/rustml/issues/8)

## What

Meta-estimator that applies a function `func` to the training target before
calling the inner regressor, and applies `inverse_func` to its predictions.
Mirrors `sklearn.compose.TransformedTargetRegressor` in its function-based form
(`func` / `inverse_func` arguments; no transformer object).

## Reference

`sklearn.compose.TransformedTargetRegressor` — sklearn 1.8.0.

## Golden test

- Generator: `test_harness/generators/gen_transformed_target.py`
- Fixture:   `crates/rustml/tests/golden_data/transformed_target.json`
- Rust test: `crates/rustml/tests/golden_transformed_target.rs`

The fixture trains a sklearn `Ridge(alpha=0.01)` wrapped with
`TransformedTargetRegressor(func=np.log, inverse_func=np.exp)` on a 50×4
dataset whose target is constructed multiplicatively
(`y = exp(Xβ + c) + noise`). The Rust test reproduces the setup with
`RidgeRegressor::new().with_lambda(0.01)` wrapped by
`TransformedTargetRegressor::new(inner, f64::ln, f64::exp)`.

Predictions match within `1e-6` element-wise (Ridge is closed-form; the only
divergence sources are the inner solver's intercept handling and floating-point
order).

## Differences from sklearn

- We accept `fn(f64) -> f64` rather than allowing arbitrary closures /
  transformer objects. A future iteration could generalise to a trait if a
  use case appears.
- `check_inverse` defaults to `true` (matches sklearn) and verifies the
  round-trip on up to 10 elements of `y`.
