# TweedieRegressor / GammaRegressor — sklearn parity

Issue: [#7](https://github.com/sipemu/anofox-ml/issues/7)

## What

Tweedie GLM family covering Gaussian (`power=0`), Poisson (`power=1`),
compound Poisson-Gamma (`1 < power < 2`), Gamma (`power=2`), and
Inverse-Gaussian (`power=3`). Implemented as a thin wrapper around
`anofox_regression::TweedieRegressor`, plus a `GammaRegressor` convenience.

Mirrors `sklearn.linear_model.TweedieRegressor` and `GammaRegressor`.

## Reference

`sklearn.linear_model.TweedieRegressor`, `GammaRegressor` — sklearn 1.8.0.

## Golden test

- Generator: `test_harness/generators/gen_tweedie.py`
- Fixture:   `crates/anofox-ml/tests/golden_data/tweedie.json`
- Rust test: `crates/anofox-ml/tests/golden_tweedie.rs`

Two cases: Tweedie(power=1.5, sklearn α=0.5) and Gamma(power=2, sklearn α=0.1).
Predictions match sklearn within **1% relative tolerance** element-wise.

## Regularization scaling

sklearn's `alpha` is per-sample-normalized:

    loss = (1 / 2n) * Σ deviance(y_i, μ_i) + (α/2) * ||β||²

anofox-regression's `lambda` is on the un-normalized loss:

    loss = Σ deviance(y_i, μ_i) + λ * ||β||²

To get parity: **`λ = n * α`**. This is encoded in the fixture as
`anofox_lambda = n * sklearn_alpha`, and the test plugs that value into
`with_alpha`.

This is mildly leaky — if a user reads the sklearn docs and passes the same
number, they'll get a *much* less-regularized fit. A future iteration should
either expose the per-sample-normalized form natively or rename the parameter
to avoid the trap.

## Not yet implemented

- Identity / inverse links beyond what anofox supports.
- `link='auto'` mode — caller must pass `link_power` explicitly for non-defaults.

## Complexity

- TweedieRegressor / PoissonRegressor / GammaRegressor: GLM IRLS — each iteration solves a weighted-least-squares of cost **O(n·p² + p³)**.
- Typical convergence in 5–20 iterations.
- Total: **O((n·p² + p³) · iter)**.
- Memory: **O(n·p)** for design matrix + **O(p²)** for the Hessian factorisation.
