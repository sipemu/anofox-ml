# BayesianRidge / ARDRegression — sklearn parity

Issue: [#1](https://github.com/sipemu/rustml/issues/1)

## What

Bayesian linear regression with evidence (type-II ML) hyperparameter updates.

- **BayesianRidge** ties all coefficient precisions `λ_j = λ`.
- **ARDRegression** allows per-feature `λ_j`, driving irrelevant features
  toward zero by lifting their precision to infinity. Features whose `λ_j`
  exceeds `threshold_lambda` are dropped from the model.

## Reference

`sklearn.linear_model.{BayesianRidge, ARDRegression}` — sklearn 1.8.0.

## Golden test

- Generator: `test_harness/generators/gen_bayesian_ridge.py`
- Fixture:   `crates/rustml/tests/golden_data/bayesian_ridge.json`
- Rust test: `crates/rustml/tests/golden_bayesian_ridge.rs`

BayesianRidge: predictions match sklearn within 2% relative tolerance on a
80×4 synthetic problem.

ARD: rather than matching predictions exactly we assert both implementations
agree on *which* features are relevant (|coef| < 0.1 for true-zero features,
|coef| > threshold for true-nonzero), and R² > 0.95 on the training set.

## Differences from sklearn

- No `compute_score` / log-marginal-likelihood track.
- No `return_std` on `predict`.
- Inverse of `S = (αX'X + diag(λ))` is computed via column-by-column
  Cholesky solves rather than a single SVD-based inverse.
