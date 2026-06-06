# MultiOutputRegressor — sklearn parity

Issue: [#9](https://github.com/sipemu/rustml/issues/9)

## What

Meta-estimator that fits one independent regressor per output column. Mirrors
`sklearn.multioutput.MultiOutputRegressor`. Lives in `rustml-core` so any
crate can use it.

## Reference

`sklearn.multioutput.MultiOutputRegressor` — sklearn 1.8.0.

## Golden test

- Generator: `test_harness/generators/gen_multi_output.py`
- Fixture:   `crates/rustml/tests/golden_data/multi_output.json`
- Rust test: `crates/rustml/tests/golden_multi_output.rs`

Fixture: 50×4 features, 3-output linear target with low noise, Ridge(alpha=0.5)
per output. Predictions match sklearn element-wise to `1e-6` (Ridge is
closed-form).

## API note

Existing rustml estimators take 1-D `y` (`Fit<F>::fit(x, y)` with
`y: Array1<F>`). `MultiOutputRegressor::fit_2d` / `predict_2d` are the 2-D
entry points; the inner estimator continues to use the 1-D contract.

## Not yet implemented

- `MultiOutputClassifier`, `RegressorChain`, `ClassifierChain` — same issue
  (#9) still tracks these. Filing a follow-up issue would be reasonable; for
  now they're called out as gaps in the issue thread.

## Complexity

- MultiOutputRegressor/Classifier: independently fits one base estimator per output → **O(n_outputs · cost(base on n_samples))**.
- Embarrassingly parallel across outputs.
- RegressorChain: sequential — output i is conditional on outputs 0..i−1 already predicted, so each step's feature dimensionality grows by 1 → **O(Σ_i cost(base on n_samples, p+i))**.
- Memory: each fitted base estimator is independently stored; chain also requires intermediate predictions.
