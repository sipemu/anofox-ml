# permutation_importance — sklearn parity

Issue: [#22](https://github.com/sipemu/rustml/issues/22)

## What

Model-agnostic permutation feature importance. Mirrors
`sklearn.inspection.permutation_importance`. For each feature, the column is
shuffled `n_repeats` times and the drop in score is recorded.

## Reference

`sklearn.inspection.permutation_importance` — sklearn 1.8.0.

## Golden test

- Generator: `test_harness/generators/gen_permutation_importance.py`
- Fixture:   `crates/rustml/tests/golden_data/permutation_importance.json`
- Rust test: `crates/rustml/tests/golden_permutation_importance.rs`

Fixture: 200×5 synthetic dataset, true coefficients `[5, 0, 2, 0, 0.3]`,
Ridge(alpha=1e-3) baseline scored by R², 50 permutation repeats.

We assert:
1. **R² baseline matches sklearn within 1e-6** (Ridge is closed-form so this is
   the strongest check available).
2. **Rank order of mean importances** matches sklearn exactly (argsort
   descending).
3. **Top-feature mean importance** is within 10% of sklearn's value. The mean
   converges with `1/sqrt(n_repeats)` and sklearn uses a different RNG order,
   so we don't demand tighter agreement.

## Differences from sklearn

- API takes a `Predict<f64>` impl, not a generic estimator with `score()`.
  The user supplies a scoring function explicitly.
- Single-threaded — sklearn parallelises across features when `n_jobs > 1`.
  Not currently a priority; the loop is small in practice.
- No support for sample weights (sklearn's `sample_weight` argument).

## Complexity

- `permutation_importance(estimator, X, y, n_repeats)`: each feature is shuffled `n_repeats` times and the estimator is scored on the permuted X.
- Total: **O(p · n_repeats · cost(predict on n samples))**.
- Parallelised across features via rayon (no inter-feature dependency).
- Memory: **O(n·p)** for the working copy of X.
