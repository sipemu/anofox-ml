# RansacRegressor / TheilSenRegressor — sklearn parity

Issue: [#4](https://github.com/sipemu/rustml/issues/4)

## What

Robust linear regression for outlier-heavy data.

- **RANSAC**: repeatedly sample `min_samples` points, fit OLS, count inliers
  (`|y - ŷ| < threshold`), keep the model with the most inliers; finally refit
  on the union of inliers.
- **TheilSen**: enumerate (or sub-sample) subsets of size `n_features + 1`,
  OLS each, then take the spatial (geometric) median of the coefficient
  vectors via Weiszfeld iterations.

## Reference

`sklearn.linear_model.{RANSACRegressor, TheilSenRegressor}` — sklearn 1.8.0.

## Golden test

- Generator: `test_harness/generators/gen_robust.py`
- Fixture:   `crates/rustml/tests/golden_data/robust.json`
- Rust test: `crates/rustml/tests/golden_robust.rs`

120-sample 1-D contaminated regression problem: 100 inliers on `y = 2x + 1`
plus 20 wild outliers at `y ∈ [15, 30]`. Both implementations must recover
slope ≈ 2 and intercept ≈ 1.

Exact agreement with sklearn isn't pursued — the random-sampling order
differs. The test asserts both land in a tight band around the true line:
slope within 0.1 (RANSAC) / 0.5 (TheilSen).

## Differences from sklearn

- RANSAC takes only OLS as the base estimator (sklearn allows any).
- TheilSen subset sampling is uniform random; sklearn uses random combinations
  of indices and falls back to deterministic enumeration when feasible.
- No `stop_score` / `stop_probability` early termination for RANSAC.

## Complexity

- HuberRegressor: each IRLS step is one weighted OLS of cost **O(np² + p³)**; outer loop iterates until convergence (~20–50 iters).
- TheilSenRegressor: enumerates n choose (n_subsamples) random subsets; each fits OLS of cost **O(s·p² + p³)** where s = n_subsamples. Total dominated by `n_subsamples_iter`.
- RANSACRegressor: `max_trials × O(min_samples · p² + p³)` for inlier model fits + `O(n)` consensus checks per trial.
- Memory: **O(n·p)** for the data matrix; no quadratic blow-up in n.
