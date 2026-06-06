# RFE / SequentialFeatureSelector — sklearn parity

Issue: [#21](https://github.com/sipemu/rustml/issues/21) (partial — RFECV not yet)

## What

- **RFE** (`Rfe`): callback-based recursive feature elimination. Caller supplies
  an `(X, y) → importance` function (e.g. `|coef_|` for Ridge, or feature
  importances for trees). RFE refits on the active set, drops `step` features
  with lowest importance, repeats until `n_features_to_select` remain.
- **SequentialFeatureSelector** (forward only): greedy forward selection. Caller
  supplies an `(X, y) → score` function (higher is better).

## Reference

`sklearn.feature_selection.{RFE, SequentialFeatureSelector}` — sklearn 1.8.0.

## Golden test

- Generator: `test_harness/generators/gen_rfe.py`
- Fixture:   `crates/rustml/tests/golden_data/rfe.json`
- Rust test: `crates/rustml/tests/golden_rfe.rs`

100×8 problem with 3 informative features at indices `{0, 2, 5}`. RFE asserts
exact match with sklearn's `support_`; SFS asserts ≥ 2 of 3 informative
features are picked (sklearn uses 3-fold CV scoring, we use train-set R²
which is biased — a tighter comparison would need CV in our scorer).

## Differences from sklearn

- **No `RFECV`** — only fixed `n_features_to_select` is supported.
- **Forward SFS only** — sklearn supports forward and backward.
- Importance / score are user-supplied closures rather than implicit access
  to `estimator.coef_` / `feature_importances_`. This generalises but loses
  the convenience of `RFE(estimator=Ridge())`.
- No `n_features_to_select='auto'`.

## Complexity

- RFE: at each step, fit the estimator (**O(fit(n, p_current))**) and rank features by importance, removing the bottom `step` features. Repeat until `n_features_to_select` features remain.
- Total: **O((p - target) / step · cost(fit))**.
- RFECV: above, performed independently on each of K CV folds, with the optimal feature count chosen by cross-validated score. Memory: **O(K · p)** for fold-wise rankings.
- Parallelisable across CV folds.
