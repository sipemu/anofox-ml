# HalvingGridSearchCV — sklearn parity

Issue: [#23](https://github.com/sipemu/anofox-ml/issues/23) (partial — no randomized variant)

## What

Successive halving for hyperparameter search. Start with `min_resources`
samples, evaluate all candidates, keep top `1 / factor`, multiply resources
by `factor`, repeat until 1 candidate remains.

API: `halving_grid_search_cv` in `anofox-ml-core::halving`.

## Reference

`sklearn.model_selection.HalvingGridSearchCV` — sklearn 1.8.0.

## Validation

The function is asserted on a hand-built fixture in the unit tests (two
candidate predictors, one perfect, one zeros — the search must pick the
perfect one). A sklearn golden comparison is intentionally skipped: sklearn
uses CV-aware resource budgets, our implementation uses a simple 80/20
internal split, and the two are not directly comparable on the same
"resources" parameter.

## Differences from sklearn

- **No** `HalvingRandomSearchCV` — pending.
- No `resource='n_samples' | parameter` switch — only sample-based resources.
- Internal scoring uses a single 80/20 split rather than full k-fold CV at
  each round, to keep cost down. sklearn does k-fold within each round.
- No `min_resources='exhaust' | 'smallest'` heuristic — caller supplies.
- No `aggressive_elimination`.

## Complexity

- HalvingGridSearchCV / HalvingRandomSearchCV: at iteration `i` (i = 0, 1, …), evaluate `n_candidates_i = max(1, n_candidates_0 / factor^i)` candidates on `n_resources_i = min_resources · factor^i` samples.
- Total cost ≈ `Σ_i n_candidates_i · cost(estimator on n_resources_i)`.
- For estimators linear in n_samples this is a geometric series, dominated by the final iteration → close to one full-data fit, but with the search effectively free.
- Memory: **O(p · candidates)** for parameter tables; negligible compared to the estimator footprint.
