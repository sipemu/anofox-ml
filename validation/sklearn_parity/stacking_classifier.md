# StackingClassifier — sklearn parity

Issue: [#10](https://github.com/sipemu/sequels/issues/10)

## What

Mirrors `sklearn.ensemble.StackingClassifier` with `stack_method='predict'`:
base classifier *hard predictions* are used as features for a meta-classifier.
Out-of-fold predictions are generated via sequential k-fold during fitting.

## Reference

`sklearn.ensemble.StackingClassifier` — sklearn 1.8.0.

## Golden test

- Generator: `test_harness/generators/gen_stacking_classifier.py`
- Fixture:   `crates/rustml/tests/golden_data/stacking_classifier.json`
- Rust test: `crates/rustml/tests/golden_stacking_classifier.rs`

Behavioral parity, not exact prediction match — we use hard predictions and
sequential k-fold; sklearn defaults to `predict_proba` and StratifiedKFold.
The fixture is a 120-sample binary problem from `make_classification` with
`class_sep=2.5` (well separated), interleaved so non-stratified k-fold sees
both classes per fold.

The test asserts:
1. Our accuracy is within ±10% of sklearn's,
2. Our accuracy is ≥ 0.85 (sanity floor).

## Differences from sklearn

- Uses hard `predict` outputs from base estimators, not `predict_proba` /
  `decision_function`. Sklearn's default is `'auto'`, which prefers the
  probabilistic path when available.
- Uses simple sequential KFold — sklearn defaults to StratifiedKFold.
- No `passthrough` option (sklearn can forward the original features to the
  meta-estimator alongside base predictions).

## Complexity

- StackingClassifier/StackingRegressor: K-fold cross-validation to produce out-of-fold predictions for each base estimator → **K × O(fit(base_i, n_samples · (K-1)/K))** per base estimator.
- Plus one final fit of each base estimator on the full data and one fit of the meta-estimator on the OOF features.
- Memory: **O(n · n_base · n_outputs)** for the meta-feature matrix.
- Parallelisable across base estimators (independent), and within each base estimator's K folds.
