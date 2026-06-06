# PassiveAggressive — sklearn parity

Issue: [#5](https://github.com/sipemu/rustml/issues/5)

## What

PassiveAggressiveClassifier (binary, hinge loss) and PassiveAggressiveRegressor
(epsilon-insensitive loss). Implements the original PA, PA-I, and PA-II
variants from Crammer et al. (2006).

Per-sample update:
- loss = max(0, 1 − y·(w·x)) for classification, max(0, |y − w·x| − ε) for regression
- if loss > 0: τ = depends on variant; w ← w + τ·sign·x

## Reference

`sklearn.linear_model.{PassiveAggressiveClassifier, PassiveAggressiveRegressor}` — sklearn 1.8.0.

## Golden test

- Generator: `test_harness/generators/gen_passive_aggressive.py`
- Fixture:   `crates/rustml/tests/golden_data/passive_aggressive.json`
- Rust test: `crates/rustml/tests/golden_passive_aggressive.rs`

Behavioral parity tests:
- Classifier on a 200-sample, 10-feature `make_classification` problem
  (class_sep=2.0). Both implementations land at ≥ 85% accuracy; rustml within
  0.15 of sklearn's accuracy.
- Regressor on a 200-sample `make_regression` problem with standardized `y`.
  Both achieve R² > 0.6; rustml within 0.15 of sklearn's R².

We don't pursue exact agreement: sklearn averages updates (`averaging`),
applies an early-stopping validation split, and uses a different RNG order.
Our implementation is the textbook PA online update with optional shuffle and
a tol-based early stop.

## Differences from sklearn

- Binary classification only; sklearn does multi-class via one-vs-rest.
- No update averaging.
- No `class_weight` / `sample_weight`.
- No `early_stopping` / `validation_fraction`.
- `PaVariant::Pa`, `PaI`, `PaII` map to the original / PA-I (sklearn default) /
  PA-II respectively.

## Complexity

- Each sample requires computing the prediction (**O(p)**), the loss-margin update step (**O(p)**), and weight regularisation.
- One epoch: **O(n·p)**.
- Total over `max_iter` epochs: **O(n·p·max_iter)**, but in practice convergence is reached well before `max_iter`.
- Memory: **O(p · k)** for one weight vector per class (k=1 for binary, OvR otherwise).
- Online-friendly: `partial_fit` extends to streaming data via the same update.
