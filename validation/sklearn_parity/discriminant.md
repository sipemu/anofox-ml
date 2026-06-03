# LDA / QDA — sklearn parity

Issue: [#14](https://github.com/sipemu/rustml/issues/14)

## What

- `LinearDiscriminantAnalysis` (LDA): pooled within-class covariance, linear
  decision boundary. Optional shrinkage toward `(tr(Σ)/d) I`.
- `QuadraticDiscriminantAnalysis` (QDA): per-class covariance.

Both compute the closed-form Bayes-optimal classifier under a Gaussian class-
conditional with maximum-likelihood mean/covariance estimates.

New crate: `rustml-discriminant`.

## Reference

`sklearn.discriminant_analysis.{LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis}` — sklearn 1.8.0.

## Golden test

- Generator: `test_harness/generators/gen_discriminant.py`
- Fixture:   `crates/rustml/tests/golden_data/discriminant.json`
- Rust test: `crates/rustml/tests/golden_discriminant.rs`

Two fixtures:
1. **LDA**: 3-class synthetic problem with shared Σ (180 samples). Sklearn
   `solver='lsqr'` matches our closed-form solve exactly.
2. **QDA**: 200-sample binary problem from `make_classification` with
   `n_clusters_per_class=1`. Sklearn QDA with `reg_param=0`.

Assertion: agreement with sklearn ≥ 97-98% on labels; rustml accuracy ≥ 85%.
We don't demand 100% sample agreement because a small number of boundary
points can flip due to floating-point differences in the Cholesky solve.

## Differences from sklearn

- LDA has no `transform` / dimensionality reduction yet (sklearn supports
  reducing to `n_classes - 1` dims via the LDA projection).
- No `predict_proba` / `decision_function`.
- No `priors` override; computed from class frequencies only.
- QDA's `reg_param` shrinks toward identity; we currently only add `reg` to
  the diagonal.
