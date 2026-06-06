# GaussianMixture — sklearn parity

Issue: [#13](https://github.com/sipemu/anofox-ml/issues/13) (partial — Bayesian GMM pending)

## What

Gaussian Mixture Model trained via EM with k-means++ init. Supports `full`
and `diag` covariance types.

## Reference

`sklearn.mixture.GaussianMixture` — sklearn 1.8.0.

## Golden test

- Generator: `test_harness/generators/gen_gmm.py`
- Fixture:   `crates/anofox-ml/tests/golden_data/gmm.json`
- Rust test: `crates/anofox-ml/tests/golden_gmm.rs`

150-sample 3-blob dataset, both covariance types. Compared via Adjusted Rand
Index against the true labels (`ARI ≥ sklearn_ARI − 0.05`).

## Differences from sklearn

- **No** `BayesianGaussianMixture` (Dirichlet-process variational) — pending.
- Covariance types `tied` and `spherical` not implemented.
- Single initialization (no `n_init`).
- No `score_samples` / `predict_proba` public API.

## Complexity

- `fit`: O(max_iter · n · k · d²) for full covariance; O(max_iter · n · k · d) for diagonal.
- E-step does a single pass per iteration (log-likelihood accumulates from the same log-sum-exp values used for responsibilities).
- `predict_proba`: O(n_test · k · d²) (full) or O(n_test · k · d) (diag).
- Memory: O(n · k) for responsibility matrix during fit; not retained.
