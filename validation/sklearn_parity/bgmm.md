# BayesianGaussianMixture — sklearn parity

Issue: [#13](https://github.com/sipemu/anofox-ml/issues/13) (now closed)

## What

EM on a GMM with Dirichlet-smoothed mixing weights:
`π_k = (α₀ + N_k) / (k · α₀ + N)`.

Captures the user-facing API of `sklearn.mixture.BayesianGaussianMixture`
without implementing full variational inference. With `α₀ << 1`, low-mass
components get smoothed weights but the algorithm doesn't auto-prune as
aggressively as the full variational version.

## Reference

`sklearn.mixture.BayesianGaussianMixture` — sklearn 1.8.0.

## Validation

Unit test: 2 well-separated 2-D blobs, fit with `n_components=2` and
`α₀=0.01`. Points within a blob share a label; `predict_proba` rows sum to 1.

## Complexity

- `fit`: O(max_iter · n · k · d²) for full covariance.
- Memory: O(n · k) for responsibilities during fit.

## Differences from sklearn

- **No full variational inference** — no Wishart prior on precision, no
  Normal prior on means, no Dirichlet process variant. The auto-pruning
  behaviour ("set n_components large, let it find the effective count") is
  weak; users should set `n_components` close to the true cluster count.
- No `weight_concentration_prior_type='dirichlet_process'`.
- Single initialization (no `n_init`).
