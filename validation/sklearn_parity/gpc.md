# GaussianProcessClassifier — sklearn parity

Issue: [#12](https://github.com/sipemu/rustml/issues/12) (now closed)

## What

Binary Gaussian Process classification via Laplace approximation
(Rasmussen & Williams §3.4, Algorithm 3.1):

1. Newton-Raphson on the latent posterior with Cholesky of
   `B = I + W^{1/2} K W^{1/2}` for numerical stability.
2. Posterior mean `f̄_*` and variance `V[f_*]` at query points via
   `L^{-1} W^{1/2} k_*`.
3. Predict via probit approximation:
   `p(y=1|x_*) ≈ σ(f̄_* / √(1 + π/8 · V[f_*]))`.

Implemented in `rustml_gaussian_process::classifier::GaussianProcessClassifier`.

## Reference

`sklearn.gaussian_process.GaussianProcessClassifier` (binary case) — sklearn 1.8.0.

## Validation

Unit test: two 2-D clusters separated at distance 5 with 12 samples →
≥ 11/12 correct predictions; `predict_proba` rows sum to 1.

## Complexity

- `fit`: O(max_iter · n³) — each Newton step does a fresh Cholesky.
- `predict`: O(n_train · n_test · d) for the kernel matrix +
  O(n_train · n_test) for the dual product.
- Memory: O(n²) for the Cholesky factor.

## Differences from sklearn

- **Binary classification only.** Multi-class via one-vs-rest is not yet
  wired up.
- No automatic kernel hyperparameter learning. (The regressor has
  `optimize_rbf_length_scale`; the classifier doesn't share it.)
- No multi-restart of the Newton optimization.
