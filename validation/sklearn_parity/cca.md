# CCA — sklearn parity

Issue: [#11](https://github.com/sipemu/rustml/issues/11) (now closed)

## What

Canonical Correlation Analysis via closed-form SVD on whitened (X, Y):

1. Centre and whiten `X` and `Y` via SVD (`K_x`, `K_y`).
2. Cross-covariance `C = X_whiteᵀ Y_white / (n − 1)`.
3. SVD `C = U Σ Vᵀ`. The first `k` columns of `K_x U` and `K_y V` are the
   canonical loadings; `Σ_ii` are the canonical correlations.

Implemented in `rustml_preprocessing::cca::Cca`.

## Reference

`sklearn.cross_decomposition.CCA` — sklearn 1.8.0.

## Validation

Unit tests:
- 100-sample (X 3-D, Y 2-D) problem where `Y[:, 0] = X[:, 0] + ε`: first
  canonical correlation > 0.9.
- Shape-only test: `transform_x` and `transform_y` produce `(n, n_components)`.

## Complexity

- `fit`: two whitening SVDs + one cross-covariance SVD. All O(min(n,d)³).
- Memory: O(d_x · d_y) for the cross-covariance.

## Differences from sklearn

- No iterative deflation (sklearn supports both algorithm='nipals' for
  iterative and 'svd' for closed-form — we implement the closed-form only).
- No `score()` (predictive correlation on held-out data).
- No `inverse_transform`.
- No `PLSCanonical` — though the closed-form CCA result coincides with
  PLSCanonical for `mode='A'` deflation.
