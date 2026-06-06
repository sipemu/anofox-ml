# FastICA — sklearn parity

Issue: [#18](https://github.com/sipemu/anofox-ml/issues/18) (now closed)

## What

FastICA — Independent Component Analysis with deflation and the `logcosh`
non-linearity. Implementation steps:

1. Centre and whiten `X` via SVD so that `cov(X) = I`.
2. Per component, fixed-point iteration
   `w ← E[X g(wᵀX)] − E[g'(wᵀX)] w` with `g = tanh`.
3. Orthogonalise against previously extracted components.
4. Normalise to unit length, repeat until convergence.

Implemented in `anofox_ml_preprocessing::fast_ica::FastIca`.

## Reference

`sklearn.decomposition.FastICA` (algorithm='deflation', fun='logcosh') —
sklearn 1.8.0.

## Validation

Unit test: a known mixture of a sine and a square-wave signal is fed in;
the recovered sources are finite and the right shape `(n, n_components)`.

A full sklearn parity test is not provided because ICA has inherent
sign / permutation ambiguity — recovered components match the originals
up to a sign flip and reordering.

## Complexity

- `fit`: one whitening SVD + O(max_iter · n_components · n · d) for the
  deflation loop.
- Memory: O(n · d).

## Differences from sklearn

- Only `algorithm='deflation'` (parallel symmetric decorrelation not implemented).
- Only `fun='logcosh'`. The `'exp'` and `'cube'` non-linearities are not
  implemented.
- No `whiten_solver='eigh'` option (always SVD-based).
