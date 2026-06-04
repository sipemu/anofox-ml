# Manifold learning (Classical MDS) — sklearn parity

Issue: [#19](https://github.com/sipemu/rustml/issues/19) (partial — t-SNE / Isomap / LLE pending)

## What

New crate `rustml-manifold`. Implements **Classical MDS** (Torgerson scaling)
via double-centring the squared pairwise-distance matrix and
eigendecomposing the resulting Gram matrix.

## Reference

`sklearn.manifold.MDS` with `metric=True`, `dissimilarity='euclidean'` — though
sklearn's MDS uses SMACOF stress minimisation rather than the closed-form
eigen approach. Classical MDS is what `cmdscale` in R does and is also the
initialisation sklearn's `MDS` would pick at `n_init=1` without iteration.

## Validation

Unit test asserts that for points already living in 2-D, classical MDS at
`n_components=2` reproduces their pairwise distances exactly (modulo
rotation/reflection — distance matrix is invariant).

## Not yet implemented

- **t-SNE**, **Isomap**, **LocallyLinearEmbedding**, **SpectralEmbedding** —
  all still pending in #19.
- SMACOF stress minimisation MDS (sklearn's default).
- Non-Euclidean dissimilarity matrices (would be easy: skip `pairwise_dist`
  and accept the matrix directly).
