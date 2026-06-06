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

## Complexity

- Isomap: k-NN graph O(n²·p) + Dijkstra/Floyd geodesic distances O(n²·log n) + classical MDS eigendecomposition O(n³). Memory **O(n²)**.
- LocallyLinearEmbedding: k-NN graph + per-point local reconstruction (k×k least squares) + bottom-eigendecomposition of (I-W)ᵀ(I-W). Time **O(n·k³ + n²·k)**, memory **O(n²)**.
- ClassicalMDS: double-centring D² + eigendecomposition. Time **O(n³)**, memory **O(n²)**.
- t-SNE (exact): **O(n²·iter)**, memory **O(n²)**.
- t-SNE (Barnes-Hut): **O(n·log n · iter)** with quadtree, memory **O(n)**.
