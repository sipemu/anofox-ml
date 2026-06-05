# Isomap + LocallyLinearEmbedding — sklearn parity

Issue: [#19](https://github.com/sipemu/rustml/issues/19) (now closed)

## What

- **Isomap** (`rustml_manifold::isomap`): k-NN graph + Floyd-Warshall geodesic
  distances + classical MDS on the geodesic distance matrix.
- **LocallyLinearEmbedding** (`rustml_manifold::lle`): per-point local
  reconstruction weights via Cholesky on the local Gram, then bottom-k
  eigenvectors of `(I − W)ᵀ(I − W)` (dropping the smallest, which
  corresponds to the constant direction).

## Reference

`sklearn.manifold.Isomap`, `sklearn.manifold.LocallyLinearEmbedding` —
sklearn 1.8.0.

## Validation

- Isomap: 25-point 1D arc embedded in 2-D is unrolled to a monotone 1-D
  embedding (geodesic distance preserves order along the curve).
- LLE: same arc test produces a monotone 1-D embedding. Also passes a 9-point
  2-D grid → 2-D test.

## Complexity

- Isomap: O(n³) for Floyd-Warshall + O(n³) for MDS eigendecomposition.
  Memory O(n²) for the geodesic distance matrix.
- LLE: O(n² · d) for k-NN + O(n³) for the eigendecomposition of M.
  Memory O(n²) for M.

## Differences from sklearn

- Isomap: brute-force pairwise distances. KD-tree / Ball-tree path not
  implemented.
- LLE: only the standard variant. No `method='modified'`, `'hessian'`,
  `'ltsa'`.
- No `transform()` for new samples — sklearn supports out-of-sample
  projection via Nyström-like extension.
