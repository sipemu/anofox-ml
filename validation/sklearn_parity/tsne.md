# t-SNE — sklearn parity

Issue: [#19](https://github.com/sipemu/rustml/issues/19) (now closed)

## What

Vanilla O(n²) t-SNE (van der Maaten & Hinton 2008): perplexity-calibrated
Gaussian conditional probabilities in input space, student-t affinities in
embedding space, gradient descent with momentum and early exaggeration.

Implemented in `rustml_manifold::tsne::TSne`. Suitable for n ≲ 1000; beyond
that the O(n²) cost per iteration dominates.

## Reference

`sklearn.manifold.TSNE` — sklearn 1.8.0.

## Validation

Unit test: 30 points across two well-separated 5-D blobs are linearly
separable along the centroid-difference direction in the 2-D embedding for
≥ 80% of points per cluster.

A full sklearn golden test is not provided because the optimization is
stochastic and different RNG paths produce different (but equivalent)
embeddings.

## Complexity

- `fit`: O(n² · max_iter) — pairwise affinities and pairwise gradient terms.
- Memory: O(n²) for the P and Q matrices.

## Differences from sklearn

- **No Barnes-Hut acceleration** — Barnes-Hut drops the cost to O(n log n)
  per iteration. Without it, our t-SNE doesn't scale past ~1000 samples.
- No `init='pca'` (sklearn's default since 1.2) — we always init with a
  small random embedding.
- No `metric='precomputed'` (always Euclidean on the raw data).
- No `learning_rate='auto'`.
