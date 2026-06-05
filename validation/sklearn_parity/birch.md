# Birch-lite — sklearn parity

Issue: [#16](https://github.com/sipemu/rustml/issues/16) (now closed)

## What

Single-pass online sub-clustering with radius threshold, then final KMeans on
sub-cluster centroids. Mirrors `sklearn.cluster.Birch`'s user-facing behaviour
without implementing the full CF (Cluster Feature) tree.

## Reference

`sklearn.cluster.Birch` — sklearn 1.8.0.

## Validation

Unit tests:
- 2 well-separated 2-D blobs are correctly assigned to 2 distinct clusters.
- With `n_clusters=None` each CF is its own cluster.

A full sklearn golden test is not provided because Birch's CF tree structure
makes labels depend on the order of insertion; sklearn's `Birch.fit` produces
different label orderings than our simpler accumulator. The behavioural
invariants (cluster separation, predict on new data) are equivalent.

## Complexity

- `fit`: O(n · m · d) where m is the number of CF subclusters that emerge
  (≤ n / threshold²-ish for a uniform spread). Final KMeans is O(m · k · d).
- Memory: O(m · d) for sub-cluster centroids.

## Differences from sklearn

- No CF tree — flat list of subclusters. With many subclusters the inner
  "nearest CF" loop is the bottleneck (the CF tree gives O(log m) lookups).
- No `branching_factor` parameter.
- No `partial_fit`.
