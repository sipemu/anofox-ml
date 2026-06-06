# AgglomerativeClustering — sklearn parity

Issue: [#15](https://github.com/sipemu/anofox-ml/issues/15) (partial — Spectral & MeanShift pending)

## What

Bottom-up hierarchical clustering with Lance-Williams updates. Linkages:
Single, Complete, Average, Ward. Ward uses Müllner's `O(n²)` nn-chain;
Single/Complete/Average use the naive `O(n³)` sweep (also `O(n²)` memory).

## Reference

`sklearn.cluster.AgglomerativeClustering` — sklearn 1.8.0.

## Golden test

- Generator: `test_harness/generators/gen_agglomerative.py`
- Fixture:   `crates/anofox-ml/tests/golden_data/agglomerative.json`
- Rust test: `crates/anofox-ml/tests/golden_agglomerative.rs`

120-sample 4-blob dataset, all four linkages. Labels are arbitrary permutations
between implementations, so we compare via Adjusted Rand Index against the
true labels and require `ARI ≥ sklearn_ARI − 0.05`.

## Differences from sklearn

- **Not yet implemented**: `SpectralClustering`, `MeanShift` (still tracked
  in #15).
- No `distance_threshold` mode — `n_clusters` only.
- No `connectivity` constraint graph.
- No `compute_full_tree` / `children_` / `distances_` output.

## Complexity

- Time: **O(n²)** for Ward (Müllner's nn-chain, default).
- Time: **O(n³)** for Single/Complete/Average (naive Lance-Williams).
- Memory: **O(n²)** (dense pairwise distance matrix).
- Set `RUSTML_AGGLO_NAIVE=1` to force the O(n³) sweep on Ward — cross-checks
  against nn-chain via a partition-equality regression test.
- Hard ceiling: ~5,000 samples before the O(n²) memory bound bites.
