# AgglomerativeClustering — sklearn parity

Issue: [#15](https://github.com/sipemu/rustml/issues/15) (partial — Spectral & MeanShift pending)

## What

Bottom-up hierarchical clustering with Lance-Williams updates. Linkages:
Single, Complete, Average, Ward. Naive `O(n²)` memory / `O(n³)` time.

## Reference

`sklearn.cluster.AgglomerativeClustering` — sklearn 1.8.0.

## Golden test

- Generator: `test_harness/generators/gen_agglomerative.py`
- Fixture:   `crates/rustml/tests/golden_data/agglomerative.json`
- Rust test: `crates/rustml/tests/golden_agglomerative.rs`

120-sample 4-blob dataset, all four linkages. Labels are arbitrary permutations
between implementations, so we compare via Adjusted Rand Index against the
true labels and require `ARI ≥ sklearn_ARI − 0.05`.

## Differences from sklearn

- **Not yet implemented**: `SpectralClustering`, `MeanShift` (still tracked
  in #15).
- No `distance_threshold` mode — `n_clusters` only.
- No `connectivity` constraint graph.
- No `compute_full_tree` / `children_` / `distances_` output.
