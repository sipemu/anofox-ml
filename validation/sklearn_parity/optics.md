# OPTICS — sklearn parity

Issue: [#16](https://github.com/sipemu/anofox-ml/issues/16) (now closed)

## What

Ordering Points To Identify the Clustering Structure (Ankerst et al. 1999).
Produces a reachability ordering of samples, then extracts clusters by
walking the ordering with a single-eps DBSCAN-like threshold.

Implemented in `anofox_ml_cluster::optics::Optics` with brute-force pairwise
distance and a priority-queue chain extension.

## Reference

`sklearn.cluster.OPTICS` — sklearn 1.8.0.

## Validation

Unit test: two 2-D blobs + a far outlier. The blobs get distinct labels and
the outlier is either marked noise or assigned to its own cluster.

A sklearn golden parity test is not provided because sklearn defaults to
the `'xi'` cluster-extraction mode (a sloped-reachability-segment heuristic)
while ours uses the simpler `'dbscan'`-style threshold; direct label
comparison is fragile under reasonable parameter choices for both.

## Complexity

- O(n² · d) pairwise distances + O(n²) reachability ordering.
- Memory: O(n²) for the pairwise distance cache.

## Differences from sklearn

- Only `extract_dbscan`-style extraction; no `xi` mode (steepness-of-slope
  cluster identification).
- No KD-tree / Ball-tree acceleration.
- `cluster_hierarchy_` / `reachability_plot_` not exposed (the reachability
  ordering and per-point reachability distances are accessible on the fitted
  model).
