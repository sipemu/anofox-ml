# HDBSCAN-lite — sklearn parity

Issue: [#16](https://github.com/sipemu/anofox-ml/issues/16) (now closed)

## What

Hierarchical Density-Based Spatial Clustering with size-gated boundary cuts.
Implements:

1. Mutual reachability distance `d_mr(a, b) = max(core(a), core(b), d(a, b))`.
2. Prim's MST of the mutual-reachability graph.
3. Single-linkage hierarchy by walking MST edges in ascending order.
4. Cluster extraction by cutting edges where both pre-merge sub-clusters have
   ≥ `min_cluster_size` points.

Full HDBSCAN's stability-based extraction (Campello et al. 2013) — which
correctly isolates singleton outliers as noise via cluster lifetime analysis
— is deferred. The current implementation handles well-formed clusters
correctly but may absorb near-cluster outliers into the nearest cluster.

## Reference

`sklearn.cluster.HDBSCAN` — sklearn 1.8.0.

## Validation

Unit test: two well-separated 2-D blobs are correctly separated into distinct
clusters.

## Complexity

- O(n² · d) pairwise distances + O(n²) Prim's MST.
- Memory: O(n²) for the distance matrix.

## Differences from sklearn

- **No stability-based extraction** — uses size-gated boundary cuts instead.
  This is the major functional gap; clusters are correctly separated but
  outliers are not reliably marked noise.
- No `cluster_persistence_`, `condensed_tree_`, or `single_linkage_tree_`
  output.
- No KD-tree / Ball-tree acceleration for k-NN core distance computation.
