# MiniBatchKMeans — sklearn parity

Issue: [#17](https://github.com/sipemu/rustml/issues/17)

## What

Mini-batch K-Means using Sculley's per-sample learning-rate update:
`cₖ ← (1 − 1/Nₖ) cₖ + (1/Nₖ) x` where `Nₖ` is the running count of samples
ever assigned to cluster `k`. k-means++ initialization.

## Reference

`sklearn.cluster.MiniBatchKMeans` — sklearn 1.8.0.

## Golden test

- Generator: `test_harness/generators/gen_mini_batch_kmeans.py`
- Fixture:   `crates/rustml/tests/golden_data/mini_batch_kmeans.json`
- Rust test: `crates/rustml/tests/golden_mini_batch_kmeans.rs`

300-sample 4-blob `make_blobs` dataset with `cluster_std=0.6`. Centroids are
matched greedily against sklearn's; we require **max matched-pair distance
< 0.5** (well within the blob spread), plus all clusters non-empty.

## Differences from sklearn

- No `n_init` — single initialization. sklearn defaults to 3 and keeps the
  best.
- No `reassignment_ratio` (sklearn reassigns near-empty clusters every
  iteration).
- No `init='random'` option; always k-means++.
- Convergence test is on the squared centroid shift across a full pass, not
  on running EWMA inertia like sklearn.
