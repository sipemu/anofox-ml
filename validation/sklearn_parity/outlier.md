# IsolationForest / LocalOutlierFactor — sklearn parity

Issue: [#20](https://github.com/sipemu/rustml/issues/20)

## What

- **IsolationForest** (`rustml-ensemble`): forest of randomized depth-limited
  binary trees. Anomaly score = `2^(-E[h(x)] / c(n))` averaged across trees.
  Threshold calibrated from the training-set contamination rate.
- **LocalOutlierFactor** (`rustml-neighbors`): density-based scoring. For each
  point `a`, `LOF(a) = mean_{b ∈ N(a)} lrd(b) / lrd(a)`. Values much greater
  than 1 indicate an outlier. We use the sklearn sign convention
  (`-LOF` higher = more normal).

## Reference

`sklearn.ensemble.IsolationForest`, `sklearn.neighbors.LocalOutlierFactor` — sklearn 1.8.0.

## Golden test

- Generator: `test_harness/generators/gen_outlier.py`
- Fixture:   `crates/rustml/tests/golden_data/outlier.json`
- Rust test: `crates/rustml/tests/golden_outlier.rs`

205-sample 2-D dataset: 200 normal Gaussians plus 5 fixed wild outliers.
Both implementations must catch ≥ 60% of outliers and within ±0.4 recall of
sklearn's number.

## Differences from sklearn

- IsolationForest: no sample-weight; no `score_samples` API exposed publicly
  beyond `predict`; tree storage is a flat `Vec<INode>` (no `extra-trees`-style
  shortcut for half-uniform splits).
- LocalOutlierFactor: only "fit" mode, no `novelty=True` (predicting on new
  data isn't supported — sklearn's LOF only supports it in novelty mode).
- LOF uses brute-force pairwise distances (O(n²)); a KD-tree path for
  Euclidean distance would be a future optimisation.

## Complexity

- **IsolationForest** fit: O(n_trees · subsample · log(subsample)). Trees are built in parallel via rayon.
- **IsolationForest** predict: O(n_test · n_trees · log(subsample)). Per-row scoring is parallelised.
- **LOF** fit: O(n² · d) (brute-force pairwise scan with a bounded heap). Memory O(n·k) for nearest-neighbour lists.
- LOF KD-tree path would drop fit to O(n log n · d) — pending; the KNN crate already has the tree implementation.
