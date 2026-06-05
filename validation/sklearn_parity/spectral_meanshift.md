# SpectralClustering + MeanShift — sklearn parity

Issue: [#15](https://github.com/sipemu/rustml/issues/15) (now fully closed in terms of clustering set; HDBSCAN/OPTICS/Birch are #16).

## What

- **SpectralClustering** (`rustml-cluster::spectral`): RBF or k-NN affinity →
  normalised Laplacian `L = I - D^{-1/2} A D^{-1/2}` → bottom-k eigenvectors
  → row-normalise → KMeans on the embedding.
- **MeanShift** (`rustml-cluster::mean_shift`): flat-kernel mean-shift
  iteration from each point with mode-merging at bandwidth/2.

## Reference

`sklearn.cluster.SpectralClustering`, `sklearn.cluster.MeanShift` — sklearn 1.8.0.

## Validation

Unit tests:
- SpectralClustering: 2 well-separated blobs recovered with both RBF and k-NN
  affinity.
- MeanShift: 2 blobs at (0,0) and (10,10) with bandwidth=2.0 — centroids land
  near `|cx| < 1` and `|cx - 10| < 1`.

## Differences from sklearn

- SpectralClustering: only RBF and k-NN affinities; sklearn also supports
  precomputed and nearest-neighbours-with-mutual.
- SpectralClustering: `assign_labels='discretize'` not implemented (KMeans-on-embedding only).
- MeanShift: uses a flat (uniform) kernel; sklearn also supports Gaussian via
  the `kernel` parameter. No automatic `estimate_bandwidth`.

## Complexity

- SpectralClustering: O(n³) for the eigendecomposition + O(n²) for the
  affinity matrix.
- MeanShift: O(n² · max_iter · d) — each mean-shift iteration scans all
  points. KD-tree would drop this to O(n · k · max_iter · d) per point.
