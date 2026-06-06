# Changelog

All notable changes to RustML are recorded here. The format is loosely based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Algorithms — new

- **HDBSCAN** (`rustml-cluster`) with stability-based flat extraction that
  correctly marks low-density outliers as noise.
- **BayesianGaussianMixture** — full variational inference with
  Normal–Wishart conjugate priors (Bishop §10.2), Dirichlet weight prior,
  posterior covariance reporting, ELBO-based convergence.
- **MeanShift, OPTICS, Birch (lite), SpectralClustering, AffinityPropagation**.
- **Isomap, LocallyLinearEmbedding, t-SNE** (exact O(n²) + Barnes-Hut quadtree
  O(n log n) for 2D embeddings).
- **GaussianProcessClassifier** with Laplace approximation, multi-class via
  one-vs-rest. Kernel zoo (Matern, RationalQuadratic, White, Constant, sums,
  products) added on the regressor side.
- **FastICA, CCA, NMF, PLS, KernelPCA + inverse_transform, TruncatedSVD, RFECV**.
- **LocalOutlierFactor** with both brute-force and KD-tree paths
  (auto-dispatch ≤ 20 dims).

### Algorithms — extensions

- **AffinityPropagation** sparse k-NN responsibility matrix path
  (`with_n_neighbors`) for O(n·k) memory and per-iter cost vs the dense
  O(n²) default.
- **t-SNE** Barnes-Hut quadtree path for 2D embeddings, gated by
  `with_method(TSneMethod::BarnesHut)`.
- **KMeans** `sample_weight` via the new `FitUnsupervisedWeighted` trait.
- **GaussianNB** and **SGDClassifier** `partial_fit` for streaming /
  incremental training.
- **GP hyperparameter tuning**: multi-parameter BFGS with finite-difference
  gradients (`optimize_kernel_lbfgs`) — extends the prior single-parameter
  golden-section sweep.

### Core / infrastructure

- New traits: `FitWeighted`, `FitUnsupervisedWeighted`, `PartialFit`,
  `PredictLogProba`, `DecisionFunction`, `RegressorScore`, `ClassifierScore`.
- `CsrMatrix` (compressed sparse row) added to `rustml-core::sparse`. Text
  vectorizers now expose `fit_transform_sparse` returning CSR — typical
  10⁴–10⁵-vocab corpora are dense-infeasible.
- PyO3 bindings extended to cover HDBSCAN, MeanShift, AffinityPropagation,
  BayesianGaussianMixture, LocalOutlierFactor, t-SNE, Isomap,
  LocallyLinearEmbedding.
- GitHub Actions CI: check, test matrix, clippy, fmt, docs, sklearn version
  pin, golden-data sync, Python bindings build, coverage (advisory).
- 35 sklearn-parity validation docs under `validation/sklearn_parity/`, each
  with a Complexity section.

### Fixes

- `LassoLarsIC` AIC formula corrected.
- Stacking estimators now prefer `PredictProba` when the base estimator
  provides it (matches sklearn behaviour).

### Algorithms — performance

- **Agglomerative Ward** now uses Müllner's O(n²) nn-chain algorithm by
  default. Earlier attempts produced an inconsistent flat clustering because
  nn-chain processes merges in chain order rather than distance order. The
  fix runs the chain to a single cluster, sorts the recorded merges by
  distance ascending, and applies them via DSU until `n_clusters` remain.
  Set `RUSTML_AGGLO_NAIVE=1` to force the O(n³) sweep (regression-test
  cross-check).
