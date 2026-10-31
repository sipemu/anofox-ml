# Changelog

All notable changes to anofox-ml are recorded here. The format is loosely based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Algorithms — new

- **HDBSCAN** (`anofox-ml-cluster`) with stability-based flat extraction that
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
- `CsrMatrix` (compressed sparse row) added to `anofox-ml-core::sparse`. Text
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
- **Cross-validation (#33):** `grid_search_cv`, `randomized_search_cv`,
  `cross_val_predict`, `cross_validate`, `learning_curve` and
  `validation_curve` now work on regression targets. They fall back to seeded
  shuffled K-fold when `y` does not look like class labels, and new
  `*_with_splits` variants accept user-supplied splits. `stratified_k_fold`
  allocates folds like sklearn, so classes smaller than k no longer produce
  empty folds. `stratified_shuffle_split` is deterministic per seed and
  honours `test_size` exactly. `learning_curve` shuffles before taking
  prefixes, and both curves fit once per point.
- **Metrics (#34):** `roc_auc_score` and `average_precision_score` handle
  tied scores like sklearn. `precision_recall_curve` uses sklearn ordering
  (increasing thresholds, decreasing recall, (1, 0) sentinel last); **the
  output order changed**. NaN scores now return an error instead of panicking.
- **RandomForest / ExtraTrees (#45):** `max_features` is sampled per split
  instead of per tree. Defaults are now `Sqrt` for classifiers and all
  features for regressors (sklearn ≥ 1.1). Trees get independent seeds, and
  classifier `predict` averages leaf probabilities. **Breaking:**
  `max_features` is `Option<MaxFeatures>`, and `DecisionTree*` gain `seed`.
- **LGBM (#44):** early stopping bins the eval set with the training bin
  edges and truncates the model to `best_iteration`. `LgbmClassifier`
  supports `eval_set` / `early_stopping_rounds`.
- **CalibratedClassifierCV (#43):** calibrates out-of-fold probabilities or
  decision scores instead of hard labels. Platt scaling uses Newton's method
  (libsvm `sigmoid_train`). It returns the original class labels and rejects
  more than two classes. **Breaking:** the base model must implement
  `PredictProba`, or use `with_decision_function`.
- **HDBSCAN (#57):** now runs the full algorithm (condensed tree, stability,
  EOM / leaf selection) and matches sklearn on golden data. Previously a
  separated third cluster could be labelled as noise. New: `probabilities`,
  `with_allow_single_cluster`, `with_cluster_selection_method`.
- **DBSCAN / OPTICS / LOF / Isomap / AffinityPropagation (#57):**
  - `FittedDbscan::predict` labels new points by the nearest core sample
    (it used to return the training labels).
  - OPTICS and HDBSCAN `min_samples` count the point itself.
  - LOF uses sklearn's `1/(mean + 1e-10)` local reachability density, so
    duplicates score correctly.
  - Isomap errors on a disconnected kNN graph instead of setting the
    geodesic distances to 0.
  - AffinityPropagation exemplars always label themselves; new `converged`
    and `n_iter` fields.
- **Robustness (#42):** clustering, KNN, LOF, trees and ensembles accept
  Fortran-order / transposed input instead of panicking. Majority votes in
  KNN, forests, bagging and voting break ties deterministically toward the
  smallest label. Sorting and binning no longer panic on NaN.
  RobustScaler, QuantileTransformer and KBinsDiscretizer ignore NaN at fit
  and pass it through at transform.
- **PCA (#51):** now uses faer (thin SVD, or covariance eigendecomposition
  for tall data) instead of power iteration, and matches sklearn components
  and signs. Adds `explained_variance_ratio`, `singular_values`,
  `noise_variance` and `whiten`. TruncatedSVD, FastICA, NMF and CCA use thin
  SVD; TruncatedSVD signs and explained variance match sklearn.
- **Text (#52):**
  - `CountVectorizer::fit` / `TfidfVectorizer::fit` return fitted
    vectorizers whose `transform` encodes new documents.
  - HashingVectorizer uses MurmurHash3 like sklearn, so `alternate_sign`
    actually cancels collisions. **Its output changes.**
  - The tokenizer is Unicode-aware (`(?u)\b\w\w+\b`).
  - `min_df > max_df` now returns an error.
- **LDA (#65):** `scalings` / `transform` were computed from the wrong
  matrix. They now match sklearn `solver='eigen'`; new
  `explained_variance_ratio`.
- **SVM (#63):**
  - SVR and OneClassSVM project exactly onto the dual feasible set.
  - LinearSVC uses liblinear dual coordinate descent with a `loss` option
    (default squared hinge, as sklearn).
  - LinearSVR uses liblinear dual coordinate descent with L1 / L2
    ε-insensitive loss.
  - LinearSVC and LinearSVR coefficients match sklearn.
- **SGD (#31):**
  - sklearn stopping rule (`tol`, `n_iter_no_change`, `early_stopping` /
    `validation_fraction`) and `n_iter()`.
  - `partial_fit` runs one epoch per call and keeps the learning-rate
    schedule; `SgdRegressor` implements `PartialFit`.
  - Stable log loss, L1 gives exact zeros, and new `average` (ASGD) and
    `predict_proba`.
  - **Breaking:** binary classifiers train a single model
    (`weights().len() == 1`).

### Algorithms — performance

- **Agglomerative Ward** now uses Müllner's O(n²) nn-chain algorithm by
  default. Earlier attempts produced an inconsistent flat clustering because
  nn-chain processes merges in chain order rather than distance order. The
  fix runs the chain to a single cluster, sorts the recorded merges by
  distance ascending, and applies them via DSU until `n_clusters` remain.
  Set `RUSTML_AGGLO_NAIVE=1` to force the O(n³) sweep (regression-test
  cross-check).
