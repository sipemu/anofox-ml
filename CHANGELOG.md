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

### sklearn parity (review P1)

Each item below is checked against hardcoded scikit-learn 1.8 reference values.
**Breaking** marks API or default changes.

- **anofox-regression 0.7 (#27):**
  - The dependency is bumped from 0.5 to 0.7.
  - TheilSen, RANSAC, BayesianRidge/ARD (now with `predict_std`), LARS /
    LassoLars / LassoLarsIC, PassiveAggressiveRegressor (now with `PartialFit`)
    and PLSRegression (now with `transform` and scores) delegate to it.
  - LassoLarsIC evaluates sklearn's criterion along a single path.
  - RANSAC no longer panics when `min_samples > n` or on NaN.
  - Upstream deviations from sklearn are tracked in #72 (no local
    workarounds): the LassoLars drop rule and `predict_std` centring.
- **Regularisation CV (#28):** LassoCV/ElasticNetCV use warm-started
  coordinate-descent paths on sklearn's alpha scale with the default 100-alpha
  grid. They select by pooled MSE over seeded or explicit folds and expose
  `alphas()`, `mse_path()` and `alpha_1se()`. RidgeCV defaults to exact
  leave-one-out from one SVD. **Breaking:** the alpha scale and the folds changed.
- **GLMs (#29):**
  - Poisson/Tweedie/Gamma/Binomial `alpha` now has sklearn's scaling
    (upstream `lambda = alpha · Σw`).
  - Offset/exposure support and opt-in inference via `with_inference(true)`
    → `summary()` (SE, z/t, p, CI).
  - `predict_interval` for OLS/WLS/Ridge.
  - Binomial `sample_weight` and probit/cloglog links.
  - Blocked on anofox-regression (#72): sklearn's default `alpha = 1.0`
    (upstream penalised IRLS does not converge; the default stays 0) and
    `sample_weight` for Poisson/Tweedie/Gamma.
- **LogisticRegressor (#30):**
  - Binary and one-vs-rest multiclass with `C`, `class_weight` and
    `sample_weight`, delegated to upstream's logit GLM.
  - Implements the core classifier traits.
  - **Breaking:** `predict_proba` is `(n, K)` and the default is `C = 1.0`.
  - Blocked on anofox-regression (#72): the multinomial loss and L1 /
    elastic-net penalties (both return an error).
- **Metrics (#35):**
  - `Average::Binary { pos_label }`, per-class scores, and `ClassificationOptions`
    (`labels`, `sample_weight`, `zero_division`).
  - Multiclass `roc_auc_score_multiclass` (OvR/OvO) and `log_loss_multiclass`.
  - Labels are encoded once with exact comparison.
  - With constant `y`, `r2_score` gives 1.0 for a perfect fit and 0.0 otherwise.
- **Halving search (#36):** per-round (stratified) k-fold CV, sklearn's schedule,
  and NaN scores ranked last. Also returns per-iteration `n_resources` /
  `n_candidates`.
- **Boosting / stacking (#46):**
  - GradientBoostingClassifier uses Newton leaf values and true multinomial deviance.
  - AdaBoost SAMME uses sample weights; AdaBoost.R2 predicts the weighted median.
  - StackingClassifier uses stratified shuffled folds and drops the binary column.
- **HGB / LGBM (#47):**
  - HGB handles NaN natively, with a learned split direction.
  - `max_bins` must be ≤ 255 (**breaking:** HGB default is now 255).
  - LGBM monotone constraints now hold globally.
- **Feature selection (#53):**
  - RFECV scores one elimination path per fold with a train/test scorer.
  - SelectFromModel uses absolute importances and a mean/median threshold.
  - k-NN mutual information (`mutual_info_regression` / `mutual_info_classif`).
  - SelectKBest gains chi2, MI scorers and p-values.
  - **Breaking:** the RFECV scorer signature changed and
    `MutualInformationSelector` lost `n_bins`.
- **Transformers (#54):**
  - OneHotEncoder: real categories, `handle_unknown`, `drop`, `min_frequency` /
    `max_categories`, `inverse_transform` and sparse output.
  - OrdinalEncoder: `UseEncodedValue`.
  - QuantileTransformer: two-sided interpolation and inverse.
  - PowerTransformer: Brent-optimised λ and Box-Cox.
  - KBinsDiscretizer: kmeans strategy and duplicate-edge removal.
  - VarianceThreshold / StandardScaler: sklearn constant-feature handling.
- **KMeans (#58):** `n_init`, `KMeansInit`, data-relative `tol`, and empty-cluster
  relocation. One shared greedy k-means++ is used by MiniBatchKMeans, GMM and
  BGMM. Assignment uses GEMM plus rayon.
- **Gaussian mixtures (#59):**
  - Tied/Spherical covariance types and k-means initialisation.
  - The precision Cholesky is computed once per iteration.
  - New `score_samples` / `bic` / `aic` / `converged`.
  - BGMM `lower_bound` is now the real ELBO.
  - A singular covariance is now an error.
- **SVM (#64):**
  - Multiclass SVC/NuSVC use one-vs-one, with `decision_function_shape` and
    `break_ties`.
  - `class_weight` and `probability = true` (Platt + pairwise coupling).
  - Sigmoid and precomputed kernels.
  - **Breaking:** `gamma` defaults to `Gamma::Scale`, and the polynomial kernel
    takes `gamma`.
- **Naive Bayes (#67):**
  - **Breaking:** GaussianNB `var_smoothing` is relative to the largest feature
    variance.
  - GaussianNB gains `priors`.
  - Multinomial/Bernoulli gain `partial_fit`, `sample_weight` and
    `fit_prior`/`class_prior`.
  - `predict_log_proba` is computed in log space.
  - New ComplementNB and CategoricalNB.
- **Gaussian processes (#68):**
  - **Breaking:** `fit` optimises kernel hyperparameters (analytic LML gradients,
    bounded L-BFGS-B in `anofox_ml_core::optim`).
  - ARD kernels and per-parameter bounds.
  - New `predict_cov` / `sample_y`.
  - The White kernel matches sklearn.
  - The GPC objective bug is fixed, and GPC uses erf-based probabilities.
- **MLP (#69):**
  - **Breaking:** sklearn L2 scaling, stopping rule, Glorot initialisation and a
    single logistic output for binary problems.
  - New lbfgs solver.
  - SGD momentum/Nesterov with learning-rate schedules.
  - Early stopping, `partial_fit` / warm start and multi-output regression.

### Cross-validation designs (#74–#79)

New `anofox_ml_core::splitters` and `anofox_ml_core::validation` modules.
Every splitter returns `Vec<CvSplit>` for the `*_with_splits` helpers.

- **Group splitters (#74):**
  - `group_k_fold` now balances folds like sklearn (largest group to the
    lightest fold). **Breaking:** fold assignment changed.
  - New `group_k_fold_shuffled`, `stratified_group_k_fold`,
    `leave_one_group_out`, `leave_p_groups_out` and `group_shuffle_split`.
  - Unshuffled splits match sklearn 1.8.
- **Purged CV (#75):** `purged_k_fold` and `combinatorial_purged_cv` (López de
  Prado) for labels spanning a time interval, with purging and embargo. CPCV
  also returns the backtest paths.
- **Spatial CV (#76):** `spatial_block_k_fold` (grid blocks), `spatial_buffer`
  for any splits, and `buffered_leave_one_out`, with Euclidean or haversine
  distance.
- **`permutation_test_score` (#77):** p-value of a CV score against label
  permutations, optionally within groups.
- **Bootstrap validation (#78):** `bootstrap_splits` (+ stratified) and
  `bootstrap_validate`, which returns the apparent, out-of-bag, .632, .632+
  and optimism-corrected errors.
- **From the sister crates (#79):**
  - `time_series_split_with(TimeSeriesSplitOptions { test_size, gap,
    max_train_size, min_train_size })`, matching sklearn.
  - `group_time_series_split` and `quantile_strata` /
    `stratified_k_fold_continuous` for continuous targets.
  - `nested_cross_validate`.

### Review P2 / P3

- **Kernel SVMs (#62, #64):** SVC, NuSVC, SVR, NuSVR and OneClassSVM share a
  port of libsvm's SMO solver (second-order working-set selection, gradient
  maintenance, shrinking, and an LRU kernel column cache bounded by
  `cache_size` MB). The full Gram matrix is no longer built.
  - NuSVC solves the ν dual directly instead of bisecting C.
  - SVC/NuSVC decision values match sklearn exactly.
  - A degenerate dual (e.g. ν too large) returns an error, as sklearn does.
  - **Breaking:** defaults follow libsvm: `tol = 1e-3`, `max_iter` is an SMO
    iteration cap (default 10⁷), and there are new `shrinking` / `cache_size`
    fields.

- **LDA / QDA (#66):**
  - LDA: sklearn `solver` `Svd` (default; handles collinear features),
    `Lsqr` and `Eigen`; `Shrinkage::{Fixed, Auto}` (Ledoit-Wolf on
    standardised data); `priors`; sklearn `transform` semantics.
  - QDA: per-class SVD with sklearn's `reg_param` (`(1 − r) S² + r`),
    `priors`, and batched scoring (no per-sample factorisation).
  - New `ledoit_wolf` and `oas` estimators. All match sklearn 1.8 to 1e-9.
  - **Breaking:** the default solver is now `Svd`; LDA's `shrinkage` is a
    `Shrinkage` enum; QDA's fitted model stores `rotations` / `scalings`
    instead of `sigmas`, and `with_reg` means sklearn's `reg_param`
    (default 0).

- **NMF / KernelPCA (#55):**
  - NMF has sklearn's coordinate-descent solver (default) and
    multiplicative updates for the Frobenius, Kullback-Leibler and
    Itakura-Saito losses, plus `alpha_W` / `alpha_H` / `l1_ratio`
    regularisation.
  - The NMF convergence check no longer forms WH.
  - `w` (the `fit_transform` result) is stored, and `transform` is
    deterministic and implements `Transform`.
  - NNDSVD zeroes entries below 1e-6, as sklearn does.
  - The CD and KL-MU results match sklearn 1.8.
  - KernelPCA builds the kernel with one GEMM, clips non-positive
    eigenvalues, projects zero-eigenvalue components to 0, adds
    `remove_zero_eig`, uses sklearn's sign convention, and matches sklearn
    to 1e-9.

- **Consistency (#41):**
  - New `anofox_ml_core::labels` (`encode_labels`, exact total-order label
    comparison, shared by metrics).
  - CSV reader: empty and `NA` fields are read as NaN, returned headers
    match the columns of X, and `CsvError` converts into `RustMlError`.
  - `CsrMatrix::from_triplets` checks bounds; new `try_from_triplets`.

- **Parallelism (#38):**
  - New `grid_search_cv_with_splits_par` and `cross_val_score_with_splits_par`
    evaluate in parallel with rayon and give results identical to the
    sequential versions.
  - Multi-output targets and ColumnTransformer branches are fitted in
    parallel.
  - `permutation_importance` copies X once per feature instead of once per
    repeat.
  - Row and column selection use ndarray's `select`.

### Algorithms — performance

- **Agglomerative Ward** now uses Müllner's O(n²) nn-chain algorithm by
  default. Earlier attempts produced an inconsistent flat clustering because
  nn-chain processes merges in chain order rather than distance order. The
  fix runs the chain to a single cluster, sorts the recorded merges by
  distance ascending, and applies them via DSU until `n_clusters` remain.
  Set `RUSTML_AGGLO_NAIVE=1` to force the O(n³) sweep (regression-test
  cross-check).
