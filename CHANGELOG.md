# Changelog

All notable changes to anofox-ml are recorded here. The format is loosely based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Correctness parity with sklearn (round 2)

- **HistGradientBoosting bins like sklearn.** The engine gets
  `BinningMethod::{LightGbm, Sklearn}`, and the HGB facades use sklearn's
  `_BinMapper` thresholds: midpoints between distinct values, or
  percentile midpoints, with a 200 000-row subsample. Before, they used
  LightGBM's greedy bins, so even a single split differed from sklearn.
  - The regressor now matches sklearn to about 1e-7 (sklearn keeps
    gradients in float32).
  - The classifier matches up to exact-gain ties, which sklearn breaks by
    float32 summation order.
  - `LgbmRegressor` / `LgbmClassifier` keep LightGBM's binning.
- **NMF initialisation follows sklearn.**
  - The default is sklearn's `init=None`: `nndsvda`, or `random` when
    `n_components > min(n_samples, n_features)` (which is now allowed with
    a random init). New `NndsvdA` and `NndsvdAr`.
  - The first NNDSVD component uses `|U₀|`, `|V₀|`. Before, a negatively
    signed leading singular vector would have zeroed it.
  - Random init draws `|N(0,1)|·sqrt(mean(X)/k)`, sklearn's distribution.
  - The default fit matches sklearn to 1e-13.
  - Python `NMF` gains `init`, `components_` and `reconstruction_err_`.
  - **Breaking:** the default init changes from `Nndsvd` to `Auto`.

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

- **Regression conversions and small solvers (#32):**
  - ndarray → faer conversion copies standard-layout arrays through a
    strided view instead of bounds-checked indexing; OLS/WLS, Ridge,
    ElasticNet and Quantile predict in ndarray from cached coefficients
    (aliased / `NaN` coefficients count as 0, as upstream).
  - Weighted Ridge delegates to the upstream solver (weighted centring and
    `√w` row scaling, as sklearn does) and matches sklearn 1.8.
    **Breaking:** `FitWeighted` returns `FittedRidgeRegressor`;
    `FittedWeightedRidgeRegressor` is removed.
  - OMP grows the Cholesky factor of the active Gram matrix by one row per
    step (sklearn's `_cholesky_omp`), stops on linear dependence and
    matches sklearn 1.8. **Breaking:** the default is `max(⌊0.1·d⌋, 1)`
    coefficients, `tol` overrides `n_nonzero_coefs`, and
    `n_nonzero_coefs > d` is an error.
  - New `SvmKernel::matrix` / `SvmKernel::gram` build kernel matrices from
    one GEMM; KernelRidge and SVC prediction use them.

- **Weights, class weights and pipelines (#37):**
  - New shared `anofox_ml_core::ClassWeight` (`Balanced`, `Manual`) with
    `compute_sample_weight`, `effective_sample_weight` and
    `check_sample_weight`. Trees, SVC, logistic regression, LGBM and SGD use
    it. **Breaking:** `svm::ClassWeight::Map` and
    `regression::ClassWeight::Custom` are now `ClassWeight::Manual`;
    `LgbmClassWeight` is an alias of the shared type.
  - `FitWeighted` for decision trees, random forests, extra trees, gradient
    boosting, AdaBoost, LGBM and SGD. Weighted gradient boosting, AdaBoost
    and SGD match sklearn 1.8 exactly.
  - Random-forest `class_weight` is computed on the full target, not on each
    bootstrap draw (sklearn's `balanced`). **Breaking:** forest results with
    `class_weight` change.
  - Weighted tree splits use running weighted sums (O(n) per feature instead
    of O(n²)); unit weights reproduce the unweighted tree exactly.
  - SGD's `optimal` and `invscaling` learning-rate schedules now follow
    sklearn (`t₀ = 1/(typw·α)`, `eta0 / t^power_t`). **Breaking:** results
    with these schedules change. `SgdClassifier` gains `class_weight`.
  - `Pipeline::fit` borrows the pipeline (refit per CV fold), and `Pipeline`
    implements `Fit`; `FittedPipeline` implements `Predict`, `Transform` and
    `PredictProba` (with the new `Pipeline::set_classifier`).
  - The target reaches every pipeline step: `FitTransform::fit_transform`
    takes `y: Option<&Array1<F>>`, and the new `Supervised(...)` adapter puts
    `Fit`-based transformers (`SelectKBest`, `Rfe`, `Rfecv`,
    `SequentialFeatureSelector`, which now implement `Fit` / `Transform`)
    into a `Pipeline`, `FeatureUnion` or `ColumnTransformer`.
    `FitUnsupervised` gains the provided hook `fit_with_target`.
    **Breaking:** custom `FitTransform` impls take the extra argument.
  - `ColumnTransformer` and `FeatureUnion` compute each branch's training
    output once instead of fitting and transforming again.
  - Unneeded `unsafe impl Send/Sync` removed from the pipeline types.
  - New `cross_validate_weighted_with_splits` /
    `cross_val_score_weighted_with_splits` pass the fold's weights to the fit
    and to the scorer.
  - Not done: view-based (`ArrayView2`) trait inputs.

- **Persistence (#39):**
  - Saved models carry a versioned envelope (`format`, `format_version`,
    `crate_version`, `type_name`, `payload`; bincode files start with the
    magic bytes `ANOFOXML`). Loading checks it, so a file read as the wrong
    type or written in a newer format fails with a clear error. Type names
    are compared without module paths. Files without an envelope still
    load.
  - New `ModelHeader`, `json_header` / `bincode_header` (read the header
    only), `short_type_name` and `ModelBundle { config, fitted }`.
  - Python `save_json` / `save_bincode` store the estimator's configuration
    with the fitted model, so `load_json` / `load_bincode` restore the
    hyperparameters instead of the defaults (older files still load, with
    default hyperparameters).
  - Pipelines with boxed steps remain non-persistable (documented).

- **Python bindings (#40):**
  - Rebuilt on one generic native class (`_native.Model`) backed by a Rust
    registry of 95 estimators: linear and GLM regression, logistic,
    SGD, passive-aggressive, trees, forests, extra trees, gradient boosting,
    AdaBoost, bagging, HistGradientBoosting, LGBM, isolation forest, SVMs,
    naive Bayes, k-NN, LOF, LDA/QDA, MLP, the scalers / imputers /
    decompositions / selectors, clustering and manifold learners.
  - A Python layer gives them sklearn's API: `get_params` / `set_params`,
    `clone` / `GridSearchCV` / pipeline compatibility (`__sklearn_tags__`),
    `fit` returning `self`, `random_state`, `predict_proba` /
    `decision_function` / `score_samples` where available, `sample_weight`,
    string labels, `NotFittedError`, pickling (joblib / multiprocessing)
    through the versioned bincode envelope.
  - Fitting and prediction release the GIL.
  - More metrics, `train_test_split`, `KFold`, `cross_val_score`.
  - CI builds the wheel and runs the Python test suite.
  - **Breaking:** sklearn class names (`KNeighborsClassifier`, `SVC`, `PCA`,
    `TSNE`, ...; the old names remain as aliases), `random_state` instead of
    `seed`, and `fit` returns `self`. The old `save_json` / `load_json`
    methods are replaced by pickling.
  - `Lasso` / `ElasticNet` use anofox-regression's penalty scaling, which
    does not yet match sklearn's `alpha` (#72).
- **Tree and boosting performance (#48):**
  - Decision trees presort every feature once per tree and split nodes by
    stable partition (no per-node sorting); classes are encoded once per
    tree instead of hashed per sample move. Trees are byte-identical to
    before and fit 2–2.7× faster.
  - New `DecisionTreeClassifier::fit_on_rows` / `DecisionTreeRegressor::
    fit_on_rows` fit on (bootstrap) row lists without copying `X`; random
    forests, bagging, gradient-boosting subsampling and AdaBoost.R2 use
    them (same results, no per-tree copy of `X`).
  - LGBM: histogram subtraction (only the smaller child's histograms are
    built), per-feature bin counts, column-major bins for histograms and
    row-major bins for prediction, no per-row allocations, O(1) categorical
    routing, histograms built in parallel for large nodes. Fitting is
    3.5–4× faster with the same predictions.
  - Weighted splits use running weighted sums (see #37) and the RF OOB score
    uses an in-bag mask.

- **HistGradientBoosting on the LGBM engine (#49):**
  - `HistGradientBoostingRegressor` / `HistGradientBoostingClassifier` are
    facades over the LGBM engine with sklearn's defaults
    (`max_leaf_nodes = 31`, `max_depth = None`, leaf-wise growth). They gain
    native missing-value handling, `FitWeighted`, feature importances and
    the engine's speed-ups; the separate histogram implementation is gone.
  - **Breaking:** `max_depth` is `Option<usize>` (default `None`), there is
    a new `max_leaf_nodes`, results change, and fitted HGB models saved by
    earlier versions do not load.
  - Ensemble helpers are shared instead of duplicated (`majority_vote`,
    `select_rows`, the boxed-estimator traits of voting / stacking); the
    tree classifier and regressor share one tree engine (#48).

- **Tree / ensemble parity, part 1 (#50):**
  - Decision trees gain `min_impurity_decrease`, `min_weight_fraction_leaf`,
    `max_leaf_nodes` (best-first growth) and `ccp_alpha` (minimal
    cost-complexity pruning); all match sklearn 1.8.
  - Leaves store their impurity, weighted sample totals and exact
    per-class weights (previously weighted counts were rounded to
    `w·1000`); feature importances use weighted node sizes and leaf
    impurities, as sklearn. Older saved trees still load.
  - ExtraTrees: `criterion` (Gini / Entropy) and an `Array2`
    `predict_proba` (plus the `PredictProba` trait). **Breaking:** the
    inherent `predict_proba` returned `Vec<Vec<(F, F)>>`.
  - GradientBoostingRegressor: `AbsoluteError`, `Huber` and `Quantile`
    losses with sklearn's terminal-region updates, early stopping
    (`n_iter_no_change`, `validation_fraction`, `tol`) and `max_features`.
  - GradientBoostingClassifier gets the same early stopping (on a
    stratified hold-out, scored by the binomial / multinomial log loss) and
    `max_features`. In both estimators each round's tree now draws its
    features with its own seed. Before, every round reused one seed, so all
    trees drew the same feature sequence.
  - Bagging{Classifier,Regressor}: `max_features` / `bootstrap_features`
    (one feature subset per estimator, as sklearn), `oob_score` with
    `oob_decision_function` / `oob_prediction`, `FitWeighted`, and
    per-estimator tree seeds. **Breaking:** `BaggingClassifier::predict` is
    the argmax of the averaged probabilities (sklearn) instead of a majority
    vote.

- **LGBM parity, part 2 (#50):** checked against LightGBM 4.7.
  - The L1 objective starts from the (weighted) median and refits every
    leaf to the median residual of its rows (LightGBM's
    `RenewTreeOutput`), with `sign(0) = 0` gradients. Previously it used the
    mean and plain Newton leaves.
  - Baselines use sample weights: the weighted mean (L2 / Huber), the
    weighted positive rate (binary) and weighted class priors (multiclass).
  - LgbmClassifier's softmax hessian carries LightGBM's `K / (K - 1)`
    factor. The HistGradientBoosting facade keeps sklearn's hessian, and
    its multiclass output matches sklearn.
  - Categorical binning keeps the most frequent categories, at most 254 and
    none rarer than `min_data_in_bin`. Every other category (and any
    category unseen in training) goes to the "other"/missing bin. Before,
    categories beyond 254 collapsed into the last bin, and unseen
    categories landed on a real category.
  - Categorical splits follow LightGBM: one-vs-rest up to
    `max_cat_to_onehot`, otherwise `G / (H + cat_smooth)` ordering scanned
    from both ends with `cat_l2`, `max_cat_threshold` and
    `min_data_per_group` (new `CategoricalSplitParams`, LightGBM defaults).
    Missing and "other" categories always go right.
  - **Breaking:** `huber_delta` defaults to LightGBM's 0.9 (was 1.0), and
    results change for the L1 objective, weighted fits, multiclass LGBM and
    categorical features.

- **Neighbours, part 1 (#60):**
  - New `NeighborIndex` (sklearn's `NearestNeighbors`): batched, parallel
    `kneighbors`, `kneighbors_self` (each sample excluded from its own
    list), `radius_neighbors` and `radius_neighbors_self`.
    `NeighborAlgorithm::Auto` follows sklearn's rule (brute force for
    non-Minkowski metrics, more than 15 features or
    `k >= n_samples / 2`). Results are exact and identical across
    algorithms, with ties going to the lower index.
  - The KD-tree is rewritten. It stores a flat permuted copy of the data,
    uses leaves of 30 points and splits at the median of the widest
    dimension (`select_nth_unstable`). It prunes with node bounding boxes,
    so it serves every Minkowski metric. Building it no longer sorts at
    every level.
  - Euclidean brute force pre-selects candidates with a matrix product,
    then ranks them by exact distances (about 4× faster, still exact).
  - New metrics `Chebyshev`, `Minkowski(p)` and `Precomputed`.
  - kNN estimators: `algorithm`, `kneighbors`, `predict_proba` /
    `PredictProba` for the classifier. `weights = Distance` uses sklearn's
    rule: neighbours at distance 0 take all the weight, replacing the old
    `1e15` cap.
  - New `RadiusNeighborsClassifier` (with `outlier_label`) and
    `RadiusNeighborsRegressor` (NaN without neighbours), also in the
    Python bindings, which gain `p`, `algorithm` and the new metrics.
  - LOF uses the shared index.
  - All of this matches sklearn 1.8 (Rust and Python parity tests).
  - **Breaking:** the public `kdtree` module is gone, and fitted kNN
    models saved by earlier versions do not load.

- **Neighbours, part 2 (#60):** clustering and manifold learning query the
  shared `NeighborIndex` instead of building dense n×n distance matrices.
  - DBSCAN computes every eps-neighbourhood up front and expands clusters
    depth-first like sklearn's `dbscan_inner`, so border points get
    sklearn's labels. It gains `metric` (including `Precomputed`) and
    `algorithm`; `predict` queries the index of core samples.
  - OPTICS takes core distances from the index and picks the next point
    from a heap with lazy deletion. Before, a linear scan plus
    `stack.contains` made it about O(n³). It now exposes `core_distances`
    and `predecessor`; ordering, reachability and labels match sklearn.
  - HDBSCAN takes core distances from the index, and its Prim MST
    computes distances on the fly (O(n) memory instead of n²).
  - SpectralClustering (k-NN affinity), Isomap and LLE build their k-NN
    graphs with the index. Isomap's geodesics use parallel Dijkstra on the
    sparse graph (sklearn's `path_method="D"`) instead of O(n³)
    Floyd–Warshall, and its embedding matches sklearn.

- **t-SNE (#61)** follows sklearn's `TSNE`:
  - The optimiser uses per-coordinate gains (+0.2 / ×0.8, minimum 0.01).
    It runs 250 exploration iterations with momentum 0.5 and exaggerated P,
    then momentum 0.8. KL is checked every 50 iterations, with
    `n_iter_without_progress` and `min_grad_norm` stopping. Before, there
    were no gains, exaggeration lasted `n_iter / 4` iterations, and the
    momentum switch was fixed at 250.
  - `learning_rate: Option<f64>` (`None` = `"auto"`), `max_iter`
    (default 1000; `n_iter` still deserialises), and `init`: `TSneInit::Pca`
    by default, or `Random` / `Array`.
  - Perplexities are searched over single-precision squared distances, as
    sklearn does.
  - Barnes–Hut is the default. It uses a sparse P from the
    `⌊3·perplexity + 1⌋` nearest neighbours (no more dense n×n P or
    distance matrix) and a port of sklearn's space-partitioning tree, so
    it works for 1–3 output dimensions instead of silently falling back to
    exact. It runs in parallel.
  - The affinities, the first objective and the first optimiser steps
    match sklearn to rounding; full runs match its iteration counts and
    mean KL. On 3,000 points Barnes–Hut takes 2.8 s (sklearn: 4.2 s) and
    reaches the same KL.
  - **Breaking:** `learning_rate` is an `Option`, `n_iter` is `max_iter`,
    and the defaults change (Barnes–Hut, PCA init, automatic learning
    rate, 1000 iterations). Perplexity must be below `n_samples`; it was
    silently clamped before.

- **LLE and spectral clustering (#61):**
  - New `sparse_eigen::smallest_eigenpairs` (manifold crate). It uses a
    dense eigendecomposition up to n = 500, and above that shift-invert
    Lanczos (faer's thick-restart Lanczos on `(M + εI)⁻¹` through a sparse
    Cholesky factorisation, like sklearn's ARPACK with `sigma`).
  - LLE regularises with sklearn's `reg · trace(G)`; it was
    `reg · trace / k`, k times weaker. It builds the sparse
    `(I − W)ᵀ(I − W)` in O(n·k²) instead of a dense O(n³) product, uses the
    new eigensolver, and reports `reconstruction_error`. Embeddings match
    sklearn for n = 100 and n = 800.
  - New `spectral_embedding` (sklearn's `manifold.spectral_embedding`).
  - SpectralClustering now follows sklearn. The k-NN affinity is
    `0.5·(C + Cᵀ)` with self loops, replacing an OR of the two directions.
    The embedding is divided by √degree, with sklearn's deterministic sign
    flip, instead of being row-normalised. k-means gets `n_init` restarts
    (default 10). Large k-NN graphs use the sparse eigensolver. Python
    gains `affinity="nearest_neighbors"`, `n_neighbors` and `n_init`.

- **AgglomerativeClustering (#61)** builds the tree like sklearn.
  - Single linkage uses Prim's MST with O(n) memory.
  - Complete, average and Ward use scipy's nearest-neighbour chain with
    Lance–Williams updates, O(n²) in total. Before, single, complete and
    average ran a naive O(n³) loop.
  - Merges are sorted and relabelled, exposing `children` and `distances`.
  - Labels come from a port of sklearn's `_hc_cut`, so the label numbering
    matches too.
  - New `distance_threshold` (also in Python).
  - The `RUSTML_AGGLO_NAIVE` environment switch is removed.
  - Children, distances and labels match sklearn exactly for all four
    linkages.

- **ClassicalMds (#61)** follows sklearn 1.8's `ClassicalMDS`. It gains a
  `metric`, including `Precomputed` dissimilarity matrices, its main use.
  It uses sklearn's sign convention. With Euclidean features it solves the
  `d × d` covariance eigenproblem (O(n·d²)) instead of an n×n one (O(n³)).
  It matches sklearn for Euclidean, Manhattan and precomputed input; Python
  adds `metric` and a `ClassicalMDS` alias.

- **MeanShift (#61)** follows sklearn.
  - `bandwidth: Option<f64>` (`None` estimates it with the new
    `estimate_bandwidth`).
  - New `seeds`, `bin_seeding` / `min_bin_freq` and `cluster_all`.
  - The stopping threshold is `1e-3 · bandwidth`.
  - Seeds shift in parallel through the neighbour index, instead of
    copying every row on every shift.
  - Centres are sorted by intensity and suppressed within the bandwidth,
    where before the first-found centre won within bandwidth / 2.
  - Centres, labels and `n_iter` match sklearn exactly.
  - **Breaking:** `bandwidth` is an `Option` and `tol` is gone.

- **OPTICS ξ extraction (#61):** new `OpticsClusterMethod::Xi`
  (`Optics::xi`), a port of sklearn's `cluster_optics_xi`. It covers steep
  areas, `min_cluster_size` and predecessor correction, and exposes
  `cluster_hierarchy`. Labels and hierarchy match sklearn exactly. DBSCAN
  extraction stays the Rust default; the Python `OPTICS` now defaults to
  ξ, like sklearn.

- **Birch (#61)** is now a real CF-tree, ported from sklearn.
  - A sample descends to the closest subcluster and merges when the merged
    radius stays within `threshold` (before: distance to the centroid).
  - Nodes split around their farthest pair, up to new roots
    (`branching_factor`, default 50).
  - Subcluster centres come out in leaf order.
  - The global step is AgglomerativeClustering, as in sklearn; it was
    KMeans, which ignored subcluster sizes.
  - Centres, subcluster labels and sample labels match sklearn exactly in
    the parity tests. On other data the leaf order can differ on ULP-level
    ties, but the partition is the same.
  - **Breaking:** the `seed` field is gone.

- **Incremental preprocessing (#56):** new core trait
  `PartialFitUnsupervised` (`partial_fit(state, X)`, plus
  `partial_fit_weighted`).
  - `StandardScaler` merges batches with sklearn's
    `_incremental_mean_and_var`, per feature and ignoring NaN. It takes
    `sample_weight` through `FitUnsupervisedWeighted` and
    `partial_fit_weighted`, and exposes `var()` and `n_samples_seen()`.
  - `MinMaxScaler` and `MaxAbsScaler` implement `partial_fit`.
  - New `IncrementalPca`, a port of sklearn's `IncrementalPCA`: batch SVD
    with mean correction; `fit` runs in batches of `batch_size` rows.
  - Scalers and IncrementalPCA match sklearn 1.8 to 1e-10.

- **Imputation (#56):**
  - New `MissingIndicator` (sklearn's, `features` = missing-only / all,
    `error_on_new`).
  - `SimpleImputer` gains sklearn's `add_indicator` and
    `keep_empty_features`, and `sample_weight` through
    `FitUnsupervisedWeighted` (weighted mean, median and mode; this is an
    extension, sklearn has no weights here).
  - **Breaking:** an all-missing column is dropped from the output, as in
    sklearn (its `fill_values` entry is NaN); before, `fit` returned an
    error. `keep_empty_features(true)` keeps it and imputes 0.

- **KnnImputer (#56):** a port of sklearn's `KNNImputer` with the
  `nan_euclidean` metric, `uniform` / `distance` weights, `add_indicator`
  and `keep_empty_features`. Rows to impute are processed in parallel.
  Matches sklearn 1.8 to 1e-12.

- **PolynomialFeatures (#56)** gains sklearn's `include_bias` and
  `degree=(min, max)` (`with_degree_range`), `powers()` and
  `feature_names_out()`. Each degree-k column is built from its degree-k−1
  prefix column times one feature, instead of recomputing every power.
  `degree = 0` with a bias is allowed, as in sklearn. Output matches
  sklearn 1.8.

- **TargetEncoder (#56):** a port of sklearn's `TargetEncoder`.
  - `smooth` is `Auto` (empirical Bayes) or a fixed value.
  - Continuous, binary and multiclass targets (one-vs-rest columns grouped
    by feature). Unknown categories map to the target mean; NaN is its own
    category.
  - `fit_transform` cross-fits (stratified folds for class targets).
  - New core hook `Fit::fit_transform_supervised`: `Supervised(encoder)` in
    a pipeline, FeatureUnion or ColumnTransformer gets the cross-fitted
    training output, as sklearn's pipeline does.
  - Encodings and `transform` match sklearn 1.8 to 1e-12. Fold assignment
    uses this crate's seeded splitters, so cross-fitted values differ from
    sklearn's.

- **RcsTransformer (#56):** restricted cubic spline expansion per feature.
  It wraps `anofox_regression::splines::Rcs` (`rms::rcs`): default knots
  at the RMS quantiles or explicit knots, `norm` options, rms-style
  column names. sklearn has no counterpart.

- **IterativeImputer (#56):** a port of sklearn's `IterativeImputer`.
  - Chained per-feature regressions; BayesianRidge by default, or any
    `ImputationEstimator` (`AnyRegressor` adapts a `Fit` + `Predict`
    regressor).
  - sklearn's early stopping, imputation orders, `skip_complete`, bounds,
    `n_nearest_features`, `add_indicator` and `keep_empty_features`.
  - `sample_posterior` draws from the truncated predictive normal.
  - `transform` replays the fitted sequence.
  - With `sample_posterior = false` it matches sklearn 1.8 (BayesianRidge
    to 1e-8, LinearRegression to 1e-9). Random draws use this crate's
    generator, so they differ from sklearn's. BayesianRidge's
    `predict_std` still has the centring deviation recorded in #72.
  - anofox-ml-preprocessing now depends on anofox-ml-regression.
- **AregImputer (#56):** multiple imputation through
  `anofox_regression::imputation::areg_impute` (`Hmisc::aregImpute`). It
  returns `n_impute` completed data sets to pool with Rubin's rules.

- **Python / facade (#56):**
  - New Python classes `IncrementalPCA`, `MissingIndicator`, `KNNImputer`,
    `IterativeImputer` (BayesianRidge estimator) and `TargetEncoder`, whose
    `fit_transform` returns the cross-fitted encoding.
  - `SimpleImputer` gains `add_indicator` / `keep_empty_features`, and
    `PolynomialFeatures` gains `include_bias` and a `degree` tuple.
  - `StandardScaler` and `SimpleImputer` accept `sample_weight`.
  - `partial_fit` is not exposed in Python yet.
  - The new Rust types are in the `anofox_ml` prelude.
  - `KnnImputer` breaks exact distance ties by training-row order. sklearn
    leaves the order unspecified (`argpartition`), so tied cases can differ.

- **Bagging with any base estimator (#50):** `BaggingRegressor::with_estimator(e)`
  and `BaggingClassifier::with_estimator(e)` bag any `Fit` + `Predict`
  estimator, like sklearn's `estimator=` parameter. They keep the sampling
  options (rows, `max_features`, `bootstrap_features`, OOB score).
  - Each estimator is fitted on its drawn rows and columns.
    `fit_weighted` passes the drawn weights on to `FitWeighted` bases.
  - The classifier votes; `predict_proba` is the vote fraction, as sklearn
    does for bases without `predict_proba`.

- **Gradient boosting `warm_start` (#50):**
  `GradientBoostingRegressor::fit_warm_start(prev, X, y, w)` and the
  classifier's counterpart add trees to a fitted model until there are
  `n_estimators`. The earlier rounds' random draws are replayed, so
  without early stopping the result is identical to one fit with
  `n_estimators` trees, which is sklearn's guarantee. The early-stopping
  history restarts with each call, as in sklearn. Not exposed in Python
  yet.

- **f32 input for clustering / manifold / LOF (#61):**
  - Affected estimators: AgglomerativeClustering, AffinityPropagation,
    GaussianMixture, BayesianGaussianMixture, Birch, SpectralClustering,
    OPTICS, MeanShift, HDBSCAN, Isomap, LLE, ClassicalMds, t-SNE and
    LocalOutlierFactor.
  - They now implement `FitUnsupervised<F>` (and `Predict<F>` /
    `PredictProba<F>` where available) for any `F: Float`.
  - They compute in f64 and convert at the boundary (new core helpers
    `as_f64` / `from_f64`, zero-copy for f64). f32 data gives the same
    result as the same data in f64. Fitted attributes stay f64.

- **SVM persistence:** `Gamma` now round-trips through bincode, so fitted
  SVMs can be saved with `save_bincode` (they previously failed to load).
- More configs derive `Serialize` / `Deserialize` (regression, clustering,
  manifold, LDA/QDA, isolation forest, LOF, decompositions).

### Algorithms — performance

- **Agglomerative Ward** now uses Müllner's O(n²) nn-chain algorithm by
  default. Earlier attempts produced an inconsistent flat clustering because
  nn-chain processes merges in chain order rather than distance order. The
  fix runs the chain to a single cluster, sorts the recorded merges by
  distance ascending, and applies them via DSU until `n_clusters` remain.
  Set `RUSTML_AGGLO_NAIVE=1` to force the O(n³) sweep (regression-test
  cross-check).
