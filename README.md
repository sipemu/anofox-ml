# anofox-ml

A scikit-learn-inspired machine learning library for Rust, built on ndarray.

[![CI](https://github.com/sipemu/anofox-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/sipemu/anofox-ml/actions/workflows/ci.yml)
[![CodSpeed](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://app.codspeed.io/sipemu/anofox-ml?utm_source=badge)
[![Crates.io](https://img.shields.io/crates/v/anofox-ml.svg)](https://crates.io/crates/anofox-ml)
[![Documentation](https://docs.rs/anofox-ml/badge.svg)](https://docs.rs/anofox-ml)
[![codecov](https://codecov.io/gh/sipemu/anofox-ml/branch/master/graph/badge.svg)](https://codecov.io/gh/sipemu/anofox-ml)
[![Rust](https://img.shields.io/badge/rust-1.87%2B-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#license)

## Features

| Category | anofox-ml | scikit-learn equivalent |
|---|---|---|
| **Preprocessing & scaling** | `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, `Normalizer`, `Binarizer`, `KBinsDiscretizer`, `PolynomialFeatures`, `PowerTransformer`, `QuantileTransformer`, `SimpleImputer`, `OneHotEncoder`, `OrdinalEncoder`, `LabelEncoder` | `sklearn.preprocessing`, `sklearn.impute` |
| **Decomposition** | `Pca`, `TruncatedSvd`, `KernelPca`, `Nmf` (NNDSVD), `FastIca` | `sklearn.decomposition` |
| **Cross-decomposition** | `PlsRegression` (PLS1), `Cca` | `sklearn.cross_decomposition` |
| **Manifold learning** | `ClassicalMds`, `Isomap`, `LocallyLinearEmbedding`, `TSne` | `sklearn.manifold` |
| **Feature selection** | `VarianceThreshold`, `MutualInformationSelector`, `SelectKBest`, `SelectFromModel`, `Rfe`, `Rfecv`, `SequentialFeatureSelector` | `sklearn.feature_selection` |
| **Neighbors** | `KnnClassifier`, `KnnRegressor` (KD-tree), `LocalOutlierFactor` | `sklearn.neighbors` |
| **Linear models** | `OlsRegressor`, `RidgeRegressor` (+CV, +sample_weight), `LassoRegressor` (+CV), `ElasticNetRegressor` (+CV), `HuberRegressor`, `QuantileRegressor`, `IsotonicRegressor`, `WlsRegressor`, `LogisticRegressor`, `BayesianRidge`, `ARDRegression`, `Lars`, `LassoLarsIC`, `OrthogonalMatchingPursuit`, `RansacRegressor`, `TheilSenRegressor`, `KernelRidge`, `TransformedTargetRegressor`, `SgdClassifier`, `SgdRegressor`, `PassiveAggressiveClassifier`, `PassiveAggressiveRegressor` | `sklearn.linear_model` |
| **GLMs** | `PoissonRegressor`, `BinomialRegressor`, `TweedieRegressor`, `GammaRegressor` | `sklearn.linear_model` GLM family |
| **Discriminant analysis** | `LinearDiscriminantAnalysis` (+`transform`), `QuadraticDiscriminantAnalysis` | `sklearn.discriminant_analysis` |
| **Trees** | `DecisionTreeClassifier` (+`predict_proba`), `DecisionTreeRegressor` | `sklearn.tree` |
| **Ensemble** | `RandomForest{Classifier,Regressor}`, `ExtraTrees{Classifier,Regressor}`, `GradientBoosting{Classifier,Regressor}`, `HistGradientBoosting{Classifier,Regressor}`, `LgbmClassifier`, `LgbmRegressor`, `CatBoost{Classifier,Regressor}`, `Bagging{Classifier,Regressor}`, `AdaBoost{Classifier,Regressor}`, `Voting{Classifier,Regressor}`, `Stacking{Classifier,Regressor}` (+`push_proba`), `CalibratedClassifierCV`, `IsolationForest` (rayon-parallel); `Perpetual{Classifier,Regressor}` with the optional `perpetual` feature | `sklearn.ensemble`, `lightgbm`, `catboost`, `perpetual` |
| **Clustering** | `KMeans`, `MiniBatchKMeans`, `Dbscan`, `Hdbscan`, `Optics`, `Birch`, `AgglomerativeClustering` (Ward/single/complete/average), `SpectralClustering`, `MeanShift`, `AffinityPropagation`, `GaussianMixture`, `BayesianGaussianMixture` | `sklearn.cluster`, `sklearn.mixture` |
| **Naive Bayes** | `GaussianNB`, `MultinomialNB`, `BernoulliNB` (all with `predict_proba`) | `sklearn.naive_bayes` |
| **SVM** | `Svc`, `Svr`, `NuSvc`, `NuSvr`, `LinearSvc`, `LinearSvr`, `OneClassSvm` | `sklearn.svm` |
| **Gaussian processes** | `GaussianProcessRegressor` (RBF, Matern, RationalQuadratic, White, Constant, sums/products), `GaussianProcessClassifier` (Laplace) | `sklearn.gaussian_process` |
| **Neural networks** | `MlpClassifier`, `MlpRegressor` | `sklearn.neural_network` |
| **Multi-output** | `MultiOutputRegressor`, `MultiOutputClassifier`, `RegressorChain`, `ClassifierChain` | `sklearn.multioutput` |
| **Text** | `CountVectorizer`, `TfidfVectorizer`, `HashingVectorizer` | `sklearn.feature_extraction.text` |
| **Inspection** | `permutation_importance` (rayon-parallel) | `sklearn.inspection` |
| **Metrics** | `accuracy_score`, `precision`, `recall`, `f1_score`, `confusion_matrix`, `roc_auc_score`, `roc_curve`, `precision_recall_curve`, `log_loss`, `brier_score_loss`, `matthews_corrcoef`, `cohen_kappa_score`, `silhouette_score`, `adjusted_rand_score`, `mse`, `mae`, `r2_score`, `mean_absolute_percentage_error`, `median_absolute_error`, `mean_squared_log_error`, ... | `sklearn.metrics` |
| **Utilities** | `train_test_split`, `cross_val_score`, `cross_validate`, `grid_search_cv`, `randomized_search_cv`, `halving_grid_search_cv`, `halving_random_search_cv`, `k_fold` (+ stratified / group / time-series / shuffle / leave-one-out), `learning_curve`, `validation_curve`, `Pipeline`, `ColumnTransformer`, `FunctionTransformer`, `FeatureUnion` | `sklearn.model_selection`, `sklearn.pipeline`, `sklearn.compose` |
| **I/O & persistence** | CSV reader with ndarray integration, versioned binary (MessagePack + zstd, tolerant to type changes) and JSON persistence for fitted models | `pandas.read_csv`, `joblib.dump` |

## Quick Start

Add anofox-ml to your project:

```toml
[dependencies]
anofox-ml = "0.4"
ndarray = "0.16"
```

Train a KNN classifier with standardized features:

```rust
use anofox_ml::prelude::*;
use ndarray::array;

fn main() -> anofox_ml::core::Result<()> {
    // Sample data
    let x_train = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0],
                          [8.0, 9.0], [9.0, 10.0], [10.0, 11.0]];
    let y_train = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    // Scale features
    let scaler = StandardScaler::new().fit(&x_train)?;
    let x_scaled = scaler.transform(&x_train)?;

    // Fit KNN classifier
    let knn = KnnClassifier::new(3);
    let model = knn.fit(&x_scaled, &y_train)?;

    // Predict and evaluate
    let x_test = array![[2.0, 3.0], [9.0, 10.0]];
    let x_test_scaled = scaler.transform(&x_test)?;
    let predictions = model.predict(&x_test_scaled)?;

    let acc = accuracy_score(&array![0.0, 1.0], &predictions);
    println!("Accuracy: {acc:.2}");

    Ok(())
}
```

## Architecture

anofox-ml is organized as a Cargo workspace with focused crates. You can depend on
the umbrella `anofox-ml` crate for everything, or pick individual crates for
smaller dependency trees.

```
anofox-ml (facade)
  +-- anofox-ml-core              Core traits, error types, Pipeline, utilities
  +-- anofox-ml-metrics           Classification, regression, clustering metrics
  +-- anofox-ml-preprocessing     Scalers, PCA, KernelPCA, NMF, FastICA, TruncatedSVD,
                               PLS, CCA, feature selection, RFE/RFECV/SFS
  +-- anofox-ml-neighbors         KNN with KD-tree, LocalOutlierFactor
  +-- anofox-ml-trees             CART decision trees with predict_proba
  +-- anofox-ml-ensemble          Random Forest, ExtraTrees, Gradient Boosting,
                               HistGradientBoosting, LightGBM-lite, AdaBoost,
                               Bagging, Voting, Stacking, Calibrated, IsolationForest
  +-- anofox-ml-cluster           KMeans, MiniBatchKMeans, DBSCAN, HDBSCAN, OPTICS,
                               Birch, Agglomerative, Spectral, MeanShift, AP,
                               GaussianMixture, BayesianGaussianMixture
  +-- anofox-ml-naive-bayes       Gaussian/Multinomial/Bernoulli NB
  +-- anofox-ml-discriminant      LDA (with transform) and QDA
  +-- anofox-ml-svm               SVC, SVR, NuSVC, NuSVR, LinearSVC/SVR, OneClassSVM
  +-- anofox-ml-regression        OLS, Ridge (+weighted), Lasso, ElasticNet, GLMs,
                               BayesianRidge, ARD, LARS, OMP, KernelRidge,
                               RANSAC, TheilSen, Tweedie, TransformedTarget
  +-- anofox-ml-linear            SGD, PassiveAggressive
  +-- anofox-ml-gaussian-process  GP regressor (5 kernels + composites) & classifier
  +-- anofox-ml-manifold          ClassicalMDS, Isomap, LLE, t-SNE
  +-- anofox-ml-neural-networks   MLPClassifier, MLPRegressor
  +-- anofox-ml-text              Count/Tfidf/Hashing vectorizers
  +-- anofox-ml-io                CSV loading
  +-- anofox-ml-conformal         Conformal prediction intervals: split conformal,
                               CV+, jackknife+, CV-minmax (checked against MAPIE)
  +-- anofox-ml-registry          Every estimator by name + JSON parameters
                               (used by the Python bindings and DuckDB)
```

### Type-state pattern

Estimators use a compile-time type-state pattern to separate unfitted
parameters from fitted models. Calling `fit()` on an unfitted struct returns a
distinct `Fitted*` type that implements `Predict` or `Transform`. This makes it
a compile error to call `predict()` on an unfitted estimator.

```
KnnClassifier --fit()--> FittedKnnClassifier --predict()--> Array1
StandardScaler --fit()--> FittedStandardScaler --transform()--> Array2
```

### Core traits

| Trait | Purpose |
|---|---|
| `Fit<F>` | Supervised fitting: `fit(&self, x, y) -> Fitted` |
| `FitUnsupervised<F>` | Unsupervised fitting: `fit(&self, x) -> Fitted` |
| `FitWeighted<F>` | Supervised fitting with per-sample `sample_weight` |
| `Predict<F>` | Generate predictions from fitted model |
| `PredictProba<F>` | Class probabilities; rows sum to 1 |
| `PredictLogProba<F>` | Log of `predict_proba` (auto-derived) |
| `DecisionFunction<F>` | Real-valued per-class decision scores |
| `RegressorScore<F>` / `ClassifierScore<F>` | `score()` (R² / accuracy) |
| `Transform<F>` | Transform feature matrix |
| `InverseTransform<F>` | Reverse a transformation |

## sklearn parity

Every estimator in `anofox-ml` is validated against scikit-learn 1.8.0 via golden
fixtures (`test_harness/generators/gen_*.py`) and corresponding Rust tests in
`crates/anofox-ml/tests/golden_*.rs`. Per-estimator parity notes — including
tolerances, sample-weight behaviour, missing options, and asymptotic
complexity — live under `validation/sklearn_parity/`.

The pinned sklearn version (1.8.0) is enforced by
`test_harness/check_sklearn_version.py`. Bumping the pin requires
`./test_harness/regenerate_all.sh` followed by a full `cargo test --workspace`
to confirm tolerances still hold.

### LightGBM and CatBoost parity

The boosting engines are validated against their reference libraries as
well (#127):

- `LgbmClassifier` / `LgbmRegressor` match **LightGBM 4.7.0**
  (`lgbm_parity.json`), including DART and exclusive feature bundling.
- `CatBoostClassifier` / `CatBoostRegressor` match **catboost 1.2.10**
  (`catboost_parity.json`, 104 golden cases, mostly to ~1e-16). That covers
  oblivious trees, ordered and plain boosting, RMSE / Logloss / MultiClass,
  categorical features through ordered CTRs (with feature combinations) and
  one-hot splits, and multi-step leaf estimation with backtracking. CatBoost's
  randomness is reproduced draw for draw: its generator, the permutations,
  MVS / Bayesian / Bernoulli bootstrap and the score noise. The estimators
  therefore default to CatBoost's own defaults, and a fit with
  `random_seed=s` gives CatBoost's model for that seed.

```rust
use anofox_ml::prelude::*;
use anofox_ml::ensemble::CatBoostClassifier;

// x: f64 matrix with category codes in columns 3 and 4.
let model = CatBoostClassifier::new()
    .with_iterations(500)
    .with_cat_features(vec![3, 4])
    .fit(&x, &y)?;
let proba = model.predict_proba(&x)?;
```

Exact agreement needs CatBoost's `thread_count=1`, because CatBoost sums
some statistics per thread. CatBoost also computes some statistics with
subtraction and keeps Logloss approximations as `exp(a)`; both differ from
ours only in the last bits and can rarely break a near-tie between two
splits differently. The generators live in
`test_harness/generators/gen_lgbm_parity.py` and `gen_catboost_parity.py`.

## Algorithms

See the feature table above for the full list. New since the original release:

- **Linear models**: BayesianRidge / ARDRegression, LARS / LassoLars /
  LassoLarsIC, OrthogonalMatchingPursuit, KernelRidge (with sample_weight),
  Tweedie / Gamma GLMs, TransformedTargetRegressor, PassiveAggressive,
  RANSAC, TheilSen, Ridge with `sample_weight`.
- **Cluster**: MiniBatchKMeans, AgglomerativeClustering (4 linkages),
  SpectralClustering, MeanShift, AffinityPropagation, Birch, OPTICS, HDBSCAN,
  GaussianMixture, BayesianGaussianMixture.
- **Decomposition / manifold**: TruncatedSVD, KernelPCA, NMF (NNDSVD),
  FastICA, ClassicalMDS, Isomap, LocallyLinearEmbedding, t-SNE.
- **Cross-decomposition**: PLSRegression, CCA.
- **Discriminant**: LinearDiscriminantAnalysis (with `transform`), QDA.
- **Gaussian processes**: regressor (5 kernels + composites) and Laplace
  classifier.
- **Outlier detection**: IsolationForest (rayon-parallel), LocalOutlierFactor.
- **Meta-estimators**: MultiOutputRegressor/Classifier, RegressorChain,
  ClassifierChain, StackingClassifier (with `predict_proba`), CalibratedClassifierCV.
- **Feature selection**: RFE, RFECV, SequentialFeatureSelector (forward).
- **Inspection**: `permutation_importance` (rayon-parallel).
- **Search**: `halving_grid_search_cv`, `halving_random_search_cv`.
- **Text**: CountVectorizer, TfidfVectorizer, HashingVectorizer.

### Metrics
- Classification: `accuracy_score`, `precision`, `recall`, `f1_score`, `confusion_matrix`, macro/micro/weighted averaging
- Regression: `mse`, `mae`, `r2_score`

### Utilities
- `train_test_split`, `cross_val_score`, `Pipeline`

## Benchmarks

anofox-ml outperforms scikit-learn across all benchmarks, with up to 22x speedups
on critical operations. Measurements taken on the same machine with identical
datasets and parameters.

### Phase 1: original sweep (verified)

| Algorithm | Operation | sklearn (ms) | anofox-ml (ms) | Speedup |
|---|---|--:|--:|--:|
| **GaussianNB** | fit 5000×20 | 6.34 | 0.29 | **21.8x** |
| **DecisionTree** | predict 5000×20 | 0.10 | 0.007 | **14.6x** |
| **KNN** | predict 1000×50 | 6.34 | 0.73 | **8.7x** |
| **KMeans** | fit 5000×20 | 114.16 | 20.51 | **5.6x** |
| **StandardScaler** | fit+transform 1000×50 | 0.59 | 0.15 | **3.9x** |
| **StandardScaler** | fit+transform 10000×100 | 6.78 | 3.11 | **2.2x** |
| **RandomForest** | fit 5000×20 | 1039.67 | 511.20 | **2.0x** |
| **RandomForest** | predict 5000×20 | 5.93 | 3.82 | **1.6x** |
| **DecisionTree** | fit 5000×20 | 78.45 | 59.95 | **1.3x** |
| **GaussianNB** | predict 5000×20 | 0.31 | 0.23 | **1.3x** |
| **KNN** | fit 1000×50 | 0.31 | 0.29 | **1.1x** |

### Gradient boosting against scikit-learn and LightGBM (#127)

`benchmarks/boosting_benchmark.py`, median of 3 fits, 100 trees, 31 leaves,
learning rate 0.1, every library on its default threads; anofox-ml through
its Python binding. Laptop i9-13900H (6 P + 8 E cores, 20 threads),
scikit-learn 1.8.0, LightGBM 4.7.0. The equal scores show the same models
(LGBM and HistGB reproduce LightGBM and sklearn).

| Model | Data | Reference | fit (s) anofox / ref | predict (s) anofox / ref |
|---|---|---|--:|--:|
| LGBM | regression 10 000×20 | LightGBM | **0.24** / 0.57 | **0.010** / 0.015 |
| LGBM | binary 10 000×20 | LightGBM | **0.22** / 0.38 | **0.009** / 0.010 |
| LGBM | regression 100 000×50 | LightGBM | 0.95 / **0.89** | **0.046** / 0.072 |
| LGBM | binary 100 000×50 | LightGBM | 1.05 / **0.71** | **0.041** / 0.068 |
| LGBM | one-hot 100 000×210 | LightGBM | 1.26 / **0.66** | 0.146 / **0.078** |
| HistGB | regression 10 000×20 | sklearn | **0.31** / 0.35 | 0.011 / **0.007** |
| HistGB | binary 10 000×20 | sklearn | 0.32 / **0.29** | 0.010 / **0.007** |
| HistGB | regression 100 000×50 | sklearn | 1.33 / **0.60** | **0.046** / 0.077 |
| HistGB | binary 100 000×50 | sklearn | 1.37 / **0.69** | **0.044** / 0.080 |
| HistGB | one-hot 100 000×210 | sklearn | 3.01 / **1.90** | **0.129** / 0.297 |
| GradientBoosting | regression 10 000×20 | sklearn | **1.15** / 4.82 | **0.006** / 0.011 |

Before the #127 speed work, LGBM fitted 100 000×50 in 3.6 s and predicted
it in 0.50 s. HistGB is slower than LGBM at 100 000 rows because its
sklearn numerics keep sequential sums and counts (for bit-exact sklearn
parity). On this hybrid CPU, all 20 logical threads are slower than the
physical cores, which `n_jobs=None` uses.

```bash
python benchmarks/boosting_benchmark.py   # needs the anofox_ml_python wheel, lightgbm, scikit-learn
```

The CatBoost-style estimators were checked on the same laptop against
catboost 1.2.10 with `thread_count=1`. The fit used 20 000 rows (5 numeric
and 3 categorical features with 50 levels), 300 trees of depth 6, and
CatBoost's defaults: MVS, score noise, permutations, CTR combinations and
10 leaf steps. The models agree to 1e-15. anofox-ml took 24.7 s (plain) and
22.8 s (ordered) against catboost's 8.9 s and 8.8 s; speed work on this
engine is still open.

### Phase 2: extended supervised sweep (harness in place; numbers TBD)

Implemented in `crates/anofox-ml/benches/benchmarks.rs` + the matching
estimators in `benchmarks/sklearn_benchmark.py`. Run the harness on your
own hardware to populate; the entries below list what gets measured.

| Algorithm | Operation | size |
|---|---|---|
| Ridge | fit + predict | 5000×20 |
| Lasso | fit + predict | 5000×20 |
| ElasticNet | fit + predict | 5000×20 |
| OLS | fit + predict | 5000×20 |
| BayesianRidge | fit + predict | 5000×20 |
| LogisticRegression | fit + predict | 5000×20 |
| SVC (RBF) | fit + predict | 1000×20 |
| LinearSVC | fit + predict | 5000×20 |
| LDA | fit + predict | 5000×20 |
| QDA | fit + predict | 5000×20 |
| MLP (1×32, 50 epochs) | fit + predict | 5000×20 |
| ExtraTrees (100 trees) | fit + predict | 5000×20 |
| GradientBoosting (100×depth-3) | fit + predict | 5000×20 |
| HistGradientBoosting | fit + predict | 5000×20 |
| AdaBoost (50 stumps) | fit + predict | 5000×20 |

Key optimizations: incremental sorted-index split finding for decision trees,
BinaryHeap-based KD-tree pruning for KNN, vectorized distance computation with
rayon parallelism for KMeans, and batch prediction for Random Forest.

Reproduce with:
```bash
cargo bench -p anofox-ml
uv run benchmarks/compare.py
```

### Phase 3: head-to-head against linfa (harness in place; numbers TBD)

[linfa](https://github.com/rust-ml/linfa) is the established Rust ML
toolkit. Where the algorithm exists in both libraries the benches run
the same seeded data through both, so the comparison is honest. The 6
overlap points are:

| Algorithm | anofox-ml bench group | linfa bench group |
|---|---|---|
| KMeans | `kmeans_5000x20` | `linfa_kmeans_5000x20` |
| Ridge | `ridge_5000x20` | `linfa_ridge_5000x20` |
| Lasso | `lasso_5000x20` | `linfa_lasso_5000x20` |
| LogisticRegression | `logistic_5000x20` | `linfa_logistic_5000x20` |
| DecisionTree | `tree_5000x20` | `linfa_decision_tree_5000x20` |
| GaussianNB | `gnb_5000x20` | `linfa_gnb_5000x20` |

Run the harness to populate:

```bash
cargo bench -p anofox-ml -- "5000x20"
```

linfa is a dev-dependency only — no runtime dep on linfa from
`anofox-ml`. Both libraries pin `ndarray = "0.16"` so the dependency
resolver doesn't fork.

### Continuous performance tracking (codspeed)

The codspeed CI job is paused: its cachegrind run took 40–60 minutes per
push. The benches still use the `codspeed-criterion-compat` shim, so
`cargo bench` works as before and the job can be restored from git
history (`.github/workflows/ci.yml`, before the conformal-registry PR).

### Scaling profiles

How wall-time grows as `n` grows, per category:

- [clustering](validation/scaling/clustering.md) — KMeans (linear), AgglomerativeClustering Ward (O(n²) nn-chain vs O(n³) naive)
- [regression](validation/scaling/regression.md) — Ridge (linear), RandomForestRegressor (n log n)
- [ensemble](validation/scaling/ensemble.md) — RandomForestClassifier (rayon-parallel) vs GradientBoostingClassifier (sequential)

The single-point comparison above tells you how anofox-ml does on a 5k
benchmark; the scaling profile docs tell you which algorithm to pick at
your actual data size.

### Additional estimator families (not in the perf sweep)

The benchmark table above covers the load-bearing fast paths. The following
families ship with correctness-only validation against sklearn and are not
part of the head-to-head perf sweep — they target API parity, not throughput
records:

- **Clustering**: HDBSCAN, AffinityPropagation, MeanShift, Birch, OPTICS,
  SpectralClustering, BayesianGaussianMixture
- **Manifold**: Isomap, LocallyLinearEmbedding, t-SNE (exact and Barnes-Hut),
  ClassicalMDS
- **Decomposition / FE**: FastICA, CCA, KernelPCA + inverse_transform,
  TruncatedSVD, NMF, PLS, RFECV
- **Neighbors**: LocalOutlierFactor (KD-tree path)
- **GP**: GaussianProcessClassifier (Laplace), kernel zoo (Matern, RQ, White,
  Constant, sums/products), L-BFGS multi-parameter optimisation

Algorithmic complexity tables for each estimator live in
`validation/sklearn_parity/*.md`.

## Documentation

API documentation is published at [docs.rs/anofox-ml](https://docs.rs/anofox-ml).

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes
before submitting a pull request. All code should include tests and pass
`cargo clippy` and `cargo fmt --check`.

## Releasing

The workspace is split into 17 publishable crates plus an umbrella `anofox-ml`.
Versions are kept in lockstep via `[workspace.package].version` in the root
`Cargo.toml` — bumping there moves every crate at once.

Releases publish through GitHub Actions, triggered by a published GitHub
release. To cut a release:

```bash
# 1. Bump workspace.package.version in Cargo.toml (e.g. 0.3.0 → 0.4.0)
# 2. Bump workspace.dependencies.anofox-ml-* `version = "..."` entries to match
# 3. Commit, tag, push
git commit -am "Release v0.4.0"
git tag v0.4.0
git push && git push --tags

# 4. Create a GitHub release pointing at the tag.
#    The `publish` job in .github/workflows/ci.yml fires on
#    `release.types: [published]`, runs after the full test matrix, and
#    invokes scripts/publish.sh --execute.
```

Required secret: `CARGO_REGISTRY_TOKEN` (a crates.io API token with publish
scope) configured in the GitHub repo settings. The workflow also asserts
the release tag matches `workspace.package.version` before uploading
anything, so a misnamed tag fails fast.

`anofox-ml-python` is marked `publish = false` — it's a PyO3 extension
module distributed via maturin (PyPI), not crates.io.

To dry-run locally before tagging:

```bash
scripts/publish.sh           # dry-run, no uploads
scripts/publish.sh --execute # requires `cargo login` against crates.io
```

## Python

`crates/anofox-ml-python` builds the `anofox_ml_python` package (PyO3 +
maturin). It exposes 150+ estimators with scikit-learn's API:
constructor arguments are stored unchanged, so `get_params` / `set_params`,
`sklearn.base.clone`, `GridSearchCV` and sklearn pipelines work; `fit`
returns `self`; seeds are `random_state`; classifiers accept any label type;
fitted models pickle (joblib / multiprocessing). Fitting and prediction run
in Rust with the GIL released.

```bash
pip install maturin
maturin develop --release -m crates/anofox-ml-python/Cargo.toml
pytest crates/anofox-ml-python/tests
```

```python
import anofox_ml_python as am
from sklearn.model_selection import GridSearchCV

clf = am.RandomForestClassifier(n_estimators=200, random_state=0).fit(X, y)
proba = clf.predict_proba(X_test)
search = GridSearchCV(am.SVC(), {"C": [0.1, 1, 10]}).fit(X, y)
```

Conformal prediction (`anofox_ml_python.conformal`) wraps any of these
estimators and gives intervals or sets with coverage `≥ 1 − alpha`:

```python
from anofox_ml_python import conformal

reg = conformal.CrossConformalRegressor(am.HistGradientBoostingRegressor(), cv=5).fit(X, y)
bounds = reg.predict_interval(X_test, alpha=0.1)      # (n, 2)
clf = conformal.SplitConformalClassifier(conformity_score="aps").fit(X, y)
sets = clf.predict_set(X_test, alpha=0.1)             # (n, n_classes), bool
```

## License

Licensed under either of

- [MIT license](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)

at your option.
