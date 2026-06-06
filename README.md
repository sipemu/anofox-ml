# RustML

A scikit-learn-inspired machine learning library for Rust, built on ndarray.

[![CI](https://github.com/sipemu/rustml/actions/workflows/ci.yml/badge.svg)](https://github.com/sipemu/rustml/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/rustml.svg)](https://crates.io/crates/rustml)
[![Documentation](https://docs.rs/rustml/badge.svg)](https://docs.rs/rustml)
[![codecov](https://codecov.io/gh/sipemu/rustml/branch/master/graph/badge.svg)](https://codecov.io/gh/sipemu/rustml)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#license)

## Features

| Category | RustML | scikit-learn equivalent |
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
| **Ensemble** | `RandomForest{Classifier,Regressor}`, `ExtraTrees{Classifier,Regressor}`, `GradientBoosting{Classifier,Regressor}`, `HistGradientBoosting{Classifier,Regressor}`, `LgbmClassifier`, `LgbmRegressor`, `Bagging{Classifier,Regressor}`, `AdaBoost{Classifier,Regressor}`, `Voting{Classifier,Regressor}`, `Stacking{Classifier,Regressor}` (+`push_proba`), `CalibratedClassifierCV`, `IsolationForest` (rayon-parallel) | `sklearn.ensemble`, `lightgbm` |
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
| **I/O & persistence** | CSV reader with ndarray integration, JSON / bincode serde round-tripping for fitted models | `pandas.read_csv`, `joblib.dump` |

## Quick Start

Add RustML to your project:

```toml
[dependencies]
rustml = "0.1"
ndarray = "0.16"
```

Train a KNN classifier with standardized features:

```rust
use rustml::prelude::*;
use ndarray::array;

fn main() -> rustml::core::Result<()> {
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

RustML is organized as a Cargo workspace with focused crates. You can depend on
the umbrella `rustml` crate for everything, or pick individual crates for
smaller dependency trees.

```
rustml (facade)
  +-- rustml-core              Core traits, error types, Pipeline, utilities
  +-- rustml-metrics           Classification, regression, clustering metrics
  +-- rustml-preprocessing     Scalers, PCA, KernelPCA, NMF, FastICA, TruncatedSVD,
                               PLS, CCA, feature selection, RFE/RFECV/SFS
  +-- rustml-neighbors         KNN with KD-tree, LocalOutlierFactor
  +-- rustml-trees             CART decision trees with predict_proba
  +-- rustml-ensemble          Random Forest, ExtraTrees, Gradient Boosting,
                               HistGradientBoosting, LightGBM-lite, AdaBoost,
                               Bagging, Voting, Stacking, Calibrated, IsolationForest
  +-- rustml-cluster           KMeans, MiniBatchKMeans, DBSCAN, HDBSCAN, OPTICS,
                               Birch, Agglomerative, Spectral, MeanShift, AP,
                               GaussianMixture, BayesianGaussianMixture
  +-- rustml-naive-bayes       Gaussian/Multinomial/Bernoulli NB
  +-- rustml-discriminant      LDA (with transform) and QDA
  +-- rustml-svm               SVC, SVR, NuSVC, NuSVR, LinearSVC/SVR, OneClassSVM
  +-- rustml-regression        OLS, Ridge (+weighted), Lasso, ElasticNet, GLMs,
                               BayesianRidge, ARD, LARS, OMP, KernelRidge,
                               RANSAC, TheilSen, Tweedie, TransformedTarget
  +-- rustml-linear            SGD, PassiveAggressive
  +-- rustml-gaussian-process  GP regressor (5 kernels + composites) & classifier
  +-- rustml-manifold          ClassicalMDS, Isomap, LLE, t-SNE
  +-- rustml-neural-networks   MLPClassifier, MLPRegressor
  +-- rustml-text              Count/Tfidf/Hashing vectorizers
  +-- rustml-io                CSV loading
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

Every estimator in `rustml` is validated against scikit-learn 1.8.0 via golden
fixtures (`test_harness/generators/gen_*.py`) and corresponding Rust tests in
`crates/rustml/tests/golden_*.rs`. Per-estimator parity notes — including
tolerances, sample-weight behaviour, missing options, and asymptotic
complexity — live under `validation/sklearn_parity/`.

The pinned sklearn version (1.8.0) is enforced by
`test_harness/check_sklearn_version.py`. Bumping the pin requires
`./test_harness/regenerate_all.sh` followed by a full `cargo test --workspace`
to confirm tolerances still hold.

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

RustML outperforms scikit-learn across all benchmarks, with up to 22x speedups
on critical operations. Measurements taken on the same machine with identical
datasets and parameters.

| Algorithm | Operation | sklearn (ms) | rustml (ms) | Speedup |
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

Key optimizations: incremental sorted-index split finding for decision trees,
BinaryHeap-based KD-tree pruning for KNN, vectorized distance computation with
rayon parallelism for KMeans, and batch prediction for Random Forest.

Reproduce with:
```bash
cargo bench -p rustml
uv run benchmarks/compare.py
```

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

API documentation is published at [docs.rs/rustml](https://docs.rs/rustml).

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes
before submitting a pull request. All code should include tests and pass
`cargo clippy` and `cargo fmt --check`.

## Releasing

The workspace is split into 17 publishable crates plus an umbrella `rustml`.
Versions are kept in lockstep via `[workspace.package].version` in the root
`Cargo.toml` — bumping there moves every crate at once.

Releases publish through GitHub Actions, triggered by a published GitHub
release. To cut a release:

```bash
# 1. Bump workspace.package.version in Cargo.toml (e.g. 0.1.0 → 0.2.0)
# 2. Bump workspace.dependencies.rustml-* `version = "..."` entries to match
# 3. Commit, tag, push
git commit -am "Release v0.2.0"
git tag v0.2.0
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

`rustml-python` is marked `publish = false` — it's a PyO3 extension
module distributed via maturin (PyPI), not crates.io.

To dry-run locally before tagging:

```bash
scripts/publish.sh           # dry-run, no uploads
scripts/publish.sh --execute # requires `cargo login` against crates.io
```

## License

Licensed under either of

- [MIT license](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)

at your option.
