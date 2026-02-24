# RustML

A scikit-learn-inspired machine learning library for Rust, built on ndarray.

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#license)
[![CI](https://github.com/sipemu/rustml/actions/workflows/ci.yml/badge.svg)](https://github.com/sipemu/rustml/actions)

## Features

| Category | RustML | scikit-learn equivalent |
|---|---|---|
| **Preprocessing** | `StandardScaler`, `MinMaxScaler` | `sklearn.preprocessing` |
| **Dimensionality Reduction** | `Pca` | `sklearn.decomposition.PCA` |
| **Feature Selection** | `VarianceThreshold`, `MutualInformationSelector` | `sklearn.feature_selection` |
| **Neighbors** | `KnnClassifier`, `KnnRegressor` (KD-tree) | `sklearn.neighbors` |
| **Trees** | `DecisionTreeClassifier`, `DecisionTreeRegressor` | `sklearn.tree` |
| **Ensemble** | `RandomForestClassifier/Regressor`, `GradientBoostingClassifier/Regressor` | `sklearn.ensemble` |
| **Clustering** | `KMeans` (k-means++), `Dbscan` | `sklearn.cluster` |
| **Naive Bayes** | `GaussianNB` | `sklearn.naive_bayes` |
| **Metrics** | `accuracy_score`, `f1_score`, `mse`, `r2_score`, ... | `sklearn.metrics` |
| **Utilities** | `train_test_split`, `cross_val_score`, `Pipeline` | `sklearn.model_selection`, `sklearn.pipeline` |
| **I/O** | CSV reader with ndarray integration | `pandas.read_csv` |

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
  +-- rustml-core          Core traits, error types, Pipeline, utilities
  +-- rustml-metrics        Classification and regression metrics
  +-- rustml-preprocessing  Scalers, PCA, feature selection
  +-- rustml-neighbors      KNN with KD-tree acceleration
  +-- rustml-trees          CART decision trees
  +-- rustml-ensemble       Random Forest, Gradient Boosting (parallel via rayon)
  +-- rustml-cluster        KMeans, DBSCAN
  +-- rustml-naive-bayes    Gaussian Naive Bayes
  +-- rustml-io             CSV loading
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
| `Predict<F>` | Generate predictions from fitted model |
| `Transform<F>` | Transform feature matrix |
| `InverseTransform<F>` | Reverse a transformation |

## Algorithms

### Classification
- K-Nearest Neighbors (`KnnClassifier`) with KD-tree and parallel query
- Decision Tree (`DecisionTreeClassifier`) using CART
- Random Forest (`RandomForestClassifier`) with parallel tree fitting
- Gradient Boosting (`GradientBoostingClassifier`)
- Gaussian Naive Bayes (`GaussianNB`)

### Regression
- K-Nearest Neighbors (`KnnRegressor`)
- Decision Tree (`DecisionTreeRegressor`)
- Random Forest (`RandomForestRegressor`)
- Gradient Boosting (`GradientBoostingRegressor`)

### Clustering
- K-Means with k-means++ initialization (`KMeans`)
- DBSCAN density-based clustering (`Dbscan`)

### Preprocessing
- `StandardScaler` -- zero mean, unit variance
- `MinMaxScaler` -- scale to [0, 1]
- `Pca` -- principal component analysis
- `VarianceThreshold` -- drop low-variance features
- `MutualInformationSelector` -- select features by mutual information

### Metrics
- Classification: `accuracy_score`, `precision`, `recall`, `f1_score`, `confusion_matrix`, macro/micro/weighted averaging
- Regression: `mse`, `mae`, `r2_score`

### Utilities
- `train_test_split`, `cross_val_score`, `Pipeline`

## Documentation

API documentation is published at [docs.rs/rustml](https://docs.rs/rustml).

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes
before submitting a pull request. All code should include tests and pass
`cargo clippy` and `cargo fmt --check`.

## License

Licensed under either of

- [MIT license](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)

at your option.
