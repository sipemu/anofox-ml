//! Golden fixtures for model persistence (#87).
//!
//! `tests/fixtures/persistence/v<N>/` holds fitted models saved by binary
//! format version `N`, next to the predictions they made when they were
//! saved. Every fixture must keep loading and predicting the same values,
//! so a change to a fitted type that breaks old files fails here.
//!
//! Fixtures of a new format version are written with
//! `ANOFOX_WRITE_FIXTURES=1 cargo test -p anofox-ml --test persistence_compat`,
//! which saves into the directory of the current format version only. Files
//! of older versions are never rewritten.

use std::path::PathBuf;

use anofox_ml::prelude::*;
use anofox_ml_core::persistence;
use ndarray::{Array1, Array2};
use serde::{de::DeserializeOwned, Serialize};

/// Format versions with committed fixtures.
const VERSIONS: &[u32] = &[1];

fn fixture_dir(version: u32) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/persistence")
        .join(format!("v{version}"))
}

fn writing() -> bool {
    std::env::var_os("ANOFOX_WRITE_FIXTURES").is_some()
}

/// Small deterministic data: 60 × 4 features, a regression and a
/// three-class target.
fn data() -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let x = Array2::from_shape_fn((60, 4), |(i, j)| {
        (((i * 37 + j * 11) % 23) as f64 / 7.0 - 1.5) + 0.1 * j as f64
    });
    let y_reg = Array1::from_shape_fn(60, |i| {
        2.0 * x[[i, 0]] - x[[i, 1]] + 0.5 * x[[i, 2]] * x[[i, 3]]
    });
    let y_cls = y_reg.mapv(|v| {
        if v < -0.5 {
            0.0
        } else if v < 1.0 {
            1.0
        } else {
            2.0
        }
    });
    (x, y_reg, y_cls)
}

fn write_fixture<M: Serialize>(name: &str, model: &M, out: &[f64]) {
    let version = persistence::FORMAT_VERSION;
    let dir = fixture_dir(version);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join(format!("{name}.bin")),
        persistence::to_bincode_bytes(model).unwrap(),
    )
    .unwrap();
    std::fs::write(
        dir.join(format!("{name}.expected.json")),
        serde_json::to_string(out).unwrap(),
    )
    .unwrap();
}

/// Check (or, when writing, save) one fixture. `out` maps a model to its
/// flattened output on the training data.
fn case<M>(name: &str, fit: impl FnOnce() -> M, out: impl Fn(&M) -> Vec<f64>)
where
    M: Serialize + DeserializeOwned,
{
    if writing() {
        let model = fit();
        write_fixture(name, &model, &out(&model));
        return;
    }
    for &version in VERSIONS {
        let dir = fixture_dir(version);
        let bytes = std::fs::read(dir.join(format!("{name}.bin")))
            .unwrap_or_else(|e| panic!("v{version} fixture {name}: {e}"));
        let expected: Vec<f64> = serde_json::from_str(
            &std::fs::read_to_string(dir.join(format!("{name}.expected.json"))).unwrap(),
        )
        .unwrap();
        let model: M = persistence::from_bincode_bytes(&bytes)
            .unwrap_or_else(|e| panic!("v{version} fixture {name} no longer loads: {e}"));
        let got = out(&model);
        assert_eq!(got.len(), expected.len(), "v{version} {name}");
        for (i, (g, w)) in got.iter().zip(&expected).enumerate() {
            assert!(
                (g - w).abs() <= 1e-9 * w.abs().max(1.0),
                "v{version} {name}[{i}]: {g} vs saved {w}"
            );
        }
    }
}

macro_rules! supervised {
    ($test:ident, $name:literal, $est:expr, discrete = $cls:expr) => {
        #[test]
        fn $test() {
            let (x, y_reg, y_cls) = data();
            let y = if $cls { &y_cls } else { &y_reg };
            case(
                $name,
                || Fit::fit(&$est, &x, y).unwrap(),
                |m| Predict::predict(m, &x).unwrap().to_vec(),
            );
        }
    };
}

macro_rules! transformer {
    ($test:ident, $name:literal, $est:expr) => {
        #[test]
        fn $test() {
            let (x, _, _) = data();
            case(
                $name,
                || FitUnsupervised::fit(&$est, &x).unwrap(),
                |m| {
                    Transform::transform(m, &x)
                        .unwrap()
                        .iter()
                        .copied()
                        .collect()
                },
            );
        }
    };
}

macro_rules! unsupervised_predict {
    ($test:ident, $name:literal, $est:expr) => {
        #[test]
        fn $test() {
            let (x, _, _) = data();
            case(
                $name,
                || FitUnsupervised::fit(&$est, &x).unwrap(),
                |m| Predict::predict(m, &x).unwrap().to_vec(),
            );
        }
    };
}

supervised!(
    decision_tree,
    "decision_tree_classifier",
    DecisionTreeClassifier::new(),
    discrete = true
);
supervised!(
    random_forest,
    "random_forest_classifier",
    RandomForestClassifier::new(10),
    discrete = true
);
supervised!(
    gradient_boosting,
    "gradient_boosting_regressor",
    GradientBoostingRegressor::new(),
    discrete = false
);
supervised!(
    hist_gradient_boosting,
    "hist_gradient_boosting_classifier",
    HistGradientBoostingClassifier::new(),
    discrete = true
);
supervised!(
    knn,
    "knn_classifier",
    KnnClassifier::new(3),
    discrete = true
);
supervised!(
    logistic,
    "logistic_regressor",
    LogisticRegressor::new(),
    discrete = true
);
supervised!(
    bayesian_ridge,
    "bayesian_ridge",
    BayesianRidge::new(),
    discrete = false
);
supervised!(
    poisson,
    "poisson_regressor",
    PoissonRegressor::new(),
    discrete = true
);
supervised!(
    gaussian_nb,
    "gaussian_nb",
    GaussianNB::new(),
    discrete = true
);
supervised!(
    lda,
    "linear_discriminant_analysis",
    LinearDiscriminantAnalysis::new(),
    discrete = true
);
supervised!(svc, "svc", Svc::new(), discrete = true);
supervised!(
    mlp,
    "mlp_classifier",
    MlpClassifier::default(),
    discrete = true
);
transformer!(standard_scaler, "standard_scaler", StandardScaler::new());
transformer!(pca, "pca", Pca::new(2));
transformer!(
    spline_transformer,
    "spline_transformer",
    SplineTransformer::default()
);
unsupervised_predict!(kmeans, "kmeans", KMeans::new(3));
unsupervised_predict!(isolation_forest, "isolation_forest", IsolationForest::new());
