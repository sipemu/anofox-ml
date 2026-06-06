use anofox_ml::prelude::*;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ---------------------------------------------------------------------------
// Shared data generators
// ---------------------------------------------------------------------------

fn generate_classification_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<f64>) {
    let mut x = Array2::<f64>::zeros((n_samples, n_features));
    let mut y = Array1::<f64>::zeros(n_samples);

    for i in 0..n_samples {
        let class = if i < n_samples / 2 { 0.0 } else { 1.0 };
        y[i] = class;
        for j in 0..n_features {
            x[[i, j]] = if class == 0.0 {
                (i * n_features + j) as f64 * 0.01
            } else {
                10.0 + (i * n_features + j) as f64 * 0.01
            };
        }
    }

    (x, y)
}

fn generate_regression_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<f64>) {
    let mut x = Array2::<f64>::zeros((n_samples, n_features));
    let mut y = Array1::<f64>::zeros(n_samples);

    for i in 0..n_samples {
        let mut sum = 0.0;
        for j in 0..n_features {
            let val = (i * n_features + j) as f64 * 0.1;
            x[[i, j]] = val;
            sum += val;
        }
        y[i] = sum;
    }

    (x, y)
}

/// Generate random classification data with a seeded RNG (matches the Python
/// script's `generate_classification_data`).
fn generate_random_classification_data(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    seed: u64,
) -> (Array2<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x = Array2::<f64>::zeros((n_samples, n_features));
    let mut y = Array1::<f64>::zeros(n_samples);

    for i in 0..n_samples {
        for j in 0..n_features {
            x[[i, j]] = rng.gen::<f64>() * 2.0 - 1.0; // roughly standard normal approximation
        }
        y[i] = (rng.gen::<u64>() % n_classes as u64) as f64;
    }

    (x, y)
}

/// Generate random regression/unsupervised data with a seeded RNG.
fn generate_random_data(n_samples: usize, n_features: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x = Array2::<f64>::zeros((n_samples, n_features));

    for i in 0..n_samples {
        for j in 0..n_features {
            x[[i, j]] = rng.gen::<f64>() * 2.0 - 1.0;
        }
    }

    x
}

// ===========================================================================
// Original benchmark groups (kept intact)
// ===========================================================================

fn bench_knn_classifier(c: &mut Criterion) {
    let mut group = c.benchmark_group("knn_classifier");

    for &n in &[100, 500, 1000] {
        let (x_train, y_train) = generate_classification_data(n, 4);
        let (x_test, _) = generate_classification_data(50, 4);

        let knn = KnnClassifier {
            n_neighbors: 5,
            ..Default::default()
        };
        let fitted = Fit::fit(&knn, &x_train, &y_train).unwrap();

        group.bench_with_input(BenchmarkId::new("predict", n), &n, |b, _| {
            b.iter(|| fitted.predict(black_box(&x_test)).unwrap());
        });
    }

    group.finish();
}

fn bench_decision_tree(c: &mut Criterion) {
    let mut group = c.benchmark_group("decision_tree");

    for &n in &[100, 500, 1000] {
        let (x_train, y_train) = generate_classification_data(n, 4);
        let (x_test, _) = generate_classification_data(50, 4);

        group.bench_with_input(BenchmarkId::new("fit", n), &n, |b, _| {
            let tree = DecisionTreeClassifier {
                max_depth: Some(10),
                ..Default::default()
            };
            b.iter(|| Fit::<f64>::fit(&tree, black_box(&x_train), black_box(&y_train)).unwrap());
        });

        let tree = DecisionTreeClassifier {
            max_depth: Some(10),
            ..Default::default()
        };
        let fitted = Fit::fit(&tree, &x_train, &y_train).unwrap();

        group.bench_with_input(BenchmarkId::new("predict", n), &n, |b, _| {
            b.iter(|| fitted.predict(black_box(&x_test)).unwrap());
        });
    }

    group.finish();
}

fn bench_random_forest(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_forest");
    group.sample_size(20);

    for &n in &[100, 500] {
        let (x_train, y_train) = generate_classification_data(n, 4);
        let (x_test, _) = generate_classification_data(50, 4);

        group.bench_with_input(BenchmarkId::new("fit", n), &n, |b, _| {
            let rf = RandomForestClassifier {
                n_estimators: 10,
                max_depth: Some(5),
                ..Default::default()
            };
            b.iter(|| Fit::<f64>::fit(&rf, black_box(&x_train), black_box(&y_train)).unwrap());
        });

        let rf = RandomForestClassifier {
            n_estimators: 10,
            max_depth: Some(5),
            ..Default::default()
        };
        let fitted = Fit::fit(&rf, &x_train, &y_train).unwrap();

        group.bench_with_input(BenchmarkId::new("predict", n), &n, |b, _| {
            b.iter(|| fitted.predict(black_box(&x_test)).unwrap());
        });
    }

    group.finish();
}

fn bench_preprocessing(c: &mut Criterion) {
    let mut group = c.benchmark_group("preprocessing");

    for &n in &[100, 1000, 5000] {
        let (x, _) = generate_regression_data(n, 10);

        group.bench_with_input(
            BenchmarkId::new("standard_scaler_fit_transform", n),
            &n,
            |b, _| {
                b.iter(|| {
                    let scaler = StandardScaler::default();
                    let fitted = FitUnsupervised::<f64>::fit(&scaler, black_box(&x)).unwrap();
                    fitted.transform(black_box(&x)).unwrap()
                });
            },
        );
    }

    group.finish();
}

fn bench_kmeans(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans");
    group.sample_size(20);

    for &n in &[100, 500] {
        let (x, _) = generate_regression_data(n, 4);

        group.bench_with_input(BenchmarkId::new("fit", n), &n, |b, _| {
            let km = KMeans {
                n_clusters: 3,
                max_iter: 100,
                ..Default::default()
            };
            b.iter(|| FitUnsupervised::<f64>::fit(&km, black_box(&x)).unwrap());
        });
    }

    group.finish();
}

fn bench_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("metrics");

    for &n in &[100, 1000, 10000] {
        let y_true: Array1<f64> = Array1::from_vec((0..n).map(|i| (i % 3) as f64).collect());
        let y_pred: Array1<f64> = Array1::from_vec((0..n).map(|i| ((i + 1) % 3) as f64).collect());

        group.bench_with_input(BenchmarkId::new("accuracy", n), &n, |b, _| {
            b.iter(|| accuracy_score(black_box(&y_true), black_box(&y_pred)).unwrap());
        });

        let y_true_reg: Array1<f64> = Array1::from_vec((0..n).map(|i| i as f64 * 0.1).collect());
        let y_pred_reg: Array1<f64> =
            Array1::from_vec((0..n).map(|i| i as f64 * 0.1 + 0.01).collect());

        group.bench_with_input(BenchmarkId::new("r2_score", n), &n, |b, _| {
            b.iter(|| r2_score(black_box(&y_true_reg), black_box(&y_pred_reg)).unwrap());
        });
    }

    group.finish();
}

// ===========================================================================
// Comparison benchmark groups (matching sklearn_benchmark.py scenarios)
// ===========================================================================

/// StandardScaler fit+transform on 1000x50.
fn bench_scaler_1000x50(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaler_1000x50");
    let x = generate_random_data(1000, 50, 42);

    group.bench_function("fit_transform", |b| {
        b.iter(|| {
            let scaler = StandardScaler::default();
            let fitted = FitUnsupervised::<f64>::fit(&scaler, black_box(&x)).unwrap();
            fitted.transform(black_box(&x)).unwrap()
        });
    });

    group.finish();
}

/// StandardScaler fit+transform on 10000x100.
fn bench_scaler_10000x100(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaler_10000x100");
    let x = generate_random_data(10000, 100, 42);

    group.bench_function("fit_transform", |b| {
        b.iter(|| {
            let scaler = StandardScaler::default();
            let fitted = FitUnsupervised::<f64>::fit(&scaler, black_box(&x)).unwrap();
            fitted.transform(black_box(&x)).unwrap()
        });
    });

    group.finish();
}

/// KNN classifier on 1000 train / 200 test, 50 features, k=5, Euclidean.
fn bench_knn_1000x50(c: &mut Criterion) {
    let mut group = c.benchmark_group("knn_1000x50");
    let (x_train, y_train) = generate_random_classification_data(1000, 50, 2, 42);
    let (x_test, _) = generate_random_classification_data(200, 50, 2, 43);

    // Bench fit
    group.bench_function("fit", |b| {
        b.iter(|| {
            let knn = KnnClassifier {
                n_neighbors: 5,
                metric: DistanceMetric::Euclidean,
                ..Default::default()
            };
            Fit::<f64>::fit(&knn, black_box(&x_train), black_box(&y_train)).unwrap()
        });
    });

    // Bench predict
    let knn = KnnClassifier {
        n_neighbors: 5,
        metric: DistanceMetric::Euclidean,
        ..Default::default()
    };
    let fitted = Fit::fit(&knn, &x_train, &y_train).unwrap();

    group.bench_function("predict", |b| {
        b.iter(|| fitted.predict(black_box(&x_test)).unwrap());
    });

    group.finish();
}

/// Decision tree classifier on 5000x20, max_depth=10.
fn bench_tree_5000x20(c: &mut Criterion) {
    let mut group = c.benchmark_group("tree_5000x20");
    group.sample_size(20);

    let (x_train, y_train) = generate_random_classification_data(5000, 20, 5, 42);
    let (x_test, _) = generate_random_classification_data(500, 20, 5, 43);

    // Bench fit
    group.bench_function("fit", |b| {
        b.iter(|| {
            let tree = DecisionTreeClassifier {
                max_depth: Some(10),
                ..Default::default()
            };
            Fit::<f64>::fit(&tree, black_box(&x_train), black_box(&y_train)).unwrap()
        });
    });

    // Bench predict
    let tree = DecisionTreeClassifier {
        max_depth: Some(10),
        ..Default::default()
    };
    let fitted = Fit::fit(&tree, &x_train, &y_train).unwrap();

    group.bench_function("predict", |b| {
        b.iter(|| fitted.predict(black_box(&x_test)).unwrap());
    });

    group.finish();
}

/// Random forest classifier on 5000x20, 100 trees, max_depth=10.
fn bench_rf_5000x20(c: &mut Criterion) {
    let mut group = c.benchmark_group("rf_5000x20");
    group.sample_size(10);

    let (x_train, y_train) = generate_random_classification_data(5000, 20, 5, 42);
    let (x_test, _) = generate_random_classification_data(500, 20, 5, 43);

    // Bench fit
    group.bench_function("fit", |b| {
        b.iter(|| {
            let rf = RandomForestClassifier {
                n_estimators: 100,
                max_depth: Some(10),
                seed: 42,
                ..Default::default()
            };
            Fit::<f64>::fit(&rf, black_box(&x_train), black_box(&y_train)).unwrap()
        });
    });

    // Bench predict
    let rf = RandomForestClassifier {
        n_estimators: 100,
        max_depth: Some(10),
        seed: 42,
        ..Default::default()
    };
    let fitted = Fit::fit(&rf, &x_train, &y_train).unwrap();

    group.bench_function("predict", |b| {
        b.iter(|| fitted.predict(black_box(&x_test)).unwrap());
    });

    group.finish();
}

/// KMeans clustering on 5000x20, k=10.
fn bench_kmeans_5000x20(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans_5000x20");
    group.sample_size(10);

    let x = generate_random_data(5000, 20, 42);

    group.bench_function("fit", |b| {
        b.iter(|| {
            let km = KMeans {
                n_clusters: 10,
                max_iter: 300,
                seed: 42,
                ..Default::default()
            };
            FitUnsupervised::<f64>::fit(&km, black_box(&x)).unwrap()
        });
    });

    group.finish();
}

/// Gaussian Naive Bayes on 5000x20.
fn bench_gnb_5000x20(c: &mut Criterion) {
    let mut group = c.benchmark_group("gnb_5000x20");

    let (x_train, y_train) = generate_random_classification_data(5000, 20, 5, 42);
    let (x_test, _) = generate_random_classification_data(500, 20, 5, 43);

    // Bench fit
    group.bench_function("fit", |b| {
        b.iter(|| {
            let gnb = GaussianNB::default();
            Fit::<f64>::fit(&gnb, black_box(&x_train), black_box(&y_train)).unwrap()
        });
    });

    // Bench predict
    let gnb = GaussianNB::default();
    let fitted = Fit::fit(&gnb, &x_train, &y_train).unwrap();

    group.bench_function("predict", |b| {
        b.iter(|| fitted.predict(black_box(&x_test)).unwrap());
    });

    group.finish();
}

// ===========================================================================
// Phase A: Extended supervised estimator comparison (vs sklearn)
//
// Sizing: 5000×20 unless the estimator's training is super-linear in n
// (SVMs are O(n²)–O(n³) per the libsvm cost model); those drop to 1000×20
// so a single bench iteration stays under a few seconds.
// ===========================================================================

// ─── Linear models ────────────────────────────────────────────────────────

fn bench_ridge_5000x20(c: &mut Criterion) {
    let mut group = c.benchmark_group("ridge_5000x20");
    let (x_train, y_train) = generate_random_regression_data(5000, 20, 42);
    let (x_test, _) = generate_random_regression_data(500, 20, 43);

    group.bench_function("fit", |b| {
        b.iter(|| {
            let m = RidgeRegressor::new().with_lambda(1.0);
            Fit::fit(&m, black_box(&x_train), black_box(&y_train)).unwrap()
        });
    });
    let fitted = Fit::fit(&RidgeRegressor::new().with_lambda(1.0), &x_train, &y_train).unwrap();
    group.bench_function("predict", |b| {
        b.iter(|| fitted.predict(black_box(&x_test)).unwrap());
    });
    group.finish();
}

fn bench_lasso_5000x20(c: &mut Criterion) {
    let mut group = c.benchmark_group("lasso_5000x20");
    let (x_train, y_train) = generate_random_regression_data(5000, 20, 42);
    let (x_test, _) = generate_random_regression_data(500, 20, 43);

    group.bench_function("fit", |b| {
        b.iter(|| {
            let m = LassoRegressor::new().with_lambda(0.1);
            Fit::fit(&m, black_box(&x_train), black_box(&y_train)).unwrap()
        });
    });
    let fitted = Fit::fit(&LassoRegressor::new().with_lambda(0.1), &x_train, &y_train).unwrap();
    group.bench_function("predict", |b| {
        b.iter(|| fitted.predict(black_box(&x_test)).unwrap());
    });
    group.finish();
}

fn bench_elasticnet_5000x20(c: &mut Criterion) {
    let mut group = c.benchmark_group("elasticnet_5000x20");
    let (x_train, y_train) = generate_random_regression_data(5000, 20, 42);
    let (x_test, _) = generate_random_regression_data(500, 20, 43);

    group.bench_function("fit", |b| {
        b.iter(|| {
            let m = ElasticNetRegressor::new().with_lambda(0.1).with_alpha(0.5);
            Fit::fit(&m, black_box(&x_train), black_box(&y_train)).unwrap()
        });
    });
    let m = ElasticNetRegressor::new().with_lambda(0.1).with_alpha(0.5);
    let fitted = Fit::fit(&m, &x_train, &y_train).unwrap();
    group.bench_function("predict", |b| {
        b.iter(|| fitted.predict(black_box(&x_test)).unwrap());
    });
    group.finish();
}

fn bench_ols_5000x20(c: &mut Criterion) {
    let mut group = c.benchmark_group("ols_5000x20");
    let (x_train, y_train) = generate_random_regression_data(5000, 20, 42);
    let (x_test, _) = generate_random_regression_data(500, 20, 43);

    group.bench_function("fit", |b| {
        b.iter(|| {
            let m = OlsRegressor::new();
            Fit::fit(&m, black_box(&x_train), black_box(&y_train)).unwrap()
        });
    });
    let fitted = Fit::fit(&OlsRegressor::new(), &x_train, &y_train).unwrap();
    group.bench_function("predict", |b| {
        b.iter(|| fitted.predict(black_box(&x_test)).unwrap());
    });
    group.finish();
}

fn bench_bayesian_ridge_5000x20(c: &mut Criterion) {
    let mut group = c.benchmark_group("bayesian_ridge_5000x20");
    group.sample_size(10);
    let (x_train, y_train) = generate_random_regression_data(5000, 20, 42);
    let (x_test, _) = generate_random_regression_data(500, 20, 43);

    group.bench_function("fit", |b| {
        b.iter(|| {
            let m = BayesianRidge::new();
            Fit::fit(&m, black_box(&x_train), black_box(&y_train)).unwrap()
        });
    });
    let fitted = Fit::fit(&BayesianRidge::new(), &x_train, &y_train).unwrap();
    group.bench_function("predict", |b| {
        b.iter(|| fitted.predict(black_box(&x_test)).unwrap());
    });
    group.finish();
}

fn bench_logistic_5000x20(c: &mut Criterion) {
    let mut group = c.benchmark_group("logistic_5000x20");
    group.sample_size(10);
    let (x_train, y_train) = generate_random_classification_data(5000, 20, 2, 42);
    let (x_test, _) = generate_random_classification_data(500, 20, 2, 43);

    group.bench_function("fit", |b| {
        b.iter(|| {
            let m = LogisticRegressor::new();
            Fit::fit(&m, black_box(&x_train), black_box(&y_train)).unwrap()
        });
    });
    let fitted = Fit::fit(&LogisticRegressor::new(), &x_train, &y_train).unwrap();
    group.bench_function("predict", |b| {
        b.iter(|| fitted.predict(black_box(&x_test)).unwrap());
    });
    group.finish();
}

// ─── SVM (smaller n; libsvm is O(n²)+) ────────────────────────────────────

fn bench_svc_rbf_1000x20(c: &mut Criterion) {
    let mut group = c.benchmark_group("svc_rbf_1000x20");
    group.sample_size(10);
    let (x_train, y_train) = generate_random_classification_data(1000, 20, 2, 42);
    let (x_test, _) = generate_random_classification_data(200, 20, 2, 43);

    group.bench_function("fit", |b| {
        b.iter(|| {
            let m = Svc {
                c: 1.0,
                kernel: SvmKernel::Rbf { gamma: 0.05 },
                max_iter: 1000,
                tol: 1e-3,
                seed: 42,
            };
            Fit::fit(&m, black_box(&x_train), black_box(&y_train)).unwrap()
        });
    });
    let m = Svc {
        c: 1.0,
        kernel: SvmKernel::Rbf { gamma: 0.05 },
        max_iter: 1000,
        tol: 1e-3,
        seed: 42,
    };
    let fitted = Fit::fit(&m, &x_train, &y_train).unwrap();
    group.bench_function("predict", |b| {
        b.iter(|| fitted.predict(black_box(&x_test)).unwrap());
    });
    group.finish();
}

fn bench_linear_svc_5000x20(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_svc_5000x20");
    group.sample_size(10);
    let (x_train, y_train) = generate_random_classification_data(5000, 20, 2, 42);
    let (x_test, _) = generate_random_classification_data(500, 20, 2, 43);

    group.bench_function("fit", |b| {
        b.iter(|| {
            let m = LinearSvc::default();
            Fit::fit(&m, black_box(&x_train), black_box(&y_train)).unwrap()
        });
    });
    let fitted = Fit::fit(&LinearSvc::default(), &x_train, &y_train).unwrap();
    group.bench_function("predict", |b| {
        b.iter(|| fitted.predict(black_box(&x_test)).unwrap());
    });
    group.finish();
}

// ─── Discriminant analysis ────────────────────────────────────────────────

fn bench_lda_5000x20(c: &mut Criterion) {
    let mut group = c.benchmark_group("lda_5000x20");
    let (x_train, y_train) = generate_random_classification_data(5000, 20, 5, 42);
    let (x_test, _) = generate_random_classification_data(500, 20, 5, 43);

    group.bench_function("fit", |b| {
        b.iter(|| {
            let m = LinearDiscriminantAnalysis::new();
            Fit::fit(&m, black_box(&x_train), black_box(&y_train)).unwrap()
        });
    });
    let fitted = Fit::fit(&LinearDiscriminantAnalysis::new(), &x_train, &y_train).unwrap();
    group.bench_function("predict", |b| {
        b.iter(|| fitted.predict(black_box(&x_test)).unwrap());
    });
    group.finish();
}

fn bench_qda_5000x20(c: &mut Criterion) {
    let mut group = c.benchmark_group("qda_5000x20");
    let (x_train, y_train) = generate_random_classification_data(5000, 20, 5, 42);
    let (x_test, _) = generate_random_classification_data(500, 20, 5, 43);

    group.bench_function("fit", |b| {
        b.iter(|| {
            let m = QuadraticDiscriminantAnalysis::new();
            Fit::fit(&m, black_box(&x_train), black_box(&y_train)).unwrap()
        });
    });
    let fitted = Fit::fit(&QuadraticDiscriminantAnalysis::new(), &x_train, &y_train).unwrap();
    group.bench_function("predict", |b| {
        b.iter(|| fitted.predict(black_box(&x_test)).unwrap());
    });
    group.finish();
}

// ─── Neural networks (smaller arch + fewer epochs to keep bench wall-time
//      manageable; sklearn parity is the goal, not absolute time) ─────────

fn bench_mlp_5000x20(c: &mut Criterion) {
    let mut group = c.benchmark_group("mlp_5000x20");
    group.sample_size(10);
    let (x_train, y_train) = generate_random_classification_data(5000, 20, 5, 42);
    let (x_test, _) = generate_random_classification_data(500, 20, 5, 43);

    group.bench_function("fit", |b| {
        b.iter(|| {
            let m = MlpClassifier {
                hidden_layer_sizes: vec![32],
                max_iter: 50,
                seed: 42,
                ..Default::default()
            };
            Fit::fit(&m, black_box(&x_train), black_box(&y_train)).unwrap()
        });
    });
    let m = MlpClassifier {
        hidden_layer_sizes: vec![32],
        max_iter: 50,
        seed: 42,
        ..Default::default()
    };
    let fitted = Fit::fit(&m, &x_train, &y_train).unwrap();
    group.bench_function("predict", |b| {
        b.iter(|| fitted.predict(black_box(&x_test)).unwrap());
    });
    group.finish();
}

// ─── Ensemble / boosting ──────────────────────────────────────────────────

fn bench_extra_trees_5000x20(c: &mut Criterion) {
    let mut group = c.benchmark_group("extra_trees_5000x20");
    group.sample_size(10);
    let (x_train, y_train) = generate_random_classification_data(5000, 20, 5, 42);
    let (x_test, _) = generate_random_classification_data(500, 20, 5, 43);

    group.bench_function("fit", |b| {
        b.iter(|| {
            let m = ExtraTreesClassifier::new(100)
                .with_max_depth(Some(10))
                .with_seed(42);
            Fit::fit(&m, black_box(&x_train), black_box(&y_train)).unwrap()
        });
    });
    let m = ExtraTreesClassifier::new(100)
        .with_max_depth(Some(10))
        .with_seed(42);
    let fitted = Fit::fit(&m, &x_train, &y_train).unwrap();
    group.bench_function("predict", |b| {
        b.iter(|| fitted.predict(black_box(&x_test)).unwrap());
    });
    group.finish();
}

fn bench_gbm_5000x20(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_boosting_5000x20");
    group.sample_size(10);
    let (x_train, y_train) = generate_random_classification_data(5000, 20, 2, 42);
    let (x_test, _) = generate_random_classification_data(500, 20, 2, 43);

    group.bench_function("fit", |b| {
        b.iter(|| {
            let m = GradientBoostingClassifier::new()
                .with_n_estimators(100)
                .with_max_depth(Some(3));
            Fit::fit(&m, black_box(&x_train), black_box(&y_train)).unwrap()
        });
    });
    let m = GradientBoostingClassifier::new()
        .with_n_estimators(100)
        .with_max_depth(Some(3));
    let fitted = Fit::fit(&m, &x_train, &y_train).unwrap();
    group.bench_function("predict", |b| {
        b.iter(|| fitted.predict(black_box(&x_test)).unwrap());
    });
    group.finish();
}

fn bench_hist_gbm_5000x20(c: &mut Criterion) {
    let mut group = c.benchmark_group("hist_gradient_boosting_5000x20");
    group.sample_size(10);
    let (x_train, y_train) = generate_random_classification_data(5000, 20, 2, 42);
    let (x_test, _) = generate_random_classification_data(500, 20, 2, 43);

    group.bench_function("fit", |b| {
        b.iter(|| {
            let m = HistGradientBoostingClassifier::new()
                .with_n_estimators(100)
                .with_max_depth(6);
            Fit::fit(&m, black_box(&x_train), black_box(&y_train)).unwrap()
        });
    });
    let m = HistGradientBoostingClassifier::new()
        .with_n_estimators(100)
        .with_max_depth(6);
    let fitted = Fit::fit(&m, &x_train, &y_train).unwrap();
    group.bench_function("predict", |b| {
        b.iter(|| fitted.predict(black_box(&x_test)).unwrap());
    });
    group.finish();
}

fn bench_adaboost_5000x20(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaboost_5000x20");
    group.sample_size(10);
    let (x_train, y_train) = generate_random_classification_data(5000, 20, 2, 42);
    let (x_test, _) = generate_random_classification_data(500, 20, 2, 43);

    group.bench_function("fit", |b| {
        b.iter(|| {
            let m = AdaBoostClassifier::new()
                .with_n_estimators(50)
                .with_seed(42);
            Fit::fit(&m, black_box(&x_train), black_box(&y_train)).unwrap()
        });
    });
    let m = AdaBoostClassifier::new()
        .with_n_estimators(50)
        .with_seed(42);
    let fitted = Fit::fit(&m, &x_train, &y_train).unwrap();
    group.bench_function("predict", |b| {
        b.iter(|| fitted.predict(black_box(&x_test)).unwrap());
    });
    group.finish();
}

/// Random regression-data generator matching the seeded Python harness.
fn generate_random_regression_data(
    n_samples: usize,
    n_features: usize,
    seed: u64,
) -> (Array2<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x = Array2::<f64>::zeros((n_samples, n_features));
    let mut y = Array1::<f64>::zeros(n_samples);
    let mut coef = Array1::<f64>::zeros(n_features);
    for j in 0..n_features {
        coef[j] = rng.gen::<f64>() * 2.0 - 1.0;
    }
    for i in 0..n_samples {
        let mut yi = 0.0;
        for j in 0..n_features {
            let v = rng.gen::<f64>() * 2.0 - 1.0;
            x[[i, j]] = v;
            yi += v * coef[j];
        }
        y[i] = yi + (rng.gen::<f64>() - 0.5) * 0.1;
    }
    (x, y)
}

// ===========================================================================
// Registration
// ===========================================================================

criterion_group!(
    benches,
    // Original benchmark groups
    bench_knn_classifier,
    bench_decision_tree,
    bench_random_forest,
    bench_preprocessing,
    bench_kmeans,
    bench_metrics,
    // Comparison benchmark groups (matching sklearn_benchmark.py)
    bench_scaler_1000x50,
    bench_scaler_10000x100,
    bench_knn_1000x50,
    bench_tree_5000x20,
    bench_rf_5000x20,
    bench_kmeans_5000x20,
    bench_gnb_5000x20,
    // Phase A: extended supervised comparison
    bench_ridge_5000x20,
    bench_lasso_5000x20,
    bench_elasticnet_5000x20,
    bench_ols_5000x20,
    bench_bayesian_ridge_5000x20,
    bench_logistic_5000x20,
    bench_svc_rbf_1000x20,
    bench_linear_svc_5000x20,
    bench_lda_5000x20,
    bench_qda_5000x20,
    bench_mlp_5000x20,
    bench_extra_trees_5000x20,
    bench_gbm_5000x20,
    bench_hist_gbm_5000x20,
    bench_adaboost_5000x20,
);
criterion_main!(benches);
