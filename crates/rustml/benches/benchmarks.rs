use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2};
use rustml::prelude::*;

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

        group.bench_with_input(BenchmarkId::new("standard_scaler_fit_transform", n), &n, |b, _| {
            b.iter(|| {
                let scaler = StandardScaler::default();
                let fitted = FitUnsupervised::<f64>::fit(&scaler, black_box(&x)).unwrap();
                fitted.transform(black_box(&x)).unwrap()
            });
        });
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

criterion_group!(
    benches,
    bench_knn_classifier,
    bench_decision_tree,
    bench_random_forest,
    bench_preprocessing,
    bench_kmeans,
    bench_metrics,
);
criterion_main!(benches);
