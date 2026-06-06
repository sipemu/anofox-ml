"""Benchmark scikit-learn algorithms on standardized datasets.

Generates reproducible random data and measures fit/predict timings,
outputting results as JSON to stdout for comparison with rustml benchmarks.
"""

import json
import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

ITERATIONS = 5
SEED = 42


def generate_classification_data(n_samples, n_features, n_classes=2, seed=SEED):
    """Generate reproducible random classification data."""
    rng = np.random.RandomState(seed)
    x = rng.randn(n_samples, n_features)
    y = rng.randint(0, n_classes, size=n_samples).astype(np.float64)
    return x, y


def generate_regression_data(n_samples, n_features, seed=SEED):
    """Generate reproducible random regression data."""
    rng = np.random.RandomState(seed)
    x = rng.randn(n_samples, n_features)
    return x


def generate_regression_xy(n_samples, n_features, seed=SEED):
    """Generate reproducible random (X, y) for regression: y = X @ coef + noise."""
    rng = np.random.RandomState(seed)
    x = rng.randn(n_samples, n_features)
    coef = rng.randn(n_features)
    y = x @ coef + rng.randn(n_samples) * 0.1
    return x, y


def time_fn(fn, iterations=ITERATIONS):
    """Time a function over multiple iterations, returning mean and std in ms."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times.append((end - start) * 1000.0)
    return np.mean(times), np.std(times)


def bench_standard_scaler():
    """Benchmark StandardScaler fit+transform on two dataset sizes."""
    results = []

    for n_samples, n_features in [(1000, 50), (10000, 100)]:
        x = generate_regression_data(n_samples, n_features)
        label = f"StandardScaler fit+transform {n_samples}x{n_features}"

        def run(x=x):
            scaler = StandardScaler()
            scaler.fit_transform(x)

        mean_ms, std_ms = time_fn(run)
        results.append({
            "name": label,
            "mean_ms": round(mean_ms, 4),
            "std_ms": round(std_ms, 4),
            "iterations": ITERATIONS,
        })

    return results


def bench_knn():
    """Benchmark KNN classifier fit+predict."""
    results = []

    n_train, n_test, n_features = 1000, 200, 50
    x_train, y_train = generate_classification_data(n_train, n_features)
    x_test, _ = generate_classification_data(n_test, n_features, seed=SEED + 1)

    # Benchmark fit
    label_fit = f"KNN fit {n_train}x{n_features}"

    def run_fit():
        knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
        knn.fit(x_train, y_train)

    mean_ms, std_ms = time_fn(run_fit)
    results.append({
        "name": label_fit,
        "mean_ms": round(mean_ms, 4),
        "std_ms": round(std_ms, 4),
        "iterations": ITERATIONS,
    })

    # Benchmark predict
    label_predict = f"KNN predict {n_train}x{n_features}"
    knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    knn.fit(x_train, y_train)

    def run_predict():
        knn.predict(x_test)

    mean_ms, std_ms = time_fn(run_predict)
    results.append({
        "name": label_predict,
        "mean_ms": round(mean_ms, 4),
        "std_ms": round(std_ms, 4),
        "iterations": ITERATIONS,
    })

    return results


def bench_decision_tree():
    """Benchmark Decision Tree classifier fit+predict."""
    results = []

    n_samples, n_features = 5000, 20
    x_train, y_train = generate_classification_data(n_samples, n_features, n_classes=5)
    x_test, _ = generate_classification_data(500, n_features, seed=SEED + 1)

    # Benchmark fit
    label_fit = f"DecisionTree fit {n_samples}x{n_features}"

    def run_fit():
        tree = DecisionTreeClassifier(max_depth=10, random_state=SEED)
        tree.fit(x_train, y_train)

    mean_ms, std_ms = time_fn(run_fit)
    results.append({
        "name": label_fit,
        "mean_ms": round(mean_ms, 4),
        "std_ms": round(std_ms, 4),
        "iterations": ITERATIONS,
    })

    # Benchmark predict
    label_predict = f"DecisionTree predict {n_samples}x{n_features}"
    tree = DecisionTreeClassifier(max_depth=10, random_state=SEED)
    tree.fit(x_train, y_train)

    def run_predict():
        tree.predict(x_test)

    mean_ms, std_ms = time_fn(run_predict)
    results.append({
        "name": label_predict,
        "mean_ms": round(mean_ms, 4),
        "std_ms": round(std_ms, 4),
        "iterations": ITERATIONS,
    })

    return results


def bench_random_forest():
    """Benchmark Random Forest classifier fit+predict."""
    results = []

    n_samples, n_features = 5000, 20
    x_train, y_train = generate_classification_data(n_samples, n_features, n_classes=5)
    x_test, _ = generate_classification_data(500, n_features, seed=SEED + 1)

    # Benchmark fit
    label_fit = f"RandomForest fit {n_samples}x{n_features}"

    def run_fit():
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=SEED
        )
        rf.fit(x_train, y_train)

    mean_ms, std_ms = time_fn(run_fit)
    results.append({
        "name": label_fit,
        "mean_ms": round(mean_ms, 4),
        "std_ms": round(std_ms, 4),
        "iterations": ITERATIONS,
    })

    # Benchmark predict
    label_predict = f"RandomForest predict {n_samples}x{n_features}"
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=SEED
    )
    rf.fit(x_train, y_train)

    def run_predict():
        rf.predict(x_test)

    mean_ms, std_ms = time_fn(run_predict)
    results.append({
        "name": label_predict,
        "mean_ms": round(mean_ms, 4),
        "std_ms": round(std_ms, 4),
        "iterations": ITERATIONS,
    })

    return results


def bench_kmeans():
    """Benchmark KMeans fit."""
    results = []

    n_samples, n_features = 5000, 20
    x = generate_regression_data(n_samples, n_features)

    label = f"KMeans fit {n_samples}x{n_features}"

    def run():
        km = KMeans(n_clusters=10, max_iter=300, random_state=SEED, n_init=1)
        km.fit(x)

    mean_ms, std_ms = time_fn(run)
    results.append({
        "name": label,
        "mean_ms": round(mean_ms, 4),
        "std_ms": round(std_ms, 4),
        "iterations": ITERATIONS,
    })

    return results


def bench_gaussian_nb():
    """Benchmark Gaussian Naive Bayes fit+predict."""
    results = []

    n_samples, n_features = 5000, 20
    x_train, y_train = generate_classification_data(n_samples, n_features, n_classes=5)
    x_test, _ = generate_classification_data(500, n_features, seed=SEED + 1)

    # Benchmark fit
    label_fit = f"GaussianNB fit {n_samples}x{n_features}"

    def run_fit():
        gnb = GaussianNB()
        gnb.fit(x_train, y_train)

    mean_ms, std_ms = time_fn(run_fit)
    results.append({
        "name": label_fit,
        "mean_ms": round(mean_ms, 4),
        "std_ms": round(std_ms, 4),
        "iterations": ITERATIONS,
    })

    # Benchmark predict
    label_predict = f"GaussianNB predict {n_samples}x{n_features}"
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)

    def run_predict():
        gnb.predict(x_test)

    mean_ms, std_ms = time_fn(run_predict)
    results.append({
        "name": label_predict,
        "mean_ms": round(mean_ms, 4),
        "std_ms": round(std_ms, 4),
        "iterations": ITERATIONS,
    })

    return results


# -------------------------------------------------------------------------
# Phase A: extended supervised estimator comparison
# -------------------------------------------------------------------------

def _bench_fit_predict(label, ctor, x_train, y_train, x_test):
    """Common fit/predict timing wrapper used by every Phase A bench."""
    results = []

    def run_fit():
        m = ctor()
        m.fit(x_train, y_train)

    mean_ms, std_ms = time_fn(run_fit)
    results.append({
        "name": f"{label} fit {x_train.shape[0]}x{x_train.shape[1]}",
        "mean_ms": round(mean_ms, 4),
        "std_ms": round(std_ms, 4),
        "iterations": ITERATIONS,
    })

    m = ctor()
    m.fit(x_train, y_train)

    def run_predict():
        m.predict(x_test)

    mean_ms, std_ms = time_fn(run_predict)
    results.append({
        "name": f"{label} predict {x_train.shape[0]}x{x_train.shape[1]}",
        "mean_ms": round(mean_ms, 4),
        "std_ms": round(std_ms, 4),
        "iterations": ITERATIONS,
    })

    return results


def bench_ridge():
    x_train, y_train = generate_regression_xy(5000, 20)
    x_test, _ = generate_regression_xy(500, 20, seed=SEED + 1)
    return _bench_fit_predict("Ridge", lambda: Ridge(alpha=1.0), x_train, y_train, x_test)


def bench_lasso():
    x_train, y_train = generate_regression_xy(5000, 20)
    x_test, _ = generate_regression_xy(500, 20, seed=SEED + 1)
    return _bench_fit_predict("Lasso", lambda: Lasso(alpha=0.1, max_iter=2000), x_train, y_train, x_test)


def bench_elasticnet():
    x_train, y_train = generate_regression_xy(5000, 20)
    x_test, _ = generate_regression_xy(500, 20, seed=SEED + 1)
    return _bench_fit_predict(
        "ElasticNet",
        lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000),
        x_train, y_train, x_test,
    )


def bench_ols():
    x_train, y_train = generate_regression_xy(5000, 20)
    x_test, _ = generate_regression_xy(500, 20, seed=SEED + 1)
    return _bench_fit_predict("OLS", LinearRegression, x_train, y_train, x_test)


def bench_bayesian_ridge():
    x_train, y_train = generate_regression_xy(5000, 20)
    x_test, _ = generate_regression_xy(500, 20, seed=SEED + 1)
    return _bench_fit_predict("BayesianRidge", BayesianRidge, x_train, y_train, x_test)


def bench_logistic():
    x_train, y_train = generate_classification_data(5000, 20, n_classes=2)
    x_test, _ = generate_classification_data(500, 20, n_classes=2, seed=SEED + 1)
    return _bench_fit_predict(
        "LogisticRegression",
        lambda: LogisticRegression(max_iter=200),
        x_train, y_train, x_test,
    )


def bench_svc_rbf():
    x_train, y_train = generate_classification_data(1000, 20, n_classes=2)
    x_test, _ = generate_classification_data(200, 20, n_classes=2, seed=SEED + 1)
    return _bench_fit_predict(
        "SVC_RBF",
        lambda: SVC(C=1.0, kernel="rbf", gamma=0.05, max_iter=1000),
        x_train, y_train, x_test,
    )


def bench_linear_svc():
    x_train, y_train = generate_classification_data(5000, 20, n_classes=2)
    x_test, _ = generate_classification_data(500, 20, n_classes=2, seed=SEED + 1)
    return _bench_fit_predict(
        "LinearSVC",
        lambda: LinearSVC(max_iter=1000, dual="auto"),
        x_train, y_train, x_test,
    )


def bench_lda():
    x_train, y_train = generate_classification_data(5000, 20, n_classes=5)
    x_test, _ = generate_classification_data(500, 20, n_classes=5, seed=SEED + 1)
    return _bench_fit_predict("LDA", LinearDiscriminantAnalysis, x_train, y_train, x_test)


def bench_qda():
    x_train, y_train = generate_classification_data(5000, 20, n_classes=5)
    x_test, _ = generate_classification_data(500, 20, n_classes=5, seed=SEED + 1)
    return _bench_fit_predict("QDA", QuadraticDiscriminantAnalysis, x_train, y_train, x_test)


def bench_mlp():
    x_train, y_train = generate_classification_data(5000, 20, n_classes=5)
    x_test, _ = generate_classification_data(500, 20, n_classes=5, seed=SEED + 1)
    return _bench_fit_predict(
        "MLP",
        lambda: MLPClassifier(hidden_layer_sizes=(32,), max_iter=50, random_state=SEED),
        x_train, y_train, x_test,
    )


def bench_extra_trees():
    x_train, y_train = generate_classification_data(5000, 20, n_classes=5)
    x_test, _ = generate_classification_data(500, 20, n_classes=5, seed=SEED + 1)
    return _bench_fit_predict(
        "ExtraTrees",
        lambda: ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=SEED),
        x_train, y_train, x_test,
    )


def bench_gbm():
    x_train, y_train = generate_classification_data(5000, 20, n_classes=2)
    x_test, _ = generate_classification_data(500, 20, n_classes=2, seed=SEED + 1)
    return _bench_fit_predict(
        "GradientBoosting",
        lambda: GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=SEED),
        x_train, y_train, x_test,
    )


def bench_hist_gbm():
    x_train, y_train = generate_classification_data(5000, 20, n_classes=2)
    x_test, _ = generate_classification_data(500, 20, n_classes=2, seed=SEED + 1)
    return _bench_fit_predict(
        "HistGradientBoosting",
        lambda: HistGradientBoostingClassifier(max_iter=100, max_depth=6, random_state=SEED),
        x_train, y_train, x_test,
    )


def bench_adaboost():
    x_train, y_train = generate_classification_data(5000, 20, n_classes=2)
    x_test, _ = generate_classification_data(500, 20, n_classes=2, seed=SEED + 1)
    return _bench_fit_predict(
        "AdaBoost",
        lambda: AdaBoostClassifier(n_estimators=50, random_state=SEED),
        x_train, y_train, x_test,
    )


def main():
    """Run all benchmarks and output JSON results to stdout."""
    all_results = []

    print("Running sklearn benchmarks...", flush=True, file=__import__("sys").stderr)

    benchmarks = [
        ("StandardScaler", bench_standard_scaler),
        ("KNN", bench_knn),
        ("DecisionTree", bench_decision_tree),
        ("RandomForest", bench_random_forest),
        ("KMeans", bench_kmeans),
        ("GaussianNB", bench_gaussian_nb),
        # Phase A: extended supervised comparison
        ("Ridge", bench_ridge),
        ("Lasso", bench_lasso),
        ("ElasticNet", bench_elasticnet),
        ("OLS", bench_ols),
        ("BayesianRidge", bench_bayesian_ridge),
        ("LogisticRegression", bench_logistic),
        ("SVC_RBF", bench_svc_rbf),
        ("LinearSVC", bench_linear_svc),
        ("LDA", bench_lda),
        ("QDA", bench_qda),
        ("MLP", bench_mlp),
        ("ExtraTrees", bench_extra_trees),
        ("GradientBoosting", bench_gbm),
        ("HistGradientBoosting", bench_hist_gbm),
        ("AdaBoost", bench_adaboost),
    ]

    for name, bench_fn in benchmarks:
        print(f"  Benchmarking {name}...", flush=True, file=__import__("sys").stderr)
        all_results.extend(bench_fn())

    output = {"benchmarks": all_results}
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
