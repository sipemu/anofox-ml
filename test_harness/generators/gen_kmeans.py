"""Generate golden data for KMeans tests."""

import numpy as np
from sklearn.cluster import KMeans


def generate():
    cases = []

    # 3 well-separated clusters
    np.random.seed(42)
    cluster1 = np.random.randn(10, 2) * 0.5 + np.array([0.0, 0.0])
    cluster2 = np.random.randn(10, 2) * 0.5 + np.array([10.0, 0.0])
    cluster3 = np.random.randn(10, 2) * 0.5 + np.array([5.0, 10.0])
    X_train = np.vstack([cluster1, cluster2, cluster3])

    X_test = np.array([[0.1, 0.2], [10.1, 0.1], [5.1, 9.8], [5.0, 5.0]])

    km = KMeans(n_clusters=3, random_state=42, n_init=1, max_iter=300, tol=1e-4)
    km.fit(X_train)

    cases.append({
        "name": "kmeans_3_clusters",
        "algorithm": "KMeans",
        "X_train": X_train.tolist(),
        "X_test": X_test.tolist(),
        "n_clusters": 3,
        "centroids": km.cluster_centers_.tolist(),
        "labels_train": km.labels_.tolist(),
        "labels_test": km.predict(X_test).tolist(),
        "inertia": float(km.inertia_),
        "n_iter": int(km.n_iter_),
    })

    # 2 clusters, simple
    X_simple = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [10.0, 10.0], [11.0, 10.0], [10.0, 11.0],
    ])
    X_test_simple = np.array([[0.5, 0.5], [10.5, 10.5]])

    km2 = KMeans(n_clusters=2, random_state=42, n_init=1, max_iter=300, tol=1e-4)
    km2.fit(X_simple)

    cases.append({
        "name": "kmeans_2_clusters_simple",
        "algorithm": "KMeans",
        "X_train": X_simple.tolist(),
        "X_test": X_test_simple.tolist(),
        "n_clusters": 2,
        "centroids": km2.cluster_centers_.tolist(),
        "labels_train": km2.labels_.tolist(),
        "labels_test": km2.predict(X_test_simple).tolist(),
        "inertia": float(km2.inertia_),
        "n_iter": int(km2.n_iter_),
    })

    return cases
