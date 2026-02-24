"""Generate golden data for DBSCAN tests."""

import numpy as np
from sklearn.cluster import DBSCAN


def generate():
    cases = []

    # Well-separated clusters with noise
    np.random.seed(42)
    cluster1 = np.random.randn(20, 2) * 0.3 + np.array([0.0, 0.0])
    cluster2 = np.random.randn(20, 2) * 0.3 + np.array([5.0, 5.0])
    cluster3 = np.random.randn(15, 2) * 0.3 + np.array([10.0, 0.0])
    noise = np.random.uniform(-2, 12, size=(5, 2))
    X = np.vstack([cluster1, cluster2, cluster3, noise])

    for eps, min_samples in [(0.8, 3), (1.5, 5), (0.5, 2)]:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        core_indices = db.core_sample_indices_.tolist()

        cases.append({
            "name": f"dbscan_eps{eps}_min{min_samples}",
            "algorithm": "Dbscan",
            "X": X.tolist(),
            "labels": labels.astype(float).tolist(),
            "eps": eps,
            "min_samples": min_samples,
            "n_clusters": n_clusters,
            "n_noise": int(n_noise),
            "core_sample_indices": core_indices,
        })

    # All noise case (very small eps)
    db_noise = DBSCAN(eps=0.01, min_samples=5)
    labels_noise = db_noise.fit_predict(X)

    cases.append({
        "name": "dbscan_all_noise",
        "algorithm": "Dbscan",
        "X": X.tolist(),
        "labels": labels_noise.astype(float).tolist(),
        "eps": 0.01,
        "min_samples": 5,
        "n_clusters": 0,
        "n_noise": len(X),
        "core_sample_indices": [],
    })

    return cases
