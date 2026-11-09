"""Golden data for BisectingKMeans (#113) against scikit-learn 1.8.0:
seeded fits (numpy RandomState init draws) with both inits and strategies."""

import json
import os

import numpy as np
from sklearn.cluster import BisectingKMeans


def generate():
    r = np.random.RandomState(9)
    centers = np.array([[0, 0], [5, 5], [0, 6], [6, 0], [3, 3]])
    X = np.round(np.vstack([c + r.randn(18, 2) for c in centers]), 3)
    w = np.round(r.uniform(0.5, 2.0, len(X)), 2)
    cases = []
    for params, weighted in [
        ({"n_clusters": 4, "random_state": 0}, False),
        ({"n_clusters": 5, "init": "k-means++", "random_state": 3}, False),
        ({"n_clusters": 3, "bisecting_strategy": "largest_cluster", "n_init": 3,
          "random_state": 1}, False),
        ({"n_clusters": 4, "random_state": 5}, True),
    ]:
        m = BisectingKMeans(**params).fit(X, sample_weight=w if weighted else None)
        cases.append({"params": params, "weighted": weighted,
                      "centers": m.cluster_centers_.tolist(), "labels": m.labels_.tolist(),
                      "inertia": float(m.inertia_),
                      "predict": m.predict(X[::7] + 0.3).tolist()})
    return {"X": X.tolist(), "w": w.tolist(), "cases": cases}


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "anofox-ml", "tests", "golden_data", "bisecting_kmeans.json")
    with open(path, "w") as f:
        json.dump(generate(), f, indent=1)
    print(f"wrote {path}")
