"""Golden data for MiniBatchKMeans (sklearn.cluster.MiniBatchKMeans).

Different RNG order means we don't try to match labels directly. Instead we
match the *centroid set* (after a greedy nearest-neighbor matching), and
require label agreement on a held-out set after permuting labels.
"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import make_blobs


def generate():
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)
    mbk = MiniBatchKMeans(
        n_clusters=4, batch_size=64, max_iter=200,
        random_state=0, n_init=3, tol=1e-4,
    )
    mbk.fit(X)
    return [{
        "name": "blobs_4",
        "X": X.tolist(),
        "n_clusters": 4,
        "batch_size": 64,
        "sklearn_centroids": mbk.cluster_centers_.tolist(),
        "sklearn_inertia": float(mbk.inertia_),
    }]


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "rustml", "tests", "golden_data",
                        "mini_batch_kmeans.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
