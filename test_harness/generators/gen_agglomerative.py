"""Golden data for AgglomerativeClustering (sklearn.cluster)."""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score


def generate():
    X, y_true = make_blobs(n_samples=120, centers=4, cluster_std=0.5, random_state=0)
    cases = []
    for link in ["ward", "complete", "average", "single"]:
        ac = AgglomerativeClustering(n_clusters=4, linkage=link)
        labels = ac.fit_predict(X)
        cases.append({
            "name": f"agglo_{link}",
            "linkage": link,
            "X": X.tolist(),
            "y_true": y_true.astype(float).tolist(),
            "sklearn_labels": labels.astype(float).tolist(),
            "sklearn_ari": float(adjusted_rand_score(y_true, labels)),
        })
    return cases


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "rustml", "tests", "golden_data",
                        "agglomerative.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
