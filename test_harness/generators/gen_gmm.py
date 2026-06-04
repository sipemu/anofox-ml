"""Golden data for GaussianMixture (sklearn.mixture.GaussianMixture)."""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture


def generate():
    X, y_true = make_blobs(n_samples=150, centers=3, cluster_std=0.7, random_state=0)
    cases = []
    for ct in ["full", "diag"]:
        gm = GaussianMixture(n_components=3, covariance_type=ct, random_state=0, n_init=3)
        labels = gm.fit_predict(X)
        cases.append({
            "name": f"gmm_{ct}",
            "covariance_type": ct,
            "X": X.tolist(),
            "y_true": y_true.astype(float).tolist(),
            "sklearn_ari": float(adjusted_rand_score(y_true, labels)),
        })
    return cases


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "rustml", "tests", "golden_data",
                        "gmm.json")
    with open(out, "w") as f:
        json.dump(generate(), f, indent=2)
    print(f"wrote {out}")
