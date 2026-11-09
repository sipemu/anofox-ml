"""Golden data for SparsePCA (#113, method="lars") against scikit-learn
1.8.0."""

import json
import os

import numpy as np
from sklearn.decomposition import SparsePCA


def generate():
    r = np.random.RandomState(4)
    latent = r.randn(30, 2)
    load = np.array([[1.0, 1.0, 0.5, 0, 0, 0], [0, 0, 0, 1.0, -1.0, 0.7]])
    X = np.round(latent @ load + 0.3 * r.randn(30, 6), 3)
    cases = []
    for params in [{"n_components": 2, "random_state": 0},
                   {"n_components": 3, "alpha": 0.3, "random_state": 0},
                   {"n_components": 2, "alpha": 2.0, "ridge_alpha": 0.5, "random_state": 0}]:
        m = SparsePCA(**params).fit(X)
        cases.append({"params": params, "components": m.components_.tolist(),
                      "n_iter": int(m.n_iter_), "error": list(map(float, m.error_)),
                      "transform": m.transform(X[:5]).tolist()})
    return {"X": X.tolist(), "cases": cases}


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "anofox-ml", "tests", "golden_data", "sparse_pca.json")
    with open(path, "w") as f:
        json.dump(generate(), f, indent=1)
    print(f"wrote {path}")
