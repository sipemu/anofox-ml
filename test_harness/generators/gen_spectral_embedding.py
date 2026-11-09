"""Golden data for SpectralEmbedding (#113) against scikit-learn 1.8.0:
`embedding_` for the nearest-neighbours affinity (default `n_neighbors`
and an explicit one) and the RBF affinity (default and explicit gamma).
Signs follow sklearn's `_deterministic_vector_sign_flip`."""

import json
import os

import numpy as np
from sklearn.manifold import SpectralEmbedding


def generate():
    n = 40
    t = np.arange(n)
    X = np.column_stack([np.cos(t / 6.0) * (1 + 0.1 * (t % 3)), np.sin(t / 6.0), 0.05 * ((t * 7) % 5)])
    cases = []
    for params in [{}, {"n_neighbors": 6}, {"affinity": "rbf"}, {"affinity": "rbf", "gamma": 2.0, "n_components": 3}]:
        m = SpectralEmbedding(random_state=0, **params).fit(X)
        cases.append({"params": params, "embedding": m.embedding_.tolist()})
    return {"X": X.tolist(), "cases": cases}


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "anofox-ml", "tests", "golden_data", "spectral_embedding.json")
    with open(path, "w") as f:
        json.dump(generate(), f, indent=1)
    print(f"wrote {path}")
