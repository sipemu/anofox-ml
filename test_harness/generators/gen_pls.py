"""Golden data for PLSCanonical and PLSSVD (#113) against scikit-learn
1.8.0: the X / Y weights and the scores of new rows, default parameters
(scale=True) and n_components=2.
"""

import json
import os

import numpy as np
from sklearn.cross_decomposition import PLSSVD, PLSCanonical


def generate():
    n = 25
    X = np.array([[((i * 7 + j * 3 + i * j) % 13) / 6 - 1 + 0.03 * ((i * (j + 2)) % 5)
                   for j in range(4)] for i in range(n)])
    Y = np.column_stack([
        X[:, 0] + 0.5 * X[:, 1] + ((np.arange(n) * 5) % 7 - 3) / 10,
        X[:, 2] - X[:, 0] + ((np.arange(n) * 3) % 5 - 2) / 10,
        0.3 * X[:, 3] + np.sin(np.arange(n)),
    ])
    Xq, Yq = X[::6], Y[::6]
    out = {"X": X.tolist(), "Y": Y.tolist(), "Xq": Xq.tolist(), "Yq": Yq.tolist(), "models": {}}
    for name, m in [("PLSCanonical", PLSCanonical(n_components=2)), ("PLSSVD", PLSSVD(n_components=2))]:
        m.fit(X, Y)
        xs, ys = m.transform(Xq, Yq)
        out["models"][name] = {"x_weights": m.x_weights_.tolist(), "y_weights": m.y_weights_.tolist(),
                               "x_scores": xs.tolist(), "y_scores": ys.tolist()}
    return out


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "anofox-ml", "tests", "golden_data", "pls_cross.json")
    with open(path, "w") as f:
        json.dump(generate(), f, indent=1)
    print(f"wrote {path}")
