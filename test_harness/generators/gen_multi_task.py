"""Golden data for MultiTaskLasso / MultiTaskElasticNet (#113) against
scikit-learn 1.8.0, solved to tol=1e-12 so that the converged coefficients
are compared (sklearn's block coordinate descent stops on the duality gap).
"""

import json
import os

import numpy as np
from sklearn.linear_model import MultiTaskElasticNet, MultiTaskLasso


def generate():
    n = 40
    t = np.arange(n)
    X = np.array([[((i * 7 + j * 3 + i * j) % 13) / 6 - 1 + 0.03 * ((i * (j + 2)) % 5)
                   for j in range(6)] for i in range(n)])
    Y = np.column_stack([
        2 * X[:, 0] - X[:, 1] + 0.1 * X[:, 4] + ((t * 5) % 7 - 3) / 10,
        1.5 * X[:, 0] - 0.5 * X[:, 1] + ((t * 3) % 5 - 2) / 10,
        -X[:, 0] + 0.8 * X[:, 2] + 0.05 * np.sin(t),
    ])
    cases = []
    for name, cls, params in [
        ("MultiTaskLasso", MultiTaskLasso, {"alpha": 0.1}),
        ("MultiTaskLasso", MultiTaskLasso, {"alpha": 0.5, "fit_intercept": False}),
        ("MultiTaskElasticNet", MultiTaskElasticNet, {"alpha": 0.1, "l1_ratio": 0.5}),
        ("MultiTaskElasticNet", MultiTaskElasticNet, {"alpha": 0.3, "l1_ratio": 0.2}),
    ]:
        m = cls(tol=1e-12, max_iter=100000, **params).fit(X, Y)
        cases.append({"kind": name, "params": params, "coef": m.coef_.tolist(),
                      "intercept": np.atleast_1d(m.intercept_).tolist(), "predict": m.predict(X[:3]).tolist()})
    return {"X": X.tolist(), "Y": Y.tolist(), "cases": cases}


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "anofox-ml", "tests", "golden_data", "multi_task.json")
    with open(path, "w") as f:
        json.dump(generate(), f, indent=1)
    print(f"wrote {path}")
