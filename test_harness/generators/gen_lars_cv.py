"""Golden data for LarsCV / LassoLarsCV (#113) against scikit-learn 1.8.0."""

import json
import os

import numpy as np
from sklearn.linear_model import LarsCV, LassoLarsCV


def generate():
    r = np.random.RandomState(7)
    n, p = 60, 8
    X = np.round(r.randn(n, p), 3)
    y = np.round(X @ np.array([2.0, -1.5, 0.0, 0.0, 0.8, 0.0, 0.3, 0.0])
                 + 0.7 * r.randn(n), 3)
    cases = []
    for name, cls, params in [
        ("LarsCV", LarsCV, {}),
        ("LarsCV", LarsCV, {"cv": 3, "fit_intercept": False}),
        ("LassoLarsCV", LassoLarsCV, {}),
        ("LassoLarsCV", LassoLarsCV, {"cv": 4, "max_n_alphas": 10}),
    ]:
        m = cls(**params).fit(X, y)
        cases.append({"kind": name, "params": params, "alpha": float(m.alpha_),
                      "cv_alphas": m.cv_alphas_.tolist(), "mse_path": m.mse_path_.tolist(),
                      "coef": m.coef_.tolist(), "intercept": float(m.intercept_),
                      "alphas": m.alphas_.tolist(), "predict": m.predict(X[:5]).tolist()})
    return {"X": X.tolist(), "y": y.tolist(), "cases": cases}


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "..", "..",
                        "crates", "anofox-ml", "tests", "golden_data", "lars_cv.json")
    with open(path, "w") as f:
        json.dump(generate(), f, indent=1)
    print(f"wrote {path}")
